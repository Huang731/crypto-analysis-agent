"""请求上下文与结构化日志，贯通 FastAPI、路由 和分析任务。"""

from __future__ import annotations

import functools
import json
import logging
import sys
import time
import uuid
from collections.abc import Callable
from contextvars import ContextVar, Token
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_ROOT = Path(__file__).resolve().parents[1]
_trace_id: ContextVar[str] = ContextVar("crypto_trace_id", default="-")
_thread_id: ContextVar[str] = ContextVar("crypto_thread_id", default="-")


class LogSettings(BaseSettings):
    """从环境变量或项目 .env 读取日志格式、级别和文件配置。"""

    model_config = SettingsConfigDict(
        env_file=_ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    log_json: bool = True
    log_level: str = "INFO"
    log_dir: Path | None = None
    log_max_bytes: int = Field(default=10 * 1024 * 1024, gt=0)
    log_backup_count: int = Field(default=5, ge=1)


def new_trace_id() -> str:
    """生成 QA 项目使用的 8 位十六进制追踪号。"""
    return uuid.uuid4().hex[:8]


def bind_request_context(
    trace_id: str, thread_id: str
) -> tuple[Token[str], Token[str]]:
    """将两个 ID 绑定到当前异步请求上下文。"""
    return _trace_id.set(trace_id), _thread_id.set(thread_id)


def reset_request_context(tokens: tuple[Token[str], Token[str]]) -> None:
    """请求结束后恢复上下文，防止后续请求继承旧 ID。"""
    _trace_id.reset(tokens[0])
    _thread_id.reset(tokens[1])


_RECORD_BUILTINS = set(logging.makeLogRecord({}).__dict__) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    """把业务事件和请求 ID 输出为单行 JSON。"""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "trace_id": _trace_id.get(),
            "thread_id": _thread_id.get(),
            "msg": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key in payload or key in _RECORD_BUILTINS or key.startswith("_"):
                continue
            payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


class TextFormatter(logging.Formatter):
    """开发时使用的可读日志格式。"""

    def __init__(self) -> None:
        super().__init__(
            "%(asctime)s | %(levelname)-7s | %(name)s | "
            "trace=%(trace_id)s | thread=%(thread_id)s | %(message)s",
            datefmt="%H:%M:%S",
        )

    def format(self, record: logging.LogRecord) -> str:
        record.trace_id = _trace_id.get()
        record.thread_id = _thread_id.get()
        return super().format(record)


_configured = False


def setup_logging() -> None:
    """按 QA 项目方式配置标准库日志，仅初始化一次。"""
    global _configured
    if _configured:
        return
    settings = LogSettings()
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    handler = logging.StreamHandler(sys.stdout)
    formatter = JsonFormatter() if settings.log_json else TextFormatter()
    handler.setFormatter(formatter)
    handler.setLevel(level)
    root.addHandler(handler)
    # 指定日志目录时追加文件输出，保留控制台日志和原有请求上下文。
    if settings.log_dir is not None:
        settings.log_dir.mkdir(parents=True, exist_ok=True)
        for filename, file_level in (
            ("app.log", level),
            ("error.log", logging.ERROR),
        ):
            file_handler = RotatingFileHandler(
                settings.log_dir / filename,
                maxBytes=settings.log_max_bytes,
                backupCount=settings.log_backup_count,
                encoding="utf-8",
            )
            file_handler.setFormatter(formatter)
            file_handler.setLevel(file_level)
            root.addHandler(file_handler)
    root.setLevel(level)
    for name in ("httpx", "httpx2", "httpcore", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """返回使用统一格式的 logger。"""
    setup_logging()
    return logging.getLogger(name)


def timed_node(
    func: Callable | None = None, *, label: Callable[[Any], str] | None = None
) -> Callable:
    """节点计时装饰器：记录进出 + 耗时。

    用法：
        @timed_node
        def researcher(state): ...

    同时支持 sync 和 async 函数（自动检测）。
    """
    if func is None:
        return functools.partial(timed_node, label=label)
    log = get_logger(f"node.{func.__name__}")
    node_name = func.__name__

    if _is_async(func):

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            # 可选标签从节点输入读取，用于区分并行任务。
            name = (
                f"{node_name} [{label(args[0] if args else kwargs['state'])}]"
                if label is not None
                else node_name
            )
            t0 = time.perf_counter()
            log.info(f"{name} start")
            try:
                result = await func(*args, **kwargs)
                duration_ms = (time.perf_counter() - t0) * 1000
                # 摘要结果大小（不打印全部，避免日志爆炸）
                summary = _summarize_result(result)
                log.info(f"{name} done | {duration_ms:.0f}ms | {summary}")
                return result
            except Exception as e:
                duration_ms = (time.perf_counter() - t0) * 1000
                log.error(
                    f"{name} failed | {duration_ms:.0f}ms | {type(e).__name__}: {e}"
                )
                raise

        return async_wrapper

    @functools.wraps(func)
    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        name = (
            f"{node_name} [{label(args[0] if args else kwargs['state'])}]"
            if label is not None
            else node_name
        )
        t0 = time.perf_counter()
        log.info(f"{name} start")
        try:
            result = func(*args, **kwargs)
            duration_ms = (time.perf_counter() - t0) * 1000
            summary = _summarize_result(result)
            log.info(f"{name} done | {duration_ms:.0f}ms | {summary}")
            return result
        except Exception as e:
            duration_ms = (time.perf_counter() - t0) * 1000
            log.error(f"{name} failed | {duration_ms:.0f}ms | {type(e).__name__}: {e}")
            raise

    return sync_wrapper


def _is_async(func: Callable) -> bool:
    import asyncio

    return asyncio.iscoroutinefunction(func)


def _summarize_result(result: Any) -> str:
    """把节点返回的 state 增量摘要成简短字符串（避免日志爆炸）。"""
    if not isinstance(result, dict):
        return f"→ {type(result).__name__}"
    parts = []
    for k, v in result.items():
        if isinstance(v, (list, str)):
            parts.append(f"{k}={len(v)}")
        elif isinstance(v, (int, float)):
            parts.append(f"{k}={v}")
        else:
            parts.append(f"{k}=?")
    return "→ {" + ", ".join(parts) + "}"
