"""请求上下文与结构化日志，贯通 FastAPI、Supervisor 和分析任务。"""

from __future__ import annotations

import functools
import json
import logging
import sys
import time
import uuid
from collections.abc import Callable
from contextvars import ContextVar, Token
from pathlib import Path
from typing import Any

from pydantic_settings import BaseSettings, SettingsConfigDict

_ROOT = Path(__file__).resolve().parents[1]
_trace_id: ContextVar[str] = ContextVar("crypto_trace_id", default="-")
_thread_id: ContextVar[str] = ContextVar("crypto_thread_id", default="-")


class LogSettings(BaseSettings):
    """从环境变量或项目 .env 读取日志格式和级别。"""

    model_config = SettingsConfigDict(
        env_file=_ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    log_json: bool = True
    log_level: str = "INFO"


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
    handler.setFormatter(JsonFormatter() if settings.log_json else TextFormatter())
    handler.setLevel(level)
    root.addHandler(handler)
    root.setLevel(level)
    for name in ("httpx", "httpx2", "httpcore", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """返回使用统一格式的 logger。"""
    setup_logging()
    return logging.getLogger(name)


def log_event(logger: logging.Logger, event: str, **fields: Any) -> None:
    """记录带事件名、trace_id 和 thread_id 的业务日志。"""
    logger.info(event, extra={"event": event, **fields})


def timed_node(func: Callable) -> Callable:
    """节点计时装饰器：记录进出 + 耗时。

    用法：
        @timed_node
        def researcher(state): ...

    同时支持 sync 和 async 函数（自动检测）。
    """
    log = get_logger(f"node.{func.__name__}")
    node_name = func.__name__

    if _is_async(func):

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            t0 = time.perf_counter()
            log.info(f"{node_name} start")
            try:
                result = await func(*args, **kwargs)
                duration_ms = (time.perf_counter() - t0) * 1000
                # 摘要结果大小（不打印全部，避免日志爆炸）
                summary = _summarize_result(result)
                log.info(f"{node_name} done | {duration_ms:.0f}ms | {summary}")
                return result
            except Exception as e:
                duration_ms = (time.perf_counter() - t0) * 1000
                log.error(
                    f"{node_name} failed | {duration_ms:.0f}ms | {type(e).__name__}: {e}"
                )
                raise

        return async_wrapper

    @functools.wraps(func)
    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        log.info(f"{node_name} start")
        try:
            result = func(*args, **kwargs)
            duration_ms = (time.perf_counter() - t0) * 1000
            summary = _summarize_result(result)
            log.info(f"{node_name} done | {duration_ms:.0f}ms | {summary}")
            return result
        except Exception as e:
            duration_ms = (time.perf_counter() - t0) * 1000
            log.error(
                f"{node_name} failed | {duration_ms:.0f}ms | {type(e).__name__}: {e}"
            )
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
