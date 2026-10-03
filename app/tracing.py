"""自写的控制台追踪器，用于记录 trace、span、generation 和估算成本。

当前实现只包含本地数据结构与控制台输出，不调用 Langfuse 或其他追踪服务。
``start_trace`` 会在上下文退出时打印 trace；``trace_span`` 和
``trace_generation`` 会加入当前异步上下文中的 trace，并按父子关系组织。
"""

from __future__ import annotations

import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from app.config.config import settings

from .observability import get_logger

_log = get_logger("crypto-analysis-agent.tracing")
_current_trace: ContextVar[_Trace | None] = ContextVar(
    "crypto_current_trace", default=None
)
_current_span: ContextVar[_Span | None] = ContextVar(
    "crypto_current_span", default=None
)


def compute_cost(model: str, usage: dict[str, int]) -> float | None:
    """按 token 用量 + 模型单价算成本（元）。generation 退出时自动调用。

    单价集中在 settings.model_price_table；未配置该模型单价时返回未知。
    """
    price = settings.model_price_table.get(model)
    if price is None:
        return None
    in_tok = usage.get("input", 0)
    out_tok = usage.get("output", 0)
    return round((in_tok * price["input"] + out_tok * price["output"]) / 1_000_000, 6)


def estimate_tokens(text: str) -> int:
    """按中文字符和其他字符长度估算 token 数，不读取或输出原文。"""
    chinese_chars = sum("\u3400" <= char <= "\u9fff" for char in text)
    other_chars = len(text) - chinese_chars
    return chinese_chars + (other_chars + 3) // 4


# ════════════════════════════════════════════════════════════════
# 自写控制台追踪后端
# ════════════════════════════════════════════════════════════════
@dataclass
class _Span:
    name: str
    kind: str = "span"  # "span" | "generation"
    input: Any = None
    output: Any = None
    start: float = 0.0
    end: float = 0.0
    metadata: dict = field(default_factory=dict)
    model: str | None = None
    usage: dict | None = None
    cost: float | None = None
    children: list[_Span] = field(default_factory=list)

    @property
    def duration_ms(self) -> int:
        return round((self.end - self.start) * 1000) if self.end else 0


@dataclass
class _Trace:
    name: str
    input: Any = None
    output: Any = None
    metadata: dict = field(default_factory=dict)
    spans: list[_Span] = field(default_factory=list)
    start: float = field(default_factory=time.perf_counter)

    @property
    def duration_ms(self) -> int:
        return round((time.perf_counter() - self.start) * 1000)

    def total_cost(self) -> float | None:
        total = 0.0
        usage_missing = False
        generation_count = 0

        def walk(spans: list[_Span]) -> None:
            nonlocal total, generation_count, usage_missing
            for s in spans:
                if s.kind == "generation":
                    generation_count += 1
                    if s.usage is None or s.cost is None:
                        usage_missing = True
                    else:
                        total += s.cost
                walk(s.children)

        walk(self.spans)
        return None if usage_missing or generation_count == 0 else total


class _ConsoleBackend:
    """自写控制台后端：在异步上下文中关联父子 span，并打印 trace 树。"""

    @contextmanager
    def trace(self, name: str, **meta: Any) -> Iterator[_Trace]:
        trace = _Trace(name=name, metadata=meta)
        trace_token = _current_trace.set(trace)
        span_token = _current_span.set(None)
        try:
            yield trace
        finally:
            self._render(trace)
            _current_span.reset(span_token)
            _current_trace.reset(trace_token)

    @contextmanager
    def span(self, name: str, **meta: Any) -> Iterator[_Span]:
        s = _Span(name=name, kind="span", metadata=meta, start=time.perf_counter())
        self._attach(s)
        token = _current_span.set(s)
        try:
            yield s
        finally:
            s.end = time.perf_counter()
            _current_span.reset(token)

    @contextmanager
    def generation(self, name: str, model: str, **meta: Any) -> Iterator[_Span]:
        g = _Span(
            name=name,
            kind="generation",
            model=model,
            metadata=meta,
            start=time.perf_counter(),
        )
        self._attach(g)
        token = _current_span.set(g)
        try:
            yield g
        finally:
            g.end = time.perf_counter()
            if g.usage:
                g.cost = compute_cost(g.model or "", g.usage)
            _current_span.reset(token)

    @staticmethod
    def _attach(s: _Span) -> None:
        current_trace = _current_trace.get()
        parent_span = _current_span.get()
        if parent_span is not None:
            parent_span.children.append(s)
        elif current_trace is not None:
            current_trace.spans.append(s)

    def _render(self, trace: _Trace) -> None:
        print(f"\n┌─ [trace] {trace.name} ({trace.duration_ms}ms)", file=sys.stderr)
        if trace.metadata:
            print(f"│  meta: {trace.metadata}", file=sys.stderr)
        if trace.output is not None:
            print(f"│  result: {trace.output}", file=sys.stderr)

        def walk(spans: list[_Span], prefix: str = "│") -> None:
            for i, s in enumerate(spans):
                last = i == len(spans) - 1
                branch = "└─" if last else "├─"
                child_prefix = prefix.replace("├", "│").replace("└", " ") + (
                    "  " if last else "│ "
                )
                tag = s.kind.upper()
                extra = ""
                if s.kind == "generation":
                    extra = f" | model={s.model}"
                    if s.usage is None:
                        extra += " token_usage_est=unknown cost_est=unknown"
                    else:
                        extra += (
                            f" input_tokens_est={s.usage['input']}"
                            f" output_tokens_est={s.usage['output']}"
                            f" cost_est={'未知' if s.cost is None else f'¥{s.cost:.6f}'}"
                        )
                if s.metadata.get("status") == "error":
                    extra += " | status=error"
                print(
                    f"{prefix}{branch} {tag} {s.name} ({s.duration_ms}ms){extra}",
                    file=sys.stderr,
                )
                if s.children:
                    walk(s.children, child_prefix)

        walk(trace.spans)
        total_cost = trace.total_cost()
        if total_cost is None:
            print("└─ 本次问答模型成本: 未记录", file=sys.stderr)
        else:
            print(f"└─ 已记录模型调用估算成本: ¥{total_cost:.6f}", file=sys.stderr)

    def flush(self) -> None:
        pass  # 控制台即时打印，无需 flush


@contextmanager
def trace_span(name: str, **meta: Any) -> Iterator[Any]:
    """创建普通 span，并自动挂到当前 trace 或父 span 下。"""
    with _ConsoleBackend().span(name, **meta) as s:
        yield s


@contextmanager
def trace_generation(name: str, model: str, **meta: Any) -> Iterator[Any]:
    """创建并返回一个本地 generation span，记录模型名及可选 token 用量。

    退出上下文时，如果 ``usage`` 已设置，会按配置中的模型单价估算成本；
    当前结果保存在 span 对象中，不会上报到外部服务。
    """
    with _ConsoleBackend().generation(name, model=model, **meta) as g:
        yield g


@contextmanager
def start_trace(name: str, **meta: Any) -> Iterator[Any]:
    """创建本地 trace，并在上下文退出时打印它当前包含的 span 树。"""
    with _ConsoleBackend().trace(name, **meta) as t:
        yield t


def flush() -> None:
    """控制台输出为同步写入，无需额外刷新缓冲区。"""
    _ConsoleBackend().flush()
