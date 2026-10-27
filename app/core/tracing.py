"""自写的控制台追踪器，用于记录 trace、span、generation 和估算成本。

当前实现只包含本地数据结构与控制台输出，不调用 Langfuse 或其他追踪服务。
``trace_model_call`` 统一记录模型调用与普通节点，按父子关系组织，
最外层调用结束时打印 trace。业务代码无需手工管理追踪上下文。
"""

from __future__ import annotations

import sys
import time
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import ExitStack, aclosing, closing, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from inspect import (
    BoundArguments,
    isasyncgenfunction,
    iscoroutinefunction,
    isgeneratorfunction,
    signature,
)
from typing import Any, TypeVar, cast

from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel
from pydantic_core import to_json

from app.core.config import settings

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


_TraceFunction = TypeVar("_TraceFunction", bound=Callable[..., Any])
_TraceLabel = str | Callable[[BoundArguments], str]


def _estimate_model_tokens(value: Any) -> int:
    """按实际消息字段或结构化结果估算，不计入响应元数据。"""
    if isinstance(value, str):
        return estimate_tokens(value)
    if isinstance(value, BaseMessage):
        payload = value.model_dump(
            mode="python",
            include={"content", "name", "tool_calls", "tool_call_id"},
            exclude_none=True,
            exclude_defaults=True,
        )
        payload["role"] = {"human": "user", "ai": "assistant"}.get(
            value.type, value.type
        )
        for field_name in ("function_call", "reasoning_content"):
            if field_name in value.additional_kwargs:
                payload[field_name] = value.additional_kwargs[field_name]
        return sum(
            estimate_tokens(
                item
                if isinstance(item, str)
                else to_json(item, bytes_mode="base64").decode("utf-8")
            )
            for item in payload.values()
        )
    if isinstance(value, Document):
        return estimate_tokens(value.page_content)
    if isinstance(value, ModelRequest):
        messages = [*value.messages]
        if value.system_message is not None:
            messages.insert(0, value.system_message)
        tools = [convert_to_openai_tool(tool) for tool in value.tools]
        return _estimate_model_tokens(messages) + (
            estimate_tokens(to_json(tools).decode("utf-8")) if tools else 0
        )
    if isinstance(value, ModelResponse):
        return _estimate_model_tokens(value.result)
    if isinstance(value, list):
        if not all(isinstance(message, (BaseMessage, Document)) for message in value):
            raise TypeError("模型追踪的列表只能包含消息或文档")
        return sum(_estimate_model_tokens(message) for message in value)
    if isinstance(value, BaseModel):
        return estimate_tokens(value.model_dump_json())
    raise TypeError(f"模型追踪不支持此输入或输出类型：{type(value).__name__}")


def trace_model_call(
    name: _TraceLabel,
    model: _TraceLabel | None = None,
) -> Callable[[_TraceFunction], _TraceFunction]:
    """统一追踪调用；指定模型时估算 token，省略模型时只记录普通节点。

    模型函数必须声明 messages，支持文本、消息、文档及 Agent 模型请求。
    文档输入只统计输入 token；其余输出支持消息或 Pydantic 结构化结果。
    需要补充追踪元数据或结果摘要的函数可声明仅限关键字的 trace 参数。
    最外层自动创建 trace；同步、异步和生成器均在实际执行结束时退出追踪。
    """

    def decorate(func: _TraceFunction) -> _TraceFunction:
        func_signature = signature(func)
        if model is not None and "messages" not in func_signature.parameters:
            raise TypeError("被追踪的模型函数必须声明 messages 参数")
        inject_trace = "trace" in func_signature.parameters
        is_generator = isasyncgenfunction(func) or isgeneratorfunction(func)
        if model is not None and is_generator:
            raise TypeError("模型 token 估算要求完整返回结果，不能装饰生成器")

        @contextmanager
        def context(
            args: tuple[Any, ...], kwargs: dict[str, Any]
        ) -> Iterator[tuple[BoundArguments, _Trace | _Span, int | None, bool]]:
            if inject_trace:
                kwargs = {**kwargs, "trace": None}
            arguments = func_signature.bind(*args, **kwargs)
            arguments.apply_defaults()
            trace_name = name(arguments) if callable(name) else name
            backend = _ConsoleBackend()
            with ExitStack() as stack:
                root = (
                    stack.enter_context(backend.trace(trace_name))
                    if _current_trace.get() is None
                    else None
                )
                if model is not None:
                    model_name = model(arguments) if callable(model) else model
                    record = stack.enter_context(
                        backend.generation(trace_name, model=model_name)
                    )
                elif root is not None:
                    record = root
                else:
                    record = stack.enter_context(backend.span(trace_name))
                if inject_trace:
                    arguments.arguments["trace"] = record
                try:
                    input_tokens = None
                    documents = False
                    if model is not None:
                        messages = arguments.arguments["messages"]
                        input_tokens = _estimate_model_tokens(messages)
                        documents = (
                            isinstance(messages, list)
                            and bool(messages)
                            and all(isinstance(item, Document) for item in messages)
                        )
                    yield arguments, record, input_tokens, documents
                except BaseException:
                    record.metadata["status"] = "error"
                    raise

        def finish(
            record: _Trace | _Span, tokens: int | None, docs: bool, output: Any
        ) -> None:
            if tokens is not None:
                generation = cast(_Span, record)
                generation.usage = {
                    "input": tokens,
                    "output": 0 if docs else _estimate_model_tokens(output),
                }

        if isasyncgenfunction(func):

            @wraps(func)
            async def async_generator(*args: Any, **kwargs: Any) -> AsyncIterator[Any]:
                with context(args, kwargs) as (arguments, _, _, _):
                    async with aclosing(
                        func(*arguments.args, **arguments.kwargs)
                    ) as stream:
                        async for item in stream:
                            yield item

            return cast(_TraceFunction, async_generator)

        if isgeneratorfunction(func):

            @wraps(func)
            def generator(*args: Any, **kwargs: Any) -> Iterator[Any]:
                with (
                    context(args, kwargs) as (arguments, _, _, _),
                    closing(func(*arguments.args, **arguments.kwargs)) as stream,
                ):
                    return (yield from stream)

            return cast(_TraceFunction, generator)

        if iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                with context(args, kwargs) as (arguments, record, tokens, docs):
                    output = await func(*arguments.args, **arguments.kwargs)
                    finish(record, tokens, docs, output)
                    return output

            return cast(_TraceFunction, async_wrapper)

        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            with context(args, kwargs) as (arguments, record, tokens, docs):
                output = func(*arguments.args, **arguments.kwargs)
                finish(record, tokens, docs, output)
                return output

        return cast(_TraceFunction, sync_wrapper)

    return decorate


def flush() -> None:
    """控制台输出为同步写入，无需额外刷新缓冲区。"""
    _ConsoleBackend().flush()
