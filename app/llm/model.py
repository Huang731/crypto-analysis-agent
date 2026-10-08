"""通过智谱的兼容接口创建并记录聊天模型错误。"""

import json
from collections.abc import AsyncIterator
from contextvars import ContextVar
from typing import Any

import openai
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGenerationChunk
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, PrivateAttr

from app.config.config import settings
from app.observability import get_logger

_LOG = get_logger("crypto_analysis.llm")


class _StreamDiagnostics(BaseModel):
    """保存单次 Supervisor 流的转换前后数据，不参与消息处理。"""

    # 每层最多保留前 31 条和最后一条，避免逐 token 日志无限增长。
    raw_chunks: list[dict[str, Any]] = Field(default_factory=list)
    parsed_chunks: list[dict[str, Any]] = Field(default_factory=list)
    raw_chunk_count: int = 0
    parsed_chunk_count: int = 0
    raw_content_present: bool = False
    raw_tool_calls_present: bool = False

    @staticmethod
    def save_sample(samples: list[dict[str, Any]], chunk: dict[str, Any]) -> None:
        """达到采样上限后持续更新末条，保留流结束信息。"""
        if len(samples) < 32:
            samples.append(chunk)
        else:
            samples[-1] = chunk


class LoggedChatOpenAI(ChatOpenAI):
    """去除流式固定元数据的重复值，并记录智谱 HTTP 错误。"""

    _stream_diagnostics: ContextVar[_StreamDiagnostics | None] = PrivateAttr(
        default_factory=lambda: ContextVar(
            "supervisor_stream_diagnostics", default=None
        )
    )

    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict[str, Any],
        default_chunk_class: type,
        base_generation_info: dict[str, Any] | None,
    ) -> ChatGenerationChunk | None:
        """在父类转换前取证；这是 SDK 对象数据，不是 HTTP 原文字节。"""
        diagnostics = self._stream_diagnostics.get()
        if diagnostics is not None:
            diagnostics.raw_chunk_count += 1
            diagnostics.save_sample(diagnostics.raw_chunks, chunk)
            # 检查所有 chunk，采样截断也不会丢失正文或工具调用是否存在的信息。
            for choice in chunk.get("choices", []):
                delta = choice.get("delta")
                if isinstance(delta, dict):
                    diagnostics.raw_content_present |= bool(delta.get("content"))
                    diagnostics.raw_tool_calls_present |= bool(delta.get("tool_calls"))
        return super()._convert_chunk_to_generation_chunk(
            chunk, default_chunk_class, base_generation_info
        )

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        # 固定元数据不是文本增量，同一次流中重复发送会被 LangChain 拼接。
        seen_metadata: dict[str, Any] = {}
        # 按交接工具识别调度调用，避免给取数、计算、writer 等调用增加日志。
        tool_names = [
            tool["function"]["name"]
            for tool in kwargs.get("tools", [])
            if tool.get("type") == "function"
        ]
        diagnostics = (
            _StreamDiagnostics()
            if any(
                name
                in {
                    "transfer_to_crypto_analysis_agent",
                    "transfer_to_topic_research_agent",
                }
                for name in tool_names
            )
            else None
        )
        stream = super()._astream(
            messages, stop=stop, run_manager=run_manager, **kwargs
        )
        completed = False
        try:
            while True:
                # 仅在读取下一块时绑定上下文，不能跨向外 yield 保持。
                # 这样独立 Task 并发、同一 Task 交错消费都不会串流。
                token = self._stream_diagnostics.set(diagnostics)
                try:
                    chunk = await anext(stream)
                except StopAsyncIteration:
                    completed = True
                    break
                finally:
                    self._stream_diagnostics.reset(token)
                if diagnostics is not None:
                    diagnostics.parsed_chunk_count += 1
                    diagnostics.save_sample(
                        diagnostics.parsed_chunks,
                        chunk.model_dump(mode="json", serialize_as_any=True),
                    )
                if chunk.generation_info:
                    for key in (
                        "finish_reason",
                        "model_name",
                        "system_fingerprint",
                        "service_tier",
                    ):
                        if key not in chunk.generation_info:
                            continue
                        value = chunk.generation_info[key]
                        if key in seen_metadata and seen_metadata[key] == value:
                            del chunk.generation_info[key]
                        else:
                            seen_metadata[key] = value
                yield chunk
        except openai.APIStatusError as exc:
            code, reason = self._response_error(exc.response.content)
            reason = reason.replace(settings.zhipuai_api_key, "[已隐藏]")
            _LOG.error(
                "llm.provider.error",
                extra={
                    "event": "llm.provider.error",
                    "provider": "zhipuai",
                    "model": self.model_name,
                    "http_status": exc.status_code,
                    "provider_code": code,
                    "provider_message": reason,
                    "content_type": exc.response.headers.get("content-type", "<缺失>"),
                },
            )
            raise RuntimeError(
                f"智谱 API HTTP {exc.status_code}，错误码 {code}：{reason}"
            ) from exc
        finally:
            # 消费者提前关闭或任务取消时，也关闭父类流，释放 HTTP 响应资源。
            await stream.aclose()
            if diagnostics is not None:
                # 只记录响应与工具配置，不记录请求头、密钥或完整输入消息。
                payload = diagnostics.model_dump_json()
                api_key = settings.zhipuai_api_key.strip()
                if api_key:
                    payload = payload.replace(api_key, "[已隐藏]")
                _LOG.info(
                    "supervisor.stream.diagnostics",
                    extra={
                        "event": "supervisor.stream.diagnostics",
                        "model": self.model_name,
                        "run_id": str(run_manager.run_id) if run_manager else None,
                        "tool_names": tool_names,
                        "tool_choice": kwargs.get("tool_choice"),
                        "capture_stage": "openai_sdk_before_langchain_conversion",
                        "stream_completed": completed,
                        "samples_truncated": diagnostics.raw_chunk_count > 32
                        or diagnostics.parsed_chunk_count > 32,
                        **json.loads(payload),
                    },
                )

    @staticmethod
    def _response_error(body: bytes) -> tuple[str, str]:
        try:
            payload = json.loads(body)
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
        ):
            return "unknown", "上游返回非 JSON 响应"
        if not isinstance(payload, dict) or not isinstance(payload.get("error"), dict):
            return "unknown", "上游响应缺少错误详情"
        error = payload["error"]
        return (
            str(error.get("code", "unknown")),
            str(error.get("message", "上游未提供错误说明"))[:200],
        )


def _make_model(model_name: str) -> LoggedChatOpenAI:
    api_key = settings.zhipuai_api_key.strip()
    if not api_key or api_key.startswith("xxxx"):
        raise ValueError("未配置 ZHIPUAI_API_KEY")
    base_url = settings.model_base_url.strip()
    if not base_url:
        raise ValueError("未配置智谱 API 地址")
    return LoggedChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        max_retries=0,
        timeout=60,
        # 智谱兼容接口不支持此参数。
        disabled_params={"parallel_tool_calls": None},
    )


def make_smart_llm() -> LoggedChatOpenAI:
    """创建决策和写作模型实例。"""
    return _make_model(settings.smart_model)


def make_fast_llm() -> LoggedChatOpenAI:
    """创建并行执行模型实例。"""
    return _make_model(settings.fast_model)
