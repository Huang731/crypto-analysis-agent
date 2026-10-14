"""通过百炼和智谱的 OpenAI 兼容接口创建并记录聊天模型错误。"""

import json
from collections.abc import AsyncIterator
from typing import Any, Literal

import httpx
import openai
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGenerationChunk
from langchain_openai import ChatOpenAI

from app.config.config import settings
from app.observability import get_logger

_LOG = get_logger("crypto_analysis.llm")


class LoggedChatOpenAI(ChatOpenAI):
    """去除流式固定元数据的重复值，并记录对应供应商的 HTTP 错误。"""

    provider_name: Literal["dashscope", "zhipuai"] = "dashscope"

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        # 固定元数据不是文本增量，同一次流中重复发送会被 LangChain 拼接。
        seen_metadata: dict[str, Any] = {}
        stream = super()._astream(
            messages, stop=stop, run_manager=run_manager, **kwargs
        )
        try:
            async for chunk in stream:
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
            reason = reason.replace(self.openai_api_key.get_secret_value(), "[已隐藏]")
            _LOG.error(
                "llm.provider.error",
                extra={
                    "event": "llm.provider.error",
                    "provider": self.provider_name,
                    "model": self.model_name,
                    "http_status": exc.status_code,
                    "provider_code": code,
                    "provider_message": reason,
                    "content_type": exc.response.headers.get("content-type", "<缺失>"),
                },
            )
            provider_label = "智谱" if self.provider_name == "zhipuai" else "百炼"
            raise RuntimeError(
                f"{provider_label} API HTTP {exc.status_code}，错误码 {code}：{reason}"
            ) from exc
        finally:
            # 消费者提前关闭或任务取消时，也关闭父类流，释放 HTTP 响应资源。
            await stream.aclose()

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


def _make_model(
    model_name: str, *, http_async_client: httpx.AsyncClient | None = None
) -> LoggedChatOpenAI:
    # 执行模型按明确配置选择供应商，其余模型使用百炼接口。
    is_fast_model = model_name == settings.fast_model
    provider_name = settings.fast_model_provider if is_fast_model else "dashscope"
    is_zhipu_model = provider_name == "zhipuai"
    api_key = (
        settings.zhipuai_api_key if is_zhipu_model else settings.dashscope_api_key
    ).strip()
    key_name = "ZHIPUAI_API_KEY" if is_zhipu_model else "DASHSCOPE_API_KEY"
    if not api_key or api_key.startswith("xxxx"):
        raise ValueError(f"未配置 {key_name}")
    base_url = (
        settings.fast_model_base_url if is_fast_model else settings.model_base_url
    ).strip()
    if not base_url:
        provider_label = "智谱" if is_zhipu_model else "百炼"
        raise ValueError(f"未配置{provider_label} API 地址")
    return LoggedChatOpenAI(
        model=model_name,
        provider_name=provider_name,
        api_key=api_key,
        base_url=base_url,
        # 执行模型沿用非思考模式，不改变决策模型的思考配置。
        extra_body=(
            {"enable_thinking": False}
            if is_fast_model and provider_name == "dashscope"
            else None
        ),
        # 重试由调用层统一控制，避免 SDK 重试与调用层叠加。
        max_retries=0,
        timeout=120,
        # 沿用现有兼容参数设置，不向上游传递 parallel_tool_calls。
        disabled_params={"parallel_tool_calls": None},
        http_async_client=http_async_client,
    )


def make_smart_llm(
    *, http_async_client: httpx.AsyncClient | None = None
) -> LoggedChatOpenAI:
    """创建决策和写作模型实例。"""
    return _make_model(settings.smart_model, http_async_client=http_async_client)


def make_fast_llm(
    *, http_async_client: httpx.AsyncClient | None = None
) -> LoggedChatOpenAI:
    """创建并行执行模型实例。"""
    return _make_model(settings.fast_model, http_async_client=http_async_client)
