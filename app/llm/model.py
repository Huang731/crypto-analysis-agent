"""统一模型创建、调用策略与百炼协议适配。"""

from __future__ import annotations

import asyncio
import json
import ssl
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from http import HTTPStatus
from typing import Any, Literal, TypeVar

import httpx
import truststore
from dashscope import (
    Generation,
    MultiModalConversation,
    TextEmbedding,
    TextReRank,
)
from httpx_sse import aconnect_sse
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import RunnableBinding
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, SecretStr

from app.config.config import settings

_Schema = TypeVar("_Schema", bound=BaseModel)


# 模型创建与参数绑定


def _make_bailian_llm(
    model_name: str,
    *,
    protocol: Literal["text", "multimodal"],
    async_client: httpx.AsyncClient | None = None,
    **parameters: Any,
) -> LoggedChatTongyi:
    """按调用方明确传入的协议和参数创建百炼模型。"""
    timeout = 120.0
    model = LoggedChatTongyi(
        model=model_name,
        api_key=_api_key(),
        model_kwargs={
            "base_address": "https://dashscope.aliyuncs.com/api/v1",
            "request_timeout": timeout,
            **parameters,
        },
        max_retries=1,
        request_timeout=timeout,
        multimodal=protocol == "multimodal",
        async_client=async_client,
    )
    # 父类 pre_init 注入的默认值会被新版 LangChain 误认为显式关闭流式。
    # 保留默认按回调决定是否流式的行为，让报告正文继续进入图的消息流。
    model.model_fields_set.discard("streaming")
    return model


def make_bailian_smart_llm(
    *, async_client: httpx.AsyncClient | None = None
) -> LoggedChatTongyi:
    """创建决策和写作模型，保留模型自身的默认思考配置。"""
    return _make_bailian_llm(
        settings.bailian_smart_model,
        protocol="text",
        async_client=async_client,
    )


def make_bailian_fast_llm(
    *, async_client: httpx.AsyncClient | None = None
) -> LoggedChatTongyi:
    """创建执行模型，沿用非思考模式。"""
    return _make_bailian_llm(
        settings.bailian_fast_model,
        protocol="multimodal",
        async_client=async_client,
        enable_thinking=False,
    )


def _zhipuai_api_key() -> str:
    """请求智谱前校验密钥，避免未配置主题研究时阻断其他功能启动。"""
    key = settings.zhipuai_api_key.strip()
    if not key or key.startswith("xxxx"):
        raise ValueError("未配置 ZHIPUAI_API_KEY")
    return key


def _make_zhipuai_llm(
    model_name: str, *, async_client: httpx.AsyncClient | None = None
) -> ChatOpenAI:
    """主题研究直连智谱，保留请求超时，重试由业务调用方负责。"""
    return ChatOpenAI(
        model=model_name,
        api_key=_zhipuai_api_key,
        base_url="https://open.bigmodel.cn/api/paas/v4/",
        timeout=120.0,
        max_retries=0,
        use_responses_api=False,
        stream_usage=True,
        http_async_client=async_client,
    )


def make_glm_smart_llm(*, async_client: httpx.AsyncClient | None = None) -> ChatOpenAI:
    """创建主题研究的汇总、写作和审稿模型，思考配置由节点绑定。"""
    return _make_zhipuai_llm(settings.glm_smart_model, async_client=async_client)


def make_glm_4_flash_llm(
    *, async_client: httpx.AsyncClient | None = None
) -> ChatOpenAI:
    """创建主题研究的拆题和工具执行模型。"""
    return _make_zhipuai_llm(settings.glm_4_flash_model, async_client=async_client)


# 统一调用：超时与结果类型，重试由业务调用方负责


async def ainvoke_model(
    model: Any,
    messages: LanguageModelInput,
    *,
    timeout: float | None = None,
) -> Any:
    """执行单次模型请求并控制超时，重试由业务调用方负责。"""
    return await asyncio.wait_for(
        model.ainvoke(messages),
        timeout,
    )


async def ainvoke_structured(
    model: Any,
    messages: LanguageModelInput,
    *,
    schema: type[_Schema],
    timeout: float | None = None,
) -> _Schema:
    """单次强制通过工具调用返回目标对象，保留调用超时。"""
    binding = model if isinstance(model, RunnableBinding) else None
    if binding is not None:
        model = binding.bound
    structured = model.with_structured_output(
        schema, method="function_calling", tool_choice="required"
    )
    if binding is not None:
        structured = structured.bind(**binding.kwargs).with_config(binding.config)
    return await ainvoke_model(
        structured,
        messages,
        timeout=timeout,
    )


def invoke_vision(messages: LanguageModelInput) -> AIMessage:
    """同步调用视觉模型，不额外重试。"""
    return _make_bailian_llm(
        settings.vision_model,
        protocol="multimodal",
        temperature=0.1,
        max_tokens=4096,
    ).invoke(messages)


# ChatTongyi 适配：连接、协议与流式响应


@asynccontextmanager
async def model_session(
    client: httpx.AsyncClient | None = None,
) -> AsyncIterator[httpx.AsyncClient]:
    """复用模型连接；未传入连接时负责创建和关闭。"""
    if client is not None:
        yield client
        return
    async with httpx.AsyncClient(
        verify=truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT), trust_env=True
    ) as session:
        yield session


def _api_key() -> str:
    key = settings.dashscope_api_key.strip()
    if not key or key.startswith("xxxx"):
        raise ValueError("未配置 DASHSCOPE_API_KEY")
    return key


def _check_response(response: Any) -> Any:
    """保留上游状态码与错误详情，脱敏后向调用方报告。"""
    if isinstance(response, httpx.Response):
        if response.status_code != HTTPStatus.OK:
            reason = response.text.replace(_api_key(), "[已隐藏]")[:200]
            raise RuntimeError(f"百炼 API HTTP {response.status_code}：{reason}")
        return response.json()
    if response.status_code != HTTPStatus.OK:
        reason = str(response.message).replace(_api_key(), "[已隐藏]")[:200]
        raise RuntimeError(
            f"百炼 API HTTP {response.status_code}，错误码 {response.code}：{reason}"
        )
    return response


class LoggedChatTongyi(ChatTongyi):
    """保留 LangChain 协议，通过 HTTPX 异步请求百炼原生接口。"""

    multimodal: bool = False
    request_timeout: float = Field(default=120, gt=0)
    async_client: httpx.AsyncClient | None = Field(
        default=None, exclude=True, repr=False
    )
    top_p: float | None = None

    def with_structured_output(
        self,
        schema: type[_Schema],
        *,
        method: Literal["function_calling"] = "function_calling",
        tool_choice: Literal["required"] = "required",
        include_raw: bool = False,
    ) -> Any:
        """沿用父类工具解析，仅补齐统一入口的调用方式与强制工具参数。"""
        if method != "function_calling":
            raise ValueError("百炼结构化输出仅支持 function_calling")
        structured = super().with_structured_output(schema, include_raw=include_raw)
        return structured.bind(tool_choice=tool_choice)

    @property
    def _default_params(self) -> dict[str, Any]:
        params = super()._default_params
        if self.top_p is None:
            # 原客户端未覆盖采样参数，继续使用模型服务端默认值。
            params.pop("top_p", None)
        return params

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        # 显式选择视觉接口，避免 qwen3.5-ocr 被父类按名称分到文本接口。
        self.client = MultiModalConversation if self.multimodal else Generation

    def _invocation_params(
        self, messages: list[BaseMessage], stop: Any, **kwargs: Any
    ) -> dict[str, Any]:
        params = super()._invocation_params(messages, stop, **kwargs)
        if params.get("stream"):
            # 直接消费原生增量响应，避免 SDK 的累计合并与二次截取。
            params["incremental_output"] = True
        # 标准工具消息可能只包含 tool_calls，补齐原生协议要求的字段。
        for original, message in zip(messages, params["messages"], strict=True):
            if isinstance(original, AIMessage) and original.tool_calls:
                message["tool_calls"] = [
                    {
                        "id": call["id"],
                        "type": "function",
                        "function": {
                            "name": call["name"],
                            "arguments": json.dumps(call["args"], ensure_ascii=False),
                        },
                    }
                    for call in original.tool_calls
                ]
            if self.multimodal and isinstance(message["content"], list):
                content = []
                for item in message["content"]:
                    if item.get("type") == "image_url":
                        content.append({"image": item["image_url"]["url"]})
                    elif item.get("type") == "text":
                        content.append({"text": item["text"]})
                    else:
                        content.append(item)
                message["content"] = content
            elif self.multimodal and isinstance(message["content"], str):
                message["content"] = [{"text": message["content"]}]
        return params

    def completion_with_retry(self, **kwargs: Any) -> Any:
        # 保留父类钩子名称，同步请求直接执行一次。
        return _check_response(self.client.call(**kwargs))

    def _generate(self, *args: Any, **kwargs: Any) -> ChatResult:
        result = super()._generate(*args, **kwargs)
        for generation in result.generations:
            if self.multimodal and isinstance(generation.message.content, list):
                generation.message.content = "".join(
                    item["text"] for item in generation.message.content
                )
        return result

    async def _async_responses(
        self, params: dict[str, Any]
    ) -> AsyncIterator[dict[str, Any]]:
        """按百炼原生协议发送请求，普通响应和 SSE 均复用 HTTPX 连接。"""
        parameters = params.copy()
        base_address = parameters.pop("base_address")
        api_key = parameters.pop("api_key")
        model = parameters.pop("model")
        messages = parameters.pop("messages")
        parameters.pop("request_timeout")
        stream = parameters.pop("stream", False)
        headers = parameters.pop("headers", {}).copy()
        headers["Authorization"] = f"Bearer {api_key}"
        task = "multimodal-generation" if self.multimodal else "text-generation"
        url = f"{base_address}/services/aigc/{task}/generation"
        body = {
            "model": model,
            "input": {"messages": messages},
            "parameters": parameters,
        }
        async with model_session(self.async_client) as session:
            if stream:
                headers["X-DashScope-SSE"] = "enable"
                async with aconnect_sse(
                    session,
                    "POST",
                    url,
                    json=body,
                    headers=headers,
                    timeout=self.request_timeout,
                ) as source:
                    if source.response.status_code != HTTPStatus.OK:
                        await source.response.aread()
                        _check_response(source.response)
                    async for event in source.aiter_sse():
                        if event.event == "error":
                            reason = event.data.replace(api_key, "[已隐藏]")[:200]
                            raise RuntimeError(f"百炼 API 流式错误：{reason}")
                        if event.data == "[DONE]":
                            return
                        yield event.json()
                return
            response = await session.post(
                url, json=body, headers=headers, timeout=self.request_timeout
            )
            yield _check_response(response)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.streaming:
            # 父类在 streaming=True 时使用 self._astream，不会进入线程池。
            return await super()._agenerate(messages, stop, run_manager, **kwargs)
        params = self._invocation_params(messages, stop, **kwargs)
        responses = self._async_responses(params)
        try:
            response = await asyncio.wait_for(
                anext(responses), timeout=self.request_timeout
            )
        finally:
            await responses.aclose()
        payload = self._normalize_response(response)
        generation = ChatGeneration(**self._chat_generation_from_qwen_resp(payload))
        self._set_usage(generation.message, payload)
        return ChatResult(
            generations=[generation], llm_output={"model_name": self.model_name}
        )

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        params = self._invocation_params(messages, stop, stream=True, **kwargs)
        deadline = asyncio.get_running_loop().time() + self.request_timeout
        responses = self._async_responses(params)
        seen_metadata: dict[str, Any] = {}
        try:
            while True:
                remaining = deadline - asyncio.get_running_loop().time()
                response = await asyncio.wait_for(anext(responses), timeout=remaining)
                payload = self._normalize_response(response)
                finish_reason = payload["output"]["choices"][0]["finish_reason"]
                final = finish_reason not in (None, "null", "")
                chunk = ChatGenerationChunk(
                    **self._chat_generation_from_qwen_resp(
                        payload, is_chunk=True, is_last_chunk=final
                    )
                )
                # 固定元数据只保留一次；文本和工具参数继续按增量传递。
                if chunk.generation_info:
                    for key, value in list(chunk.generation_info.items()):
                        if seen_metadata.get(key) == value:
                            del chunk.generation_info[key]
                        else:
                            seen_metadata[key] = value
                    if "token_usage" in chunk.generation_info:
                        self._set_usage(chunk.message, payload)
                if run_manager:
                    await run_manager.on_llm_new_token(chunk.text, chunk=chunk)
                yield chunk
        except StopAsyncIteration:
            return
        finally:
            await responses.aclose()

    @staticmethod
    def _normalize_response(response: Any) -> dict[str, Any]:
        """多模态文本转换为标准正文，保留工具调用、推理和用量字段。"""
        payload = dict(response)
        message = payload["output"]["choices"][0]["message"]
        if isinstance(message["content"], list):
            # 只复制正文所在的层级，避免修改原始响应。
            output = dict(payload["output"])
            choices = list(output["choices"])
            choice = dict(choices[0])
            message = dict(message)
            message["content"] = "".join(item["text"] for item in message["content"])
            choice["message"] = message
            choices[0] = choice
            output["choices"] = choices
            payload["output"] = output
        return payload

    @staticmethod
    def _set_usage(message: Any, payload: dict[str, Any]) -> None:
        usage = payload["usage"]
        message.usage_metadata = {
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
            "total_tokens": usage["input_tokens"] + usage["output_tokens"],
        }


# 向量与重排


def get_embeddings() -> BailianEmbeddings:
    """统一向量模型入口，不改变现有 FAISS 维度及索引目录。"""
    return BailianEmbeddings(
        model=settings.embedding_model,
        api_key=SecretStr(_api_key()),
        base_address="https://dashscope.aliyuncs.com/api/v1",
    )


class BailianEmbeddings(BaseModel, Embeddings):
    """保留向量模型及默认维度，每次显式传入密钥、地址和超时。"""

    model: str
    api_key: SecretStr = Field(repr=False)
    base_address: str

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), settings.embed_batch_size):
            vectors.extend(
                self._embed(
                    texts[start : start + settings.embed_batch_size], "document"
                )
            )
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], "query")[0]

    def _embed(self, texts: list[str], text_type: str) -> list[list[float]]:
        response = _check_response(
            TextEmbedding.call(
                model=self.model,
                api_key=self.api_key.get_secret_value(),
                base_address=self.base_address,
                request_timeout=120.0,
                input=texts,
                text_type=text_type,
            ),
        )
        items = sorted(
            response.output["embeddings"], key=lambda item: item["text_index"]
        )
        if len(items) != len(texts):
            raise ValueError("百炼向量返回数量与输入数量不一致")
        return [item["embedding"] for item in items]


def rerank(query: str, documents: list[str]) -> Any:
    """只封装重排模型请求，文档排序与元数据合并由检索层保留。"""
    return _check_response(
        TextReRank.call(
            api_key=_api_key(),
            base_address="https://dashscope.aliyuncs.com/api/v1",
            request_timeout=120.0,
            model=settings.rerank_model,
            query=query,
            documents=documents,
            top_n=len(documents),
            return_documents=False,
        ),
    )
