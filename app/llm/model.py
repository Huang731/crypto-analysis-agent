"""统一百炼模型创建、调用策略和协议转换。"""

from __future__ import annotations

import asyncio
import copy
import json
import ssl
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from http import HTTPStatus
from typing import Any, Literal, TypeVar

import aiohttp
import truststore
from dashscope import (
    AioGeneration,
    AioMultiModalConversation,
    Generation,
    MultiModalConversation,
    TextEmbedding,
    TextReRank,
)
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun
from langchain_core.callbacks.base import Callbacks
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable, RunnableBinding
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, SecretStr
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

from app.config.config import settings

_Result = TypeVar("_Result")
_Schema = TypeVar("_Schema", bound=BaseModel)


@asynccontextmanager
async def model_session() -> AsyncIterator[aiohttp.ClientSession]:
    """由应用或评测案例持有连接，结束时关闭，不修改 SDK 全局配置。"""
    connector = aiohttp.TCPConnector(ssl=truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT))
    async with aiohttp.ClientSession(connector=connector, trust_env=True) as session:
        yield session


@asynccontextmanager
async def _request_session(
    client: aiohttp.ClientSession | None,
) -> AsyncIterator[aiohttp.ClientSession]:
    """复用明确注入的连接；独立调用则在本次请求结束时释放连接。"""
    if client is not None:
        yield client
    else:
        async with model_session() as session:
            yield session


def _api_key() -> str:
    key = settings.dashscope_api_key.strip()
    if not key or key.startswith("xxxx"):
        raise ValueError("未配置 DASHSCOPE_API_KEY")
    return key


def _native_address(address: str) -> str:
    """将已有兼容接口配置确定性转换为同地域的 DashScope 原生地址。"""
    address = address.strip().rstrip("/")
    if not address:
        raise ValueError("未配置百炼 API 地址")
    if address.endswith("/compatible-mode/v1"):
        address = address.removesuffix("/compatible-mode/v1") + "/api/v1"
    if not address.endswith("/api/v1"):
        raise ValueError("百炼原生 API 地址必须以 /api/v1 结尾")
    return address


def _check_response(response: Any, model_name: str) -> Any:
    """保留上游状态码与错误详情，脱敏后向调用方报告。"""
    if response.status_code != HTTPStatus.OK:
        reason = str(response.message).replace(_api_key(), "[已隐藏]")[:200]
        raise RuntimeError(
            f"百炼 API HTTP {response.status_code}，错误码 {response.code}：{reason}"
        )
    return response


def model_error_message(error: Exception) -> str:
    """统一错误脱敏，避免对空密钥执行字符串替换。"""
    message = str(error)
    if settings.dashscope_api_key:
        message = message.replace(settings.dashscope_api_key, "[已隐藏]")
    return message


class LoggedChatTongyi(ChatTongyi):
    """保留 LangChain 协议，异步请求直接使用 DashScope 的 asyncio 客户端。"""

    multimodal: bool = False
    request_timeout: float = Field(default=120, gt=0)
    async_client: Any = Field(default=None, exclude=True, repr=False)
    top_p: float | None = None

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
        if params.get("stream") and params.get("tools"):
            # 原生接口默认可能返回增量；工具流明确请求累计格式，再统一截取前缀。
            params["incremental_output"] = False
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
        # SDK 边界只执行一次，调用策略在统一入口或业务整体校验边界执行。
        return _check_response(self.client.call(**kwargs), self.model_name)

    @staticmethod
    def _normalize_response(response: Any) -> dict[str, Any]:
        """多模态文本转换为标准正文，保留工具调用、推理和用量字段。"""
        # SDK 累计响应会复用嵌套对象，JSON 快照确保前后响应互不影响。
        payload = json.loads(json.dumps(response, ensure_ascii=False))
        message = payload["output"]["choices"][0]["message"]
        if isinstance(message["content"], list):
            message["content"] = "".join(item["text"] for item in message["content"])
        return payload

    @staticmethod
    def _delta_response(
        response: dict[str, Any], previous: dict[str, Any] | None
    ) -> dict[str, Any]:
        """按前缀截取累计响应，避免删除正文或工具参数中的重复片段。"""
        if previous is None:
            return response
        payload = copy.deepcopy(response)
        message = payload["output"]["choices"][0]["message"]
        prior = previous["output"]["choices"][0]["message"]
        for field in ("content", "reasoning_content"):
            if field in message and prior.get(field):
                if not message[field].startswith(prior[field]):
                    raise ValueError(f"累计响应的 {field} 不是前一响应的延续")
                message[field] = message[field][len(prior[field]) :]
        prior_calls = {
            call.get("index", index): call
            for index, call in enumerate(prior.get("tool_calls", []))
        }
        for index, call in enumerate(message.get("tool_calls", [])):
            old = prior_calls.get(call.get("index", index))
            if old is None:
                continue
            if call.get("id") == old.get("id"):
                call.pop("id", None)
            for field in ("name", "arguments"):
                value = call["function"].get(field)
                old_value = old["function"].get(field)
                if value is not None and old_value:
                    if not value.startswith(old_value):
                        raise ValueError(f"累计工具调用的 {field} 不是前一响应的延续")
                    call["function"][field] = value[len(old_value) :]
        return payload

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
        client = AioMultiModalConversation if self.multimodal else AioGeneration
        async with _request_session(self.async_client) as session:
            response = await asyncio.wait_for(
                client.call(**params, session=session), timeout=self.request_timeout
            )
        payload = self._normalize_response(_check_response(response, self.model_name))
        generation = ChatGeneration(**self._chat_generation_from_qwen_resp(payload))
        self._set_usage(generation.message, payload)
        return ChatResult(
            generations=[generation], llm_output={"model_name": self.model_name}
        )

    @staticmethod
    def _set_usage(message: Any, payload: dict[str, Any]) -> None:
        usage = payload["usage"]
        message.usage_metadata = {
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
            "total_tokens": usage["input_tokens"] + usage["output_tokens"],
        }

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        async with _request_session(self.async_client) as session:
            params = self._invocation_params(messages, stop, stream=True, **kwargs)
            params["session"] = session
            client = AioMultiModalConversation if self.multimodal else AioGeneration
            deadline = asyncio.get_running_loop().time() + self.request_timeout
            responses = await asyncio.wait_for(
                client.call(**params), timeout=self.request_timeout
            )
            previous = None
            seen_metadata: dict[str, Any] = {}
            try:
                while True:
                    remaining = deadline - asyncio.get_running_loop().time()
                    response = await asyncio.wait_for(
                        anext(responses), timeout=remaining
                    )
                    payload = self._normalize_response(
                        _check_response(response, self.model_name)
                    )
                    delta = (
                        payload
                        if params.get("incremental_output", False)
                        else self._delta_response(payload, previous)
                    )
                    previous = payload
                    finish_reason = payload["output"]["choices"][0]["finish_reason"]
                    final = finish_reason not in (None, "null", "")
                    chunk = ChatGenerationChunk(
                        **self._chat_generation_from_qwen_resp(
                            delta, is_chunk=True, is_last_chunk=final
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

    @retry(
        stop=stop_after_attempt(settings.model_max_attempts),
        wait=wait_random_exponential(min=1, max=2),
        reraise=True,
    )
    def _generate(self, *args: Any, **kwargs: Any) -> ChatResult:
        result = super()._generate(*args, **kwargs)
        for generation in result.generations:
            if self.multimodal and isinstance(generation.message.content, list):
                generation.message.content = "".join(
                    item["text"] for item in generation.message.content
                )
        return result


def make_model(
    model_name: str,
    *,
    async_client: aiohttp.ClientSession | None = None,
    thinking: bool | None = None,
    role: Literal["smart", "fast", "vision"] = "smart",
    protocol: Literal["text", "multimodal"] | None = None,
) -> LoggedChatTongyi:
    """按明确角色选择地址与参数，模型名称不用于推断供应商和角色。"""
    timeout = settings.model_timeout
    if protocol is None:
        protocol = {
            "smart": settings.smart_model_protocol,
            "fast": settings.fast_model_protocol,
            "vision": "multimodal",
        }[role]
    params: dict[str, Any] = {
        "base_address": _native_address(settings.model_base_url),
        "request_timeout": timeout,
    }
    if thinking is not None:
        params["enable_thinking"] = thinking
    if role == "vision":
        params.update(temperature=0.1, max_tokens=4096)
    model = LoggedChatTongyi(
        model=model_name,
        api_key=_api_key(),
        model_kwargs=params,
        max_retries=1,
        request_timeout=timeout,
        multimodal=protocol == "multimodal",
        async_client=async_client,
    )
    # 父类 pre_init 注入的默认值会被新版 LangChain 误认为显式关闭流式。
    # 保留默认按回调决定是否流式的行为，让报告正文继续进入图的消息流。
    model.model_fields_set.discard("streaming")
    return model


def make_smart_llm(
    *, async_client: aiohttp.ClientSession | None = None
) -> LoggedChatTongyi:
    """创建决策和写作模型，保留模型自身的默认思考配置。"""
    return make_model(settings.smart_model, async_client=async_client, role="smart")


def make_fast_llm(
    *, async_client: aiohttp.ClientSession | None = None
) -> LoggedChatTongyi:
    """创建执行模型，沿用非思考模式。"""
    return make_model(
        settings.fast_model, async_client=async_client, role="fast", thinking=False
    )


def make_vision_llm() -> LoggedChatTongyi:
    """创建视觉模型，显式指定多模态调用协议。"""
    return make_model(settings.vision_model, role="vision")


def with_thinking(model: Any, enabled: bool) -> Any:
    """生成独立的节点参数绑定，不修改共享模型。"""
    return model.bind(enable_thinking=enabled)


def with_tools(model: Any, tools: Sequence[BaseTool]) -> Any:
    """绑定项目已有的 Pydantic 工具 schema。"""
    if isinstance(model, RunnableBinding):
        return (
            with_tools(model.bound, tools)
            .bind(**model.kwargs)
            .with_config(model.config)
        )
    return model.bind_tools(tools)


def structured_model(model: Any, schema: type[_Schema]) -> Runnable[Any, _Schema]:
    """显式绑定输出工具，只从 Function Calling 参数解析 Pydantic 结果。"""
    if isinstance(model, RunnableBinding):
        return (
            structured_model(model.bound, schema)
            .bind(**model.kwargs)
            .with_config(model.config)
        )
    return model.bind_tools(
        [schema],
        tool_choice={"type": "function", "function": {"name": schema.__name__}},
    ) | PydanticToolsParser(
        tools=[schema],
        first_tool_only=True,
    )


async def await_model_call(
    call: Awaitable[_Result], *, timeout: float | None = None
) -> _Result:
    """为普通调用和 Agent 框架回调统一设置单次总超时。"""
    return await asyncio.wait_for(
        call, timeout=settings.model_timeout if timeout is None else timeout
    )


@retry(
    stop=stop_after_attempt(settings.model_max_attempts),
    wait=wait_random_exponential(min=1, max=2),
    reraise=True,
)
async def ainvoke_model(
    model: Any,
    messages: LanguageModelInput,
    *,
    callbacks: Callbacks = None,
    timeout: float | None = None,
    parse_result: Callable[[Any], Any] | None = None,
    on_attempt: Callable[[], None] | None = None,
) -> Any:
    """统一重试请求和结果解析，业务节点不再包装重试。"""
    if on_attempt is not None:
        on_attempt()
    call = (
        model.ainvoke(messages)
        if callbacks is None
        else model.ainvoke(messages, config={"callbacks": callbacks})
    )
    result = await await_model_call(call, timeout=timeout)
    return result if parse_result is None else parse_result(result)


async def ainvoke_structured(
    model: Any,
    messages: LanguageModelInput,
    *,
    schema: type[_Schema],
    callbacks: Callbacks = None,
    timeout: float | None = None,
) -> _Schema:
    """调用、Pydantic 解析和类型校验整体重试，不叠加 SDK 重试。"""

    def parse(result: Any) -> _Schema:
        if not isinstance(result, schema):
            raise TypeError(f"模型未返回 {schema.__name__} 结构化结果")
        return result

    return await ainvoke_model(
        structured_model(model, schema),
        messages,
        callbacks=callbacks,
        timeout=timeout,
        parse_result=parse,
    )


async def ainvoke_text(
    model: Any,
    messages: LanguageModelInput,
    *,
    on_attempt: Callable[[], None] | None = None,
) -> str:
    """统一文本调用及非空校验，整体最多尝试两次。"""

    def parse(response: Any) -> str:
        if not isinstance(response.content, str):
            raise TypeError("模型未返回文本内容")
        content = response.content.strip()
        if not content:
            raise ValueError("模型返回了空文本")
        return content

    return await ainvoke_model(
        model, messages, parse_result=parse, on_attempt=on_attempt
    )


async def ainvoke_tools(model: Any, messages: LanguageModelInput) -> AIMessage:
    """统一工具响应类型与参数解析检查，错误进入同一层模型重试。"""

    def parse(response: Any) -> AIMessage:
        if not isinstance(response, AIMessage):
            raise TypeError("指标决策模型未返回 AIMessage")
        if response.invalid_tool_calls:
            raise ValueError("指标决策模型返回了无法解析的工具调用")
        return response

    return await ainvoke_model(model, messages, parse_result=parse)


@retry(
    stop=stop_after_attempt(settings.model_max_attempts),
    wait=wait_random_exponential(min=1, max=2),
    reraise=True,
)
async def ainvoke_agent_model(
    call: Callable[[], Awaitable[_Result]], *, timeout: float = 60
) -> _Result:
    """只重试 Agent 当前一轮模型请求，不重复执行工具和整张图。"""
    return await await_model_call(call(), timeout=timeout)


def invoke_vision(messages: LanguageModelInput) -> AIMessage:
    """同步视觉调用沿用模型 _generate 上的单层 @retry。"""
    return make_vision_llm().invoke(messages)


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

    @retry(
        stop=stop_after_attempt(settings.model_max_attempts),
        wait=wait_random_exponential(min=1, max=2),
        reraise=True,
    )
    def _embed(self, texts: list[str], text_type: str) -> list[list[float]]:
        response = _check_response(
            TextEmbedding.call(
                model=self.model,
                api_key=self.api_key.get_secret_value(),
                base_address=self.base_address,
                request_timeout=settings.model_timeout,
                input=texts,
                text_type=text_type,
            ),
            self.model,
        )
        items = sorted(
            response.output["embeddings"], key=lambda item: item["text_index"]
        )
        if len(items) != len(texts):
            raise ValueError("百炼向量返回数量与输入数量不一致")
        return [item["embedding"] for item in items]


def get_embeddings() -> BailianEmbeddings:
    """统一向量模型入口，不改变现有 FAISS 维度及索引目录。"""
    return BailianEmbeddings(
        model=settings.embedding_model,
        api_key=SecretStr(_api_key()),
        base_address=_native_address(settings.model_base_url),
    )


@retry(
    stop=stop_after_attempt(settings.model_max_attempts),
    wait=wait_random_exponential(min=1, max=2),
    reraise=True,
)
def rerank(query: str, documents: list[str]) -> Any:
    """只封装重排模型请求，文档排序与元数据合并由检索层保留。"""
    return _check_response(
        TextReRank.call(
            api_key=_api_key(),
            base_address=_native_address(settings.model_base_url),
            request_timeout=settings.model_timeout,
            model=settings.rerank_model,
            query=query,
            documents=documents,
            top_n=len(documents),
            return_documents=False,
        ),
        settings.rerank_model,
    )
