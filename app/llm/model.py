"""统一模型创建、调用策略与百炼协议适配。"""

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
import httpx
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
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable, RunnableBinding
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, SecretStr
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

from app.config.config import settings

_Result = TypeVar("_Result")
_Schema = TypeVar("_Schema", bound=BaseModel)


# 模型创建与参数绑定


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
            "smart": settings.bailian_smart_model_protocol,
            "fast": settings.bailian_fast_model_protocol,
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


def make_bailian_smart_llm(
    *, async_client: aiohttp.ClientSession | None = None
) -> LoggedChatTongyi:
    """创建决策和写作模型，保留模型自身的默认思考配置。"""
    return make_model(
        settings.bailian_smart_model, async_client=async_client, role="smart"
    )


def make_bailian_fast_llm(
    *, async_client: aiohttp.ClientSession | None = None
) -> LoggedChatTongyi:
    """创建执行模型，沿用非思考模式。"""
    return make_model(
        settings.bailian_fast_model,
        async_client=async_client,
        role="fast",
        thinking=False,
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
    """主题研究直连智谱，超时与重试沿用统一调用层。"""
    return ChatOpenAI(
        model=model_name,
        api_key=_zhipuai_api_key,
        base_url=settings.zhipuai_base_url,
        timeout=settings.model_timeout,
        # 由 ainvoke_model 和 ainvoke_agent_model 统一重试，避免叠加。
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


def with_thinking(model: Any, enabled: bool) -> Any:
    """生成独立的节点参数绑定，不修改共享模型。"""
    if isinstance(model, ChatOpenAI):
        return model.bind(
            extra_body={
                **(model.extra_body or {}),
                "thinking": {"type": "enabled" if enabled else "disabled"},
            }
        )
    return model.bind(enable_thinking=enabled)


def with_tools(
    model: Any, tools: Sequence[BaseTool | type[BaseModel]], **kwargs: Any
) -> Any:
    """绑定项目已有的 Pydantic 工具 schema。"""
    if isinstance(model, RunnableBinding):
        return (
            with_tools(model.bound, tools, **kwargs)
            .bind(**model.kwargs)
            .with_config(model.config)
        )
    return model.bind_tools(tools, **kwargs)


def structured_model(model: Any, schema: type[_Schema]) -> Runnable[Any, _Schema]:
    """显式绑定输出工具，只从 Function Calling 参数解析 Pydantic 结果。"""
    return with_tools(
        model,
        [schema],
        tool_choice={"type": "function", "function": {"name": schema.__name__}},
    ) | PydanticToolsParser(
        tools=[schema],
        first_tool_only=True,
    )


# 统一调用：重试、超时与结果类型


@retry(
    stop=stop_after_attempt(settings.model_max_attempts),
    wait=wait_random_exponential(min=1, max=2),
    reraise=True,
)
async def ainvoke_model(
    model: Any,
    messages: LanguageModelInput,
    *,
    timeout: float | None = None,
) -> Any:
    """统一执行模型请求、超时和重试，业务节点不再包装重试。"""
    return await asyncio.wait_for(
        model.ainvoke(messages),
        timeout=settings.model_timeout if timeout is None else timeout,
    )


async def ainvoke_structured(
    model: Any,
    messages: LanguageModelInput,
    *,
    schema: type[_Schema],
    timeout: float | None = None,
) -> _Schema:
    """通过 Function Calling 返回 Pydantic 结果。"""
    return await ainvoke_model(
        structured_model(model, schema),
        messages,
        timeout=timeout,
    )


async def ainvoke_text(model: Any, messages: LanguageModelInput) -> str:
    """调用模型并返回文本正文。"""
    response = await ainvoke_model(model, messages)
    return response.content.strip()


@retry(
    stop=stop_after_attempt(settings.model_max_attempts),
    wait=wait_random_exponential(min=1, max=2),
    reraise=True,
)
async def ainvoke_agent_model(
    call: Callable[[], Awaitable[_Result]], *, timeout: float = 60
) -> _Result:
    """只重试 Agent 当前一轮模型请求，不重复执行工具和整张图。"""
    return await asyncio.wait_for(call(), timeout=timeout)


def invoke_vision(messages: LanguageModelInput) -> AIMessage:
    """同步视觉调用沿用模型 _generate 上的单层 @retry。"""
    return make_model(settings.vision_model, role="vision").invoke(messages)


# ChatTongyi 适配：连接、协议与流式响应


@asynccontextmanager
async def model_session(
    client: aiohttp.ClientSession | None = None,
) -> AsyncIterator[aiohttp.ClientSession]:
    """复用调用方连接；未传入连接时负责创建和关闭。"""
    if client is not None:
        yield client
        return
    connector = aiohttp.TCPConnector(ssl=truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT))
    async with aiohttp.ClientSession(connector=connector, trust_env=True) as session:
        yield session


@asynccontextmanager
async def zhipuai_model_session() -> AsyncIterator[httpx.AsyncClient]:
    """主题研究共享异步连接，随应用或命令行生命周期关闭。"""
    async with httpx.AsyncClient(
        verify=truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ) as session:
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


def _check_response(response: Any) -> Any:
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
    for key in (settings.dashscope_api_key, settings.zhipuai_api_key):
        if key:
            message = message.replace(key, "[已隐藏]")
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
        # 保留父类钩子名称；同步重试由 _generate 上的 @retry 负责。
        return _check_response(self.client.call(**kwargs))

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
        async with model_session(self.async_client) as session:
            response = await asyncio.wait_for(
                client.call(**params, session=session), timeout=self.request_timeout
            )
        payload = self._normalize_response(_check_response(response))
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
        async with model_session(self.async_client) as session:
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
                    payload = self._normalize_response(_check_response(response))
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
        base_address=_native_address(settings.model_base_url),
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
        )
        items = sorted(
            response.output["embeddings"], key=lambda item: item["text_index"]
        )
        if len(items) != len(texts):
            raise ValueError("百炼向量返回数量与输入数量不一致")
        return [item["embedding"] for item in items]


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
    )
