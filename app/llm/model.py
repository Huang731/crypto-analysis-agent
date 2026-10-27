"""统一模型创建、调用策略与服务商协议适配。"""

from __future__ import annotations

import ssl
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from http import HTTPStatus
from json import JSONDecodeError
from typing import Any, TypeVar

import httpx
import truststore
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import Runnable, RunnableBinding, RunnableConfig
from langchain_openai import ChatOpenAI
from openai import APIStatusError, OpenAI
from pydantic import BaseModel, Field, SecretStr

from app.core.config import settings

_Schema = TypeVar("_Schema", bound=BaseModel)


class BailianFreeQuotaExhaustedError(APIStatusError):
    """百炼模型免费额度已耗尽，仅此错误允许切换备用模型。"""


# 模型创建与参数绑定


def _make_bailian_llm(
    model_name: str,
    **parameters: Any,
) -> ChatOpenAI:
    """通过 OpenAI 兼容接口调用百炼，消息与流式处理使用官方实现。"""
    tls_context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    return ChatOpenAI(
        model=model_name,
        api_key=_api_key(),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        timeout=120.0,
        streaming=True,
        stream_usage=True,
        use_responses_api=False,
        http_client=httpx.Client(
            verify=tls_context,
            event_hooks={"response": [_check_bailian_quota]},
        ),
        http_async_client=httpx.AsyncClient(
            verify=tls_context,
            event_hooks={"response": [_acheck_bailian_quota]},
        ),
        **parameters,
    )


def _make_bailian_llm_with_fallbacks(
    model_names: list[str],
    **parameters: Any,
) -> Runnable[LanguageModelInput, AIMessage]:
    """按配置顺序创建模型，仅免费额度耗尽时切换。"""
    models = [_make_bailian_llm(name, **parameters) for name in model_names]
    if len(models) == 1:
        return models[0]
    return models[0].with_fallbacks(
        models[1:], exceptions_to_handle=(BailianFreeQuotaExhaustedError,)
    )


def make_bailian_smart_llm() -> Runnable[LanguageModelInput, AIMessage]:
    """创建决策和写作模型，保留模型自身的默认思考配置。"""
    return _make_bailian_llm_with_fallbacks(settings.bailian_smart_models)


def make_bailian_fast_llm() -> Runnable[LanguageModelInput, AIMessage]:
    """创建执行模型，思考开关由调用方设置。"""
    return _make_bailian_llm_with_fallbacks(settings.bailian_fast_models)


def make_qwen3_max_llm() -> Runnable[LanguageModelInput, AIMessage]:
    """按列表创建百炼评测裁判，仅免费额度耗尽时切换下一项。"""
    return _make_bailian_llm_with_fallbacks(
        settings.qwen3_max_model,
        max_retries=0,
        extra_body={"enable_thinking": False},
    )


def make_qwen3_8b_llm(
    *,
    async_client: httpx.AsyncClient | None = None,
    enable_thinking: bool = False,
) -> ChatOpenAI:
    """创建 Qwen3-8B，按节点配置思考开关，温度统一为 0.1。"""
    return ChatOpenAI(
        model=settings.qwen3_8b_model,
        api_key=_siliconflow_api_key,
        base_url="https://api.siliconflow.cn/v1",
        temperature=0.1,
        timeout=120.0,
        max_retries=1,
        extra_body={"enable_thinking": enable_thinking, "max_tokens": 8192},
        use_responses_api=False,
        stream_usage=True,
        http_async_client=async_client,
    )


def _zhipuai_api_key(model_name: str) -> str:
    """请求智谱前按模型系列选择并校验对应密钥。"""
    if model_name.lower().startswith("glm-4"):
        key = settings.zhipuai_4_api_key.strip()
        variable_name = "ZHIPUAI_4_API_KEY"
    elif model_name.lower().startswith("glm-5"):
        key = settings.zhipuai_5_api_key.strip()
        variable_name = "ZHIPUAI_5_API_KEY"
    else:
        raise ValueError(f"不支持的智谱模型系列：{model_name}")
    if not key or key.startswith("xxxx"):
        raise ValueError(f"未配置 {variable_name}")
    return key


def _make_zhipuai_llm(
    model_name: str,
    *,
    async_client: httpx.AsyncClient | None = None,
) -> ChatOpenAI:
    """主题研究直连智谱，保留请求超时，重试由业务调用方负责。"""
    return ChatOpenAI(
        model=model_name,
        api_key=lambda: _zhipuai_api_key(model_name),
        base_url="https://open.bigmodel.cn/api/paas/v4/",
        timeout=120.0,
        max_retries=0,
        use_responses_api=False,
        stream_usage=True,
        http_async_client=async_client,
    )


def make_glm_4_5_air_llm(
    *, async_client: httpx.AsyncClient | None = None
) -> ChatOpenAI:
    """创建配置指定的智谱 Air 模型，思考开关由调用节点设置。"""
    return _make_zhipuai_llm(settings.glm_smart_model, async_client=async_client)


def make_glm_4_flash_llm(
    *, async_client: httpx.AsyncClient | None = None
) -> ChatOpenAI:
    """创建主题研究的拆题和工具执行模型。"""
    return _make_zhipuai_llm(settings.glm_4_flash_model, async_client=async_client)


def make_glm_5_3_flash_llm(
    *, async_client: httpx.AsyncClient | None = None
) -> ChatOpenAI:
    """创建 RAG 评测裁判模型，沿用智谱 5 系列密钥与统一调用配置。"""
    return _make_zhipuai_llm(settings.glm_5_3_flash_model, async_client=async_client)


# 统一调用与结构化结果解析，重试由业务调用方负责


async def ainvoke_model(
    model: Any,
    messages: LanguageModelInput,
) -> Any:
    """执行单次模型请求，重试由业务调用方负责。"""
    return await model.ainvoke(messages)


async def ainvoke_structured(
    model: Any,
    messages: list[BaseMessage],
    *,
    schema: type[_Schema],
) -> _Schema:
    """统一使用 JSON 模式，返回经过 Schema 校验的对象。"""
    parameters: dict[str, Any] = {}
    config: RunnableConfig = {}
    if isinstance(model, RunnableBinding):
        parameters = model.kwargs
        config = model.config
        model = model.bound
    structured = model.with_structured_output(schema, method="json_mode")
    return await ainvoke_model(
        structured.bind(**parameters).with_config(config),
        messages,
    )


def invoke_vision(messages: LanguageModelInput) -> AIMessage:
    """同步调用视觉模型，不额外重试。"""
    return _make_bailian_llm(
        settings.vision_model,
        temperature=0.1,
        max_tokens=4096,
    ).invoke(messages)


# 连接管理与百炼额度识别


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


def _check_bailian_quota(response: httpx.Response) -> None:
    """只识别免费额度耗尽，其他 HTTP 错误继续由 OpenAI SDK 处理。"""
    if response.status_code != HTTPStatus.FORBIDDEN:
        return
    response.read()
    try:
        body = response.json()
    except JSONDecodeError:
        return
    if not isinstance(body, dict):
        return
    error = body.get("error", body)
    if isinstance(error, dict) and error.get("code") == "AllocationQuota.FreeTierOnly":
        raise BailianFreeQuotaExhaustedError(
            "百炼模型免费额度已耗尽", response=response, body=body
        )


async def _acheck_bailian_quota(response: httpx.Response) -> None:
    """异步读取额度错误响应，不预读正常的流式正文。"""
    if response.status_code == HTTPStatus.FORBIDDEN:
        await response.aread()
        _check_bailian_quota(response)


# 向量与重排


def _siliconflow_api_key() -> str:
    """硅基流动请求共用环境密钥，未配置时直接报错。"""
    key = settings.siliconflow_api_key.strip()
    if not key:
        raise ValueError("未配置 SILICONFLOW_API_KEY")
    return key


def get_embeddings() -> SiliconFlowEmbeddings:
    """创建硅基流动向量适配器，索引维度及目录沿用配置。"""
    return SiliconFlowEmbeddings(
        model=settings.embedding_model,
        api_key=SecretStr(_siliconflow_api_key()),
        base_address="https://api.siliconflow.cn/v1",
    )


class SiliconFlowEmbeddings(BaseModel, Embeddings):
    """保留分批、排序及数量校验，通过兼容接口获取文本向量。"""

    model: str
    api_key: SecretStr = Field(repr=False)
    base_address: str

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), settings.embed_batch_size):
            vectors.extend(
                self._embed(texts[start : start + settings.embed_batch_size])
            )
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]

    def _embed(self, texts: list[str]) -> list[list[float]]:
        with OpenAI(
            api_key=self.api_key.get_secret_value(),
            base_url=self.base_address,
            timeout=120.0,
            max_retries=1,
        ) as client:
            response = client.embeddings.create(
                model=self.model,
                input=texts,
                encoding_format="float",
            )
        items = sorted(response.data, key=lambda item: item.index)
        if len(items) != len(texts):
            raise ValueError("硅基流动向量返回数量与输入数量不一致")
        vectors = [item.embedding for item in items]
        if any(len(vector) != settings.embedding_dimensions for vector in vectors):
            raise ValueError("硅基流动向量返回维度与配置不一致")
        return vectors


def rerank(query: str, documents: list[str]) -> dict[str, Any]:
    """只封装重排模型请求，文档排序与元数据合并由检索层保留。"""
    with OpenAI(
        api_key=_siliconflow_api_key(),
        base_url="https://api.siliconflow.cn/v1",
        timeout=120.0,
        max_retries=1,
    ) as client:
        return client.post(
            "/rerank",
            cast_to=dict[str, Any],
            body={
                "model": settings.rerank_model,
                "query": query,
                "documents": documents,
                "top_n": len(documents),
                "return_documents": False,
            },
        )
