"""通过智谱的兼容接口创建并记录聊天模型错误。"""

import json  # 解析智谱接口返回的错误详情。
from collections.abc import AsyncIterator  # 标注异步流的返回类型。
from typing import Any  # 透传模型的流式调用参数。

import openai  # 识别兼容接口返回的 HTTP 错误。
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun  # 保留流式回调。
from langchain_core.messages import BaseMessage  # 标注模型输入消息。
from langchain_core.outputs import ChatGenerationChunk  # 标注流式输出块。
from langchain_openai import ChatOpenAI  # 使用兼容 OpenAI 协议的模型客户端。

from app.config.config import settings  # 读取模型、密钥和接口地址。
from app.observability import get_logger  # 延续现有结构化日志。

_LOG = get_logger("crypto_analysis.llm")  # 复用原有模型日志名称。


class LoggedChatOpenAI(ChatOpenAI):  # 在 ChatOpenAI 的流式调用外记录智谱接口错误。
    """保留原有的智谱 HTTP 错误日志与明确异常。"""

    async def _astream(  # 透传正常流式响应，只处理服务商的 HTTP 错误。
        self,  # 当前模型实例。
        messages: list[BaseMessage],  # 本轮聊天消息。
        stop: list[str] | None = None,  # 可选停止词。
        run_manager: AsyncCallbackManagerForLLMRun | None = None,  # 流式回调管理器。
        **kwargs: Any,  # 原样透传其他调用参数。
    ) -> AsyncIterator[ChatGenerationChunk]:  # 返回 LangChain 的消息块。
        try:  # 正常数据流交给 ChatOpenAI 解析。
            async for chunk in super()._astream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            ):  # 不改动流式回调行为。
                yield chunk  # 向原有分析图交付消息块。
        except openai.APIStatusError as exc:  # 保留原有的 HTTP 错误记录入口。
            code, reason = self._response_error(
                exc.response.content
            )  # 从响应体读取智谱错误码和原因。
            reason = reason.replace(
                settings.zhipuai_api_key, "[已隐藏]"
            )  # 避免密钥进入日志。
            _LOG.error(  # 维持原有的结构化错误事件。
                "llm.provider.error",  # 日志消息与事件名一致。
                extra={  # 保留现有日志字段。
                    "event": "llm.provider.error",  # 标记提供商错误事件。
                    "provider": "zhipuai",  # 模型服务仍然是智谱。
                    "model": self.model_name,  # 记录出错的模型名。
                    "http_status": exc.status_code,  # 记录 HTTP 状态码。
                    "provider_code": code,  # 记录智谱错误码。
                    "provider_message": reason,  # 记录脱敏后的原因。
                    "content_type": exc.response.headers.get(
                        "content-type", "<缺失>"
                    ),  # 记录响应类型。
                },  # 结束结构化日志字段。
            )  # 完成错误日志。
            raise RuntimeError(
                f"智谱 API HTTP {exc.status_code}，错误码 {code}：{reason}"
            ) from exc  # 保留原有错误形式。

    @staticmethod  # 错误体解析不依赖模型实例。
    def _response_error(body: bytes) -> tuple[str, str]:  # 沿用原有智谱错误格式解析。
        try:  # 先尝试读取 JSON 响应。
            payload = json.loads(body)  # 解码服务商返回的错误体。
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
        ):  # 非 JSON 响应没有可解析错误码。
            return "unknown", "上游返回非 JSON 响应"  # 保留原有错误说明。
        if not isinstance(payload, dict) or not isinstance(
            payload.get("error"), dict
        ):  # 检查智谱错误结构。
            return "unknown", "上游响应缺少错误详情"  # 保留原有缺失说明。
        error = payload["error"]  # 获取智谱错误对象。
        return (
            str(error.get("code", "unknown")),
            str(  # 返回字符串错误码和限长消息。
                error.get("message", "上游未提供错误说明")  # 消息缺失时给出明确说明。
            )[:200],
        )  # 限制日志中的错误消息长度。


def _make_model(model_name: str) -> LoggedChatOpenAI:  # 为快模型和主模型共用连接配置。
    api_key = settings.zhipuai_api_key.strip()  # 从已有 ZHIPUAI_API_KEY 读取密钥。
    if not api_key or api_key.startswith("xxxx"):  # 缺少密钥或仍使用示例占位值时停止。
        raise ValueError("未配置 ZHIPUAI_API_KEY")  # 给出明确配置错误。
    base_url = settings.model_base_url.strip()  # 从配置读取智谱兼容接口地址。
    if not base_url:  # 空接口地址无法创建有效模型。
        raise ValueError("未配置智谱 API 地址")  # 保留原有地址错误语义。
    return LoggedChatOpenAI(  # 使用用户指定的 ChatOpenAI 接入方式。
        model=model_name,  # 保留配置中的 GLM 模型名。
        api_key=api_key,  # 继续使用现有智谱密钥。
        base_url=base_url,  # 将请求发送到智谱兼容接口。
        max_retries=0,  # 遵循示例，不在 SDK 层重试请求。
        timeout=60,  # 保留原有流式请求的 60 秒超时。
        disabled_params={
            "parallel_tool_calls": None
        },  # 智谱接口未声明此参数，避免结构化调用附带发送。
    )  # 返回带原有错误日志的聊天模型。


def make_smart_llm() -> LoggedChatOpenAI:  # 创建质量优先的主模型。
    """创建决策和写作模型实例。"""
    return _make_model(settings.smart_model)  # 沿用原有主模型配置。


def make_fast_llm() -> LoggedChatOpenAI:  # 创建速度优先的快模型。
    """创建并行执行模型实例。"""
    return _make_model(settings.fast_model)  # 沿用原有快模型配置。
