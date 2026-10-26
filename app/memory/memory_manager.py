"""按消息数量压缩旧对话，返回需要整体保存的消息列表。"""

import asyncio

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AnyMessage, BaseMessage, HumanMessage, SystemMessage
from tenacity import retry, stop_after_attempt, wait_random_exponential

from app.core.config import settings
from app.core.tracing import trace_generation_call
from app.llm.model import ainvoke_model
from prompts.summarize_history import history_summary_messages


class MemoryManager:
    """超过阈值才压缩旧历史，近期消息保留原文。"""

    def __init__(self, threshold: int = 10, keep_recent: int = 4) -> None:
        self.threshold = threshold
        self.keep_recent = keep_recent

    async def manage(
        self, llm: BaseChatModel, messages: list[AnyMessage]
    ) -> list[AnyMessage]:
        """统计非系统消息，未超过阈值时保留现有列表。"""
        other = [
            message for message in messages if not isinstance(message, SystemMessage)
        ]
        if len(other) > self.threshold:
            return await compress_with_summary(llm, messages, self.keep_recent)
        return messages


async def compress_with_summary(
    llm: BaseChatModel, messages: list[AnyMessage], keep_recent: int = 4
) -> list[AnyMessage]:
    """旧历史压成一条摘要，系统消息和最近指定条数的消息原样保留。"""
    system_msgs = [
        message for message in messages if isinstance(message, SystemMessage)
    ]
    other_msgs = [
        message for message in messages if not isinstance(message, SystemMessage)
    ]
    if len(other_msgs) <= keep_recent:
        return messages.copy()

    to_compress = other_msgs[:-keep_recent]
    recent = other_msgs[-keep_recent:]
    messages_to_summarize = history_summary_messages(to_compress)
    summary = await summarize_history(llm, messages_to_summarize)
    summary_msg = HumanMessage(content=f"[之前的对话摘要] {summary}")
    return [*system_msgs, summary_msg, *recent]


@trace_generation_call(
    name="conversation_summary",
    model=settings.glm_4_flash_model,
    input_arg="messages",
    input_format="messages",
    record_io=True,
    kind="generation",
)
@retry(
    stop=stop_after_attempt(2),
    wait=wait_random_exponential(min=1, max=2),
    reraise=True,
)
async def summarize_history(llm: BaseChatModel, messages: list[BaseMessage]) -> str:
    """直接读取摘要文本，保留整体重试和每次调用的超时限制。"""
    response = await asyncio.wait_for(
        ainvoke_model(llm, messages),
        timeout=60.0,
    )
    if not isinstance(response.content, str):
        raise TypeError("历史摘要模型未返回文本")
    summary = response.content.strip()
    if not summary:
        raise ValueError("历史摘要模型返回了空文本")
    return summary
