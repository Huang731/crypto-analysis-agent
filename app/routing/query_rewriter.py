"""分派前用 LLM 把口语化问题改写为意图明确的完整问题。"""

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    messages_to_dict,
)
from langchain_core.runnables import Runnable
from tenacity import retry, stop_after_attempt, wait_random_exponential

from app.core.candidate_capture import append_node_candidate
from app.core.config import settings
from app.core.tracing import trace_model_call
from app.llm.model import ainvoke_model
from prompts.query_rewriter import query_rewriter_messages


async def rewrite_question(
    quick_think_llm: Runnable[LanguageModelInput, AIMessage],
    messages: list[HumanMessage],
) -> str:
    """结合消息列表中的全部用户消息改写本轮原话，失败时向上报告。"""
    messages = query_rewriter_messages(messages)
    rewritten = await _rewrite_messages(quick_think_llm, messages)
    if settings.rewrite_capture_enabled:
        append_node_candidate(
            node="rewrite",
            input=messages_to_dict(messages),
            actual_output=rewritten,
            path=settings.rewrite_capture_path,
        )
    return rewritten


@trace_model_call(name="question_rewriter", model="quick-model")
@retry(
    stop=stop_after_attempt(2),
    wait=wait_random_exponential(min=1, max=2),
    reraise=True,
)
async def _rewrite_messages(
    quick_think_llm: Runnable[LanguageModelInput, AIMessage],
    messages: list[BaseMessage],
) -> str:
    """保留文本校验和整体重试，统计规范化后的输出。"""
    # 使用轻量模型改写问题，保留原有文本校验。
    response = await ainvoke_model(
        quick_think_llm,
        messages,
    )
    rewritten = response.content.strip()
    if not rewritten:
        raise ValueError("问题改写模型返回了空文本")
    return rewritten
