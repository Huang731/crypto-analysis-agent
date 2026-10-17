"""分派前用 LLM 把口语化问题改写为意图明确的完整问题。"""

from langchain_core.language_models.chat_models import BaseChatModel  # 标注模型接口。
from langchain_core.messages import BaseMessage, messages_to_dict
from tenacity import retry, stop_after_attempt, wait_random_exponential

from app.candidate_capture import append_node_candidate
from app.config.config import settings
from app.llm.model import ainvoke_model
from app.observability import get_logger
from app.tracing import trace_generation_call
from prompts.query_rewriter import query_rewriter_messages

_LOG = get_logger("crypto_analysis.query_rewriter")


async def rewrite_question(
    glm_smart_llm: BaseChatModel,
    question: str,
    previous_question: str,
) -> str:
    """结合上一轮问题改写本轮用户原话，失败时记录日志并向上报告。"""
    messages = query_rewriter_messages(question, previous_question)
    rewritten = await _rewrite_messages(glm_smart_llm, messages)
    if settings.rewrite_capture_enabled:
        append_node_candidate(
            node="rewrite",
            input=messages_to_dict(messages),
            actual_output=rewritten,
            path=settings.rewrite_capture_path,
        )
    return rewritten


@trace_generation_call(
    name="question_rewriter",
    model=settings.glm_smart_model,
    input_arg="messages",
    input_format="messages",
    kind="generation",
)
@retry(
    stop=stop_after_attempt(2),
    wait=wait_random_exponential(min=1, max=2),
    reraise=True,
)
async def _rewrite_messages(
    glm_smart_llm: BaseChatModel, messages: list[BaseMessage]
) -> str:
    """保留文本校验和整体重试，统计规范化后的输出。"""
    # 仅问题改写关闭深度思考，不修改共享模型配置。
    rewrite_llm = glm_smart_llm.bind(extra_body={"thinking": {"type": "disabled"}})
    response = await ainvoke_model(rewrite_llm, messages, timeout=120)
    rewritten = response.content.strip()
    if not rewritten:
        raise ValueError("问题改写模型返回了空文本")
    return rewritten
