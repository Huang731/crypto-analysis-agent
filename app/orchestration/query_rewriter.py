"""分派前用 LLM 把口语化问题改写为意图明确的完整问题。"""

from langchain_core.language_models.chat_models import BaseChatModel  # 标注模型接口。
from langchain_core.messages import HumanMessage, SystemMessage

from app.config.config import settings
from app.llm.model import ainvoke_model
from app.observability import get_logger
from app.tracing import trace_generation_call
from prompts.query_rewriter import query_rewriter_prompt

_LOG = get_logger("crypto_analysis.query_rewriter")


async def rewrite_question(
    llm: BaseChatModel,
    question: str,
    previous_question: str,
) -> str:
    """结合上一轮问题改写本轮用户原话，失败时记录日志并向上报告。"""
    messages = [
        SystemMessage(content=query_rewriter_prompt(previous_question)),
        HumanMessage(content=question),
    ]
    try:
        rewritten = await _rewrite_messages(llm, messages)
    except Exception:
        _LOG.warning(
            "query.rewrite.failed",
            extra={"event": "query.rewrite.failed"},
            exc_info=True,
        )
        raise

    return rewritten


@trace_generation_call(
    name="question_rewriter",
    model=settings.bailian_fast_model,
    input_arg="messages",
    input_format="messages",
)
async def _rewrite_messages(
    llm: BaseChatModel, messages: list[SystemMessage | HumanMessage]
) -> str:
    """保留文本校验和整体重试，统计规范化后的输出。"""
    response = await ainvoke_model(llm, messages)
    rewritten = response.content.strip()
    if not rewritten:
        raise ValueError("问题改写模型返回了空文本")
    return rewritten
