"""分派前用 LLM 把口语化问题改写为意图明确的完整问题。"""

import json

from langchain_core.language_models.chat_models import BaseChatModel  # 标注模型接口。
from langchain_core.messages import HumanMessage, SystemMessage

from app.config.config import settings
from app.observability import get_logger, log_event
from app.tracing import estimate_tokens, trace_generation
from prompts.query_rewriter import query_rewriter_prompt

_LOG = get_logger("crypto_analysis.query_rewriter")


async def rewrite_question(
    llm: BaseChatModel,
    question: str,
    history_summary: str,
) -> str:
    """结合历史摘要改写本轮用户原话，失败时记录日志并向上报告。"""
    messages = [
        SystemMessage(content=query_rewriter_prompt(history_summary)),
        HumanMessage(content=question),
    ]
    try:
        with trace_generation(
            "question_rewriter", model=settings.fast_model
        ) as generation:
            response = await llm.with_retry(stop_after_attempt=2).ainvoke(messages)
            if not isinstance(response.content, str):
                raise TypeError("问题改写模型未返回文本内容")
            rewritten = response.content.strip()
            if not rewritten:
                raise ValueError("问题改写模型返回了空文本")
            generation.usage = {
                "input": estimate_tokens(
                    json.dumps(
                        [message.model_dump(mode="json") for message in messages],
                        ensure_ascii=False,
                    )
                ),
                "output": estimate_tokens(rewritten),
            }
    except Exception:
        _LOG.warning(
            "query.rewrite.failed",
            extra={"event": "query.rewrite.failed"},
            exc_info=True,
        )
        raise

    log_event(_LOG, "query.rewrite.done", original=question, rewritten=rewritten)
    return rewritten
