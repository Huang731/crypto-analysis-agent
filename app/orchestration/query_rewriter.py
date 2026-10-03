"""分派前用 LLM 把口语化问题改写为意图明确的完整问题。"""

import asyncio

from langchain_core.language_models.chat_models import BaseChatModel  # 标注模型接口。
from langchain_core.messages import HumanMessage, SystemMessage  # 构造改写对话。

from app.config.config import settings
from app.observability import get_logger, log_event
from app.tracing import estimate_tokens, trace_generation
from prompts.query_rewriter import SYSTEM_PROMPT as _SYSTEM_PROMPT

_LOG = get_logger("crypto_analysis.query_rewriter")
_TIMEOUT_SECONDS = 15  # 改写只是前置优化，超时即放弃，不能拖慢主流程。
_MAX_CHARS = 200  # 改写结果超过该长度视为模型跑题。


async def rewrite_question(llm: BaseChatModel, question: str) -> str:
    """返回改写后的问题；模型失败、超时或输出异常时返回原问题。"""
    try:
        with trace_generation(
            "question_rewriter", model=settings.fast_model
        ) as generation:
            response = await asyncio.wait_for(
                llm.ainvoke(
                    [
                        SystemMessage(content=_SYSTEM_PROMPT),
                        HumanMessage(content=question),
                    ]
                ),
                timeout=_TIMEOUT_SECONDS,
            )
            generation.usage = {
                "input": estimate_tokens(_SYSTEM_PROMPT + question),
                "output": estimate_tokens(response.text),
            }
    except Exception:  # noqa: BLE001 - 改写失败不应阻断分派，退回原问题即可。
        _LOG.warning(
            "query.rewrite.failed",
            extra={"event": "query.rewrite.failed"},
            exc_info=True,
        )
        return question

    rewritten = response.text.strip().strip("\"'“”")
    if not rewritten or len(rewritten) > _MAX_CHARS:
        log_event(_LOG, "query.rewrite.discarded", output_chars=len(rewritten))
        return question
    log_event(_LOG, "query.rewrite.done", original=question, rewritten=rewritten)
    return rewritten
