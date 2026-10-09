"""分派前用 LLM 把口语化问题改写为意图明确的完整问题。"""

import asyncio
import json

from langchain_core.language_models.chat_models import BaseChatModel  # 标注模型接口。
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field

from app.config.config import settings
from app.observability import get_logger, log_event
from app.tracing import estimate_tokens, trace_generation
from prompts.query_rewriter import query_rewriter_prompt

_LOG = get_logger("crypto_analysis.query_rewriter")
_TIMEOUT_SECONDS = 15  # 限制单次问题改写调用耗时。
_MAX_CHARS = 200  # 改写结果超过该长度视为模型跑题。


class RewrittenQuestion(BaseModel):
    """问题改写的结构化结果，禁止空问题或额外字段。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    question: str = Field(min_length=1, max_length=_MAX_CHARS)


async def rewrite_question(
    llm: BaseChatModel,
    question: str,
) -> str:
    """仅规范改写本轮用户原话，失败时记录日志并向上报告。"""
    messages = [
        SystemMessage(content=query_rewriter_prompt()),
        HumanMessage(content=question),
    ]
    rewrite_model = llm.with_structured_output(RewrittenQuestion, method="json_mode")
    try:
        with trace_generation(
            "question_rewriter", model=settings.fast_model
        ) as generation:
            response = await asyncio.wait_for(
                rewrite_model.ainvoke(messages),
                timeout=_TIMEOUT_SECONDS,
            )
            if not isinstance(response, RewrittenQuestion):
                raise TypeError("问题改写模型未返回 RewrittenQuestion 结构化结果")
            generation.usage = {
                "input": estimate_tokens(
                    json.dumps(
                        [message.model_dump(mode="json") for message in messages],
                        ensure_ascii=False,
                    )
                ),
                "output": estimate_tokens(response.model_dump_json()),
            }
    except Exception:
        _LOG.warning(
            "query.rewrite.failed",
            extra={"event": "query.rewrite.failed"},
            exc_info=True,
        )
        raise

    rewritten = response.question
    log_event(_LOG, "query.rewrite.done", original=question, rewritten=rewritten)
    return rewritten
