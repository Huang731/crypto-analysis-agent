"""从上一轮报告提取结论，并增量更新会话摘要。"""

import asyncio
import json

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field

from app.config.config import settings
from app.observability import get_logger, log_event
from app.tracing import estimate_tokens, trace_generation
from prompts.memory import memory_summary_prompt

_LOG = get_logger("crypto_analysis.memory")
_SUMMARY_MAX_CHARS = 600
_SUMMARY_TIMEOUT_SECONDS = 60


class ConversationSummary(BaseModel):
    """限制摘要长度并拒绝空结果。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    summary: str = Field(min_length=1, max_length=_SUMMARY_MAX_CHARS)


async def update_memory(
    llm: BaseChatModel, summary: str, report: str
) -> dict[str, str]:
    """改写本轮问题前，将上一轮报告结论合并到已有摘要。"""
    # 首轮尚无报告，直接使用已有摘要，不调用记忆模型。
    if not report.strip():
        return {"summary": summary}

    payload = json.dumps(
        {
            "previous_summary": summary,
            "previous_report": report,
        },
        ensure_ascii=False,
    )
    model = llm.with_structured_output(
        ConversationSummary, method="json_mode"
    ).with_retry(stop_after_attempt=2)
    system_prompt = memory_summary_prompt()
    with trace_generation("memory_summary", model=settings.smart_model) as generation:
        response = await asyncio.wait_for(
            model.ainvoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=payload),
                ]
            ),
            timeout=_SUMMARY_TIMEOUT_SECONDS,
        )
        if not isinstance(response, ConversationSummary):
            raise TypeError("记忆模型未返回 ConversationSummary 结构化结果")
        generation.usage = {
            "input": estimate_tokens(system_prompt + payload),
            "output": estimate_tokens(response.model_dump_json()),
        }
    # 调试时在控制台输出完整压缩摘要，立即刷新避免缓冲。
    print(f"\n[记忆摘要]\n{response.summary}\n", flush=True)
    log_event(
        _LOG,
        "memory.summary.updated",
        report_chars=len(report),
        summary_chars=len(response.summary),
    )
    return {"summary": response.summary}
