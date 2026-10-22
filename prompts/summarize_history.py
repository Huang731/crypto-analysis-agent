"""供相邻轮次理解追问的问答摘要提示词。"""

import json

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field

from prompts.common import build_system_prompt


class AnswerSummary(BaseModel):
    """只保存本轮问答的短摘要，不累计整段会话。"""

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(strict=True, min_length=1, max_length=200)


class HistorySummaryPrompt:
    instruction = """\
你是对话摘要整理器，将本轮问答压缩为供下一轮理解追问的上下文，不回答问题，不重新分析行情。

保留本轮讨论的资产、主题和分析范围，回答中的主要结论。

summary 使用一段约150～200字的中文。
"""

    user_prompt = """\
本轮补全后的用户问题：
{question}

本轮最终回答：
{report}
"""

    system_prompt = build_system_prompt(
        instruction,
        pydantic_schema=json.dumps(
            AnswerSummary.model_json_schema(), ensure_ascii=False
        ),
    )


def history_summary_messages(question: str, report: str) -> list[BaseMessage]:
    """只传入本轮问题与最终报告，生成可覆盖保存的回答摘要。"""
    return [
        SystemMessage(content=HistorySummaryPrompt.system_prompt),
        HumanMessage(
            content=HistorySummaryPrompt.user_prompt.format(
                question=question, report=report
            )
        ),
    ]
