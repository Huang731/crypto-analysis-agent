"""审核结果相关性评测使用的任务描述。"""

from langchain_core.messages import BaseMessage, HumanMessage

from prompts.common import build_system_prompt
from prompts.crypto_analysis_reviewer import AnalysisReviewPrompt


class ReviewerEvaluationPrompt:
    user_prompt = (
        "请审核关于{asset}的以下报告是否回应用户问题、是否符合本轮证据，"
        "给出核查过程、是否通过和修改意见。\n"
        "用户问题：{question}\n待审核报告：\n{report}"
    )

    pydantic_schema = AnalysisReviewPrompt.pydantic_schema

    system_prompt_with_schema = build_system_prompt(pydantic_schema=pydantic_schema)


def reviewer_target_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    """保留固定案例中的业务提示词，仅补充当前输出结构。"""
    return [
        *messages,
        HumanMessage(content=ReviewerEvaluationPrompt.system_prompt_with_schema),
    ]


def reviewer_evaluation_messages(
    asset: str, question: str, report: str
) -> list[BaseMessage]:
    """相关性对应审核任务，避免把审核结果当成投资报告。"""
    return [
        HumanMessage(
            content=ReviewerEvaluationPrompt.user_prompt.format(
                asset=asset, question=question, report=report
            )
        )
    ]
