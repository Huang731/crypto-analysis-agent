"""主题研究 Agent 的父图与并行子图状态。"""

import operator  # 合并并行研究节点的发现。
from typing import Annotated, Any  # 声明共享状态字段。

from langchain.agents import AgentState
from typing_extensions import TypedDict  # 定义 LangGraph 状态。


class SystemState(TypedDict, total=False):  # 主题研究父图的状态。
    scope: dict[str, str]  # 请求追踪信息和固定的分析时间。
    findings: Annotated[list[dict[str, Any]], operator.add]  # 并行研究发现。
    research_summary: str  # 子图汇总的主题摘要。
    report: str  # 最终研究报告。
    review_decision: str  # 审稿结果。
    review_count: int  # 当前报告的审稿次数。
    feedback: str  # 审稿人要求修改的内容。


class AnalysisState(TypedDict, total=False):  # 拆题与并行研究子图的状态。
    question: str  # 用户原始研究主题。
    subquestions: list[str]  # 模型拆出的三个独立问题。
    findings: Annotated[list[dict[str, Any]], operator.add]  # 并行节点合并后的发现。
    research_summary: str  # 汇总全部子问题的摘要。
    scope: dict[str, str]  # 本次请求的固定范围。


def _merge_records(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """按工具调用 ID 合并记录，原文始终来自真实工具响应。"""
    return {**left, **right}


class ResearchAgentState(AgentState):
    """每个子问题独立持有搜索记录、网页原文及工具错误。"""

    search_records: Annotated[dict[str, dict[str, Any]], _merge_records]
    research_evidence: Annotated[dict[str, dict[str, Any]], _merge_records]
    tool_errors: Annotated[list[dict[str, str]], operator.add]
