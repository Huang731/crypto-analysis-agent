"""主题研究 Agent 的父图与并行子图状态。"""

import operator  # 合并并行研究节点的发现。
from typing import Annotated, Any  # 声明共享状态字段。

from langgraph.graph.message import add_messages  # 累积对话消息。
from typing_extensions import TypedDict  # 定义 LangGraph 状态。


class SystemState(TypedDict, total=False):  # 主题研究父图的状态。
    messages: Annotated[list[Any], add_messages]  # 用户问题与最终报告。
    scope: dict[str, str]  # 请求追踪信息和固定的分析时间。
    findings: Annotated[list[dict[str, str]], operator.add]  # 并行研究发现。
    research_summary: str  # 子图汇总的主题摘要。
    report: str  # 最终研究报告。
    review_decision: str  # 审稿结果。
    review_count: int  # 当前报告的审稿次数。
    feedback: str  # 审稿人要求修改的内容。


class AnalysisState(TypedDict, total=False):  # 拆题与并行研究子图的状态。
    question: str  # 用户原始研究主题。
    subquestions: list[str]  # 模型拆出的三个独立问题。
    findings: Annotated[list[dict[str, str]], operator.add]  # 并行节点合并后的发现。
    research_summary: str  # 汇总全部子问题的摘要。
    scope: dict[str, str]  # 本次请求的固定范围。
