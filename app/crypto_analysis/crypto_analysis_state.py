"""加密货币分析的图状态、运行上下文及状态合并函数。"""

import operator
from typing import Annotated, Any, Literal

from langchain.agents import AgentState
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from app.crypto_analysis.skill_loader import LoadedSkill
from app.crypto_analysis.skill_memory import merge_skill_memory


def _merge_records(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """合并并行工具记录，同一指标以后续真实计算结果为准。"""
    return {**left, **right}


class DynamicAnalysisState(TypedDict, total=False):
    scope: dict[str, str]
    plan: dict[str, Any]
    skill_results: Annotated[list[dict[str, Any]], operator.add]
    skill_memory: Annotated[dict[str, str], merge_skill_memory]
    history_summary: str
    report: str
    review_decision: Literal["pending", "pass", "rework"]
    review_count: int
    review_feedback: str


class MetricAgentState(AgentState):
    """指标 Agent 独立维护消息、原始证据和计算结果。"""

    evidence: Annotated[dict[str, dict[str, Any]], _merge_records]
    indicator_results: Annotated[dict[str, dict[str, Any]], _merge_records]


class MetricAgentContext(TypedDict):
    """由程序传入当前技能与请求范围，不由模型生成。"""

    skill: LoadedSkill
    scope: dict[str, str]


class SkillWorkerGraphState(TypedDict, total=False):
    """技能子图的输入及内部状态，工具消息和原始证据不进入父图。"""

    messages: Annotated[list[AnyMessage], add_messages]
    skill_name: str
    skill: LoadedSkill
    scope: dict[str, str]
    cached_memory: str | None
    evidence: Annotated[dict[str, dict[str, Any]], _merge_records]
    indicator_results: Annotated[dict[str, dict[str, Any]], _merge_records]
    metric_result: dict[str, Any]
    skill_result: dict[str, Any]
    skill_memory: dict[str, str]
