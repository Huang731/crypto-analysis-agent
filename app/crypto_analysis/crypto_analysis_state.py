"""加密货币分析的图状态、运行上下文及状态合并函数。"""

import operator
from typing import Annotated, Any

from langchain.agents import AgentState
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from app.crypto_analysis.skill_loader import LoadedSkill
from app.crypto_analysis.skill_memory import merge_skill_memory


def _merge_records(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """合并并行工具记录，同一指标以后续真实计算结果为准。"""
    return {**left, **right}


class MetricToolContext(TypedDict):
    """由程序注入的技能与请求范围，模型不能修改。"""

    skill: LoadedSkill
    scope: dict[str, str]


class MetricAgentState(AgentState):
    """每个技能独立的证据与结果，不与其他请求共享原始数据。"""

    # 键是 evidence_id，值是取数原始记录。并行取数时按键合并，避免后返回的结果覆盖先返回的。
    evidence: Annotated[dict[str, dict[str, Any]], _merge_records]
    indicator_results: Annotated[dict[str, dict[str, Any]], _merge_records]


class DynamicAnalysisState(TypedDict, total=False):
    messages: Annotated[list[Any], add_messages]
    scope: dict[str, str]
    skill_sources: dict[str, str]
    plan: dict[str, Any]
    skill_results: Annotated[list[dict[str, Any]], operator.add]
    skill_memory: Annotated[dict[str, dict[str, Any]], merge_skill_memory]
    report: str


class SkillWorkerState(TypedDict):
    """并行技能节点的独立输入，仅包含当前技能快照和请求范围。"""

    skill_name: str
    skill_source: str
    scope: dict[str, str]
    cached_memory: dict[str, Any] | None
