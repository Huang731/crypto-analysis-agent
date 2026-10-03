"""加密货币分析的图状态、运行上下文及状态合并函数。"""

import operator
from typing import Annotated, Any

from langchain.agents import AgentState
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from app.crypto_analysis.crypto_dimension_catalog import (
    DimensionCatalog,
    DimensionConfig,
)


def _merge_records(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """合并并行工具记录，同一指标以后续真实计算结果为准。"""
    return {**left, **right}


class MetricContext(TypedDict):
    """由程序注入的维度与请求范围，模型不能修改。"""

    dimension: DimensionConfig
    scope: dict[str, str]


class MetricAgentState(AgentState):
    """每个维度独立的证据与结果，不与其他请求共享原始数据。"""

    evidence: Annotated[dict[str, dict[str, Any]], _merge_records]
    indicator_results: Annotated[dict[str, dict[str, Any]], _merge_records]
    research_evidence: Annotated[list[dict[str, Any]], operator.add]


class DynamicAnalysisState(TypedDict, total=False):
    messages: Annotated[list[Any], add_messages]
    scope: dict[str, str]
    catalog: DimensionCatalog
    plan: dict[str, Any]
    dimension_results: Annotated[list[dict[str, Any]], operator.add]
    findings: Annotated[list[str], operator.add]
    report: str
