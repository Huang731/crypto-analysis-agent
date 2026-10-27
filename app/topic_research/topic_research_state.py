"""主题研究 Agent 的父图与并行子图状态。"""

import operator
from typing import Annotated, Any

from langchain.agents import AgentState
from typing_extensions import TypedDict

RESEARCH_EVIDENCE_TARGET = 1  # 每个子问题至少取得一条网页原文证据。


class SystemState(TypedDict):
    scope: dict[str, str]
    # 外层由 research_team 每轮整体替换，并行结果仍在 AnalysisState 中合并。
    findings: list[dict[str, Any]]
    research_summary: str
    report: str
    review_decision: str
    review_count: int
    feedback: str


class AnalysisState(TypedDict):
    question: str
    subquestions: list[str]
    findings: Annotated[list[dict[str, Any]], operator.add]
    research_summary: str
    scope: dict[str, str]


def _merge_records(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """按工具调用 ID 合并记录，原文始终来自真实工具响应。"""
    return {**left, **right}


def _merge_evidence(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """按实际页面 URL 去重，并行工具结果最多保留一条原文证据。"""
    records = left.copy()
    urls = {item["url"] for item in records.values()}
    for key, item in right.items():
        if len(records) >= RESEARCH_EVIDENCE_TARGET:
            break
        if item["url"] not in urls:
            records[key] = item
            urls.add(item["url"])
    return records


class ResearchAgentState(AgentState):
    """每个子问题独立持有搜索记录、网页原文及工具错误。"""

    search_records: Annotated[dict[str, dict[str, Any]], _merge_records]
    research_evidence: Annotated[dict[str, dict[str, Any]], _merge_evidence]
    visited_urls: Annotated[list[str], operator.add]
    tool_errors: Annotated[list[dict[str, str]], operator.add]
