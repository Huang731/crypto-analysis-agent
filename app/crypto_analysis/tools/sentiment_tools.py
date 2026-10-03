"""维度06工具：市场情绪的取数和计算。"""

from langchain.tools import ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from langgraph.types import Command

from app.crypto_analysis.crypto_analysis_agent import (
    _calculate_fear_greed,
    _compute_metric,
    _fetch_evidence,
)
from app.crypto_analysis.crypto_analysis_state import MetricContext
from app.crypto_analysis.sources.coinmarketcap_source import get_fear_and_greed


@tool
async def get_cmc_fear_greed(runtime: ToolRuntime[MetricContext]) -> Command:
    """复用CMC工具获取最新市场恐惧贪婪指数，返回计算所需的evidence_id。"""
    return await _fetch_evidence(
        runtime,
        "fear_greed",
        "cmc.get_fear_and_greed(latest)",
        get_fear_and_greed,
        {"mode": "latest"},
    )


@tool
def calculate_fear_greed_indicator(
    evidence_id: str, runtime: ToolRuntime[MetricContext]
) -> Command | ToolMessage:
    """按维度06指数区间计算Fear_Greed固定分档，记录时间仅用于展示。"""
    return _compute_metric(
        runtime, evidence_id, "Fear_Greed", "fear_greed", _calculate_fear_greed
    )


def build_sentiment_tools() -> list[BaseTool]:
    """维度06：市场情绪的取数和计算工具。"""
    return [get_cmc_fear_greed, calculate_fear_greed_indicator]
