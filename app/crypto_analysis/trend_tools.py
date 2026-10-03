"""维度05工具：趋势与均线的取数和计算。"""

from langchain.tools import ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from langgraph.types import Command

from app.crypto_analysis.crypto_analysis_agent import (
    MetricContext,
    _calculate_trend,
    _compute_metric,
    _fetch_evidence,
)
from app.crypto_analysis.sources.binance_source import get_binance_trend_klines


@tool
async def get_binance_trend_data(runtime: ToolRuntime[MetricContext]) -> Command:
    """通过官方SDK获取维度05完整UTC日线，返回交给calculate_trend_indicator的evidence_id。"""
    return await _fetch_evidence(
        runtime,
        "trend",
        "binance_sdk.klines(1d)",
        get_binance_trend_klines,
        {
            "target_asset": runtime.context["scope"]["target_asset"],
            "as_of": runtime.context["scope"]["as_of"],
        },
    )


@tool
def calculate_trend_indicator(
    evidence_id: str, runtime: ToolRuntime[MetricContext]
) -> Command | ToolMessage:
    """按维度05公式用代码计算300根日线的均线、Wilder ADX、DI及固定评分。"""
    return _compute_metric(runtime, evidence_id, "Trend", "trend", _calculate_trend)


def build_trend_tools() -> list[BaseTool]:
    """维度05：趋势与均线的取数和计算工具。"""
    return [get_binance_trend_data, calculate_trend_indicator]
