"""趋势与均线的取数和计算。"""

from typing import Any

from langchain.tools import ToolRuntime, tool
from langchain_core.tools import BaseTool
from langgraph.types import Command

from app.crypto_analysis.calculators.trend_calculator import calculate_trend
from app.crypto_analysis.metric_service import (
    MetricJob,
    _execute_metrics,
)
from app.crypto_analysis.sources.binance_source import get_binance_trend_klines


@tool
async def analyze_trend(
    runtime: ToolRuntime[dict[str, Any]],
) -> Command:
    """取得目标资产日线并计算、记录 Trend，返回指标状态。"""
    scope = runtime.context["scope"]
    return await _execute_metrics(
        runtime,
        [
            MetricJob(
                data_kind="trend",
                source_action="binance_sdk.klines(1d)",
                source_function=get_binance_trend_klines,
                arguments={"target_asset": scope["target_asset"]},
                calculators={"Trend": lambda raw: calculate_trend(raw, scope)},
            )
        ],
    )


def build_trend_tools() -> list[BaseTool]:
    """趋势与均线的取数和计算工具。"""
    return [analyze_trend]
