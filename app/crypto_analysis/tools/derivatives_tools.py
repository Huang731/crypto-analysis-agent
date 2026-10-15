"""衍生品OI与Funding的取数和计算。"""

from typing import Any

from langchain.tools import ToolRuntime, tool
from langchain_core.tools import BaseTool
from langgraph.types import Command

from app.crypto_analysis.calculators.derivatives_calculator import calculate_derivatives
from app.crypto_analysis.metric_service import (
    MetricJob,
    _execute_metrics,
)
from app.crypto_analysis.sources.binance_source import get_binance_derivatives


@tool
async def analyze_derivatives(
    runtime: ToolRuntime[dict[str, Any]],
) -> Command:
    """取一次目标资产衍生品数据，计算并记录 OI、Funding，返回两个指标状态。"""
    scope = runtime.context["scope"]
    return await _execute_metrics(
        runtime,
        [
            MetricJob(
                data_kind="derivatives",
                source_action="binance_sdk.get_derivatives",
                source_function=get_binance_derivatives,
                arguments={"target_asset": scope["target_asset"]},
                calculators={
                    "OI": lambda raw: calculate_derivatives(raw, scope, "OI"),
                    "Funding": lambda raw: calculate_derivatives(raw, scope, "Funding"),
                },
            )
        ],
    )


def build_derivatives_tools() -> list[BaseTool]:
    """衍生品OI与Funding的取数和计算工具。"""
    return [analyze_derivatives]
