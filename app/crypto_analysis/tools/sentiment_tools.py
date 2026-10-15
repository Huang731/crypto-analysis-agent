"""市场情绪的取数和计算。"""

from typing import Any

from langchain.tools import ToolRuntime, tool
from langchain_core.tools import BaseTool
from langgraph.types import Command

from app.crypto_analysis.calculators.fear_greed_calculator import (
    calculate_fear_and_greed,
)
from app.crypto_analysis.metric_service import (
    MetricJob,
    _execute_metrics,
)
from app.crypto_analysis.sources.coinmarketcap_source import get_fear_and_greed


@tool
async def analyze_sentiment(
    runtime: ToolRuntime[dict[str, Any]],
) -> Command:
    """取得恐惧贪婪指数并计算、记录 Fear_Greed，返回指标状态。"""
    return await _execute_metrics(
        runtime,
        [
            MetricJob(
                data_kind="fear_greed",
                source_action="cmc.get_fear_and_greed(latest)",
                source_function=get_fear_and_greed,
                arguments={"mode": "latest"},
                calculators={"Fear_Greed": calculate_fear_and_greed},
            )
        ],
    )


def build_sentiment_tools() -> list[BaseTool]:
    """市场情绪的取数和计算工具。"""
    return [analyze_sentiment]
