"""衍生品OI与Funding的取数和计算。"""

from langchain.tools import ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from langgraph.types import Command

from app.crypto_analysis.calculators.derivatives_calculator import calculate_derivatives
from app.crypto_analysis.crypto_analysis_state import MetricToolContext
from app.crypto_analysis.metric_service import (
    _compute_metric,
    _fetch_evidence,
)
from app.crypto_analysis.sources.binance_source import get_binance_derivatives


@tool
async def get_binance_derivatives_data(
    runtime: ToolRuntime[MetricToolContext],
) -> Command:
    """通过Binance官方SDK取得同合约OI、Funding及现货小时线，返回计算用evidence_id。"""
    return await _fetch_evidence(
        runtime,
        "derivatives",
        "binance_sdk.get_derivatives",
        get_binance_derivatives,
        {
            "target_asset": runtime.context["scope"]["target_asset"],
            "as_of": runtime.context["scope"]["as_of"],
        },
    )


@tool
def calculate_oi_indicator(
    evidence_id: str, runtime: ToolRuntime[MetricToolContext]
) -> Command | ToolMessage:
    """读取衍生品证据，按7日OI、价格及Funding拥挤规则计算OI。"""
    return _compute_metric(
        runtime,
        evidence_id,
        "OI",
        "derivatives",
        lambda raw, scope: calculate_derivatives(raw, scope, "OI"),
    )


@tool
def calculate_funding_indicator(
    evidence_id: str, runtime: ToolRuntime[MetricToolContext]
) -> Command | ToolMessage:
    """读取衍生品证据，核验完整7日结算记录并计算8小时等效Funding及固定评分。"""
    return _compute_metric(
        runtime,
        evidence_id,
        "Funding",
        "derivatives",
        lambda raw, scope: calculate_derivatives(raw, scope, "Funding"),
    )


def build_derivatives_tools() -> list[BaseTool]:
    """衍生品OI与Funding的取数和计算工具。"""
    return [
        get_binance_derivatives_data,
        calculate_oi_indicator,
        calculate_funding_indicator,
    ]
