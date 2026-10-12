"""ETF资金流的取数和计算。"""

from langchain.tools import ToolRuntime, tool
from langchain_core.tools import BaseTool
from langgraph.types import Command

from app.crypto_analysis.calculators.etf_flow_calculator import calculate_etf_flows
from app.crypto_analysis.crypto_analysis_state import MetricToolContext
from app.crypto_analysis.metric_service import (
    MetricJob,
    _execute_metrics,
)
from app.crypto_analysis.sources.crypto_etf_source import get_crypto_etf_flows


@tool
async def analyze_etf_flows(
    runtime: ToolRuntime[MetricToolContext],
) -> Command:
    """取得目标资产 ETF 流量并计算、记录 ETF_Flows，返回指标状态。"""
    scope = runtime.context["scope"]
    return await _execute_metrics(
        runtime,
        [
            MetricJob(
                data_kind="etf",
                source_action="crypto_etf.get_asset_flows",
                source_function=get_crypto_etf_flows,
                arguments={"target_asset": scope["target_asset"]},
                calculators={"ETF_Flows": lambda raw: calculate_etf_flows(raw, scope)},
            )
        ],
    )


def build_etf_tools() -> list[BaseTool]:
    """ETF资金流的取数和计算工具。"""
    return [analyze_etf_flows]
