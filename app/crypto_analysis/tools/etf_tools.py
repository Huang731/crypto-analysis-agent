"""维度03工具：ETF资金流的取数和计算。"""

from langchain.tools import ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from langgraph.types import Command

from app.crypto_analysis.calculators.etf_flow_calculator import calculate_etf_flows
from app.crypto_analysis.crypto_analysis_state import MetricToolContext
from app.crypto_analysis.metric_service import (
    _compute_metric,
    _fetch_evidence,
)
from app.crypto_analysis.sources.crypto_etf_source import get_crypto_etf_flows


@tool
async def get_etf_flows(runtime: ToolRuntime[MetricToolContext]) -> Command:
    """通过MCP取得目标资产ETF资金流，返回计算工具使用的evidence_id。"""
    return await _fetch_evidence(
        runtime,
        "etf",
        "crypto_etf.get_asset_flows",
        get_crypto_etf_flows,
        {
            "target_asset": runtime.context["scope"]["target_asset"],
            "as_of": runtime.context["scope"]["as_of"],
        },
    )


@tool
def calculate_etf_indicator(
    evidence_id: str, runtime: ToolRuntime[MetricToolContext]
) -> Command | ToolMessage:
    """读取ETF原始证据，按5日40%、20日60%的金额分档规则计算ETF_Flows。"""
    return _compute_metric(
        runtime, evidence_id, "ETF_Flows", "etf", calculate_etf_flows
    )


def build_etf_tools() -> list[BaseTool]:
    """维度03：ETF资金流的取数和计算工具。"""
    return [get_etf_flows, calculate_etf_indicator]
