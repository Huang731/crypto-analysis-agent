"""维度02工具：取得政策目标区间，并计算三个月利率净变化。"""

from langchain.tools import ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from langgraph.types import Command

from app.crypto_analysis.calculators.fed_policy_calculator import (
    calculate_fed_rate_change,
)
from app.crypto_analysis.crypto_analysis_agent import (
    MetricContext,
    _compute_metric,
    _fetch_evidence,
    _tool_error,
)
from app.crypto_analysis.sources.fred_source import get_fed_target_range


@tool
async def get_fred_target_range(
    runtime: ToolRuntime[MetricContext],
) -> Command | ToolMessage:
    """从 FRED 取得 DFEDTARL/DFEDTARU 原始日度序列。返回 evidence_id，供利率变化计算工具使用。"""
    if runtime.context["dimension"].dimension_id != "02":
        return _tool_error(runtime, "此工具只用于维度02")
    return await _fetch_evidence(
        runtime,
        "fed_target_range",
        "fred.get_fed_target_range",
        get_fed_target_range,
        {"as_of": runtime.context["scope"]["as_of"]},
    )


@tool
def calculate_fed_rate_change_indicator(
    evidence_id: str, runtime: ToolRuntime[MetricContext]
) -> Command | ToolMessage:
    """用 FRED 证据计算 Fed_Rate_Change_3M：同日上下限、中点、三个月基点净变化与固定评分。"""
    return _compute_metric(
        runtime,
        evidence_id,
        "Fed_Rate_Change_3M",
        "fed_target_range",
        calculate_fed_rate_change,
    )


def build_fed_policy_tools() -> list[BaseTool]:
    """维度02只注册取数与程序计算工具。"""
    return [get_fred_target_range, calculate_fed_rate_change_indicator]
