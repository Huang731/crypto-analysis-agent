"""取得政策目标区间，并计算三个月利率净变化。"""

from langchain.tools import ToolRuntime, tool
from langchain_core.tools import BaseTool
from langgraph.types import Command

from app.crypto_analysis.calculators.fed_policy_calculator import (
    calculate_fed_rate_change,
)
from app.crypto_analysis.crypto_analysis_state import MetricToolContext
from app.crypto_analysis.metric_service import (
    MetricJob,
    _execute_metrics,
)
from app.crypto_analysis.sources.fred_source import get_fed_target_range


@tool
async def analyze_fed_policy(
    runtime: ToolRuntime[MetricToolContext],
) -> Command:
    """取得政策利率序列并计算、记录 Fed_Rate_Change_3M，返回指标状态。"""
    return await _execute_metrics(
        runtime,
        [
            MetricJob(
                data_kind="fed_target_range",
                source_action="fred.get_fed_target_range",
                source_function=get_fed_target_range,
                arguments={},
                calculators={"Fed_Rate_Change_3M": calculate_fed_rate_change},
            )
        ],
    )


def build_fed_policy_tools() -> list[BaseTool]:
    """政策利率的取数与程序计算工具。"""
    return [analyze_fed_policy]
