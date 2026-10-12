"""美元流动性与加密内部流动性的取数和计算。"""

from langchain.tools import ToolRuntime, tool
from langchain_core.tools import BaseTool
from langgraph.types import Command

from app.crypto_analysis.calculators.dxy_calculator import calculate_dxy_metrics
from app.crypto_analysis.calculators.fred_calculator import (
    calculate_m2_yoy_change,
    calculate_walcl_ma4,
)
from app.crypto_analysis.calculators.stablecoin_calculator import _stablecoin_metrics
from app.crypto_analysis.crypto_analysis_state import MetricToolContext
from app.crypto_analysis.metric_service import (
    MetricJob,
    _execute_metrics,
)
from app.crypto_analysis.sources.coinmarketcap_source import (
    get_stablecoin_historical_market_cap,
)
from app.crypto_analysis.sources.eodhd_source import query_dxy_from_official_mcp
from app.crypto_analysis.sources.fred_source import get_m2_yoy_change


@tool
async def analyze_liquidity(
    runtime: ToolRuntime[MetricToolContext],
) -> Command:
    """并行取得四类流动性数据，计算并记录 DXY、稳定币、M2 和美联储资产负债表指标。"""
    return await _execute_metrics(
        runtime,
        [
            MetricJob(
                data_kind="dxy",
                source_action="eodhd.get_historical_stock_prices",
                source_function=query_dxy_from_official_mcp,
                arguments={},
                calculators={"DXY_Trend": calculate_dxy_metrics},
            ),
            MetricJob(
                data_kind="stablecoin",
                source_action="cmc.get_stablecoin_historical_market_cap",
                source_function=get_stablecoin_historical_market_cap,
                arguments={},
                calculators={"Stable_MCap": _stablecoin_metrics},
            ),
            MetricJob(
                data_kind="m2",
                source_action="fred.get_m2_yoy_change(M2SL)",
                source_function=get_m2_yoy_change,
                arguments={"series_id": "M2SL"},
                calculators={"M2_YoY": calculate_m2_yoy_change},
            ),
            MetricJob(
                data_kind="walcl",
                source_action="fred.get_m2_yoy_change(WALCL)",
                source_function=get_m2_yoy_change,
                arguments={"series_id": "WALCL"},
                calculators={"Fed_Balance": calculate_walcl_ma4},
            ),
        ],
    )


def build_liquidity_tools() -> list[BaseTool]:
    """美元流动性与加密内部流动性的取数和计算工具。"""
    return [analyze_liquidity]
