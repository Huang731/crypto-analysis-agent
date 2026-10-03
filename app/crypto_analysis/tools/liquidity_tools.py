"""维度01工具：美元流动性与加密内部流动性的取数和计算。"""

from typing import Literal

from langchain.tools import ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from langgraph.types import Command

from app.crypto_analysis.crypto_analysis_agent import (
    _calculate_dxy,
    _calculate_m2,
    _calculate_stable,
    _calculate_walcl,
    _compute_metric,
    _fetch_evidence,
)
from app.crypto_analysis.crypto_analysis_state import MetricContext
from app.crypto_analysis.sources.coinmarketcap_source import (
    get_stablecoin_historical_market_cap,
)
from app.crypto_analysis.sources.eodhd_source import query_dxy_from_official_mcp
from app.crypto_analysis.sources.fred_source import get_m2_yoy_change


@tool
async def get_eodhd_dxy(runtime: ToolRuntime[MetricContext]) -> Command:
    """从 EODHD 官方 MCP 获取 DXY 历史日线。返回 evidence_id，供 calculate_dxy_indicator 计算。"""
    return await _fetch_evidence(
        runtime,
        "dxy",
        "eodhd.get_historical_stock_prices",
        query_dxy_from_official_mcp,
        {"as_of": runtime.context["scope"]["as_of"]},
    )


@tool
async def get_cmc_stablecoin_market_cap(runtime: ToolRuntime[MetricContext]) -> Command:
    """获取 CMC 的 USDT 与 USDC 历史市值。返回 evidence_id，供 calculate_stablecoin_indicator 计算。"""
    return await _fetch_evidence(
        runtime,
        "stablecoin",
        "cmc.get_stablecoin_historical_market_cap",
        get_stablecoin_historical_market_cap,
        {"as_of": runtime.context["scope"]["as_of"]},
    )


@tool
async def get_fred_observations(
    series_id: Literal["M2SL", "WALCL"], runtime: ToolRuntime[MetricContext]
) -> Command:
    """从 FRED 获取原始序列。M2SL 是货币供应月值，WALCL 是美联储资产负债表周值。返回 evidence_id。"""
    return await _fetch_evidence(
        runtime,
        "m2" if series_id == "M2SL" else "walcl",
        f"fred.get_m2_yoy_change({series_id})",
        get_m2_yoy_change,
        {"series_id": series_id, "as_of": runtime.context["scope"]["as_of"]},
    )


@tool
def calculate_dxy_indicator(
    evidence_id: str, runtime: ToolRuntime[MetricContext]
) -> Command | ToolMessage:
    """用 get_eodhd_dxy 返回的 evidence_id 计算 DXY_Trend：MA50、MA200、偏离比例与固定评分。"""
    return _compute_metric(runtime, evidence_id, "DXY_Trend", "dxy", _calculate_dxy)


@tool
def calculate_stablecoin_indicator(
    evidence_id: str, runtime: ToolRuntime[MetricContext]
) -> Command | ToolMessage:
    """用稳定币证据计算 Stable_MCap：USDT+USDC、MA30、7日及30日变化、分歧和固定评分。"""
    return _compute_metric(
        runtime, evidence_id, "Stable_MCap", "stablecoin", _calculate_stable
    )


@tool
def calculate_m2_indicator(
    evidence_id: str, runtime: ToolRuntime[MetricContext]
) -> Command | ToolMessage:
    """用 FRED M2SL 证据计算 M2_YoY：最新与上月同比、同比变化百分点和固定评分。"""
    return _compute_metric(runtime, evidence_id, "M2_YoY", "m2", _calculate_m2)


@tool
def calculate_fed_balance_indicator(
    evidence_id: str, runtime: ToolRuntime[MetricContext]
) -> Command | ToolMessage:
    """用 FRED WALCL 证据计算 Fed_Balance：连续四周 MA4、偏离比例和固定评分。"""
    return _compute_metric(
        runtime, evidence_id, "Fed_Balance", "walcl", _calculate_walcl
    )


def build_liquidity_tools() -> list[BaseTool]:
    """维度01：美元流动性与加密内部流动性的取数和计算工具。"""
    return [
        get_eodhd_dxy,
        get_cmc_stablecoin_market_cap,
        get_fred_observations,
        calculate_dxy_indicator,
        calculate_stablecoin_indicator,
        calculate_m2_indicator,
        calculate_fed_balance_indicator,
    ]
