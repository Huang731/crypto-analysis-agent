"""通过Binance官方SDK读取维度04公开行情。"""

import asyncio
from datetime import datetime
from typing import Any

from binance_common.configuration import ConfigurationRestAPI
from binance_sdk_derivatives_trading_usds_futures.derivatives_trading_usds_futures import (
    DerivativesTradingUsdsFutures,
)
from binance_sdk_spot.spot import Spot
from app.observability import timed_node
from langchain_core.tools import tool

_HOUR = 3_600_000
_SOURCES = {
    "oi": "https://fapi.binance.com/futures/data/openInterestHist",
    "funding": "https://fapi.binance.com/fapi/v1/fundingRate",
    "klines": "https://api.binance.com/api/v3/klines",
}


def _fetch_derivatives(target_asset: str, as_of: str) -> dict[str, Any]:
    """同步SDK在工作线程执行；各接口缺失保留独立原因，不换源。"""
    cutoff = datetime.fromisoformat(as_of)
    if cutoff.tzinfo is None:
        raise ValueError("as_of 必须包含时区")
    asset = target_asset.upper()
    if not asset.isascii() or not asset.isalnum():
        raise ValueError("目标资产代码无效")
    symbol = f"{asset}USDT"
    end = int(cutoff.timestamp() * 1000)
    futures = DerivativesTradingUsdsFutures(config_rest_api=ConfigurationRestAPI(timeout=30000, retries=0))
    spot = Spot(config_rest_api=ConfigurationRestAPI(timeout=30000, retries=0))
    payload: dict[str, Any] = {"symbol": symbol}

    def funding_history():
        rows = []
        start = end - 200 * _HOUR
        while True:
            page = [row.to_dict() for row in futures.rest_api.get_funding_rate_history(symbol=symbol, start_time=start, end_time=end, limit=1000).data()]
            if not isinstance(page, list):
                raise TypeError("Funding响应不是数组")
            rows.extend(page)
            if len(page) < 1000:
                return rows
            next_start = int(page[-1]["fundingTime"]) + 1
            if next_start <= start:
                raise ValueError("Funding分页时间未推进")
            if next_start > end:
                return rows
            start = next_start

    calls = {
        "oi": lambda: futures.rest_api.open_interest_statistics(symbol=symbol, period="1h", limit=200, end_time=end).data(),
        "funding": funding_history,
        "klines": lambda: spot.rest_api.klines(symbol=symbol, interval="1h", limit=200, end_time=end, time_zone="0").data(),
    }
    try:
        for kind, fetch in calls.items():
            try:
                rows = fetch()
                if not isinstance(rows, list):
                    raise TypeError(f"{kind}响应不是数组")
                payload[kind] = [row.to_dict() for row in rows] if kind == "oi" else rows
            except Exception as exc:  # noqa: BLE001 - 单个行情接口失败只记录该序列缺失，不中断其余接口。
                payload[kind] = {"status": "data_missing", "reason": f"Binance SDK {kind}：{exc}"}
    finally:
        # 关闭SDK持有的HTTP连接，不在请求之间共享状态。
        futures.rest_api._session.close()
        spot.rest_api._session.close()
    return {"payload": payload, "sources": _SOURCES.copy(), "source": "https://fapi.binance.com"}


@tool
@timed_node
async def get_binance_derivatives(target_asset: str, as_of: str) -> dict[str, Any]:
    """用官方SDK获取目标USDT永续合约小时OI、Funding及对应现货小时线。"""
    return await asyncio.to_thread(_fetch_derivatives, target_asset, as_of)


def _fetch_trend_klines(target_asset: str, as_of: str) -> dict[str, Any]:
    """获取截至UTC前一日收盘的301根原始现货日线。"""
    from datetime import timezone

    cutoff = datetime.fromisoformat(as_of)
    if cutoff.tzinfo is None:
        raise ValueError("as_of 必须包含时区")
    asset = target_asset.upper()
    if not asset.isascii() or not asset.isalnum():
        raise ValueError("目标资产代码无效")
    day = cutoff.astimezone(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    end = int(day.timestamp() * 1000) - 1
    client = Spot(config_rest_api=ConfigurationRestAPI(timeout=30000, retries=0))
    try:
        rows = client.rest_api.klines(symbol=f"{asset}USDT", interval="1d", limit=301, time_zone="0", end_time=end).data()
    finally:
        client.rest_api._session.close()
    return {"symbol": f"{asset}USDT", "rows": rows, "source": _SOURCES["klines"]}


@tool
@timed_node
async def get_binance_trend_klines(target_asset: str, as_of: str) -> dict[str, Any]:
    """用官方SDK获取维度05所需的完整UTC现货日线，不计算技术指标。"""
    return await asyncio.to_thread(_fetch_trend_klines, target_asset, as_of)
