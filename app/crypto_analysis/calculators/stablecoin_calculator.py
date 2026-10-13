"""CMC 稳定币原始报价的校验、日期对齐与评分。"""

from datetime import date, datetime, timedelta
from math import isfinite
from typing import Any

import pandas as pd

from app.crypto_analysis.calculators.score_calculator import linear_score
from app.crypto_analysis.tools.source_time_utils import utc_time

_BASE_URL = "https://pro-api.coinmarketcap.com"


def _parse_quotes(
    quotes: list[Any], symbol: str
) -> dict[date, tuple[float | None, datetime]]:
    """校验单个币种的报价，保留 UTC 日期、市值和原始观测时间。"""
    values: dict[date, tuple[float | None, datetime]] = {}
    for row in quotes:
        if not isinstance(row, dict):
            raise TypeError(f"{symbol} 历史报价格式无效")
        quote = row.get("quote")
        usd = quote.get("USD") if isinstance(quote, dict) else None
        if not isinstance(usd, dict):
            raise TypeError(f"{symbol} 缺少 USD 报价")
        observed_at = utc_time(usd.get("timestamp"))
        market_cap = usd.get("market_cap")
        if market_cap is None:
            value = None
        elif isinstance(market_cap, bool) or not isinstance(market_cap, (int, float)):
            raise ValueError(f"{symbol} 市值格式无效")
        else:
            value = float(market_cap)
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{symbol} 市值必须是正的有限数值")
        day = observed_at.date()
        if day in values:
            raise ValueError(f"{symbol} 的 {day} UTC 日期重复")
        values[day] = (value, observed_at)
    return values


def _stablecoin_metrics(raw: dict[str, Any]) -> dict[str, Any]:
    """对齐连续 31 个 UTC 日期，沿用原有计算顺序和评分公式。"""
    data = raw["payload"].get("data")
    if not isinstance(data, dict):
        raise TypeError("CMC 稳定币响应缺少 data 对象")

    by_coin: dict[str, dict[date, tuple[float | None, datetime]]] = {}
    market_caps: dict[str, pd.Series] = {}
    for coin_id, symbol in (("825", "USDT"), ("3408", "USDC")):
        coin = data.get(coin_id)
        if not isinstance(coin, dict) or not isinstance(coin.get("quotes"), list):
            return {"status": "data_missing", "reason": f"缺少 {symbol} 历史报价"}
        quotes = _parse_quotes(coin["quotes"], symbol)
        by_coin[symbol] = quotes
        market_caps[symbol] = pd.Series(
            {day: value for day, (value, _) in quotes.items()}, dtype="float64"
        )

    # 以两种币的共同最新日期为截止点，不能被某个币单独的新报价推进。
    shared_days = market_caps["USDT"].index.intersection(market_caps["USDC"].index)
    if shared_days.empty:
        return {"status": "data_missing", "reason": "USDT 与 USDC 没有共同 UTC 日期"}
    latest_day = max(shared_days)
    required_days = [
        latest_day - timedelta(days=offset) for offset in range(30, -1, -1)
    ]

    # 先检查日期缺口，再检查市值缺失，保留原有错误优先级，不填充数据。
    required_index = pd.Index(required_days)
    if any(
        not required_index.isin(series.index).all() for series in market_caps.values()
    ):
        return {
            "status": "data_missing",
            "reason": "缺少连续 31 个 UTC 日期，无法核验 MA30、7 日和 30 日变化",
        }
    aligned = pd.DataFrame(market_caps).reindex(required_days)
    if aligned.isna().any().any():
        return {"status": "data_missing", "reason": "所需日期的稳定币市值缺失"}

    totals = aligned["USDT"] + aligned["USDC"]
    latest = float(totals.iloc[-1])
    # 保留 Python 顺序求和，避免 pandas 聚合的浮点差异影响未舍入的分数。
    ma30 = sum(totals.iloc[-30:].tolist()) / 30
    seven_day = float(totals.iloc[-8])
    thirty_day = float(totals.iloc[0])
    ma30_change = 100 * (latest / ma30 - 1)
    seven_day_change = 100 * (latest / seven_day - 1)
    ma30_score = linear_score(ma30_change, -0.5, 0.5)
    seven_day_score = linear_score(seven_day_change, -1, 1)

    return {
        "status": "ok",
        "source": f"{_BASE_URL}/v3/cryptocurrency/quotes/historical",
        "observed_at": latest_day.isoformat(),
        "quote_timestamps": {
            symbol: by_coin[symbol][latest_day][1].isoformat()
            for symbol in ("USDT", "USDC")
        },
        # 评分使用原始美元市值，返回的市值统一换算为亿美元。
        "unit": "亿美元",
        "usdt_market_cap": round(by_coin["USDT"][latest_day][0] / 100_000_000, 2),
        "usdc_market_cap": round(by_coin["USDC"][latest_day][0] / 100_000_000, 2),
        "total_market_cap": round(latest / 100_000_000, 2),
        "ma30_market_cap": round(ma30 / 100_000_000, 2),
        "seven_day_market_cap": round(seven_day / 100_000_000, 2),
        "thirty_day_market_cap": round(thirty_day / 100_000_000, 2),
        "ma30_change_pct": round(ma30_change, 2),
        "seven_day_change_pct": round(seven_day_change, 2),
        "score": round(0.5 * ma30_score + 0.5 * seven_day_score, 2),
        "thirty_day_change_pct": round(100 * (latest / thirty_day - 1), 2),
        "conflict_flag": (ma30_change > 0 > seven_day_change)
        or (seven_day_change > 0 > ma30_change),
    }
