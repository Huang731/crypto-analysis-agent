"""计算现货ETF净流量及金额分档评分，金额统一使用百万美元。"""

from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any


def _flow_score(value: Decimal, neutral: int, extreme: int) -> float:
    """中性区间包含两端，极端流量包含极值边界。"""
    if value <= -extreme:
        return 0
    if value < -neutral:
        return 2.5
    if value <= neutral:
        return 5
    if value < extreme:
        return 7.5
    return 10


def calculate_etf_flows(raw: dict[str, Any], scope: dict[str, str]) -> dict[str, Any]:
    """按接口日期序列计算最近5条、20条记录，不依赖额外交易日历。"""
    payload = raw["payload"]
    if payload["symbol"].upper() != scope["target_asset"].upper():
        raise ValueError("ETF响应资产与目标资产不一致")
    days = payload["days"]
    if not isinstance(days, list):
        raise TypeError("ETF days 必须为数组")
    rows: dict[date, Decimal] = {}
    for point in days:
        day = date.fromisoformat(point["date"])
        if day in rows:
            raise ValueError("ETF数据存在重复日期")
        value = point["netFlowUsdM"]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("ETF净流量必须为有效数值")
        try:
            amount = Decimal(str(value))
        except InvalidOperation as exc:
            raise ValueError("ETF净流量不是有效金额") from exc
        if not amount.is_finite():
            raise ValueError("ETF净流量不是有限金额")
        rows[day] = amount
    # 直接按日期取接口返回的最近20条记录。
    window_dates = sorted(rows)[-20:]
    values = [rows[day] for day in window_dates]
    five = sum(values[-5:], Decimal(0))
    twenty = sum(values, Decimal(0))
    five_score = _flow_score(five, 500, 1500)
    twenty_score = _flow_score(twenty, 1500, 5000)
    return {
        "status": "ok",
        "source": raw["source"],
        "unit": "USD million",
        "observed_at": window_dates[-1].isoformat(),
        "trading_dates": [day.isoformat() for day in window_dates],
        "daily_flows_usd_m": [float(value) for value in values],
        "latest_flow_usd_m": float(values[-1]),
        "previous_five_day_flow_usd_m": float(sum(values[-10:-5], Decimal(0))),
        "five_day_flow_usd_m": float(five),
        "twenty_day_flow_usd_m": float(twenty),
        "five_day_score": five_score,
        "twenty_day_score": twenty_score,
        "conflict_flag": five * twenty < 0,
        "score": float(
            Decimal(str(five_score)) * Decimal("0.4")
            + Decimal(str(twenty_score)) * Decimal("0.6")
        ),
    }
