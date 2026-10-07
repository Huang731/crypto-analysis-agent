"""按既定规则校验时间窗口，并计算OI及Funding评分。"""

from datetime import datetime, timezone
from decimal import Decimal
from math import isfinite
from typing import Any

_HOUR = 3_600_000
_WEEK = 168 * _HOUR


def _number(value: Any) -> float:
    """拒绝非有限数值。"""
    number = float(value)
    if not isfinite(number):
        raise ValueError("行情数值不是有限数")
    return number


def _rows(payload: dict[str, Any], key: str) -> list[Any]:
    """接口缺失不能转换为空值或零分。"""
    rows = payload[key]
    if isinstance(rows, dict):
        raise TypeError(rows["reason"])
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{key} 原始记录缺失")
    return rows


def _price(payload: dict[str, Any]) -> dict[int, float]:
    """小时线收盘价对应下一小时起点，按返回的UTC小时线计算。"""
    prices = {}
    for row in _rows(payload, "klines"):
        start, close = int(row[0]), int(row[6])
        if start % _HOUR or close != start + _HOUR - 1:
            raise ValueError("现货小时线时间格式无效")
        timestamp = start + _HOUR
        value = _number(row[4])
        if timestamp in prices or value <= 0:
            raise ValueError("现货小时线重复或价格无效")
        prices[timestamp] = value
    return prices


def _change(values: dict[int, float], latest: int, hours: int) -> float | None:
    """按精确时间戳比较，不以相邻记录替代基准。"""
    base = values.get(latest - hours * _HOUR)
    return (
        None
        if base is None
        else float((Decimal(str(values[latest])) / Decimal(str(base)) - 1) * 100)
    )


def _funding(payload: dict[str, Any], prices: dict[int, float]) -> dict[str, Any]:
    """按最新已返回结算记录取168小时窗口，起点记录不计入累计值。"""
    records = {}
    actual_times = {}
    for row in _rows(payload, "funding"):
        actual = int(row["fundingTime"])
        if actual % _HOUR >= 1000:
            raise ValueError("Funding结算时间偏离小时边界超过允许范围")
        # 保留原始结算时间，仅将不足一秒的发布偏移映射到结算小时。
        timestamp = actual - actual % _HOUR
        actual_times[timestamp] = actual
        if row["symbol"] != payload["symbol"]:
            raise ValueError("Funding 合约与目标不一致")
        if timestamp in records:
            raise ValueError("Funding 结算时间重复")
        records[timestamp] = _number(row["fundingRate"])
    times = sorted(records)
    if len(times) < 2:
        raise ValueError("Funding 缺少7日窗口记录")
    latest = times[-1]
    start = latest - _WEEK
    window = [t for t in times if start < t <= latest]
    predecessors = [t for t in times if t <= start]
    if not predecessors or predecessors[-1] != start:
        raise ValueError("Funding 缺少7日窗口起点结算记录")
    if latest not in prices or start not in prices:
        raise ValueError("Funding 评分缺少同时间戳7日现货价格")
    price_change = _change(prices, latest, 168)
    # 使用十进制保留分档边界，不把0.03%误判到相邻档。
    average = (
        sum((Decimal(str(records[t])) for t in window), Decimal(0)) * 100 * 8 / 168
    )
    if average <= Decimal("-0.03") or average >= Decimal("0.05"):
        score = 0
    elif average < Decimal("-0.01") or average > Decimal("0.03"):
        score = 2.5
    elif price_change > 2 and 0 < average <= Decimal("0.03"):
        score = 7.5
    else:
        score = 5
    return {
        "funding_times": [actual_times[t] for t in window],
        "funding_rates": [records[t] for t in window],
        "funding_average_pct": float(average),
        "funding_score": score,
        "funding_timestamp": latest,
        "funding_price_change_pct": price_change,
    }


def calculate_derivatives(
    raw: dict[str, Any], scope: dict[str, str], metric_id: str
) -> dict[str, Any]:
    """Funding独立评分；OI依赖Funding和同时间戳价格，缺失时不给分。"""
    if metric_id not in {"OI", "Funding"}:
        raise ValueError("不支持的衍生品指标")
    payload = raw["payload"]
    try:
        if payload["symbol"] != f"{scope['target_asset'].upper()}USDT":
            raise ValueError("原始证据目标合约不一致")
        prices = _price(payload)
        if not prices:
            raise ValueError("现货小时线缺失")
        funding = _funding(payload, prices)
        result = {**funding, "symbol": payload["symbol"]}
        latest = funding["funding_timestamp"]
        score = funding["funding_score"]
        source = raw["sources"]["funding"]
        if metric_id == "OI":
            oi = {}
            for row in _rows(payload, "oi"):
                timestamp = int(row["timestamp"])
                if row["symbol"] != payload["symbol"]:
                    raise ValueError("OI 合约与目标不一致")
                value = _number(row["sumOpenInterest"])
                if timestamp in oi or value <= 0:
                    raise ValueError("OI记录重复或数量无效")
                oi[timestamp] = value
            if not oi:
                raise ValueError("OI 最新记录缺失")
            latest = max(oi)
            if abs(latest - max(prices)) > _HOUR:
                raise ValueError("OI与现货小时线时间差超过1小时")
            if latest not in prices:
                raise ValueError("OI 最新采样与现货收盘时间无法匹配")
            oi_change = _change(oi, latest, 168)
            price_change = _change(prices, latest, 168)
            if oi_change is None or price_change is None:
                raise ValueError("OI或现货缺少168小时前精确基准")
            if (score < 5 and oi_change > 5) or (price_change < -2 and oi_change > 5):
                score = 0
            elif price_change < -2 and oi_change < -5:
                score = 2.5
            elif price_change > 2 and oi_change < -5:
                score = 7.5
            elif price_change > 2 and oi_change > 5:
                score = 10
            else:
                score = 5
            state = (
                "杠杆压力"
                if funding["funding_score"] < 5
                else "去杠杆"
                if oi_change < -5
                else "健康扩张候选"
                if price_change > 2 and oi_change > 5
                else "未见明确方向"
            )
            result.update(
                oi_timestamp=latest,
                latest_oi=oi[latest],
                baseline_oi=oi[latest - _WEEK],
                latest_price=prices[latest],
                baseline_price=prices[latest - _WEEK],
                oi_change_pct=oi_change,
                price_change_pct=price_change,
                oi_24h_change_pct=_change(oi, latest, 24),
                price_24h_change_pct=_change(prices, latest, 24),
                market_state=state,
            )
            source = raw["sources"]["oi"]
        # 评分和市场状态判定完成后，再统一处理返回值的小数精度。
        result["funding_average_pct"] = round(result["funding_average_pct"], 6)
        for field in (
            "funding_price_change_pct",
            "oi_change_pct",
            "price_change_pct",
            "oi_24h_change_pct",
            "price_24h_change_pct",
        ):
            if field in result and result[field] is not None:
                result[field] = round(result[field], 2)
        return {
            **result,
            "status": "ok",
            "score": score,
            "unit": "%",
            "observed_at": datetime.fromtimestamp(latest / 1000, timezone.utc)
            .date()
            .isoformat(),
            "source": source,
            "price_source": raw["sources"]["klines"],
            "funding_source": raw["sources"]["funding"],
        }
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        return {"status": "data_missing", "reason": str(exc)}
