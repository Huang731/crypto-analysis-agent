"""维度 01 的请求范围、指标记录和维度结果契约。"""

from datetime import date, datetime, timezone
from math import isfinite
import re
from typing import Any


INDICATOR_WEIGHTS = {
    "DXY_Trend": 0.35,
    "Stable_MCap": 0.30,
    "M2_YoY": 0.20,
    "Fed_Balance": 0.15,
}
_ASSETS = {"BTC", "ETH", "SOL", "XRP", "BNB", "ADA", "DOGE", "AVAX", "LTC", "LINK", "DOT"}
_NON_ASSET_TOKENS = {"USD", "USDT", "USDC", "DXY", "M2", "M2SL", "WALCL", "MA4", "MA30", "MA50", "MA200", "ETF", "CMC", "FRED"}
_EXPLICIT_ASSET = re.compile(r"(?:分析|评估|币种|标的|看看)\s*[:：]?\s*([A-Z][A-Z0-9]{1,9})\b")
_UPPER_TOKEN = re.compile(r"(?<![A-Za-z0-9_])[A-Z][A-Z0-9]{1,9}(?![A-Za-z0-9_])")
_HISTORICAL = re.compile(r"(?:历史|回看|回测|去年|上个月|截至\s*\d|\d{4}[-/年]\d{1,2})")
_RAW_FIELDS = {
    "DXY_Trend": ("dxy", "ma50", "ma200"),
    "Stable_MCap": ("usdt_market_cap", "usdc_market_cap", "total_market_cap", "ma30_market_cap", "seven_day_market_cap", "thirty_day_market_cap", "quote_timestamps"),
    "M2_YoY": ("latest_value", "previous_value", "latest_year_ago_value", "previous_year_ago_value", "comparison_months"),
    "Fed_Balance": ("latest_value", "ma4_value", "comparison_dates", "weekly_values"),
}
_CALCULATED_FIELDS = {
    "DXY_Trend": ("ma50_deviation_pct",),
    "Stable_MCap": ("ma30_change_pct", "seven_day_change_pct", "thirty_day_change_pct", "conflict_flag"),
    "M2_YoY": ("latest_yoy_pct", "previous_yoy_pct", "yoy_change_pp"),
    "Fed_Balance": ("ma4_change_pct",),
}
_ACTIONS = {
    "DXY_Trend": "eodhd.get_historical_stock_prices",
    "Stable_MCap": "cmc.get_stablecoin_historical_market_cap",
    "M2_YoY": "fred.get_m2_yoy_change(M2SL)",
    "Fed_Balance": "fred.get_m2_yoy_change(WALCL)",
}


def resolve_scope(question: str, thread_id: str, trace_id: str) -> dict[str, str]:
    """一次性确定当前时点分析的目标与 UTC 截止时间。"""
    if _HISTORICAL.search(question):
        raise ValueError("当前仅支持当前时点分析，暂不支持历史截止时间")
    upper_question = question.upper()
    assets: set[str] = set()
    unknown: set[str] = set()
    for token in _UPPER_TOKEN.findall(upper_question):
        symbol = token[:-4] if token.endswith(("USDT", "USDC")) and token not in _NON_ASSET_TOKENS else token
        if symbol in _ASSETS:
            assets.add(symbol)
        elif token not in _NON_ASSET_TOKENS:
            unknown.add(token)
    if unknown:
        raise ValueError(f"无法识别目标币种：{', '.join(sorted(unknown))}")
    explicit = _EXPLICIT_ASSET.findall(upper_question)
    for token in explicit:
        symbol = token[:-4] if token.endswith(("USDT", "USDC")) and token not in _NON_ASSET_TOKENS else token
        if symbol not in _ASSETS:
            raise ValueError(f"无法识别目标币种：{token}")
        assets.add(symbol)
    if len(assets) > 1:
        raise ValueError("当前仅支持单个目标币种")
    return {
        "target_asset": next(iter(assets)) if assets else "BTC",
        "as_of": datetime.now(timezone.utc).isoformat(),
        "dimension_id": "01",
        "thread_id": thread_id,
        "trace_id": trace_id,
    }


def missing_indicator(indicator_id: str, status: str, reason: str) -> dict[str, Any]:
    """保留缺失或计算失败的指标位置与原因。"""
    return {
        "indicator_id": indicator_id,
        "status": status,
        "reason": reason,
        "raw_values": {},
        "calculated_values": {},
        "unit": None,
        "observed_at": None,
        "source": None,
        "score": None,
        "confidence": 0,
    }


def normalize_indicator(indicator_id: str, result: dict[str, Any]) -> dict[str, Any]:
    """校验工具结果，保留报告核对所需的原始值与计算值。"""
    status = result.get("status")
    if status != "ok":
        if status not in {"data_missing", "calculation_error", "not_applicable"}:
            raise ValueError(f"{indicator_id} 返回了无效状态")
        reason = result.get("reason")
        if not isinstance(reason, str) or not reason:
            raise ValueError(f"{indicator_id} 缺少失败原因")
        return missing_indicator(indicator_id, status, reason)

    required = (*_RAW_FIELDS[indicator_id], *_CALCULATED_FIELDS[indicator_id], "unit", "observed_at", "source", "score")
    missing = [name for name in required if result.get(name) is None]
    if missing:
        raise ValueError(f"{indicator_id} 缺少必要字段：{', '.join(missing)}")
    score = result["score"]
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not isfinite(score) or not 0 <= score <= 10:
        raise ValueError(f"{indicator_id} 分数必须是 0 到 10 的有限数值")
    if not all(isinstance(result[name], str) and result[name] for name in ("unit", "observed_at", "source")):
        raise ValueError(f"{indicator_id} 的单位、日期或来源无效")
    try:
        date.fromisoformat(result["observed_at"])
    except ValueError as exc:
        raise ValueError(f"{indicator_id} 的观测日期格式无效") from exc
    return {
        "indicator_id": indicator_id,
        "status": "ok",
        "reason": None,
        "raw_values": {name: result[name] for name in _RAW_FIELDS[indicator_id]},
        "calculated_values": {name: result[name] for name in _CALCULATED_FIELDS[indicator_id]},
        "unit": result["unit"],
        "observed_at": result["observed_at"],
        "source": {"action": _ACTIONS[indicator_id], "url": result["source"]},
        "score": score,
        "confidence": 0.75 if indicator_id == "Stable_MCap" and result["conflict_flag"] else 1,
    }


def aggregate_dimension(indicators: list[dict[str, Any]]) -> dict[str, Any]:
    """依维度 01 的权重和稳定币必需条件计算正式结果。"""
    by_id = {item["indicator_id"]: item for item in indicators}
    if set(by_id) != set(INDICATOR_WEIGHTS) or len(indicators) != len(INDICATOR_WEIGHTS):
        raise ValueError("维度 01 指标记录不完整或重复")
    valid = [key for key, item in by_id.items() if item["status"] == "ok"]
    coverage = round(sum(INDICATOR_WEIGHTS[key] for key in valid), 10)
    stable_valid = by_id["Stable_MCap"]["status"] == "ok"
    formal = stable_valid and coverage >= 0.7 - 1e-9
    if formal:
        score = sum(INDICATOR_WEIGHTS[key] * by_id[key]["score"] for key in valid) / coverage
        raw_confidence = sum(INDICATOR_WEIGHTS[key] * by_id[key]["confidence"] for key in INDICATOR_WEIGHTS)
        confidence = next(level for level in (1, 0.75, 0.5, 0) if raw_confidence >= level - 1e-9)
        status = "ok" if len(valid) == 4 else "partial"
        reason = None
    else:
        score = None
        confidence = 0
        if all(item["status"] == "not_applicable" for item in indicators):
            status = "not_applicable"
        elif any(item["status"] == "calculation_error" for item in indicators):
            status = "calculation_error"
        else:
            status = "data_missing"
        reason = "Stable_MCap 无效，Macro_Only；正式维度分不可用" if not stable_valid else "有效指标权重覆盖率低于 70%"
    return {
        "dimension_id": "01",
        "status": status,
        "reason": reason,
        "score": score,
        "confidence": confidence,
        "coverage": coverage,
        "valid_indicators": len(valid),
        "total_indicators": 4,
    }
