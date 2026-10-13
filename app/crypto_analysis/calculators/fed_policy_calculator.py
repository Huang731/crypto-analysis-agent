"""根据 FRED 政策利率上下限计算三个月净变化。"""

import calendar
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any

import pandas as pd

FED_RATE_FIELD_LABELS = (
    {
        "current_date": "当前利率观测日期",
        "baseline_date": "基准利率观测日期",
        "comparison_date": "三个月前比较日期",
        "current_lower_pct": "当前目标利率下限（%）",
        "current_upper_pct": "当前目标利率上限（%）",
        "baseline_lower_pct": "基准目标利率下限（%）",
        "baseline_upper_pct": "基准目标利率上限（%）",
    },
    {
        "current_midpoint_pct": "当前目标利率中点（%）",
        "baseline_midpoint_pct": "基准目标利率中点（%）",
        "change_bp": "三个月利率净变化（基点）",
        "policy_change": "三个月政策利率变化方向",
    },
    {},
)


def comparison_date(day: date) -> date:
    """按日历向前移动三个月，月末取目标月实际最后一天。"""
    index = day.year * 12 + day.month - 1 - 3
    year, month = index // 12, index % 12 + 1
    return date(year, month, min(day.day, calendar.monthrange(year, month)[1]))


def _fed_target_values(payload: dict[str, Any]) -> dict[date, Decimal]:
    """解析有效日度观测；缺失标记不转成零。"""
    values = {}
    dates = set()
    for row in payload["observations"]:
        day = date.fromisoformat(row["date"])
        if day in dates:
            raise ValueError("FRED 政策利率观测日期重复")
        dates.add(day)
        if row["value"] == ".":
            continue
        try:
            value = Decimal(row["value"])
        except InvalidOperation as exc:
            raise ValueError("FRED 政策利率观测值无效") from exc
        if not value.is_finite():
            raise ValueError("FRED 政策利率观测值不是有限数值")
        values[day] = value
    return values


def calculate_fed_rate_change(raw: dict[str, Any]) -> dict[str, Any]:
    """用同日上下限计算三个月中点净变化，并按五档评分。"""
    payload = raw["payload"]
    lower = _fed_target_values(payload["lower_payload"])
    upper = _fed_target_values(payload["upper_payload"])
    # 只保留上下限同日有效的观测；金额始终保留 Decimal。
    rates = pd.DataFrame(
        {
            "lower": pd.Series(lower, dtype=object),
            "upper": pd.Series(upper, dtype=object),
        }
    ).dropna()
    if rates.empty:
        return {
            "status": "data_missing",
            "reason": "缺少上下限共同有效的观测日期",
        }
    current_day = max(rates.index)
    comparison_day = comparison_date(current_day)
    baseline_days = rates.index[rates.index <= comparison_day]
    if baseline_days.empty:
        return {
            "status": "data_missing",
            "reason": "缺少当前或三个月前上下限共同有效的观测日期",
        }
    baseline_day = max(baseline_days)
    current = rates.loc[current_day]
    baseline = rates.loc[baseline_day]
    for interval in (current, baseline):
        if interval["lower"] > interval["upper"]:
            raise ValueError("FRED 政策目标区间下限大于上限")
    current_midpoint = (current["lower"] + current["upper"]) / 2
    baseline_midpoint = (baseline["lower"] + baseline["upper"]) / 2
    change_bp = (current_midpoint - baseline_midpoint) * 100
    if change_bp >= 50:
        score = 0
    elif change_bp > 0:
        score = 2.5
    elif change_bp == 0:
        score = 5
    elif change_bp > -50:
        score = 7.5
    else:
        score = 10
    return {
        "status": "ok",
        "current_date": current_day.isoformat(),
        "baseline_date": baseline_day.isoformat(),
        "comparison_date": comparison_day.isoformat(),
        "current_lower_pct": float(current["lower"]),
        "current_upper_pct": float(current["upper"]),
        "baseline_lower_pct": float(baseline["lower"]),
        "baseline_upper_pct": float(baseline["upper"]),
        "current_midpoint_pct": float(current_midpoint),
        "baseline_midpoint_pct": float(baseline_midpoint),
        "change_bp": float(change_bp),
        "policy_change": "净加息"
        if change_bp > 0
        else "净降息"
        if change_bp < 0
        else "净变化为零",
        "unit": "basis_points",
        "observed_at": current_day.isoformat(),
        "source": raw["source"],
        "score": round(score, 2),
    }
