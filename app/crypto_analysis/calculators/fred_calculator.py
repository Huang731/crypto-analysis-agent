"""FRED 月度、周度观测的校验、日期对齐与变化计算。"""

from datetime import date, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any, Literal

import pandas as pd

from app.crypto_analysis.tools.source_time_utils import shift_month

_FRED_SOURCE = "https://api.stlouisfed.org/fred/series/observations"

M2_FIELD_LABELS = (
    {
        "latest_value": "最新月M2（亿美元）",
        "previous_value": "上月M2（亿美元）",
        "latest_year_ago_value": "最新月去年同月M2（亿美元）",
        "previous_year_ago_value": "上月去年同月M2（亿美元）",
        "comparison_months": "同比比较月份",
    },
    {
        "latest_yoy_pct": "最新月M2同比增速（%）",
        "previous_yoy_pct": "上月M2同比增速（%）",
        "yoy_change_pp": "M2同比增速较上月变化（百分点）",
    },
    {},
)

WALCL_FIELD_LABELS = (
    {
        "latest_value": "最新周美联储总资产（亿美元）",
        "ma4_value": "最近4周平均总资产（亿美元）",
        "comparison_dates": "最近4周观测日期",
        "weekly_values": "最近4周总资产（亿美元）",
    },
    {"ma4_change_pct": "相对4周均值偏离（%）"},
    {},
)


def _parse_observations(
    raw: dict[str, Any], series_id: Literal["M2SL", "WALCL"]
) -> pd.Series:
    """统一校验观测，保留各序列的日期规则、错误信息和 Decimal 精度。"""
    rows = raw["payload"].get("observations")
    if not isinstance(rows, list):
        raise TypeError("FRED 响应缺少 observations 数组")
    monthly = series_id == "M2SL"
    frequency = "月度" if monthly else "周度"
    values: dict[date, Decimal] = {}
    for row in rows:
        if not isinstance(row, dict):
            prefix = "FRED" if monthly else "WALCL"
            raise TypeError(f"{prefix} {frequency}观测格式无效")
        try:
            observed_at = date.fromisoformat(row["date"])
            raw_value = row["value"]
        except (KeyError, TypeError, ValueError) as exc:
            prefix = "FRED" if monthly else "WALCL"
            raise ValueError(f"{prefix} {frequency}观测缺少有效日期或数值") from exc
        if monthly and observed_at.day != 1:
            raise ValueError("M2SL 观测日期不是月份的第一天")
        if not monthly and observed_at.weekday() != 2:
            raise ValueError("WALCL 观测日期不是周三")
        if observed_at in values:
            raise ValueError(f"{series_id} {frequency}观测日期重复")
        if raw_value == ".":
            continue
        if not isinstance(raw_value, str):
            raise TypeError(f"{series_id} 观测值不是字符串")
        try:
            value = Decimal(raw_value)
        except InvalidOperation as exc:
            raise ValueError(f"{series_id} 观测值格式无效") from exc
        if not value.is_finite() or value <= 0:
            raise ValueError(f"{series_id} 观测值必须是正的有限数值")
        values[observed_at] = value
    # object 类型保留 Decimal；不将金额转换成浮点数。
    return pd.Series(values, dtype=object)


def calculate_m2_yoy_change(raw: dict[str, Any]) -> dict[str, Any]:
    """按精确月份对齐，计算最新月与上月的同比增速之差。"""
    values = _parse_observations(raw, "M2SL")
    if values.empty:
        return {
            "indicator": "M2_YoY",
            "status": "data_missing",
            "reason": "没有已发布的 M2SL 月值",
        }
    latest_month = max(values.index)
    required = (
        latest_month,
        shift_month(latest_month, -1),
        shift_month(latest_month, -12),
        shift_month(latest_month, -13),
    )
    window = values.reindex(required)
    if window.isna().any():
        return {
            "indicator": "M2_YoY",
            "status": "data_missing",
            "reason": "缺少最新月、上月或其去年同月的 M2SL 月值",
        }
    latest, previous, latest_year_ago, previous_year_ago = window.tolist()
    latest_yoy = Decimal(100) * (latest / latest_year_ago - 1)
    previous_yoy = Decimal(100) * (previous / previous_year_ago - 1)
    yoy_change = latest_yoy - previous_yoy
    return {
        "indicator": "M2_YoY",
        "status": "ok",
        "source": _FRED_SOURCE,
        "series_id": "M2SL",
        "observed_at": latest_month.isoformat(),
        # M2SL 原始单位为十亿美元，返回金额统一换算为亿美元。
        "unit": "亿美元，季节调整",
        "latest_value": float(latest * 10),
        "previous_value": float(previous * 10),
        "latest_year_ago_value": float(latest_year_ago * 10),
        "previous_year_ago_value": float(previous_year_ago * 10),
        "latest_yoy_pct": round(float(latest_yoy), 2),
        "previous_yoy_pct": round(float(previous_yoy), 2),
        "yoy_change_pp": round(float(yoy_change), 2),
        "comparison_months": [month.isoformat() for month in required],
    }


def calculate_walcl_ma4(raw: dict[str, Any]) -> dict[str, Any]:
    """按最近四个连续周三对齐，计算 MA4 与最新值偏离百分比。"""
    values = _parse_observations(raw, "WALCL")
    if values.empty:
        return {
            "indicator": "Fed_Balance（WALCL）",
            "status": "data_missing",
            "reason": "没有已发布的 WALCL 周值",
        }
    latest_day = max(values.index)
    required = [latest_day - timedelta(days=7 * index) for index in range(4)]
    window = values.reindex(required)
    if window.isna().any():
        return {
            "indicator": "Fed_Balance（WALCL）",
            "status": "data_missing",
            "reason": "缺少最近四个连续周三的 WALCL 观测",
        }
    weekly_values = window.tolist()
    # 沿用由近到远的 Decimal 求和顺序，保持原有计算精度。
    ma4 = sum(weekly_values, Decimal(0)) / 4
    latest = weekly_values[0]
    ma4_change_pct = Decimal(100) * (latest / ma4 - 1)
    return {
        "indicator": "Fed_Balance（WALCL）",
        "status": "ok",
        "source": _FRED_SOURCE,
        "series_id": "WALCL",
        "observed_at": latest_day.isoformat(),
        # WALCL 原始单位为百万美元，返回金额统一换算为亿美元。
        "unit": "亿美元，未季节调整",
        "latest_value": round(float(latest / 100), 2),
        "ma4_value": round(float(ma4 / 100), 2),
        "ma4_change_pct": round(float(ma4_change_pct), 2),
        "comparison_dates": [day.isoformat() for day in required],
        "weekly_values": [round(float(value / 100), 2) for value in weekly_values],
    }
