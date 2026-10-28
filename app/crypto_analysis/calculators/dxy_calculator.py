"""DXY 原始日线的校验、均线与偏离计算。"""

from datetime import date
from math import isfinite
from typing import Any

import pandas as pd

_DXY_SOURCE = "https://eodhd.com/api/eod/DXY.INDX"

# 原始值、计算值的中文名称与计算窗口一起维护，供指标服务校验和正文展示。
DXY_FIELD_LABELS = (
    {"dxy": "美元指数", "ma50": "50日均线", "ma200": "200日均线"},
    {
        "ma50_deviation_pct": "相对50日均线偏离（%）",
        "ma50_trend": "50日均线方向",
        "ma200_trend": "200日均线方向",
    },
    {},
)


def _parse_closes(raw: dict[str, Any]) -> pd.Series:
    """校验日线后按原始顺序保存收盘价，拒绝乱序和重复日期。"""
    rows = raw["payload"]
    if not isinstance(rows, list):
        raise TypeError("EODHD 日线响应必须是数组")
    bars: list[tuple[date, float]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise TypeError("EODHD 日线记录格式无效")
        try:
            trading_date = date.fromisoformat(row["date"])
            close = float(row["close"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("EODHD 日线缺少有效的 date 或 close") from exc
        if not isfinite(close) or close <= 0:
            raise ValueError("EODHD 日线 close 必须是正的有限数值")
        if bars and trading_date <= bars[-1][0]:
            raise ValueError("EODHD 日线日期未严格升序或存在重复")
        bars.append((trading_date, close))
    if len(bars) < 201:
        raise ValueError("DXY 完整日线不足 201 根，无法判断 MA200 方向")

    return pd.Series(
        [close for _, close in bars],
        index=[day for day, _ in bars],
        dtype="float64",
    )


def calculate_dxy_metrics(raw: dict[str, Any]) -> dict[str, Any]:
    """用日线 close 计算 DXY、MA50、MA200 及相邻交易日的均线方向。"""
    closes = _parse_closes(raw)

    def ma(period: int, offset: int = 0) -> float:
        """保留窗口内的 Python 顺序求和与原有浮点精度。"""
        end = len(closes) - offset
        return sum(closes.iloc[end - period : end].tolist()) / period

    latest_close = float(closes.iloc[-1])
    ma50, previous_ma50 = ma(50), ma(50, 1)
    ma200, previous_ma200 = ma(200), ma(200, 1)
    ma50_deviation_pct = 100 * (latest_close / ma50 - 1)

    def direction(current: float, previous: float) -> str:
        if current > previous:
            return "上行"
        if current < previous:
            return "下行"
        return "走平"

    return {
        "status": "ok",
        "symbol": "DXY.INDX",
        "source": _DXY_SOURCE,
        "observed_at": closes.index[-1].isoformat(),
        "unit": "指数点",
        "bar_count": len(closes),
        "dxy": latest_close,
        "ma50": round(ma50, 2),
        "ma200": round(ma200, 2),
        "ma50_deviation_pct": round(ma50_deviation_pct, 2),
        "report_fields": {
            "原始值": latest_close,
            "原始值单位": "指数点",
            "基准值": ma50,
            "基准值名称": "MA50",
            "计算值": ma50_deviation_pct,
            "计算值单位": "%",
        },
        "above_ma50": latest_close > ma50,
        "above_ma200": latest_close > ma200,
        "ma50_trend": direction(ma50, previous_ma50),
        "ma200_trend": direction(ma200, previous_ma200),
    }
