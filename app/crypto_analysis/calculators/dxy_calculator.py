"""DXY 原始日线的校验、均线与评分。"""

from datetime import date, timedelta
from math import isfinite
from typing import Any

from app.crypto_analysis.calculators.score_calculator import linear_score

_DXY_SOURCE = "https://eodhd.com/api/eod/DXY.INDX"


def calculate_dxy_metrics(rows: Any, end_date: date) -> dict[str, Any]:
    """用日线 close 计算 DXY、MA50、MA200 及均线方向。"""
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
        if trading_date > end_date:
            raise ValueError("EODHD 返回了未来交易日期")
        if bars and trading_date <= bars[-1][0]:
            raise ValueError("EODHD 日线日期未严格升序或存在重复")
        bars.append((trading_date, close))
    if len(bars) < 201:
        raise ValueError("DXY 完整日线不足 201 根，无法判断 MA200 方向")
    if end_date - bars[-1][0] > timedelta(days=5):
        raise ValueError("DXY 最新日线已超过 5 天有效期")

    # 使用相邻交易日的均线判断方向，不改变评分基准。
    closes = [close for _, close in bars]
    latest_close = closes[-1]
    ma50 = sum(closes[-50:]) / 50
    previous_ma50 = sum(closes[-51:-1]) / 50
    ma200 = sum(closes[-200:]) / 200
    previous_ma200 = sum(closes[-201:-1]) / 200
    ma50_deviation_pct = 100 * (latest_close / ma50 - 1)
    # 美元越强，流动性分数越低；沿用相对 MA50 的反向评分。
    score = 10 - linear_score(ma50_deviation_pct, -0.5, 0.5)

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
        "observed_at": bars[-1][0].isoformat(),
        "unit": "指数点",
        "bar_count": len(bars),
        "dxy": latest_close,
        "ma50": ma50,
        "ma200": ma200,
        "ma50_deviation_pct": ma50_deviation_pct,
        "score_baseline": ma50,
        "score_calculation_pct": ma50_deviation_pct,
        "score_upper_threshold_pct": 0.5,
        "score": score,
        "report_fields": {
            "原始值": latest_close,
            "原始值单位": "指数点",
            "基准值": ma50,
            "基准值名称": "MA50",
            "计算值": ma50_deviation_pct,
            "计算值单位": "%",
            "分数": score,
        },
        "above_ma50": latest_close > ma50,
        "above_ma200": latest_close > ma200,
        "ma50_trend": direction(ma50, previous_ma50),
        "ma200_trend": direction(ma200, previous_ma200),
    }
