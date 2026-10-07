"""按既定公式校验日线并计算均线及Wilder ADX。"""

from datetime import datetime, timezone
from math import isfinite
from typing import Any

_DAY = 86_400_000


def validate_trend_klines(
    raw: dict[str, Any], scope: dict[str, str]
) -> list[list[Any]]:
    """检查完整原始数组，只选返回数据中最近300根连续的UTC日线。"""
    if raw["symbol"] != f"{scope['target_asset'].upper()}USDT":
        raise ValueError("现货日线交易对与目标不一致")
    rows = raw["payload"]
    if not isinstance(rows, list):
        raise TypeError("现货日线响应不是数组")
    selected = []
    for row in rows:
        if not isinstance(row, list) or len(row) < 7:
            raise ValueError("现货日线字段缺失或响应截断")
        if not isinstance(row[0], int) or not isinstance(row[6], int):
            raise TypeError("现货日线时间戳必须为整数毫秒")
        selected.append(row)
    selected = sorted(selected, key=lambda row: row[0])[-300:]
    if len(selected) != 300:
        raise ValueError("完整UTC日线不足300根")
    previous = None
    for row in selected:
        opened, closed = row[0], row[6]
        if opened % _DAY or closed != opened + _DAY - 1:
            raise ValueError("日线不是完整UTC自然日")
        if previous is not None and opened - previous != _DAY:
            raise ValueError("日线日期缺口或重复")
        previous = opened
        high, low, close = (float(row[index]) for index in (2, 3, 4))
        if any(not isfinite(value) or value <= 0 for value in (high, low, close)):
            raise ValueError("日线价格不是有限正数")
        if not low <= close <= high:
            raise ValueError("日线最高、最低及收盘价格不一致")
    return selected


def calculate_trend(raw: dict[str, Any], scope: dict[str, str]) -> dict[str, Any]:
    """程序计算完整300根日线，不调用大模型，不缩短计算窗口。"""
    try:
        rows = validate_trend_klines(raw, scope)
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        return {"status": "data_missing", "reason": str(exc)}
    high = [float(row[2]) for row in rows]
    low = [float(row[3]) for row in rows]
    close = [float(row[4]) for row in rows]

    def ma(period: int, index: int) -> float:
        return sum(close[index - period + 1 : index + 1]) / period

    tr, plus_dm, minus_dm = [], [], []
    for i in range(1, 300):
        up = high[i] - high[i - 1]
        down = low[i - 1] - low[i]
        tr.append(
            max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1]),
            )
        )
        plus_dm.append(up if up > down and up > 0 else 0.0)
        minus_dm.append(down if down > up and down > 0 else 0.0)
    smooth_tr, smooth_plus, smooth_minus = (
        sum(tr[:14]),
        sum(plus_dm[:14]),
        sum(minus_dm[:14]),
    )
    intermediate = {
        "smoothed_tr_14": smooth_tr,
        "smoothed_plus_dm_14": smooth_plus,
        "smoothed_minus_dm_14": smooth_minus,
    }
    dx_values = []
    adx = 0.0
    for i in range(14, 300):
        if i > 14:
            smooth_tr = smooth_tr - smooth_tr / 14 + tr[i - 1]
            smooth_plus = smooth_plus - smooth_plus / 14 + plus_dm[i - 1]
            smooth_minus = smooth_minus - smooth_minus / 14 + minus_dm[i - 1]
        pdi = 100 * smooth_plus / smooth_tr if smooth_tr else 0.0
        mdi = 100 * smooth_minus / smooth_tr if smooth_tr else 0.0
        dx = 100 * abs(pdi - mdi) / (pdi + mdi) if pdi + mdi else 0.0
        dx_values.append(dx)
        if i == 27:
            adx = sum(dx_values) / 14
            intermediate["adx_27"] = adx
        elif i > 27:
            adx = (13 * adx + dx) / 14
        if i in (298, 299):
            intermediate.update(
                {
                    f"smoothed_tr_{i}": smooth_tr,
                    f"smoothed_plus_dm_{i}": smooth_plus,
                    f"smoothed_minus_dm_{i}": smooth_minus,
                    f"adx_{i}": adx,
                    f"dx_{i}": dx,
                }
            )
    ma50, ma200 = ma(50, 299), ma(200, 299)
    ma50_before, ma200_before = ma(50, 279), ma(200, 279)
    admission = close[-1] > ma200 > ma200_before
    timing = adx > 25 and pdi > mdi
    confirmation = ma50 > ma200 and ma50 > ma50_before
    if not admission:
        score = 0 if close[-1] < ma200 < ma200_before else 2.5
    else:
        score = 5 if not timing else 7.5 if not confirmation else 10
    return {
        "status": "ok",
        "score": score,
        "unit": "price / ADX / DI",
        "observed_at": datetime.fromtimestamp(rows[-1][6] / 1000, timezone.utc)
        .date()
        .isoformat(),
        "source": raw["source"],
        "symbol": raw["symbol"],
        "kline_count": len(rows),
        "first_open_time": rows[0][0],
        "last_close_time": rows[-1][6],
        "close": close[-1],
        "ma50": ma50,
        "ma200": ma200,
        "ma50_20_days_ago": ma50_before,
        "ma200_20_days_ago": ma200_before,
        "adx14": adx,
        "plus_di": pdi,
        "minus_di": mdi,
        "admission": admission,
        "timing": timing,
        "confirmation": confirmation,
        "intermediate_values": intermediate,
        "calculation_method": "python",
    }
