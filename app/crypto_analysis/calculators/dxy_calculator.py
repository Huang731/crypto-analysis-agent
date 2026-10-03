"""DXY 原始日线的校验、均线与评分。"""

from datetime import date, timedelta  # 校验日线日期和有效期。
from math import isfinite  # 排除无效收盘价。
from typing import Any  # 接收工具返回的原始记录。

from app.crypto_analysis.calculators.score_calculator import (
    linear_score,  # 使用原有线性评分口径。
)

_DXY_SOURCE = "https://eodhd.com/api/eod/DXY.INDX"  # 保留原有结果来源标识。


def calculate_dxy_metrics(
    rows: Any, end_date: date
) -> dict[str, Any]:  # 定义该指标的程序计算函数。
    """用日线 close 计算 DXY、MA50、MA200 及均线方向。"""
    if not isinstance(rows, list):  # 校验当前条件并决定后续计算。
        raise TypeError("EODHD 日线响应必须是数组")  # 拒绝无效或不足的数据。
    bars: list[tuple[date, float]] = []  # 计算或保存已校验的日线列表。
    for row in rows:  # 依次处理原始数据中的记录。
        if not isinstance(row, dict):  # 校验当前条件并决定后续计算。
            raise TypeError("EODHD 日线记录格式无效")  # 拒绝无效或不足的数据。
        try:  # 开始解析外部数据字段。
            trading_date = date.fromisoformat(row["date"])  # 计算或保存当前交易日期。
            close = float(row["close"])  # 计算或保存当前收盘价。
        except (KeyError, TypeError, ValueError) as exc:  # 捕获格式错误并保留明确原因。
            raise ValueError(
                "EODHD 日线缺少有效的 date 或 close"
            ) from exc  # 拒绝无效或不足的数据。
        if not isfinite(close) or close <= 0:  # 校验当前条件并决定后续计算。
            raise ValueError(
                "EODHD 日线 close 必须是正的有限数值"
            )  # 拒绝无效或不足的数据。
        if trading_date > end_date:  # 校验当前条件并决定后续计算。
            raise ValueError("EODHD 返回了未来交易日期")  # 拒绝无效或不足的数据。
        if bars and trading_date <= bars[-1][0]:  # 校验当前条件并决定后续计算。
            raise ValueError(
                "EODHD 日线日期未严格升序或存在重复"
            )  # 拒绝无效或不足的数据。
        bars.append((trading_date, close))  # 延续上一行的数据处理。
    if len(bars) < 201:  # 校验当前条件并决定后续计算。
        raise ValueError(
            "DXY 完整日线不足 201 根，无法判断 MA200 方向"
        )  # 拒绝无效或不足的数据。
    if end_date - bars[-1][0] > timedelta(days=5):  # 校验当前条件并决定后续计算。
        raise ValueError("DXY 最新日线已超过 5 天有效期")  # 拒绝无效或不足的数据。

    closes = [close for _, close in bars]  # 计算或保存按时间排序的收盘价。
    latest_close = closes[-1]  # 计算或保存最新收盘价。
    ma50 = sum(closes[-50:]) / 50  # 计算或保存最新五十日均线。
    previous_ma50 = sum(closes[-51:-1]) / 50  # 计算或保存前一根五十日均线。
    ma200 = sum(closes[-200:]) / 200  # 计算或保存最新二百日均线。
    previous_ma200 = sum(closes[-201:-1]) / 200  # 计算或保存前一根二百日均线。
    ma50_deviation_pct = 100 * (
        latest_close / ma50 - 1
    )  # 计算或保存收盘价相对五十日均线的偏离百分比。
    score = 10 - linear_score(ma50_deviation_pct, -0.5, 0.5)  # 计算或保存指标分数。

    def direction(current: float, previous: float) -> str:  # 定义该指标的程序计算函数。
        if current > previous:  # 校验当前条件并决定后续计算。
            return "上行"  # 返回本步骤的结果。
        if current < previous:  # 校验当前条件并决定后续计算。
            return "下行"  # 返回本步骤的结果。
        return "走平"  # 返回本步骤的结果。

    return {  # 返回本步骤的结果。
        "status": "ok",  # 输出 status 字段供后续核对。
        "symbol": "DXY.INDX",  # 输出 symbol 字段供后续核对。
        "source": _DXY_SOURCE,  # 输出 source 字段供后续核对。
        "observed_at": bars[-1][0].isoformat(),  # 输出 observed_at 字段供后续核对。
        "unit": "指数点",  # 输出 unit 字段供后续核对。
        "bar_count": len(bars),  # 输出 bar_count 字段供后续核对。
        "dxy": latest_close,  # 输出 dxy 字段供后续核对。
        "ma50": ma50,  # 输出 ma50 字段供后续核对。
        "ma200": ma200,  # 输出 ma200 字段供后续核对。
        "ma50_deviation_pct": ma50_deviation_pct,  # 输出 ma50_deviation_pct 字段供后续核对。
        "score_baseline": ma50,  # 输出 score_baseline 字段供后续核对。
        "score_calculation_pct": ma50_deviation_pct,  # 输出 score_calculation_pct 字段供后续核对。
        "score_upper_threshold_pct": 0.5,  # 输出 score_upper_threshold_pct 字段供后续核对。
        "score": score,  # 输出 score 字段供后续核对。
        "report_fields": {  # 输出 report_fields 字段供后续核对。
            "原始值": latest_close,  # 输出 原始值 字段供后续核对。
            "原始值单位": "指数点",  # 输出 原始值单位 字段供后续核对。
            "基准值": ma50,  # 输出 基准值 字段供后续核对。
            "基准值名称": "MA50",  # 输出 基准值名称 字段供后续核对。
            "计算值": ma50_deviation_pct,  # 输出 计算值 字段供后续核对。
            "计算值单位": "%",  # 输出 计算值单位 字段供后续核对。
            "分数": score,  # 输出 分数 字段供后续核对。
        },  # 结束当前数据结构。
        "above_ma50": latest_close > ma50,  # 输出 above_ma50 字段供后续核对。
        "above_ma200": latest_close > ma200,  # 输出 above_ma200 字段供后续核对。
        "ma50_trend": direction(
            ma50, previous_ma50
        ),  # 输出 ma50_trend 字段供后续核对。
        "ma200_trend": direction(
            ma200, previous_ma200
        ),  # 输出 ma200_trend 字段供后续核对。
    }  # 结束当前数据结构。
