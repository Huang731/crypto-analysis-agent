"""CMC 稳定币原始报价的校验、对齐与评分。"""

from datetime import date, datetime, timedelta  # 对齐 UTC 日期。
from math import isfinite  # 排除无效市值。
from typing import Any  # 接收 CMC 原始响应。

from app.crypto_analysis.calculators.score_calculator import (
    linear_score,  # 使用原有线性评分口径。
)
from app.crypto_analysis.tools.source_time_utils import (
    utc_time,  # 复用数据源时间解析规则。
)

_BASE_URL = "https://pro-api.coinmarketcap.com"  # 保留原有结果来源标识。


def _stablecoin_metrics(
    raw: dict[str, Any], scope: dict[str, str]
) -> dict[str, Any]:  # 定义该指标的程序计算函数。
    """按两个币种的 UTC 日期对齐，计算维度 01 所需指标。"""
    data = raw["payload"].get("data")  # 计算或保存接口中的币种数据。
    if not isinstance(data, dict):  # 校验当前条件并决定后续计算。
        raise TypeError("CMC 稳定币响应缺少 data 对象")  # 拒绝无效或不足的数据。
    by_coin: dict[
        str, dict[date, tuple[float | None, datetime]]
    ] = {}  # 计算或保存各稳定币按日期索引的报价。
    for coin_id, symbol in (
        ("825", "USDT"),
        ("3408", "USDC"),
    ):  # 依次处理原始数据中的记录。
        coin = data.get(coin_id)  # 计算或保存当前稳定币原始报价。
        if not isinstance(coin, dict) or not isinstance(
            coin.get("quotes"), list
        ):  # 校验当前条件并决定后续计算。
            return {
                "status": "data_missing",
                "reason": f"缺少 {symbol} 历史报价",
            }  # 返回本步骤的结果。
        values: dict[
            date, tuple[float | None, datetime]
        ] = {}  # 计算或保存按日期整理的有效观测。
        for row in coin["quotes"]:  # 依次处理原始数据中的记录。
            if not isinstance(row, dict):  # 校验当前条件并决定后续计算。
                raise TypeError(f"{symbol} 历史报价格式无效")  # 拒绝无效或不足的数据。
            quote = row.get("quote")  # 计算或保存当前报价对象。
            usd = (
                quote.get("USD") if isinstance(quote, dict) else None
            )  # 计算或保存美元计价报价。
            if not isinstance(usd, dict):  # 校验当前条件并决定后续计算。
                raise TypeError(f"{symbol} 缺少 USD 报价")  # 拒绝无效或不足的数据。
            observed_at = utc_time(usd.get("timestamp"))  # 计算或保存实际观测时间。
            market_cap = usd.get("market_cap")  # 计算或保存原始市值字段。
            if market_cap is None:  # 校验当前条件并决定后续计算。
                value = None  # 计算或保存已解析的有效数值。
            elif isinstance(market_cap, bool) or not isinstance(
                market_cap, (int, float)
            ):  # 检查另一种明确的计算条件。
                raise ValueError(f"{symbol} 市值格式无效")  # 拒绝无效或不足的数据。
            else:  # 处理剩余的有效情况。
                value = float(market_cap)  # 计算或保存已解析的有效数值。
                if not isfinite(value) or value <= 0:  # 校验当前条件并决定后续计算。
                    raise ValueError(
                        f"{symbol} 市值必须是正的有限数值"
                    )  # 拒绝无效或不足的数据。
            day = observed_at.date()  # 计算或保存观测的 UTC 日期。
            if day in values:  # 校验当前条件并决定后续计算。
                raise ValueError(
                    f"{symbol} 的 {day} UTC 日期重复"
                )  # 拒绝无效或不足的数据。
            values[day] = (value, observed_at)  # 计算或保存values[day]。
        by_coin[symbol] = values  # 计算或保存by_coin[symbol]。

    shared_days = sorted(set(by_coin["USDT"]) & set(by_coin["USDC"]))
    if not shared_days:  # 校验当前条件并决定后续计算。
        return {
            "status": "data_missing",
            "reason": "USDT 与 USDC 没有共同 UTC 日期",
        }  # 返回本步骤的结果。
    latest_day = shared_days[-1]  # 计算或保存最新有效观测日期。
    required_days = [
        latest_day - timedelta(days=offset) for offset in range(30, -1, -1)
    ]  # 计算或保存连续三十一天的必要日期。
    if any(
        day not in by_coin["USDT"] or day not in by_coin["USDC"]
        for day in required_days
    ):  # 校验当前条件并决定后续计算。
        return {
            "status": "data_missing",
            "reason": "缺少连续 31 个 UTC 日期，无法核验 MA30、7 日和 30 日变化",
        }  # 返回本步骤的结果。
    if any(
        by_coin[symbol][day][0] is None
        for symbol in ("USDT", "USDC")
        for day in required_days
    ):  # 校验当前条件并决定后续计算。
        return {
            "status": "data_missing",
            "reason": "所需日期的稳定币市值缺失",
        }  # 返回本步骤的结果。

    totals = {  # 计算或保存每天两种稳定币的合计市值。
        day: float(by_coin["USDT"][day][0])
        + float(by_coin["USDC"][day][0])  # 延续上一行的数据处理。
        for day in required_days  # 依次处理原始数据中的记录。
    }  # 结束当前数据结构。
    latest = totals[latest_day]  # 计算或保存最新指标原始值。
    ma30 = (
        sum(totals[day] for day in required_days[-30:]) / 30
    )  # 计算或保存三十日平均市值。
    seven_day = totals[latest_day - timedelta(days=7)]  # 计算或保存七日前的合计市值。
    thirty_day = totals[
        latest_day - timedelta(days=30)
    ]  # 计算或保存三十日前的合计市值。
    ma30_change = 100 * (latest / ma30 - 1)  # 计算或保存相对三十日均值的变化率。
    seven_day_change = 100 * (latest / seven_day - 1)  # 计算或保存七日市值变化率。
    ma30_score = linear_score(ma30_change, -0.5, 0.5)  # 计算或保存三十日均值子项分数。
    seven_day_score = linear_score(
        seven_day_change, -1, 1
    )  # 计算或保存七日变化子项分数。
    return {  # 返回本步骤的结果。
        "status": "ok",  # 输出 status 字段供后续核对。
        "source": f"{_BASE_URL}/v3/cryptocurrency/quotes/historical",  # 输出 source 字段供后续核对。
        "observed_at": latest_day.isoformat(),  # 输出 observed_at 字段供后续核对。
        "quote_timestamps": {  # 输出 quote_timestamps 字段供后续核对。
            symbol: by_coin[symbol][latest_day][1].isoformat()  # 延续上一行的数据处理。
            for symbol in ("USDT", "USDC")  # 依次处理原始数据中的记录。
        },  # 结束当前数据结构。
        "unit": "USD",  # 输出 unit 字段供后续核对。
        "usdt_market_cap": by_coin["USDT"][latest_day][
            0
        ],  # 输出 usdt_market_cap 字段供后续核对。
        "usdc_market_cap": by_coin["USDC"][latest_day][
            0
        ],  # 输出 usdc_market_cap 字段供后续核对。
        "total_market_cap": latest,  # 输出 total_market_cap 字段供后续核对。
        "ma30_market_cap": ma30,  # 输出 ma30_market_cap 字段供后续核对。
        "seven_day_market_cap": seven_day,  # 输出 seven_day_market_cap 字段供后续核对。
        "thirty_day_market_cap": thirty_day,  # 输出 thirty_day_market_cap 字段供后续核对。
        "ma30_change_pct": ma30_change,  # 输出 ma30_change_pct 字段供后续核对。
        "seven_day_change_pct": seven_day_change,  # 输出 seven_day_change_pct 字段供后续核对。
        "score": 0.5 * ma30_score
        + 0.5 * seven_day_score,  # 输出 score 字段供后续核对。
        "thirty_day_change_pct": 100
        * (latest / thirty_day - 1),  # 输出 thirty_day_change_pct 字段供后续核对。
        "conflict_flag": (
            ma30_change > 0 > seven_day_change
        )  # 输出 conflict_flag 字段供后续核对。
        or (seven_day_change > 0 > ma30_change),  # 延续上一行的数据处理。
    }  # 结束当前数据结构。
