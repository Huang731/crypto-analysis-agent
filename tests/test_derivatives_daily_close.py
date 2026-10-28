"""只验证衍生品与趋势日线的收盘价一致，不请求网络或模型。

运行：python -m pytest tests/test_derivatives_daily_close.py -v -s
"""

from typing import Any

from app.crypto_analysis.calculators.derivatives_calculator import calculate_derivatives
from app.crypto_analysis.calculators.trend_calculator import calculate_trend

HOUR = 3_600_000
DAY = 24 * HOUR
BOUNDARY = 1_793_145_600_000  # 2026-10-28 00:00:00 UTC。
SYMBOL = "ETHUSDT"
SCOPE = {"target_asset": "ETH"}


def make_kline(opened: int, period: int, close: float) -> list[Any]:
    """构造包含 Binance 十二个标准字段的完整 K 线。"""
    return [
        opened,
        str(close),
        str(close + 10),
        str(close - 10),
        str(close),
        "100",
        opened + period - 1,
        "200000",
        100,
        "50",
        "100000",
        "0",
    ]


def test_derivatives_close_matches_trend_close() -> None:
    """保留计算器必需输入，只检查最终收盘价。"""
    timestamps = [BOUNDARY - 168 * HOUR, BOUNDARY]
    raw_data = {
        "payload": {
            "symbol": SYMBOL,
            "klines": [
                make_kline(BOUNDARY - 169 * HOUR, HOUR, 2455.96),
                make_kline(BOUNDARY - 2 * HOUR, HOUR, 2355.36),
                make_kline(BOUNDARY - HOUR, HOUR, 2350.36),
            ],
            "oi": [
                {
                    "symbol": SYMBOL,
                    "timestamp": t,
                    "sumOpenInterest": "2300000",
                }
                for t in timestamps
            ],
            "funding": [
                {
                    "symbol": SYMBOL,
                    "fundingTime": t + 1,
                    "fundingRate": "0.00005",
                }
                for t in range(BOUNDARY - 168 * HOUR, BOUNDARY + 1, 8 * HOUR)
            ],
        },
        "sources": {
            "oi": "https://fapi.binance.com/futures/data/openInterestHist",
            "funding": "https://fapi.binance.com/fapi/v1/fundingRate",
            "klines": "https://api.binance.com/api/v3/klines",
        },
    }
    trend = calculate_trend(
        {
            "symbol": SYMBOL,
            "payload": [
                make_kline(BOUNDARY - (301 - i) * DAY, DAY, 2350.36) for i in range(301)
            ],
            "source": raw_data["sources"]["klines"],
        },
        SCOPE,
    )
    oi = calculate_derivatives(raw_data, SCOPE, "OI")
    print(f"趋势日线收盘价：{trend['close']} USDT")
    print(f"衍生品现货收盘价：{oi['latest_price']} USDT")
    assert oi["latest_price"] == trend["close"] == 2350.36
