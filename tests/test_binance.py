"""Binance 公开行情真实接口测试，需要允许网络访问，无需配置 API 密钥。"""

import unittest

from app.crypto_analysis.new_tools import binance


class BinanceLiveTests(unittest.TestCase):
    """验证三个 Binance 接口返回的中文行情报告。"""

    target_asset: str = "BTC"
    look_back_days: int = binance.DEFAULT_LOOKBACK_DAYS

    def test_get_daily_klines_returns_report(self) -> None:
        """请求现货日线并打印报告。"""
        report = binance.get_binance_klines(self.target_asset)
        print(report)
        self.assertIn(f"## Binance：{self.target_asset}USDT 现货 K 线", report)
        self.assertIn("- 周期：1d", report)
        self.assertIn(f"- 回溯窗口：最近 {binance.DEFAULT_LOOKBACK_DAYS} 天", report)
        self.assertIn("**最新记录：**", report)
        self.assertIn("的收盘价变化：**", report)
        self.assertIn("起始值为", report)
        self.assertIn("| 开盘时间（UTC） |", report)

    def test_get_open_interest_returns_report(self) -> None:
        """按指定天数请求日度持仓量采样并打印报告。"""
        report = binance.get_binance_open_interest(
            self.target_asset, look_back_days=self.look_back_days
        )
        print(report)
        self.assertIn(f"## Binance：{self.target_asset}USDT 历史持仓量", report)
        self.assertIn("- 采样周期：1 天", report)
        self.assertIn(f"- 回溯窗口：最近 {self.look_back_days} 天", report)
        self.assertIn("**最新记录：**", report)
        self.assertIn("的持仓量变化：**", report)
        self.assertIn("起始值为", report)
        self.assertIn("| 采样时间（UTC） | 持仓量 | 持仓名义价值（USDT） |", report)

    def test_get_funding_rates_returns_report(self) -> None:
        """按指定天数请求完整日的累计资金费率并打印报告。"""
        report = binance.get_binance_funding_rates(
            self.target_asset, look_back_days=self.look_back_days
        )
        print(report)
        self.assertIn(f"## Binance：{self.target_asset}USDT 历史资金费率", report)
        self.assertIn("**最新记录：**", report)
        self.assertIn("- 频率：日度（UTC）", report)
        self.assertIn(f"- 回溯窗口：{self.look_back_days} 个完整日", report)
        self.assertIn("的当日累计资金费率变化：**", report)
        self.assertIn("个百分点，起始值为", report)
        self.assertIn("| 日期（UTC） | 当日累计资金费率 |", report)
        self.assertIn("% |", report)


if __name__ == "__main__":
    unittest.main(verbosity=2)
