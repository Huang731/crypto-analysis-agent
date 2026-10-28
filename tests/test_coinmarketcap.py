"""CoinMarketCap 真实接口测试，需要配置 CMC_PRO_API_KEY 并允许网络访问。"""

import unittest

from app.crypto_analysis.new_tools import coinmarketcap


class CoinMarketCapLiveTests(unittest.TestCase):
    """验证稳定币市值、最新情绪指数和历史情绪指数的中文报告。"""

    target_asset: str = "USDT"

    def test_get_stablecoin_historical_market_cap_returns_report(self) -> None:
        """请求稳定币日度市值并打印报告，可将币种改为 USDC。"""
        report = coinmarketcap.get_stablecoin_historical_market_cap(self.target_asset)
        print(report)
        self.assertIn(f"## CoinMarketCap：{self.target_asset} 历史市值", report)
        self.assertIn(f"最近 {coinmarketcap.DEFAULT_LOOKBACK_DAYS} 条日度记录", report)
        self.assertIn("**最新记录：**", report)
        self.assertIn("- 单位：亿美元", report)
        self.assertIn("的市值（亿美元）变化：**", report)
        self.assertIn("起始值为", report)
        self.assertIn("| 观测时间（UTC） | 市值（亿美元） |", report)
        self.assertIn(coinmarketcap.STABLECOIN_MARKET_CAP_URL, report)

    def test_get_fear_and_greed_latest_returns_report(self) -> None:
        """请求最新恐惧贪婪指数并打印报告。"""
        report = coinmarketcap.get_fear_and_greed_latest()
        print(report)
        self.assertIn("## CoinMarketCap：最新恐惧贪婪指数", report)
        self.assertIn("**最新记录：**", report)
        self.assertTrue(
            any(
                f"情绪分类：{label}" in report
                for label in ("极度恐惧", "恐惧", "中性", "贪婪", "极度贪婪")
            )
        )
        self.assertIn("| 观测时间（UTC） | 恐惧贪婪指数 | 情绪分类 |", report)
        self.assertIn(coinmarketcap.FEAR_AND_GREED_LATEST_URL, report)

    def test_get_fear_and_greed_historical_returns_report(self) -> None:
        """请求历史日度恐惧贪婪指数并打印报告。"""
        report = coinmarketcap.get_fear_and_greed_historical()
        print(report)
        self.assertIn("## CoinMarketCap：历史恐惧贪婪指数", report)
        self.assertTrue(
            any(
                f"情绪分类：{label}" in report
                for label in ("极度恐惧", "恐惧", "中性", "贪婪", "极度贪婪")
            )
        )
        self.assertIn(f"最近 {coinmarketcap.DEFAULT_LOOKBACK_DAYS} 条日度记录", report)
        self.assertIn("**最新记录：**", report)
        self.assertIn("的恐惧贪婪指数变化：**", report)
        self.assertIn("起始值为", report)
        self.assertIn("| 观测时间（UTC） | 恐惧贪婪指数 | 情绪分类 |", report)
        self.assertIn(coinmarketcap.FEAR_AND_GREED_HISTORICAL_URL, report)


if __name__ == "__main__":
    unittest.main(verbosity=2)
