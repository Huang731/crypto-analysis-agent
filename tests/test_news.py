"""Yahoo Finance 新闻真实接口测试，需要允许网络访问，无需配置 API 密钥。"""

import unittest

from app.crypto_analysis.new_tools import news


class YahooFinanceNewsLiveTests(unittest.TestCase):
    """验证目标币种新闻的中文报告。"""

    target_asset: str = "BTC"

    def test_get_news_yfinance_returns_report(self) -> None:
        """请求新闻并打印报告，可将币种改为 ETH 等代码。"""
        report = news.get_news_yfinance(self.target_asset)
        print(report)
        asset = self.target_asset.strip().upper()
        symbol = f"{asset}-USD"
        self.assertIn(f"## Yahoo Finance：{asset} 新闻", report)
        self.assertIn(f"- 查询代码：{symbol}", report)
        self.assertIn(f"- 回溯窗口：{news.DEFAULT_LOOKBACK_DAYS} 天", report)
        self.assertIn("UTC", report)
        self.assertIn("共筛选出", report)
        self.assertIn("\n### ", report)
        self.assertIn("（来源：", report)
        self.assertIn(news.YAHOO_NEWS_URL.format(symbol=symbol), report)


if __name__ == "__main__":
    unittest.main(verbosity=2)
