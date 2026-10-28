"""CryptoETF 真实接口测试，需要配置 CRYPTO_ETF_MCP_API_KEY 并允许网络访问。"""

import unittest

from app.crypto_analysis.new_tools import cryptoetf


class CryptoETFLiveTests(unittest.TestCase):
    """验证目标资产 ETF 日度净流量的中文报告。"""

    target_asset: str = "BTC"
    look_back_days: int = cryptoetf.DEFAULT_LOOKBACK_DAYS

    def test_get_crypto_etf_flows_returns_report(self) -> None:
        """请求日度 ETF 净流量并打印报告，可修改币种和回溯天数。"""
        report = cryptoetf.get_crypto_etf_flows(
            self.target_asset, look_back_days=self.look_back_days
        )
        print(report)
        self.assertIn(f"## CryptoETF：{self.target_asset} 现货 ETF 资金流", report)
        self.assertIn("- 频率：日度（UTC）", report)
        self.assertIn(f"- 回溯窗口：{self.look_back_days} 天", report)
        self.assertIn("- 单位：亿美元", report)
        self.assertIn("**最新记录：**", report)
        self.assertIn("| 日期（UTC） | 净流量（亿美元） |", report)
        table_rows = [line for line in report.splitlines() if line.startswith("|")]
        self.assertGreaterEqual(len(table_rows), 3)
        if len(table_rows) > 3:
            self.assertIn("的单日净流量变化：**", report)
            self.assertIn("起始值为", report)
        else:
            self.assertNotIn("的单日净流量变化：**", report)
        self.assertIn(
            cryptoetf.CRYPTO_ETF_FLOWS_URL.format(asset=self.target_asset.lower()),
            report,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
