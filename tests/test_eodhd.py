"""EODHD 美元指数真实接口测试，需要配置 EODHD_API_KEY 并允许网络访问。"""

import unittest

from app.crypto_analysis.new_tools import eodhd


class EODHDDxyLiveTests(unittest.TestCase):
    """验证真实 EODHD 接口能返回美元指数的历史收盘价报告。"""

    def test_get_market_data_returns_dxy_report(self) -> None:
        out = eodhd.get_market_data("dxy", "2026-10-22")
        print(out)
        self.assertIn("## EODHD:", out)
        self.assertIn("DXY.INDX", out)
        self.assertIn("| 日期 | 收盘价 |", out)
