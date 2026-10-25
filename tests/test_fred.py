"""FRED 指标别名解析与 M2 真实接口测试。

别名解析测试无需网络；M2 接口测试使用实际 FRED_API_KEY 并请求真实数据。
"""

import unittest

import pytest

from app.crypto_analysis.new_tools import fred

# 用于格式化校验的少量固定元数据和观测样本。
_META = {
    "seriess": [
        {
            "title": "Unemployment Rate",
            "units_short": "%",
            "frequency": "Monthly",
            "seasonal_adjustment_short": "SA",
        }
    ]
}
_OBS = {
    "observations": [
        {"date": "2025-06-01", "value": "4.1"},
        {"date": "2025-07-01", "value": "4.3"},
        {"date": "2025-08-01", "value": "."},  # 缺失值，应跳过。
        {"date": "2025-09-01", "value": "4.4"},
    ]
}


def _request_stub(meta=_META, obs=_OBS):
    """构造按接口路径返回固定数据的 _request 替身。"""

    def _impl(path, params):
        if path == "series":
            return meta
        if path == "series/observations":
            return obs
        raise AssertionError(f"未预期的 FRED 接口路径：{path}")

    return _impl


@pytest.mark.unit
class FredSeriesResolutionTests(unittest.TestCase):
    def test_alias_maps_to_series_id(self):
        self.assertEqual(fred._resolve_series_id("cpi"), "CPIAUCSL")
        self.assertEqual(fred._resolve_series_id("unemployment"), "UNRATE")

    def test_alias_is_case_and_separator_insensitive(self):
        self.assertEqual(fred._resolve_series_id("Fed Funds Rate"), "FEDFUNDS")
        self.assertEqual(fred._resolve_series_id("10y-treasury"), "DGS10")

    def test_euro_area_aliases_map_to_series_ids(self):
        # 为欧洲交易所 .PA 等非美国证券代码提供欧元区宏观背景。
        # FRED 收录欧洲央行政策利率、欧盟统计局 HICP 与 GDP，以及 OECD 长期国债收益率。
        expected = {
            "ecb_deposit_rate": "ECBDFR",
            "ecb_main_refi_rate": "ECBMRRFR",
            "euro_hicp": "CP0000EZ19M086NEST",
            "euro_core_hicp": "00XEFDEZ19M086NEST",
            "euro_real_gdp": "CLVMNACSCAB1GQEA19",
            "germany_10y": "IRLTLT01DEM156N",
            "france_10y": "IRLTLT01FRM156N",
            "eur_usd": "DEXUSEU",
        }
        for alias, series_id in expected.items():
            with self.subTest(alias=alias):
                self.assertEqual(fred._resolve_series_id(alias), series_id)
        self.assertEqual(fred._resolve_series_id("ECB Deposit Rate"), "ECBDFR")
        self.assertEqual(fred._resolve_series_id("euro-hicp"), "CP0000EZ19M086NEST")

    def test_unknown_alias_is_treated_as_raw_series_id(self):
        # 支持直接传入任意 FRED 序列 ID，并按约定统一转换为大写。
        self.assertEqual(fred._resolve_series_id("dgs30"), "DGS30")
        self.assertEqual(fred._resolve_series_id("MyCustomSeries"), "MYCUSTOMSERIES")

    def test_descriptive_phrase_is_rejected(self):
        # LLM 传入的描述性短语（包含空格或过长）不符合序列 ID 要求，
        # 应提前拒绝并提示，避免向 API 发送会触发 HTTP 400 的请求。
        for bad in ("bank of japan rate", "the unemployment number", "X" * 31):
            with self.assertRaises(ValueError):
                fred._resolve_series_id(bad)


class FredM2LiveTests(unittest.TestCase):
    """验证真实 FRED 接口能返回 M2 数据，需要配置密钥并允许网络访问。"""

    def test_get_macro_data_returns_m2_report(self) -> None:
        out = fred.get_macro_data("m2", "2026-10-22")
        print(out)
        self.assertIn("## FRED:", out)
        self.assertIn("(M2SL)", out)
        self.assertIn("| 日期 | 数值 |", out)
