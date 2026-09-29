"""Tests for data-source resilience — the fallbacks that keep the system
running when akshare endpoints degrade.

Two production bugs motivated these tests:
  1. 000300 (bare index code) was treated as a stock → index endpoint never
     hit → synthetic fallback → benchmark understated (+8.58% vs real +32.47%).
  2. eastmoney spot intermittently returns no sector column → every stock
     labelled "未知" → sector selector picked garbage ("零食/金融信息服务").
Both are now guarded; these tests pin the guard.
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from data.providers import _cn_code, _is_cn_index_symbol  # noqa: E402
from data.universe import AShareUniverseProvider, _INDUSTRY_MAP_CACHE  # noqa: E402


class IndexDetectionTest(unittest.TestCase):
    """Bare index codes (000300, 399001) must be recognised as indices."""

    def test_csi300_bare_code_is_index(self) -> None:
        self.assertTrue(_is_cn_index_symbol("000300"))
        self.assertTrue(_is_cn_index_symbol("000300.SH"))
        self.assertTrue(_is_cn_index_symbol("sh000300"))

    def test_csi500_and_szse_component_indices(self) -> None:
        for code in ["000905", "000016", "000688", "000852", "399001", "399006"]:
            with self.subTest(code=code):
                self.assertTrue(_is_cn_index_symbol(code), f"{code} should be an index")

    def test_stock_code_is_not_index(self) -> None:
        # 000001 is 平安银行 (a stock), NOT the index — the bare code must
        # not be whitelisted, or it would shadow the stock.
        self.assertFalse(_is_cn_index_symbol("000001"))
        self.assertFalse(_is_cn_index_symbol("600519"))  # 贵州茅台

    def test_cn_code_strips_market_prefix(self) -> None:
        self.assertEqual(_cn_code("sh000300"), "000300")
        self.assertEqual(_cn_code("000300.SH"), "000300")
        self.assertEqual(_cn_code("sz000001"), "000001")


class SpotSectorFallbackTest(unittest.TestCase):
    """When eastmoney spot has no sector column, the SW industry_map must
    fill in real sector labels instead of leaving '未知'.

    This is the regression test for the "零食/金融信息服务" sweep bug.
    """

    @classmethod
    def setUpClass(cls) -> None:
        # Stash the real (possibly empty) cache so we can inject a tiny map
        cls._saved = dict(_INDUSTRY_MAP_CACHE) if _INDUSTRY_MAP_CACHE else None

    @classmethod
    def tearDownClass(cls) -> None:
        # Restore cache state for other test modules
        import data.universe as um
        um._INDUSTRY_MAP_CACHE = cls._saved

    def test_spot_without_sector_column_falls_back_to_industry_map(self) -> None:
        import data.universe as um
        # Inject a minimal mapping so the fallback has something to draw on
        um._INDUSTRY_MAP_CACHE = {"600519": "食品饮料", "000001": "银行"}
        provider = AShareUniverseProvider()

        # Simulate eastmoney spot frame with NO sector column
        spot = pd.DataFrame({
            "代码": ["600519", "000001"],
            "名称": ["贵州茅台", "平安银行"],
            "最新价": [1500.0, 10.0],
            "涨跌幅": [2.1, 0.5],
            "成交量": [10000, 5000],
            "成交额": [1.5e7, 5e4],
        })
        normalized = provider._normalize_spot(spot)
        sectors = normalized["sector"].tolist()
        self.assertNotIn("未知", sectors, "sector column should not be '未知' when industry_map has entries")
        self.assertIn("食品饮料", sectors)
        self.assertIn("银行", sectors)

    def test_spot_with_sector_column_preserves_it(self) -> None:
        import data.universe as um
        um._INDUSTRY_MAP_CACHE = None  # force no fallback needed
        provider = AShareUniverseProvider()
        spot = pd.DataFrame({
            "代码": ["600519"],
            "名称": ["贵州茅台"],
            "最新价": [1500.0],
            "涨跌幅": [2.1],
            "成交量": [10000],
            "成交额": [1.5e7],
            "所属行业": ["白酒"],   # eastmoney returned the column this time
        })
        normalized = provider._normalize_spot(spot)
        self.assertEqual(normalized["sector"].iloc[0], "白酒")

    def test_all_unknown_falls_back_only_when_fully_missing(self) -> None:
        """The fallback triggers only when EVERY row is '未知' — a partial
        sector column is respected, not overwritten."""
        import data.universe as um
        um._INDUSTRY_MAP_CACHE = {"600519": "食品饮料"}
        provider = AShareUniverseProvider()
        spot = pd.DataFrame({
            "代码": ["600519", "000001"],
            "名称": ["贵州茅台", "平安银行"],
            "最新价": [1500.0, 10.0],
            "涨跌幅": [2.1, 0.5],
            "成交量": [10000, 5000],
            "成交额": [1.5e7, 5e4],
            "所属行业": ["白酒", "未知"],  # partial — not all unknown
        })
        normalized = provider._normalize_spot(spot)
        # Partial column is NOT overwritten (not all rows are '未知')
        self.assertEqual(normalized["sector"].iloc[0], "白酒")
        self.assertEqual(normalized["sector"].iloc[1], "未知")


if __name__ == "__main__":
    unittest.main()
