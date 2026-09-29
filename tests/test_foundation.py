import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.providers import _cn_code, _cn_instrument_info, _is_cn_index_symbol, _tushare_code  # noqa: E402
from tools.market import MarketEnv  # noqa: E402


class FoundationTest(unittest.TestCase):
    """Tests for the supplied market, broker, and portfolio foundation."""

    def test_manual_orders_and_benchmark_metrics(self) -> None:
        env = MarketEnv(dataset="trend", strategy="multi_agent")
        state = env.manual_order("buy", 100)
        self.assertEqual(state["portfolio"]["position"], 100)
        self.assertIn("account", state)
        self.assertIn("benchmark_return", state["report"])
        self.assertIn("active_return", state["report"])
        state = env.manual_order("sell", 1000)
        self.assertEqual(state["portfolio"]["position"], 0)
        self.assertEqual(state["trades"][-1]["shares"], 100)

    def test_rebalance_to_target_exposure(self) -> None:
        env = MarketEnv(dataset="trend", strategy="multi_agent")
        state = env.rebalance_to(0.50)
        self.assertGreater(state["portfolio"]["position"], 0)
        self.assertGreater(state["account"]["position_ratio"], 0.45)
        self.assertLess(state["account"]["position_ratio"], 0.55)

    def test_cn_stock_and_index_symbol_disambiguation(self) -> None:
        self.assertFalse(_is_cn_index_symbol("000001"))
        self.assertTrue(_is_cn_index_symbol("000001.SH"))
        self.assertTrue(_is_cn_index_symbol("sh000001"))
        self.assertEqual(_cn_code("sh000001"), "000001")
        self.assertEqual(_tushare_code("000001", index=False), "000001.SZ")
        self.assertEqual(_cn_instrument_info("000001")["name"], "平安银行")
        self.assertEqual(_cn_instrument_info("000001.SH")["name"], "上证指数")

    def test_bare_index_codes_identified_as_indices(self) -> None:
        """Regression: bare codes like 000300 must be recognised as indices.

        Before the _BARE_INDEX_CODES allowlist was added, these were treated as
        stocks → akshare stock endpoint failed → synthetic fallback → benchmark
        understated.  This test guards against that regression.
        """
        bare_indices = [
            "000300",  # 沪深300
            "000016",  # 上证50
            "000905",  # 中证500
            "000688",  # 科创50
            "000852",  # 中证1000
            "399001",  # 深证成指
        ]
        for code in bare_indices:
            with self.subTest(code=code):
                self.assertTrue(
                    _is_cn_index_symbol(code),
                    f"Bare index code '{code}' should be recognised as an index",
                )
                info = _cn_instrument_info(code)
                self.assertIsNotNone(info, f"_cn_instrument_info({code}) returned None")
                self.assertIn("name", info, f"name missing for {code}")
                # name should not be a stock name (like "平安银行")
                self.assertNotIn(
                    info["name"],
                    {"平安银行", "万科A", "深发展A"},
                    f"Unexpected stock name for index {code}: {info['name']}",
                )


if __name__ == "__main__":
    unittest.main()
