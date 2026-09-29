"""Tests for the enhanced market-trend regime and the sector earnings-growth proxy."""
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.top_down import TopDownSelector  # noqa: E402
from core.schemas import BacktestConfig, PortfolioSnapshot  # noqa: E402


def _candidates():
    return pd.DataFrame(
        {
            "symbol": ["000001", "000002", "600000", "601012"],
            "name": ["a", "b", "c", "d"],
            "sector": ["银行", "电子", "银行", "电子"],
            "close": [10.0, 12.0, 8.0, 15.0],
            "change_pct": [0.01, 0.03, -0.005, 0.025],
            "volume": [1e6, 2e6, 5e5, 3e6],
            "rsi_14": [50.0, 60.0, 45.0, 65.0],
            "volatility_20": [0.10, 0.20, 0.08, 0.25],
            "sma_5": [10.0, 12.0, 8.0, 15.0],
            "sma_20": [10.0, 12.0, 8.0, 15.0],
            "sma_60": [10.0, 12.0, 8.0, 15.0],
            "atr_14": [0.1, 0.2, 0.08, 0.25],
            "volume_z": [0.0, 0.5, -0.5, 1.0],
            "return_5d": [0.01, 0.04, -0.01, 0.05],
            "return_20d": [0.02, 0.10, -0.03, 0.12],
            "dist_sma_20": [0.0, 0.01, -0.01, 0.02],
            "dist_sma_60": [0.0, 0.01, -0.01, 0.02],
        }
    )


class SectorEarningsProxyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = BacktestConfig(provider="synthetic", symbol="DEMO", universe=["000001"], agent_mode="offline")
        self.selector = TopDownSelector(self.config)

    def test_sector_score_combines_rs_and_growth(self) -> None:
        candidates = _candidates()
        # Pre-filter adds relative_strength/factor_score; then select_sectors
        # aggregates and computes sector_score = z(RS) + 0.5*z(growth).
        pre = self.selector._pre_filter(candidates, None)
        market_snapshot = {"benchmark_return": 0.0, "market_view": "bullish"}
        portfolio = PortfolioSnapshot(cash=100000, equity=100000, peak=100000, max_drawdown=0.0)
        view, selected = self.selector._select_sectors(pre, market_snapshot, portfolio)
        ranking = view["sector_ranking"]
        # 电子 (higher 1d change AND much higher 20d growth) must rank above 银行.
        self.assertEqual(ranking[0]["sector"], "电子")
        # earnings_growth_proxy field is surfaced for transparency.
        self.assertIn("earnings_growth_proxy", ranking[0])
        self.assertIn("sector_score", ranking[0])

    def test_growth_proxy_breaks_tie_when_rs_equal(self) -> None:
        # Two sectors with identical 1-day change but different 20d return:
        # the higher-growth sector must rank first.
        candidates = pd.DataFrame(
            {
                "symbol": ["000001", "000002"],
                "name": ["a", "b"],
                "sector": ["银行", "电子"],
                "close": [10.0, 10.0],
                "change_pct": [0.02, 0.02],
                "volume": [1e6, 1e6],
                "rsi_14": [50.0, 50.0],
                "volatility_20": [0.10, 0.10],
                "sma_5": [10.0, 10.0], "sma_20": [10.0, 10.0], "sma_60": [10.0, 10.0],
                "atr_14": [0.1, 0.1],
                "volume_z": [0.0, 0.0],
                "return_5d": [0.0, 0.0],
                "return_20d": [0.01, 0.08],
                "dist_sma_20": [0.0, 0.0], "dist_sma_60": [0.0, 0.0],
            }
        )
        pre = self.selector._pre_filter(candidates, None)
        portfolio = PortfolioSnapshot(cash=100000, equity=100000, peak=100000, max_drawdown=0.0)
        view, selected = self.selector._select_sectors(pre, {"benchmark_return": 0.0, "market_view": "bullish"}, portfolio)
        self.assertEqual(view["sector_ranking"][0]["sector"], "电子")


class MarketTrendScoreTest(unittest.TestCase):
    def test_trend_snapshot_includes_signals(self) -> None:
        # Build a minimal engine just to exercise _build_market_snapshot; the
        # synthetic trend dataset gives a stacked bullish MA structure after a
        # few days, which must surface trend_score + trend_signals.
        from engine.backtest import TradingBacktestEngine

        cfg = BacktestConfig(provider="synthetic", symbol="DEMO", dataset="trend", initial_cash=100000)
        eng = TradingBacktestEngine(cfg)
        # Advance a few days so indicators form.
        for _ in range(10):
            eng.step()
        snap = eng._build_market_snapshot()
        self.assertIn("trend_score", snap)
        self.assertIn("trend_signals", snap)
        self.assertIsInstance(snap["trend_score"], float)
        # market_view must be one of the three regimes.
        self.assertIn(snap["market_view"], {"bullish", "neutral", "bearish"})


if __name__ == "__main__":
    unittest.main()
