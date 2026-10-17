"""Regression tests for the robust factor model and risk-budget allocator."""

import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.top_down import TopDownSelector  # noqa: E402
from core.schemas import BacktestConfig, PortfolioSnapshot  # noqa: E402


class RobustFactorAndRiskBudgetTest(unittest.TestCase):
    @staticmethod
    def _selector(*, strength: float = 0.65) -> TopDownSelector:
        return TopDownSelector(
            BacktestConfig(
                provider="synthetic",
                agent_mode="offline",
                risk_parity_strength=strength,
            )
        )

    def test_robust_zscore_limits_single_outlier(self) -> None:
        values = pd.Series([0.0, 0.01, 0.02, 10.0])
        result = self._selector()._robust_zscore(values)

        self.assertLessEqual(float(result.max()), 3.5)
        self.assertGreater(float(result.iloc[-1]), float(result.iloc[-2]))

    def test_inverse_volatility_equalizes_risk_when_strength_is_one(self) -> None:
        selected = pd.DataFrame(
            {
                "symbol": ["LOW", "HIGH"],
                "target_weight": [0.5, 0.5],
                "volatility_20": [0.10, 0.30],
            }
        )
        result = self._selector(strength=1.0)._apply_risk_budget(selected)

        by_symbol = result.set_index("symbol")
        self.assertAlmostEqual(float(by_symbol.loc["LOW", "risk_adjusted_weight"]), 0.75, places=6)
        self.assertAlmostEqual(float(by_symbol.loc["HIGH", "risk_adjusted_weight"]), 0.25, places=6)
        self.assertAlmostEqual(float(by_symbol.loc["LOW", "risk_contribution"]), 0.5, places=6)
        self.assertAlmostEqual(float(by_symbol.loc["HIGH", "risk_contribution"]), 0.5, places=6)

    def test_zero_strength_preserves_signal_weights(self) -> None:
        selected = pd.DataFrame(
            {
                "symbol": ["A", "B"],
                "target_weight": [0.8, 0.2],
                "volatility_20": [0.10, 0.30],
            }
        )
        result = self._selector(strength=0.0)._apply_risk_budget(selected)
        self.assertEqual(result["risk_adjusted_weight"].round(6).tolist(), [0.8, 0.2])

    def test_allocator_exposes_risk_budget_diagnostics(self) -> None:
        selector = self._selector(strength=1.0)
        selected = selector._apply_risk_budget(
            pd.DataFrame(
                {
                    "symbol": ["AAA", "BBB"],
                    "close": [10.0, 10.0],
                    "target_weight": [0.5, 0.5],
                    "volatility_20": [0.10, 0.30],
                }
            )
        )
        portfolio = PortfolioSnapshot(cash=100_000, equity=100_000, peak=100_000, max_drawdown=0.0)
        view = {"market_view": "bullish"}

        targets = selector._allocate(portfolio, selected, view)

        self.assertGreater(targets["AAA"], targets["BBB"])
        self.assertEqual(view["risk_budget"]["method"], "signal_inverse_volatility")
        self.assertAlmostEqual(sum(view["risk_budget"]["risk_contributions"].values()), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
