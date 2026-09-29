"""Tests for the constraint layer — the SimulatedBroker hard risk controls.

These guard the invariants the LLM cannot bypass: fees on every trade, slippage
that makes buys dearer and sells cheaper, no buying without cash, no selling
without a position, and average-cost tracking on partial fills.
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.schemas import BacktestConfig, MarketBar, OrderSide, PortfolioDecision  # noqa: E402
from tools.broker import SimulatedBroker  # noqa: E402


def _bar(close: float, day: int = 1) -> MarketBar:
    return MarketBar(day=day, date=f"2024-01-{day:02d}", open=close, high=close, low=close, close=close, volume=10000)


def _buy(shares: int, symbol: str = "000001") -> PortfolioDecision:
    return PortfolioDecision(symbol=symbol, side=OrderSide.buy, shares=shares, signal="test", confidence=0.7, reason="test")


def _sell(shares: int, symbol: str = "000001") -> PortfolioDecision:
    return PortfolioDecision(symbol=symbol, side=OrderSide.sell, shares=shares, signal="test", confidence=0.7, reason="test")


class BrokerFeeTest(unittest.TestCase):
    """Every trade pays a fee proportional to gross notional."""

    def test_buy_charges_fee_on_gross(self) -> None:
        cfg = BacktestConfig(initial_cash=1_000_000, fee_rate=0.0003, slippage_bps=1.0)
        broker = SimulatedBroker(cfg)
        fill = broker.execute(_bar(10.0), _buy(1000))
        # price 10 * (1 + 1bp) = 10.001; gross = 10000.1; fee = 3.00003
        self.assertGreater(fill.fee, 0)
        self.assertAlmostEqual(fill.fee, 10000.1 * 0.0003, places=2)
        self.assertEqual(fill.shares, 1000)
        self.assertFalse(fill.blocked)

    def test_sell_charges_fee_on_gross(self) -> None:
        cfg = BacktestConfig(initial_cash=1_000_000, fee_rate=0.0003, slippage_bps=1.0)
        broker = SimulatedBroker(cfg)
        broker.execute(_bar(10.0), _buy(1000))  # accumulate position
        fill = broker.execute(_bar(10.0), _sell(1000))
        self.assertGreater(fill.fee, 0)
        # sell price 10 * (1 - 1bp) = 9.999; gross = 9999; fee ≈ 3.0
        self.assertAlmostEqual(fill.fee, 9999.0 * 0.0003, places=2)

    def test_zero_fee_rate_means_no_fee(self) -> None:
        cfg = BacktestConfig(initial_cash=1_000_000, fee_rate=0.0, slippage_bps=0.0)
        broker = SimulatedBroker(cfg)
        fill = broker.execute(_bar(10.0), _buy(1000))
        self.assertEqual(fill.fee, 0.0)


class BrokerSlippageTest(unittest.TestCase):
    """Slippage makes buys cost more and sells realise less."""

    def test_buy_price_above_close(self) -> None:
        cfg = BacktestConfig(initial_cash=1_000_000, fee_rate=0.0, slippage_bps=10.0)
        broker = SimulatedBroker(cfg)
        fill = broker.execute(_bar(10.0), _buy(100))
        # 10bp slippage → 10 * 1.001 = 10.01
        self.assertAlmostEqual(fill.price, 10.01, places=4)

    def test_sell_price_below_close(self) -> None:
        cfg = BacktestConfig(initial_cash=1_000_000, fee_rate=0.0, slippage_bps=10.0)
        broker = SimulatedBroker(cfg)
        broker.execute(_bar(10.0), _buy(100))
        fill = broker.execute(_bar(10.0), _sell(100))
        self.assertAlmostEqual(fill.price, 9.99, places=4)

    def test_zero_slippage_uses_close(self) -> None:
        cfg = BacktestConfig(initial_cash=1_000_000, fee_rate=0.0, slippage_bps=0.0)
        broker = SimulatedBroker(cfg)
        fill = broker.execute(_bar(10.0), _buy(100))
        self.assertEqual(fill.price, 10.0)


class BrokerGuardTest(unittest.TestCase):
    """Hard guards: no buying without cash, no selling without a position."""

    def test_buy_blocked_when_cannot_afford_one_share(self) -> None:
        cfg = BacktestConfig(initial_cash=5, fee_rate=0.0, slippage_bps=0.0)
        broker = SimulatedBroker(cfg)
        # 5 cash, price 10 → can't afford a single share → blocked
        fill = broker.execute(_bar(10.0), _buy(200))
        self.assertTrue(fill.blocked)
        self.assertEqual(fill.shares, 0)

    def test_buy_partial_fill_when_cash_tight(self) -> None:
        cfg = BacktestConfig(initial_cash=1500, fee_rate=0.0, slippage_bps=0.0)
        broker = SimulatedBroker(cfg)
        # 1500 cash / 10 = 150 shares affordable, want 200 → partial 150
        fill = broker.execute(_bar(10.0), _buy(200))
        self.assertFalse(fill.blocked)
        self.assertEqual(fill.shares, 150)

    def test_sell_blocked_when_no_position(self) -> None:
        cfg = BacktestConfig(initial_cash=1_000_000, fee_rate=0.0, slippage_bps=0.0)
        broker = SimulatedBroker(cfg)
        fill = broker.execute(_bar(10.0), _sell(100))
        self.assertTrue(fill.blocked)
        self.assertEqual(fill.shares, 0)

    def test_sell_capped_at_current_position(self) -> None:
        cfg = BacktestConfig(initial_cash=1_000_000, fee_rate=0.0, slippage_bps=0.0)
        broker = SimulatedBroker(cfg)
        broker.execute(_bar(10.0), _buy(100))
        # hold 100, try sell 500 → capped at 100
        fill = broker.execute(_bar(10.0), _sell(500))
        self.assertEqual(fill.shares, 100)

    def test_hold_side_does_nothing(self) -> None:
        cfg = BacktestConfig(initial_cash=1_000_000, fee_rate=0.0003, slippage_bps=1.0)
        broker = SimulatedBroker(cfg)
        decision = PortfolioDecision(symbol="000001", side=OrderSide.hold, shares=100, signal="test", confidence=0.7, reason="test")
        fill = broker.execute(_bar(10.0), decision)
        self.assertEqual(fill.shares, 0)
        self.assertEqual(fill.fee, 0.0)


class BrokerAccountingTest(unittest.TestCase):
    """Average-cost tracking and realized PnL on partial closes."""

    def test_avg_entry_updates_on_scaled_buy(self) -> None:
        cfg = BacktestConfig(initial_cash=1_000_000, fee_rate=0.0, slippage_bps=0.0)
        broker = SimulatedBroker(cfg)
        broker.execute(_bar(10.0), _buy(100))   # avg 10
        broker.execute(_bar(12.0), _buy(100))   # avg (10*100 + 12*100)/200 = 11
        snap = broker.snapshot({"000001": 12.0}, symbol="000001")
        self.assertAlmostEqual(snap.avg_entry, 11.0, places=4)

    def test_realized_pnl_positive_on_profit_sell(self) -> None:
        cfg = BacktestConfig(initial_cash=1_000_000, fee_rate=0.0, slippage_bps=0.0)
        broker = SimulatedBroker(cfg)
        broker.execute(_bar(10.0), _buy(100))
        fill = broker.execute(_bar(12.0), _sell(100))
        # avg cost 10, sell 12 → (12-10)*100 = 200
        self.assertGreater(fill.realized_pnl, 0)
        self.assertAlmostEqual(fill.realized_pnl, 200.0, places=2)

    def test_realized_pnl_negative_on_loss_sell(self) -> None:
        cfg = BacktestConfig(initial_cash=1_000_000, fee_rate=0.0, slippage_bps=0.0)
        broker = SimulatedBroker(cfg)
        broker.execute(_bar(10.0), _buy(100))
        fill = broker.execute(_bar(8.0), _sell(100))
        self.assertLess(fill.realized_pnl, 0)
        self.assertAlmostEqual(fill.realized_pnl, -200.0, places=2)


if __name__ == "__main__":
    unittest.main()
