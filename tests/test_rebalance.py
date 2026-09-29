"""Tests for the continuous-decision rebalance diff logic.

The rebalance step generates a *new* target position each cycle and diffs it
against the current holdings to produce buy/sell orders. These tests pin the
diff invariant: target > current → buy, target < current → sell, equal → hold,
and sells are ordered before buys (to free cash).
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.schemas import OrderSide  # noqa: E402
from tools.sizing import infer_lot_size, round_order_shares  # noqa: E402
from core.schemas import BacktestConfig  # noqa: E402


class LotSizingTest(unittest.TestCase):
    """A-share 100-share lot rounding — the constraint layer's last guard."""

    def test_akshare_provider_uses_100_share_lots(self) -> None:
        cfg = BacktestConfig(provider="akshare")
        self.assertEqual(infer_lot_size(cfg), 100)

    def test_synthetic_provider_uses_single_share_lots(self) -> None:
        cfg = BacktestConfig(provider="synthetic")
        self.assertEqual(infer_lot_size(cfg), 1)

    def test_round_down_to_lot_multiple(self) -> None:
        cfg = BacktestConfig(provider="akshare")
        # 250 shares → 200 (next 100-lot)
        self.assertEqual(round_order_shares(250, cfg, OrderSide.buy), 200)

    def test_round_below_one_lot_is_zero(self) -> None:
        cfg = BacktestConfig(provider="akshare")
        # 50 shares → 0 (below one lot)
        self.assertEqual(round_order_shares(50, cfg, OrderSide.buy), 0)

    def test_sell_capped_at_current_position(self) -> None:
        cfg = BacktestConfig(provider="akshare")
        # hold 150, sell 500 → capped at 150 (full close allowed even if < lot)
        self.assertEqual(round_order_shares(500, cfg, OrderSide.sell, current_position=150), 150)


class RebalanceDiffTest(unittest.TestCase):
    """The target-vs-current diff that drives buy/sell ordering.

    Mirrors engine/backtest.py:337-361 logic: sells first (to free cash),
    then buys; equal positions are skipped.
    """

    @staticmethod
    def _diff_orders(current: dict, target: dict) -> list:
        """Replicate the rebalance diff: returns [(symbol, side, shares)]."""
        current_symbols = set(current.keys()) | set(target.keys())
        orders = []
        for symbol in sorted(
            current_symbols,
            key=lambda s: 0 if current.get(s, 0) > target.get(s, 0) else 1,
        ):
            tgt = target.get(symbol, 0)
            cur = current.get(symbol, 0)
            if tgt == cur:
                continue
            side = OrderSide.buy if tgt > cur else OrderSide.sell
            orders.append((symbol, side, abs(tgt - cur)))
        return orders

    def test_new_position_generates_buy(self) -> None:
        orders = self._diff_orders(current={}, target={"000001": 2000})
        self.assertEqual(orders, [("000001", OrderSide.buy, 2000)])

    def test_reduced_position_generates_sell(self) -> None:
        orders = self._diff_orders(
            current={"000001": 2000}, target={"000001": 1200}
        )
        self.assertEqual(orders, [("000001", OrderSide.sell, 800)])

    def test_dropped_stock_generates_full_sell(self) -> None:
        orders = self._diff_orders(
            current={"000001": 1500}, target={}
        )
        self.assertEqual(orders, [("000001", OrderSide.sell, 1500)])

    def test_equal_position_is_skipped(self) -> None:
        orders = self._diff_orders(
            current={"000001": 2000}, target={"000001": 2000}
        )
        self.assertEqual(orders, [])

    def test_sells_ordered_before_buys(self) -> None:
        """Sells must precede buys so freed cash funds the buys."""
        orders = self._diff_orders(
            current={"000002": 1500},          # to be sold
            target={"000003": 1000},           # to be bought
        )
        # sell (000002) sorts before buy (000003) because current>target → key 0
        self.assertEqual(orders[0][1], OrderSide.sell)
        self.assertEqual(orders[-1][1], OrderSide.buy)

    def test_mixed_rebalance_produces_correct_sides(self) -> None:
        orders = self._diff_orders(
            current={"000001": 2000, "000002": 1500},
            target={"000001": 1200, "000003": 1000},
        )
        sides = {sym: side for sym, side, _ in orders}
        self.assertEqual(sides["000001"], OrderSide.sell)   # 2000→1200
        self.assertEqual(sides["000002"], OrderSide.sell)   # 1500→0
        self.assertEqual(sides["000003"], OrderSide.buy)    # 0→1000


if __name__ == "__main__":
    unittest.main()
