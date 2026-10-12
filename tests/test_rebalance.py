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
from engine.rebalance import plan_rebalance_orders  # noqa: E402
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
    """Exercise the production target-vs-current planning function directly."""

    @staticmethod
    def _orders(current: dict, target: dict, config=None) -> list:
        decisions = plan_rebalance_orders(
            current,
            target,
            config or BacktestConfig(provider="synthetic"),
        )
        return [(decision.symbol, decision.side, decision.shares) for decision in decisions]

    @staticmethod
    def _legacy_orders(current: dict, target: dict, config: BacktestConfig) -> list:
        """Frozen copy of the pre-A-06 inline engine behavior for parity."""
        orders = []
        symbols = set(current.keys()) | set(target.keys())
        for symbol in sorted(
            symbols,
            key=lambda item: 0 if current.get(item, 0) > target.get(item, 0) else 1,
        ):
            target_shares = target.get(symbol, 0)
            current_shares = current.get(symbol, 0)
            if target_shares == current_shares:
                continue
            side = OrderSide.buy if target_shares > current_shares else OrderSide.sell
            shares = round_order_shares(
                abs(target_shares - current_shares),
                config,
                side,
                current_shares,
            )
            if shares > 0:
                orders.append((symbol, side, shares))
        return orders

    def test_new_position_generates_buy(self) -> None:
        orders = self._orders(current={}, target={"000001": 2000})
        self.assertEqual(orders, [("000001", OrderSide.buy, 2000)])

    def test_reduced_position_generates_sell(self) -> None:
        orders = self._orders(
            current={"000001": 2000}, target={"000001": 1200}
        )
        self.assertEqual(orders, [("000001", OrderSide.sell, 800)])

    def test_dropped_stock_generates_full_sell(self) -> None:
        orders = self._orders(
            current={"000001": 1500}, target={}
        )
        self.assertEqual(orders, [("000001", OrderSide.sell, 1500)])

    def test_equal_position_is_skipped(self) -> None:
        orders = self._orders(
            current={"000001": 2000}, target={"000001": 2000}
        )
        self.assertEqual(orders, [])

    def test_sells_ordered_before_buys(self) -> None:
        """Sells must precede buys so freed cash funds the buys."""
        orders = self._orders(
            current={"000002": 1500},          # to be sold
            target={"000003": 1000},           # to be bought
        )
        # sell (000002) sorts before buy (000003) because current>target → key 0
        self.assertEqual(orders[0][1], OrderSide.sell)
        self.assertEqual(orders[-1][1], OrderSide.buy)

    def test_mixed_rebalance_produces_correct_sides(self) -> None:
        orders = self._orders(
            current={"000001": 2000, "000002": 1500},
            target={"000001": 1200, "000003": 1000},
        )
        sides = {sym: side for sym, side, _ in orders}
        self.assertEqual(sides["000001"], OrderSide.sell)   # 2000→1200
        self.assertEqual(sides["000002"], OrderSide.sell)   # 1500→0
        self.assertEqual(sides["000003"], OrderSide.buy)    # 0→1000

    def test_planner_does_not_mutate_input_positions(self) -> None:
        current = {"000001": 350}
        target = {"000001": 200, "000002": 250}
        current_before = dict(current)
        target_before = dict(target)

        plan_rebalance_orders(current, target, BacktestConfig(provider="akshare"))

        self.assertEqual(current, current_before)
        self.assertEqual(target, target_before)

    def test_a_share_rounding_is_applied_to_planned_orders(self) -> None:
        orders = self._orders(
            current={"000001": 350},
            target={"000001": 200, "000002": 250, "000003": 50},
            config=BacktestConfig(provider="akshare"),
        )
        by_symbol = {symbol: (side, shares) for symbol, side, shares in orders}

        self.assertEqual(by_symbol["000001"], (OrderSide.sell, 100))
        self.assertEqual(by_symbol["000002"], (OrderSide.buy, 200))
        self.assertNotIn("000003", by_symbol)

    def test_full_exit_keeps_odd_lot_position(self) -> None:
        orders = self._orders(
            current={"000001": 150},
            target={},
            config=BacktestConfig(provider="akshare"),
        )
        self.assertEqual(orders, [("000001", OrderSide.sell, 150)])

    def test_order_intent_metadata_matches_legacy_engine_contract(self) -> None:
        decisions = plan_rebalance_orders(
            {},
            {"000001": 200},
            BacktestConfig(provider="synthetic"),
        )
        decision = decisions[0]

        self.assertEqual(decision.signal, "top_down_rebalance")
        self.assertEqual(decision.confidence, 0.7)
        self.assertEqual(
            decision.reason,
            "top-down rebalance to 200 shares (current 0)",
        )
        self.assertFalse(decision.blocked)

    def test_extracted_planner_matches_legacy_direction_quantity_and_order(self) -> None:
        cases = [
            (
                {"000001": 350, "000002": 1500},
                {"000001": 200, "000003": 250, "000004": 50},
                BacktestConfig(provider="akshare"),
            ),
            (
                {"AAA": 10, "BBB": 5},
                {"BBB": 5, "CCC": 7, "DDD": 9},
                BacktestConfig(provider="synthetic"),
            ),
        ]
        for current, target, config in cases:
            with self.subTest(provider=config.provider):
                self.assertEqual(
                    self._orders(current, target, config),
                    self._legacy_orders(current, target, config),
                )


if __name__ == "__main__":
    unittest.main()
