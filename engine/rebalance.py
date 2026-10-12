"""Pure target-position diff planning for multi-symbol rebalances."""

from __future__ import annotations

from typing import List, Mapping

from core.schemas import BacktestConfig, OrderSide, PortfolioDecision
from tools.sizing import round_order_shares


def plan_rebalance_orders(
    current_positions: Mapping[str, int],
    target_positions: Mapping[str, int],
    config: BacktestConfig,
) -> List[PortfolioDecision]:
    """Convert current and target holdings into executable order intents.

    The ordering deliberately matches the legacy engine behavior: every sell
    is planned before every buy so sales can release cash.  Symbols within the
    same side retain the set iteration order used by the previous inline code;
    changing that tie order could change partial fills when cash is scarce.
    The function mutates neither input and performs no broker I/O.
    """
    current_symbols = set(current_positions.keys()) | set(target_positions.keys())
    decisions: List[PortfolioDecision] = []
    for symbol in sorted(
        current_symbols,
        key=lambda item: (
            0
            if current_positions.get(item, 0) > target_positions.get(item, 0)
            else 1
        ),
    ):
        target = target_positions.get(symbol, 0)
        current = current_positions.get(symbol, 0)
        if target == current:
            continue
        side = OrderSide.buy if target > current else OrderSide.sell
        shares = round_order_shares(
            abs(target - current),
            config,
            side,
            current,
        )
        if shares <= 0:
            continue
        decisions.append(
            PortfolioDecision(
                symbol=symbol,
                side=side,
                shares=shares,
                signal="top_down_rebalance",
                confidence=0.7,
                reason=f"top-down rebalance to {target} shares (current {current})",
                blocked=False,
            )
        )
    return decisions


__all__ = ["plan_rebalance_orders"]
