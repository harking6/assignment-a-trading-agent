from __future__ import annotations

from core.schemas import BacktestConfig, OrderSide


def infer_lot_size(config: BacktestConfig) -> int:
    if config.lot_size and config.lot_size > 0:
        return max(1, int(config.lot_size))
    if config.provider in {"akshare", "tushare"}:
        return 100
    symbol = str(config.symbol).strip().upper()
    if symbol.startswith(("SH", "SZ")) or symbol.endswith((".SH", ".SZ")):
        return 100
    return 1


def round_order_shares(shares: int, config: BacktestConfig, side: OrderSide, current_position: int = 0) -> int:
    shares = max(0, int(shares))
    lot = infer_lot_size(config)
    if shares <= 0 or lot <= 1:
        return shares
    if side == OrderSide.sell and current_position > 0 and shares >= current_position:
        return current_position
    return (shares // lot) * lot
