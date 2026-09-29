from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Union

from core.schemas import BacktestConfig, ExecutionFill, MarketBar, OrderSide, PortfolioDecision, PortfolioSnapshot


@dataclass
class BrokerState:
    cash: float
    positions: Dict[str, int] = field(default_factory=dict)
    avg_entries: Dict[str, float] = field(default_factory=dict)
    peak: float = 0.0
    max_drawdown: float = 0.0
    realized_trades: int = 0
    realized_wins: int = 0
    realized_pnl: float = 0.0
    total_fees: float = 0.0
    turnover: float = 0.0

    @property
    def position(self) -> int:
        """Backward-compatible scalar position (single-symbol mode)."""
        return sum(self.positions.values())

    @property
    def avg_entry(self) -> Optional[float]:
        """Backward-compatible scalar avg_entry (single-symbol mode)."""
        if len(self.avg_entries) == 1:
            return next(iter(self.avg_entries.values()))
        return None

    def position_for(self, symbol: str) -> int:
        return self.positions.get(symbol, 0)

    def avg_entry_for(self, symbol: str) -> Optional[float]:
        return self.avg_entries.get(symbol)


class SimulatedBroker:
    def __init__(self, config: BacktestConfig) -> None:
        self.config = config
        self.state = BrokerState(cash=config.initial_cash, peak=config.initial_cash)

    def reset(self, config: BacktestConfig) -> None:
        self.config = config
        self.state = BrokerState(cash=config.initial_cash, peak=config.initial_cash)

    def _default_symbol(self, symbol: Optional[str] = None) -> str:
        return symbol or self.config.symbol

    def _normalize_prices(self, prices: Union[float, Dict[str, float]], symbol: Optional[str] = None) -> Dict[str, float]:
        if isinstance(prices, (int, float)):
            return {self._default_symbol(symbol): float(prices)}
        return prices

    def snapshot(self, prices: Union[float, Dict[str, float]], symbol: Optional[str] = None) -> PortfolioSnapshot:
        price_map = self._normalize_prices(prices, symbol)
        equity = self.equity(price_map)
        default_symbol = self._default_symbol(symbol)
        return PortfolioSnapshot(
            cash=self.state.cash,
            position=self.state.position_for(default_symbol),
            avg_entry=self.state.avg_entry_for(default_symbol),
            equity=equity,
            peak=self.state.peak,
            max_drawdown=self.state.max_drawdown,
            realized_trades=self.state.realized_trades,
            realized_wins=self.state.realized_wins,
            realized_pnl=self.state.realized_pnl,
            total_fees=self.state.total_fees,
            turnover=self.state.turnover,
            positions=dict(self.state.positions),
            avg_entries=dict(self.state.avg_entries),
        )

    def equity(self, prices: Union[float, Dict[str, float]]) -> float:
        price_map = self._normalize_prices(prices)
        market_value = sum(
            self.state.position_for(sym) * price for sym, price in price_map.items()
        )
        return self.state.cash + market_value

    def execute(
        self,
        bar: MarketBar,
        decision: PortfolioDecision,
        symbol: Optional[str] = None,
        all_prices: Optional[Union[float, Dict[str, float]]] = None,
    ) -> ExecutionFill:
        sym = self._default_symbol(symbol or decision.symbol)
        price = self.execution_price(bar.close, decision.side)
        fee = 0.0
        realized_pnl = 0.0
        shares = max(0, int(decision.shares))
        blocked = decision.blocked
        reason = decision.reason

        current_position = self.state.position_for(sym)

        if blocked or shares <= 0 or decision.side == OrderSide.hold:
            shares = 0
            blocked = blocked or (decision.side != OrderSide.hold and decision.shares > 0)
        elif decision.side == OrderSide.buy:
            max_affordable = int(self.state.cash // (price * (1.0 + self.config.fee_rate))) if price > 0 else 0
            requested = shares
            shares = min(requested, max_affordable)
            if shares <= 0:
                blocked = True
                reason = f"{reason}; blocked by broker: insufficient cash"
            elif shares < requested:
                reason = f"{reason}; partial fill due to cash limit ({shares}/{requested})"
            if shares > 0:
                gross = shares * price
                fee = gross * self.config.fee_rate
                old_cost = (self.state.avg_entry_for(sym) or price) * current_position
                self.state.cash -= gross + fee
                self.state.positions[sym] = current_position + shares
                total_shares = self.state.positions[sym]
                self.state.avg_entries[sym] = (old_cost + gross) / total_shares if total_shares else 0.0
        elif decision.side == OrderSide.sell:
            requested = shares
            shares = min(requested, current_position)
            if shares <= 0:
                blocked = True
                reason = f"{reason}; blocked by broker: no shares to sell"
            elif shares < requested:
                reason = f"{reason}; partial fill due to position limit ({shares}/{requested})"
            if shares > 0:
                gross = shares * price
                fee = gross * self.config.fee_rate
                self.state.cash += gross - fee
                avg_cost = self.state.avg_entry_for(sym) or price
                realized_pnl = ((price - avg_cost) * shares) - fee
                self.state.positions[sym] = current_position - shares
                self.state.realized_trades += 1
                if realized_pnl > 0:
                    self.state.realized_wins += 1
                if self.state.positions[sym] <= 0:
                    del self.state.positions[sym]
                    self.state.avg_entries.pop(sym, None)
        if shares > 0:
            self.state.total_fees += fee
            self.state.turnover += shares * price
            self.state.realized_pnl += realized_pnl
        equity = self.equity(all_prices if all_prices is not None else bar.close)
        self.state.peak = max(self.state.peak, equity)
        self.state.max_drawdown = min(self.state.max_drawdown, equity / self.state.peak - 1.0 if self.state.peak else 0.0)
        return ExecutionFill(
            symbol=sym,
            day=bar.day,
            date=bar.date,
            price=price,
            signal=decision.signal,
            side=decision.side,
            shares=shares,
            reason=reason,
            confidence=decision.confidence,
            cash=self.state.cash,
            position=self.state.position_for(sym),
            equity=equity,
            fee=fee,
            realized_pnl=realized_pnl,
            blocked=blocked,
        )

    def execution_price(self, close: float, side: OrderSide) -> float:
        slippage = self.config.slippage_bps / 10000.0
        if side == OrderSide.buy:
            return close * (1.0 + slippage)
        if side == OrderSide.sell:
            return close * (1.0 - slippage)
        return close
