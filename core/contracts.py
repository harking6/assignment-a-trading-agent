"""Structural service boundaries used by the backtest runtime.

The protocols describe the behavior the engine consumes without requiring
concrete implementations to inherit from a project base class.  This keeps the
default implementations compatible while allowing deterministic substitutes
to be injected by later architecture work.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Protocol, runtime_checkable

from core.schemas import (
    BacktestConfig,
    ExecutionFill,
    GraphResult,
    MarketBar,
    MultiGraphResult,
    PortfolioSnapshot,
    QuoteTick,
)

if TYPE_CHECKING:
    from pandas import DataFrame
else:
    # Keep pandas out of the runtime dependency path for this foundational
    # module.  Concrete graph and universe implementations still use frames.
    DataFrame = Any


@runtime_checkable
class GraphRunner(Protocol):
    """Decision workflow consumed by ``TradingBacktestEngine``."""

    def reconfigure_selector(
        self,
        config: BacktestConfig,
        universe_frame: DataFrame | None = None,
    ) -> None:
        """Bind multi-symbol selection services to a backtest configuration."""
        ...

    def run(
        self,
        frame: DataFrame,
        day: int,
        portfolio: PortfolioSnapshot,
        config: BacktestConfig,
    ) -> GraphResult:
        """Return one single-symbol decision cycle."""
        ...

    def run_multi(
        self,
        day: int,
        portfolio: PortfolioSnapshot,
        config: BacktestConfig,
        market_snapshot: Dict[str, Any] | None,
        prices: Dict[str, float] | None,
        technical_context: Dict[str, Dict[str, float]] | None,
    ) -> MultiGraphResult:
        """Return absolute target positions for one multi-symbol cycle."""
        ...

    def record_outcome(
        self,
        config: BacktestConfig,
        day: int,
        graph_result: GraphResult,
        fill: ExecutionFill,
    ) -> None:
        """Record the observable outcome of a single-symbol decision."""
        ...


@runtime_checkable
class MarketDataSource(Protocol):
    """Historical and point-in-time market data required by the engine."""

    def load(
        self,
        dataset: str,
        symbol: str = "DEMO",
        start: str | None = None,
        end: str | None = None,
        adjust: str | None = None,
    ) -> List[MarketBar]:
        """Load normalized OHLCV bars for a symbol and date range."""
        ...

    def latest_quote(self, dataset: str, symbol: str = "DEMO") -> QuoteTick:
        """Return the latest normalized quote available to the provider."""
        ...

    def instrument_info(self, symbol: str = "DEMO") -> Dict[str, str]:
        """Return display metadata for a symbol."""
        ...


@runtime_checkable
class UniverseSource(Protocol):
    """Normalized A-share universe inputs used by top-down selection."""

    def list_spot(self) -> DataFrame:
        """Return the normalized full-market snapshot."""
        ...

    def list_sectors(self) -> DataFrame:
        """Return normalized sector summary rows."""
        ...

    def list_sector_stocks(self, sector: str) -> DataFrame:
        """Return normalized constituents for one sector."""
        ...


__all__ = ["GraphRunner", "MarketDataSource", "UniverseSource"]
