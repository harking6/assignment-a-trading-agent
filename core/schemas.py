from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class Stance(str, Enum):
    bullish = "bullish"
    bearish = "bearish"
    neutral = "neutral"


class OrderSide(str, Enum):
    buy = "buy"
    sell = "sell"
    hold = "hold"


class MarketBar(BaseModel):
    day: int
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int

    @property
    def price(self) -> float:
        return self.close

    def public_dict(self) -> Dict[str, Any]:
        return {
            "day": self.day,
            "date": self.date,
            "open": round(self.open, 4),
            "high": round(self.high, 4),
            "low": round(self.low, 4),
            "close": round(self.close, 4),
            "price": round(self.close, 4),
            "volume": self.volume,
        }


class QuoteTick(BaseModel):
    provider: str
    symbol: str
    timestamp: str
    price: float
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    prev_close: Optional[float] = None
    volume: Optional[int] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    source: str = "unknown"
    raw: Dict[str, Any] = Field(default_factory=dict)

    def to_bar(self, day: int = 1) -> MarketBar:
        base = self.price
        return MarketBar(
            day=day,
            date=self.timestamp[:10],
            open=self.open if self.open is not None else base,
            high=self.high if self.high is not None else base,
            low=self.low if self.low is not None else base,
            close=base,
            volume=self.volume or 0,
        )


class PortfolioSnapshot(BaseModel):
    """Portfolio snapshot supporting both single-symbol and multi-symbol modes.

    In single-symbol mode ``position`` and ``avg_entry`` hold the only holding.
    In multi-symbol mode ``positions`` and ``avg_entries`` hold per-symbol data,
    and ``for_symbol()`` returns a single-symbol view.
    """

    cash: float
    position: int = 0
    avg_entry: Optional[float] = None
    equity: float
    peak: float
    max_drawdown: float
    realized_trades: int = 0
    realized_wins: int = 0
    realized_pnl: float = 0.0
    total_fees: float = 0.0
    turnover: float = 0.0
    positions: Dict[str, int] = Field(default_factory=dict)
    avg_entries: Dict[str, float] = Field(default_factory=dict)

    def for_symbol(self, symbol: str) -> "PortfolioSnapshot":
        """Return a single-symbol view for the given symbol."""
        return self.model_copy(
            update={
                "position": self.positions.get(symbol, 0),
                "avg_entry": self.avg_entries.get(symbol),
                "positions": {symbol: self.positions.get(symbol, 0)},
                "avg_entries": {symbol: self.avg_entries.get(symbol, 0.0)} if self.avg_entries.get(symbol) is not None else {},
            }
        )

    @property
    def total_position(self) -> int:
        return sum(self.positions.values()) if self.positions else self.position


class AnalystReport(BaseModel):
    agent: str
    stance: Stance
    score: float = Field(ge=-1.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    summary: str
    evidence: List[str] = Field(default_factory=list)
    metrics: Dict[str, float] = Field(default_factory=dict)


class ResearchDebate(BaseModel):
    bull_thesis: str
    bear_thesis: str
    bull_score: float
    bear_score: float
    verdict: Stance
    conviction: float = Field(ge=0.0, le=1.0)
    manager_notes: str
    rounds: List[Dict[str, Any]] = Field(default_factory=list)


class TradeIntent(BaseModel):
    symbol: Optional[str] = None
    signal: str
    side: OrderSide
    target_position: int
    shares: int
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    evidence: List[str] = Field(default_factory=list)


class RiskAssessment(BaseModel):
    symbol: Optional[str] = None
    approved: bool
    side: OrderSide
    shares: int
    risk_score: float = Field(ge=0.0, le=1.0)
    reasons: List[str] = Field(default_factory=list)
    limits: Dict[str, float] = Field(default_factory=dict)


class RiskDebate(BaseModel):
    aggressive_case: str
    neutral_case: str
    conservative_case: str
    final_judgement: str
    final_side: OrderSide
    final_shares: int
    risk_score: float = Field(ge=0.0, le=1.0)
    rounds: List[Dict[str, Any]] = Field(default_factory=list)


class PortfolioDecision(BaseModel):
    symbol: Optional[str] = None
    side: OrderSide
    shares: int
    signal: str
    confidence: float
    reason: str
    blocked: bool = False


class ExecutionFill(BaseModel):
    symbol: Optional[str] = None
    day: int
    date: str
    price: float
    signal: str
    side: OrderSide
    shares: int
    reason: str
    confidence: float
    cash: float
    position: int
    equity: float
    fee: float
    realized_pnl: float
    blocked: bool = False

    def public_dict(self) -> Dict[str, Any]:
        payload = self.model_dump()
        payload["side"] = self.side.value
        return payload


class BacktestConfig(BaseModel):
    provider: str = "synthetic"
    symbol: str = "DEMO"
    benchmark_symbol: str = "BENCH"
    start: Optional[str] = None
    end: Optional[str] = None
    price_adjust: str = "qfq"
    dataset: str = "trend"
    strategy: str = "multi_agent"
    agent_mode: str = "auto"
    llm_model: str = "gpt-4o-mini"
    deep_model: str = "gpt-4o"
    llm_base_url: Optional[str] = None
    max_debate_rounds: int = 2
    max_risk_debate_rounds: int = 1
    memory_window: int = 6
    risk_budget: float = 0.30
    fee_rate: float = 0.0012
    slippage_bps: float = 2.0
    initial_cash: float = 100000.0
    max_single_order_shares: int = 5000
    max_order_value_ratio: float = 0.12
    rebalance_tolerance: float = 0.015
    min_trade_notional: float = 1000.0
    lot_size: int = 0
    max_position_ratio: float = 0.70
    cash_reserve_ratio: float = 0.12
    lookback: int = 30
    # Multi-symbol / top-down settings
    universe: List[str] = Field(default_factory=list)
    dynamic_universe: bool = False
    dynamic_universe_limit: int = 30
    dynamic_sector_pool: int = 8
    dynamic_min_turnover: float = 50_000_000.0
    max_positions: int = 10
    top_n_sectors: int = 3
    stocks_per_sector: int = 3
    rebalance_frequency: int = 5
    # Optional synthetic dataset length (used by synthetic/csv providers when set)
    days: Optional[int] = None


class GraphResult(BaseModel):
    reports: List[AnalystReport]
    debate: ResearchDebate
    intent: TradeIntent
    risk: RiskAssessment
    risk_debate: Optional[RiskDebate] = None
    decision: PortfolioDecision
    trace: List[Dict[str, str]]


class MultiGraphResult(BaseModel):
    """Output of the multi-symbol (top-down) LangGraph workflow.

    Carries the sector view dict (consumed verbatim by the frontend) and the
    absolute target positions ``{symbol: shares}`` that the engine diffs
    against current holdings to synthesize buy/sell decisions.
    """

    sector_view: Dict[str, Any]
    target_positions: Dict[str, int]
    trace: List[Dict[str, str]]
