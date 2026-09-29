from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict

from core.schemas import (
    BacktestConfig,
    ExecutionFill,
    PortfolioDecision,
    PortfolioSnapshot,
    ResearchDebate,
    RiskAssessment,
    RiskDebate,
    TradeIntent,
)


class TradingGraphState(TypedDict, total=False):
    """Typed state shared by the LangGraph analyst/researcher/trader/risk nodes.

    The pandas DataFrame and LLM client are intentionally NOT stored here; they
    are kept as runtime attributes on ``TradingAgentsGraph``.
    """

    day: int
    portfolio: PortfolioSnapshot
    config: BacktestConfig
    memory: List[Dict[str, Any]]
    memory_messages: List[Dict[str, Any]]
    reports: List[Any]
    rag_context: Dict[str, Any]
    tool_context: Dict[str, Any]
    debate: ResearchDebate
    intent: TradeIntent
    risk_debate: Optional[RiskDebate]
    risk: RiskAssessment
    decision: PortfolioDecision
    trace: List[Dict[str, str]]

    # --- Multi-symbol (top-down) channels; additive, do not remove the 14 above. ---
    market_snapshot: Dict[str, Any]
    prices: Dict[str, float]
    technical_context: Dict[str, Dict[str, float]]
    selected_sectors: List[str]
    sector_view: Dict[str, Any]
    target_positions: Dict[str, int]
    market_regime: Dict[str, Any]
    candidate_analysis: List[Dict[str, Any]]
    debate_result: Dict[str, Any]
    portfolio_plan: Dict[str, Any]
    # Experience-learning channels (recall/inject/record loop). Additive; the
    # single-symbol graph never touches them.
    kline_features: Dict[str, Any]
    sentiment: Dict[str, Any]
    memory_lesson: str
