from __future__ import annotations

import os
from typing import Any, Dict

import pandas as pd

from agents.analysts import FundamentalsAnalyst, MarketAnalyst, NewsAnalyst, SentimentAnalyst
from agents.llm_client import LLMClient, clamp_float
from agents.memory import TradingMemory
from agents.portfolio_manager import PortfolioManager
from agents.researchers import ResearchManager
from agents.risk import RiskCommittee, RiskDebatePanel
from agents.state import TradingGraphState
from agents.top_down import TopDownSelector
from core.assignment import assignment_todo
from core.schemas import BacktestConfig, ExecutionFill, GraphResult, MultiGraphResult, OrderSide, PortfolioSnapshot, Stance, TradeIntent
from tools.langchain_tools import invoke_langchain_tools
from tools.news import MultiSentimentFeed
from tools.rag import MultiKnowledgeBase, TradingRAG
from tools.sizing import infer_lot_size, round_order_shares

try:
    from langgraph.graph import END, START, StateGraph
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    _LANGGRAPH_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    _LANGGRAPH_AVAILABLE = False


# Pydantic models and enums that may travel through checkpointed state.
# with_msgpack_allowlist expects tuples ("module", "ClassName") or type objects;
# plain strings are iterated char-by-char and silently fail to register.
_JSONPLUS_ALLOWLIST = [
    ("core.schemas", "AnalystReport"),
    ("core.schemas", "BacktestConfig"),
    ("core.schemas", "ExecutionFill"),
    ("core.schemas", "GraphResult"),
    ("core.schemas", "MarketBar"),
    ("core.schemas", "PortfolioDecision"),
    ("core.schemas", "PortfolioSnapshot"),
    ("core.schemas", "QuoteTick"),
    ("core.schemas", "ResearchDebate"),
    ("core.schemas", "RiskAssessment"),
    ("core.schemas", "RiskDebate"),
    ("core.schemas", "TradeIntent"),
    ("core.schemas", "MultiGraphResult"),
    ("core.schemas", "OrderSide"),
    ("core.schemas", "Stance"),
]


class TradingAgentsGraph:
    """TradingAgents-inspired orchestration graph.

    Nodes mirror the open-source TradingAgents pattern: analyst team,
    researcher debate, trader, risk team, and portfolio manager. LangGraph is
    used when available; otherwise the same nodes run in deterministic order.
    """

    def __init__(self) -> None:
        self.market_analyst = MarketAnalyst()
        self.sentiment_analyst = SentimentAnalyst()
        self.news_analyst = NewsAnalyst()
        self.fundamentals_analyst = FundamentalsAnalyst()
        self.research_manager = ResearchManager()
        self.risk_debate_panel = RiskDebatePanel()
        self.risk_committee = RiskCommittee()
        self.portfolio_manager = PortfolioManager()
        self.memory = TradingMemory()
        self.rag = TradingRAG()
        # Multi-symbol experience-learning helpers: market/per-stock sentiment
        # and book-excerpt + industry-note knowledge recall. Both degrade to
        # empty when their inputs are missing, so the LLM keeps working.
        self.sentiment_feed = MultiSentimentFeed()
        self.knowledge_base = MultiKnowledgeBase()
        self.checkpointer = None
        self.checkpoint_enabled = False
        self.checkpoint_error: str | None = None
        self._runtime_frame: pd.DataFrame | None = None
        self._runtime_llm: LLMClient | None = None
        # Multi-symbol (top-down) runtime. The selector is NOT constructed here
        # because __init__ has no BacktestConfig; the engine injects a fresh,
        # reset-safe selector via reconfigure_selector() at reset() time.
        self.top_down: TopDownSelector | None = None
        self._runtime_candidates: pd.DataFrame | None = None
        self._runtime_selected: pd.DataFrame | None = None
        self.checkpointer_multi = None
        self.checkpoint_enabled_multi = False
        self.checkpoint_error_multi: str | None = None
        self._compiled_multi = self._try_build_langgraph_multi()
        self._compiled = self._try_build_langgraph()

    def reconfigure_selector(self, config: BacktestConfig, universe_frame: pd.DataFrame | None = None) -> None:
        """Inject a fresh top-down selector bound to ``config``.

        Called by the engine on every reset() so the selector never goes stale
        when the user changes universe / agent_mode / llm_model / etc.
        """
        self.top_down = TopDownSelector(config)
        self.top_down.set_universe_frame(universe_frame)

    def run(
        self,
        frame: pd.DataFrame,
        day: int,
        portfolio: PortfolioSnapshot,
        config: BacktestConfig,
    ) -> GraphResult:
        llm = LLMClient(config.agent_mode, config.llm_model, config.deep_model, config.llm_base_url)
        self._runtime_frame = frame
        self._runtime_llm = llm
        memory_history = self.memory.chat_history(config.symbol, config.memory_window)
        initial: TradingGraphState = {
            "day": day,
            "portfolio": portfolio,
            "config": config,
            "memory": self.memory.recent(config.symbol, config.memory_window),
            "memory_messages": [{"type": message.type, "content": str(message.content)} for message in memory_history.messages],
            "trace": [{"tool": "agent_runtime", "detail": f"mode={config.agent_mode}, llm_enabled={llm.enabled}"}],
        }
        if self._compiled is None:
            assignment_todo(
                "LG-02",
                "LangGraph workflow",
                "define TradingGraphState and compile the node/edge workflow in _try_build_langgraph",
            )
        invoke_config = {"configurable": {"thread_id": f"{config.provider}:{config.symbol}:{config.dataset}"}}
        try:
            result = self._compiled.invoke(initial, config=invoke_config)
        except TypeError as exc:
            if self.checkpoint_enabled:
                self.checkpoint_enabled = False
                self.checkpoint_error = str(exc)
                self._compiled = self._compile_without_checkpoint()
                if self._compiled is None:
                    raise
                result = self._compiled.invoke(initial, config=invoke_config)
            else:
                raise
        if self.checkpoint_enabled:
            result["trace"].append({"tool": "langgraph_checkpoint", "detail": "MemorySaver checkpoint enabled"})
        elif self.checkpoint_error:
            result["trace"].append({"tool": "langgraph_checkpoint", "detail": f"disabled: {self.checkpoint_error}"})
        return GraphResult(
            reports=result["reports"],
            debate=result["debate"],
            intent=result["intent"],
            risk=result["risk"],
            risk_debate=result.get("risk_debate"),
            decision=result["decision"],
            trace=result["trace"],
        )

    def run_multi(
        self,
        day: int,
        portfolio: PortfolioSnapshot,
        config: BacktestConfig,
        market_snapshot: Dict[str, Any] | None,
        prices: Dict[str, float] | None,
        technical_context: Dict[str, Dict[str, float]] | None,
    ) -> MultiGraphResult:
        """Run the multi-symbol LLM-first graph: regime → sector → stock → debate → portfolio.

        Returns the sector view (frontend contract) and absolute target
        positions ``{symbol: shares}`` that the engine diffs against current
        holdings to synthesize buy/sell decisions.
        """
        if self.top_down is None:
            self.reconfigure_selector(config)
        if config.agent_mode.lower() == "llm" and not self.top_down.llm.enabled:
            raise RuntimeError(
                "Multi-symbol agent_mode='llm' requires OPENAI_API_KEY; "
                "use agent_mode='offline' only when an explicit rule-only run is intended."
            )
        if self._compiled_multi is None:
            assignment_todo(
                "LG-MULTI",
                "multi-symbol LangGraph workflow",
                "compile the top-down node/edge workflow in _try_build_langgraph_multi",
            )
        self._runtime_candidates = None
        self._runtime_selected = None
        initial: TradingGraphState = {
            "day": day,
            "portfolio": portfolio,
            "config": config,
            "market_snapshot": market_snapshot or {},
            "prices": prices or {},
            "technical_context": technical_context or {},
            "selected_sectors": [],
            "sector_view": {},
            "target_positions": {},
            "market_regime": {},
            "candidate_analysis": [],
            "debate_result": {},
            "portfolio_plan": {},
            "kline_features": {},
            "sentiment": {},
            "memory_lesson": "",
            "trace": [
                {
                    "tool": "multi_agent_runtime",
                    "detail": f"mode={config.agent_mode}, universe={config.universe}, top_n_sectors={config.top_n_sectors}",
                }
            ],
        }
        invoke_config = {"configurable": {"thread_id": f"{config.provider}:multi:{config.dataset}"}}
        try:
            result = self._compiled_multi.invoke(initial, config=invoke_config)
        except TypeError as exc:
            if self.checkpoint_enabled_multi:
                self.checkpoint_enabled_multi = False
                self.checkpoint_error_multi = str(exc)
                self._compiled_multi = self._compile_multi_without_checkpoint()
                if self._compiled_multi is None:
                    raise
                result = self._compiled_multi.invoke(initial, config=invoke_config)
            else:
                raise
        if self.checkpoint_enabled_multi:
            result["trace"].append({"tool": "langgraph_checkpoint", "detail": "MemorySaver checkpoint enabled (multi)"})
        elif self.checkpoint_error_multi:
            result["trace"].append({"tool": "langgraph_checkpoint", "detail": f"disabled: {self.checkpoint_error_multi}"})
        return MultiGraphResult(
            sector_view=result.get("sector_view") or {},
            target_positions=result.get("target_positions") or {},
            trace=result.get("trace") or [],
        )

    def _frame(self) -> pd.DataFrame:
        if self._runtime_frame is None:
            raise RuntimeError("runtime frame is not initialized")
        return self._runtime_frame

    def _llm(self) -> LLMClient | None:
        return self._runtime_llm

    def _fallback_run(self, initial: TradingGraphState) -> TradingGraphState:
        """Optional TODO LG-05: implement a deterministic graph-free fallback."""

        assignment_todo(
            "LG-05",
            "deterministic orchestration fallback",
            "execute the same node order and conditional branches without StateGraph",
        )

    def _try_build_langgraph(self):
        """LG-02: construct and compile the complete LangGraph workflow.

        Nodes: analysts, langchain_tools, research, hold, trader, risk_debate,
        risk, portfolio. Edges follow START→analyst→tools→research; a
        conditional edge after research routes neutral verdicts to ``hold`` and
        directional verdicts to ``trader``; a conditional edge after the
        trader routes executable orders through ``risk_debate`` and otherwise
        straight to ``risk``; ``hold`` and ``risk_debate`` also pass through
        ``risk`` before all paths funnel into ``portfolio`` and then END.

        Compiled with ``MemorySaver`` using a ``JsonPlusSerializer`` whose
        msgpack allow-list covers the Pydantic models and enums in
        ``core.schemas`` so checkpointed state survives round-trips. Graph
        construction errors return None so the web server still starts and
        reports the LG-02 error when /api/step is called.
        """
        if not _LANGGRAPH_AVAILABLE:
            self.checkpoint_enabled = False
            self.checkpoint_error = "[LG-02] LangGraph is not installed"
            return None
        try:
            workflow = StateGraph(TradingGraphState)
            workflow.add_node("analyst", self._analyst_node)
            workflow.add_node("langchain_tools", self._langchain_tools_node)
            workflow.add_node("research", self._research_node)
            workflow.add_node("hold", self._hold_node)
            workflow.add_node("trader", self._trader_node)
            workflow.add_node("risk_debate", self._risk_debate_node)
            workflow.add_node("risk", self._risk_node)
            workflow.add_node("portfolio", self._portfolio_node)

            workflow.add_edge(START, "analyst")
            workflow.add_edge("analyst", "langchain_tools")
            workflow.add_edge("langchain_tools", "research")
            workflow.add_conditional_edges(
                "research",
                self._route_after_research,
                {"hold": "hold", "trade": "trader"},
            )
            workflow.add_conditional_edges(
                "trader",
                self._route_after_trader,
                {"debate_risk": "risk_debate", "risk_only": "risk"},
            )
            workflow.add_edge("hold", "risk")
            workflow.add_edge("risk_debate", "risk")
            workflow.add_edge("risk", "portfolio")
            workflow.add_edge("portfolio", END)

            serde = JsonPlusSerializer(allowed_msgpack_modules=tuple(_JSONPLUS_ALLOWLIST))
            self.checkpointer = MemorySaver(serde=serde)
            compiled = workflow.compile(checkpointer=self.checkpointer)
            self.checkpoint_enabled = True
            self.checkpoint_error = None
            return compiled
        except Exception as exc:  # pragma: no cover - defensive
            self.checkpoint_enabled = False
            self.checkpointer = None
            self.checkpoint_error = f"[LG-02] {type(exc).__name__}: {exc}"
            return None

    def _compile_without_checkpoint(self):
        """LG-03: compile the same workflow without a checkpointer."""

        if not _LANGGRAPH_AVAILABLE:
            assignment_todo(
                "LG-03",
                "checkpoint-free LangGraph compilation",
                "install LangGraph to rebuild the LG-02 topology without MemorySaver",
            )
        workflow = StateGraph(TradingGraphState)
        workflow.add_node("analyst", self._analyst_node)
        workflow.add_node("langchain_tools", self._langchain_tools_node)
        workflow.add_node("research", self._research_node)
        workflow.add_node("hold", self._hold_node)
        workflow.add_node("trader", self._trader_node)
        workflow.add_node("risk_debate", self._risk_debate_node)
        workflow.add_node("risk", self._risk_node)
        workflow.add_node("portfolio", self._portfolio_node)

        workflow.add_edge(START, "analyst")
        workflow.add_edge("analyst", "langchain_tools")
        workflow.add_edge("langchain_tools", "research")
        workflow.add_conditional_edges(
            "research",
            self._route_after_research,
            {"hold": "hold", "trade": "trader"},
        )
        workflow.add_conditional_edges(
            "trader",
            self._route_after_trader,
            {"debate_risk": "risk_debate", "risk_only": "risk"},
        )
        workflow.add_edge("hold", "risk")
        workflow.add_edge("risk_debate", "risk")
        workflow.add_edge("risk", "portfolio")
        workflow.add_edge("portfolio", END)
        return workflow.compile()

    # ------------------------------------------------------------------
    # Multi-symbol topology: factor pre-filter followed by a multi-stage LLM
    # decision chain. In explicit offline mode the same nodes use deterministic
    # fallbacks, preserving reproducible classroom runs.
    # ------------------------------------------------------------------

    def _build_multi_workflow(self):
        """Return the un-compiled multi-symbol StateGraph (shared by both compiles)."""
        workflow = StateGraph(TradingGraphState)
        workflow.add_node("universe", self._universe_node)
        workflow.add_node("market_regime", self._market_regime_node)
        workflow.add_node("sector_selector", self._sector_selector_node)
        workflow.add_node("stock_selector", self._stock_selector_node)
        workflow.add_node("candidate_analyst", self._candidate_analyst_node)
        workflow.add_node("debate_ranker", self._debate_ranker_node)
        workflow.add_node("portfolio_planner", self._portfolio_planner_node)
        workflow.add_node("allocator", self._allocator_node)

        workflow.add_edge(START, "universe")
        workflow.add_conditional_edges(
            "universe",
            self._route_after_universe,
            {"end": END, "continue": "market_regime"},
        )
        workflow.add_edge("market_regime", "sector_selector")
        workflow.add_edge("sector_selector", "stock_selector")
        workflow.add_edge("stock_selector", "candidate_analyst")
        workflow.add_edge("candidate_analyst", "debate_ranker")
        workflow.add_edge("debate_ranker", "portfolio_planner")
        workflow.add_edge("portfolio_planner", "allocator")
        workflow.add_edge("allocator", END)
        return workflow

    def _try_build_langgraph_multi(self):
        """Compile the multi-symbol top-down graph with checkpointing."""
        if not _LANGGRAPH_AVAILABLE:
            self.checkpoint_enabled_multi = False
            self.checkpoint_error_multi = "[LG-MULTI] LangGraph is not installed"
            return None
        try:
            workflow = self._build_multi_workflow()
            serde = JsonPlusSerializer(allowed_msgpack_modules=tuple(_JSONPLUS_ALLOWLIST))
            self.checkpointer_multi = MemorySaver(serde=serde)
            compiled = workflow.compile(checkpointer=self.checkpointer_multi)
            self.checkpoint_enabled_multi = True
            self.checkpoint_error_multi = None
            return compiled
        except Exception as exc:  # pragma: no cover - defensive
            self.checkpoint_enabled_multi = False
            self.checkpointer_multi = None
            self.checkpoint_error_multi = f"[LG-MULTI] {type(exc).__name__}: {exc}"
            return None

    def _compile_multi_without_checkpoint(self):
        """Compile the multi-symbol top-down graph without a checkpointer."""
        if not _LANGGRAPH_AVAILABLE:
            assignment_todo(
                "LG-MULTI",
                "checkpoint-free multi-symbol LangGraph compilation",
                "install LangGraph to rebuild the multi topology without MemorySaver",
            )
        workflow = self._build_multi_workflow()
        return workflow.compile()

    def _analyst_node(self, state: TradingGraphState) -> TradingGraphState:
        frame = self._frame()
        day = state["day"]
        portfolio = state["portfolio"]
        config = state["config"]
        reports = [
            self.market_analyst.run(frame, day, portfolio),
            self.sentiment_analyst.run(frame, day, config.dataset),
            self.news_analyst.run(frame, day, config.dataset),
            self.fundamentals_analyst.run(config.dataset),
        ]
        reports = self._llm_refine_reports(reports, state)
        state["reports"] = reports
        state["rag_context"] = self.rag.build_context(
            symbol=config.symbol,
            day=day,
            frame=frame,
            reports=reports,
            memory=state.get("memory") or [],
            config=config,
        )
        state["trace"].append({"tool": "analyst_team", "detail": "market/sentiment/news/fundamentals reports generated"})
        state["trace"].append(
            {
                "tool": "langchain_rag",
                "detail": f"documents={state['rag_context']['document_count']}, sources={state['rag_context']['sources']}",
            }
        )
        return state

    def _langchain_tools_node(self, state: TradingGraphState) -> TradingGraphState:
        row = self._frame().iloc[state["day"]]
        tool_context = invoke_langchain_tools(row, state["portfolio"], state["config"])
        state["tool_context"] = tool_context
        state["trace"].append({"tool": "langchain_tools", "detail": str(tool_context)})
        return state

    def _research_node(self, state: TradingGraphState) -> TradingGraphState:
        config = state["config"]
        debate = self.research_manager.debate(
            state["reports"],
            rounds=config.max_debate_rounds,
            memory=state.get("memory") or [],
            memory_messages=state.get("memory_messages") or [],
            rag_context=state.get("rag_context"),
            tool_context=state.get("tool_context"),
            llm=self._llm(),
        )
        state["debate"] = debate
        state["trace"].append({"tool": "research_debate", "detail": debate.manager_notes})
        return state

    def _route_after_research(self, state: TradingGraphState) -> str:
        """LG-04A: route neutral research to hold and directional research to trade."""

        debate = state.get("debate")
        verdict = getattr(debate, "verdict", None)
        route = "hold" if verdict == Stance.neutral else "trade"
        state.setdefault("trace", []).append(
            {"tool": "research_router", "detail": f"verdict={getattr(verdict,'value',verdict)} -> {route}"}
        )
        return route

    def _hold_node(self, state: TradingGraphState) -> TradingGraphState:
        debate = state["debate"]
        intent = TradeIntent(
            signal="hold",
            side=OrderSide.hold,
            target_position=state["portfolio"].position,
            shares=0,
            confidence=debate.conviction,
            rationale="LangGraph 条件边进入 hold 节点：研究裁决中性，等待更强信号",
            evidence=[report.summary for report in state["reports"]],
        )
        state["intent"] = intent
        state["trace"].append({"tool": "hold_node", "detail": intent.model_dump_json()})
        return state

    def _trader_node(self, state: TradingGraphState) -> TradingGraphState:
        frame = self._frame()
        day = state["day"]
        portfolio = state["portfolio"]
        config = state["config"]
        row = frame.iloc[day]
        debate = state["debate"]
        target = portfolio.position
        signal = "hold"
        side = OrderSide.hold
        rationale = "研究团队无明确方向，维持当前仓位"
        confidence = debate.conviction

        if debate.verdict == Stance.bullish:
            style_multiplier = 1.0
            if config.strategy == "risk":
                style_multiplier = 0.62
            elif config.strategy == "aggressive":
                style_multiplier = 1.28
            risk_scale = min(1.0, max(0.2, debate.conviction))
            effective_budget = min(config.max_position_ratio, config.risk_budget * style_multiplier)
            target = int((portfolio.equity * effective_budget * risk_scale) // float(row["close"]))
            signal = "accumulate"
            side = OrderSide.buy if target > portfolio.position else OrderSide.hold
            rationale = f"研究裁决偏多：{debate.bull_thesis}"
        elif debate.verdict == Stance.bearish:
            if config.strategy == "aggressive" and debate.conviction < 0.62:
                target = int(portfolio.position * 0.6)
            else:
                target = 0 if debate.conviction > 0.5 else int(portfolio.position * 0.5)
            signal = "de_risk"
            side = OrderSide.sell if target < portfolio.position else OrderSide.hold
            rationale = f"研究裁决偏空：{debate.bear_thesis}"

        shares, sizing_note = self._size_rebalance_order(target, portfolio, row, config, confidence)
        if shares <= 0:
            side = OrderSide.hold
            if target != portfolio.position:
                rationale = f"{rationale} | {sizing_note}"
        else:
            rationale = f"{rationale} | {sizing_note}"
        intent = TradeIntent(
            signal=signal,
            side=side,
            target_position=target,
            shares=shares,
            confidence=confidence,
            rationale=rationale,
            evidence=[report.summary for report in state["reports"]],
        )
        intent = self._llm_trade_overlay(intent, state)
        state["intent"] = intent
        state["trace"].append({"tool": "trader_agent", "detail": intent.model_dump_json()})
        return state

    def _size_rebalance_order(
        self,
        target: int,
        portfolio: PortfolioSnapshot,
        row: pd.Series,
        config: BacktestConfig,
        confidence: float,
    ) -> tuple[int, str]:
        price = max(0.0, float(row["close"]))
        delta = int(target) - int(portfolio.position)
        side = OrderSide.buy if delta > 0 else OrderSide.sell if delta < 0 else OrderSide.hold
        if side == OrderSide.hold or price <= 0:
            return 0, "仓位已接近目标，不交易"

        abs_delta = abs(delta)
        delta_notional = abs_delta * price
        tolerance_notional = max(float(config.min_trade_notional), float(portfolio.equity) * float(config.rebalance_tolerance))
        lot = infer_lot_size(config)
        if delta_notional < tolerance_notional or abs_delta < lot:
            return 0, (
                f"目标差额 {abs_delta} 股 / {delta_notional:.0f} 元低于再平衡阈值 "
                f"{tolerance_notional:.0f} 元，避免日内噪声交易"
            )

        atr_pct = float(row.get("atr_14", 0.0) or 0.0) / price if price else 0.0
        volatility = float(row.get("volatility_20", 0.0) or 0.0)
        conviction = max(0.15, min(1.0, float(confidence)))
        volatility_penalty = min(0.55, max(0.0, (atr_pct - 0.025) * 6.0 + (volatility - 0.020) * 4.0))
        tranche_ratio = max(0.12, (0.25 + 0.55 * conviction) * (1.0 - volatility_penalty))
        tranche_shares = max(1, int(abs_delta * tranche_ratio))
        cap_notional = max(float(config.min_trade_notional), float(portfolio.equity) * float(config.max_order_value_ratio) * max(0.45, conviction))
        cap_shares = max(1, int(cap_notional // price))

        raw_shares = min(abs_delta, tranche_shares, cap_shares, int(config.max_single_order_shares))
        shares = round_order_shares(raw_shares, config, side, portfolio.position)
        if shares <= 0 and abs_delta >= lot and delta_notional >= tolerance_notional:
            shares = min(abs_delta, lot)
            shares = round_order_shares(shares, config, side, portfolio.position)
        if shares <= 0:
            return 0, f"订单小于最小交易手数 {lot}，跳过"
        return shares, (
            f"仓位管理：target={target}, current={portfolio.position}, delta={delta}, "
            f"order={shares}, lot={lot}, tranche={tranche_ratio:.2f}, "
            f"cap_notional={cap_notional:.0f}, atr={atr_pct:.2%}, vol={volatility:.2%}"
        )

    def _route_after_trader(self, state: TradingGraphState) -> str:
        """LG-04B: route executable orders through risk debate."""

        intent = state.get("intent")
        side = getattr(intent, "side", None)
        shares = getattr(intent, "shares", 0) or 0
        route = "debate_risk" if side != OrderSide.hold and shares > 0 else "risk_only"
        state.setdefault("trace", []).append(
            {"tool": "trader_router", "detail": f"side={getattr(side,'value',side)}, shares={shares} -> {route}"}
        )
        return route

    def _risk_debate_node(self, state: TradingGraphState) -> TradingGraphState:
        price = float(self._frame().iloc[state["day"]]["close"])
        risk_debate = self.risk_debate_panel.debate(state["intent"], state["portfolio"], price, state["config"])
        state["risk_debate"] = risk_debate
        if risk_debate.final_side != state["intent"].side or risk_debate.final_shares != state["intent"].shares:
            payload = state["intent"].model_dump()
            payload["side"] = risk_debate.final_side
            payload["shares"] = risk_debate.final_shares
            payload["rationale"] = f"{state['intent'].rationale} | {risk_debate.final_judgement}"
            state["intent"] = TradeIntent(**payload)
        state["trace"].append({"tool": "risk_debate", "detail": risk_debate.model_dump_json()})
        return state

    def _risk_node(self, state: TradingGraphState) -> TradingGraphState:
        price = float(self._frame().iloc[state["day"]]["close"])
        risk = self.risk_committee.assess(state["intent"], state["portfolio"], price, state["config"])
        state["risk"] = risk
        state["trace"].append({"tool": "risk_committee", "detail": risk.model_dump_json()})
        return state

    def _portfolio_node(self, state: TradingGraphState) -> TradingGraphState:
        decision = self.portfolio_manager.decide(state["intent"], state["risk"])
        state["decision"] = decision
        state["trace"].append({"tool": "portfolio_manager", "detail": decision.model_dump_json()})
        return state

    # ------------------------------------------------------------------
    # Multi-symbol (top-down) nodes. Each wraps one TopDownSelector step so the
    # sector → stock → weight pipeline runs as LangGraph nodes. State mirrors
    # the assembly order in TopDownSelector.run_with_view.
    # ------------------------------------------------------------------

    def _universe_node(self, state: TradingGraphState) -> TradingGraphState:
        prices = state.get("prices") or {}
        technical_context = state.get("technical_context") or {}
        universe = self.top_down._load_universe()
        if prices:
            price_series = pd.Series(prices)
            universe = universe.copy()
            universe["close"] = universe["symbol"].map(price_series).fillna(universe["close"])
            if technical_context:
                returns = {s: ctx.get("return_1d", 0.0) for s, ctx in technical_context.items()}
                return_series = pd.Series(returns)
                universe["change_pct"] = universe["symbol"].map(return_series).fillna(universe["change_pct"])
        if universe.empty:
            self._runtime_candidates = None
            state["sector_view"] = {}
            state["target_positions"] = {}
            state["trace"].append({"tool": "universe_node", "detail": "empty universe → END"})
            return state
        candidates = self.top_down._pre_filter(universe, technical_context)
        self._runtime_candidates = candidates
        if candidates.empty:
            state["sector_view"] = {}
            state["target_positions"] = {}
            state["trace"].append({"tool": "universe_node", "detail": "empty candidates after pre_filter → END"})
            return state
        # Experience-learning context. K-line form snapshot + market sentiment
        # are derived from the candidate cross-section; recalled past
        # decisions whose form was similar (and whose outcome is known) become
        # the lesson the LLM sees in every downstream stage. Knowledge recall
        # happens later, once sectors are chosen.
        kline_features = self.top_down._market_kline_features(candidates)
        sentiment = self.sentiment_feed.snapshot(candidates, technical_context)
        # Ablation: ABLATE_SENTIMENT=1 zeroes the sentiment snapshot so the LLM
        # gets no market-mood signal, isolating its marginal contribution.
        if os.environ.get("ABLATE_SENTIMENT"):
            sentiment = {}
        market_view = (state.get("market_snapshot") or {}).get("market_view", "neutral")
        recalled: list = []
        lesson = ""
        # Ablation: ABLATE_EXPERIENCE=1 suppresses experience injection so the
        # LLM decides without any recalled historical outcomes.
        if not os.environ.get("ABLATE_EXPERIENCE"):
            try:
                recalled = self.memory.recall_similar(kline_features, market_view, limit=3)
                # Gate experience injection behind a minimum sample: with 1-2
                # back-filled records the "lesson" is noise (e.g. a single late
                # bullish entry that underperformed reads as "bullish is bad" and
                # spooks the LLM into neutral right at a melt-up top). Only inject
                # once there are enough learned outcomes for the pattern to be
                # meaningful. The graph still WRITES every decision to memory
                # regardless, so the sample accumulates.
                if len(recalled) >= 3:
                    lesson = self.memory.to_lesson(recalled)
            except Exception as exc:
                lesson = ""
                state.setdefault("trace", []).append({"tool": "memory_recall", "detail": f"failed: {exc}"})
        self.top_down.set_experience_context(
            experience_lesson=lesson,
            sentiment=sentiment,
        )
        state["kline_features"] = kline_features
        state["sentiment"] = sentiment
        state["memory_lesson"] = lesson
        state["trace"].append(
            {
                "tool": "universe_node",
                "detail": f"universe={len(universe)}, candidates={len(candidates)}, "
                f"breadth={sentiment.get('breadth_up')}, recalled={len(recalled)}",
            }
        )
        return state

    def _route_after_universe(self, state: TradingGraphState) -> str:
        candidates = self._runtime_candidates
        empty = candidates is None or len(candidates) == 0
        route = "end" if empty else "continue"
        state.setdefault("trace", []).append(
            {"tool": "universe_router", "detail": f"candidates={0 if candidates is None else len(candidates)} → {route}"}
        )
        return route

    def _sector_selector_node(self, state: TradingGraphState) -> TradingGraphState:
        candidates = self._runtime_candidates
        portfolio = state["portfolio"]
        market_snapshot = state.get("market_snapshot") or {}
        sector_view, selected_sectors = self.top_down._select_sectors(candidates, market_snapshot, portfolio)
        state["selected_sectors"] = selected_sectors
        state["sector_view"] = sector_view
        # Now that sectors are known, recall book/industry knowledge and add it
        # to the experience context before the stock selector runs. The recall
        # is bounded and degrades to empty context on any error.
        # Ablation: ABLATE_KNOWLEDGE=1 suppresses RAG injection.
        if os.environ.get("ABLATE_KNOWLEDGE"):
            knowledge = {"context": "", "sources": [], "document_count": 0}
        else:
            try:
                knowledge = self.knowledge_base.recall(
                    selected_sectors,
                    market_view=str(sector_view.get("market_view", "neutral")),
                    k=4,
                )
            except Exception as exc:
                knowledge = {"context": "", "sources": [], "document_count": 0}
                state.setdefault("trace", []).append({"tool": "knowledge_recall", "detail": f"failed: {exc}"})
        self.top_down.knowledge_context = knowledge
        state["trace"].append(
            {"tool": "sector_selector", "detail": f"selected={selected_sectors}, market_view={sector_view.get('market_view')}, kb_docs={knowledge.get('document_count')}"}
        )
        return state

    def _market_regime_node(self, state: TradingGraphState) -> TradingGraphState:
        regime = self.top_down.analyze_market_regime(
            state.get("market_snapshot") or {},
            state["portfolio"],
            self._runtime_candidates,
        )
        state["market_regime"] = regime
        market_snapshot = dict(state.get("market_snapshot") or {})
        if regime.get("market_view"):
            market_snapshot["market_view"] = regime["market_view"]
        state["market_snapshot"] = market_snapshot
        state["trace"].append(
            {"tool": "llm_market_regime", "detail": f"view={regime.get('market_view')}, runtime={regime.get('runtime')}"}
        )
        return state

    def _stock_selector_node(self, state: TradingGraphState) -> TradingGraphState:
        candidates = self._runtime_candidates
        sectors = state["selected_sectors"]
        portfolio = state["portfolio"]
        sector_view = state.get("sector_view") or {}
        selected = self.top_down._select_stocks(candidates, sectors, portfolio, sector_view)
        self._runtime_selected = selected
        sector_view["selected_stocks"] = self.top_down._stocks_to_view(selected)
        sector_view["aggregate_weight_reasoning"] = (
            selected["allocation_reasoning"].iloc[0]
            if not selected.empty and "allocation_reasoning" in selected.columns
            else ""
        )
        state["sector_view"] = sector_view
        state["trace"].append({"tool": "stock_selector", "detail": f"selected_stocks={len(selected)}"})
        return state

    def _candidate_analyst_node(self, state: TradingGraphState) -> TradingGraphState:
        analysis = self.top_down.analyze_candidates(
            self._runtime_selected,
            state["portfolio"],
            state.get("market_regime") or {},
        )
        state["candidate_analysis"] = analysis
        state["trace"].append({"tool": "llm_candidate_team", "detail": f"reports={len(analysis)}"})
        return state

    def _debate_ranker_node(self, state: TradingGraphState) -> TradingGraphState:
        debate = self.top_down.debate_and_rank(
            self._runtime_selected,
            state.get("candidate_analysis") or [],
            state.get("market_regime") or {},
        )
        state["debate_result"] = debate
        state["trace"].append(
            {"tool": "llm_cross_sectional_debate", "detail": f"ranked={debate.get('ranked_symbols', [])}"}
        )
        return state

    def _portfolio_planner_node(self, state: TradingGraphState) -> TradingGraphState:
        plan = self.top_down.plan_portfolio(
            self._runtime_selected,
            state.get("debate_result") or {},
            state["portfolio"],
            state.get("sector_view") or {},
        )
        state["portfolio_plan"] = plan
        self._runtime_selected = self.top_down.apply_portfolio_plan(self._runtime_selected, plan)
        sector_view = state.get("sector_view") or {}
        sector_view["market_regime"] = state.get("market_regime") or {}
        sector_view["candidate_analysis"] = state.get("candidate_analysis") or []
        sector_view["debate_result"] = state.get("debate_result") or {}
        sector_view["portfolio_plan"] = plan
        sector_view["selected_stocks"] = self.top_down._stocks_to_view(self._runtime_selected)
        state["sector_view"] = sector_view
        state["trace"].append(
            {"tool": "llm_portfolio_manager", "detail": f"weights={plan.get('weights', {})}"}
        )
        return state

    def _allocator_node(self, state: TradingGraphState) -> TradingGraphState:
        selected = self._runtime_selected
        portfolio = state["portfolio"]
        sector_view = state.get("sector_view") or {}
        sector_view["llm_trace"] = self.top_down._build_llm_trace()
        targets = self.top_down._allocate(portfolio, selected, sector_view)
        state["sector_view"] = sector_view
        state["target_positions"] = targets
        # Write the multi-symbol decision to memory so the next rebalance can
        # recall it by form similarity and learn from the realized outcome
        # (the engine back-fills active_return). All inputs degrade safely.
        try:
            market_snapshot = state.get("market_snapshot") or {}
            kline_features = dict(state.get("kline_features") or {})
            kline_features["selected"] = self.top_down._selected_kline_features(selected)
            self.memory.append_multi(
                day=int(state.get("day", 0)),
                date=str(market_snapshot.get("date", "")),
                market_view=str(sector_view.get("market_view", "neutral")),
                benchmark_return=float(market_snapshot.get("benchmark_return", 0.0)),
                selected_sectors=list(state.get("selected_sectors") or []),
                target_positions=targets,
                llm_reasoning=str(sector_view.get("aggregate_weight_reasoning", "")),
                llm_trace=sector_view.get("llm_trace") or {},
                kline_features=kline_features,
                sentiment=dict(state.get("sentiment") or {}),
            )
        except Exception as exc:
            state.setdefault("trace", []).append({"tool": "memory_append_multi", "detail": f"failed: {exc}"})
        state["trace"].append(
            {"tool": "allocator", "detail": f"targets={len(targets)}, cash_reserve_view={sector_view.get('cash_reserve_ratio')}"}
        )
        return state

    def _llm_refine_reports(self, reports: list, state: Dict[str, Any]) -> list:
        llm: LLMClient | None = self._llm()
        if not llm or not llm.enabled:
            return reports
        refined = []
        for report in reports:
            result = llm.complete_json(
                report.agent,
                "Refine this analyst report without inventing unsupported facts. "
                "Return JSON with optional stance, score, confidence, summary, evidence.\n"
                f"Report: {report.model_dump(mode='json')}\nRecent memory: {state.get('memory') or []}",
            )
            if result:
                payload = report.model_dump()
                stance = str(result.get("stance", report.stance.value)).lower()
                payload["stance"] = stance if stance in {"bullish", "bearish", "neutral"} else report.stance
                payload["score"] = clamp_float(result.get("score"), -1.0, 1.0, report.score)
                payload["confidence"] = clamp_float(result.get("confidence"), 0.0, 1.0, report.confidence)
                payload["summary"] = str(result.get("summary") or report.summary)
                if isinstance(result.get("evidence"), list):
                    payload["evidence"] = [str(item) for item in result["evidence"]][:6]
                refined.append(type(report)(**payload))
            else:
                refined.append(report)
        state["trace"].append({"tool": "llm_analyst_overlay", "detail": llm.last_error or "reports refined by LLM"})
        return refined

    def _llm_trade_overlay(self, intent: TradeIntent, state: Dict[str, Any]) -> TradeIntent:
        llm: LLMClient | None = self._llm()
        if not llm or not llm.enabled:
            return intent
        result = llm.complete_json(
            "trader_agent",
            "Review the deterministic trading intent. Return JSON with side, target_position, shares, confidence, rationale. "
            "side must be buy/sell/hold; keep shares within configured max_single_order_shares.\n"
            f"Intent: {intent.model_dump(mode='json')}\n"
            f"Debate: {state['debate'].model_dump(mode='json')}\n"
            f"Portfolio: {state['portfolio'].model_dump(mode='json')}\n"
            f"Config: {state['config'].model_dump()}\n"
            f"LangChain RAG context: {state.get('rag_context', {}).get('context', '')}\n"
            f"Recent memory: {state.get('memory') or []}",
            deep=True,
        )
        if not result:
            state["trace"].append({"tool": "llm_trader_overlay", "detail": llm.last_error or "not used"})
            return intent
        side_value = str(result.get("side", intent.side.value)).lower()
        side = OrderSide(side_value) if side_value in {"buy", "sell", "hold"} else intent.side
        max_shares = state["config"].max_single_order_shares
        try:
            shares = int(result.get("shares", intent.shares))
        except Exception:
            shares = intent.shares
        payload = intent.model_dump()
        payload["side"] = side
        payload["shares"] = max(0, min(max_shares, shares))
        payload["target_position"] = int(result.get("target_position", intent.target_position) or intent.target_position)
        payload["confidence"] = clamp_float(result.get("confidence"), 0.0, 1.0, intent.confidence)
        payload["rationale"] = str(result.get("rationale") or intent.rationale)
        state["trace"].append({"tool": "llm_trader_overlay", "detail": "trader intent reviewed by LLM"})
        return TradeIntent(**payload)

    def record_outcome(self, config: BacktestConfig, day: int, graph_result: GraphResult, fill: ExecutionFill) -> None:
        self.memory.append(
            {
                "symbol": config.symbol,
                "day": day,
                "side": fill.side.value,
                "shares": fill.shares,
                "price": fill.price,
                "equity": fill.equity,
                "realized_pnl": fill.realized_pnl,
                "signal": fill.signal,
                "debate_verdict": graph_result.debate.verdict.value,
                "risk_score": graph_result.risk.risk_score,
                "reason": fill.reason[:500],
            }
        )
