from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from agents.llm_client import LLMClient
from core.schemas import BacktestConfig, OrderSide, PortfolioSnapshot
from data.universe import AShareUniverseProvider, UniverseProviderError, load_a_share_universe, synthetic_a_share_universe
from tools.sizing import infer_lot_size, round_order_shares


def _infer_sector(symbol: str) -> str:
    """Infer a plausible A-share sector from the symbol code for synthetic universe fallback."""

    code = "".join(ch for ch in str(symbol) if ch.isdigit())[:6]
    known = {
        "000001": "银行",
        "000002": "房地产",
        "600000": "银行",
        "601012": "电力设备/光伏",
        "300750": "电力设备/电池",
    }
    if code in known:
        return known[code]
    if code.startswith(("600", "601", "602", "603", "605", "688")):
        return "主板/科技制造"
    if code.startswith("300"):
        return "创业板/成长"
    if code.startswith("000"):
        return "深圳主板"
    if code.startswith("002"):
        return "中小板"
    if code.startswith("8") or code.startswith("4"):
        return "新三板"
    return "综合"



class TopDownSelector:
    """Top-down A-share selector: sectors first, then stocks, then weights.

    The selector is designed to run without an LLM (rule-based fallback) so the
    multi-stock backtest works offline. When ``OPENAI_API_KEY`` is available and
    the config agent_mode is not ``offline``, it asks an LLM to review the
    pre-filtered candidate list and returns structured target positions.
    """

    def __init__(
        self,
        config: BacktestConfig,
        llm_client: LLMClient | None = None,
        universe_provider: AShareUniverseProvider | None = None,
    ) -> None:
        self.config = config
        self.llm = llm_client or LLMClient(
            mode=config.agent_mode,
            quick_model=config.llm_model,
            deep_model=config.deep_model,
            base_url=config.llm_base_url,
        )
        self.universe_provider = universe_provider
        self._injected_universe: pd.DataFrame | None = None
        # Experience-learning context injected by the graph before the LLM
        # stages run: a recalled lesson from similar past decisions, the
        # market/per-stock sentiment snapshot, and the knowledge-base context.
        # Empty by default; the offline rule path ignores them entirely.
        self.experience_lesson: str = ""
        self.sentiment_context: Dict[str, Any] = {}
        self.knowledge_context: Dict[str, Any] = {}

    def set_universe_frame(self, frame: pd.DataFrame | None) -> None:
        """Inject the engine's dynamically resolved A-share universe."""
        self._injected_universe = frame.copy() if frame is not None else None

    def set_experience_context(
        self,
        *,
        experience_lesson: str = "",
        sentiment: Dict[str, Any] | None = None,
        knowledge: Dict[str, Any] | None = None,
    ) -> None:
        """Inject recalled experience + sentiment + knowledge before LLM stages.

        Called by the multi-symbol graph's universe node so every downstream
        LLM stage (regime, sector, stock, candidate, portfolio) sees the same
        experiential context. The rule-based fallback path ignores these.
        """
        self.experience_lesson = experience_lesson or ""
        self.sentiment_context = sentiment or {}
        self.knowledge_context = knowledge or {}

    def _experience_block(self) -> str:
        """Render the injected experience/sentiment/knowledge as a prompt block."""
        parts: List[str] = []
        if self.experience_lesson:
            parts.append("Historical experience from similar past market forms (mimic winners, avoid losers):\n" + self.experience_lesson)
        if self.knowledge_context.get("context"):
            parts.append("Knowledge base (book notes + industry character):\n" + str(self.knowledge_context["context"]))
        if self.sentiment_context:
            score = self.sentiment_context.get("market_score", 0.0)
            events = self.sentiment_context.get("top_events", [])
            breadth = self.sentiment_context.get("breadth_up")
            line = f"Market sentiment score={score:+.2f}"
            if breadth is not None:
                line += f", breadth_up={breadth:.0%}"
            if events:
                line += f", events={events}"
            parts.append(line)
        return "\n\n".join(parts)

    @staticmethod
    def _market_kline_features(candidates: pd.DataFrame) -> Dict[str, Any]:
        """Market-layer form snapshot for similarity recall.

        Vector = [avg_rsi, avg_volatility, breadth_up, avg_dist_sma20]. This is
        the anchor that ``TradingMemory.recall_similar`` matches against past
        decisions, so a similar cross-section surface similar past outcomes.
        """
        if candidates is None or candidates.empty:
            return {"market": {}}
        rsi = pd.to_numeric(candidates.get("rsi_14"), errors="coerce").dropna()
        vol = pd.to_numeric(candidates.get("volatility_20"), errors="coerce").dropna()
        dist20 = pd.to_numeric(candidates.get("dist_sma_20"), errors="coerce").dropna()
        changes = pd.to_numeric(candidates["change_pct"], errors="coerce").fillna(0.0)
        return {
            "market": {
                "avg_rsi": float(rsi.mean()) if len(rsi) else 50.0,
                "avg_volatility": float(vol.mean()) if len(vol) else 0.2,
                "breadth_up": float((changes > 0).mean()),
                "avg_dist_sma20": float(dist20.mean()) if len(dist20) else 0.0,
            }
        }

    @staticmethod
    def _selected_kline_features(selected: pd.DataFrame) -> Dict[str, Any]:
        """Per-stock feature snapshot for the chosen holdings (for later review)."""
        out: Dict[str, Any] = {}
        if selected is None or selected.empty:
            return out
        for _, row in selected.iterrows():
            sym = str(row["symbol"])
            out[sym] = {
                "rsi": float(pd.to_numeric(row.get("rsi_14"), errors="coerce") or 50.0),
                "volatility_20": float(pd.to_numeric(row.get("volatility_20"), errors="coerce") or 0.2),
                "dist_sma20": float(pd.to_numeric(row.get("dist_sma_20"), errors="coerce") or 0.0),
                "factor_score": float(pd.to_numeric(row.get("factor_score"), errors="coerce") or 0.0),
            }
        return out

    def run(
        self,
        portfolio: PortfolioSnapshot,
        market_snapshot: Dict[str, Any] | None = None,
        prices: Dict[str, float] | None = None,
    ) -> Dict[str, int]:
        """Return target positions ``{symbol: shares}`` for the next rebalance."""
        _, targets = self.run_with_view(portfolio, market_snapshot, prices)
        return targets

    def run_with_view(
        self,
        portfolio: PortfolioSnapshot,
        market_snapshot: Dict[str, Any] | None = None,
        prices: Dict[str, float] | None = None,
        technical_context: Dict[str, Dict[str, float]] | None = None,
    ) -> Tuple[Dict[str, Any], Dict[str, int]]:
        """Return both the sector view and target positions for the next rebalance."""
        universe = self._load_universe()
        if prices:
            price_series = pd.Series(prices)
            universe = universe.copy()
            universe["close"] = universe["symbol"].map(price_series).fillna(universe["close"])
            # Use 1-day return from the engine as the momentum field when available
            if technical_context:
                returns = {s: ctx.get("return_1d", 0.0) for s, ctx in technical_context.items()}
                return_series = pd.Series(returns)
                universe["change_pct"] = universe["symbol"].map(return_series).fillna(universe["change_pct"])
        if universe.empty:
            return {}, {}

        candidates = self._pre_filter(universe, technical_context)
        if candidates.empty:
            return {}, {}

        sector_view, selected_sectors = self._select_sectors(candidates, market_snapshot, portfolio)
        selected_stocks = self._select_stocks(candidates, selected_sectors, portfolio, sector_view)
        sector_view["selected_stocks"] = self._stocks_to_view(selected_stocks)
        sector_view["aggregate_weight_reasoning"] = (
            selected_stocks["allocation_reasoning"].iloc[0]
            if not selected_stocks.empty and "allocation_reasoning" in selected_stocks.columns
            else ""
        )
        sector_view["llm_trace"] = self._build_llm_trace()
        targets = self._allocate(portfolio, selected_stocks, sector_view)
        return sector_view, targets

    def _load_universe(self) -> pd.DataFrame:
        if self._injected_universe is not None:
            return self._injected_universe.copy()
        # Offline providers never need network universe data; build a synthetic
        # universe that mirrors the configured symbol list so prices line up.
        if self.config.provider in {"synthetic", "csv"}:
            return self._synthetic_universe_from_config()

        universe = pd.DataFrame(columns=["symbol", "name", "close", "change_pct", "volume", "sector"])
        try:
            if self.universe_provider is not None:
                universe = self.universe_provider.list_spot()
            else:
                universe = load_a_share_universe(allow_fallback=True)
        except UniverseProviderError:
            universe = load_a_share_universe(allow_fallback=True)

        configured = set(self.config.universe or [])
        loaded = set(universe["symbol"].astype(str).tolist()) if not universe.empty else set()
        # If the real-time/full-market universe is missing the configured symbols
        # (e.g. network fallback), build a symbol-aligned synthetic universe so the
        # selector can actually pick from the stocks we have price data for.
        if configured and not configured.issubset(loaded):
            return self._synthetic_universe_from_config()
        return universe

    def _synthetic_universe_from_config(self) -> pd.DataFrame:
        symbols = self.config.universe or [self.config.symbol]
        rows: List[Dict[str, object]] = []
        for symbol in symbols:
            rows.append(
                {
                    "symbol": symbol,
                    "name": f"合成{symbol}",
                    "close": 10.0,
                    "change_pct": 0.0,
                    "volume": 100000,
                    "turnover": 1000000.0,
                    "sector": _infer_sector(symbol),
                }
            )
        return pd.DataFrame(rows)

    def _pre_filter(
        self,
        universe: pd.DataFrame,
        technical_context: Dict[str, Dict[str, float]] | None = None,
    ) -> pd.DataFrame:
        required = {"symbol", "close", "change_pct", "volume", "sector"}
        if not required.issubset(universe.columns):
            return pd.DataFrame(columns=list(required))

        frame = universe.copy()
        # Drop rows with missing critical fields
        frame = frame.dropna(subset=["symbol", "close", "volume", "sector"])
        frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
        frame["change_pct"] = pd.to_numeric(frame["change_pct"], errors="coerce").fillna(0.0)
        frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce").fillna(0)
        frame = frame[frame["close"] > 0]
        # Lightweight liquidity filter: keep stocks with positive volume
        frame = frame[frame["volume"] > 0]

        # Merge optional technical indicators from the engine
        tech = technical_context or {}
        tech_cols = ["rsi_14", "volatility_20", "sma_5", "sma_20", "sma_60", "atr_14", "volume_z",
                     "return_5d", "return_20d"]
        for col in tech_cols:
            frame[col] = frame["symbol"].map(lambda s: tech.get(s, {}).get(col)).astype(float)

        # Distance from moving averages
        for ma in ["sma_5", "sma_20", "sma_60"]:
            if ma in frame.columns:
                frame[f"dist_{ma}"] = ((frame["close"] - frame[ma]) / frame["close"]).replace([np.inf, -np.inf], 0.0).fillna(0.0)

        # Multi-factor score for rule-based fallback
        frame["momentum_score"] = self._zscore(frame["change_pct"]).fillna(0.0)
        frame["liquidity_score"] = self._zscore(np.log1p(frame["volume"].astype(float))).fillna(0.0)
        vol = frame["volatility_20"].fillna(frame["volatility_20"].median())
        frame["volatility_score"] = -self._zscore(vol).fillna(0.0)
        trend = (frame.get("dist_sma_20", pd.Series(0.0, index=frame.index)).fillna(0.0)
                 + frame.get("dist_sma_60", pd.Series(0.0, index=frame.index)).fillna(0.0))
        frame["trend_score"] = self._zscore(trend).fillna(0.0)

        # Relative strength = momentum + trend, the liquidity-free backbone of
        # both sector and stock selection. Liquidity is still scored (the LLM
        # may consult it) but no longer drives ranking — weighting it was
        # biasing picks toward high-turnover names that crashed in this regime
        # while the index rallied.
        frame["relative_strength"] = (frame["momentum_score"] + frame["trend_score"]).fillna(0.0)

        # Rank within sector by relative strength so groupby(sector).head(N)
        # later picks the regime leaders, not just single-day movers.
        frame["sector_momentum_rank"] = frame.groupby("sector")["relative_strength"].rank(ascending=False, method="first")

        # Composite factor score for LLM context / tie-breaking. Liquidity is
        # intentionally zero-weighted in the decision.
        frame["factor_score"] = (
            0.45 * frame["momentum_score"]
            + 0.35 * frame["trend_score"]
            + 0.20 * frame["volatility_score"]
        )

        return frame.sort_values(["sector", "sector_momentum_rank"]).reset_index(drop=True)

    @staticmethod
    def _zscore(series: pd.Series) -> pd.Series:
        mean = series.mean()
        std = series.std()
        if std is None or math.isclose(std, 0.0):
            return pd.Series(0.0, index=series.index)
        return (series - mean) / std

    @staticmethod
    def _clean_float(value: Any) -> Any:
        """Coerce NaN/inf floats to None so they survive json.dumps / checkpoint round-trips."""
        if isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                return None
        elif isinstance(value, (np.floating, np.integer)):
            f = float(value)
            if math.isnan(f) or math.isinf(f):
                return None
            return f if isinstance(value, np.floating) else int(value)
        return value

    @classmethod
    def _sanitize_records(cls, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Launder NaN/inf out of pandas to_dict("records") output for JSON safety."""
        cleaned: List[Dict[str, Any]] = []
        for record in records:
            cleaned.append({key: cls._clean_float(val) for key, val in record.items()})
        return cleaned

    def _complete_stage(self, role: str, prompt: str, *, deep: bool = False) -> Dict[str, Any] | None:
        """Invoke one LLM decision stage.

        In ``llm`` mode the LLM is preferred, but a transient timeout/error
        degrades gracefully to the caller's rule-based fallback instead of
        crashing the whole backtest — one failed LLM call mid-run must not
        sink 10 rebalances. The failure is recorded so the trace surfaces it.
        """
        result = self.llm.complete_json(role, prompt, deep=deep)
        if result is None:
            # Record the failure for visibility, but do NOT raise: callers
            # (_llm_sector_view / _llm_stock_view / etc.) already branch into
            # rule-based fallback when this returns None.
            self._llm_stage_errors = getattr(self, "_llm_stage_errors", [])
            self._llm_stage_errors.append(
                f"{role}: {self.llm.last_error or 'no response'}"
            )
        return result

    def analyze_market_regime(
        self,
        market_snapshot: Dict[str, Any],
        portfolio: PortfolioSnapshot,
        candidates: pd.DataFrame | None,
    ) -> Dict[str, Any]:
        """Use an LLM macro agent to review the deterministic market regime."""
        rule_view = str(market_snapshot.get("market_view", "neutral"))
        if not self.llm.enabled:
            return {
                "market_view": rule_view,
                "confidence": 0.5,
                "reasoning": "显式离线模式：使用基准累计收益判断市场状态。",
                "risk_budget_scale": 1.0,
                "runtime": "offline-rule",
            }
        summary = {}
        if candidates is not None and not candidates.empty:
            summary = {
                "candidate_count": int(len(candidates)),
                "positive_factor_ratio": float((candidates["factor_score"] > 0).mean()),
                "median_change": float(candidates["change_pct"].median()),
                "median_volatility": self._clean_float(float(candidates["volatility_20"].median())) or 0.0,
            }
        result = self._complete_stage(
            "multi_market_regime_agent",
            (
                "Assess the market regime for a multi-stock paper portfolio using only the supplied data. "
                "Return JSON with market_view (bullish|neutral|bearish), confidence (0..1), "
                "reasoning, and risk_budget_scale (0.25..1.25).\n"
                f"Market snapshot: {market_snapshot}\nBreadth summary: {summary}\n"
                f"Portfolio: cash={portfolio.cash}, equity={portfolio.equity}, drawdown={portfolio.max_drawdown}\n"
                f"{self._experience_block()}"
            ),
            deep=True,
        ) or {}
        view = str(result.get("market_view", rule_view)).lower()
        if view not in {"bullish", "neutral", "bearish"}:
            view = rule_view
        return {
            "market_view": view,
            "confidence": max(0.0, min(1.0, float(result.get("confidence", 0.5)))),
            "reasoning": str(result.get("reasoning", "")),
            "risk_budget_scale": max(0.25, min(1.25, float(result.get("risk_budget_scale", 1.0)))),
            "runtime": self.llm.last_runtime,
        }

    def analyze_candidates(
        self,
        selected: pd.DataFrame | None,
        portfolio: PortfolioSnapshot,
        market_regime: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Run a structured LLM analyst team over the shortlisted stocks."""
        if selected is None or selected.empty:
            return []
        records = self._sanitize_records(
            selected[
                [
                    "symbol", "name", "sector", "close", "change_pct", "volume",
                    "rsi_14", "volatility_20", "dist_sma_20", "dist_sma_60", "factor_score",
                ]
            ].to_dict("records")
        )
        if not self.llm.enabled:
            per_stock = self.sentiment_context.get("per_stock") or {}
            return [
                {
                    "symbol": str(row["symbol"]),
                    "technical_score": float(row.get("factor_score") or 0.0),
                    "sentiment_score": float(per_stock.get(str(row["symbol"]), 0.0)),
                    "fundamental_score": 0.0,
                    "overall_score": float(row.get("factor_score") or 0.0),
                    "bull_case": "离线因子得分为正时支持配置。",
                    "bear_case": "离线模式没有新闻和基本面语义判断。",
                    "data_gaps": ["news", "fundamentals"],
                }
                for row in records
            ]
        result = self._complete_stage(
            "multi_candidate_analyst_team",
            (
                "Act as a technical, sentiment, news, and fundamentals analyst team. "
                "Score every supplied candidate from -1 to 1. Do not invent unavailable news or fundamentals; "
                "list missing evidence in data_gaps and use 0 for unsupported component scores. "
                "Return JSON: {\"analyses\":[{\"symbol\":\"...\",\"technical_score\":0.0,"
                "\"sentiment_score\":0.0,\"fundamental_score\":0.0,\"overall_score\":0.0,"
                "\"bull_case\":\"...\",\"bear_case\":\"...\",\"data_gaps\":[]}]}\n"
                f"Market regime: {market_regime}\nPortfolio: {portfolio.model_dump(mode='json')}\n"
                f"Candidates: {records}\n{self._experience_block()}"
            ),
            deep=False,
        ) or {}
        analyses = result.get("analyses")
        return analyses if isinstance(analyses, list) else []

    def debate_and_rank(
        self,
        selected: pd.DataFrame | None,
        analyses: List[Dict[str, Any]],
        market_regime: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Use a cross-sectional bull/bear debate to rank the candidates."""
        if selected is None or selected.empty:
            return {"ranked_symbols": [], "rounds": []}
        symbols = selected["symbol"].astype(str).tolist()
        if not self.llm.enabled:
            ranked = selected.sort_values("factor_score", ascending=False)["symbol"].astype(str).tolist()
            return {
                "ranked_symbols": ranked,
                "conviction": 0.5,
                "manager_notes": "显式离线模式：按综合因子得分排序。",
                "rounds": [],
            }
        result = self._complete_stage(
            "multi_bull_bear_research_manager",
            (
                "Conduct a concise bull-versus-bear cross-sectional debate over the candidate stocks, "
                "then rank them for a multi-stock paper portfolio. Use only supplied evidence. "
                "Return JSON with ranked_symbols, conviction (0..1), manager_notes, and "
                "rounds [{symbol,bull_case,bear_case,verdict,score}].\n"
                f"Allowed symbols: {symbols}\nMarket regime: {market_regime}\nAnalyst reports: {analyses}"
            ),
            deep=True,
        ) or {}
        ranked = [str(s) for s in result.get("ranked_symbols", []) if str(s) in symbols]
        ranked.extend(symbol for symbol in symbols if symbol not in ranked)
        result["ranked_symbols"] = ranked
        return result

    def plan_portfolio(
        self,
        selected: pd.DataFrame | None,
        debate: Dict[str, Any],
        portfolio: PortfolioSnapshot,
        sector_view: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Ask the LLM portfolio manager for normalized advisory weights."""
        if selected is None or selected.empty:
            return {"weights": {}, "cash_ratio": 1.0, "reasoning": "没有候选股票。"}
        symbols = selected["symbol"].astype(str).tolist()
        if not self.llm.enabled:
            weights = dict(zip(symbols, selected["target_weight"].astype(float).tolist()))
            return {"weights": weights, "cash_ratio": self.config.cash_reserve_ratio, "reasoning": "离线 Softmax 权重。"}
        result = self._complete_stage(
            "multi_portfolio_manager",
            (
                "Construct advisory weights for the ranked candidates. Return JSON with weights as a "
                "symbol-to-number object, cash_ratio, and reasoning. Use only allowed symbols; non-cash "
                "weights must be non-negative and sum to at most 1. Hard limits are enforced later.\n"
                f"Allowed symbols: {symbols}\nDebate: {debate}\nSector view: {sector_view}\n"
                f"Portfolio: {portfolio.model_dump(mode='json')}\n"
                f"Constraints: max_positions={self.config.max_positions}, "
                f"max_position_ratio={self.config.max_position_ratio}, "
                f"cash_reserve_ratio={self.config.cash_reserve_ratio}\n"
                f"{self._experience_block()}"
            ),
            deep=True,
        ) or {}
        raw = result.get("weights") if isinstance(result.get("weights"), dict) else {}
        weights = {s: max(0.0, float(raw.get(s, 0.0))) for s in symbols}
        result["weights"] = weights
        return result

    def apply_portfolio_plan(self, selected: pd.DataFrame | None, plan: Dict[str, Any]) -> pd.DataFrame:
        """Apply LLM advisory weights while preserving deterministic allocation limits."""
        if selected is None or selected.empty:
            return pd.DataFrame() if selected is None else selected
        frame = selected.copy()
        weights = plan.get("weights") if isinstance(plan.get("weights"), dict) else {}
        if weights and sum(max(0.0, float(v)) for v in weights.values()) > 0:
            frame["target_weight"] = frame["symbol"].astype(str).map(
                lambda symbol: max(0.0, float(weights.get(symbol, 0.0)))
            )
            frame = frame[frame["target_weight"] > 0].sort_values("target_weight", ascending=False).reset_index(drop=True)
            frame["allocation_reasoning"] = str(plan.get("reasoning", "LLM portfolio manager"))
        return frame

    def _select_sectors(
        self,
        candidates: pd.DataFrame,
        market_snapshot: Dict[str, Any] | None,
        portfolio: PortfolioSnapshot,
    ) -> Tuple[Dict[str, Any], List[str]]:
        benchmark_return = float((market_snapshot or {}).get("benchmark_return", 0.0))
        sector_perf = (
            candidates.groupby("sector")
            .agg(
                avg_change=("change_pct", "mean"),
                median_change=("change_pct", "median"),
                avg_factor_score=("factor_score", "mean"),
                avg_relative_strength=("relative_strength", "mean"),
                avg_volatility=("volatility_20", "mean"),
                avg_volume=("volume", "mean"),
                avg_return_5d=("return_5d", "mean"),
                avg_return_20d=("return_20d", "mean"),
                count=("symbol", "count"),
            )
            .reset_index()
        )
        # Sector relative strength = sector avg return minus the benchmark's
        # return on the same day. Outperforming sectors rank first; when every
        # sector lags the benchmark we still pick the relatively-strongest
        # top_n rather than going to cash (keeps the strategy comparable across
        # regimes). Replaces the old avg_factor_score sort, which was diluted
        # by the liquidity term and tilted toward high-turnover losers.
        sector_perf["relative_strength"] = sector_perf["avg_change"] - benchmark_return
        # Earnings-growth proxy: longer-horizon price momentum (20d return)
        # is the market pricing in forward earnings growth ahead of the print.
        # Real fundamentals (akshare 财务摘要) are often blocked from this
        # network and lag quarterly, so the price-based proxy is the runnable
        # substitute — it adds a growth dimension to what was pure 1-day
        # momentum, without depending on an endpoint that may be unreachable.
        sector_perf["earnings_growth_proxy"] = sector_perf["avg_return_20d"].fillna(0.0)
        # Composite: short-horizon relative strength (regime leaders) plus a
        # weight on the growth proxy. Z-scored within the cross-section so the
        # two scales (1d ~0.02, 20d ~0.10) contribute on equal footing.
        rs_z = self._zscore(sector_perf["relative_strength"]).fillna(0.0)
        growth_z = self._zscore(sector_perf["earnings_growth_proxy"]).fillna(0.0)
        # Ablation: ABLATE_GROWTH=1 zeroes the growth-proxy weight, isolating
        # the marginal contribution of the earnings-growth dimension.
        growth_weight = 0.5 if not os.environ.get("ABLATE_GROWTH") else 0.0
        sector_perf["sector_score"] = rs_z + growth_weight * growth_z
        sector_perf = sector_perf.sort_values("sector_score", ascending=False)
        top_n = max(1, self.config.top_n_sectors)
        selected = sector_perf.head(top_n)["sector"].tolist()
        market_view = (market_snapshot or {}).get("market_view", "neutral")
        view = {
            "market_view": market_view,
            "benchmark_return": round(benchmark_return, 6),
            "sector_ranking": self._sanitize_records(sector_perf.to_dict("records")),
            "selected_sectors": selected,
            "cash_reserve_ratio": self.config.cash_reserve_ratio,
            "max_position_ratio": self.config.max_position_ratio,
        }
        # Try LLM sector refinement when online
        llm_view = self._llm_sector_view(candidates, sector_perf, market_snapshot, portfolio)
        if llm_view and llm_view.get("selected_sectors"):
            selected = llm_view["selected_sectors"][:top_n]
            view.update(llm_view)
        return view, selected

    def _select_stocks(
        self,
        candidates: pd.DataFrame,
        sectors: List[str],
        portfolio: PortfolioSnapshot,
        sector_view: Dict[str, Any],
    ) -> pd.DataFrame:
        mask = candidates["sector"].isin(sectors)
        selected = candidates[mask].copy()
        stocks_per_sector = max(1, self.config.stocks_per_sector)
        selected = selected.groupby("sector").head(stocks_per_sector).reset_index(drop=True)

        # In non-bullish markets, avoid catching falling knives: keep only
        # stocks with positive relative strength (leading the market via
        # momentum + trend) if any exist.
        market_view = sector_view.get("market_view", "neutral")
        if market_view != "bullish":
            positive = selected[selected["relative_strength"] > 0]
            if not positive.empty:
                selected = positive.reset_index(drop=True)

        llm_stocks, allocation_reasoning = self._llm_stock_view(selected, portfolio)
        if llm_stocks:
            symbols = [s["symbol"] for s in llm_stocks if s.get("symbol")]
            if symbols:
                selected = selected[selected["symbol"].isin(symbols)].copy()
                # Preserve LLM ordering by merging weights back
                weights = {s["symbol"]: s.get("weight", 0.0) for s in llm_stocks}
                rationales = {s["symbol"]: s.get("rationale", "") for s in llm_stocks}
                selected["target_weight"] = selected["symbol"].map(weights).fillna(0.0)
                selected["llm_rationale"] = selected["symbol"].map(rationales).fillna("")
                selected = selected.sort_values("target_weight", ascending=False)
                selected["allocation_reasoning"] = allocation_reasoning or "由 LLM 分配权重"
                return selected

        # Rule-based fallback: relative strength (momentum + trend) → softmax
        # weights. Liquidity is no longer in the weighting — it only gates the
        # pre-filter, so leaders of the strongest sectors get the weight.
        scores = selected["relative_strength"].astype(float).fillna(0.0).values
        selected["target_weight"] = self._softmax_weights(scores)
        selected["llm_rationale"] = ""
        selected["allocation_reasoning"] = (
            "离线规则：板块按相对基准强度排序，个股按相对强度（动量+趋势）分配权重。"
        )
        return selected

    @staticmethod
    def _softmax_weights(scores: np.ndarray) -> np.ndarray:
        """Convert factor scores to positive weights summing to 1."""
        scores = np.asarray(scores, dtype=float)
        if len(scores) == 0:
            return scores
        # Shift so max is 0 for numerical stability
        max_score = np.max(scores)
        exp_scores = np.exp(scores - max_score)
        total = np.sum(exp_scores)
        if total <= 0 or not np.isfinite(total):
            return np.ones_like(scores) / len(scores)
        return exp_scores / total

    def _allocate(
        self,
        portfolio: PortfolioSnapshot,
        selected: pd.DataFrame,
        sector_view: Dict[str, Any],
    ) -> Dict[str, int]:
        if selected.empty:
            return {}

        max_positions = max(1, self.config.max_positions)
        selected = selected.head(max_positions)

        total_weight = selected["target_weight"].sum()
        if total_weight <= 0:
            selected["target_weight"] = 1.0 / len(selected)
            total_weight = selected["target_weight"].sum()

        # Trend-aware cash reserve: raise cash in bearish/neutral down-markets to protect
        # relative performance, deploy more in bullish markets.
        market_view = sector_view.get("market_view", "neutral")
        extra_reserve = 0.0
        if market_view == "bearish":
            extra_reserve = 0.60
        elif market_view == "neutral":
            extra_reserve = 0.35
        effective_reserve = min(0.75, self.config.cash_reserve_ratio + extra_reserve)

        cash_reserve = portfolio.equity * effective_reserve
        investable = max(0.0, portfolio.cash - cash_reserve)
        # Per-position cap is a fraction of total equity.
        max_position_value = portfolio.equity * self.config.max_position_ratio

        targets: Dict[str, int] = {}
        lot = infer_lot_size(self.config)
        for _, row in selected.iterrows():
            symbol = str(row["symbol"])
            price = float(row["close"])
            if price <= 0:
                continue
            weight = float(row["target_weight"]) / total_weight
            target_value = min(investable * weight, max_position_value)
            target_shares = int(target_value // price)
            target_shares = round_order_shares(target_shares, self.config, OrderSide.buy)
            if target_shares > 0:
                targets[symbol] = target_shares
        return targets

    def _llm_sector_view(
        self,
        candidates: pd.DataFrame,
        sector_perf: pd.DataFrame,
        market_snapshot: Dict[str, Any] | None,
        portfolio: PortfolioSnapshot,
    ) -> Dict[str, Any] | None:
        if not self.llm.enabled:
            return None
        top_sectors = sector_perf.head(10).to_dict("records")
        prompt = self._sector_prompt(top_sectors, market_snapshot, portfolio, self.config)
        result = self._complete_stage("multi_sector_selector", prompt, deep=True)
        if not isinstance(result, dict):
            return None
        return {
            "market_view": result.get("market_view", "neutral"),
            "selected_sectors": result.get("selected_sectors", []),
            "reasoning": result.get("reasoning", ""),
            "confidence": result.get("confidence", 0.0),
            "excluded_sectors": result.get("excluded_sectors", []),
        }

    def _llm_stock_view(
        self,
        selected: pd.DataFrame,
        portfolio: PortfolioSnapshot,
    ) -> tuple[List[Dict[str, Any]] | None, str | None]:
        if not self.llm.enabled or selected.empty:
            return None, None
        prompt = self._stock_prompt(selected, portfolio, self.config)
        result = self._complete_stage("multi_stock_selector", prompt, deep=True)
        if not isinstance(result, dict):
            return None, None
        return result.get("selected_stocks"), result.get("allocation_reasoning")

    def _sector_prompt(
        self,
        sectors: List[Dict[str, Any]],
        market_snapshot: Dict[str, Any] | None,
        portfolio: PortfolioSnapshot,
        config: BacktestConfig,
    ) -> str:
        snapshot = market_snapshot or {}
        lines = [
            "You are a macro/sector strategist for an A-share paper-trading portfolio.",
            "Select the most attractive sectors for the next rebalance based on the data below.",
            "",
            "Portfolio constraints:",
            f"- cash: {portfolio.cash:.2f}",
            f"- equity: {portfolio.equity:.2f}",
            f"- cash_reserve_ratio: {config.cash_reserve_ratio:.2%}",
            f"- max_position_ratio: {config.max_position_ratio:.2%}",
            f"- top_n_sectors target: {config.top_n_sectors}",
            f"- stocks_per_sector target: {config.stocks_per_sector}",
            "",
            "Market snapshot:",
            f"- day: {snapshot.get('day', 'N/A')}",
            f"- date: {snapshot.get('date', 'N/A')}",
            f"- benchmark_symbol: {snapshot.get('benchmark_symbol', 'N/A')}",
            f"- benchmark_return: {snapshot.get('benchmark_return', 0.0):.2%}",
            f"- market_view (rule-based): {snapshot.get('market_view', 'neutral')}",
            f"- trend_score: {snapshot.get('trend_score', 0.0)}",
            f"- trend_signals: {snapshot.get('trend_signals', {})}",
            "",
            "Sector performance (top candidates by sector_score):",
        ]
        for s in sectors:
            lines.append(
                f"- {s['sector']}: avg_change={s['avg_change']:.2f}%, "
                f"median={s['median_change']:.2f}%, "
                f"factor_score={s.get('avg_factor_score', 0.0):.2f}, "
                f"return_5d={s.get('avg_return_5d', 0.0) or 0.0:.2%}, "
                f"return_20d(盈利预期代理)={s.get('avg_return_20d', 0.0) or 0.0:.2%}, "
                f"volatility={s.get('avg_volatility', 0.0):.4f}, "
                f"count={s['count']}"
            )
        lines.extend([
            "",
            "Return strict JSON only:",
            '{"market_view": "bullish|neutral|bearish", "selected_sectors": ["sector1", "sector2"], '
            '"reasoning": "...", "confidence": 0.75, "excluded_sectors": ["sector3"]}',
        ])
        block = self._experience_block()
        if block:
            lines += ["", "Experience / knowledge / sentiment context:", block]
        return "\n".join(lines)

    def _stock_prompt(self, selected: pd.DataFrame, portfolio: PortfolioSnapshot, config: BacktestConfig) -> str:
        lines = [
            "You are a stock picker for an A-share paper-trading portfolio.",
            "From the candidate stocks below, select the best ones and assign target weights.",
            "Weights must sum to <= 1.0. Cash ratio is implicit (1 - sum(weights)).",
            f"Put most capital to work: target total weight close to {1 - config.cash_reserve_ratio:.0%}, "
            f"so only the required cash reserve ({config.cash_reserve_ratio:.0%}) stays idle.",
            f"No single stock weight should exceed the per-position cap of {config.max_position_ratio:.0%}.",
            "Prefer stocks with positive momentum, strong relative strength, and lower volatility.",
            "",
            "Portfolio constraints:",
            f"- cash: {portfolio.cash:.2f}",
            f"- equity: {portfolio.equity:.2f}",
            f"- cash_reserve_ratio: {config.cash_reserve_ratio:.2%}",
            f"- max_position_ratio per stock: {config.max_position_ratio:.2%}",
            f"- max_positions target: {config.max_positions}",
            "",
            "Current holdings:",
        ]
        if portfolio.positions:
            for sym, shares in portfolio.positions.items():
                avg = portfolio.avg_entries.get(sym, 0.0)
                lines.append(f"- {sym}: {shares} shares @ avg_cost={avg:.2f}")
        else:
            lines.append("- none")
        lines.extend([
            "",
            "Candidate stocks:",
        ])
        for _, row in selected.iterrows():
            sector_rank = int(row.get("sector_momentum_rank", 0))
            rsi = row.get("rsi_14")
            vol = row.get("volatility_20")
            dist_20 = row.get("dist_sma_20")
            dist_60 = row.get("dist_sma_60")
            factor = row.get("factor_score", 0.0)
            extras = []
            if rsi is not None and not math.isnan(rsi):
                extras.append(f"rsi={rsi:.1f}")
            if vol is not None and not math.isnan(vol):
                extras.append(f"vol={vol:.4f}")
            if dist_20 is not None and not math.isnan(dist_20):
                extras.append(f"dist_sma20={dist_20:.2%}")
            if dist_60 is not None and not math.isnan(dist_60):
                extras.append(f"dist_sma60={dist_60:.2%}")
            extras.append(f"factor_score={factor:.2f}")
            lines.append(
                f"- {row['symbol']} {row['name']} | sector={row['sector']} "
                f"close={row['close']:.2f} change_pct={row['change_pct']:.2f}% "
                f"volume={row['volume']} sector_rank={sector_rank} "
                + " ".join(extras)
            )
        lines.extend([
            "",
            "Return strict JSON only:",
            '{"selected_stocks": [{"symbol": "000001", "weight": 0.20, "rationale": "..."}], '
            '"cash_ratio": 0.30, "allocation_reasoning": "..."}',
        ])
        block = self._experience_block()
        if block:
            lines += ["", "Experience / knowledge / sentiment context:", block]
        return "\n".join(lines)

    def _stocks_to_view(self, selected: pd.DataFrame) -> List[Dict[str, Any]]:
        if selected.empty:
            return []
        rows: List[Dict[str, Any]] = []
        for _, row in selected.iterrows():
            rows.append(
                {
                    "symbol": str(row["symbol"]),
                    "name": str(row.get("name", "")),
                    "sector": str(row.get("sector", "")),
                    "weight": self._clean_float(round(float(row.get("target_weight", 0.0)), 6)) or 0.0,
                    "rationale": str(row.get("llm_rationale", "")),
                    "factor_score": self._clean_float(round(float(row.get("factor_score", 0.0)), 4)) or 0.0,
                }
            )
        return rows

    def _build_llm_trace(self) -> Dict[str, Any]:
        enabled = self.llm.enabled
        return {
            "enabled": enabled,
            # Report the model actually resolved by the LLMClient (honours
            # OPENAI_MODEL/OPENAI_DEEP_MODEL from .env), not the raw config
            # default which may still be the gpt-4o placeholder.
            "model_used": self.llm.deep_model if enabled else self.config.deep_model,
            "runtime": self.llm.last_runtime,
            "error": self.llm.last_error,
            "stage": "sector+stock",
        }
