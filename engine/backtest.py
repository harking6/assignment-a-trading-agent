from __future__ import annotations

import math
from statistics import mean, pstdev
from typing import Any, Dict, List

from agents.graph import TradingAgentsGraph
from core.schemas import BacktestConfig, ExecutionFill, MarketBar, OrderSide
from core.schemas import PortfolioDecision
from data.providers import MarketDataProvider, SyntheticDataProvider, bars_to_frame, build_provider
from data.universe import AShareUniverseProvider, UniverseProviderError
from tools.broker import SimulatedBroker
from tools.indicators import enrich_indicators
from tools.sizing import infer_lot_size, round_order_shares


class TradingBacktestEngine:
    def __init__(self, config: BacktestConfig | None = None, provider: MarketDataProvider | None = None) -> None:
        self.config = config or BacktestConfig()
        self.provider = provider or build_provider(self.config.provider, __import__("pathlib").Path(__file__).resolve().parents[1] / "data")
        self.graph = TradingAgentsGraph()
        self.reset(self.config)

    def reset(self, config: BacktestConfig | None = None) -> None:
        if config is not None:
            self.config = config
        data_dir = __import__("pathlib").Path(__file__).resolve().parents[1] / "data"
        if self.config.provider == "synthetic" and self.config.days:
            self.provider = SyntheticDataProvider(total_days=self.config.days)
        else:
            self.provider = build_provider(self.config.provider, data_dir)
        self.dynamic_universe_frame = self._resolve_dynamic_universe()
        self.effective_universe = (
            self.dynamic_universe_frame["symbol"].astype(str).tolist()
            if self.dynamic_universe_frame is not None and not self.dynamic_universe_frame.empty
            else list(self.config.universe)
        )
        self.bars = self.provider.load(
            self.config.dataset,
            self.config.symbol,
            self.config.start,
            self.config.end,
            self.config.price_adjust,
        )
        self.symbol_bars: Dict[str, List[MarketBar]] = {}
        self.symbol_frames: Dict[str, Any] = {}
        if self.multi_symbol_mode:
            universe = self.effective_universe or [self.config.symbol]
            for symbol in universe:
                try:
                    bars = self.provider.load(
                        self.config.dataset,
                        symbol,
                        self.config.start,
                        self.config.end,
                        self.config.price_adjust,
                    )
                except Exception as exc:
                    self.trace = getattr(self, "trace", [])
                    self.trace.append({"tool": "dynamic_universe_load", "detail": f"skip {symbol}: {exc}"})
                    continue
                if bars:
                    self.symbol_bars[symbol] = bars
                    self.symbol_frames[symbol] = enrich_indicators(bars_to_frame(bars))
            # A spot quote does not guarantee that the requested historical
            # window exists (new listings and suspended securities are common).
            # Keep selectors and execution on the same actually tradable set.
            loaded_symbols = set(self.symbol_bars)
            self.effective_universe = [
                symbol for symbol in self.effective_universe if symbol in loaded_symbols
            ]
            if self.dynamic_universe_frame is not None:
                self.dynamic_universe_frame = (
                    self.dynamic_universe_frame[
                        self.dynamic_universe_frame["symbol"].astype(str).isin(loaded_symbols)
                    ]
                    .reset_index(drop=True)
                )
            if self.config.dynamic_universe and not self.effective_universe:
                raise UniverseProviderError(
                    "动态股票池中的标的均未取得所选日期范围内的历史行情"
                )
        self.benchmark_error: str | None = None
        self.benchmark_bars = self._load_benchmark_bars()
        self.frame = enrich_indicators(bars_to_frame(self.bars))
        self.day_index = 0
        self.broker = SimulatedBroker(self.config)
        # Inject a fresh, reset-safe top-down selector into the LangGraph so the
        # multi-symbol workflow never runs against a stale config (universe /
        # agent_mode / llm_model changed via /api/reset).
        if self.multi_symbol_mode:
            self.graph.reconfigure_selector(self.config, self.dynamic_universe_frame)
        self.equity_history: List[float] = [self.config.initial_cash]
        self.returns: List[float] = []
        self.trades: List[ExecutionFill] = []
        self.trace: List[Dict[str, str]] = []
        self.research_log: List[Dict[str, Any]] = []
        self.blocked_orders = 0
        self.last_target_positions: Dict[str, int] = {}
        self.last_sector_view: Dict[str, Any] = {}
        # Last multi-symbol rebalance anchor (day/equity/benchmark close) used
        # to back-fill the realized active return onto the prior decision in
        # the experience memory at the next rebalance.
        self._last_rebalance: Dict[str, Any] | None = None

    @property
    def multi_symbol_mode(self) -> bool:
        return bool(self.config.dynamic_universe or self.config.universe)

    def _resolve_dynamic_universe(self):
        """Build a liquid cross-sector A-share universe before loading histories."""
        if not self.config.dynamic_universe:
            return None
        if self.config.provider != "akshare":
            raise ValueError("A股全市场动态股票池当前仅支持 provider='akshare'")
        try:
            provider = AShareUniverseProvider()
            try:
                sectors = provider.list_sectors().sort_values("change_pct", ascending=False)
                sector_count = max(self.config.top_n_sectors, self.config.dynamic_sector_pool)
                frames = []
                for sector in sectors.head(sector_count)["sector"].astype(str):
                    stocks = provider.list_sector_stocks(sector)
                    if not stocks.empty:
                        frames.append(stocks)
                if not frames:
                    raise UniverseProviderError("industry boards returned no constituents")
                frame = __import__("pandas").concat(frames, ignore_index=True)
            except Exception:
                # Eastmoney's industry-board endpoint is occasionally less
                # stable than the full-market quote endpoint. Keep the
                # universe dynamic by falling back to the complete spot table.
                frame = provider.list_spot()
                if frame.empty:
                    raise UniverseProviderError("full-market spot table is empty")
                frame = frame.copy()
                # list_spot now attaches a real Shenwan industry label via
                # industry_map(); only fall back to the coarse exchange-board
                # grouping when the industry mapping failed entirely (every
                # symbol unmapped), so sector-relative-strength selection still
                # has something to work with.
                if frame["sector"].isin(["未分类", "未知"]).all():
                    frame["sector"] = frame.apply(
                        lambda row: self._fallback_dynamic_sector(str(row["symbol"]), str(row.get("name", ""))),
                        axis=1,
                    )
            frame = frame.drop_duplicates("symbol")
            frame["turnover"] = __import__("pandas").to_numeric(frame["turnover"], errors="coerce").fillna(0.0)
            frame["close"] = __import__("pandas").to_numeric(frame["close"], errors="coerce")
            frame["volume"] = __import__("pandas").to_numeric(frame["volume"], errors="coerce").fillna(0)
            frame = frame[
                (frame["close"] > 0)
                & (frame["volume"] > 0)
                & (frame["turnover"] >= self.config.dynamic_min_turnover)
                & ~frame["name"].astype(str).str.contains(r"ST|退", regex=True)
            ]
            return (
                frame.sort_values(["turnover", "change_pct"], ascending=[False, False])
                .head(max(5, self.config.dynamic_universe_limit))
                .reset_index(drop=True)
            )
        except Exception as akshare_exc:
            try:
                return self._resolve_dynamic_universe_tushare()
            except Exception as tushare_exc:
                raise UniverseProviderError(
                    f"动态A股股票池构建失败: akshare={akshare_exc}; tushare={tushare_exc}"
                ) from tushare_exc

    def _resolve_dynamic_universe_tushare(self):
        """Fallback full-market snapshot using the configured Tushare token."""
        import os
        import pandas as pd
        import tushare as ts  # type: ignore

        token = os.getenv("TUSHARE_TOKEN")
        if not token:
            raise UniverseProviderError("TUSHARE_TOKEN is not configured")
        pro = ts.pro_api(token)
        end_date = __import__("datetime").date.today().strftime("%Y%m%d")
        calendar = pro.trade_cal(exchange="", end_date=end_date, is_open="1", fields="cal_date", limit=1)
        if calendar is None or calendar.empty:
            raise UniverseProviderError("Tushare trading calendar returned no open date")
        trade_date = str(calendar.iloc[0]["cal_date"])
        basic = pro.stock_basic(
            exchange="",
            list_status="L",
            fields="ts_code,symbol,name,industry",
        )
        daily = pro.daily(trade_date=trade_date, fields="ts_code,close,pct_chg,vol,amount")
        if basic is None or daily is None or basic.empty or daily.empty:
            raise UniverseProviderError(f"Tushare returned no full-market data for {trade_date}")
        frame = basic.merge(daily, on="ts_code", how="inner")
        normalized = pd.DataFrame(
            {
                "symbol": frame["symbol"].astype(str).str.zfill(6),
                "name": frame["name"].astype(str),
                "close": pd.to_numeric(frame["close"], errors="coerce"),
                "change_pct": pd.to_numeric(frame["pct_chg"], errors="coerce").fillna(0.0),
                "volume": pd.to_numeric(frame["vol"], errors="coerce").fillna(0.0) * 100,
                "turnover": pd.to_numeric(frame["amount"], errors="coerce").fillna(0.0) * 1000,
                "sector": frame["industry"].fillna("未知行业").astype(str),
            }
        )
        normalized = normalized[
            (normalized["close"] > 0)
            & (normalized["volume"] > 0)
            & (normalized["turnover"] >= self.config.dynamic_min_turnover)
            & ~normalized["name"].str.contains(r"ST|退", regex=True)
        ]
        return (
            normalized.sort_values(["turnover", "change_pct"], ascending=[False, False])
            .head(max(5, self.config.dynamic_universe_limit))
            .reset_index(drop=True)
        )

    @staticmethod
    def _fallback_dynamic_sector(symbol: str, name: str) -> str:
        """Coarse exchange/board grouping when industry metadata is unavailable."""
        code = "".join(ch for ch in symbol if ch.isdigit())[:6]
        if "银行" in name:
            return "银行"
        if code.startswith("688"):
            return "科创板"
        if code.startswith("300"):
            return "创业板"
        if code.startswith(("600", "601", "603", "605")):
            return "沪市主板"
        if code.startswith("002"):
            return "深市中小盘"
        if code.startswith("000"):
            return "深市主板"
        if code.startswith(("4", "8", "92")):
            return "北交所"
        return "其他A股"

    @property
    def finished(self) -> bool:
        return self.day_index >= len(self.bars)

    def current_price(self) -> float:
        idx = max(0, min(self.day_index - 1, len(self.bars) - 1))
        return self.bars[idx].close

    def mark_bar(self) -> MarketBar:
        idx = max(0, min(self.day_index, len(self.bars) - 1))
        return self.bars[idx]

    def step(self) -> Dict[str, Any]:
        if self.finished:
            self.trace = [{"tool": "done", "detail": "backtest finished"}]
            return self.to_dict()
        if self.multi_symbol_mode:
            return self._multi_symbol_step()
        return self._single_symbol_step()

    def _single_symbol_step(self) -> Dict[str, Any]:
        bar = self.bars[self.day_index]
        portfolio = self.broker.snapshot(bar.close)
        graph_result = self.graph.run(self.frame, self.day_index, portfolio, self.config)
        fill = self.broker.execute(bar, graph_result.decision)
        self.graph.record_outcome(self.config, bar.day, graph_result, fill)
        previous_equity = self.equity_history[-1]
        self.equity_history.append(fill.equity)
        self.returns.append(fill.equity / previous_equity - 1.0 if previous_equity else 0.0)
        self.trades.append(fill)
        if fill.blocked:
            self.blocked_orders += 1
        self.trace = graph_result.trace + [{"tool": "broker.execute", "detail": fill.model_dump_json()}]
        self.research_log.append(
            {
                "day": bar.day,
                "reports": [report.model_dump(mode="json") for report in graph_result.reports],
                "debate": graph_result.debate.model_dump(mode="json"),
                "risk_debate": graph_result.risk_debate.model_dump(mode="json") if graph_result.risk_debate else None,
                "intent": graph_result.intent.model_dump(mode="json"),
                "risk": graph_result.risk.model_dump(mode="json"),
            }
        )
        self.day_index += 1
        return self.to_dict()

    def _multi_symbol_step(self) -> Dict[str, Any]:
        prices = self._current_prices()
        portfolio = self.broker.snapshot(prices)
        bar = self.bars[self.day_index]
        fills: List[ExecutionFill] = []

        if self._rebalance_day():
            # Back-fill the realized active return onto the previous multi
            # decision so it becomes eligible for experience recall. Active
            # return = portfolio interval return − benchmark interval return.
            prev = self._last_rebalance
            if prev is not None:
                try:
                    bench_now = float(bar.close)
                    bench_prev = float(prev.get("bench", bench_now))
                    eq_prev = float(prev.get("equity", portfolio.equity))
                    port_ret = portfolio.equity / eq_prev - 1.0 if eq_prev else 0.0
                    bench_ret = bench_now / bench_prev - 1.0 if bench_prev else 0.0
                    self.graph.memory.record_outcome(int(prev["day"]), port_ret - bench_ret)
                except Exception:
                    pass
            market_snapshot = self._build_market_snapshot()
            technical_context = self._technical_context()
            graph_result = self.graph.run_multi(
                self.day_index,
                portfolio,
                self.config,
                market_snapshot,
                prices,
                technical_context,
            )
            sector_view = graph_result.sector_view
            target_positions = graph_result.target_positions
            self.last_sector_view = sector_view
            self.last_target_positions = target_positions
            # Anchor for the next rebalance's active-return back-fill. The
            # ``day`` key mirrors the 0-indexed day_index the memory record is
            # written with (run_multi writes state["day"] = day_index), so
            # record_outcome can match it — bar.day is 1-indexed and would
            # miss.
            self._last_rebalance = {
                "day": self.day_index,
                "equity": portfolio.equity,
                "bench": float(bar.close),
            }
            self.research_log.append(
                {
                    "day": bar.day,
                    "sector_view": sector_view,
                    "target_positions": target_positions,
                    "llm_trace": sector_view.get("llm_trace"),
                }
            )
            # Generate sell decisions first to free cash, then buys
            current_symbols = set(portfolio.positions.keys()) | set(target_positions.keys())
            for symbol in sorted(current_symbols, key=lambda s: 0 if portfolio.positions.get(s, 0) > (target_positions.get(s, 0)) else 1):
                target = target_positions.get(symbol, 0)
                current = portfolio.positions.get(symbol, 0)
                if target == current:
                    continue
                side = OrderSide.buy if target > current else OrderSide.sell
                shares = abs(target - current)
                shares = round_order_shares(shares, self.config, side, current)
                if shares <= 0:
                    continue
                decision = PortfolioDecision(
                    symbol=symbol,
                    side=side,
                    shares=shares,
                    signal="top_down_rebalance",
                    confidence=0.7,
                    reason=f"top-down rebalance to {target} shares (current {current})",
                    blocked=False,
                )
                sym_bar = self._bar_for_symbol(symbol, bar.day)
                if sym_bar is None:
                    continue
                fill = self.broker.execute(sym_bar, decision, symbol=symbol, all_prices=self._current_prices())
                fills.append(fill)
                self.trades.append(fill)
                if fill.blocked:
                    self.blocked_orders += 1
                # Refresh portfolio snapshot after each fill for conservative cash accounting
                portfolio = self.broker.snapshot(self._current_prices())
            self.trace = (
                graph_result.trace
                + [{"tool": "broker.execute", "detail": f"fills={len(fills)}, targets={self.last_target_positions}"}]
            )
        else:
            self.trace = [{"tool": "top_down_step", "detail": f"non-rebalance day {bar.day}, fills=0"}]

        previous_equity = self.equity_history[-1]
        equity = portfolio.equity
        self.equity_history.append(equity)
        self.returns.append(equity / previous_equity - 1.0 if previous_equity else 0.0)
        self.day_index += 1
        return self.to_dict()

    def _current_prices(self) -> Dict[str, float]:
        prices: Dict[str, float] = {}
        idx = max(0, min(self.day_index, len(self.bars) - 1))
        prices[self.config.symbol] = self.bars[idx].close
        for symbol, bars in self.symbol_bars.items():
            if bars and self.day_index < len(bars):
                prices[symbol] = bars[self.day_index].close
            elif bars:
                prices[symbol] = bars[-1].close
        return prices

    def _bar_for_symbol(self, symbol: str, day: int) -> MarketBar | None:
        bars = self.symbol_bars.get(symbol)
        if not bars:
            return None
        idx = max(0, min(self.day_index, len(bars) - 1))
        return bars[idx]

    def _rebalance_day(self) -> bool:
        freq = max(1, self.config.rebalance_frequency)
        return self.day_index % freq == 0

    def _build_market_snapshot(self) -> Dict[str, Any]:
        # Lightweight market snapshot for the top-down selector.
        idx = max(0, min(self.day_index, len(self.bars) - 1))
        bar = self.bars[idx]
        first = self.bars[0]
        benchmark_return = bar.close / first.close - 1.0 if first.close else 0.0
        # Composite benchmark trend score replaces the old single ±5% return
        # threshold: a sustained trend needs price structure (moving-average
        # stack), momentum (MACD), volume confirmation, and the cumulative
        # move to agree, not just one number crossing a line. This catches
        # bullish regimes earlier (924-style ramps where cumulative return
        # is still < 5% but the MA stack has just turned positive).
        trend_score = 0.0
        trend_signals: Dict[str, Any] = {}
        try:
            row = self.frame.iloc[idx]
            close = float(row.get("close", 0.0) or 0.0)
            sma_5 = float(row.get("sma_5", 0.0) or 0.0)
            sma_20 = float(row.get("sma_20", 0.0) or 0.0)
            sma_60 = float(row.get("sma_60", 0.0) or 0.0)
            macd_hist = float(row.get("macd_hist", 0.0) or 0.0)
            volume_z = float(row.get("volume_z", 0.0) or 0.0)
            # Moving-average stack: bullish when stacked short>mid>long, bearish reversed.
            if sma_5 and sma_20 and sma_60:
                if close > sma_20 > sma_60:
                    trend_score += 0.30
                    trend_signals["ma_stack"] = "bullish"
                elif close < sma_20 < sma_60:
                    trend_score -= 0.30
                    trend_signals["ma_stack"] = "bearish"
                else:
                    trend_signals["ma_stack"] = "mixed"
            # MACD momentum confirmation.
            if macd_hist > 0:
                trend_score += 0.15
                trend_signals["macd"] = "up"
            elif macd_hist < 0:
                trend_score -= 0.15
                trend_signals["macd"] = "down"
            # Volume confirmation of the move.
            if volume_z > 0.5:
                trend_score += 0.10
            elif volume_z < -0.5:
                trend_score -= 0.10
            # Cumulative return still matters — it is the magnitude of the move.
            trend_score += max(-0.25, min(0.25, benchmark_return))
            trend_signals["volume_z"] = volume_z
        except Exception:
            trend_signals["error"] = "trend calc failed"
        regime = "neutral"
        if trend_score > 0.20:
            regime = "bullish"
        elif trend_score < -0.20:
            regime = "bearish"
        return {
            "day": bar.day,
            "date": bar.date,
            "benchmark_symbol": self.config.symbol,
            "benchmark_return": benchmark_return,
            "market_view": regime,
            "trend_score": round(trend_score, 4),
            "trend_signals": trend_signals,
            "cash_reserve_ratio": self.config.cash_reserve_ratio,
            "max_position_ratio": self.config.max_position_ratio,
        }

    def _technical_context(self) -> Dict[str, Dict[str, float]]:
        """Return latest technical indicators and recent returns for each symbol in the universe."""

        context: Dict[str, Dict[str, float]] = {}
        indicator_cols = ["rsi_14", "volatility_20", "sma_5", "sma_20", "sma_60", "atr_14", "volume_z"]
        for symbol, frame in self.symbol_frames.items():
            if frame.empty or self.day_index >= len(frame):
                continue
            row = frame.iloc[self.day_index]
            metrics: Dict[str, float] = {}
            for col in indicator_cols:
                if col in row.index:
                    value = row[col]
                    if isinstance(value, (int, float)) and not math.isnan(value):
                        metrics[col] = float(value)
            # Add recent returns so the top-down selector can rank sectors by momentum
            close = float(row.get("close", 0.0))
            for lookback, key in [(1, "return_1d"), (5, "return_5d"), (20, "return_20d")]:
                prev_idx = self.day_index - lookback
                if prev_idx >= 0 and close > 0:
                    prev_close = float(frame.iloc[prev_idx].get("close", 0.0))
                    if prev_close > 0:
                        metrics[key] = close / prev_close - 1.0
            if metrics:
                context[symbol] = metrics
        return context

    def run_to_end(self) -> Dict[str, Any]:
        while not self.finished:
            self.step()
        return self.to_dict()

    def quote(self) -> Dict[str, Any]:
        return self.mark_bar().public_dict()

    def live_quote(self) -> Dict[str, Any]:
        return self.provider.latest_quote(self.config.dataset, self.config.symbol).model_dump(mode="json")

    def manual_order(self, side: str, shares: int, reason: str = "manual paper order", realtime: bool = False) -> Dict[str, Any]:
        idx = max(0, min(self.day_index, len(self.bars) - 1))
        bar = self.provider.latest_quote(self.config.dataset, self.config.symbol).to_bar(day=idx + 1) if realtime else self.mark_bar()
        side_enum = OrderSide(side)
        decision = PortfolioDecision(
            side=side_enum,
            shares=max(0, int(shares)),
            signal="manual_paper_order",
            confidence=1.0,
            reason=reason,
            blocked=False,
        )
        fill = self.broker.execute(bar, decision)
        self.trades.append(fill)
        if fill.blocked:
            self.blocked_orders += 1
        self.trace = [{"tool": "paper_broker.manual_order", "detail": fill.model_dump_json()}]
        return self.to_dict()

    def rebalance_to(self, target_ratio: float, realtime: bool = False, reason: str = "manual rebalance") -> Dict[str, Any]:
        bar = self.provider.latest_quote(self.config.dataset, self.config.symbol).to_bar(day=self.mark_bar().day) if realtime else self.mark_bar()
        target_ratio = max(0.0, min(1.0, float(target_ratio)))
        snapshot = self.broker.snapshot(bar.close)
        target_shares = int((snapshot.equity * target_ratio) // bar.close) if bar.close > 0 else 0
        delta = target_shares - snapshot.position
        if delta == 0:
            self.trace = [
                {
                    "tool": "paper_broker.rebalance",
                    "detail": f"already near target exposure {target_ratio:.2%}; position={snapshot.position}",
                }
            ]
            return self.to_dict()
        side = "buy" if delta > 0 else "sell"
        return self.manual_order(side, abs(delta), f"{reason}; target exposure {target_ratio:.2%}", realtime)

    def adjust_position(self, direction: str, ratio_delta: float, realtime: bool = False) -> Dict[str, Any]:
        account = self.account(realtime=False)
        current_ratio = float(account["position_ratio"])
        delta = abs(float(ratio_delta))
        target = current_ratio + delta if direction == "increase" else current_ratio - delta
        return self.rebalance_to(target, realtime, reason=f"manual {direction} by {delta:.2%}")

    def account(self, realtime: bool = False) -> Dict[str, Any]:
        if self.multi_symbol_mode:
            return self._multi_symbol_account(realtime)
        return self._single_symbol_account(realtime)

    def _single_symbol_account(self, realtime: bool = False) -> Dict[str, Any]:
        quote_payload: Dict[str, Any] | None = None
        instrument = self.provider.instrument_info(self.config.symbol)
        if realtime:
            tick = self.provider.latest_quote(self.config.dataset, self.config.symbol)
            bar = tick.to_bar(day=self.mark_bar().day)
            quote_payload = tick.model_dump(mode="json")
            raw_name = quote_payload.get("raw", {}).get("名称") if isinstance(quote_payload.get("raw"), dict) else None
            if raw_name:
                instrument["name"] = str(raw_name)
            price_mode = "realtime"
        else:
            bar = self.mark_bar()
            quote_payload = bar.public_dict()
            quote_payload["provider"] = self.config.provider
            quote_payload["symbol"] = self.config.symbol
            quote_payload["source"] = "historical-backtest-bar"
            quote_payload["timestamp"] = f"{bar.date}T15:00:00"
            quote_payload["adjust"] = self.config.price_adjust
            price_mode = "historical"

        snapshot = self.broker.snapshot(bar.close)
        market_value = snapshot.position * bar.close
        cost_basis = (snapshot.avg_entry or 0.0) * snapshot.position
        unrealized_pnl = market_value - cost_basis if snapshot.position else 0.0
        unrealized_return = unrealized_pnl / cost_basis if cost_basis else 0.0
        position_ratio = market_value / snapshot.equity if snapshot.equity else 0.0
        cash_ratio = snapshot.cash / snapshot.equity if snapshot.equity else 0.0
        total_return = snapshot.equity / self.config.initial_cash - 1.0 if self.config.initial_cash else 0.0
        benchmark = self.benchmark_summary()
        symbol_return = self.symbol_buy_hold_return(bar.close)
        active_return = total_return - benchmark["return"]

        return {
            "symbol": self.config.symbol,
            "instrument": instrument,
            "price": round(bar.close, 4),
            "price_date": bar.date,
            "price_day": bar.day,
            "price_mode": price_mode,
            "price_adjust": self.config.price_adjust,
            "price_source": quote_payload.get("source", "unknown") if quote_payload else "unknown",
            "price_timestamp": quote_payload.get("timestamp") if quote_payload else None,
            "quote": quote_payload,
            "cash": round(snapshot.cash, 4),
            "position": snapshot.position,
            "avg_entry": round(snapshot.avg_entry or 0.0, 4),
            "market_value": round(market_value, 4),
            "cost_basis": round(cost_basis, 4),
            "equity": round(snapshot.equity, 4),
            "initial_cash": round(self.config.initial_cash, 4),
            "position_ratio": round(position_ratio, 6),
            "cash_ratio": round(cash_ratio, 6),
            "unrealized_pnl": round(unrealized_pnl, 4),
            "unrealized_return": round(unrealized_return, 6),
            "realized_pnl": round(snapshot.realized_pnl, 4),
            "total_fees": round(snapshot.total_fees, 4),
            "turnover": round(snapshot.turnover, 4),
            "total_return": round(total_return, 6),
            "symbol_buy_hold_return": round(symbol_return, 6),
            "benchmark_return": round(benchmark["return"], 6),
            "active_return": round(active_return, 6),
            "outperforming": active_return >= 0,
            "benchmark": benchmark,
        }

    def _multi_symbol_account(self, realtime: bool = False) -> Dict[str, Any]:
        # Realtime quotes are currently single-symbol only; use historical prices for aggregation.
        prices = self._current_prices()
        bar = self.mark_bar()
        quote_payload = bar.public_dict()
        quote_payload["provider"] = self.config.provider
        quote_payload["symbol"] = self.config.symbol
        quote_payload["source"] = "historical-backtest-bar-multi"
        quote_payload["timestamp"] = f"{bar.date}T15:00:00"
        quote_payload["adjust"] = self.config.price_adjust
        quote_payload["note"] = "multi-symbol aggregate; price reflects main symbol"

        snapshot = self.broker.snapshot(prices)
        market_value = sum(
            int(snapshot.positions.get(symbol, 0)) * price for symbol, price in prices.items()
        )
        cost_basis = sum(
            (snapshot.avg_entries.get(symbol) or 0.0) * int(shares)
            for symbol, shares in snapshot.positions.items()
        )
        unrealized_pnl = market_value - cost_basis if snapshot.positions else 0.0
        unrealized_return = unrealized_pnl / cost_basis if cost_basis else 0.0
        position_ratio = market_value / snapshot.equity if snapshot.equity else 0.0
        cash_ratio = snapshot.cash / snapshot.equity if snapshot.equity else 0.0
        total_return = snapshot.equity / self.config.initial_cash - 1.0 if self.config.initial_cash else 0.0
        benchmark = self.benchmark_summary()
        active_return = total_return - benchmark["return"]

        instrument = self.provider.instrument_info(self.config.symbol)
        instrument["name"] = f"多标组合 ({len(snapshot.positions)} 只持仓)"
        instrument["asset_type"] = "multi-symbol portfolio"
        instrument["note"] = "main symbol displayed for price reference"

        return {
            "symbol": self.config.symbol,
            "instrument": instrument,
            "price": round(bar.close, 4),
            "price_date": bar.date,
            "price_day": bar.day,
            "price_mode": "realtime" if realtime else "historical",
            "price_adjust": self.config.price_adjust,
            "price_source": quote_payload.get("source", "unknown"),
            "price_timestamp": quote_payload.get("timestamp"),
            "quote": quote_payload,
            "cash": round(snapshot.cash, 4),
            "position": snapshot.total_position,
            "avg_entry": round(
                cost_basis / snapshot.total_position if snapshot.total_position else 0.0, 4
            ),
            "market_value": round(market_value, 4),
            "cost_basis": round(cost_basis, 4),
            "equity": round(snapshot.equity, 4),
            "initial_cash": round(self.config.initial_cash, 4),
            "position_ratio": round(position_ratio, 6),
            "cash_ratio": round(cash_ratio, 6),
            "unrealized_pnl": round(unrealized_pnl, 4),
            "unrealized_return": round(unrealized_return, 6),
            "realized_pnl": round(snapshot.realized_pnl, 4),
            "total_fees": round(snapshot.total_fees, 4),
            "turnover": round(snapshot.turnover, 4),
            "total_return": round(total_return, 6),
            "symbol_buy_hold_return": round(self.symbol_buy_hold_return(bar.close), 6),
            "benchmark_return": round(benchmark["return"], 6),
            "active_return": round(active_return, 6),
            "outperforming": active_return >= 0,
            "benchmark": benchmark,
            "multi_symbol": True,
            "positions": {k: int(v) for k, v in snapshot.positions.items()},
            "avg_entries": {k: round(v, 4) for k, v in snapshot.avg_entries.items()},
        }

    def report(self) -> Dict[str, Any]:
        if self.multi_symbol_mode:
            prices = self._current_prices()
            snapshot = self.broker.snapshot(prices)
            price = self.mark_bar().close
        else:
            price = self.mark_bar().close
            snapshot = self.broker.snapshot(price)
        total_return = snapshot.equity / self.config.initial_cash - 1.0
        baseline_return = self.symbol_buy_hold_return(price)
        benchmark = self.benchmark_summary()
        active_return = total_return - benchmark["return"]
        returns_std = pstdev(self.returns) if len(self.returns) > 1 else 0.0
        returns_mean = mean(self.returns) if self.returns else 0.0
        sharpe = (returns_mean / returns_std) * (252 ** 0.5) if returns_std else 0.0
        win_rate = snapshot.realized_wins / snapshot.realized_trades if snapshot.realized_trades else 0.0
        return {
            "equity": round(snapshot.equity, 4),
            "total_return": round(total_return, 6),
            "max_drawdown": round(snapshot.max_drawdown, 6),
            "baseline_return": round(baseline_return, 6),
            "benchmark_symbol": benchmark["symbol"],
            "benchmark_return": round(benchmark["return"], 6),
            "active_return": round(active_return, 6),
            "outperforming": active_return >= 0,
            "sharpe": round(sharpe, 4),
            "win_rate": round(win_rate, 6),
            "blocked_orders": self.blocked_orders,
            "trade_count": len([trade for trade in self.trades if trade.side.value != "hold"]),
            "finished": self.finished,
        }

    def symbol_buy_hold_return(self, price: float) -> float:
        return price / self.bars[0].close - 1.0 if self.bars and self.bars[0].close else 0.0

    def benchmark_summary(self) -> Dict[str, Any]:
        if not self.benchmark_bars:
            return {"symbol": self.config.benchmark_symbol, "price": 0.0, "return": 0.0, "source": "none", "error": self.benchmark_error}
        idx = max(0, min(self.day_index, len(self.benchmark_bars) - 1))
        first = self.benchmark_bars[0]
        current = self.benchmark_bars[idx]
        value = current.close / first.close - 1.0 if first.close else 0.0
        source = "provider" if self.benchmark_error is None else "synthetic-fallback"
        return {
            "symbol": self.config.benchmark_symbol,
            "date": current.date,
            "price": round(current.close, 4),
            "return": round(value, 6),
            "source": source,
            "error": self.benchmark_error,
        }

    def _load_benchmark_bars(self) -> List[MarketBar]:
        fallback_dataset = "range" if self.config.dataset != "range" else "trend"
        fallback = SyntheticDataProvider(total_days=len(self.bars)).load(fallback_dataset, self.config.benchmark_symbol)
        if self.config.provider in {"synthetic", "csv"}:
            return fallback
        try:
            bars = self.provider.load(
                self.config.dataset,
                self.config.benchmark_symbol,
                self.config.start,
                self.config.end,
                self.config.price_adjust,
            )
            if not bars:
                raise RuntimeError("benchmark provider returned no rows")
            return bars
        except Exception as exc:
            self.benchmark_error = str(exc)
            return fallback

    def _benchmark_equity_curve(self) -> List[float]:
        if not self.benchmark_bars:
            return [self.config.initial_cash]
        first = self.benchmark_bars[0].close
        if not first or math.isclose(first, 0.0):
            return [self.config.initial_cash for _bar in self.benchmark_bars]
        return [self.config.initial_cash * (bar.close / first) for bar in self.benchmark_bars]

    def to_dict(self) -> Dict[str, Any]:
        if self.multi_symbol_mode:
            prices = self._current_prices()
            snapshot = self.broker.snapshot(prices)
        else:
            price = self.mark_bar().close
            snapshot = self.broker.snapshot(price)
        portfolio_block: Dict[str, Any] = {
            "cash": round(snapshot.cash, 4),
            "position": snapshot.position,
            "avg_entry": round(snapshot.avg_entry or 0.0, 4),
            "equity": round(snapshot.equity, 4),
            "peak": round(snapshot.peak, 4),
            "max_drawdown": round(snapshot.max_drawdown, 6),
            "realized_pnl": round(snapshot.realized_pnl, 4),
            "total_fees": round(snapshot.total_fees, 4),
            "turnover": round(snapshot.turnover, 4),
        }
        if self.multi_symbol_mode:
            portfolio_block["positions"] = {k: int(v) for k, v in snapshot.positions.items()}
            portfolio_block["avg_entries"] = {k: round(v, 4) for k, v in snapshot.avg_entries.items()}
            portfolio_block["market_value"] = round(
                sum(int(snapshot.positions.get(s, 0)) * p for s, p in self._current_prices().items()), 4
            )
            portfolio_block["position_ratio"] = round(
                portfolio_block["market_value"] / snapshot.equity, 6
            ) if snapshot.equity else 0.0
        else:
            price = self.mark_bar().close
            portfolio_block["market_value"] = round(snapshot.position * price, 4)
            portfolio_block["position_ratio"] = round(
                (snapshot.position * price) / snapshot.equity, 6
            ) if snapshot.equity else 0.0
        return {
            "config": self.config.model_dump(),
            "day": self.day_index,
            "total_days": len(self.bars),
            "series": [bar.public_dict() for bar in self.bars],
            "benchmark_series": [bar.public_dict() for bar in self.benchmark_bars],
            "benchmark_equity": [round(value, 4) for value in self._benchmark_equity_curve()],
            "portfolio": portfolio_block,
            "account": self.account(realtime=False),
            "equity_history": [round(value, 4) for value in self.equity_history],
            "trades": [trade.public_dict() for trade in self.trades],
            "trace": self.trace,
            "research_log": self.research_log[-20:],
            "report": self.report(),
            "sector_view": self.last_sector_view,
        }
