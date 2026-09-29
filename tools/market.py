from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from core.schemas import BacktestConfig
from engine.backtest import TradingBacktestEngine


class MarketEnv:
    """Compatibility facade over the full TradingAgents-like engine.

    The original course API names are preserved for the frontend and tests:
    reset, observe, act, report, step, run_to_end. Internally this now delegates
    to a multi-agent graph and an event-driven broker.
    """

    def __init__(
        self,
        dataset: str = "trend",
        strategy: str = "multi_agent",
        risk_budget: float = 0.30,
        fee_rate: float = 0.0012,
        provider: str = "synthetic",
        symbol: str = "DEMO",
        benchmark_symbol: str = "BENCH",
        price_adjust: str = "qfq",
        agent_mode: str | None = "offline",
        llm_base_url: str | None = None,
        start: str | None = None,
        end: str | None = None,
        universe: List[str] | None = None,
        dynamic_universe: bool = False,
        dynamic_universe_limit: int = 30,
        cash_reserve_ratio: float = 0.05,
        max_position_ratio: float = 0.25,
        rebalance_frequency: int = 5,
    ) -> None:
        mode = agent_mode or os.getenv("TRADING_AGENT_MODE", "llm")
        self.config = BacktestConfig(
            dataset=dataset,
            strategy=strategy,
            risk_budget=risk_budget,
            fee_rate=fee_rate,
            provider=provider,
            symbol=symbol,
            benchmark_symbol=benchmark_symbol,
            price_adjust=price_adjust,
            start=start,
            end=end,
            agent_mode=mode,
            llm_model=os.getenv("OPENAI_MODEL", "glm-5.2"),
            deep_model=os.getenv("OPENAI_DEEP_MODEL", "glm-5.2"),
            llm_base_url=llm_base_url or os.getenv("OPENAI_BASE_URL") or None,
            universe=list(universe) if universe is not None else [],
            dynamic_universe=dynamic_universe,
            dynamic_universe_limit=dynamic_universe_limit,
            cash_reserve_ratio=cash_reserve_ratio,
            max_position_ratio=max_position_ratio,
            rebalance_frequency=rebalance_frequency,
        )
        self.engine = TradingBacktestEngine(self.config)

    def reset(
        self,
        dataset: Optional[str] = None,
        strategy: Optional[str] = None,
        risk_budget: Optional[float] = None,
        fee_rate: Optional[float] = None,
        provider: Optional[str] = None,
        symbol: Optional[str] = None,
        benchmark_symbol: Optional[str] = None,
        price_adjust: Optional[str] = None,
        agent_mode: Optional[str] = None,
        llm_model: Optional[str] = None,
        deep_model: Optional[str] = None,
        llm_base_url: Optional[str] = None,
        max_debate_rounds: Optional[int] = None,
        max_risk_debate_rounds: Optional[int] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        universe: Optional[List[str]] = None,
        dynamic_universe: Optional[bool] = None,
        dynamic_universe_limit: Optional[int] = None,
        max_positions: Optional[int] = None,
        top_n_sectors: Optional[int] = None,
        stocks_per_sector: Optional[int] = None,
        rebalance_frequency: Optional[int] = None,
        cash_reserve_ratio: Optional[float] = None,
        max_position_ratio: Optional[float] = None,
        initial_cash: Optional[float] = None,
        days: Optional[int] = None,
    ) -> None:
        payload = self.config.model_dump()
        if dataset is not None:
            payload["dataset"] = dataset
        if strategy is not None:
            payload["strategy"] = strategy
        if risk_budget is not None:
            payload["risk_budget"] = risk_budget
        if fee_rate is not None:
            payload["fee_rate"] = fee_rate
        if provider is not None:
            payload["provider"] = provider
        if symbol is not None:
            payload["symbol"] = symbol
        if benchmark_symbol is not None:
            payload["benchmark_symbol"] = benchmark_symbol
        if price_adjust is not None:
            payload["price_adjust"] = price_adjust
        if agent_mode is not None:
            payload["agent_mode"] = agent_mode
        if llm_model is not None:
            payload["llm_model"] = llm_model
        if deep_model is not None:
            payload["deep_model"] = deep_model
        if llm_base_url is not None:
            payload["llm_base_url"] = llm_base_url or None
        if max_debate_rounds is not None:
            payload["max_debate_rounds"] = max_debate_rounds
        if max_risk_debate_rounds is not None:
            payload["max_risk_debate_rounds"] = max_risk_debate_rounds
        if start is not None:
            payload["start"] = start or None
        if end is not None:
            payload["end"] = end or None
        if universe is not None:
            payload["universe"] = universe
        if dynamic_universe is not None:
            payload["dynamic_universe"] = dynamic_universe
        if dynamic_universe_limit is not None:
            payload["dynamic_universe_limit"] = dynamic_universe_limit
        if max_positions is not None:
            payload["max_positions"] = max_positions
        if top_n_sectors is not None:
            payload["top_n_sectors"] = top_n_sectors
        if stocks_per_sector is not None:
            payload["stocks_per_sector"] = stocks_per_sector
        if rebalance_frequency is not None:
            payload["rebalance_frequency"] = rebalance_frequency
        if cash_reserve_ratio is not None:
            payload["cash_reserve_ratio"] = cash_reserve_ratio
        if max_position_ratio is not None:
            payload["max_position_ratio"] = max_position_ratio
        if initial_cash is not None:
            payload["initial_cash"] = initial_cash
        if days is not None:
            payload["days"] = days
        self.config = BacktestConfig(**payload)
        self.engine.reset(self.config)

    @property
    def finished(self) -> bool:
        return self.engine.finished

    @property
    def dataset(self) -> str:
        return self.config.dataset

    @property
    def strategy(self) -> str:
        return self.config.strategy

    @property
    def risk_budget(self) -> float:
        return self.config.risk_budget

    @property
    def fee_rate(self) -> float:
        return self.config.fee_rate

    def observe(self) -> Dict[str, Any]:
        idx = min(self.engine.day_index, len(self.engine.frame) - 1)
        row = self.engine.frame.iloc[idx]
        keys = ["close", "sma_5", "sma_20", "sma_60", "macd_hist", "rsi_14", "atr_14", "volatility_20", "volume_z"]
        return {key: float(row[key]) for key in keys}

    def act(self, *_args: Any, **_kwargs: Any) -> Dict[str, Any]:
        return self.step()

    def step(self) -> Dict[str, Any]:
        return self.engine.step()

    def run_to_end(self) -> Dict[str, Any]:
        return self.engine.run_to_end()

    def quote(self) -> Dict[str, Any]:
        return self.engine.quote()

    def live_quote(self) -> Dict[str, Any]:
        return self.engine.live_quote()

    def manual_order(self, side: str, shares: int, reason: str = "manual paper order", realtime: bool = False) -> Dict[str, Any]:
        return self.engine.manual_order(side, shares, reason, realtime)

    def rebalance_to(self, target_ratio: float, realtime: bool = False, reason: str = "manual rebalance") -> Dict[str, Any]:
        return self.engine.rebalance_to(target_ratio, realtime, reason)

    def adjust_position(self, direction: str, ratio_delta: float, realtime: bool = False) -> Dict[str, Any]:
        return self.engine.adjust_position(direction, ratio_delta, realtime)

    def account(self, realtime: bool = False) -> Dict[str, Any]:
        return self.engine.account(realtime)

    def report(self) -> Dict[str, Any]:
        return self.engine.report()

    @property
    def trades(self):
        return self.engine.trades

    def to_dict(self) -> Dict[str, Any]:
        return self.engine.to_dict()
