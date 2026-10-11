"""Behavior contracts that protect member A's architecture refactors.

These tests intentionally assert public shapes and observable behavior rather
than private LangGraph state fields.  Additive work by member D therefore stays
possible while later dependency-injection and extraction changes get a stable
regression baseline.
"""

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.graph import TradingAgentsGraph  # noqa: E402
from agents.memory import TradingMemory  # noqa: E402
from core.schemas import (  # noqa: E402
    BacktestConfig,
    GraphResult,
    MultiGraphResult,
    OrderSide,
    PortfolioDecision,
)
from data.providers import SyntheticDataProvider  # noqa: E402
from engine.backtest import TradingBacktestEngine  # noqa: E402


class _FixedMultiGraph:
    """Minimal deterministic graph double for observing engine order flow."""

    def __init__(self, targets: dict[str, int]) -> None:
        self.targets = targets
        self.calls = 0

    def run_multi(self, *_args, **_kwargs) -> MultiGraphResult:
        self.calls += 1
        return MultiGraphResult(
            sector_view={"source": "architecture-contract"},
            target_positions=self.targets,
            trace=[{"tool": "fixed_multi_graph", "detail": "deterministic targets"}],
        )


class ArchitectureBehaviorContractTest(unittest.TestCase):
    """Pin external behavior before member A changes internal boundaries."""

    @staticmethod
    def _config(**updates) -> BacktestConfig:
        values = {
            "provider": "synthetic",
            "agent_mode": "offline",
            "dataset": "trend",
            "days": 8,
        }
        values.update(updates)
        return BacktestConfig(**values)

    def test_default_construction_path_exposes_stable_public_state(self) -> None:
        engine = TradingBacktestEngine(self._config())

        self.assertIsInstance(engine.provider, SyntheticDataProvider)
        self.assertIsInstance(engine.graph, TradingAgentsGraph)
        self.assertFalse(engine.multi_symbol_mode)
        self.assertEqual(engine.day_index, 0)

        state = engine.to_dict()
        required_top_level = {
            "config",
            "day",
            "total_days",
            "series",
            "portfolio",
            "account",
            "equity_history",
            "trades",
            "trace",
            "research_log",
            "report",
            "sector_view",
        }
        required_report = {
            "equity",
            "total_return",
            "max_drawdown",
            "benchmark_return",
            "active_return",
            "sharpe",
            "trade_count",
            "finished",
        }
        self.assertTrue(required_top_level.issubset(state))
        self.assertTrue(required_report.issubset(state["report"]))
        self.assertEqual(state["equity_history"], [engine.config.initial_cash])

    def test_reset_clears_run_state_and_preserves_graph_service(self) -> None:
        engine = TradingBacktestEngine(self._config(initial_cash=100_000.0))
        graph = engine.graph
        changed = engine.manual_order("buy", 10, reason="architecture reset baseline")
        self.assertTrue(changed["trades"])
        self.assertGreater(changed["portfolio"]["position"], 0)

        replacement = self._config(initial_cash=250_000.0, days=6)
        engine.reset(replacement)
        state = engine.to_dict()

        self.assertIs(engine.graph, graph)
        self.assertEqual(engine.config, replacement)
        self.assertEqual(state["day"], 0)
        self.assertEqual(state["total_days"], 6)
        self.assertEqual(state["trades"], [])
        self.assertEqual(state["trace"], [])
        self.assertEqual(state["research_log"], [])
        self.assertEqual(state["portfolio"]["cash"], 250_000.0)
        self.assertEqual(state["portfolio"]["position"], 0)
        self.assertEqual(state["equity_history"], [250_000.0])

    def test_single_symbol_graph_entry_returns_typed_result_offline(self) -> None:
        engine = TradingBacktestEngine(self._config(symbol="ARCH-SINGLE"))
        portfolio = engine.broker.snapshot(engine.mark_bar().close)

        with tempfile.TemporaryDirectory() as tmpdir:
            engine.graph.memory = TradingMemory(Path(tmpdir) / "memory.jsonl")
            result = engine.graph.run(
                engine.frame,
                engine.day_index,
                portfolio,
                engine.config,
            )

        self.assertIsInstance(result, GraphResult)
        self.assertTrue(result.reports)
        self.assertIn(result.decision.side, set(OrderSide))
        self.assertIsInstance(result.trace, list)

    def test_multi_symbol_graph_entry_returns_typed_result_offline(self) -> None:
        config = self._config(
            symbol="BENCH",
            universe=["000001", "000002", "600000"],
            rebalance_frequency=1,
        )
        engine = TradingBacktestEngine(config)
        prices = engine._current_prices()
        portfolio = engine.broker.snapshot(prices)

        with tempfile.TemporaryDirectory() as tmpdir:
            engine.graph.memory = TradingMemory(Path(tmpdir) / "memory.jsonl")
            result = engine.graph.run_multi(
                engine.day_index,
                portfolio,
                config,
                engine._build_market_snapshot(),
                prices,
                engine._technical_context(),
            )

        self.assertIsInstance(result, MultiGraphResult)
        self.assertIsInstance(result.sector_view, dict)
        self.assertIsInstance(result.target_positions, dict)
        self.assertIsInstance(result.trace, list)

    def test_multi_symbol_rebalance_executes_sell_before_buy(self) -> None:
        config = self._config(
            symbol="BENCH",
            universe=["AAA", "BBB"],
            rebalance_frequency=1,
            fee_rate=0.0,
            slippage_bps=0.0,
        )
        engine = TradingBacktestEngine(config)
        prices = engine._current_prices()
        buy = PortfolioDecision(
            symbol="AAA",
            side=OrderSide.buy,
            shares=10,
            signal="test_setup",
            confidence=1.0,
            reason="establish current holding",
        )
        engine.broker.execute(
            engine._bar_for_symbol("AAA", 0),
            buy,
            symbol="AAA",
            all_prices=prices,
        )
        fixed_graph = _FixedMultiGraph({"BBB": 5})
        engine.graph = fixed_graph

        engine.step()

        self.assertEqual(fixed_graph.calls, 1)
        self.assertEqual(
            [(fill.symbol, fill.side, fill.shares) for fill in engine.trades],
            [
                ("AAA", OrderSide.sell, 10),
                ("BBB", OrderSide.buy, 5),
            ],
        )


if __name__ == "__main__":
    unittest.main()
