"""Dependency-injection tests for ``TradingBacktestEngine``."""

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.contracts import GraphRunner, MarketDataSource  # noqa: E402
from core.schemas import (  # noqa: E402
    AnalystReport,
    BacktestConfig,
    GraphResult,
    MultiGraphResult,
    OrderSide,
    PortfolioDecision,
    ResearchDebate,
    RiskAssessment,
    Stance,
    TradeIntent,
)
from data.providers import SyntheticDataProvider  # noqa: E402
from engine.backtest import TradingBacktestEngine  # noqa: E402


class _RecordingProvider:
    def __init__(self, total_days: int = 6) -> None:
        self.delegate = SyntheticDataProvider(total_days=total_days)
        self.load_calls = []

    def load(self, dataset, symbol="DEMO", start=None, end=None, adjust=None):
        self.load_calls.append((dataset, symbol, start, end, adjust))
        return self.delegate.load(dataset, symbol, start, end, adjust)

    def latest_quote(self, dataset, symbol="DEMO"):
        return self.delegate.latest_quote(dataset, symbol)

    def instrument_info(self, symbol="DEMO"):
        return self.delegate.instrument_info(symbol)


class _RecordingGraph:
    def __init__(self) -> None:
        self.reconfigure_calls = []
        self.single_calls = []
        self.multi_calls = []
        self.outcome_calls = []

    def reconfigure_selector(self, config, universe_frame=None) -> None:
        self.reconfigure_calls.append((config, universe_frame))

    def run(self, frame, day, portfolio, config) -> GraphResult:
        self.single_calls.append((frame, day, portfolio, config))
        return _hold_result()

    def run_multi(
        self,
        day,
        portfolio,
        config,
        market_snapshot,
        prices,
        technical_context,
    ) -> MultiGraphResult:
        self.multi_calls.append(
            (day, portfolio, config, market_snapshot, prices, technical_context)
        )
        return MultiGraphResult(
            sector_view={"source": "injected-graph"},
            target_positions={},
            trace=[{"tool": "recording_graph", "detail": "multi"}],
        )

    def record_outcome(self, config, day, graph_result, fill) -> None:
        self.outcome_calls.append((config, day, graph_result, fill))


class _RecordingMemory:
    def __init__(self) -> None:
        self.outcome_calls = []

    def record_outcome(self, day, active_return) -> None:
        self.outcome_calls.append((day, active_return))


class _LearningMemory:
    def __init__(self) -> None:
        self.outcome_calls = []
        self.execution_calls = []

    def record_outcome(self, day, active_return, *, outcome_day=None, holding_days=None) -> None:
        self.outcome_calls.append((day, active_return, outcome_day, holding_days))

    def record_execution(self, day, execution) -> None:
        self.execution_calls.append((day, execution))


def _hold_result() -> GraphResult:
    report = AnalystReport(
        agent="injected_test_analyst",
        stance=Stance.neutral,
        score=0.0,
        confidence=1.0,
        summary="deterministic hold",
    )
    debate = ResearchDebate(
        bull_thesis="none",
        bear_thesis="none",
        bull_score=0.0,
        bear_score=0.0,
        verdict=Stance.neutral,
        conviction=1.0,
        manager_notes="deterministic hold",
    )
    intent = TradeIntent(
        signal="hold",
        side=OrderSide.hold,
        target_position=0,
        shares=0,
        confidence=1.0,
        rationale="deterministic hold",
    )
    risk = RiskAssessment(
        approved=True,
        side=OrderSide.hold,
        shares=0,
        risk_score=0.0,
        reasons=["no order"],
    )
    decision = PortfolioDecision(
        side=OrderSide.hold,
        shares=0,
        signal="hold",
        confidence=1.0,
        reason="deterministic hold",
    )
    return GraphResult(
        reports=[report],
        debate=debate,
        intent=intent,
        risk=risk,
        decision=decision,
        trace=[{"tool": "recording_graph", "detail": "single"}],
    )


class EngineDependencyInjectionTest(unittest.TestCase):
    @staticmethod
    def _config(**updates) -> BacktestConfig:
        values = {
            "provider": "synthetic",
            "agent_mode": "offline",
            "dataset": "trend",
            "days": 9,
        }
        values.update(updates)
        return BacktestConfig(**values)

    def test_constructor_uses_injected_graph_and_provider(self) -> None:
        provider = _RecordingProvider(total_days=6)
        graph = _RecordingGraph()

        engine = TradingBacktestEngine(self._config(), provider=provider, graph=graph)

        self.assertIsInstance(provider, MarketDataSource)
        self.assertIsInstance(graph, GraphRunner)
        self.assertIs(engine.provider, provider)
        self.assertIs(engine.graph, graph)
        self.assertEqual(len(engine.bars), 6)
        self.assertEqual(provider.load_calls[0][1], "DEMO")

    def test_reset_preserves_injected_services_and_reconfigures_multi_graph(self) -> None:
        provider = _RecordingProvider(total_days=6)
        graph = _RecordingGraph()
        engine = TradingBacktestEngine(self._config(), provider=provider, graph=graph)
        replacement = self._config(
            symbol="BENCH",
            universe=["AAA", "BBB"],
            days=20,
            initial_cash=250_000.0,
        )

        engine.reset(replacement)

        self.assertIs(engine.provider, provider)
        self.assertIs(engine.graph, graph)
        self.assertEqual(len(engine.bars), 6)
        self.assertEqual(engine.effective_universe, ["AAA", "BBB"])
        self.assertEqual(graph.reconfigure_calls[-1][0], replacement)
        self.assertIsNone(graph.reconfigure_calls[-1][1])
        self.assertEqual(engine.to_dict()["portfolio"]["cash"], 250_000.0)

    def test_injected_graph_drives_single_symbol_step_and_outcome(self) -> None:
        provider = _RecordingProvider()
        graph = _RecordingGraph()
        engine = TradingBacktestEngine(self._config(), provider=provider, graph=graph)

        state = engine.step()

        self.assertEqual(len(graph.single_calls), 1)
        self.assertEqual(len(graph.outcome_calls), 1)
        self.assertEqual(state["day"], 1)
        self.assertEqual(state["trades"][0]["side"], "hold")

    def test_multi_rebalances_do_not_require_graph_memory_attribute(self) -> None:
        provider = _RecordingProvider()
        graph = _RecordingGraph()
        config = self._config(
            symbol="BENCH",
            universe=["AAA", "BBB"],
            rebalance_frequency=1,
        )
        engine = TradingBacktestEngine(config, provider=provider, graph=graph)

        engine.step()
        state = engine.step()

        self.assertFalse(hasattr(graph, "memory"))
        self.assertEqual(len(graph.multi_calls), 2)
        self.assertEqual(state["day"], 2)
        self.assertEqual(state["sector_view"], {"source": "injected-graph"})

    def test_multi_outcome_hook_remains_compatible_when_graph_exposes_memory(self) -> None:
        provider = _RecordingProvider()
        graph = _RecordingGraph()
        graph.memory = _RecordingMemory()
        config = self._config(
            symbol="BENCH",
            universe=["AAA"],
            rebalance_frequency=1,
        )
        engine = TradingBacktestEngine(config, provider=provider, graph=graph)

        engine.step()
        engine.step()

        self.assertEqual(len(graph.memory.outcome_calls), 1)
        self.assertEqual(graph.memory.outcome_calls[0][0], 0)
        self.assertIsInstance(graph.memory.outcome_calls[0][1], float)

    def test_multi_learning_hooks_preserve_execution_and_holding_metadata(self) -> None:
        provider = _RecordingProvider()
        graph = _RecordingGraph()
        graph.memory = _LearningMemory()
        config = self._config(
            symbol="BENCH",
            universe=["AAA"],
            rebalance_frequency=1,
        )
        engine = TradingBacktestEngine(config, provider=provider, graph=graph)

        engine.step()
        engine.step()

        self.assertEqual(graph.memory.outcome_calls[0][0], 0)
        self.assertEqual(graph.memory.outcome_calls[0][2:], (1, 1))
        self.assertEqual(graph.memory.execution_calls[0], (0, {
            "attempted_orders": 0,
            "filled_orders": 0,
            "blocked_orders": 0,
        }))


if __name__ == "__main__":
    unittest.main()
