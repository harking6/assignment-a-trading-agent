"""Cross-module parity and dependency-boundary tests for member A's refactors.

The scenarios use deterministic offline collaborators and compare observable
engine results.  They deliberately avoid pinning prompts, factor scores,
memory ranking, or private graph state owned by other team members.
"""

from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.schemas import BacktestConfig, MultiGraphResult  # noqa: E402
from data.providers import SyntheticDataProvider  # noqa: E402
from data.universe_service import UniverseSnapshot  # noqa: E402
from engine.backtest import TradingBacktestEngine  # noqa: E402


class _ScriptedMultiGraph:
    """Return deterministic target holdings without LLM or memory I/O."""

    def __init__(self) -> None:
        self.reconfigure_calls = 0

    def reconfigure_selector(self, _config, universe_frame=None) -> None:
        self.reconfigure_calls += 1

    def run_multi(
        self,
        day,
        _portfolio,
        _config,
        _market_snapshot,
        _prices,
        _technical_context,
    ) -> MultiGraphResult:
        targets = {
            0: {"AAA": 20},
            1: {"BBB": 10},
            2: {"AAA": 5, "BBB": 10},
        }.get(day, {"AAA": 5, "BBB": 10})
        return MultiGraphResult(
            sector_view={"source": "architecture-parity", "day": day},
            target_positions=targets,
            trace=[{"tool": "scripted_multi_graph", "detail": f"day={day}"}],
        )


class _FixedSnapshotService:
    def __init__(self, snapshot: UniverseSnapshot) -> None:
        self.snapshot = snapshot
        self.calls = 0

    def capture(self, _config: BacktestConfig) -> UniverseSnapshot:
        self.calls += 1
        return self.snapshot


class _FailingSnapshotService:
    def capture(self, _config: BacktestConfig) -> UniverseSnapshot:
        raise AssertionError("fixed snapshot replay must not capture live universe data")


def _snapshot() -> UniverseSnapshot:
    frame = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "name": "Alpha",
                "close": 10.0,
                "change_pct": 0.01,
                "volume": 1_000_000,
                "turnover": 100_000_000.0,
                "sector": "Sector A",
            },
            {
                "symbol": "BBB",
                "name": "Beta",
                "close": 20.0,
                "change_pct": -0.01,
                "volume": 800_000,
                "turnover": 90_000_000.0,
                "sector": "Sector B",
            },
        ]
    )
    return UniverseSnapshot.from_frame(
        frame,
        source="test:architecture-parity",
        as_of="2026-09-24",
        captured_at="2026-10-13T11:38:00+00:00",
        parameters={"purpose": "A-07 parity"},
    )


def _observable_result(state: dict) -> dict:
    """Select public behavior while excluding collaborator call counters."""
    return {
        "day": state["day"],
        "total_days": state["total_days"],
        "portfolio": state["portfolio"],
        "equity_history": state["equity_history"],
        "trades": state["trades"],
        "report": state["report"],
        "sector_view": state["sector_view"],
        "universe_snapshot": state["universe_snapshot"],
    }


def _imported_modules(relative_path: str) -> set[str]:
    tree = ast.parse((ROOT / relative_path).read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


class ArchitectureParityTest(unittest.TestCase):
    @staticmethod
    def _config(**updates) -> BacktestConfig:
        values = {
            "provider": "synthetic",
            "agent_mode": "offline",
            "dataset": "trend",
            "symbol": "BENCH",
            "universe": ["AAA", "BBB"],
            "days": 6,
            "rebalance_frequency": 1,
            "initial_cash": 100_000.0,
            "fee_rate": 0.0003,
            "slippage_bps": 2.0,
        }
        values.update(updates)
        return BacktestConfig(**values)

    def test_default_and_injected_provider_have_identical_observable_results(self) -> None:
        config = self._config()
        default_engine = TradingBacktestEngine(config, graph=_ScriptedMultiGraph())
        injected_engine = TradingBacktestEngine(
            config,
            provider=SyntheticDataProvider(total_days=config.days),
            graph=_ScriptedMultiGraph(),
        )

        default_state = default_engine.run_to_end()
        injected_state = injected_engine.run_to_end()

        self.assertEqual(
            _observable_result(default_state),
            _observable_result(injected_state),
        )
        self.assertGreater(len(default_state["trades"]), 0)
        self.assertEqual(
            [
                (trade["symbol"], trade["side"], trade["shares"])
                for trade in default_state["trades"]
            ],
            [
                (trade["symbol"], trade["side"], trade["shares"])
                for trade in injected_state["trades"]
            ],
        )

    def test_captured_and_replayed_universe_have_identical_observable_results(self) -> None:
        snapshot = _snapshot()
        service = _FixedSnapshotService(snapshot)
        config = self._config(universe=[], dynamic_universe=True)
        captured_engine = TradingBacktestEngine(
            config,
            provider=SyntheticDataProvider(total_days=config.days),
            graph=_ScriptedMultiGraph(),
            universe_service=service,
        )
        replayed_engine = TradingBacktestEngine(
            config,
            provider=SyntheticDataProvider(total_days=config.days),
            graph=_ScriptedMultiGraph(),
            universe_service=_FailingSnapshotService(),
            universe_snapshot=snapshot,
        )

        captured_state = captured_engine.run_to_end()
        replayed_state = replayed_engine.run_to_end()

        self.assertEqual(service.calls, 1)
        self.assertEqual(captured_engine.effective_universe, ["AAA", "BBB"])
        self.assertEqual(replayed_engine.effective_universe, ["AAA", "BBB"])
        self.assertEqual(
            _observable_result(captured_state),
            _observable_result(replayed_state),
        )


class ArchitectureDependencyBoundaryTest(unittest.TestCase):
    def test_core_contracts_do_not_depend_on_implementation_packages(self) -> None:
        modules = _imported_modules("core/contracts.py")
        forbidden_roots = {"agents", "data", "engine", "tools"}

        self.assertFalse({name.split(".")[0] for name in modules} & forbidden_roots)

    def test_snapshot_service_does_not_depend_on_engine_agents_or_tools(self) -> None:
        modules = _imported_modules("data/universe_service.py")
        forbidden_roots = {"agents", "engine", "tools"}

        self.assertFalse({name.split(".")[0] for name in modules} & forbidden_roots)

    def test_rebalance_planner_isolated_from_graph_data_and_broker(self) -> None:
        modules = _imported_modules("engine/rebalance.py")

        self.assertFalse(
            {name.split(".")[0] for name in modules} & {"agents", "data"}
        )
        self.assertNotIn("tools.broker", modules)

    def test_engine_delegates_rebalance_without_importing_sizing_details(self) -> None:
        modules = _imported_modules("engine/backtest.py")

        self.assertIn("engine.rebalance", modules)
        self.assertNotIn("tools.sizing", modules)


if __name__ == "__main__":
    unittest.main()
