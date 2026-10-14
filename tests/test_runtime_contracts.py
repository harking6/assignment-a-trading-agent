"""Contract tests for member A's runtime service boundaries."""

import ast
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.graph import TradingAgentsGraph  # noqa: E402
from core.contracts import GraphRunner, MarketDataSource, UniverseSource  # noqa: E402
from data.providers import SyntheticDataProvider  # noqa: E402
from data.universe import AShareUniverseProvider  # noqa: E402


class _OfflineUniverseSource:
    """Network-free structural implementation used only for protocol checks."""

    def list_spot(self):
        return []

    def list_sectors(self):
        return []

    def list_sector_stocks(self, _sector: str):
        return []


class _IncompleteService:
    def load(self):
        return []


class RuntimeServiceContractTest(unittest.TestCase):
    def test_current_graph_implements_graph_runner_without_adapter(self) -> None:
        self.assertIsInstance(TradingAgentsGraph(), GraphRunner)

    def test_current_synthetic_provider_implements_market_data_source(self) -> None:
        self.assertIsInstance(SyntheticDataProvider(total_days=2), MarketDataSource)

    def test_structural_universe_source_needs_no_inheritance(self) -> None:
        self.assertIsInstance(_OfflineUniverseSource(), UniverseSource)

    def test_current_universe_provider_implements_contract_without_network(self) -> None:
        provider = object.__new__(AShareUniverseProvider)
        self.assertIsInstance(provider, UniverseSource)

    def test_incomplete_objects_do_not_satisfy_service_contracts(self) -> None:
        service = _IncompleteService()
        self.assertNotIsInstance(service, GraphRunner)
        self.assertNotIsInstance(service, MarketDataSource)
        self.assertNotIsInstance(service, UniverseSource)

    def test_core_contracts_do_not_import_concrete_runtime_layers(self) -> None:
        source = (ROOT / "core" / "contracts.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported_roots = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".", 1)[0])

        self.assertTrue({"agents", "data", "engine", "tools"}.isdisjoint(imported_roots))


if __name__ == "__main__":
    unittest.main()
