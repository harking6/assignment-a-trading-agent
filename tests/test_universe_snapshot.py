"""Persistence and offline replay tests for dynamic-universe snapshots."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.schemas import BacktestConfig  # noqa: E402
from data.providers import SyntheticDataProvider  # noqa: E402
from data.universe_service import UniverseSnapshot  # noqa: E402
from engine.backtest import TradingBacktestEngine  # noqa: E402


def _universe_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "symbol": "000001",
                "name": "平安银行",
                "close": 10.5,
                "change_pct": 0.01,
                "volume": 1_000_000,
                "turnover": 120_000_000.0,
                "sector": "银行",
            },
            {
                "symbol": "300750",
                "name": "宁德时代",
                "close": 210.0,
                "change_pct": -0.005,
                "volume": 800_000,
                "turnover": 180_000_000.0,
                "sector": "电力设备",
            },
        ]
    )


def _snapshot(captured_at: str = "2026-10-12T06:00:00+00:00") -> UniverseSnapshot:
    return UniverseSnapshot.from_frame(
        _universe_frame(),
        source="test:fixed",
        as_of="2026-09-24",
        captured_at=captured_at,
        parameters={"dynamic_universe_limit": 30, "dynamic_min_turnover": 50_000_000.0},
    )


class _FailIfCaptured:
    def __init__(self) -> None:
        self.calls = 0

    def capture(self, _config):
        self.calls += 1
        raise AssertionError("live universe capture must not run during snapshot replay")


class UniverseSnapshotTest(unittest.TestCase):
    def test_content_id_is_stable_across_capture_times(self) -> None:
        first = _snapshot("2026-10-12T06:00:00+00:00")
        second = _snapshot("2026-10-12T07:00:00+00:00")

        self.assertEqual(first.snapshot_id, second.snapshot_id)
        self.assertNotEqual(first.captured_at, second.captured_at)
        self.assertEqual(first.summary()["as_of"], "2026-09-24")
        self.assertEqual(first.summary()["symbol_count"], 2)

    def test_snapshot_round_trip_preserves_frame_and_metadata(self) -> None:
        snapshot = _snapshot()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = snapshot.save(Path(tmpdir) / "universe.json")
            restored = UniverseSnapshot.load(path)

        self.assertEqual(restored.snapshot_id, snapshot.snapshot_id)
        self.assertEqual(restored.summary(), snapshot.summary())
        pd.testing.assert_frame_equal(restored.to_frame(), snapshot.to_frame())

    def test_snapshot_load_rejects_tampered_records(self) -> None:
        snapshot = _snapshot()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = snapshot.save(Path(tmpdir) / "universe.json")
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["records"][0]["symbol"] = "999999"
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                UniverseSnapshot.load(path)

    def test_engine_replays_injected_snapshot_without_live_capture(self) -> None:
        snapshot = _snapshot()
        service = _FailIfCaptured()
        config = BacktestConfig(
            provider="synthetic",
            symbol="BENCH",
            agent_mode="offline",
            days=5,
            dynamic_universe=True,
            rebalance_frequency=1,
        )
        provider = SyntheticDataProvider(total_days=5)

        engine = TradingBacktestEngine(
            config,
            provider=provider,
            universe_service=service,
            universe_snapshot=snapshot,
        )
        engine.dynamic_universe_frame.loc[0, "sector"] = "mutated in one run"
        engine.reset(config)
        state = engine.to_dict()

        self.assertEqual(service.calls, 0)
        self.assertIs(engine.provider, provider)
        self.assertEqual(engine.effective_universe, ["000001", "300750"])
        self.assertEqual(engine.dynamic_universe_frame.loc[0, "sector"], "银行")
        self.assertEqual(state["universe_snapshot"]["snapshot_id"], snapshot.snapshot_id)
        self.assertEqual(state["universe_snapshot"]["as_of"], "2026-09-24")
        self.assertEqual(
            state["report"]["universe_snapshot"]["snapshot_id"],
            snapshot.snapshot_id,
        )


if __name__ == "__main__":
    unittest.main()
