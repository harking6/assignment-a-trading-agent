"""Experience-learning layer tests: memory recall loop, sentiment, knowledge base.

Isolated from the real data/agent_memory.jsonl (tmpdir) so repeated runs do not
leak multi-symbol decisions into each other or into single-symbol contract tests.
"""
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.memory import TradingMemory  # noqa: E402
from agents.top_down import TopDownSelector  # noqa: E402
from tools.news import MultiSentimentFeed  # noqa: E402
from tools.rag import MultiKnowledgeBase  # noqa: E402


def _multi_record(
    day: int,
    *,
    rsi: float,
    vol: float,
    breadth: float,
    dist: float,
    market_view: str,
    sectors,
    active: float | None,
) -> dict:
    return {
        "type": "decision",
        "scope": "multi",
        "day": day,
        "date": f"2024-09-{day:02d}",
        "market_view": market_view,
        "benchmark_return": 0.05,
        "selected_sectors": list(sectors),
        "target_positions": {"000001": 100},
        "llm_reasoning": "test",
        "llm_trace": {},
        "kline_features": {
            "market": {
                "avg_rsi": rsi,
                "avg_volatility": vol,
                "breadth_up": breadth,
                "avg_dist_sma20": dist,
            }
        },
        "sentiment": {},
        "active_return": active,
    }


class MemoryRecallLoopTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.memory = TradingMemory(Path(self.tmp.name) / "mem.jsonl")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_recall_returns_nothing_when_no_outcomes(self) -> None:
        # Decisions without a back-filled outcome must not be recalled — they
        # have not been learned from yet.
        self.memory.append(_multi_record(1, rsi=50, vol=0.2, breadth=0.5, dist=0.0, market_view="bullish", sectors=["电子"], active=None))
        self.assertEqual(self.memory.recall_similar({"market": {"avg_rsi": 50, "avg_volatility": 0.2, "breadth_up": 0.5, "avg_dist_sma20": 0.0}}, "bullish"), [])

    def test_record_outcome_backfills_and_enables_recall(self) -> None:
        self.memory.append(_multi_record(1, rsi=50, vol=0.2, breadth=0.5, dist=0.0, market_view="bullish", sectors=["电子"], active=None))
        self.memory.record_outcome(1, 0.021)
        recalled = self.memory.recall_similar({"market": {"avg_rsi": 50, "avg_volatility": 0.2, "breadth_up": 0.5, "avg_dist_sma20": 0.0}}, "bullish")
        self.assertEqual(len(recalled), 1)
        self.assertAlmostEqual(recalled[0]["active_return"], 0.021)

    def test_same_regime_nearest_neighbour_wins(self) -> None:
        # Two decisions with IDENTICAL market form but different regimes, both
        # filled. The same-regime bias (0.5 distance multiplier) must surface
        # the bullish record first when the query regime is bullish, even
        # though the raw form distance is identical.
        self.memory.append(_multi_record(2, rsi=55, vol=0.2, breadth=0.6, dist=0.01, market_view="bearish", sectors=["银行"], active=-0.01))
        self.memory.append(_multi_record(5, rsi=55, vol=0.2, breadth=0.6, dist=0.01, market_view="bullish", sectors=["电子"], active=0.03))
        recalled = self.memory.recall_similar({"market": {"avg_rsi": 52, "avg_volatility": 0.2, "breadth_up": 0.6, "avg_dist_sma20": 0.01}}, "bullish")
        self.assertEqual(recalled[0]["day"], 5)

    def test_to_lesson_contains_result(self) -> None:
        self.memory.append(_multi_record(1, rsi=50, vol=0.2, breadth=0.5, dist=0.0, market_view="bullish", sectors=["电子"], active=0.021))
        self.memory.record_outcome(1, 0.021)
        recalled = self.memory.recall_similar({"market": {"avg_rsi": 50, "avg_volatility": 0.2, "breadth_up": 0.5, "avg_dist_sma20": 0.0}}, "bullish")
        lesson = self.memory.to_lesson(recalled)
        self.assertIn("电子", lesson)
        self.assertIn("+2.10%", lesson)

    def test_record_outcome_idempotent(self) -> None:
        self.memory.append(_multi_record(1, rsi=50, vol=0.2, breadth=0.5, dist=0.0, market_view="bullish", sectors=["电子"], active=None))
        self.memory.record_outcome(1, 0.021)
        self.memory.record_outcome(1, 0.099)  # second call must not overwrite
        recalled = self.memory.recall_similar({"market": {"avg_rsi": 50, "avg_volatility": 0.2, "breadth_up": 0.5, "avg_dist_sma20": 0.0}}, "bullish")
        self.assertAlmostEqual(recalled[0]["active_return"], 0.021)


class KlineFeaturesTest(unittest.TestCase):
    def test_market_features_aggregate(self) -> None:
        candidates = pd.DataFrame(
            {
                "symbol": ["000001", "000002", "000003"],
                "rsi_14": [40.0, 50.0, 60.0],
                "volatility_20": [0.10, 0.20, 0.30],
                "change_pct": [-0.01, 0.0, 0.02],
                "dist_sma_20": [-0.02, 0.0, 0.02],
            }
        )
        feats = TopDownSelector._market_kline_features(candidates)
        market = feats["market"]
        self.assertAlmostEqual(market["avg_rsi"], 50.0)
        self.assertAlmostEqual(market["avg_volatility"], 0.20)
        self.assertAlmostEqual(market["breadth_up"], 1 / 3)
        self.assertAlmostEqual(market["avg_dist_sma20"], 0.0)

    def test_empty_candidates_safe(self) -> None:
        feats = TopDownSelector._market_kline_features(pd.DataFrame())
        self.assertEqual(feats, {"market": {}})


class MultiSentimentFeedTest(unittest.TestCase):
    def test_snapshot_market_and_per_stock(self) -> None:
        candidates = pd.DataFrame(
            {
                "symbol": ["000001", "000002"],
                "change_pct": [0.03, -0.03],
                "volume_z": [1.5, -0.5],
            }
        )
        tech = {
            "000001": {"return_1d": 0.03, "volume_z": 1.5},
            "000002": {"return_1d": -0.03, "volume_z": -0.5},
        }
        snap = MultiSentimentFeed().snapshot(candidates, tech)
        self.assertIn("market_score", snap)
        self.assertIn("per_stock", snap)
        self.assertGreater(snap["per_stock"]["000001"], 0)
        self.assertLess(snap["per_stock"]["000002"], 0)
        self.assertTrue(snap["top_events"])

    def test_empty_candidates_neutral(self) -> None:
        snap = MultiSentimentFeed().snapshot(None, None)
        self.assertEqual(snap["market_score"], 0.0)
        self.assertEqual(snap["per_stock"], {})


class MultiKnowledgeBaseTest(unittest.TestCase):
    def test_recall_returns_industry_and_book_context(self) -> None:
        kb = MultiKnowledgeBase()
        result = kb.recall(["电子", "银行"], market_view="bullish", k=4)
        self.assertIn("context", result)
        self.assertGreater(result["document_count"], 0)
        # Industry notes for the selected sectors are always included as a
        # structural prior, so their sources must appear regardless of book
        # lexical scores.
        sectors_in_sources = {s.get("sector", "") for s in result["sources"]}
        self.assertIn("电子", sectors_in_sources)
        self.assertIn("银行", sectors_in_sources)


if __name__ == "__main__":
    unittest.main()
