"""Tests for the experience-learning gate and recall ranking.

The 924 debugging taught us: injecting experience from 1-2 back-filled
records is noise — a single ``bullish → -7.8%`` spooked the LLM into
neutral at the melt-up top. The graph gates injection behind ``len(recalled)
>= 3``. These tests pin the recall ranking invariants that gate relies on:
same-regime records rank closer, only back-filled records are eligible,
and the threshold logic itself.
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.memory import TradingMemory, _feature_vector, _normalize, _euclidean  # noqa: E402


def _kline(rsi=50, vol=0.2, breadth=0.5, dist=0.0):
    return {"market": {"avg_rsi": rsi, "avg_volatility": vol, "breadth_up": breadth, "avg_dist_sma20": dist}}


def _record(day, market_view, kline, active_return, sectors=None):
    return {
        "scope": "multi",
        "day": day,
        "market_view": market_view,
        "kline_features": kline,
        "selected_sectors": sectors or [],
        "active_return": active_return,
    }


class RecallEligibilityTest(unittest.TestCase):
    """Only multi-scope records with a back-filled active_return are recalled."""

    def setUp(self) -> None:
        self.tmp = Path(__file__).resolve().parent / "_tmp_gate.jsonl"
        if self.tmp.exists():
            self.tmp.unlink()
        self.mem = TradingMemory(path=self.tmp)

    def tearDown(self) -> None:
        if self.tmp.exists():
            self.tmp.unlink()

    def test_unfilled_record_is_not_recalled(self) -> None:
        import json
        # Write a multi record with active_return=None directly (mirrors what
        # append_multi produces before record_outcome back-fills it).
        with open(self.tmp, "a") as f:
            f.write(json.dumps({"scope": "multi", "day": 0, "market_view": "bullish",
                                "kline_features": _kline(), "selected_sectors": ["电子"],
                                "active_return": None}) + "\n")
        recalled = self.mem.recall_similar(_kline(), "bullish", limit=3)
        self.assertEqual(recalled, [])

    def test_single_symbol_record_is_not_recalled(self) -> None:
        # A single-symbol record (no scope=multi) must not leak into multi recall
        import json
        with open(self.tmp, "a") as f:
            f.write(json.dumps({"symbol": "000001", "scope": None, "active_return": 0.05,
                                "kline_features": _kline()}) + "\n")
        recalled = self.mem.recall_similar(_kline(), "bullish", limit=3)
        self.assertEqual(recalled, [])


class RecallRankingTest(unittest.TestCase):
    """Same-regime bias: a bullish query ranks bullish records ahead of bearish
    ones even when the feature distance is identical."""

    def setUp(self) -> None:
        self.tmp = Path(__file__).resolve().parent / "_tmp_rank.jsonl"
        if self.tmp.exists():
            self.tmp.unlink()
        self.mem = TradingMemory(path=self.tmp)

    def tearDown(self) -> None:
        if self.tmp.exists():
            self.tmp.unlink()

    def _fill(self, records) -> None:
        import json
        with open(self.tmp, "a") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def test_same_regime_ranks_ahead_of_cross_regime(self) -> None:
        # Query RSI 50; both records RSI 60 (identical, NON-ZERO distance).
        # The same-regime bias (×0.5) must make the matching regime win.
        kline_q = _kline(rsi=50)
        kline_rec = _kline(rsi=60)   # dist > 0 so the ×0.5 bias is meaningful
        self._fill([
            _record(1, "bearish", kline_rec, -0.05, ["银行"]),
            _record(2, "bullish", kline_rec, +0.08, ["电子"]),
        ])
        # query is bullish → bullish record dist ×0.5 < bearish dist
        recalled = self.mem.recall_similar(kline_q, "bullish", limit=2)
        self.assertEqual(len(recalled), 2)
        self.assertEqual(recalled[0]["market_view"], "bullish")
        self.assertEqual(recalled[1]["market_view"], "bearish")

    def test_limit_truncates_to_closest_n(self) -> None:
        kline = _kline(rsi=50)
        # 5 identical-regime records, all eligible
        self._fill([_record(i, "neutral", _kline(rsi=50 + i), 0.01 * i) for i in range(5)])
        recalled = self.mem.recall_similar(kline, "neutral", limit=3)
        self.assertEqual(len(recalled), 3)

    def test_nearest_feature_vector_wins_regardless_of_regime(self) -> None:
        # query RSI 50; record A RSI 50 (close), record B RSI 80 (far)
        # both bullish → pure feature distance decides, A wins
        self._fill([
            _record(1, "bullish", _kline(rsi=80), 0.02, ["电子"]),
            _record(2, "bullish", _kline(rsi=50), 0.03, ["通信"]),
        ])
        recalled = self.mem.recall_similar(_kline(rsi=50), "bullish", limit=2)
        self.assertEqual(recalled[0]["day"], 2)  # RSI 50 → nearest


class GateThresholdTest(unittest.TestCase):
    """The graph-layer gate: inject lesson only when len(recalled) >= 3.

    This mirrors agents/graph.py:681 ``if len(recalled) >= 3: lesson = ...``.
    We replicate the threshold decision here so a regression in the count
    (e.g. recall returning unfilled records) would flip the gate.
    """

    def _gate_decision(self, recalled):
        """Replica of the graph's gate: returns lesson string or empty."""
        if len(recalled) >= 3:
            return TradingMemory().to_lesson(recalled)
        return ""

    def setUp(self) -> None:
        self.tmp = Path(__file__).resolve().parent / "_tmp_thresh.jsonl"
        if self.tmp.exists():
            self.tmp.unlink()
        self.mem = TradingMemory(path=self.tmp)

    def tearDown(self) -> None:
        if self.tmp.exists():
            self.tmp.unlink()

    def test_two_records_do_not_trigger_injection(self) -> None:
        import json
        for i in range(2):
            with open(self.tmp, "a") as f:
                f.write(json.dumps(_record(i, "bullish", _kline(), 0.05)) + "\n")
        recalled = self.mem.recall_similar(_kline(), "bullish", limit=3)
        self.assertEqual(len(recalled), 2)
        # Gate blocks injection below 3 — the 924 bug root cause
        self.assertEqual(self._gate_decision(recalled), "")

    def test_three_records_trigger_injection(self) -> None:
        import json
        for i in range(3):
            with open(self.tmp, "a") as f:
                f.write(json.dumps(_record(i, "bullish", _kline(), 0.05 * (i + 1))) + "\n")
        recalled = self.mem.recall_similar(_kline(), "bullish", limit=3)
        self.assertEqual(len(recalled), 3)
        lesson = self._gate_decision(recalled)
        self.assertNotEqual(lesson, "")
        self.assertIn("区间超额", lesson)

    def test_injected_lesson_contains_outcome_sign(self) -> None:
        """A positive active_return must surface as +x% in the lesson text,
        so the LLM can tell winning from losing historical calls."""
        import json
        records = [_record(0, "bullish", _kline(), 0.082, ["电子"])]
        records.append(_record(1, "bullish", _kline(), -0.064, ["通信"]))
        records.append(_record(2, "bullish", _kline(), 0.031, ["电力设备"]))
        with open(self.tmp, "a") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        recalled = self.mem.recall_similar(_kline(), "bullish", limit=3)
        lesson = self._gate_decision(recalled)
        self.assertIn("+8.20%", lesson)
        self.assertIn("-6.40%", lesson)


class FeatureVectorTest(unittest.TestCase):
    """The 4-dim K-line feature vector used for similarity matching."""

    def test_extracts_four_market_features(self) -> None:
        vec = _feature_vector(_kline(rsi=55, vol=0.3, breadth=0.7, dist=0.05))
        self.assertEqual(vec, [55.0, 0.3, 0.7, 0.05])

    def test_none_features_uses_neutral_defaults(self) -> None:
        vec = _feature_vector(None)
        # RSI 50 (neutral), vol 0.2, breadth 0.5, dist 0.0
        self.assertEqual(vec, [50.0, 0.2, 0.5, 0.0])

    def test_normalize_scales_rsi_to_unit_range(self) -> None:
        # RSI 0-100 → 0-1; other components unchanged
        n = _normalize([50.0, 0.2, 0.5, 0.0])
        self.assertAlmostEqual(n[0], 0.5)
        self.assertEqual(n[1], 0.2)

    def test_euclidean_distance_zero_for_identical_vectors(self) -> None:
        self.assertEqual(_euclidean([1, 2, 3], [1, 2, 3]), 0.0)


if __name__ == "__main__":
    unittest.main()
