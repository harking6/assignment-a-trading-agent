"""Regression tests for the executable Financial-LLM Five-Sins audit."""

import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.memory import TradingMemory  # noqa: E402
from agents.top_down import TopDownSelector  # noqa: E402
from core.schemas import BacktestConfig  # noqa: E402
from tools.financial_llm_bias_audit import FinancialLLMBiasAuditor, NA, PASS, WARN  # noqa: E402


class FinancialLLMBiasAuditTest(unittest.TestCase):
    def test_target_must_be_in_factor_evidence_set(self) -> None:
        auditor = FinancialLLMBiasAuditor()
        entry = auditor.record_rebalance(
            day=10,
            sector_view={
                "selected_stocks": [{"symbol": "AAA"}],
                "aggregate_weight_reasoning": "AAA has stronger trend.",
            },
            target_positions={"ZZZ": 100},
        )
        self.assertEqual(entry["narrative"]["status"], "fail")
        self.assertEqual(entry["narrative"]["evidence"]["unsupported_target_symbols"], ["ZZZ"])

    def test_static_universe_does_not_claim_survivorship_proof(self) -> None:
        result = FinancialLLMBiasAuditor().summary(
            config=BacktestConfig(universe=["AAA"]),
            universe_snapshot=None,
            total_fees=1.0,
            turnover=2.0,
            llm_call_count=3,
            llm_latency_seconds=0.5,
        )
        self.assertEqual(result["checks"]["survivorship"]["status"], NA)
        self.assertEqual(result["checks"]["cost"]["evidence"]["total_fees"], 1.0)

    def test_snapshot_without_as_of_is_warning(self) -> None:
        result = FinancialLLMBiasAuditor().summary(
            config=BacktestConfig(dynamic_universe=True, provider="akshare"),
            universe_snapshot={"snapshot_id": "x", "source": "spot", "symbol_count": 10},
            total_fees=0.0,
            turnover=0.0,
            llm_call_count=0,
            llm_latency_seconds=0.0,
        )
        self.assertEqual(result["checks"]["survivorship"]["status"], WARN)

    def test_recall_audit_counts_future_outcomes_as_excluded(self) -> None:
        memory = TradingMemory(path=ROOT / ".tmp_test_bias_audit_memory.json")
        try:
            memory._rewrite_all([
                {"scope": "multi", "active_return": 0.01, "outcome_day": 5},
                {"scope": "multi", "active_return": 0.02, "outcome_day": 3},
            ])
            audit = memory.recall_temporal_audit(5)
            self.assertEqual(audit["future_outcomes_rejected"], 1)
            self.assertEqual(audit["eligible"], 1)
        finally:
            memory.path.unlink(missing_ok=True)

    def test_empty_target_is_calibrated_abstention(self) -> None:
        auditor = FinancialLLMBiasAuditor()
        row = auditor.record_rebalance(day=3, sector_view={"selected_stocks": []}, target_positions={})
        self.assertEqual(row["objective"]["status"], PASS)
        self.assertTrue(row["objective"]["evidence"]["abstained"])

    def test_explicit_all_cash_plan_does_not_fall_back_to_positions(self) -> None:
        selector = TopDownSelector(BacktestConfig(agent_mode="offline"))
        selected = pd.DataFrame([{"symbol": "AAA", "target_weight": 1.0}])
        output = selector.apply_portfolio_plan(
            selected,
            {"weights": {"AAA": 0.0}, "cash_ratio": 1.0, "abstain": True},
        )
        self.assertTrue(output.empty)


if __name__ == "__main__":
    unittest.main()
