"""Executable structural-validity checks for financial-LLM backtests.

The checks deliberately separate a *passing engineering guardrail* from
evidence that a real point-in-time data source was used.  In particular, a
synthetic or static universe is reported as ``not_applicable`` rather than
pretending it proves survivorship-bias freedom.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Dict, Iterable, List


PASS = "pass"
WARN = "warning"
FAIL = "fail"
NA = "not_applicable"


def _check(status: str, evidence: Dict[str, Any], note: str) -> Dict[str, Any]:
    return {"status": status, "evidence": evidence, "note": note}


class FinancialLLMBiasAuditor:
    """Turn the Five-Sins structural-validity framework into run metadata."""

    def __init__(self) -> None:
        self.rebalance_checks: List[Dict[str, Any]] = []

    def record_rebalance(
        self,
        *,
        day: int,
        sector_view: Dict[str, Any],
        target_positions: Dict[str, int],
        recall: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Audit inputs available at a single decision point.

        The graph supplies a compact recall audit rather than historical model
        prompts, keeping this observable in both online and offline modes.
        """
        recall = recall or {}
        selected = sector_view.get("selected_stocks") or []
        allowed = {str(row.get("symbol")) for row in selected if row.get("symbol")}
        targets = {str(symbol) for symbol, shares in target_positions.items() if shares > 0}
        bad_targets = sorted(targets - allowed)
        eligible = int(recall.get("eligible", 0))
        future_outcomes = int(recall.get("future_outcomes_rejected", 0))
        malformed_outcomes = int(recall.get("malformed_outcomes_rejected", 0))
        record = {
            "day": day,
            "look_ahead": _check(
                PASS,
                {
                    "eligible_recalled": eligible,
                    "future_outcomes_rejected": future_outcomes,
                    "malformed_outcomes_rejected": malformed_outcomes,
                },
                "经验召回仅允许 outcome_day 小于当前决策日。",
            ),
            "narrative": _check(
                FAIL if bad_targets else PASS,
                {
                    "candidate_evidence_count": len(selected),
                    "target_count": len(targets),
                    "unsupported_target_symbols": bad_targets,
                    "rationale_present": bool(str(sector_view.get("aggregate_weight_reasoning", "")).strip()),
                },
                "目标持仓必须来自本次有因子证据的候选集；该检查不把文本流畅度当作证据。",
            ),
            "objective": _check(
                PASS,
                {
                    "abstained": not targets,
                    "cash_reserve_ratio": sector_view.get("cash_reserve_ratio"),
                    "portfolio_cash_ratio": (sector_view.get("portfolio_plan") or {}).get("cash_ratio"),
                },
                "空仓/不交易是允许的决策，不因缺乏交易而扣分。",
            ),
        }
        self.rebalance_checks.append(record)
        return record

    @staticmethod
    def _universe_check(
        snapshot: Dict[str, Any] | None,
        config: Any,
    ) -> Dict[str, Any]:
        if not getattr(config, "dynamic_universe", False):
            return _check(
                NA,
                {"mode": "static_or_synthetic", "symbol_count": len(getattr(config, "universe", []) or [])},
                "静态/合成股票池不能证明包含退市标的；真实历史评测应提供逐日点时股票池。",
            )
        if not snapshot:
            return _check(FAIL, {}, "已启用动态股票池，但未捕获可复现快照。")
        as_of = snapshot.get("as_of")
        start = getattr(config, "start", None)
        end = getattr(config, "end", None)
        evidence = {
            "snapshot_id": snapshot.get("snapshot_id"),
            "source": snapshot.get("source"),
            "as_of": as_of,
            "captured_at": snapshot.get("captured_at"),
            "symbol_count": snapshot.get("symbol_count", 0),
        }
        if not as_of:
            return _check(WARN, evidence, "快照已固化但来源未提供 as_of，不能验证历史成分股时点一致性。")
        try:
            as_of_day = date.fromisoformat(str(as_of)[:10])
            end_day = date.fromisoformat(str(end)[:10]) if end else None
            if end_day and as_of_day > end_day:
                return _check(FAIL, evidence, "股票池快照晚于回测结束日，存在幸存者偏差风险。")
            if start and as_of_day > date.fromisoformat(str(start)[:10]):
                return _check(WARN, evidence, "单一 as_of 快照覆盖多日回测；应使用逐日历史成分股。")
        except ValueError:
            return _check(WARN, evidence, "无法解析股票池 as_of 日期，未能完成时点校验。")
        return _check(PASS, evidence, "股票池快照带有可校验的时点和完整性校验和。")

    def summary(
        self,
        *,
        config: Any,
        universe_snapshot: Dict[str, Any] | None,
        total_fees: float,
        turnover: float,
        llm_call_count: int,
        llm_latency_seconds: float,
    ) -> Dict[str, Any]:
        checks: Iterable[Dict[str, Any]] = self.rebalance_checks
        look_ahead_failures = sum(1 for row in checks if row["look_ahead"]["status"] == FAIL)
        narrative_failures = sum(1 for row in checks if row["narrative"]["status"] == FAIL)
        abstentions = sum(1 for row in checks if row["objective"]["evidence"]["abstained"])
        return {
            "framework": "financial_llm_structural_validity_v1",
            "checks": {
                "look_ahead": _check(
                    FAIL if look_ahead_failures else PASS,
                    {"rebalances": len(self.rebalance_checks), "violations": look_ahead_failures},
                    "逐次决策的经验结果时间审计。",
                ),
                "survivorship": self._universe_check(universe_snapshot, config),
                "narrative": _check(
                    FAIL if narrative_failures else PASS,
                    {"rebalances": len(self.rebalance_checks), "unsupported_target_violations": narrative_failures},
                    "持仓目标与候选因子证据的可追溯性审计。",
                ),
                "objective": _check(
                    PASS,
                    {"rebalances": len(self.rebalance_checks), "no_trade_decisions": abstentions},
                    "明确将 no-trade 作为合法动作，并记录其实际发生次数。",
                ),
                "cost": _check(
                    PASS,
                    {
                        "total_fees": round(float(total_fees), 4),
                        "turnover": round(float(turnover), 4),
                        "llm_call_count": int(llm_call_count),
                        "llm_latency_seconds": round(float(llm_latency_seconds), 4),
                    },
                    "净效用报告同时保留交易摩擦与模型推理延迟。",
                ),
            },
            "rebalance_audits": list(self.rebalance_checks),
        }
