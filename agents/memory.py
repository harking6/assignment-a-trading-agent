from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from langchain_core.chat_history import InMemoryChatMessageHistory


_MARKET_FEATURES = (
    "avg_rsi",
    "avg_volatility",
    "breadth_up",
    "avg_dist_sma20",
)
_QUALITY_THRESHOLD = 0.65
_HALF_LIFE_DAYS = 20.0
_SIMILARITY_TAU = 0.35
_CROSS_REGIME_FACTOR = 0.45


class TradingMemory:
    """Append-only decision memory used by later agent steps.

    The store holds both single-symbol records (``symbol`` set, no ``scope``)
    and multi-symbol portfolio decisions (``scope="multi"``). Single-symbol
    recall (``recent``/``chat_history``) filters by ``symbol`` and ignores the
    multi records naturally; multi-symbol recall (``recall_similar``) reads
    only ``scope="multi"`` records whose outcome has been back-filled, so the
    agent only learns from decisions it has already seen the result of.
    """

    def __init__(self, path: Path | None = None) -> None:
        root = Path(__file__).resolve().parents[1]
        configured_path = os.environ.get("TRADING_AGENT_MEMORY_PATH")
        self.path = path or (Path(configured_path) if configured_path else root / "data" / "agent_memory.jsonl")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.last_error: str | None = None

    def recent(self, symbol: str, limit: int = 6) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        rows: List[Dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                item = json.loads(line)
            except Exception:
                continue
            if item.get("symbol") == symbol:
                rows.append(item)
        return rows[-max(0, limit) :]

    def chat_history(self, symbol: str, limit: int = 6) -> InMemoryChatMessageHistory:
        """Convert durable JSONL decisions into LangChain chat messages.

        Create an ``InMemoryChatMessageHistory``. For each item returned by
        ``recent(symbol, limit)``, append one human message describing the
        prior situation and one AI message describing the decision and risk
        result. Preserve chronological order and return the history object.
        """

        history = InMemoryChatMessageHistory()
        for item in self.recent(symbol, limit):
            situation = {
                key: item.get(key)
                for key in ("symbol", "day", "date", "price", "portfolio", "market")
                if item.get(key) is not None
            }
            decision = {
                key: item.get(key)
                for key in ("side", "shares", "signal", "confidence", "reason", "risk", "blocked")
                if item.get(key) is not None
            }
            history.add_user_message(
                "Prior trading situation: "
                + json.dumps(situation, ensure_ascii=False, sort_keys=True)
            )
            history.add_ai_message(
                "Prior trading decision and risk result: "
                + json.dumps(decision, ensure_ascii=False, sort_keys=True)
            )
        return history

    def append(self, item: Dict[str, Any]) -> None:
        try:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
            self.last_error = None
        except Exception as exc:
            self.last_error = str(exc)

    # ------------------------------------------------------------------
    # Multi-symbol experience-learning loop (decision → outcome → recall).
    # These are additive: single-symbol paths never call them, so LC-03
    # (jsonl → chat_history) is unaffected.
    # ------------------------------------------------------------------

    def _read_all(self) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        rows: List[Dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
        return rows

    def append_multi(
        self,
        *,
        day: int,
        date: str,
        market_view: str,
        benchmark_return: float,
        selected_sectors: List[str],
        target_positions: Dict[str, int],
        llm_reasoning: str,
        llm_trace: Dict[str, Any],
        kline_features: Dict[str, Any],
        sentiment: Dict[str, Any],
    ) -> None:
        """Write one multi-symbol portfolio decision.

        ``active_return`` starts null and is back-filled by ``record_outcome``
        at the next rebalance, once the realized excess return over the
        holding interval is known. Recall only surfaces records whose outcome
        has been filled — an unfilled record has no learning value.
        """
        self.append(
            {
                "type": "decision",
                "scope": "multi",
                "day": day,
                "date": date,
                "market_view": market_view,
                "benchmark_return": round(float(benchmark_return), 6),
                "selected_sectors": list(selected_sectors or []),
                "target_positions": dict(target_positions or {}),
                "llm_reasoning": (llm_reasoning or "")[:1500],
                "llm_trace": llm_trace or {},
                "kline_features": kline_features or {},
                "sentiment": sentiment or {},
                "active_return": None,
                "outcome_day": None,
                "holding_days": None,
                "execution": {},
                "memory_schema_version": 2,
            }
        )

    def _rewrite_all(self, rows: List[Dict[str, Any]]) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        tmp.replace(self.path)

    def record_execution(self, ref_day: int, execution: Dict[str, Any]) -> None:
        """Attach the execution summary for a multi-symbol decision.

        The allocator writes the intended target before the broker runs.  The
        engine updates this field immediately afterwards, allowing recall to
        distinguish a fully traceable decision from one with rejected orders.
        """
        try:
            rows = self._read_all()
            for row in rows:
                if row.get("scope") == "multi" and row.get("day") == ref_day:
                    row["execution"] = dict(execution or {})
                    self._rewrite_all(rows)
                    break
            self.last_error = None
        except Exception as exc:
            self.last_error = str(exc)

    def record_outcome(
        self,
        ref_day: int,
        active_return: float,
        *,
        outcome_day: int | None = None,
        holding_days: int | None = None,
    ) -> None:
        """Back-fill the realized active return onto the decision at ``ref_day``.

        Rewrites the jsonl in place. Only the earliest unfilled multi decision
        at ``ref_day`` is patched, so repeated calls are idempotent. Failures
        are swallowed into ``last_error`` — the backtest must not crash because
        a memory write failed.
        """
        try:
            rows = self._read_all()
            patched = False
            for row in rows:
                if (
                    row.get("scope") == "multi"
                    and row.get("day") == ref_day
                    and row.get("active_return") is None
                ):
                    row["active_return"] = round(float(active_return), 6)
                    row["outcome_day"] = outcome_day
                    row["holding_days"] = holding_days
                    patched = True
                    break
            if patched:
                self._rewrite_all(rows)
            self.last_error = None
        except Exception as exc:
            self.last_error = str(exc)

    def recall_similar(
        self,
        kline_features: Dict[str, Any] | None,
        market_view: str,
        limit: int = 3,
        *,
        current_day: int | None = None,
        min_quality: float = _QUALITY_THRESHOLD,
    ) -> List[Dict[str, Any]]:
        """Recall complete, recent and regime-compatible past decisions.

        Retrieval keeps negative outcomes: quality measures observability and
        execution integrity, never whether a decision made money.  When
        ``current_day`` is supplied, an outcome observed on that same decision
        day is excluded to avoid using information unavailable before the new
        decision is made.
        """
        rows = self._read_all()
        query_vec = _feature_vector(kline_features)
        recall_mode = os.environ.get("MEMORY_RECALL_MODE", "full").lower()
        scored: List[Tuple[float, int, Dict[str, Any]]] = []
        for idx, row in enumerate(rows):
            if row.get("scope") != "multi" or not _is_finite_number(row.get("active_return")):
                continue
            outcome_day = row.get("outcome_day")
            if current_day is not None and outcome_day is not None:
                try:
                    if int(outcome_day) >= int(current_day):
                        continue
                except (TypeError, ValueError):
                    continue
            quality, quality_parts = _sample_quality(row)
            if recall_mode != "legacy" and quality < min_quality:
                continue
            cand_vec = _feature_vector(row.get("kline_features"))
            dist = _euclidean(_normalize(query_vec), _normalize(cand_vec))
            same_regime = row.get("market_view") == market_view
            regime_factor = 1.0 if same_regime else _CROSS_REGIME_FACTOR
            age_days = _age_days(current_day, row.get("day"))
            recency = 2.0 ** (-age_days / _HALF_LIFE_DAYS)
            similarity = math.exp(-dist / _SIMILARITY_TAU)
            if recall_mode == "legacy":
                # Exact pre-improvement behavior: only feature distance and a
                # same-regime distance multiplier influence ordering.
                legacy_distance = dist * (0.5 if same_regime else 1.0)
                score = 1.0 / (1.0 + legacy_distance)
            elif recall_mode == "quality":
                score = quality * similarity * regime_factor
            else:  # quality_recency and full
                score = quality * similarity * recency * regime_factor
            recalled = dict(row)
            recalled["_recall"] = {
                "score": round(score, 6),
                "quality": round(quality, 4),
                "quality_parts": quality_parts,
                "distance": round(dist, 6),
                "similarity": round(similarity, 6),
                "age_days": age_days,
                "recency": round(recency, 6),
                "same_regime": same_regime,
                "regime_factor": regime_factor,
            }
            scored.append((score, idx, recalled))
        scored.sort(key=lambda item: (-item[0], -item[1]))
        return [row for _, _, row in scored[: max(0, limit)]]

    @staticmethod
    def effective_sample_size(records: List[Dict[str, Any]]) -> float:
        """Return the concentration-adjusted count of recalled experience."""
        scores = [
            float((row.get("_recall") or {}).get("score", 0.0))
            for row in records
            if _is_finite_number((row.get("_recall") or {}).get("score"))
        ]
        denominator = sum(score * score for score in scores)
        return (sum(scores) ** 2 / denominator) if denominator else 0.0

    def to_lesson(self, records: List[Dict[str, Any]]) -> str:
        """Render recalled decisions as a compact experience note for the LLM."""
        if not records:
            return ""
        mean_quality = sum(float((row.get("_recall") or {}).get("quality", 0.0)) for row in records) / len(records)
        effective_n = self.effective_sample_size(records)
        lines: List[str] = [
            f"[相似经验 {len(records)} 条 | Neff={effective_n:.2f} | 平均质量={mean_quality:.2f}]"
        ]
        for row in records:
            market = (row.get("kline_features") or {}).get("market") or {}
            sectors = ", ".join(row.get("selected_sectors") or []) or "—"
            active = row.get("active_return")
            active_txt = f"{float(active):+.2%}" if isinstance(active, (int, float)) else "N/A"
            recall = row.get("_recall") or {}
            age = recall.get("age_days")
            age_txt = f" | {age} 个交易日前" if isinstance(age, int) else ""
            lines.append(
                f"[历史 day{row.get('day')} {row.get('market_view')}{age_txt}] "
                f"RSI={float(market.get('avg_rsi', 0)):.0f} "
                f"vol={float(market.get('avg_volatility', 0)):.2f} "
                f"breadth={float(market.get('breadth_up', 0)):.0%} "
                f"选中[{sectors}] → 区间超额 {active_txt}"
            )
        return "\n".join(lines)


def _feature_vector(kline_features: Dict[str, Any] | None) -> List[float]:
    market = (kline_features or {}).get("market") or {}
    return [
        float(market.get("avg_rsi", 50.0)),
        float(market.get("avg_volatility", 0.2)),
        float(market.get("breadth_up", 0.5)),
        float(market.get("avg_dist_sma20", 0.0)),
    ]


def _is_finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _age_days(current_day: int | None, record_day: Any) -> int:
    if current_day is None:
        return 0
    try:
        return max(0, int(current_day) - int(record_day))
    except (TypeError, ValueError):
        return 0


def _sample_quality(row: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
    """Score record reliability without looking at the sign of its return."""
    market = (row.get("kline_features") or {}).get("market") or {}
    feature_quality = sum(_is_finite_number(market.get(key)) for key in _MARKET_FEATURES) / len(_MARKET_FEATURES)

    execution = row.get("execution") or {}
    if not execution:
        # Version-1 records do not have execution summaries. Preserve backward
        # compatibility while preferring fully traceable version-2 records.
        execution_quality = 0.70
    else:
        attempted = max(0, int(execution.get("attempted_orders", 0) or 0))
        filled = max(0, int(execution.get("filled_orders", 0) or 0))
        blocked = max(0, int(execution.get("blocked_orders", 0) or 0))
        if attempted == 0:
            execution_quality = 1.0  # A deliberate hold is a valid decision.
        else:
            execution_quality = max(0.0, min(1.0, filled / attempted - 0.5 * blocked / attempted))

    outcome_quality = 1.0 if _is_finite_number(row.get("active_return")) else 0.0
    trace_fields = (
        bool(row.get("market_view")),
        isinstance(row.get("selected_sectors"), list),
        isinstance(row.get("target_positions"), dict) or bool(row.get("llm_reasoning")) or bool(row.get("llm_trace")),
    )
    trace_quality = sum(trace_fields) / len(trace_fields)
    parts = {
        "feature": round(feature_quality, 4),
        "execution": round(execution_quality, 4),
        "outcome": round(outcome_quality, 4),
        "trace": round(trace_quality, 4),
    }
    quality = (
        0.40 * feature_quality
        + 0.25 * execution_quality
        + 0.20 * outcome_quality
        + 0.15 * trace_quality
    )
    return quality, parts


def _normalize(vec: List[float]) -> List[float]:
    # RSI spans 0..100; the other components are already roughly 0..1 (or a
    # small signed distance). Scale RSI down so it does not dominate the
    # euclidean distance.
    return [vec[0] / 100.0, vec[1], vec[2], vec[3]]


def _euclidean(a: List[float], b: List[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
