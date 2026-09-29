from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

from langchain_core.chat_history import InMemoryChatMessageHistory


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
        self.path = path or (root / "data" / "agent_memory.jsonl")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.last_error: str | None = None

    def recent(self, symbol: str, limit: int = 6) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        rows: List[Dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
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
        for line in self.path.read_text(encoding="utf-8").splitlines():
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
            }
        )

    def record_outcome(self, ref_day: int, active_return: float) -> None:
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
                    patched = True
                    break
            if patched:
                tmp = self.path.with_suffix(self.path.suffix + ".tmp")
                with tmp.open("w", encoding="utf-8") as handle:
                    for row in rows:
                        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                tmp.replace(self.path)
            self.last_error = None
        except Exception as exc:
            self.last_error = str(exc)

    def recall_similar(
        self,
        kline_features: Dict[str, Any] | None,
        market_view: str,
        limit: int = 3,
    ) -> List[Dict[str, Any]]:
        """Recall past multi decisions whose market form was most similar.

        Similarity is euclidean distance over a normalized market feature
        vector (avg RSI / 100, avg volatility, breadth-up ratio, avg distance
        to SMA-20). A same-regime bias halves the distance for records taken
        in the same ``market_view``, so a bearish day does not surface bullish
        experience as the nearest neighbour. Only records with a back-filled
        ``active_return`` are eligible — the rest have not been learned from.
        """
        rows = self._read_all()
        query_vec = _feature_vector(kline_features)
        scored: List[Tuple[float, int, Dict[str, Any]]] = []
        for idx, row in enumerate(rows):
            if row.get("scope") != "multi" or row.get("active_return") is None:
                continue
            cand_vec = _feature_vector(row.get("kline_features"))
            dist = _euclidean(_normalize(query_vec), _normalize(cand_vec))
            if row.get("market_view") == market_view:
                dist *= 0.5
            scored.append((dist, idx, row))
        scored.sort(key=lambda item: (item[0], item[1]))
        return [row for _, _, row in scored[: max(0, limit)]]

    def to_lesson(self, records: List[Dict[str, Any]]) -> str:
        """Render recalled decisions as a compact experience note for the LLM."""
        if not records:
            return ""
        lines: List[str] = []
        for row in records:
            market = (row.get("kline_features") or {}).get("market") or {}
            sectors = ", ".join(row.get("selected_sectors") or []) or "—"
            active = row.get("active_return")
            active_txt = f"{float(active):+.2%}" if isinstance(active, (int, float)) else "N/A"
            lines.append(
                f"[历史 day{row.get('day')} {row.get('market_view')}] "
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


def _normalize(vec: List[float]) -> List[float]:
    # RSI spans 0..100; the other components are already roughly 0..1 (or a
    # small signed distance). Scale RSI down so it does not dominate the
    # euclidean distance.
    return [vec[0] / 100.0, vec[1], vec[2], vec[3]]


def _euclidean(a: List[float], b: List[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
