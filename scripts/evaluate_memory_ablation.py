"""Evaluate the member-B memory layer with controlled, non-destructive ablations.

Each test regime is preceded by a strictly earlier warm-up period that builds
episodic memory.  Every run uses its own temporary JSONL store, so this script
never deletes or changes ``data/agent_memory.jsonl`` used by the application.

Example (requires a configured OpenAI-compatible model for LLM mode)::

    .\.venv\Scripts\python.exe scripts\evaluate_memory_ablation.py --agent-mode llm --runs 5

For a quick wiring check without an LLM, use ``--agent-mode offline``.  Offline
mode is deterministic but does not demonstrate the effect of memory injection,
because the rule-only selector intentionally ignores LLM context.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Keep this script consistent with the existing benchmark scripts: credentials
# live in a local ignored file and are never written into result artifacts.
for _name in (".env", ".env.local"):
    _path = ROOT / _name
    if not _path.exists():
        continue
    for _line in _path.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if not _line or _line.startswith("#") or "=" not in _line:
            continue
        _key, _value = _line.split("=", 1)
        os.environ.setdefault(_key.strip(), _value.strip().strip('"').strip("'"))

from core.schemas import BacktestConfig
from engine.backtest import TradingBacktestEngine


REGIMES = [
    ("春节反弹", "2023-11-01", "2024-02-05", "2024-02-05", "2024-03-15"),
    ("1月杀跌", "2023-10-02", "2024-01-15", "2024-01-15", "2024-02-05"),
    ("震荡阴跌", "2023-07-24", "2023-10-23", "2023-10-23", "2023-12-29"),
    ("924急涨", "2024-07-01", "2024-09-23", "2024-09-23", "2024-10-08"),
]
QUICK_REGIME = (
    "短窗连通验证",
    "2024-01-22",
    "2024-02-05",
    "2024-02-05",
    "2024-02-16",
)
VARIANTS = [
    ("A 无经验", "none"),
    ("B 现有基线", "legacy"),
    ("C 质量门控", "quality"),
    ("D 质量加时效", "quality_recency"),
    ("E 完整方案", "full"),
]


@contextmanager
def _memory_environment(path: Path, recall_mode: str) -> Iterator[None]:
    """Temporarily select a run-specific memory store and ablation mode."""
    keys = ("TRADING_AGENT_MEMORY_PATH", "MEMORY_RECALL_MODE", "ABLATE_EXPERIENCE")
    previous = {key: os.environ.get(key) for key in keys}
    os.environ["TRADING_AGENT_MEMORY_PATH"] = str(path)
    os.environ["MEMORY_RECALL_MODE"] = recall_mode
    if recall_mode == "none":
        os.environ["ABLATE_EXPERIENCE"] = "1"
    else:
        os.environ.pop("ABLATE_EXPERIENCE", None)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _config(agent_mode: str, provider: str, start: str, end: str, *, quick: bool = False) -> BacktestConfig:
    return BacktestConfig(
        provider=provider,
        symbol="000300",
        benchmark_symbol="000300",
        dataset="trend",
        agent_mode=agent_mode,
        strategy="aggressive",
        dynamic_universe=True,
        dynamic_universe_limit=4 if quick else 12,
        dynamic_min_turnover=2e8,
        top_n_sectors=2 if quick else 4,
        stocks_per_sector=1 if quick else 2,
        max_positions=2 if quick else 6,
        # A short run needs at least three *settled* warm-up decisions before
        # the quality gate can legitimately inject an experience lesson.  A
        # three-day cadence produces that evidence within QUICK_REGIME without
        # extending the evaluation window or leaking test prices into warm-up.
        rebalance_frequency=3 if quick else 3,
        cash_reserve_ratio=0.05,
        max_position_ratio=0.25,
        initial_cash=1_000_000,
        start=start,
        end=end,
    )


def _run_period(agent_mode: str, provider: str, start: str, end: str, *, quick: bool = False) -> TradingBacktestEngine:
    engine = TradingBacktestEngine(_config(agent_mode, provider, start, end, quick=quick))
    # ``engine.trace`` describes only the most recent trading day.  Preserve
    # the full period trace here so a short run can prove whether recall was
    # attempted and whether the lesson passed the injection gate.
    trace_history: list[dict[str, Any]] = []
    while not engine.finished:
        engine.step()
        trace_history.extend(engine.trace)
    engine.ablation_trace = trace_history
    return engine


def _rebase_warmup_memory(path: Path, warmup_days: int) -> None:
    """Express pre-test experience on the test engine's relative day axis.

    Each ``TradingBacktestEngine`` starts its day index from zero. After a
    separate warm-up engine finishes, its outcome days would otherwise look
    *later* than every test decision day and correctly be blocked by the
    anti-leakage guard. Shift those completed warm-up records into negative
    relative days so they are known before the test begins.
    """
    if not path.exists() or warmup_days <= 0:
        return
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("scope") == "multi":
            for field in ("day", "outcome_day"):
                try:
                    if row.get(field) is not None:
                        row[field] = int(row[field]) - warmup_days
                except (TypeError, ValueError):
                    pass
        rows.append(row)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _one_run(
    *,
    regime: tuple[str, str, str, str, str],
    variant: tuple[str, str],
    repeat: int,
    agent_mode: str,
    provider: str,
    root: Path,
    quick: bool = False,
) -> dict[str, Any]:
    name, warm_start, warm_end, test_start, test_end = regime
    label, recall_mode = variant
    memory_path = root / f"{name}_{recall_mode}_{repeat}.jsonl"
    with _memory_environment(memory_path, recall_mode):
        # Warm-up is deliberately run before the evaluation interval. It may
        # write decisions and outcomes, but never sees a test-period price.
        warmup_engine = _run_period(agent_mode, provider, warm_start, warm_end, quick=quick)
        _rebase_warmup_memory(memory_path, warmup_engine.day_index)
        test_engine = _run_period(agent_mode, provider, test_start, test_end, quick=quick)
        report = test_engine.report()
        recall_trace = [
            item
            for item in getattr(test_engine, "ablation_trace", test_engine.trace)
            if item.get("tool") == "memory_recall"
        ]
        return {
            "regime": name,
            "variant": label,
            "recall_mode": recall_mode,
            "repeat": repeat,
            "warmup": {"start": warm_start, "end": warm_end},
            "test": {"start": test_start, "end": test_end},
            "total_return": report["total_return"],
            "benchmark_return": report["benchmark_return"],
            "active_return": report["active_return"],
            "sharpe": report["sharpe"],
            "max_drawdown": report["max_drawdown"],
            "turnover": report["turnover"],
            "trade_count": report["trade_count"],
            "memory_trace": recall_trace,
        }


def _summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["regime"], row["variant"]), []).append(row)
    summary = []
    for (regime, variant), samples in grouped.items():
        active = [float(sample["active_return"]) for sample in samples]
        summary.append(
            {
                "regime": regime,
                "variant": variant,
                "runs": len(samples),
                "median_active_return": statistics.median(active),
                "min_active_return": min(active),
                "max_active_return": max(active),
                "median_sharpe": statistics.median(float(sample["sharpe"]) for sample in samples),
                "median_max_drawdown": statistics.median(float(sample["max_drawdown"]) for sample in samples),
                "median_turnover": statistics.median(float(sample["turnover"]) for sample in samples),
            }
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-mode", choices=("llm", "offline"), default="llm")
    parser.add_argument("--provider", default="akshare")
    parser.add_argument("--runs", type=int, default=5, help="repetitions per regime and variant")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "memory_ablation_results.json")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use one short warm-up/test window and a smaller universe for inexpensive end-to-end validation.",
    )
    parser.add_argument(
        "--recall-modes",
        nargs="+",
        choices=[mode for _, mode in VARIANTS],
        help="Restrict variants by recall mode, for example: --recall-modes full.",
    )
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    if args.agent_mode == "llm" and not os.environ.get("OPENAI_API_KEY"):
        parser.error("LLM mode requires OPENAI_API_KEY. Use --agent-mode offline only for wiring checks.")

    regimes = [QUICK_REGIME] if args.quick else REGIMES
    variants = [item for item in VARIANTS if not args.recall_modes or item[1] in args.recall_modes]
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="trading-memory-ablation-") as temp_dir:
        temp_root = Path(temp_dir)
        for regime in regimes:
            for variant in variants:
                for repeat in range(args.runs):
                    print(f"{regime[0]} | {variant[0]} | run {repeat + 1}/{args.runs}", flush=True)
                    rows.append(
                        _one_run(
                            regime=regime,
                            variant=variant,
                            repeat=repeat,
                            agent_mode=args.agent_mode,
                            provider=args.provider,
                            root=temp_root,
                            quick=args.quick,
                        )
                    )

    payload = {"runs": rows, "summary": _summary(rows)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"results -> {args.output}")


if __name__ == "__main__":
    main()
