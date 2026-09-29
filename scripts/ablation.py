"""Ablation study: run LLM with each module removed, to quantify the
marginal contribution of each layer's injection.

Ablations (via env vars read by graph.py / top_down.py):
  - ABLATE_EXPERIENCE  → suppress experience-lesson injection
  - ABLATE_KNOWLEDGE   → suppress RAG knowledge-base injection
  - ABLATE_SENTIMENT   → zero the sentiment snapshot
  - ABLATE_GROWTH      → zero the earnings-growth-proxy weight in sector scoring

Each variant runs LLM on the spring-rebound regime (2024-02-05 → 2024-03-15),
which has a known positive active return (+5.83% offline) — a non-extreme
regime where each module's contribution is visible rather than drowned out
by a V-reversal.

Output: a table of active return per ablation vs the full (no-ablation) LLM.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Manual .env loader
for _name in (".env", ".env.local"):
    _path = ROOT / _name
    if not _path.exists():
        continue
    for _line in _path.read_text(encoding="utf-8").splitlines():
        _s = _line.strip()
        if not _s or _s.startswith("#") or "=" not in _s:
            continue
        _k, _v = _s.split("=", 1)
        os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

from core.schemas import BacktestConfig  # noqa: E402
from engine.backtest import TradingBacktestEngine  # noqa: E402

REGIME_NAME = "春节反弹"
START, END = "2024-02-05", "2024-03-15"

ABLATIONS = [
    ("full (无消融)", {}),
    ("−经验层", {"ABLATE_EXPERIENCE": "1"}),
    ("−知识库", {"ABLATE_KNOWLEDGE": "1"}),
    ("−情绪快照", {"ABLATE_SENTIMENT": "1"}),
    ("−盈利代理", {"ABLATE_GROWTH": "1"}),
]


def _run(ablate_env: dict) -> dict:
    # Clear any prior ablation env, then set this variant's
    for key in ("ABLATE_EXPERIENCE", "ABLATE_KNOWLEDGE", "ABLATE_SENTIMENT", "ABLATE_GROWTH"):
        os.environ.pop(key, None)
    for k, v in ablate_env.items():
        os.environ[k] = v

    MEMORY = Path("data/agent_memory.jsonl")
    if MEMORY.exists():
        MEMORY.unlink()

    cfg = BacktestConfig(
        provider="akshare",
        symbol="000300",
        benchmark_symbol="000300",
        dataset="trend",
        agent_mode="llm",
        strategy="aggressive",
        dynamic_universe=True,
        dynamic_universe_limit=12,
        dynamic_min_turnover=2e8,
        top_n_sectors=4,
        stocks_per_sector=2,
        max_positions=6,
        rebalance_frequency=3,
        cash_reserve_ratio=0.05,
        max_position_ratio=0.25,
        initial_cash=1_000_000,
        start=START,
        end=END,
    )
    eng = TradingBacktestEngine(cfg)
    eng.run_to_end()
    rep = eng.report()
    sv = eng.last_sector_view or {}
    return {
        "total": rep["total_return"],
        "bench": rep["benchmark_return"],
        "active": rep["active_return"],
        "sharpe": rep["sharpe"],
        "trades": rep["trade_count"],
        "market_view": sv.get("market_view"),
        "sectors": sv.get("selected_sectors"),
    }


def main() -> None:
    results = []
    for name, env in ABLATIONS:
        print(f"\n===== {name} =====", flush=True)
        r = _run(env)
        r["variant"] = name
        results.append(r)
        print(f"  active={r['active']:.4%} sectors={r['sectors']}", flush=True)

    # Summary
    baseline = results[0]["active"]
    print("\n" + "=" * 70)
    print(f"{'消融变体':<16} {'总收益':>10} {'基准':>10} {'超额':>10} {'相对full':>10}")
    print("-" * 70)
    for r in results:
        delta = r["active"] - baseline
        print(f"{r['variant']:<16} {r['total']:>10.2%} {r['bench']:>10.2%} "
              f"{r['active']:>10.2%} {delta:>+10.2%}")

    out = Path("data/ablation_results.json")
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str))
    print(f"\nraw -> {out}")


if __name__ == "__main__":
    main()
