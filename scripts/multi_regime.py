"""Multi-regime comparison: run offline vs LLM across 4 distinct market
regimes to see whether the strategy's active return holds up outside the
924 melt-up.

Regimes:
  - 924 急涨   (V 反暴涨)   2024-09-23 → 2024-10-08   bench +32.47%
  - 春节反弹   (超跌反弹)   2024-02-05 → 2024-03-15   bench +11.55%
  - 1月杀跌   (小盘崩跌)   2024-01-15 → 2024-02-05   bench  -2.45%
  - 震荡阴跌   (慢熊)       2023-10-23 → 2023-12-29   bench  -1.24%

Same config as the 924 run (dynamic universe, aggressive, every-3-day,
12 candidates, 6 positions) so results are directly comparable.
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

REGIMES = [
    ("924急涨", "2024-09-23", "2024-10-08"),
    ("春节反弹", "2024-02-05", "2024-03-15"),
    ("1月杀跌", "2024-01-15", "2024-02-05"),
    ("震荡阴跌", "2023-10-23", "2023-12-29"),
]


def _run(*, agent_mode: str, label: str, start: str, end: str) -> dict:
    MEMORY = Path("data/agent_memory.jsonl")
    if MEMORY.exists():
        MEMORY.unlink()

    cfg = BacktestConfig(
        provider="akshare",
        symbol="000300",
        benchmark_symbol="000300",
        dataset="trend",
        agent_mode=agent_mode,
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
        start=start,
        end=end,
    )
    eng = TradingBacktestEngine(cfg)
    eng.run_to_end()
    rep = eng.report()
    sv = eng.last_sector_view or {}
    return {
        "label": label,
        "mode": agent_mode,
        "start": start,
        "end": end,
        "total": rep["total_return"],
        "bench": rep["benchmark_return"],
        "active": rep["active_return"],
        "sharpe": rep["sharpe"],
        "maxdd": rep["max_drawdown"],
        "trades": rep["trade_count"],
        "market_view": sv.get("market_view"),
        "sectors": sv.get("selected_sectors"),
    }


def main() -> None:
    results = []
    for name, start, end in REGIMES:
        print(f"\n===== {name} {start}~{end} =====", flush=True)
        # offline first (fast)
        results.append(_run(agent_mode="offline", label=name, start=start, end=end))
        print(f"  offline done: active={results[-1]['active']:.4%}", flush=True)
        # then LLM
        results.append(_run(agent_mode="llm", label=name, start=start, end=end))
        print(f"  llm done:     active={results[-1]['active']:.4%}", flush=True)

    # Summary table
    print("\n" + "=" * 78)
    print(f"{'段':<8} {'模式':<8} {'总收益':>10} {'基准':>10} {'超额':>10} {'夏普':>8} {'最大回撤':>10}")
    print("-" * 78)
    for r in results:
        print(f"{r['label']:<8} {r['mode']:<8} {r['total']:>10.2%} {r['bench']:>10.2%} "
              f"{r['active']:>10.2%} {r['sharpe']:>8.2f} {r['maxdd']:>10.2%}")

    # Per-regime LLM vs offline delta
    print("\n" + "=" * 50)
    print(f"{'段':<10} {'offline超额':>12} {'LLM超额':>12} {'LLM增量':>12}")
    print("-" * 50)
    for name, _, _ in REGIMES:
        o = next(r for r in results if r["label"] == name and r["mode"] == "offline")
        l = next(r for r in results if r["label"] == name and r["mode"] == "llm")
        print(f"{name:<10} {o['active']:>12.2%} {l['active']:>12.2%} {l['active']-o['active']:>12.2%}")

    out = Path("data/multi_regime_results.json")
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str))
    print(f"\nraw -> {out}")


if __name__ == "__main__":
    main()
