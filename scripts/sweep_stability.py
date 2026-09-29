"""Stability sweep: 4 LLM runs + 1 offline on 924 window (benchmark FIXED).

Reports per-round numbers + summary statistics so we can quantify LLM
non-determinism and the LLM-vs-offline gap with confidence intervals.
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


def _run(*, agent_mode: str, label: str, run: int) -> dict:
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
        start="2024-09-23",
        end="2024-10-08",
    )
    eng = TradingBacktestEngine(cfg)
    eng.run_to_end()
    rep = eng.report()
    sv = eng.last_sector_view or {}
    return {
        "label": label,
        "run": run,
        "total": rep["total_return"],
        "bench": rep["benchmark_return"],
        "active": rep["active_return"],
        "sharpe": rep["sharpe"],
        "maxdd": rep["max_drawdown"],
        "trades": rep["trade_count"],
        "market_view": sv.get("market_view"),
        "sectors": sv.get("selected_sectors"),
    }


def _stats(vals: list[float]) -> str:
    import math
    m = sum(vals) / len(vals)
    if len(vals) < 2:
        return f"mean={m:.4%}"
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))
    return f"mean={m:.4%} ±{sd:.2%}  [{min(vals):.4%}, {max(vals):.4%}]"


def main() -> None:
    N_LLM = 4
    results = []

    # Offline once
    results.append(_run(agent_mode="offline", label="offline", run=0))

    # LLM N times
    for i in range(N_LLM):
        results.append(_run(agent_mode="llm", label="llm", run=i + 1))

    print()
    print("=" * 72)
    print("  Stable sweep: 924 每3天 aggressive — 基准 000300 (已修复)")
    print("=" * 72)
    print(f"  {'Run':>10} {'Mode':>8} {'总收益':>10} {'超额':>10} {'夏普':>8} {'行业':>16}")
    print(f"  {'-'*10} {'-'*8} {'-'*10} {'-'*10} {'-'*8} {'-'*16}")
    for r in results:
        sectors = ",".join(r["sectors"][:2]) if r["sectors"] else "—"
        print(f"  {r['run']:>4}/{r['label']:>4}  {r['total']:>10.4%} {r['active']:>10.4%} {r['sharpe']:>8.2f} {sectors:>16}")

    print()
    offline = [r for r in results if r["label"] == "offline"]
    llm = [r for r in results if r["label"] == "llm"]
    print("  --- Summary ---")
    for key, fmt in [("total", ".2%"), ("active", ".2%"), ("sharpe", ".2f")]:
        name = {"total": "总收益", "active": "超额", "sharpe": "夏普"}[key]
        ovals = [r[key] for r in offline]
        lvals = [r[key] for r in llm]
        print(f"  {name:>8}  offline: {ovals[0]:{fmt}}")
        print(f"  {'':>8}  LLM:     {_stats(lvals)}")

    # Save raw JSON for report
    out_path = Path("data/sweep_924_fixed.json")
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str))
    print(f"\n  raw -> {out_path}")


if __name__ == "__main__":
    main()
