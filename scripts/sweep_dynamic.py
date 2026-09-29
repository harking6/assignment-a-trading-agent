"""Run multiple dynamic-universe backtest rounds and print a comparison table."""
from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

for _k in list(os.environ):
    if "proxy" in _k.lower():
        del os.environ[_k]
urllib.request.getproxies = lambda: {}

from core.schemas import BacktestConfig
from engine.backtest import TradingBacktestEngine


def run(label: str, **overrides) -> dict:
    base = dict(
        provider="akshare",
        agent_mode="offline",
        dynamic_universe=True,
        dynamic_universe_limit=12,
        dynamic_sector_pool=4,
        dynamic_min_turnover=500_000_000,
        symbol="000001",
        benchmark_symbol="000300",
        start="2024-01-01",
        end="2024-01-31",
        rebalance_frequency=5,
        cash_reserve_ratio=0.05,
        max_position_ratio=0.20,
        top_n_sectors=2,
        stocks_per_sector=3,
        max_positions=5,
    )
    base.update(overrides)
    cfg = BacktestConfig(**base)
    eng = TradingBacktestEngine(cfg)
    out = eng.run_to_end()
    r = out["report"]
    eff = eng.effective_universe
    return {
        "label": label,
        "total": r["total_return"],
        "bench": r["benchmark_return"],
        "active": r["active_return"],
        "maxdd": r["max_drawdown"],
        "trades": r["trade_count"],
        "sharpe": r["sharpe"],
        "n_universe": len(eff) if eff else 0,
        "final_positions": list(out["portfolio"].get("positions", {}).keys()),
    }


configs = [
    ("baseline rbf5 cr5%", {}),
    ("rebalance every 3d", {"rebalance_frequency": 3}),
    ("rebalance daily", {"rebalance_frequency": 1}),
    ("more cash 15%", {"cash_reserve_ratio": 0.15}),
    ("aggressive cr2% pos25%", {"cash_reserve_ratio": 0.02, "max_position_ratio": 0.25}),
    ("top3 sectors 8 stocks", {"top_n_sectors": 3, "stocks_per_sector": 3, "max_positions": 8}),
    ("wider universe 20", {"dynamic_universe_limit": 20, "max_positions": 8}),
]

results = []
for label, ov in configs:
    try:
        res = run(label, **ov)
        results.append(res)
        print(f"  done: {label}")
    except Exception as exc:
        print(f"  FAILED: {label}: {exc!r:.120}")

print("\n=== 动态选股回测对比 (2024-01, offline) ===")
hdr = f"{'config':30s} {'total':>8s} {'bench':>8s} {'active':>8s} {'maxdd':>8s} {'trades':>7s} {'sharpe':>7s} {'univ':>5s}"
print(hdr)
print("-" * len(hdr))
for r in results:
    print(
        f"{r['label']:30s} {r['total']*100:7.2f}% {r['bench']*100:7.2f}% "
        f"{r['active']*100:7.2f}% {r['maxdd']*100:7.2f}% {r['trades']:7d} "
        f"{r['sharpe']:7.2f} {r['n_universe']:5d}"
    )
