"""Run 924 offline vs LLM comparison with the FIXED benchmark (000300).

Prints a side-by-side comparison table so we can see whether LLM adds value
over pure-rule under the real (non-synthetic) benchmark.
"""
from __future__ import annotations

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


def _run(*, agent_mode: str, label: str) -> dict:
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
    # Offline first (no LLM key needed)
    results.append(_run(agent_mode="offline", label="offline 规则"))
    # Then LLM
    results.append(_run(agent_mode="llm", label="deepseek LLM"))

    # Side-by-side
    print()
    print(f"{'':>16} {'offline 规则':>14} {'deepseek LLM':>14}")
    print("-" * 46)
    for key, fmt in [
        ("total", ".2%"),
        ("bench", ".2%"),
        ("active", ".2%"),
        ("sharpe", ".2f"),
        ("maxdd", ".2%"),
        ("trades", ""),
    ]:
        vals = [r[key] for r in results]
        label = {"total": "总收益", "bench": "基准", "active": "超额", "sharpe": "夏普",
                 "maxdd": "最大回撤", "trades": "交易笔数"}[key]
        if fmt:
            row = f"{label:>12}  " + "  ".join(f"{v:{fmt}}" for v in vals)
        else:
            row = f"{label:>12}  " + "  ".join(f"{v}" for v in vals)
        print(row)
    print()
    for r in results:
        print(f"{r['label']}: market_view={r['market_view']} sectors={r['sectors']}")


if __name__ == "__main__":
    main()
