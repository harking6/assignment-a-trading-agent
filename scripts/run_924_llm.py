"""Run one deepseek LLM multi-symbol backtest over the 924 暴涨段.

Clears the experience memory first so the run starts clean (no cross-run
recall contamination). Prints the engine.report() metrics plus LLM stage
errors / timeouts and the last sector view.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Manual .env loader (mirrors app.py; python-dotenv is not installed) so the
# deepseek API key / base URL reach the LLMClient when run as a standalone
# script instead of through the web server.
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

from core.schemas import BacktestConfig
from engine.backtest import TradingBacktestEngine

MEMORY = Path("data/agent_memory.jsonl")
if MEMORY.exists():
    MEMORY.unlink()


def main() -> None:
    cfg = BacktestConfig(
        provider="akshare",
        symbol="000300",  # 沪深300 as benchmark
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
        start="2024-09-23",
        end="2024-10-08",
    )
    eng = TradingBacktestEngine(cfg)
    eng.run_to_end()
    rep = eng.report()
    print(
        "total={total_return:.4%} bench={benchmark_return:.4%} active={active_return:.4%} "
        "sharpe={sharpe} maxdd={max_drawdown:.2%} trades={trade_count} blocked={blocked_orders}".format(**rep)
    )
    errs = getattr(eng.graph.top_down, "_llm_stage_errors", [])
    print(f"llm_errors={len(errs)}")
    for e in errs[:8]:
        print(f"  err: {e}")
    sv = eng.last_sector_view or {}
    print("market_view:", sv.get("market_view"))
    print("selected_sectors:", sv.get("selected_sectors"))
    print("llm_trace:", sv.get("llm_trace"))
    # How many multi decisions were written + backfilled (experience loop health).
    import json
    if MEMORY.exists():
        rows = [json.loads(l) for l in MEMORY.read_text().splitlines() if l.strip()]
        multi = [r for r in rows if r.get("scope") == "multi"]
        filled = [r for r in multi if r.get("active_return") is not None]
        print(f"memory: {len(multi)} multi decisions, {len(filled)} back-filled")


if __name__ == "__main__":
    main()
