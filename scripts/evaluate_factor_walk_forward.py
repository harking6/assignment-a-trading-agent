"""Long-horizon, reproducible factor and risk-budget walk-forward evaluation.

This intentionally uses the project's deterministic synthetic market regimes,
not a live quote endpoint: every comparison has identical inputs and can be
re-run offline. It compares the factor-selected portfolio with risk budgeting
disabled versus the signal-aware inverse-volatility allocation.

Example:
    .\\.venv\\Scripts\\python.exe scripts\\evaluate_factor_walk_forward.py --days 504
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from core.schemas import BacktestConfig
from engine.backtest import TradingBacktestEngine


SCENARIOS = (
    ("趋势行情", "trend"),
    ("震荡行情", "range"),
    ("压力下跌后修复", "stress"),
    ("事件冲击", "event"),
)
UNIVERSE = ("000001", "000002", "600000", "601012", "300750", "002001")
VARIANTS = (
    ("信号权重基线", 0.0),
    ("信号+逆波动风险预算", 0.65),
)


def _metrics(engine: TradingBacktestEngine) -> dict[str, float | int]:
    report = engine.report()
    daily = engine.returns
    annualized_volatility = statistics.pstdev(daily) * math.sqrt(252) if len(daily) > 1 else 0.0
    downside = [value for value in daily if value < 0]
    downside_volatility = statistics.pstdev(downside) * math.sqrt(252) if len(downside) > 1 else 0.0
    total_return = float(report["total_return"])
    annualized_return = (1.0 + total_return) ** (252 / max(1, len(daily))) - 1.0 if total_return > -1.0 else -1.0
    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "active_return": float(report["active_return"]),
        "sharpe": float(report["sharpe"]),
        "max_drawdown": float(report["max_drawdown"]),
        "annualized_volatility": annualized_volatility,
        "sortino": annualized_return / downside_volatility if downside_volatility else 0.0,
        "turnover": float(report["turnover"]),
        "trade_count": int(report["trade_count"]),
    }


def _run(scenario: str, dataset: str, label: str, strength: float, days: int, memory_path: Path) -> dict[str, Any]:
    config = BacktestConfig(
        provider="synthetic",
        dataset=dataset,
        days=days,
        symbol="BENCH",
        benchmark_symbol="BENCH",
        universe=list(UNIVERSE),
        agent_mode="offline",
        rebalance_frequency=5,
        max_positions=4,
        top_n_sectors=3,
        stocks_per_sector=2,
        initial_cash=1_000_000.0,
        cash_reserve_ratio=0.08,
        max_position_ratio=0.35,
        risk_parity_strength=strength,
    )
    old_memory_path = os.environ.get("TRADING_AGENT_MEMORY_PATH")
    os.environ["TRADING_AGENT_MEMORY_PATH"] = str(memory_path)
    try:
        engine = TradingBacktestEngine(config)
        engine.run_to_end()
    finally:
        if old_memory_path is None:
            os.environ.pop("TRADING_AGENT_MEMORY_PATH", None)
        else:
            os.environ["TRADING_AGENT_MEMORY_PATH"] = old_memory_path

    return {
        "scenario": scenario,
        "dataset": dataset,
        "variant": label,
        "risk_parity_strength": strength,
        "days": days,
        "metrics": _metrics(engine),
        "final_risk_budget": engine.last_sector_view.get("risk_budget", {}),
        "structural_validity": engine.report()["bias_audit"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=504, help="Trading days per scenario (default: 504, about two years).")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "factor_walk_forward_results.json")
    args = parser.parse_args()
    if args.days < 60:
        parser.error("--days must be at least 60 for a meaningful walk-forward run")

    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="factor-walk-forward-") as temp_dir:
        root = Path(temp_dir)
        for scenario, dataset in SCENARIOS:
            for label, strength in VARIANTS:
                print(f"{scenario} | {label} | {args.days} days", flush=True)
                rows.append(_run(scenario, dataset, label, strength, args.days, root / f"{dataset}_{strength}.jsonl"))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "methodology": {
            "provider": "synthetic",
            "days_per_scenario": args.days,
            "scenarios": [name for name, _ in SCENARIOS],
            "comparison": "factor signal weights vs signal-aware inverse-volatility risk budgeting",
            "bias_framework": "Five-Sins structural-validity audit: look-ahead, survivorship, narrative, objective, cost",
            "note": "Synthetic scenarios are deterministic engineering validation, not claims about live-market performance.",
        },
        "runs": rows,
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"results -> {args.output}")


if __name__ == "__main__":
    main()
