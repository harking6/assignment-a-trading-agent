from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

@tool
def market_window_tool(close: float, sma_5: float, sma_20: float, rsi_14: float, volatility_20: float) -> str:
    """Summarize the current technical market window."""
    trend = "uptrend" if sma_5 > sma_20 else "downtrend" if sma_5 < sma_20 else "flat"
    if rsi_14 >= 70:
        rsi_state = "overbought"
    elif rsi_14 <= 30:
        rsi_state = "oversold"
    else:
        rsi_state = "neutral"
    return (
        f"trend={trend}, close={close:.4f}, sma_5={sma_5:.4f}, "
        f"sma_20={sma_20:.4f}, rsi_14={rsi_14:.2f} ({rsi_state}), "
        f"volatility_20={volatility_20:.4%}"
    )


@tool
def portfolio_exposure_tool(cash: float, position: int, equity: float, price: float) -> str:
    """Calculate the portfolio's current market exposure."""
    market_value = max(0, position) * max(0.0, price)
    exposure = market_value / equity if equity > 0 else 0.0
    cash_ratio = cash / equity if equity > 0 else 0.0
    return (
        f"market_value={market_value:.2f}, exposure={exposure:.2%}, "
        f"cash_ratio={cash_ratio:.2%}, position={position}"
    )


@tool
def risk_limit_tool(risk_budget: float, max_position_ratio: float, max_drawdown: float) -> str:
    """Summarize configured risk limits and current drawdown status."""
    drawdown_limit = -abs(risk_budget)
    status = "within_budget" if max_drawdown >= drawdown_limit else "breached"
    return (
        f"risk_budget={risk_budget:.2%}, max_position_ratio={max_position_ratio:.2%}, "
        f"max_drawdown={max_drawdown:.2%}, drawdown_limit={drawdown_limit:.2%}, "
        f"status={status}"
    )


def invoke_langchain_tools(row: Any, portfolio: Any, config: Any) -> dict[str, str]:
    """Invoke all three StructuredTool objects with normalized values.

    Normalize values from the pandas row and Pydantic models, call each tool
    through ``.invoke({...})``, and return a dictionary keyed by tool name.
    Do not call the original Python function directly; the assignment is meant
    to exercise LangChain's tool schema and invocation path.
    """

    market_args = {
        "close": float(row.get("close", 0.0)),
        "sma_5": float(row.get("sma_5", 0.0)),
        "sma_20": float(row.get("sma_20", 0.0)),
        "rsi_14": float(row.get("rsi_14", 0.0)),
        "volatility_20": float(row.get("volatility_20", 0.0)),
    }
    exposure_args = {
        "cash": float(portfolio.cash),
        "position": int(portfolio.position),
        "equity": float(portfolio.equity),
        "price": market_args["close"],
    }
    risk_args = {
        "risk_budget": float(config.risk_budget),
        "max_position_ratio": float(config.max_position_ratio),
        "max_drawdown": float(portfolio.max_drawdown),
    }
    return {
        market_window_tool.name: str(market_window_tool.invoke(market_args)),
        portfolio_exposure_tool.name: str(portfolio_exposure_tool.invoke(exposure_args)),
        risk_limit_tool.name: str(risk_limit_tool.invoke(risk_args)),
    }
