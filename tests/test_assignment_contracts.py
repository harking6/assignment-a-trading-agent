import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.graph import TradingAgentsGraph  # noqa: E402
from agents.llm_client import LLMClient  # noqa: E402
from agents.memory import TradingMemory  # noqa: E402
from agents.state import TradingGraphState  # noqa: E402
from core.schemas import (  # noqa: E402
    AnalystReport,
    BacktestConfig,
    OrderSide,
    PortfolioSnapshot,
    ResearchDebate,
    Stance,
    TradeIntent,
)
from tools.langchain_tools import invoke_langchain_tools  # noqa: E402
from tools.rag import TradingRAG  # noqa: E402


class AssignmentContractTest(unittest.TestCase):
    """Each test isolates one LangChain or LangGraph assignment contract."""

    def test_lg01_typed_state_contains_required_channels(self) -> None:
        required = {
            "day",
            "portfolio",
            "config",
            "memory",
            "memory_messages",
            "reports",
            "rag_context",
            "tool_context",
            "debate",
            "intent",
            "risk_debate",
            "risk",
            "decision",
            "trace",
        }
        self.assertTrue(required.issubset(TradingGraphState.__annotations__))

    def test_lc01_lcel_chain_parses_json_without_network(self) -> None:
        fake_module = types.ModuleType("langchain_openai")
        fake_module.ChatOpenAI = lambda **_kwargs: RunnableLambda(
            lambda _prompt: AIMessage(content='{"side":"hold","confidence":0.7}')
        )
        client = LLMClient(mode="llm", quick_model="fake-model")
        with patch.dict(sys.modules, {"langchain_openai": fake_module}):
            result = client._complete_json_langchain("test_role", "Return a decision")
        self.assertEqual(result, {"side": "hold", "confidence": 0.7})
        self.assertEqual(client.last_runtime, "langchain-openai")

    def test_lc02_tools_are_invoked_through_langchain(self) -> None:
        row = pd.Series({"close": 10.0, "sma_5": 10.5, "sma_20": 9.5, "rsi_14": 61.0, "volatility_20": 0.02})
        portfolio = PortfolioSnapshot(cash=50000, position=5000, equity=100000, peak=100000, max_drawdown=-0.03)
        result = invoke_langchain_tools(row, portfolio, BacktestConfig())
        self.assertEqual(set(result), {"market_window_tool", "portfolio_exposure_tool", "risk_limit_tool"})
        self.assertIn("uptrend", result["market_window_tool"])
        self.assertIn("exposure=50.00%", result["portfolio_exposure_tool"])

    def test_lc03_jsonl_memory_becomes_chat_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = TradingMemory(Path(tmpdir) / "memory.jsonl")
            memory.append({"symbol": "MEMTEST", "day": 1, "side": "buy", "shares": 100, "price": 10.0, "reason": "test"})
            history = memory.chat_history("MEMTEST", 3)
            self.assertEqual(len(history.messages), 2)
            self.assertEqual(history.messages[0].type, "human")
            self.assertEqual(history.messages[1].type, "ai")

    def test_lc04_rag_returns_context_and_sources(self) -> None:
        frame = pd.DataFrame([{"close": 10.0, "return_1d": 0.02, "sma_5": 10.2, "sma_20": 9.8, "macd_hist": 0.1, "rsi_14": 61.0, "volatility_20": 0.015}])
        reports = [
            AnalystReport(
                agent="market_analyst",
                stance=Stance.bullish,
                score=0.4,
                confidence=0.8,
                summary="Trend is bullish",
                evidence=["SMA5 is above SMA20"],
            )
        ]
        context = TradingRAG().build_context(
            symbol="000001",
            day=0,
            frame=frame,
            reports=reports,
            memory=[{"symbol": "000001", "day": 1, "side": "buy", "shares": 800, "reason": "prior buy"}],
            config=BacktestConfig(provider="akshare", symbol="000001"),
        )
        self.assertGreaterEqual(context["document_count"], 3)
        self.assertEqual(context["retriever"], "LexicalTradingRetriever")
        self.assertIn("Market snapshot", context["context"])
        self.assertTrue(any(source["source"] == "analyst_report" for source in context["sources"]))

    def test_lg02_graph_compiles_with_checkpoint(self) -> None:
        graph = TradingAgentsGraph()
        self.assertIsNotNone(graph._compiled)
        self.assertTrue(graph.checkpoint_enabled)
        self.assertIsNotNone(graph.checkpointer)

    def test_lg03_graph_compiles_without_checkpoint(self) -> None:
        graph = TradingAgentsGraph()
        compiled = graph._compile_without_checkpoint()
        self.assertIsNotNone(compiled)

    def test_lg04_conditional_routes(self) -> None:
        graph = TradingAgentsGraph()
        neutral = ResearchDebate(
            bull_thesis="bull",
            bear_thesis="bear",
            bull_score=0.5,
            bear_score=0.5,
            verdict=Stance.neutral,
            conviction=0.2,
            manager_notes="neutral",
        )
        self.assertEqual(graph._route_after_research({"debate": neutral, "trace": []}), "hold")
        intent = TradeIntent(signal="buy", side=OrderSide.buy, target_position=100, shares=100, confidence=0.8, rationale="test")
        self.assertEqual(graph._route_after_trader({"intent": intent, "trace": []}), "debate_risk")


if __name__ == "__main__":
    unittest.main()
