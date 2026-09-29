import unittest

from agents.graph import TradingAgentsGraph
from core.schemas import BacktestConfig, PortfolioSnapshot


class FakeMultiLLM:
    enabled = True
    last_runtime = "fake-llm"
    last_error = None
    deep_model = "fake-model"
    quick_model = "fake-model"
    mode = "llm"

    def __init__(self) -> None:
        self.roles = []

    def complete_json(self, role, _prompt, deep=False):
        self.roles.append((role, deep))
        if role == "multi_market_regime_agent":
            return {
                "market_view": "bullish",
                "confidence": 0.8,
                "reasoning": "broad participation",
                "risk_budget_scale": 1.0,
            }
        if role == "multi_sector_selector":
            return {}
        if role == "multi_stock_selector":
            return {}
        if role == "multi_candidate_analyst_team":
            return {
                "analyses": [
                    {
                        "symbol": "000001",
                        "technical_score": 0.6,
                        "sentiment_score": 0.0,
                        "fundamental_score": 0.0,
                        "overall_score": 0.4,
                        "bull_case": "positive trend",
                        "bear_case": "limited evidence",
                        "data_gaps": ["news", "fundamentals"],
                    }
                ]
            }
        if role == "multi_bull_bear_research_manager":
            return {
                "ranked_symbols": ["000001", "000002", "600000"],
                "conviction": 0.7,
                "manager_notes": "ranked by supplied evidence",
                "rounds": [],
            }
        if role == "multi_portfolio_manager":
            return {
                "weights": {"000001": 0.4, "000002": 0.35, "600000": 0.25},
                "cash_ratio": 0.1,
                "reasoning": "diversified allocation",
            }
        raise AssertionError(f"unexpected role: {role}")


class MultiLLMPipelineTest(unittest.TestCase):
    def test_all_multi_llm_roles_are_invoked(self) -> None:
        config = BacktestConfig(
            provider="synthetic",
            symbol="DEMO",
            universe=["000001", "000002", "600000"],
            agent_mode="llm",
            top_n_sectors=3,
            stocks_per_sector=2,
        )
        graph = TradingAgentsGraph()
        graph.reconfigure_selector(config)
        fake = FakeMultiLLM()
        graph.top_down.llm = fake
        portfolio = PortfolioSnapshot(
            cash=100000,
            equity=100000,
            peak=100000,
            max_drawdown=0.0,
        )
        prices = {"000001": 10.0, "000002": 12.0, "600000": 8.0}
        technical = {
            symbol: {
                "return_1d": 0.01,
                "rsi_14": 55.0,
                "volatility_20": 0.02,
                "sma_5": price * 1.01,
                "sma_20": price,
                "sma_60": price * 0.98,
                "atr_14": 0.2,
                "volume_z": 0.5,
            }
            for symbol, price in prices.items()
        }
        result = graph.run_multi(
            day=0,
            portfolio=portfolio,
            config=config,
            market_snapshot={"market_view": "neutral", "benchmark_return": 0.0},
            prices=prices,
            technical_context=technical,
        )
        roles = [role for role, _deep in fake.roles]
        self.assertEqual(
            roles,
            [
                "multi_market_regime_agent",
                "multi_sector_selector",
                "multi_stock_selector",
                "multi_candidate_analyst_team",
                "multi_bull_bear_research_manager",
                "multi_portfolio_manager",
            ],
        )
        self.assertTrue(result.target_positions)
        self.assertIn("candidate_analysis", result.sector_view)
        self.assertIn("debate_result", result.sector_view)
        self.assertIn("portfolio_plan", result.sector_view)

    def test_llm_mode_requires_an_available_model(self) -> None:
        config = BacktestConfig(provider="synthetic", universe=["000001"], agent_mode="llm")
        graph = TradingAgentsGraph()
        graph.reconfigure_selector(config)
        graph.top_down.llm.mode = "offline"
        portfolio = PortfolioSnapshot(cash=100000, equity=100000, peak=100000, max_drawdown=0.0)
        with self.assertRaisesRegex(RuntimeError, "requires OPENAI_API_KEY"):
            graph.run_multi(0, portfolio, config, {}, {"000001": 10.0}, {})


if __name__ == "__main__":
    unittest.main()
