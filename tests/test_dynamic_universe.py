import unittest

import pandas as pd

from core.schemas import BacktestConfig
from data.universe_service import DynamicUniverseService
from engine.backtest import TradingBacktestEngine


class FakeUniverseProvider:
    def list_sectors(self):
        return pd.DataFrame(
            [
                {"sector": "银行", "change_pct": 2.0},
                {"sector": "电力设备", "change_pct": 1.0},
            ]
        )

    def list_sector_stocks(self, sector):
        base = [
            {
                "symbol": "000001" if sector == "银行" else "300750",
                "name": "样本股票",
                "close": 10.0,
                "change_pct": 1.0,
                "volume": 100000,
                "turnover": 100_000_000.0,
                "sector": sector,
            },
            {
                "symbol": "000002" if sector == "银行" else "601012",
                "name": "ST过滤样本",
                "close": 8.0,
                "change_pct": -1.0,
                "volume": 100000,
                "turnover": 90_000_000.0,
                "sector": sector,
            },
        ]
        return pd.DataFrame(base)


class FakeSpotUniverseProvider:
    def list_sectors(self):
        raise RuntimeError("industry endpoint unavailable")

    def list_spot(self):
        return pd.DataFrame(
            [
                {
                    "symbol": "000001",
                    "name": "示例银行",
                    "close": 10.0,
                    "change_pct": 0.01,
                    "volume": 100000,
                    "turnover": 100_000_000.0,
                    "sector": "未知",
                },
                {
                    "symbol": "300750",
                    "name": "正常样本",
                    "close": 20.0,
                    "change_pct": 0.02,
                    "volume": 100000,
                    "turnover": 120_000_000.0,
                    "sector": "未分类",
                },
                {
                    "symbol": "000002",
                    "name": "ST过滤样本",
                    "close": 8.0,
                    "change_pct": -0.01,
                    "volume": 100000,
                    "turnover": 90_000_000.0,
                    "sector": "未知",
                },
            ]
        )


class DynamicUniverseTest(unittest.TestCase):
    def test_dynamic_universe_filters_and_preserves_sector(self):
        engine = TradingBacktestEngine.__new__(TradingBacktestEngine)
        engine.config = BacktestConfig(
            provider="akshare",
            dynamic_universe=True,
            dynamic_universe_limit=10,
            dynamic_sector_pool=2,
            dynamic_min_turnover=50_000_000,
        )
        engine.universe_service = DynamicUniverseService(FakeUniverseProvider)
        engine._injected_universe_snapshot = None
        frame = engine._resolve_dynamic_universe()
        self.assertEqual(set(frame["symbol"]), {"000001", "300750"})
        self.assertEqual(set(frame["sector"]), {"银行", "电力设备"})
        self.assertEqual(engine.active_universe_snapshot.source, "akshare:industry_boards")

    def test_spot_fallback_keeps_existing_filters_and_sector_fallback(self):
        config = BacktestConfig(
            provider="akshare",
            dynamic_universe=True,
            dynamic_universe_limit=10,
            dynamic_min_turnover=50_000_000,
        )
        snapshot = DynamicUniverseService(FakeSpotUniverseProvider).capture(config)
        frame = snapshot.to_frame()

        self.assertEqual(snapshot.source, "akshare:spot")
        self.assertIsNone(snapshot.as_of)
        self.assertEqual(frame["symbol"].tolist(), ["300750", "000001"])
        self.assertEqual(frame["sector"].tolist(), ["创业板", "银行"])


if __name__ == "__main__":
    unittest.main()
