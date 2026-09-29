import unittest
from unittest.mock import patch

import pandas as pd

from core.schemas import BacktestConfig
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
        with patch("engine.backtest.AShareUniverseProvider", FakeUniverseProvider):
            frame = engine._resolve_dynamic_universe()
        self.assertEqual(set(frame["symbol"]), {"000001", "300750"})
        self.assertEqual(set(frame["sector"]), {"银行", "电力设备"})


if __name__ == "__main__":
    unittest.main()
