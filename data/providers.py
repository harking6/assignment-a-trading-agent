from __future__ import annotations

import csv
import hashlib
import math
import os
from abc import ABC, abstractmethod
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, List

import pandas as pd

from core.schemas import MarketBar, QuoteTick


class ProviderUnavailable(RuntimeError):
    pass


class MarketDataProvider(ABC):
    @abstractmethod
    def load(
        self,
        dataset: str,
        symbol: str = "DEMO",
        start: str | None = None,
        end: str | None = None,
        adjust: str | None = None,
    ) -> List[MarketBar]:
        raise NotImplementedError

    def latest_quote(self, dataset: str, symbol: str = "DEMO") -> QuoteTick:
        bars = self.load(dataset, symbol)
        if not bars:
            raise ProviderUnavailable("provider returned no rows")
        return _bar_to_quote(bars[-1], provider=self.__class__.__name__, symbol=symbol, source="historical-fallback")

    def instrument_info(self, symbol: str = "DEMO") -> dict[str, str]:
        return {"symbol": symbol, "name": symbol, "asset_type": "unknown", "market": "", "note": ""}


class SyntheticDataProvider(MarketDataProvider):
    """Deterministic OHLCV generator for offline classroom runs.

    Different symbols receive different but still deterministic price paths so
    multi-symbol backtests do not collapse into identical series.
    """

    def __init__(self, total_days: int = 180) -> None:
        self.total_days = total_days

    def load(
        self,
        dataset: str,
        symbol: str = "DEMO",
        start: str | None = None,
        end: str | None = None,
        adjust: str | None = None,
    ) -> List[MarketBar]:
        symbol_offset = self._symbol_offset(symbol)
        start_date = date(2026, 1, 5)
        base_price = 120.0 if dataset == "stress" else 96.0
        price = base_price + symbol_offset * 8.0
        bars: List[MarketBar] = []
        for idx in range(self.total_days):
            phase = idx + symbol_offset * 3.7
            wave = math.sin(phase / 4.8) * 1.7 + math.cos(phase / 9.5) * 1.2
            cycle = math.sin(phase / 2.7) * 0.7
            drift = 0.22
            if dataset == "range":
                drift = math.sin(phase / 8.0) * 0.16
            elif dataset == "stress":
                drift = 0.12 if idx < 48 else -0.72 if idx < 92 else 0.36
            elif dataset == "event":
                drift = 0.36 if idx < 70 else -0.22 if idx < 110 else 0.18

            previous = price
            close = max(38.0, previous + drift + wave * 0.18 + cycle)
            open_price = previous + math.sin(phase / 3.2) * 0.35
            high = max(open_price, close) + 0.55 + abs(wave) * 0.16
            low = min(open_price, close) - 0.55 - abs(cycle) * 0.13
            volume = round(8500 + abs(wave) * 1700 + idx * 22 + (1800 if dataset == "event" and 66 < idx < 78 else 0))
            bars.append(
                MarketBar(
                    day=idx + 1,
                    date=(start_date + timedelta(days=idx)).isoformat(),
                    open=round(open_price, 2),
                    high=round(high, 2),
                    low=round(low, 2),
                    close=round(close, 2),
                    volume=volume,
                )
            )
            price = close
        return bars

    @staticmethod
    def _symbol_offset(symbol: str) -> float:
        """Map a symbol to a deterministic float in [-1, 1]."""
        if not symbol or symbol == "DEMO":
            return 0.0
        # Python randomizes hash() between interpreter processes.  A stable
        # digest keeps classroom backtests reproducible after every restart.
        hashed = int.from_bytes(hashlib.sha256(symbol.encode("utf-8")).digest()[:4], "big") % 10000
        return (hashed / 10000.0) * 2.0 - 1.0

    def latest_quote(self, dataset: str, symbol: str = "DEMO") -> QuoteTick:
        bars = self.load(dataset, symbol)
        idx = min(len(bars) - 1, max(0, datetime.now().second % len(bars)))
        return _bar_to_quote(bars[idx], provider="synthetic", symbol=symbol, source="synthetic-live-clock")

    def instrument_info(self, symbol: str = "DEMO") -> dict[str, str]:
        return {
            "symbol": symbol,
            "name": "合成行情样本",
            "asset_type": "synthetic",
            "market": "offline",
            "note": "教学用模拟价格，不是真实股票/指数行情",
        }


class CSVDataProvider(MarketDataProvider):
    """CSV provider for course datasets with day/open/high/low/close/volume."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.synthetic = SyntheticDataProvider()

    def load(
        self,
        dataset: str,
        symbol: str = "DEMO",
        start: str | None = None,
        end: str | None = None,
        adjust: str | None = None,
    ) -> List[MarketBar]:
        path = self.data_dir / f"{dataset}.csv"
        if not path.exists():
            path = self.data_dir / "sample_prices.csv"
        if not path.exists():
            return self.synthetic.load(dataset, symbol, start, end, adjust)

        rows: List[MarketBar] = []
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for idx, row in enumerate(reader):
                close = float(row.get("close") or row.get("price") or 0)
                open_price = float(row.get("open") or close)
                high = float(row.get("high") or max(open_price, close))
                low = float(row.get("low") or min(open_price, close))
                rows.append(
                    MarketBar(
                        day=int(row.get("day") or idx + 1),
                        date=row.get("date") or f"sample-{idx + 1}",
                        open=open_price,
                        high=high,
                        low=low,
                        close=close,
                        volume=int(float(row.get("volume") or 0)),
                    )
                )
        return rows or self.synthetic.load(dataset, symbol, start, end, adjust)

    def latest_quote(self, dataset: str, symbol: str = "DEMO") -> QuoteTick:
        bars = self.load(dataset, symbol)
        return _bar_to_quote(bars[-1], provider="csv", symbol=symbol, source="csv-last-row")

    def instrument_info(self, symbol: str = "DEMO") -> dict[str, str]:
        return {"symbol": symbol, "name": "本地CSV标的", "asset_type": "csv", "market": "local", "note": "名称来自本地数据集配置"}


class AkShareDataProvider(MarketDataProvider):
    def load(
        self,
        dataset: str,
        symbol: str = "000001",
        start: str | None = None,
        end: str | None = None,
        adjust: str | None = None,
    ) -> List[MarketBar]:
        try:
            import akshare as ak  # type: ignore
        except Exception as exc:
            raise ProviderUnavailable("akshare is not installed. Install optional dependency: pip install akshare") from exc
        start_date = (start or "20240101").replace("-", "")
        end_date = (end or "20261231").replace("-", "")
        code = _cn_code(symbol)
        errors: list[str] = []

        # Primary: Eastmoney endpoints (adjustable, index-aware).
        try:
            if _is_cn_index_symbol(symbol):
                frame = ak.index_zh_a_hist(symbol=code, period="daily", start_date=start_date, end_date=end_date)
            else:
                frame = ak.stock_zh_a_hist(
                    symbol=code,
                    period="daily",
                    start_date=start_date,
                    end_date=end_date,
                    adjust=_akshare_adjust(adjust),
                )
            return _bars_from_chinese_frame(frame)
        except Exception as exc:
            errors.append(f"eastmoney: {exc}")

        # Fallback: Tencent historical data. Tencent exposes different
        # endpoints for indices and ordinary A-share securities.
        try:
            tx_symbol = _tx_symbol(symbol)
            if _is_cn_index_symbol(symbol):
                frame = ak.stock_zh_index_daily_tx(
                    symbol=tx_symbol,
                    start_date=start_date,
                    end_date=end_date,
                )
            else:
                frame = ak.stock_zh_a_hist_tx(
                    symbol=tx_symbol,
                    start_date=start_date,
                    end_date=end_date,
                    adjust=_akshare_adjust(adjust),
                )
            return _bars_from_chinese_frame(frame)
        except Exception as exc:
            errors.append(f"tencent: {exc}")

        raise ProviderUnavailable(f"akshare failed to load {symbol}: {'; '.join(errors)}")

    def latest_quote(self, dataset: str, symbol: str = "000001") -> QuoteTick:
        try:
            import akshare as ak  # type: ignore
        except Exception as exc:
            raise ProviderUnavailable("akshare is not installed. Install optional dependency: pip install akshare") from exc
        is_index = _is_cn_index_symbol(symbol)
        frame = ak.stock_zh_index_spot_em() if is_index else ak.stock_zh_a_spot_em()
        code_col = _first_existing(frame, ["代码", "code"])
        if code_col is None:
            raise ProviderUnavailable("akshare realtime frame has no code column")
        normalized = _cn_code(symbol)
        rows = frame[frame[code_col].astype(str).map(_cn_code) == normalized]
        if rows.empty:
            raise ProviderUnavailable(f"akshare realtime quote not found for {symbol}")
        row = rows.iloc[0]
        return QuoteTick(
            provider="akshare",
            symbol=symbol,
            timestamp=datetime.now().isoformat(timespec="seconds"),
            price=_num(row, ["最新价", "price", "close"]),
            open=_num(row, ["今开", "open"], none_ok=True),
            high=_num(row, ["最高", "high"], none_ok=True),
            low=_num(row, ["最低", "low"], none_ok=True),
            prev_close=_num(row, ["昨收", "pre_close", "prev_close"], none_ok=True),
            volume=int(_num(row, ["成交量", "volume"], default=0)),
            source="akshare.stock_zh_index_spot_em" if is_index else "akshare.stock_zh_a_spot_em",
            raw=_safe_row(row),
        )

    def instrument_info(self, symbol: str = "000001") -> dict[str, str]:
        return _cn_instrument_info(symbol)


class YFinanceDataProvider(MarketDataProvider):
    def load(
        self,
        dataset: str,
        symbol: str = "AAPL",
        start: str | None = None,
        end: str | None = None,
        adjust: str | None = None,
    ) -> List[MarketBar]:
        try:
            import yfinance as yf  # type: ignore
        except Exception as exc:
            raise ProviderUnavailable("yfinance is not installed. Install optional dependency: pip install yfinance") from exc
        frame = yf.download(symbol, start=start or "2024-01-01", end=end, auto_adjust=False, progress=False)
        if frame.empty:
            raise ProviderUnavailable(f"yfinance returned no rows for {symbol}")
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = [col[0] for col in frame.columns]
        rows: List[MarketBar] = []
        for idx, (_, row) in enumerate(frame.reset_index().iterrows()):
            date_value = str(row.get("Date") or row.get("Datetime") or idx + 1)[:10]
            rows.append(
                MarketBar(
                    day=idx + 1,
                    date=date_value,
                    open=float(row["Open"]),
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    close=float(row["Close"]),
                    volume=int(float(row.get("Volume", 0) or 0)),
                )
            )
        return rows

    def latest_quote(self, dataset: str, symbol: str = "AAPL") -> QuoteTick:
        try:
            import yfinance as yf  # type: ignore
        except Exception as exc:
            raise ProviderUnavailable("yfinance is not installed. Install optional dependency: pip install yfinance") from exc
        ticker = yf.Ticker(symbol)
        frame = ticker.history(period="1d", interval="1m", auto_adjust=False)
        if frame.empty:
            frame = ticker.history(period="5d", interval="1d", auto_adjust=False)
        if frame.empty:
            raise ProviderUnavailable(f"yfinance returned no realtime rows for {symbol}")
        row = frame.iloc[-1]
        info = getattr(ticker, "fast_info", {}) or {}
        return QuoteTick(
            provider="yfinance",
            symbol=symbol,
            timestamp=str(frame.index[-1]),
            price=float(row["Close"]),
            open=float(row.get("Open", row["Close"])),
            high=float(row.get("High", row["Close"])),
            low=float(row.get("Low", row["Close"])),
            prev_close=_float_or_none(info.get("previous_close") or info.get("previousClose")),
            volume=int(float(row.get("Volume", 0) or 0)),
            source="yfinance.history(1m)",
            raw={"fast_info": {str(k): str(v) for k, v in dict(info).items()}},
        )

    def instrument_info(self, symbol: str = "AAPL") -> dict[str, str]:
        known = {"AAPL": "Apple Inc.", "MSFT": "Microsoft Corp.", "SPY": "SPDR S&P 500 ETF", "^GSPC": "S&P 500 Index"}
        return {"symbol": symbol, "name": known.get(symbol.upper(), symbol.upper()), "asset_type": "equity_or_etf", "market": "US", "note": ""}


class TushareDataProvider(MarketDataProvider):
    def load(
        self,
        dataset: str,
        symbol: str = "000001.SZ",
        start: str | None = None,
        end: str | None = None,
        adjust: str | None = None,
    ) -> List[MarketBar]:
        token = os.getenv("TUSHARE_TOKEN")
        if not token:
            raise ProviderUnavailable("TUSHARE_TOKEN is missing. Put it in environment or .env before using tushare.")
        try:
            import tushare as ts  # type: ignore
        except Exception as exc:
            raise ProviderUnavailable("tushare is not installed. Install optional dependency: pip install tushare") from exc
        pro = ts.pro_api(token)
        ts_code = _tushare_code(symbol, index=_is_cn_index_symbol(symbol))
        loader = pro.index_daily if _is_cn_index_symbol(symbol) else pro.daily
        frame = loader(ts_code=ts_code, start_date=(start or "20240101").replace("-", ""), end_date=(end or "20261231").replace("-", ""))
        if frame.empty:
            raise ProviderUnavailable(f"tushare returned no rows for {symbol}")
        frame = frame.sort_values("trade_date")
        rows: List[MarketBar] = []
        for idx, (_, row) in enumerate(frame.iterrows()):
            rows.append(
                MarketBar(
                    day=idx + 1,
                    date=str(row["trade_date"]),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=int(float(row.get("vol", 0) or 0)),
                )
            )
        return rows

    def latest_quote(self, dataset: str, symbol: str = "000001.SZ") -> QuoteTick:
        token = os.getenv("TUSHARE_TOKEN")
        if not token:
            raise ProviderUnavailable("TUSHARE_TOKEN is missing. Put it in environment or .env before using tushare.")
        try:
            import tushare as ts  # type: ignore
        except Exception as exc:
            raise ProviderUnavailable("tushare is not installed. Install optional dependency: pip install tushare") from exc
        if not hasattr(ts, "realtime_quote"):
            raise ProviderUnavailable("installed tushare does not expose realtime_quote; use AkShare/yfinance or upgrade tushare")
        frame = ts.realtime_quote(ts_code=symbol)
        if frame.empty:
            raise ProviderUnavailable(f"tushare realtime_quote returned no rows for {symbol}")
        row = frame.iloc[0]
        return QuoteTick(
            provider="tushare",
            symbol=symbol,
            timestamp=datetime.now().isoformat(timespec="seconds"),
            price=_num(row, ["PRICE", "price", "现价", "最新价"]),
            open=_num(row, ["OPEN", "open", "今开"], none_ok=True),
            high=_num(row, ["HIGH", "high", "最高"], none_ok=True),
            low=_num(row, ["LOW", "low", "最低"], none_ok=True),
            prev_close=_num(row, ["PRE_CLOSE", "pre_close", "昨收"], none_ok=True),
            volume=int(_num(row, ["VOLUME", "volume", "成交量"], default=0)),
            source="tushare.realtime_quote",
            raw=_safe_row(row),
        )

    def instrument_info(self, symbol: str = "000001.SZ") -> dict[str, str]:
        return _cn_instrument_info(symbol)


def bars_to_frame(bars: Iterable[MarketBar]) -> pd.DataFrame:
    frame = pd.DataFrame([bar.model_dump() for bar in bars])
    if not frame.empty:
        frame = frame.set_index("day", drop=False)
    return frame


def build_provider(name: str, data_dir: Path) -> MarketDataProvider:
    if name == "csv":
        return CSVDataProvider(data_dir)
    if name == "akshare":
        return AkShareDataProvider()
    if name == "yfinance":
        return YFinanceDataProvider()
    if name == "tushare":
        return TushareDataProvider()
    return SyntheticDataProvider()


# Bare 6-digit codes that are unambiguously CSI/SH indices (no A-share stock
# shares these codes), so a symbol like "000300" can be detected as an index
# without forcing the caller to pass "SH000300". 000001/000002 are NOT here —
# those are SZ stocks (平安银行/万科), so they stay stock-side.
_BARE_INDEX_CODES = {
    # 沪市指数（000xxx — 注意：不含 000001，那是平安银行）
    "000010", "000015", "000016", "000132", "000300",
    "000688", "000852", "000905",
    # 深市指数（399xxx）
    "399001", "399005", "399006",
}


def _is_cn_index_symbol(symbol: str) -> bool:
    normalized = symbol.strip().upper()
    if normalized.startswith("SH000") or normalized.startswith("SZ399"):
        return True
    if "." not in normalized:
        return normalized in _BARE_INDEX_CODES
    code, market = normalized.split(".", 1)
    return (market == "SH" and code.startswith("000")) or (market == "SZ" and code.startswith("399"))


def _cn_code(symbol: object) -> str:
    normalized = str(symbol).strip().upper()
    if normalized.startswith(("SH", "SZ")) and len(normalized) >= 8:
        normalized = normalized[2:]
    if "." in normalized:
        normalized = normalized.split(".", 1)[0]
    digits = "".join(char for char in normalized if char.isdigit())
    return digits.zfill(6) if digits else normalized


def _tushare_code(symbol: str, index: bool = False) -> str:
    normalized = symbol.strip().upper()
    if "." in normalized:
        return normalized
    if normalized.startswith("SH") and len(normalized) >= 8:
        return f"{normalized[2:]}.SH"
    if normalized.startswith("SZ") and len(normalized) >= 8:
        return f"{normalized[2:]}.SZ"
    code = _cn_code(normalized)
    if index:
        market = "SZ" if code.startswith("399") else "SH"
    else:
        market = "SH" if code.startswith(("5", "6", "9")) else "SZ"
    return f"{code}.{market}"


def _akshare_adjust(adjust: str | None) -> str:
    normalized = (adjust or "").strip().lower()
    if normalized in {"qfq", "前复权"}:
        return "qfq"
    if normalized in {"hfq", "后复权"}:
        return "hfq"
    return ""


def _tx_symbol(symbol: str) -> str:
    """Map an A-share/index symbol to the Tencent ``sh/sz`` prefix format."""

    normalized = str(symbol).strip().upper()
    if normalized.startswith(("SH", "SZ")) and len(normalized) >= 8:
        return normalized.lower()
    code = _cn_code(normalized)
    if _is_cn_index_symbol(normalized):
        market = "SZ" if code.startswith("399") else "SH"
    else:
        market = "SH" if code.startswith(("6", "9", "5")) else "SZ"
    return f"{market.lower()}{code}"


def _cn_instrument_info(symbol: str) -> dict[str, str]:
    code = _cn_code(symbol)
    is_index = _is_cn_index_symbol(symbol)
    known = {
        ("000001", "stock"): ("平安银行", "A股股票", "SZ"),
        ("600000", "stock"): ("浦发银行", "A股股票", "SH"),
        ("000001", "index"): ("上证指数", "A股指数", "SH"),
        ("000300", "index"): ("沪深300", "A股指数", "SH"),
        ("399001", "index"): ("深证成指", "A股指数", "SZ"),
        ("399006", "index"): ("创业板指", "A股指数", "SZ"),
    }
    name, asset_type, market = known.get((code, "index" if is_index else "stock"), (symbol, "A股指数" if is_index else "A股股票", ""))
    note = "裸 000001 在股票接口中是平安银行；上证指数请用 000001.SH 或 sh000001" if code == "000001" and not is_index else ""
    return {"symbol": symbol, "name": name, "asset_type": asset_type, "market": market, "note": note}


def _bars_from_chinese_frame(frame: pd.DataFrame) -> List[MarketBar]:
    if frame.empty:
        raise ProviderUnavailable("provider returned no rows")
    rows: List[MarketBar] = []
    for idx, (_, row) in enumerate(frame.iterrows()):
        rows.append(
            MarketBar(
                day=idx + 1,
                date=str(row.get("日期") or row.get("date") or idx + 1),
                open=float(row.get("开盘") or row.get("open")),
                high=float(row.get("最高") or row.get("high")),
                low=float(row.get("最低") or row.get("low")),
                close=float(row.get("收盘") or row.get("close")),
                volume=int(float(row.get("成交量") or row.get("volume") or 0)),
            )
        )
    return rows


def _bar_to_quote(bar: MarketBar, provider: str, symbol: str, source: str) -> QuoteTick:
    return QuoteTick(
        provider=provider,
        symbol=symbol,
        timestamp=f"{bar.date}T15:00:00",
        price=bar.close,
        open=bar.open,
        high=bar.high,
        low=bar.low,
        prev_close=None,
        volume=bar.volume,
        source=source,
        raw=bar.public_dict(),
    )


def _first_existing(frame: pd.DataFrame, names: list[str]) -> str | None:
    for name in names:
        if name in frame.columns:
            return name
    return None


def _float_or_none(value: object) -> float | None:
    try:
        if value in (None, "", "-", "None"):
            return None
        result = float(value)
        if math.isnan(result):
            return None
        return result
    except Exception:
        return None


def _num(row: pd.Series, names: list[str], default: float | None = None, none_ok: bool = False) -> float:
    for name in names:
        if name in row:
            parsed = _float_or_none(row[name])
            if parsed is not None:
                return parsed
    if none_ok:
        return None  # type: ignore[return-value]
    if default is not None:
        return default
    raise ProviderUnavailable(f"missing numeric field among {names}")


def _safe_row(row: pd.Series) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in row.to_dict().items():
        try:
            jsonable = value.item() if hasattr(value, "item") else value
        except Exception:
            jsonable = str(value)
        result[str(key)] = jsonable if isinstance(jsonable, (str, int, float, bool)) or jsonable is None else str(jsonable)
    return result
