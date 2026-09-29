from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd


class UniverseProviderError(RuntimeError):
    pass


# Module-level cache for the SW industry map. Building it requires ~30 HTTP
# calls (one per SW first-level industry), so it is built once per process and
# shared across all AShareUniverseProvider instances — every engine reset
# would otherwise pay that 10s cost again.
_INDUSTRY_MAP_CACHE: Dict[str, str] | None = None


class AShareUniverseProvider:
    """A-share universe, sector, and constituent data via akshare.

    This provider is read-only and returns normalized DataFrames so the
    top-down selector can work with a consistent schema regardless of the
    underlying akshare column names.
    """

    def __init__(self) -> None:
        try:
            import akshare as ak  # type: ignore
        except Exception as exc:
            raise UniverseProviderError("akshare is not installed. Install optional dependency: pip install akshare") from exc
        self._ak = ak

    def industry_map(self) -> Dict[str, str]:
        """Map A-share symbol -> Shenwan first-level industry name.

        Built once per provider instance from the Shenwan first-level industry
        index constituents (``sw_index_first_info`` + ``index_component_sw``)
        and cached. Eastmoney's industry-board endpoints are unreachable from
        this network, but the Shenwan endpoints are not, so this is the real
        industry label source that makes sector-relative-strength selection
        meaningful (Sina's spot table has no sector column of its own).
        """
        global _INDUSTRY_MAP_CACHE
        if _INDUSTRY_MAP_CACHE is not None:
            return _INDUSTRY_MAP_CACHE
        mapping: Dict[str, str] = {}
        try:
            industries = self._ak.sw_index_first_info()
            for code, name in zip(
                industries["行业代码"].astype(str),
                industries["行业名称"].astype(str),
            ):
                bare = code.split(".")[0]  # '801010.SI' -> '801010'
                try:
                    cons = self._ak.index_component_sw(symbol=bare)
                except Exception:
                    continue
                if cons is None or cons.empty or "证券代码" not in cons.columns:
                    continue
                for sym in cons["证券代码"].astype(str).str.zfill(6):
                    mapping[sym] = name
        except Exception:
            pass
        _INDUSTRY_MAP_CACHE = mapping
        return mapping

    def list_spot(self) -> pd.DataFrame:
        """Return full-market A-share spot quote.

        Primary source is eastmoney's ``stock_zh_a_spot_em``; when eastmoney
        blocks the connection (common from this network), fall back to Sina's
        ``stock_zh_a_spot``. Both results are normalized to turnover in yuan
        and volume in shares before applying the liquidity filter.
        """
        errors: List[str] = []
        try:
            frame = self._ak.stock_zh_a_spot_em()
            return self._normalize_spot(frame)
        except Exception as exc:
            errors.append(f"eastmoney: {exc}")
        try:
            frame = self._ak.stock_zh_a_spot()
            return self._normalize_spot_sina(frame)
        except Exception as exc:
            errors.append(f"sina: {exc}")
        raise UniverseProviderError(f"failed to load A-share spot data: {'; '.join(errors)}")

    def _normalize_spot_sina(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Normalize Sina spot quote.

        Sina reports 成交量 in 股 (shares) and 成交额 in 元 (yuan), so no unit
        conversion is needed — the liquidity filter compares turnover in yuan.
        The 涨跌幅 column is in percentage points (e.g. 1.877 = 1.877%), so
        it is divided by 100 here to a decimal fraction for consistency with
        benchmark_return and the engine's return_1d, which are both decimals.
        """
        if frame is None or frame.empty:
            return pd.DataFrame(columns=["symbol", "name", "close", "change_pct", "volume", "turnover", "sector"])
        col = lambda *names: self._first_existing(frame, list(names))
        code_col = col("代码", "code", "股票代码")
        name_col = col("名称", "name", "股票名称")
        close_col = col("最新价", "close", "收盘价")
        change_col = col("涨跌幅", "change_pct")
        volume_col = col("成交量", "volume")
        turnover_col = col("成交额", "turnover", "amount")
        normalized = pd.DataFrame(
            {
                "symbol": frame[code_col].astype(str).str.strip().str.replace(r"^(sh|sz|bj)", "", regex=True) if code_col else None,
                "name": frame[name_col].astype(str).str.strip() if name_col else None,
                "close": pd.to_numeric(frame[close_col], errors="coerce") if close_col else None,
                "change_pct": (pd.to_numeric(frame[change_col], errors="coerce").fillna(0.0) / 100.0) if change_col else 0.0,
                "volume": pd.to_numeric(frame[volume_col], errors="coerce").fillna(0).astype(int) if volume_col else 0,
                "turnover": pd.to_numeric(frame[turnover_col], errors="coerce").fillna(0.0) if turnover_col else 0.0,
            }
        )
        normalized["sector"] = normalized["symbol"].map(self.industry_map()).fillna("未分类")
        return normalized.dropna(subset=["symbol"]).reset_index(drop=True)


    def list_sectors(self) -> pd.DataFrame:
        """Return industry board list."""
        try:
            frame = self._ak.stock_board_industry_name_em()
        except Exception as exc:
            raise UniverseProviderError(f"failed to load sector list: {exc}") from exc
        return self._normalize_sectors(frame)

    def list_sector_stocks(self, sector: str) -> pd.DataFrame:
        """Return constituent stocks for an industry board."""
        try:
            frame = self._ak.stock_board_industry_cons_em(symbol=sector)
        except Exception as exc:
            raise UniverseProviderError(f"failed to load sector stocks for {sector}: {exc}") from exc
        return self._normalize_spot(frame, sector=sector)

    def list_index_constituents(self, index: str) -> pd.DataFrame:
        """Return index constituent stocks.

        ``index`` is the akshare symbol, e.g. ``"000300"`` for CSI 300.
        """
        try:
            frame = self._ak.index_stock_cons(symbol=index)
        except Exception as exc:
            raise UniverseProviderError(f"failed to load index constituents for {index}: {exc}") from exc
        return self._normalize_constituents(frame)

    def sector_performance(self, sector: str) -> Dict[str, float]:
        """Aggregate performance metrics for a sector from its constituents."""
        stocks = self.list_sector_stocks(sector)
        if stocks.empty:
            return {
                "sector": sector,
                "change_pct": 0.0,
                "avg_change_pct": 0.0,
                "median_change_pct": 0.0,
                "up_count": 0,
                "down_count": 0,
                "constituent_count": 0,
            }
        change_col = "change_pct"
        if change_col not in stocks.columns:
            change_col = next((c for c in stocks.columns if "涨跌幅" in c or "涨跌" in c), None)
        if change_col and change_col in stocks.columns:
            series = pd.to_numeric(stocks[change_col], errors="coerce").dropna()
        else:
            series = pd.Series(dtype=float)
        return {
            "sector": sector,
            "change_pct": round(float(series.mean()), 4) if len(series) else 0.0,
            "avg_change_pct": round(float(series.mean()), 4) if len(series) else 0.0,
            "median_change_pct": round(float(series.median()), 4) if len(series) else 0.0,
            "up_count": int((series > 0).sum()),
            "down_count": int((series < 0).sum()),
            "constituent_count": len(stocks),
        }

    def _normalize_spot(self, frame: pd.DataFrame, sector: Optional[str] = None) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame(columns=["symbol", "name", "close", "change_pct", "volume", "turnover", "sector"])
        code_col = self._first_existing(frame, ["代码", "code", "股票代码", "symbol"])
        name_col = self._first_existing(frame, ["名称", "name", "股票名称", "股票简称"])
        close_col = self._first_existing(frame, ["最新价", "close", "收盘价", "最新价"])
        change_col = self._first_existing(frame, ["涨跌幅", "change_pct", "涨跌(%)"])
        volume_col = self._first_existing(frame, ["成交量", "volume"])
        turnover_col = self._first_existing(frame, ["成交额", "turnover", "amount"])

        normalized = pd.DataFrame(
            {
                "symbol": frame[code_col].astype(str).str.strip() if code_col else None,
                "name": frame[name_col].astype(str).str.strip() if name_col else None,
                "close": pd.to_numeric(frame[close_col], errors="coerce") if close_col else None,
                "change_pct": (pd.to_numeric(frame[change_col], errors="coerce").fillna(0.0) / 100.0) if change_col else 0.0,
                "volume": pd.to_numeric(frame[volume_col], errors="coerce").fillna(0).astype(int) if volume_col else 0,
                "turnover": pd.to_numeric(frame[turnover_col], errors="coerce").fillna(0.0) if turnover_col else 0.0,
            }
        )
        if sector:
            normalized["sector"] = sector
        else:
            sector_col = self._first_existing(frame, ["所属行业", "sector", "行业"])
            normalized["sector"] = frame[sector_col].astype(str).str.strip() if sector_col else "未知"
        # If eastmoney returned no sector column (happens intermittently), fall
        # back to the Shenwan industry map so the top-down selector still sees
        # real industry labels instead of "未知".
        if (normalized["sector"] == "未知").all():
            normalized["sector"] = normalized["symbol"].map(self.industry_map()).fillna("未分类")
        return normalized.dropna(subset=["symbol"]).reset_index(drop=True)

    def _normalize_sectors(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame(columns=["sector", "change_pct", "constituent_count"])
        name_col = self._first_existing(frame, ["板块名称", "行业名称", "name", "sector"])
        change_col = self._first_existing(frame, ["涨跌幅", "change_pct"])
        count_col = self._first_existing(frame, ["家数", "constituent_count", "count"])
        return pd.DataFrame(
            {
                "sector": frame[name_col].astype(str).str.strip() if name_col else None,
                "change_pct": pd.to_numeric(frame[change_col], errors="coerce").fillna(0.0) if change_col else 0.0,
                "constituent_count": pd.to_numeric(frame[count_col], errors="coerce").fillna(0).astype(int) if count_col else 0,
            }
        ).dropna(subset=["sector"]).reset_index(drop=True)

    def _normalize_constituents(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame(columns=["symbol", "name"])
        code_col = self._first_existing(frame, ["代码", "code", "成分股代码", "股票代码"])
        name_col = self._first_existing(frame, ["名称", "name", "成分股名称", "股票名称"])
        return pd.DataFrame(
            {
                "symbol": frame[code_col].astype(str).str.strip() if code_col else None,
                "name": frame[name_col].astype(str).str.strip() if name_col else None,
            }
        ).dropna(subset=["symbol"]).reset_index(drop=True)

    @staticmethod
    def _first_existing(frame: pd.DataFrame, names: List[str]) -> Optional[str]:
        for name in names:
            if name in frame.columns:
                return name
        return None


def load_a_share_universe(top_n: Optional[int] = None, min_turnover: float = 0.0, allow_fallback: bool = True) -> pd.DataFrame:
    """Convenience helper: load full A-share spot and optionally filter."""
    try:
        provider = AShareUniverseProvider()
        frame = provider.list_spot()
    except UniverseProviderError:
        if not allow_fallback:
            raise
        frame = _synthetic_a_share_universe()
    if min_turnover > 0 and "turnover" in frame.columns:
        frame = frame[frame["turnover"] >= min_turnover]
    if top_n:
        frame = frame.head(top_n)
    return frame


def synthetic_a_share_universe() -> pd.DataFrame:
    """Offline fallback universe for testing when akshare is unreachable."""
    sectors = ["半导体", "新能源", "白酒", "银行", "医药"]
    rows: List[Dict[str, object]] = []
    for idx in range(30):
        sector = sectors[idx % len(sectors)]
        rows.append(
            {
                "symbol": f"{idx + 1:06d}",
                "name": f"合成股票{idx + 1}",
                "close": 10.0 + idx * 0.5,
                "change_pct": (idx % 7 - 3) * 1.5,
                "volume": 100000 + idx * 1000,
                "turnover": 1000000.0 + idx * 50000,
                "sector": sector,
            }
        )
    return pd.DataFrame(rows)


def _synthetic_a_share_universe() -> pd.DataFrame:
    return synthetic_a_share_universe()
