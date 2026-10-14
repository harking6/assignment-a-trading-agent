"""Reproducible dynamic-universe capture and replay services."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Protocol

import pandas as pd
from pydantic import BaseModel, Field, model_validator

from core.contracts import UniverseSource
from core.schemas import BacktestConfig
from data.universe import AShareUniverseProvider, UniverseProviderError


REQUIRED_UNIVERSE_COLUMNS = {
    "symbol",
    "name",
    "close",
    "change_pct",
    "volume",
    "turnover",
    "sector",
}


class UniverseSnapshot(BaseModel):
    """Serializable point-in-time input for a dynamic-universe backtest.

    ``as_of`` is only populated when the upstream source reports an explicit
    market date.  ``captured_at`` records collection time and must not be
    interpreted as historical constituent membership.
    """

    schema_version: int = 1
    snapshot_id: str
    captured_at: str
    source: str
    as_of: str | None = None
    columns: List[str]
    records: List[Dict[str, Any]] = Field(default_factory=list)
    parameters: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        *,
        source: str,
        as_of: str | None = None,
        captured_at: str | None = None,
        parameters: Dict[str, Any] | None = None,
    ) -> "UniverseSnapshot":
        missing = REQUIRED_UNIVERSE_COLUMNS.difference(frame.columns)
        if missing:
            raise ValueError(f"universe snapshot missing columns: {sorted(missing)}")
        normalized_records = json.loads(
            frame.to_json(orient="records", force_ascii=False, date_format="iso")
        )
        columns = [str(column) for column in frame.columns]
        payload = {
            "schema_version": 1,
            "source": source,
            "as_of": as_of,
            "columns": columns,
            "records": normalized_records,
            "parameters": parameters or {},
        }
        return cls(
            snapshot_id=cls._content_id(payload),
            captured_at=captured_at or datetime.now(timezone.utc).isoformat(),
            source=source,
            as_of=as_of,
            columns=columns,
            records=normalized_records,
            parameters=parameters or {},
        )

    @model_validator(mode="after")
    def verify_content_id(self) -> "UniverseSnapshot":
        expected = self._content_id(self._content_payload())
        if self.snapshot_id != expected:
            raise ValueError(
                f"universe snapshot checksum mismatch: expected {expected}, got {self.snapshot_id}"
            )
        missing = REQUIRED_UNIVERSE_COLUMNS.difference(self.columns)
        if missing:
            raise ValueError(f"universe snapshot missing columns: {sorted(missing)}")
        return self

    def _content_payload(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source": self.source,
            "as_of": self.as_of,
            "columns": self.columns,
            "records": self.records,
            "parameters": self.parameters,
        }

    @staticmethod
    def _content_id(payload: Dict[str, Any]) -> str:
        canonical = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return f"sha256:{hashlib.sha256(canonical).hexdigest()}"

    def to_frame(self) -> pd.DataFrame:
        """Return an isolated DataFrame copy suitable for one engine reset."""
        return pd.DataFrame.from_records(self.records, columns=self.columns).copy(deep=True)

    def summary(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "snapshot_id": self.snapshot_id,
            "captured_at": self.captured_at,
            "source": self.source,
            "as_of": self.as_of,
            "symbol_count": len(self.records),
            "symbols": [str(record.get("symbol", "")) for record in self.records],
            "parameters": dict(self.parameters),
        }

    def save(self, path: str | Path) -> Path:
        """Atomically save the complete replay payload as UTF-8 JSON."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(f"{target.suffix}.tmp")
        temporary.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        temporary.replace(target)
        return target

    @classmethod
    def load(cls, path: str | Path) -> "UniverseSnapshot":
        """Load a snapshot and reject content changed without a new checksum."""
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))


class UniverseSnapshotSource(Protocol):
    def capture(self, config: BacktestConfig) -> UniverseSnapshot:
        """Capture one normalized and auditable universe snapshot."""
        ...


class DynamicUniverseService:
    """Capture the existing AkShare/Tushare dynamic universe as a snapshot."""

    def __init__(
        self,
        universe_factory: Callable[[], UniverseSource] = AShareUniverseProvider,
    ) -> None:
        self.universe_factory = universe_factory

    def capture(self, config: BacktestConfig) -> UniverseSnapshot:
        if not config.dynamic_universe:
            raise ValueError("dynamic_universe must be enabled to capture a snapshot")
        if config.provider != "akshare":
            raise ValueError("A股全市场动态股票池当前仅支持 provider='akshare'")
        try:
            frame, source = self._capture_akshare(config)
            as_of = None
        except Exception as akshare_exc:
            try:
                frame, source, as_of = self._capture_tushare(config)
            except Exception as tushare_exc:
                raise UniverseProviderError(
                    f"动态A股股票池构建失败: akshare={akshare_exc}; tushare={tushare_exc}"
                ) from tushare_exc
        return UniverseSnapshot.from_frame(
            frame,
            source=source,
            as_of=as_of,
            parameters=self._snapshot_parameters(config),
        )

    def _capture_akshare(self, config: BacktestConfig) -> tuple[pd.DataFrame, str]:
        provider = self.universe_factory()
        try:
            sectors = provider.list_sectors().sort_values("change_pct", ascending=False)
            sector_count = max(config.top_n_sectors, config.dynamic_sector_pool)
            frames = []
            for sector in sectors.head(sector_count)["sector"].astype(str):
                stocks = provider.list_sector_stocks(sector)
                if not stocks.empty:
                    frames.append(stocks)
            if not frames:
                raise UniverseProviderError("industry boards returned no constituents")
            frame = pd.concat(frames, ignore_index=True)
            source = "akshare:industry_boards"
        except Exception:
            frame = provider.list_spot()
            if frame.empty:
                raise UniverseProviderError("full-market spot table is empty")
            frame = frame.copy()
            if frame["sector"].isin(["未分类", "未知"]).all():
                frame["sector"] = frame.apply(
                    lambda row: self._fallback_dynamic_sector(
                        str(row["symbol"]), str(row.get("name", ""))
                    ),
                    axis=1,
                )
            source = "akshare:spot"
        frame = frame.drop_duplicates("symbol")
        frame["turnover"] = pd.to_numeric(frame["turnover"], errors="coerce").fillna(0.0)
        frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
        frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce").fillna(0)
        frame = frame[
            (frame["close"] > 0)
            & (frame["volume"] > 0)
            & (frame["turnover"] >= config.dynamic_min_turnover)
            & ~frame["name"].astype(str).str.contains(r"ST|退", regex=True)
        ]
        return (
            frame.sort_values(["turnover", "change_pct"], ascending=[False, False])
            .head(max(5, config.dynamic_universe_limit))
            .reset_index(drop=True),
            source,
        )

    def _capture_tushare(
        self,
        config: BacktestConfig,
    ) -> tuple[pd.DataFrame, str, str]:
        import tushare as ts  # type: ignore

        token = os.getenv("TUSHARE_TOKEN")
        if not token:
            raise UniverseProviderError("TUSHARE_TOKEN is not configured")
        pro = ts.pro_api(token)
        end_date = date.today().strftime("%Y%m%d")
        calendar = pro.trade_cal(
            exchange="", end_date=end_date, is_open="1", fields="cal_date", limit=1
        )
        if calendar is None or calendar.empty:
            raise UniverseProviderError("Tushare trading calendar returned no open date")
        trade_date = str(calendar.iloc[0]["cal_date"])
        basic = pro.stock_basic(
            exchange="",
            list_status="L",
            fields="ts_code,symbol,name,industry",
        )
        daily = pro.daily(
            trade_date=trade_date,
            fields="ts_code,close,pct_chg,vol,amount",
        )
        if basic is None or daily is None or basic.empty or daily.empty:
            raise UniverseProviderError(f"Tushare returned no full-market data for {trade_date}")
        frame = basic.merge(daily, on="ts_code", how="inner")
        normalized = pd.DataFrame(
            {
                "symbol": frame["symbol"].astype(str).str.zfill(6),
                "name": frame["name"].astype(str),
                "close": pd.to_numeric(frame["close"], errors="coerce"),
                "change_pct": pd.to_numeric(frame["pct_chg"], errors="coerce").fillna(0.0),
                "volume": pd.to_numeric(frame["vol"], errors="coerce").fillna(0.0) * 100,
                "turnover": pd.to_numeric(frame["amount"], errors="coerce").fillna(0.0) * 1000,
                "sector": frame["industry"].fillna("未知行业").astype(str),
            }
        )
        normalized = normalized[
            (normalized["close"] > 0)
            & (normalized["volume"] > 0)
            & (normalized["turnover"] >= config.dynamic_min_turnover)
            & ~normalized["name"].str.contains(r"ST|退", regex=True)
        ]
        result = (
            normalized.sort_values(["turnover", "change_pct"], ascending=[False, False])
            .head(max(5, config.dynamic_universe_limit))
            .reset_index(drop=True)
        )
        as_of = datetime.strptime(trade_date, "%Y%m%d").date().isoformat()
        return result, "tushare:daily", as_of

    @staticmethod
    def _snapshot_parameters(config: BacktestConfig) -> Dict[str, Any]:
        return {
            "provider": config.provider,
            "dynamic_universe_limit": config.dynamic_universe_limit,
            "dynamic_sector_pool": config.dynamic_sector_pool,
            "dynamic_min_turnover": config.dynamic_min_turnover,
            "top_n_sectors": config.top_n_sectors,
        }

    @staticmethod
    def _fallback_dynamic_sector(symbol: str, name: str) -> str:
        code = "".join(ch for ch in symbol if ch.isdigit())[:6]
        if "银行" in name:
            return "银行"
        if code.startswith("688"):
            return "科创板"
        if code.startswith("300"):
            return "创业板"
        if code.startswith(("600", "601", "603", "605")):
            return "沪市主板"
        if code.startswith("002"):
            return "深市中小盘"
        if code.startswith("000"):
            return "深市主板"
        if code.startswith(("4", "8", "92")):
            return "北交所"
        return "其他A股"


__all__ = [
    "DynamicUniverseService",
    "UniverseSnapshot",
    "UniverseSnapshotSource",
]
