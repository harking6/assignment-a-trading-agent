from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def enrich_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    df = frame.copy()
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)
    df["return_1d"] = close.pct_change().fillna(0.0)
    df["sma_5"] = close.rolling(5, min_periods=1).mean()
    df["sma_20"] = close.rolling(20, min_periods=1).mean()
    df["sma_60"] = close.rolling(60, min_periods=1).mean()
    df["ema_12"] = close.ewm(span=12, adjust=False).mean()
    df["ema_26"] = close.ewm(span=26, adjust=False).mean()
    df["macd"] = df["ema_12"] - df["ema_26"]
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    delta = close.diff().fillna(0.0)
    gain = delta.clip(lower=0).rolling(14, min_periods=1).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi_14"] = (100 - (100 / (1 + rs))).fillna(50.0)
    true_range = pd.concat([(high - low), (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    df["atr_14"] = true_range.rolling(14, min_periods=1).mean()
    df["volatility_20"] = df["return_1d"].rolling(20, min_periods=2).std().fillna(0.0)
    df["volume_z"] = ((volume - volume.rolling(20, min_periods=1).mean()) / volume.rolling(20, min_periods=2).std()).replace([np.inf, -np.inf], 0).fillna(0)
    return df


def latest_metrics(frame: pd.DataFrame, day: int) -> Dict[str, float]:
    if frame.empty:
        return {}
    row = frame.iloc[max(0, min(day, len(frame) - 1))]
    keys = ["close", "return_1d", "sma_5", "sma_20", "sma_60", "macd_hist", "rsi_14", "atr_14", "volatility_20", "volume_z"]
    return {key: float(row[key]) for key in keys if key in row}
