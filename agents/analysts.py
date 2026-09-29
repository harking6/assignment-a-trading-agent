from __future__ import annotations

from typing import Dict, List

import pandas as pd

from core.schemas import AnalystReport, PortfolioSnapshot, Stance
from tools.fundamentals import FundamentalsFeed
from tools.news import SentimentFeed


class MarketAnalyst:
    name = "market_analyst"

    def run(self, frame: pd.DataFrame, day: int, portfolio: PortfolioSnapshot) -> AnalystReport:
        row = frame.iloc[day]
        score = 0.0
        evidence: List[str] = []
        if row["sma_5"] > row["sma_20"]:
            score += 0.28
            evidence.append("5日均线位于20日均线上方")
        else:
            score -= 0.22
            evidence.append("5日均线弱于20日均线")
        if row["macd_hist"] > 0:
            score += 0.22
            evidence.append("MACD histogram 为正")
        else:
            score -= 0.18
            evidence.append("MACD histogram 为负")
        if row["rsi_14"] > 72:
            score -= 0.22
            evidence.append("RSI 进入过热区")
        elif row["rsi_14"] < 32:
            score += 0.20
            evidence.append("RSI 接近超卖区")
        if row["close"] > row["sma_60"]:
            score += 0.14
            evidence.append("价格位于60日趋势线上方")
        stance = Stance.bullish if score > 0.15 else Stance.bearish if score < -0.15 else Stance.neutral
        return AnalystReport(
            agent=self.name,
            stance=stance,
            score=max(-1.0, min(1.0, score)),
            confidence=min(0.92, 0.48 + abs(score) * 0.55),
            summary=f"技术面 {stance.value}，组合持仓 {portfolio.position} 股。",
            evidence=evidence,
            metrics={
                "sma_5": float(row["sma_5"]),
                "sma_20": float(row["sma_20"]),
                "sma_60": float(row["sma_60"]),
                "rsi_14": float(row["rsi_14"]),
                "macd_hist": float(row["macd_hist"]),
                "volatility_20": float(row["volatility_20"]),
            },
        )


class SentimentAnalyst:
    name = "sentiment_analyst"

    def __init__(self, feed: SentimentFeed | None = None) -> None:
        self.feed = feed or SentimentFeed()

    def run(self, frame: pd.DataFrame, day: int, dataset: str) -> AnalystReport:
        snapshot = self.feed.snapshot(frame, day, dataset)
        score = float(snapshot["score"])
        stance = Stance.bullish if score > 0.12 else Stance.bearish if score < -0.12 else Stance.neutral
        return AnalystReport(
            agent=self.name,
            stance=stance,
            score=score,
            confidence=min(0.85, 0.42 + abs(score) * 0.65),
            summary=f"舆情/事件面 {stance.value}",
            evidence=list(snapshot["events"]),
            metrics={"sentiment_score": score},
        )


class NewsAnalyst:
    name = "news_analyst"

    def __init__(self, feed: SentimentFeed | None = None) -> None:
        self.feed = feed or SentimentFeed()

    def run(self, frame: pd.DataFrame, day: int, dataset: str) -> AnalystReport:
        snapshot = self.feed.snapshot(frame, day, dataset)
        score = float(snapshot["score"]) * 0.75
        row = frame.iloc[max(0, min(day, len(frame) - 1))]
        if float(row.get("volume_z", 0.0)) > 1.8:
            score += 0.08 if float(row.get("return_1d", 0.0)) >= 0 else -0.08
        stance = Stance.bullish if score > 0.10 else Stance.bearish if score < -0.10 else Stance.neutral
        events = list(snapshot["events"])
        return AnalystReport(
            agent=self.name,
            stance=stance,
            score=max(-1.0, min(1.0, score)),
            confidence=min(0.78, 0.40 + abs(score) * 0.7),
            summary=f"新闻/事件面 {stance.value}",
            evidence=events + [f"成交量Z分数 {float(row.get('volume_z', 0.0)):.2f}"],
            metrics={"news_score": score, "volume_z": float(row.get("volume_z", 0.0))},
        )


class FundamentalsAnalyst:
    name = "fundamentals_analyst"

    def __init__(self, feed: FundamentalsFeed | None = None) -> None:
        self.feed = feed or FundamentalsFeed()

    def run(self, dataset: str) -> AnalystReport:
        snapshot = self.feed.snapshot(dataset)
        score = float(snapshot["quality"])
        stance = Stance.bullish if score > 0.10 else Stance.bearish if score < -0.10 else Stance.neutral
        evidence = [
            f"增长 {snapshot['growth']:.2f}",
            f"利润率 {snapshot['margin']:.2f}",
            f"债务压力 {snapshot['debt']:.2f}",
            f"估值分位 {snapshot['valuation']:.2f}",
        ]
        return AnalystReport(
            agent=self.name,
            stance=stance,
            score=score,
            confidence=min(0.82, 0.45 + abs(score) * 0.5),
            summary=f"基本面质量 {stance.value}",
            evidence=evidence,
            metrics=snapshot,
        )
