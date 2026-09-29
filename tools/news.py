from __future__ import annotations

from typing import Dict, List

import pandas as pd


class SentimentFeed:
    """Offline event/sentiment feed derived from market micro-events.

    This keeps the classroom project runnable without API keys while preserving
    the same interface a real news provider would expose.
    """

    def snapshot(self, frame: pd.DataFrame, day: int, dataset: str) -> Dict[str, object]:
        row = frame.iloc[max(0, min(day, len(frame) - 1))]
        ret = float(row.get("return_1d", 0.0))
        volume_z = float(row.get("volume_z", 0.0))
        events: List[str] = []
        score = 0.0
        if ret > 0.018:
            events.append("价格强势上涨，市场关注度升高")
            score += 0.35
        if ret < -0.018:
            events.append("价格快速回落，短线风险偏好下降")
            score -= 0.35
        if volume_z > 1.2:
            events.append("成交量异常放大，可能存在事件驱动")
            score += 0.12 if ret >= 0 else -0.12
        if dataset == "event" and 68 <= int(row["day"]) <= 76:
            events.append("公司发布新产品试点消息，舆情分歧上升")
            score += 0.22
        if not events:
            events.append("暂无重大事件，舆情保持中性")
        return {"score": max(-1.0, min(1.0, score)), "events": events}


class MultiSentimentFeed:
    """Market + per-stock sentiment for the multi-symbol top-down graph.

    A real news source (akshare ``stock_news_em``, 财联社, 雪球) can plug in
    behind this interface later. Offline, sentiment is derived from market
    micro-events on the candidate cross-section: breadth of advancers, average
    one-day return, and per-stock return/volume-z signals. This gives the LLM a
    directional mood the old multi path lacked (it hard-coded 0.0), and
    degrades gracefully to neutral when the inputs are empty.
    """

    def snapshot(
        self,
        candidates: pd.DataFrame | None,
        technical_context: Dict[str, Dict[str, float]] | None,
    ) -> Dict[str, object]:
        if candidates is None or candidates.empty:
            return {"market_score": 0.0, "per_stock": {}, "top_events": ["候选股为空，舆情中性"]}
        tech = technical_context or {}

        changes = pd.to_numeric(candidates["change_pct"], errors="coerce").fillna(0.0)
        breadth_up = float((changes > 0).mean())
        avg_return = float(changes.mean())
        vol_z_series = pd.to_numeric(candidates.get("volume_z"), errors="coerce").fillna(0.0)
        avg_vol_z = float(vol_z_series.mean()) if not vol_z_series.empty else 0.0

        market_score = 0.0
        events: List[str] = []
        if breadth_up > 0.65:
            market_score += 0.3
            events.append(f"市场广度偏强，{breadth_up:.0%} 候选上涨")
        elif breadth_up < 0.35:
            market_score -= 0.3
            events.append(f"市场广度偏弱，仅 {breadth_up:.0%} 候选上涨")
        if avg_return > 0.015:
            market_score += 0.18
            events.append(f"候选平均涨幅 {avg_return:+.2%}")
        elif avg_return < -0.015:
            market_score -= 0.18
            events.append(f"候选平均跌幅 {avg_return:+.2%}")
        if abs(avg_vol_z) > 1.0:
            tag = "放量" if avg_vol_z > 0 else "缩量"
            market_score += 0.1 if avg_vol_z > 0 else -0.1
            events.append(f"整体{tag}，volume_z 均值 {avg_vol_z:+.2f}")

        per_stock: Dict[str, float] = {}
        for _, row in candidates.iterrows():
            symbol = str(row["symbol"])
            ret = float(tech.get(symbol, {}).get("return_1d", row.get("change_pct", 0.0) or 0.0))
            v_z = float(tech.get(symbol, {}).get("volume_z", 0.0))
            s = 0.0
            if ret > 0.018:
                s += 0.35
            elif ret < -0.018:
                s -= 0.35
            if v_z > 1.2:
                s += 0.12 if ret >= 0 else -0.12
            per_stock[symbol] = max(-1.0, min(1.0, s))

        if not events:
            events.append("暂无明显事件，舆情保持中性")
        return {
            "market_score": max(-1.0, min(1.0, market_score)),
            "per_stock": per_stock,
            "top_events": events[:5],
            "breadth_up": breadth_up,
            "avg_return": avg_return,
        }
