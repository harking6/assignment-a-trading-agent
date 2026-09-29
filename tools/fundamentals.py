from __future__ import annotations

from typing import Dict


class FundamentalsFeed:
    """Synthetic fundamentals provider with a real provider-compatible shape."""

    BASELINES = {
        "trend": {"growth": 0.16, "margin": 0.22, "debt": 0.28, "valuation": 0.55},
        "range": {"growth": 0.06, "margin": 0.17, "debt": 0.34, "valuation": 0.48},
        "stress": {"growth": -0.02, "margin": 0.12, "debt": 0.52, "valuation": 0.68},
        "event": {"growth": 0.12, "margin": 0.19, "debt": 0.31, "valuation": 0.61},
    }

    def snapshot(self, dataset: str) -> Dict[str, float]:
        item = self.BASELINES.get(dataset, self.BASELINES["trend"])
        quality = item["growth"] * 1.8 + item["margin"] * 1.2 - item["debt"] * 0.8 - item["valuation"] * 0.35
        return {**item, "quality": max(-1.0, min(1.0, quality))}
