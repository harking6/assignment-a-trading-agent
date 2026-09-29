from __future__ import annotations

from core.schemas import OrderSide, PortfolioDecision, RiskAssessment, TradeIntent


class PortfolioManager:
    def decide(self, intent: TradeIntent, risk: RiskAssessment) -> PortfolioDecision:
        blocked = risk.side == OrderSide.hold and intent.side != OrderSide.hold
        signal = intent.signal
        if blocked:
            signal = "risk_blocked"
        reason = intent.rationale
        if risk.reasons:
            reason = f"{reason} | 风控：{'；'.join(risk.reasons)}"
        confidence = max(0.05, intent.confidence * (1.0 - risk.risk_score * 0.25))
        return PortfolioDecision(
            side=risk.side,
            shares=risk.shares,
            signal=signal,
            confidence=confidence,
            reason=reason,
            blocked=blocked,
        )
