from __future__ import annotations

from core.schemas import BacktestConfig, OrderSide, PortfolioSnapshot, RiskAssessment, RiskDebate, TradeIntent
from tools.sizing import round_order_shares


class AggressiveRiskManager:
    def argue(self, intent: TradeIntent, portfolio: PortfolioSnapshot, price: float, config: BacktestConfig) -> tuple[str, int, float]:
        if intent.side == OrderSide.buy:
            boosted = min(config.max_single_order_shares, max(intent.shares, int(intent.shares * 1.15)))
            return "进攻风险经理：趋势/研究共识允许更快建仓，但仍受单笔上限约束", boosted, 0.42
        if intent.side == OrderSide.sell:
            reduced = max(0, int(intent.shares * 0.85))
            return "进攻风险经理：避免过度降仓，保留反弹参与度", reduced, 0.38
        return "进攻风险经理：没有明确交易意图，等待更强信号", 0, 0.30


class NeutralRiskManager:
    def argue(self, intent: TradeIntent, portfolio: PortfolioSnapshot, price: float, config: BacktestConfig) -> tuple[str, int, float]:
        return "中性风险经理：接受 trader 的目标，但要求现金保留和最大仓位校验", intent.shares, 0.50


class ConservativeRiskManager:
    def argue(self, intent: TradeIntent, portfolio: PortfolioSnapshot, price: float, config: BacktestConfig) -> tuple[str, int, float]:
        drawdown_pressure = abs(min(0.0, portfolio.max_drawdown))
        if intent.side == OrderSide.buy:
            scaled = int(intent.shares * max(0.25, 1.0 - drawdown_pressure * 4.0))
            return "保守风险经理：买入前优先控制回撤和现金安全垫", scaled, min(0.85, 0.55 + drawdown_pressure)
        if intent.side == OrderSide.sell:
            boosted = min(portfolio.position, max(intent.shares, int(intent.shares * 1.15)))
            return "保守风险经理：下行信号出现时倾向更快降低风险敞口", boosted, 0.68
        return "保守风险经理：无交易优于弱信号交易", 0, 0.45


class RiskDebatePanel:
    def __init__(self) -> None:
        self.aggressive = AggressiveRiskManager()
        self.neutral = NeutralRiskManager()
        self.conservative = ConservativeRiskManager()

    def debate(self, intent: TradeIntent, portfolio: PortfolioSnapshot, price: float, config: BacktestConfig) -> RiskDebate:
        aggressive_case, aggressive_shares, aggressive_risk = self.aggressive.argue(intent, portfolio, price, config)
        neutral_case, neutral_shares, neutral_risk = self.neutral.argue(intent, portfolio, price, config)
        conservative_case, conservative_shares, conservative_risk = self.conservative.argue(intent, portfolio, price, config)

        if intent.side == OrderSide.hold:
            final_shares = 0
            risk_score = max(aggressive_risk, neutral_risk, conservative_risk)
        elif config.strategy == "aggressive":
            final_shares = aggressive_shares
            risk_score = aggressive_risk
        elif config.strategy == "risk":
            final_shares = conservative_shares
            risk_score = conservative_risk
        else:
            candidates = sorted([aggressive_shares, neutral_shares, conservative_shares])
            final_shares = candidates[1]
            risk_score = (aggressive_risk + neutral_risk + conservative_risk) / 3.0

        final_side = intent.side if final_shares > 0 else OrderSide.hold
        final_shares = round_order_shares(final_shares, config, final_side, portfolio.position)
        final_side = intent.side if final_shares > 0 else OrderSide.hold
        judgement = f"风险辩论裁决：style={config.strategy}, side={final_side.value}, shares={final_shares}"
        return RiskDebate(
            aggressive_case=aggressive_case,
            neutral_case=neutral_case,
            conservative_case=conservative_case,
            final_judgement=judgement,
            final_side=final_side,
            final_shares=final_shares,
            risk_score=max(0.0, min(1.0, risk_score)),
            rounds=[
                {"role": "aggressive", "shares": aggressive_shares, "risk": aggressive_risk, "case": aggressive_case},
                {"role": "neutral", "shares": neutral_shares, "risk": neutral_risk, "case": neutral_case},
                {"role": "conservative", "shares": conservative_shares, "risk": conservative_risk, "case": conservative_case},
            ],
        )


class RiskCommittee:
    """TradingAgents-style risk team with hard guardrails."""

    def assess(self, intent: TradeIntent, portfolio: PortfolioSnapshot, price: float, config: BacktestConfig) -> RiskAssessment:
        reasons: list[str] = []
        side = intent.side
        shares = intent.shares
        equity = portfolio.equity
        max_notional = equity * min(config.max_position_ratio, config.risk_budget + 0.25)
        max_shares = int(max_notional // price) if price else 0
        cash_reserve = equity * config.cash_reserve_ratio
        drawdown_limit = -abs(config.risk_budget)
        risk_score = 0.25

        if portfolio.max_drawdown < drawdown_limit and side == OrderSide.buy:
            reasons.append("组合回撤已超过风险预算，禁止加仓")
            side = OrderSide.hold
            shares = 0
            risk_score = 0.95

        if side == OrderSide.buy:
            allowed = max(0, max_shares - portfolio.position)
            if shares > allowed:
                reasons.append(f"订单超过最大仓位，{shares} 调整为 {allowed}")
                shares = allowed
                risk_score = max(risk_score, 0.62)
            rounded = round_order_shares(shares, config, side, portfolio.position)
            if rounded != shares:
                reasons.append(f"按交易手数从 {shares} 调整为 {rounded}")
                shares = rounded
            cost = shares * price * (1 + config.fee_rate)
            if shares <= 0 or portfolio.cash - cost < cash_reserve:
                reasons.append("现金保留线触发，买入被拒绝")
                side = OrderSide.hold
                shares = 0
                risk_score = 0.88

        if side == OrderSide.sell:
            if portfolio.position <= 0:
                reasons.append("无持仓，卖出被拒绝")
                side = OrderSide.hold
                shares = 0
                risk_score = 0.76
            elif shares > portfolio.position:
                reasons.append(f"卖出数量超过持仓，{shares} 调整为 {portfolio.position}")
                shares = portfolio.position
                risk_score = max(risk_score, 0.5)
            rounded = round_order_shares(shares, config, side, portfolio.position)
            if rounded != shares:
                reasons.append(f"按交易手数从 {shares} 调整为 {rounded}")
                shares = rounded

        if not reasons:
            reasons.append("风险团队通过订单")
        return RiskAssessment(
            approved=side != OrderSide.hold or intent.side == OrderSide.hold,
            side=side,
            shares=shares,
            risk_score=risk_score,
            reasons=reasons,
            limits={
                "max_shares": max_shares,
                "cash_reserve": cash_reserve,
                "drawdown_limit": drawdown_limit,
            },
        )
