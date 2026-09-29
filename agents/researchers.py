from __future__ import annotations

from typing import Any, Iterable

from core.schemas import AnalystReport, ResearchDebate, Stance
from agents.llm_client import LLMClient, clamp_float


class BullResearcher:
    def argue(self, reports: Iterable[AnalystReport]) -> tuple[str, float]:
        positives = []
        score = 0.0
        for report in reports:
            contribution = max(0.0, report.score) * report.confidence
            score += contribution
            if report.stance == Stance.bullish:
                positives.extend(report.evidence[:2])
        thesis = "；".join(positives[:4]) or "多头证据不足，仅建议观察。"
        return thesis, min(1.0, score / 2.2)


class BearResearcher:
    def argue(self, reports: Iterable[AnalystReport]) -> tuple[str, float]:
        negatives = []
        score = 0.0
        for report in reports:
            contribution = max(0.0, -report.score) * report.confidence
            score += contribution
            if report.stance == Stance.bearish:
                negatives.extend(report.evidence[:2])
        thesis = "；".join(negatives[:4]) or "空头证据不足，主要风险来自仓位和波动。"
        return thesis, min(1.0, score / 2.2)


class ResearchManager:
    def __init__(self) -> None:
        self.bull = BullResearcher()
        self.bear = BearResearcher()

    def debate(
        self,
        reports: list[AnalystReport],
        rounds: int = 2,
        memory: list[dict[str, Any]] | None = None,
        memory_messages: list[dict[str, str]] | None = None,
        rag_context: dict[str, Any] | None = None,
        tool_context: dict[str, Any] | None = None,
        llm: LLMClient | None = None,
    ) -> ResearchDebate:
        rounds = max(1, min(4, int(rounds or 1)))
        bull_thesis, bull_score = self.bull.argue(reports)
        bear_thesis, bear_score = self.bear.argue(reports)
        transcript: list[dict[str, Any]] = []
        for idx in range(rounds):
            bull_round = f"第{idx + 1}轮多头：{bull_thesis}；强度={bull_score:.2f}"
            bear_round = f"第{idx + 1}轮空头：{bear_thesis}；强度={bear_score:.2f}"
            transcript.append({"round": idx + 1, "bull": bull_round, "bear": bear_round})
            if idx < rounds - 1:
                bull_score = min(1.0, bull_score + max(0.0, bear_score - bull_score) * 0.18)
                bear_score = min(1.0, bear_score + max(0.0, bull_score - bear_score) * 0.16)

        if llm and llm.enabled:
            llm_result = llm.complete_json(
                "research_manager",
                self._llm_prompt(reports, transcript, memory or [], memory_messages or [], rag_context or {}, tool_context or {}),
                deep=True,
            )
            if llm_result:
                bull_thesis = str(llm_result.get("bull_thesis") or bull_thesis)
                bear_thesis = str(llm_result.get("bear_thesis") or bear_thesis)
                bull_score = clamp_float(llm_result.get("bull_score"), 0.0, 1.0, bull_score)
                bear_score = clamp_float(llm_result.get("bear_score"), 0.0, 1.0, bear_score)
                transcript.append({"round": "llm", "manager": str(llm_result.get("manager_notes") or "LLM research synthesis")})

        spread = bull_score - bear_score
        if spread > 0.08:
            verdict = Stance.bullish
        elif spread < -0.08:
            verdict = Stance.bearish
        else:
            verdict = Stance.neutral
        conviction = min(0.95, 0.42 + abs(spread) * 0.9)
        notes = f"研究经理裁决：bull={bull_score:.2f}, bear={bear_score:.2f}, verdict={verdict.value}, rounds={rounds}"
        return ResearchDebate(
            bull_thesis=bull_thesis,
            bear_thesis=bear_thesis,
            bull_score=bull_score,
            bear_score=bear_score,
            verdict=verdict,
            conviction=conviction,
            manager_notes=notes,
            rounds=transcript,
        )

    def _llm_prompt(
        self,
        reports: list[AnalystReport],
        transcript: list[dict[str, Any]],
        memory: list[dict[str, Any]],
        memory_messages: list[dict[str, str]],
        rag_context: dict[str, Any],
        tool_context: dict[str, Any],
    ) -> str:
        report_payload = [report.model_dump(mode="json") for report in reports]
        return (
            "Synthesize the analyst reports into a bull/bear debate judgement.\n"
            "Return JSON with bull_thesis, bear_thesis, bull_score, bear_score, manager_notes.\n"
            f"Analyst reports: {report_payload}\n"
            f"Debate transcript so far: {transcript}\n"
            f"LangChain RAG context: {rag_context.get('context', '')}\n"
            f"RAG sources: {rag_context.get('sources', [])}\n"
            f"LangChain tool outputs: {tool_context}\n"
            f"Recent memory: {memory}\n"
            f"LangChain chat history messages: {memory_messages}\n"
        )
