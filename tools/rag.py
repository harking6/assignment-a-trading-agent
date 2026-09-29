from __future__ import annotations

import re
from pathlib import Path
from typing import Any, List

import pandas as pd
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda, RunnableParallel
from pydantic import Field

from core.schemas import AnalystReport, BacktestConfig
from core.assignment import assignment_todo


# Built-in industry character notes. These are always available even when the
# data/knowledge/ book-excerpt directory is empty, so the multi-symbol LLM has
# a minimal structural prior for each Shenwan first-level sector.
_INDUSTRY_NOTES: dict[str, str] = {
    "电子": "周期成长，高弹性，趋势行情领涨；RS 领先时主升方向，回撤也深。",
    "电力设备": "周期成长，光伏/锂电波动大；政策与景气驱动，量能配合时爆发。",
    "银行": "防御红利，低波动低估值；下跌市抗跌，普涨时滞涨，牛市初期领先。",
    "公用事业": "防御，现金流稳定；避险配置，普涨时跑输。",
    "食品饮料": "稳健消费，白酒弹性较高；防御属性强，牛市中后段补涨。",
    "医药生物": "防御成长，政策敏感；震荡市稳健，创新药弹性大。",
    "计算机": "高弹性成长，流动性敏感；主题驱动，量能配合时爆发。",
    "通信": "成长，周期与主题并存；5G/算力主题驱动。",
    "传媒": "高弹性成长，主题驱动；AI/内容主题活跃。",
    "机械设备": "中游制造，周期性中等；跟随制造业景气。",
    "有色金属": "强周期，资源品；全球定价，通胀周期领涨。",
    "基础化工": "周期，跟随油价与景气；波动中等。",
    "汽车": "可选消费，周期成长；新能源转型弹性大。",
    "房地产": "周期，政策驱动；高波动，右侧交易为主。",
    "非银金融": "牛市旗手，券商弹性最高；牛市初期领涨。",
    "商贸零售": "可选消费，稳健偏弱；跟随消费景气。",
    "农林牧渔": "周期，猪周期驱动；独立行情。",
    "国防军工": "成长，主题与装备周期；事件驱动。",
    "钢铁": "强周期，跟随地产与基建；低估值。",
    "煤炭": "红利周期，高股息；煤价驱动，近年抗跌。",
    "石油石化": "强周期，油价驱动；中字头高股息。",
    "建筑装饰": "周期，跟随基建；低估值低波动。",
    "建筑材料": "周期，跟随地产基建；水泥玻璃季节性。",
    "纺织服饰": "可选消费，外需敏感；低弹性。",
    "轻工制造": "可选消费，稳健偏低；家居跟随地产。",
    "家用电器": "可选消费，白电稳健高股息；出海弹性。",
    "社会服务": "可选消费，疫后修复弹性大；事件驱动。",
    "综合": "无明显行业属性，按个股特征处理。",
    "未分类": "无明确行业标签，按个股技术面处理。",
    "未知": "无明确行业标签，按个股技术面处理。",
}


class LexicalTradingRetriever(BaseRetriever):
    """Tiny local retriever implementing LangChain's retriever interface."""

    documents: list[Document] = Field(default_factory=list)
    k: int = 5

    def _get_relevant_documents(self, query: str, *, run_manager: Any = None) -> list[Document]:
        """LC-04A: custom LangChain retriever backed by simple lexical scoring."""

        return retrieve_lexically(query, self.documents, self.k)


class TradingRAG:
    """Lightweight LangChain Document RAG for local, offline trading context.

    This intentionally avoids a mandatory vector database so the classroom demo
    stays runnable. Chroma/FAISS can replace the lexical retriever later without
    changing the graph nodes that consume the returned context.
    """

    def build_context(
        self,
        *,
        symbol: str,
        day: int,
        frame: pd.DataFrame,
        reports: list[AnalystReport],
        memory: list[dict[str, Any]],
        config: BacktestConfig,
        k: int = 5,
    ) -> dict[str, Any]:
        """LC-04B: assemble the local RAG pipeline with LCEL runnables.

        Uses ``RunnableParallel`` with two ``RunnableLambda`` branches to prepare
        the query and split the raw documents. A ``LexicalTradingRetriever`` is
        then invoked and its results are joined into a single context string
        together with source metadata.
        """

        documents = self._documents(symbol, day, frame, reports, memory, config)

        def _query_branch(inputs: dict[str, Any]) -> str:
            return self._query(inputs["symbol"], inputs["day"], inputs["config"])

        def _split_branch(inputs: dict[str, Any]) -> list[Document]:
            return self._split_documents(inputs["documents"])

        pipeline = RunnableParallel(
            query=RunnableLambda(_query_branch),
            documents=RunnableLambda(_split_branch),
        )
        prepared = pipeline.invoke({"symbol": symbol, "day": day, "config": config, "documents": documents})
        retriever = LexicalTradingRetriever(documents=prepared["documents"], k=k)
        relevant = retriever.invoke(prepared["query"])

        context_parts: list[str] = []
        sources: list[dict[str, Any]] = []
        for doc in relevant:
            context_parts.append(doc.page_content)
            source_meta: dict[str, Any] = dict(doc.metadata)
            source_meta.setdefault("source", "unknown")
            source_meta.setdefault("symbol", symbol)
            sources.append(source_meta)

        return {
            "query": prepared["query"],
            "context": "\n\n".join(context_parts),
            "sources": sources,
            "document_count": len(documents),
            "retriever": retriever.__class__.__name__,
            "runnable": "RunnableParallel(query|documents) -> LexicalTradingRetriever.invoke",
        }

    def _documents(
        self,
        symbol: str,
        day: int,
        frame: pd.DataFrame,
        reports: list[AnalystReport],
        memory: list[dict[str, Any]],
        config: BacktestConfig,
    ) -> list[Document]:
        docs: list[Document] = []
        row = frame.iloc[max(0, min(day, len(frame) - 1))]
        market_text = (
            f"Market snapshot for {symbol} day={day + 1}: "
            f"close={float(row.get('close', 0.0)):.4f}, "
            f"return_1d={float(row.get('return_1d', 0.0)):.4%}, "
            f"sma_5={float(row.get('sma_5', 0.0)):.4f}, "
            f"sma_20={float(row.get('sma_20', 0.0)):.4f}, "
            f"macd_hist={float(row.get('macd_hist', 0.0)):.4f}, "
            f"rsi_14={float(row.get('rsi_14', 0.0)):.2f}, "
            f"volatility_20={float(row.get('volatility_20', 0.0)):.4%}, "
            f"risk_budget={config.risk_budget:.2%}, strategy={config.strategy}."
        )
        docs.append(Document(page_content=market_text, metadata={"source": "market_snapshot", "symbol": symbol, "day": day + 1}))

        for report in reports:
            content = (
                f"Analyst report {report.agent}: stance={report.stance.value}, "
                f"score={report.score:.3f}, confidence={report.confidence:.3f}. "
                f"Summary: {report.summary}. Evidence: {'; '.join(report.evidence)}"
            )
            docs.append(
                Document(
                    page_content=content,
                    metadata={"source": "analyst_report", "agent": report.agent, "stance": report.stance.value, "symbol": symbol},
                )
            )

        for item in memory[-max(0, config.memory_window) :]:
            content = (
                f"Decision memory for {item.get('symbol', symbol)} day={item.get('day')}: "
                f"side={item.get('side')}, shares={item.get('shares')}, price={item.get('price')}, "
                f"equity={item.get('equity')}, realized_pnl={item.get('realized_pnl')}, "
                f"debate_verdict={item.get('debate_verdict')}, risk_score={item.get('risk_score')}. "
                f"Reason: {item.get('reason')}"
            )
            docs.append(Document(page_content=content, metadata={"source": "decision_memory", "symbol": item.get("symbol", symbol), "day": item.get("day")}))
        return docs

    def _split_documents(self, documents: list[Document]) -> list[Document]:
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore

            splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)
            return splitter.split_documents(documents)
        except Exception:
            return documents

    def _query(self, symbol: str, day: int, config: BacktestConfig) -> str:
        return (
            f"{symbol} day {day + 1} trading decision risk memory analyst report "
            f"strategy {config.strategy} provider {config.provider} buy sell hold"
        )


def retrieve_lexically(query: str, documents: list[Document], k: int) -> list[Document]:
    query_terms = set(_tokens(query))
    scored: list[tuple[float, int, Document]] = []
    for idx, doc in enumerate(documents):
        text_terms = set(_tokens(doc.page_content))
        overlap = len(query_terms & text_terms)
        source_bonus = 1.0 if doc.metadata.get("source") in {"market_snapshot", "decision_memory"} else 0.0
        stance_bonus = 0.5 if any(term in text_terms for term in ("bullish", "bearish", "risk", "sell", "buy")) else 0.0
        scored.append((overlap + source_bonus + stance_bonus, idx, doc))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [doc for score, _idx, doc in scored[: max(1, k)] if score > 0] or documents[: max(1, k)]


def _tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_.]+|[\u4e00-\u9fff]{2,}", text.lower())


class MultiKnowledgeBase:
    """Knowledge recall for the multi-symbol top-down graph.

    Combines three document sources, all retrieved lexically (a vector store
    can drop in behind the same interface later):

    1. Book excerpts / notes from ``data/knowledge/*.md`` \u2014 trading books,
       trend following, sector rotation, risk discipline, market sentiment.
       Drop a new ``.md`` file in that directory to extend the knowledge base
       without touching code.
    2. Built-in Shenwan first-level industry character notes, so the LLM has
       a structural prior even when no book excerpt matches.
    3. (Optionally) recalled memory cases \u2014 ``MultiKnowledgeBase.recall`` does
       not read memory itself; the graph passes the experience lesson in so
       the knowledge context and the experience context travel together.

    Failures degrade to an empty context \u2014 the LLM keeps working on rules.
    """

    def __init__(self, knowledge_dir: Path | None = None) -> None:
        root = Path(__file__).resolve().parents[1]
        self.knowledge_dir = knowledge_dir or (root / "data" / "knowledge")
        self.last_error: str | None = None

    def _book_documents(self) -> list[Document]:
        docs: list[Document] = []
        try:
            if not self.knowledge_dir.exists():
                return docs
            for path in sorted(self.knowledge_dir.glob("*.md")):
                if path.name == "README.md":
                    continue
                text = path.read_text(encoding="utf-8")
                # Split on markdown headers so a single book file yields
                # multiple retrievable chunks instead of one huge blob.
                chunks = re.split(r"\n(?=#{1,3}\s)", text)
                for chunk in chunks:
                    chunk = chunk.strip()
                    if not chunk:
                        continue
                    docs.append(Document(page_content=chunk, metadata={"source": "book", "file": path.name}))
        except Exception as exc:
            self.last_error = str(exc)
        return docs

    def _industry_documents(self, sectors: list[str]) -> list[Document]:
        docs: list[Document] = []
        for sector in sectors:
            note = _INDUSTRY_NOTES.get(sector)
            if note:
                docs.append(Document(page_content=f"{sector}\u884c\u4e1a\u7279\u5f81\uff1a{note}", metadata={"source": "industry", "sector": sector}))
        return docs

    def recall(
        self,
        sectors: list[str],
        market_view: str = "neutral",
        k: int = 4,
    ) -> dict[str, Any]:
        """Retrieve knowledge chunks relevant to the selected sectors + regime.

        Returns ``{context, sources}`` mirroring the single-symbol RAG
        contract so the caller can render it into a prompt uniformly.
        """
        # Industry notes for the SELECTED sectors are always included — they
        # are the structural prior the LLM should see for exactly those
        # sectors, not a lexical maybe-match. Book excerpts fill the remainder
        # of the budget by lexical relevance to the query.
        industry_docs = self._industry_documents(sectors or [])
        book_docs = self._book_documents()
        query = self._query(sectors, market_view)
        book_budget = max(1, k - len(industry_docs))
        relevant_book = retrieve_lexically(query, book_docs, book_budget) if book_docs else []
        relevant = industry_docs + relevant_book
        context_parts = [doc.page_content for doc in relevant]
        sources = [
            {"source": doc.metadata.get("source", "unknown"), "file": doc.metadata.get("file", ""), "sector": doc.metadata.get("sector", "")}
            for doc in relevant
        ]
        return {
            "context": "\n\n".join(context_parts),
            "sources": sources,
            "document_count": len(industry_docs) + len(book_docs),
        }

    @staticmethod
    def _query(sectors: list[str], market_view: str) -> str:
        return (
            f"{market_view} \u5e02\u573a\u89c2 "
            f"\u677f\u5757 {' '.join(sectors or [])} "
            "\u8d8b\u52bf \u5747\u7ebf \u7a81\u7834 \u9f99\u5934 \u76f8\u5bf9\u5f3a\u5ea6 \u4ed3\u4f4d \u6b62\u635f \u73b0\u91d1 \u5e7f\u5ea6 \u91cf\u80fd \u60c5\u7eea"
        )

