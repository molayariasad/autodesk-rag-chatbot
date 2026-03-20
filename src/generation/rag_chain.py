"""
RAG Chain: Orchestrates retrieval → context building → generation.

This is the main inference pipeline that ties all components together.
Supports two modes as required by the assignment:
1. corpus: Answer from Autodesk documents only.
2. blended: Merge corpus results with web search.
"""

from dataclasses import dataclass, field
from loguru import logger

from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.web_search import web_search
from src.generation.llm import LLMClient
from src.generation.prompts import build_prompt


@dataclass
class RAGResponse:
    """Structured response from the RAG chain."""
    answer: str
    sources: list[dict] = field(default_factory=list)
    mode: str = "corpus"
    query: str = ""
    retrieval_scores: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        sources = []
        for s in self.sources:
            metadata = s.get("metadata", {})
            source_file = metadata.get("source_file", "")
            raw_url = metadata.get("url", "")

            # Build the correct URL:
            # - Web search results already have a full http:// URL
            # - Corpus docs get a /files/ API URL (served by StaticFiles in main.py)
            if raw_url and raw_url.startswith("http"):
                url = raw_url
            elif source_file and not source_file.startswith("web"):
                url = f"/files/{source_file}"
            else:
                url = raw_url

            sources.append({
                "title": metadata.get("title", "Unknown"),
                "file": source_file,
                "source_path": metadata.get("source_path", ""),
                "type": s.get("source", ""),
                "score": round(
                    s.get("rerank_score", s.get("rrf_score", s.get("score", 0))), 4
                ),
                "url": url,
            })

        return {
            "answer": self.answer,
            "sources": sources,
            "mode": self.mode,
            "query": self.query,
        }


class RAGChain:
    """
    Main RAG pipeline.
    
    Orchestrates: Query → Retrieval → Context Assembly → LLM Generation.
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        llm_client: LLMClient | None = None,
    ):
        self.retriever = retriever
        self.llm = llm_client or LLMClient()

    def query(
        self,
        question: str,
        mode: str = "corpus",
        chat_history: list[dict] | None = None,
        top_k: int = 5,
        category: str | None = None,   # NEW — passed from eval loop
    ) -> RAGResponse:
        """
        Full RAG pipeline execution.
        
        Args:
            question: User's question.
            mode: "corpus" (docs only) or "blended" (docs + web).
            chat_history: Previous conversation turns for context.
            top_k: Number of context chunks to use.
        
        Returns:
            RAGResponse with answer, sources, and metadata.
        """
        logger.info(f"RAG query (mode={mode}): {question[:80]}...")

        # --- Stage 1: Retrieval ---
        corpus_results = self.retriever.retrieve(query=question, top_k=top_k)
        all_results = list(corpus_results)

        if mode == "blended":
            web_results = web_search(question, max_results=3)
            all_results.extend(web_results)
            logger.debug(
                f"Blended: {len(corpus_results)} corpus + {len(web_results)} web results"
            )

        if not all_results:
            return RAGResponse(
                answer=(
                    "I couldn't find relevant information in the available documentation. "
                    "Please try rephrasing your question or check Autodesk's official "
                    "website at autodesk.com."
                ),
                sources=[],
                mode=mode,
                query=question,
            )

        # --- Stage 2: Build prompt ---
        system_prompt, user_message = build_prompt(
            question=question,
            retrieved_chunks=all_results,
            chat_history=chat_history,
            mode=mode,
            category=category,   # NEW
        )

        # --- Stage 3: Generate ---
        answer = self.llm.generate(system_prompt, user_message)

        # --- Build response ---
        response = RAGResponse(
            answer=answer,
            sources=all_results,
            mode=mode,
            query=question,
        )

        logger.info(f"RAG response generated: {len(answer)} chars, {len(all_results)} sources")
        return response

    def is_healthy(self) -> dict:
        """Check health of all components."""
        return {
            "llm_available": self.llm.is_available(),
            "vector_store_count": self.retriever.vector_store.count(),
        }
