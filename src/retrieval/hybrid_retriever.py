"""
Hybrid Retriever: Vector Search + BM25 + Cross-Encoder Reranking.

Architecture:
1. Vector search (semantic): Captures meaning-based matches.
2. BM25 (keyword, uni-gram, bigram): Captures exact term matches (product names, versions).
3. Reciprocal Rank Fusion (RRF): Merges both ranked lists without needing
   score normalization across different scoring functions.
4. Cross-encoder reranking: Final precision pass using a cross-encoder model
   that jointly encodes (query, passage) for much more accurate relevance scoring.

This is a well-established pattern from information retrieval research.
"""

from sentence_transformers import CrossEncoder
from loguru import logger

from src.config import settings
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_retriever import BM25Index


class HybridRetriever:
    """Hybrid retrieval with RRF fusion and cross-encoder reranking."""

    def __init__(
        self,
        vector_store: VectorStore,
        bm25_index: BM25Index,
    ):
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self._reranker: CrossEncoder | None = None

    @property
    def reranker(self) -> CrossEncoder:
        if self._reranker is None:
            logger.info(f"Loading reranker: {settings.reranker_model}")
            self._reranker = CrossEncoder(settings.reranker_model)
        return self._reranker

    def _reciprocal_rank_fusion(
        self,
        vector_hits: list[dict],
        bm25_hits: list[dict],
        k: int = 60,
    ) -> list[dict]:
        """
        Reciprocal Rank Fusion (RRF) to merge two ranked lists.
        
        RRF score = sum( 1 / (k + rank_i) ) across all lists.
        k=60 is the standard constant from the original paper (Cormack et al., 2009).
        """
        # Build a mapping: chunk_id → {hit_data, rrf_score}
        fused = {}

        for rank, hit in enumerate(vector_hits):
            cid = hit["chunk_id"]
            rrf_score = 1.0 / (k + rank + 1)
            if cid not in fused:
                fused[cid] = {**hit, "rrf_score": 0.0, "sources": set()}
            fused[cid]["rrf_score"] += rrf_score * settings.vector_weight
            fused[cid]["sources"].add("vector")

        for rank, hit in enumerate(bm25_hits):
            cid = hit["chunk_id"]
            rrf_score = 1.0 / (k + rank + 1)
            if cid not in fused:
                fused[cid] = {**hit, "rrf_score": 0.0, "sources": set()}
            fused[cid]["rrf_score"] += rrf_score * settings.bm25_weight
            fused[cid]["sources"].add("bm25")

        # Sort by RRF score descending
        results = sorted(fused.values(), key=lambda x: x["rrf_score"], reverse=True)

        # Convert sources set to list for serialization
        for r in results:
            r["sources"] = list(r["sources"])

        return results

    def _rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        """Apply cross-encoder reranking to the top candidates."""
        if not candidates:
            return []

        pairs = [(query, c["text"]) for c in candidates]
        scores = self.reranker.predict(pairs)

        for i, score in enumerate(scores):
            candidates[i]["rerank_score"] = float(score)

        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        use_reranker: bool = True,
        filter_dict: dict | None = None,
    ) -> list[dict]:
        """
        Full hybrid retrieval pipeline.

        Args:
            query: User question.
            top_k: Number of final results to return.
            use_reranker: Whether to apply cross-encoder reranking.
            filter_dict: Optional ChromaDB metadata filter.

        Returns:
            List of dicts with chunk_id, text, metadata, and various scores.
        """
        top_k = top_k or settings.rerank_top_k
        # Retrieve more candidates for RRF fusion
        candidate_k = settings.retrieval_top_k

        # Stage 1: Parallel retrieval
        vector_hits = self.vector_store.query(query, top_k=candidate_k, filter_dict=filter_dict)
        bm25_hits = self.bm25_index.query(query, top_k=candidate_k)

        logger.debug(
            f"Retrieved: {len(vector_hits)} vector + {len(bm25_hits)} BM25 hits"
        )

        # Stage 2: RRF fusion
        fused = self._reciprocal_rank_fusion(vector_hits, bm25_hits)

        if not fused:
            logger.warning("No results from either retriever")
            return []

        # Stage 3: Rerank (optional)
        if use_reranker and len(fused) > top_k:
            # Rerank the top candidates (limit to avoid latency)
            rerank_candidates = fused[: max(top_k * 3, 15)]
            results = self._rerank(query, rerank_candidates, top_k)
            logger.debug(f"Reranked to {len(results)} results")
        else:
            results = fused[:top_k]

        return results

    def retrieve_vector_only(
        self, query: str, top_k: int | None = None
    ) -> list[dict]:
        """Vector-only retrieval (for comparison / evaluation)."""
        return self.vector_store.query(query, top_k=top_k or settings.rerank_top_k)

    def retrieve_bm25_only(
        self, query: str, top_k: int | None = None
    ) -> list[dict]:
        """BM25-only retrieval (for comparison / evaluation)."""
        return self.bm25_index.query(query, top_k=top_k or settings.rerank_top_k)
