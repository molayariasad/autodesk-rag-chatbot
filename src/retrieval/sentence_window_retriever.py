# src/retrieval/sentence_window_retriever.py
"""
Sentence Window Retriever.

Drop-in replacement for HybridRetriever that adds post-retrieval
window expansion: each retrieved sentence chunk is replaced by a
window of its surrounding sentences from the same parent document.

Architecture
────────────
  1. Delegate to the underlying HybridRetriever
     (vector + BM25 + RRF + cross-encoder reranking — unchanged).
  2. For each result, read parent_doc_id, sentence_index, window_size
     from its metadata.
  3. Query ChromaDB for siblings: sentence_index in
     [i - window_size … i + window_size], same parent_doc_id.
  4. Sort siblings by sentence_index and concatenate their text.
  5. Replace chunk["text"] with the expanded window text.
  6. The chunk_id, metadata, and all scores stay unchanged so that
     TruLens, the evaluator, and the RAG chain receive the exact
     same dict schema they expect from a standard retrieval call.

TruLens compatibility
─────────────────────
  The evaluator in evaluator.py does:

      retrieval_results = rag_chain.retriever.retrieve(query, top_k=5)
      context_chunks    = [r.get("text", "") for r in retrieval_results]

  Because we only modify chunk["text"] (widening it), and every other
  field is preserved, TruLens sees:
    • Context Relevance: scored against richer, coherent window text  ✓
    • Groundedness:      LLM answer grounded in coherent paragraphs   ✓
    • Answer Relevance:  unchanged — same question, same pipeline      ✓

  Zero changes are needed in evaluator.py, rag_chain.py, or prompts.py.
"""

import chromadb
from loguru import logger

from src.config import settings
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_retriever import BM25Index


class SentenceWindowRetriever:
    """
    Hybrid retriever with sentence-window expansion.

    Parameters
    ----------
    vector_store : VectorStore
        The same ChromaDB-backed store used for retrieval.
        We also use it directly to fetch sibling sentence chunks.
    bm25_index   : BM25Index
        The keyword index (unchanged from standard pipeline).
    window_size  : int
        Number of sentences on each side to include in the expanded
        context.  Overrides the window_size stored in chunk metadata
        if provided.  Default None → use per-chunk metadata value
        (set during ingestion by SentenceWindowChunker).
    """

    def __init__(
        self,
        vector_store: VectorStore,
        bm25_index:   BM25Index,
        window_size:  int | None = None,
    ):
        self._hybrid = HybridRetriever(vector_store, bm25_index)
        self._vector_store = vector_store
        self._window_size  = window_size

        # Expose sub-components that RAGChain.is_healthy() accesses
        self.vector_store = vector_store
        self.bm25_index   = bm25_index

    # ──────────────────────────────────────────────────────────
    # Window expansion
    # ──────────────────────────────────────────────────────────

    def _expand_chunk(self, hit: dict) -> dict:
        """
        Fetch sibling sentence chunks from ChromaDB and replace the
        hit's text with the concatenated window.

        If metadata is missing (e.g. non-sentence-window chunks in the
        same collection) the original text is returned unchanged.
        """
        meta = hit.get("metadata", {})
        parent_doc_id  = meta.get("parent_doc_id")
        sentence_index = meta.get("sentence_index")
        total_sentences = meta.get("total_sentences")

        if parent_doc_id is None or sentence_index is None:
            # Not a sentence-window chunk — return as-is
            return hit

        # Determine how many neighbors to fetch
        w = self._window_size if self._window_size is not None else meta.get("window_size", 2)

        lo = max(0, sentence_index - w)
        hi_exclusive = sentence_index + w + 1
        if total_sentences is not None:
            hi_exclusive = min(hi_exclusive, total_sentences)

        # Build list of sibling chunk_ids  (format: {parent_doc_id}::sent_{idx:04d})
        sibling_ids = [
            f"{parent_doc_id}::sent_{i:04d}"
            for i in range(lo, hi_exclusive)
        ]

        try:
            results = self._vector_store.collection.get(
                ids=sibling_ids,
                include=["documents", "metadatas"],
            )
        except Exception as e:
            logger.warning(f"Window expansion fetch failed for {hit['chunk_id']}: {e}")
            return hit

        if not results or not results.get("ids"):
            return hit

        # Sort siblings by sentence_index and concatenate
        pairs = list(zip(
            results["ids"],
            results["documents"],
            results["metadatas"],
        ))
        pairs.sort(key=lambda x: x[2].get("sentence_index", 0))

        window_text = " ".join(doc for _, doc, _ in pairs if doc)

        expanded = dict(hit)
        expanded["text"] = window_text
        expanded["metadata"] = {
            **meta,
            "window_lo":      lo,
            "window_hi":      hi_exclusive - 1,
            "window_expanded": True,
        }
        return expanded

    # ──────────────────────────────────────────────────────────
    # Public retrieval API  (mirrors HybridRetriever exactly)
    # ──────────────────────────────────────────────────────────

    def retrieve(
        self,
        query:         str,
        top_k:         int | None = None,
        use_reranker:  bool       = True,
        filter_dict:   dict | None = None,
    ) -> list[dict]:
        """
        Full sentence-window pipeline:
          hybrid retrieval → dedup → window expansion.

        The returned dicts have identical keys to HybridRetriever.retrieve()
        so every downstream consumer (RAGChain, evaluator, TruLens) works
        without modification.
        """
        top_k = top_k or settings.rerank_top_k

        hits = self._hybrid.retrieve(
            query        = query,
            top_k        = top_k,
            use_reranker = use_reranker,
            filter_dict  = filter_dict,
        )

        expanded = [self._expand_chunk(h) for h in hits]

        # Deduplicate: if two sentence hits belong to the same parent and
        # their expanded windows overlap, keep only the one with the higher
        # rerank/rrf score (already at the top after sorting).
        seen_parents: set[str] = set()
        deduped: list[dict] = []
        for hit in expanded:
            pid = hit.get("metadata", {}).get("parent_doc_id", hit["chunk_id"])
            if pid not in seen_parents:
                seen_parents.add(pid)
                deduped.append(hit)

        logger.debug(
            f"SentenceWindowRetriever: {len(hits)} hits → "
            f"{len(deduped)} after window expansion + dedup"
        )
        return deduped

    # Pass-through methods used by evaluator / health checks

    def retrieve_vector_only(self, query: str, top_k: int | None = None) -> list[dict]:
        hits = self._hybrid.retrieve_vector_only(query, top_k)
        return [self._expand_chunk(h) for h in hits]

    def retrieve_bm25_only(self, query: str, top_k: int | None = None) -> list[dict]:
        hits = self._hybrid.retrieve_bm25_only(query, top_k)
        return [self._expand_chunk(h) for h in hits]
