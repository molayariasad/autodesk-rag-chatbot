"""
BM25 keyword-based retriever for hybrid search.

Why BM25 alongside vector search:
- Vector search excels at semantic similarity but can miss exact product names,
  version numbers, and technical terms.
- BM25 catches exact keyword matches like "AutoCAD LT", "Fusion 360", "Maya 2024".
- Hybrid fusion (RRF) of both gives consistently better recall than either alone.
"""

"""
BM25 keyword-based retriever for hybrid search.

v2 changes (Issue 2b — BM25 Exact-Match Recall = 0):
- _tokenize() now emits unigrams + bigrams.
  This makes compound product names ("AutoCAD LT", "Fusion 360", "3ds Max")
  searchable as single units, not just as disconnected tokens.
  Example: query "AutoCAD LT 2D drafting" → tokens include "autocad_lt" which
  matches the bigram stored during indexing of the corpus.
- Added verify_term_in_index() diagnostic to distinguish two failure modes:
    (a) term not in corpus (parsing/ingestion bug upstream)
    (b) term in corpus but BM25 score = 0 (retriever config bug)
  This was the missing observability that made Issue 2b hard to debug.
- min token length changed from >1 (len > 1) to >=2 (len >= 2) — explicit
  and consistent with the docstring.
"""

import re
import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi
from loguru import logger

from src.config import settings
from src.ingestion.chunker import Chunk


def _tokenize(text: str) -> list[str]:
    """
    Tokenize text into unigrams and bigrams for BM25 indexing.

    Unigrams handle single-word terms: "solidworks", "revit", "maya".
    Bigrams handle compound product names: "autocad_lt", "fusion_360",
    "3ds_max", "autodesk_construction_cloud".

    Trade-off: index size approximately doubles vs unigrams-only.
    At <10k chunks this is negligible. BM25Okapi scoring is slightly
    diluted because the vocabulary is larger, but recall improves
    substantially for multi-word product name queries.
    """
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    unigrams = [t for t in tokens if len(t) >= 2]

    # Bigrams: slide a window of size 2 over the unigram list.
    # "autocad lt" → unigrams ["autocad", "lt"] → bigram "autocad_lt"
    bigrams = [
        f"{unigrams[i]}_{unigrams[i + 1]}"
        for i in range(len(unigrams) - 1)
    ]

    return unigrams + bigrams


class BM25Index:
    """BM25Okapi-based keyword retriever with bigram support."""

    def __init__(self):
        self._index: BM25Okapi | None = None
        self._chunks: list[Chunk] = []
        self._tokenized_corpus: list[list[str]] = []

    def build_index(self, chunks: list[Chunk]) -> None:
        """Build BM25 index from chunks."""
        self._chunks = chunks
        self._tokenized_corpus = [_tokenize(c.text) for c in chunks]
        self._index = BM25Okapi(self._tokenized_corpus)
        logger.info(f"BM25 index built: {len(chunks)} chunks, bigram tokenizer active")

    def query(self, query_text: str, top_k: int | None = None) -> list[dict]:
        """
        Query BM25 index.

        Returns list of dicts with keys: chunk_id, text, metadata, score, source.
        """
        if self._index is None:
            raise RuntimeError("BM25 index not built. Call build_index() first.")

        top_k = top_k or settings.retrieval_top_k
        tokenized_query = _tokenize(query_text)

        if not tokenized_query:
            return []

        scores = self._index.get_scores(tokenized_query)

        scored_indices = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )[:top_k]

        hits = []
        for idx, score in scored_indices:
            if score <= 0:
                continue
            chunk = self._chunks[idx]
            hits.append({
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "metadata": chunk.metadata,
                "score": float(score),
                "source": "bm25",
            })

        return hits

    def verify_term_in_index(self, term: str) -> dict:
        """
        Diagnostic: verify whether a specific term is present in the index corpus.

        Distinguishes two failure modes that both result in empty BM25 results:
        (a) Term is NOT in any chunk → upstream parsing/ingestion bug (html_parser).
        (b) Term IS in chunks but scores 0 → tokenizer or BM25 config bug.

        Usage during debugging:
            result = bm25_index.verify_term_in_index("solidworks")
            print(result)
            # → {'term': 'solidworks', 'bm25_hits': 3, 'term_in_raw_text': True,
            #    'found_in_files': ['adsk-abc.html'], 'diagnosis': 'OK'}
        """
        if self._index is None:
            return {"error": "Index not built"}

        term_lower = term.lower()

        # Check 1: is the term literally present in any chunk's raw text?
        files_with_term = [
            chunk.metadata.get("source_file", "?")
            for chunk in self._chunks
            if term_lower in chunk.text.lower()
        ]

        # Check 2: what does BM25 actually return?
        bm25_results = self.query(term, top_k=5)

        if not files_with_term:
            diagnosis = "UPSTREAM BUG: term not in any chunk — check html_parser.py (Issue 2a)"
        elif not bm25_results:
            diagnosis = "TOKENIZER BUG: term in corpus but BM25 score=0 — check _tokenize()"
        else:
            diagnosis = "OK"

        return {
            "term": term,
            "bm25_hits": len(bm25_results),
            "term_in_raw_text": len(files_with_term) > 0,
            "chunks_containing_term": len(files_with_term),
            "found_in_files": files_with_term[:5],  # cap at 5 for readability
            "diagnosis": diagnosis,
        }

    def save(self, path: str | Path) -> None:
        """Persist BM25 index to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "chunks": self._chunks,
            "tokenized_corpus": self._tokenized_corpus,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"BM25 index saved to {path}")

    def load(self, path: str | Path) -> None:
        """Load BM25 index from disk."""
        path = Path(path)
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._chunks = data["chunks"]
        self._tokenized_corpus = data["tokenized_corpus"]
        self._index = BM25Okapi(self._tokenized_corpus)
        logger.info(f"BM25 index loaded from {path} ({len(self._chunks)} chunks)")