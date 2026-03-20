# src/ingestion/sentence_window_chunker.py
"""
Sentence Window Chunker for Sentence Window Retrieval.

Core idea (popularised by LlamaIndex):
─────────────────────────────────────
  Index:    embed tiny "sentence" chunks  → highly precise embedding match
  Retrieve: find best sentence chunks via hybrid search
  Expand:   replace each retrieved sentence with its *surrounding window*
            of sentences before feeding to the LLM

Why this improves the RAG Triad:
  • Context Relevance ↑  — tiny index units match the query more tightly
                           than 1200-char blocks (less off-topic content
                           gets embedded alongside the relevant sentence).
  • Groundedness     ↑  — LLM receives a coherent paragraph window instead
                           of a hard-cut fixed-size chunk, so answers stay
                           grounded in natural prose.
  • Answer Relevance ↑  — wider window preserves the sentence's full
                           argumentative context, enabling more complete answers.

Metadata stored per sentence chunk
───────────────────────────────────
  parent_doc_id   : shared across all sentences in one source document
  sentence_index  : 0-based position in the parent document sentence list
  total_sentences : total sentence count in parent document
  window_size     : int — number of sentences on each side to expand to
  + all normal chunk metadata (title, source_file, product_name, …)

The parent_doc_id + sentence_index scheme allows the SentenceWindowRetriever
to reconstruct any window without a separate store — it queries ChromaDB
for the sibling sentence chunks by metadata filter.
"""

import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

# Re-use the existing Chunk dataclass from the standard chunker so that
# VectorStore.add_chunks() and BM25Index.build_index() accept our output
# without any modification.
from src.ingestion.chunker import Chunk


# ──────────────────────────────────────────────────────────────
# Sentence splitter (no NLTK dependency, Python 3.13 compatible)
# ──────────────────────────────────────────────────────────────

# Abbreviations that should NOT trigger a sentence split when followed by
# a capital letter.  All lowercase — we match against text.lower().
_ABBREVS = frozenset({
    # titles
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "rev", "gen", "sgt",
    # common English
    "etc", "vs", "approx", "dept", "est", "fig", "govt", "inc", "corp",
    "ltd", "no", "vol", "jan", "feb", "mar", "apr", "jun", "jul", "aug",
    "sep", "oct", "nov", "dec",
    # version / tech  (e.g. "v1", "v2", "rev")
    "v", "ver", "rev",
    # single letters  (e.g. "U.S.", "e.g.", "i.e.")
    *list("abcdefghijklmnopqrstuvwxyz"),
})

# Split on  .  !  ?  optionally followed by closing quote/paren,
# then whitespace.  Uses only fixed-width lookbehind (Python 3.13 safe).
_CANDIDATE_SPLIT = re.compile(r'(?<=[.!?])["\')]?\s+')


def split_into_sentences(text: str) -> list[str]:
    """
    Heuristic sentence splitter compatible with Python 3.13.

    Strategy:
      1. Split on any [.!?] followed by whitespace (broad candidates).
      2. Post-filter: re-join splits that look like abbreviations
         (single letter, known short word, or digit before the period).

    Falls back to returning the whole text as a single sentence if no
    split points are found (e.g. a heading or a very short snippet).
    """
    # Step 1 — raw candidate splits
    raw = _CANDIDATE_SPLIT.split(text)

    if len(raw) <= 1:
        return [text.strip()] if text.strip() else []

    # Step 2 — re-join false positives
    sentences: list[str] = []
    carry = raw[0]

    for fragment in raw[1:]:
        # Look at the last "word" of carry (before the period that caused the split)
        last_token = re.split(r'\s+', carry)[-1].rstrip('.!?"\')')

        # Rejoin if:
        #   (a) last token is a known abbreviation, OR
        #   (b) last token is all digits (e.g. "version 2.0 supports"),  OR
        #   (c) last token is a single uppercase letter (e.g. "U.S.")
        is_abbrev    = last_token.lower() in _ABBREVS
        is_digit     = last_token.isdigit()
        is_single_uc = len(last_token) == 1 and last_token.isupper()

        if is_abbrev or is_digit or is_single_uc:
            carry = carry + " " + fragment
        else:
            sentences.append(carry.strip())
            carry = fragment

    if carry.strip():
        sentences.append(carry.strip())

    return sentences if sentences else [text.strip()]


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────

def build_sentence_window_chunks(
    text: str,
    base_metadata: dict[str, Any],
    source_file: str,
    window_size: int = 2,
    min_sentence_len: int = 20,
) -> list[Chunk]:
    """
    Convert one document's extracted text into sentence-level Chunk objects.

    Parameters
    ----------
    text            : full extracted text of the document
    base_metadata   : metadata dict already assembled by the HTML parser
                      (title, product_name, page_type, …)
    source_file     : filename — stored in metadata for citation display
    window_size     : number of sentences on each side used during retrieval
                      expansion (stored in metadata; retriever reads it)
    min_sentence_len: sentences shorter than this (chars) are merged with
                      the preceding sentence.  Prevents stubs like "Yes."
                      from becoming their own embedding unit.

    Returns
    -------
    List of Chunk objects, one per (merged) sentence.  Each chunk carries
    parent_doc_id and sentence_index so the retriever can fetch neighbours.
    """
    sentences = split_into_sentences(text)

    # Merge very short sentences into the preceding one
    merged: list[str] = []
    for sent in sentences:
        if merged and len(sent) < min_sentence_len:
            merged[-1] = merged[-1] + " " + sent
        else:
            merged.append(sent)

    if not merged:
        return []

    # Stable parent ID shared by all sentence chunks of this document.
    # Using source_file as the seed so re-ingestion produces the same IDs
    # (important for ChromaDB upsert idempotency).
    parent_doc_id = str(uuid.uuid5(uuid.NAMESPACE_URL, source_file))
    total = len(merged)

    chunks: list[Chunk] = []
    for idx, sentence in enumerate(merged):
        # chunk_id encodes parent + position → deterministic, sortable
        chunk_id = f"{parent_doc_id}::sent_{idx:04d}"

        meta = {
            **base_metadata,
            "source_file":     source_file,
            "parent_doc_id":   parent_doc_id,
            "sentence_index":  idx,
            "total_sentences": total,
            "window_size":     window_size,
            "chunk_type":      "sentence_window",
        }

        chunks.append(Chunk(
            chunk_id=chunk_id,
            text=sentence,
            metadata=meta,
        ))

    return chunks


class SentenceWindowChunker:
    """
    Stateless chunker that converts a list of parsed documents into
    sentence-level chunks ready for dual-indexing (ChromaDB + BM25).

    Usage
    -----
    From the ingestion pipeline:

        from src.ingestion.sentence_window_chunker import SentenceWindowChunker

        chunker = SentenceWindowChunker(window_size=2)
        chunks  = chunker.chunk_documents(parsed_docs)
        # → pass to VectorStore.add_chunks() and BM25Index.build_index()
    """

    def __init__(self, window_size: int = 2, min_sentence_len: int = 50):
        self.window_size      = window_size
        self.min_sentence_len = min_sentence_len

    def chunk_documents(self, parsed_docs) -> list[Chunk]:
        """
        Parameters
        ----------
        parsed_docs : list of ParsedDocument objects from html_parser.parse_directory()

        Returns
        -------
        Flat list of sentence-level Chunk objects.
        """
        all_chunks: list[Chunk] = []

        for doc in parsed_docs:
            # Use full_text() to include tables, matching chunker.py's behaviour
            text = doc.full_text().strip()

            if not text:
                continue

            base_meta = {
                "source_file":  doc.source_file,
                "source_path":  doc.source_path,
                "title":        doc.title,
                "product_name": doc.product_name or "",
                "page_type":    doc.page_type,
            }

            doc_chunks = build_sentence_window_chunks(
                text             = text,
                base_metadata    = base_meta,
                source_file      = doc.source_file,
                window_size      = self.window_size,
                min_sentence_len = self.min_sentence_len,
            )
            all_chunks.extend(doc_chunks)

        logger.info(
            f"SentenceWindowChunker: {len(parsed_docs)} docs → "
            f"{len(all_chunks)} sentence chunks  "
            f"(window_size={self.window_size})"
        )
        return all_chunks