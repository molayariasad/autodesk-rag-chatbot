# src/ingestion/chunker_baseline.py
"""
Baseline Chunker — Experiment 01_baseline.

Intentional differences from the optimized chunker (chunker.py):

  1. chunk_size=512, chunk_overlap=64 — the initial defaults before
     corpus analysis drove us to chunk_size=1200, chunk_overlap=200.
     At 512 chars, the median document (1,235 chars) splits into ~3
     fragments, each too small to carry a coherent answer. This produces
     ~9,000+ chunks vs ~5,000 with the optimized settings.

  2. Tables as separate chunks — comparison tables are not merged into
     the main text. This breaks feature comparisons apart from their
     surrounding context, hurting "difference between X and Y" queries.

  3. Prefix not counted against chunk budget — the metadata prefix
     inflates chunk sizes past the target, creating inconsistent sizing.

  4. No single-chunk optimization — documents that fit within chunk_size
     are still run through the splitter unnecessarily.

These deficiencies are preserved intentionally to establish the baseline
lower bound for the experiment tracking comparison.
"""

from dataclasses import dataclass
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from src.ingestion.html_parser_baseline import ParsedDocument

# Baseline config — NOT read from settings to ensure experiment isolation.
# Changing settings.chunk_size would affect the optimized pipeline too.
BASELINE_CHUNK_SIZE    = 512
BASELINE_CHUNK_OVERLAP = 64


@dataclass
class Chunk:
    chunk_id: str
    text: str
    metadata: dict

    def __repr__(self):
        return (
            f"Chunk(id={self.chunk_id}, chars={len(self.text)}, "
            f"src={self.metadata.get('source_file', '?')})"
        )


def _build_context_prefix(doc: ParsedDocument) -> str:
    parts = []
    if doc.product_name:
        parts.append(f"Product: {doc.product_name}")
    if doc.title and doc.title != doc.source_file:
        parts.append(f"Title: {doc.title[:80]}")
    if doc.page_type != "unknown":
        parts.append(f"Type: {doc.page_type}")
    return " | ".join(parts) + "\n\n" if parts else ""


def _build_metadata(doc: ParsedDocument, chunk_type: str, chunk_index: int) -> dict:
    return {
        "source_file": doc.source_file,
        "source_path": doc.source_path,
        "title": doc.title,
        "product_name": doc.product_name or "",
        "page_type": doc.page_type,
        "chunk_type": chunk_type,
        "chunk_index": chunk_index,
    }


def chunk_document(doc: ParsedDocument) -> list[Chunk]:
    """
    Baseline chunking strategy.

    Differences from optimized:
    - chunk_size=512 (not data-driven 1200)
    - Tables chunked separately (not merged into main content)
    - Prefix NOT counted against chunk budget (can inflate chunk size)
    - No single-chunk optimization for small documents
    """
    chunks = []
    prefix = _build_context_prefix(doc)

    # Baseline: prefix size NOT subtracted from budget (inflates chunks)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=BASELINE_CHUNK_SIZE,
        chunk_overlap=BASELINE_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", ", ", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )

    # Main content chunks
    text_parts = splitter.split_text(doc.content)
    for i, text in enumerate(text_parts):
        chunks.append(Chunk(
            chunk_id=f"{doc.source_file}::text::{i}",
            text=prefix + text,
            metadata=_build_metadata(doc, "text", i),
        ))

    # Baseline: tables as SEPARATE chunks (not merged into main text)
    # This breaks comparison tables away from their surrounding context.
    for j, table in enumerate(doc.tables_markdown):
        table_parts = splitter.split_text(table)
        for k, part in enumerate(table_parts):
            chunks.append(Chunk(
                chunk_id=f"{doc.source_file}::table::{j}::{k}",
                text=prefix + part,
                metadata=_build_metadata(doc, "table", j * 100 + k),
            ))

    logger.debug(
        f"  {doc.source_file}: {len(chunks)} baseline chunks "
        f"(chunk_size={BASELINE_CHUNK_SIZE})"
    )
    return chunks


def chunk_documents(documents: list[ParsedDocument]) -> list[Chunk]:
    all_chunks = []
    for doc in documents:
        all_chunks.extend(chunk_document(doc))
    logger.info(
        f"Baseline chunking complete: {len(all_chunks)} chunks "
        f"from {len(documents)} docs "
        f"(chunk_size={BASELINE_CHUNK_SIZE}, overlap={BASELINE_CHUNK_OVERLAP})"
    )
    return all_chunks