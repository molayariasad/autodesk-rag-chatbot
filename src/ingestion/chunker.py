"""
Document chunking with semantic awareness.

Strategy:
- Use RecursiveCharacterTextSplitter as the base (proven, robust).
- Prepend metadata (title, product name) to each chunk so the LLM has context
  even when a chunk is retrieved in isolation.
- Separate table chunks from text chunks to avoid breaking table structure.
- Track source provenance for citation.
- Small documents (content <= chunk_size) are kept as a SINGLE chunk.
  With median doc at 1,235 chars and chunk_size at 1,200, this keeps ~50%
  of the corpus whole — the ideal scenario for RAG.
- Prefix is counted against the chunk budget so it doesn't inflate sizes.
- Tables are merged into the main text instead of separate chunks when they
  fit, reducing total chunk count and keeping related content together.
"""

from dataclasses import dataclass
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from src.ingestion.html_parser import ParsedDocument
from src.config import settings


@dataclass
class Chunk:
    """A single chunk ready for embedding and storage."""
    chunk_id: str
    text: str
    metadata: dict

    def __repr__(self):
        return (
            f"Chunk(id={self.chunk_id}, chars={len(self.text)}, "
            f"src={self.metadata.get('source_file', '?')})"
        )


def _build_context_prefix(doc: ParsedDocument) -> str:
    """
    Build a metadata prefix to prepend to each chunk.

    This gives the LLM product/page context even for isolated chunks.
    Uses normalized product names (e.g., "Fusion 360" not "F360").
    """
    parts = []
    if doc.product_name:
        parts.append(f"Product: {doc.product_name}")
    if doc.title and doc.title != doc.source_file:
        # Truncate very long titles to save chunk budget
        title_display = doc.title[:80] + "..." if len(doc.title) > 80 else doc.title
        parts.append(f"Title: {title_display}")
    if doc.page_type != "unknown":
        parts.append(f"Type: {doc.page_type}")

    if parts:
        return " | ".join(parts) + "\n\n"
    return ""


def _build_metadata(doc: ParsedDocument, chunk_type: str, chunk_index: int) -> dict:
    """Build metadata dict for a chunk, including source_path for UI links."""
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
    Split a parsed document into chunks suitable for embedding.

    Strategy:
    - Merge tables into the main content (keeps related info together).
    - If the full text fits in one chunk, keep it whole.
    - Otherwise, split with RecursiveCharacterTextSplitter.
    - Prefix is counted against the chunk budget.
    """
    chunks = []
    prefix = _build_context_prefix(doc)
    prefix_len = len(prefix)

    # --- Merge content + tables into a single text ---
    # This keeps product comparison tables with their surrounding text,
    # which is critical for questions like "difference between X and Y"
    full_text = doc.content
    if doc.tables_markdown:
        full_text += "\n\n" + "\n\n".join(doc.tables_markdown)

    # --- Effective chunk budget (minus prefix) ---
    effective_chunk_size = max(settings.chunk_size - prefix_len, 200)

    # --- If the full text fits, keep it as ONE chunk ---
    if len(full_text) <= effective_chunk_size:
        chunk = Chunk(
            chunk_id=f"{doc.source_file}::full::0",
            text=prefix + full_text,
            metadata=_build_metadata(doc, "full", 0),
        )
        chunks.append(chunk)
        logger.debug(f"  {doc.source_file}: kept whole ({len(full_text)} chars)")
        return chunks

    # --- Otherwise, split ---
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=effective_chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", ", ", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )

    text_parts = splitter.split_text(full_text)

    for i, text in enumerate(text_parts):
        chunk = Chunk(
            chunk_id=f"{doc.source_file}::text::{i}",
            text=prefix + text,
            metadata=_build_metadata(doc, "text", i),
        )
        chunks.append(chunk)

    logger.debug(
        f"  {doc.source_file}: split into {len(chunks)} chunks "
        f"(content={len(full_text)}, effective_budget={effective_chunk_size})"
    )
    return chunks


def chunk_documents(documents: list[ParsedDocument]) -> list[Chunk]:
    """Chunk a list of parsed documents."""
    all_chunks = []
    whole_count = 0

    for doc in documents:
        doc_chunks = chunk_document(doc)
        if len(doc_chunks) == 1 and doc_chunks[0].metadata["chunk_type"] == "full":
            whole_count += 1
        all_chunks.extend(doc_chunks)

    logger.info(
        f"Chunking complete: {len(all_chunks)} chunks from {len(documents)} docs "
        f"({whole_count} kept whole, {len(documents) - whole_count} split)"
    )
    return all_chunks