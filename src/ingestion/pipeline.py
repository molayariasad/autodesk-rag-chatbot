"""
Ingestion pipeline: HTML files → Parsed Docs → Chunks → Vector Store + BM25 Index.

- This is the main entry point for data processing.
- Added force parameter to run_ingestion_pipeline(). When force=False (default),
  ingestion is skipped if ChromaDB already contains documents. This reduces
  repeated eval startup.
- Added _build_bm25_from_vector_store() to reconstruct the BM25 index from
  already-stored ChromaDB documents without re-parsing any HTML files.
"""

import json
import time
import uuid
from pathlib import Path

from loguru import logger

from src.config import settings
from src.ingestion.html_parser import parse_directory, ParsedDocument
from src.ingestion.chunker import chunk_documents, Chunk
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_retriever import BM25Index


def save_processing_report(
    documents: list[ParsedDocument],
    chunks: list[Chunk],
    output_dir: str,
):
    """Save a JSON report of the ingestion process for documentation."""
    report = {
        "total_html_files_processed": len(documents),
        "total_chunks_created": len(chunks),
        "documents": [],
    }

    for doc in documents:
        doc_chunks = [c for c in chunks if c.metadata["source_file"] == doc.source_file]
        report["documents"].append({
            "file": doc.source_file,
            "title": doc.title,
            "product": doc.product_name,
            "page_type": doc.page_type,
            "char_count": doc.char_count,
            "num_chunks": len(doc_chunks),
            "num_tables": len(doc.tables_markdown),
        })

    report["stats"] = {
        "avg_chars_per_doc": (
            sum(d.char_count for d in documents) / len(documents) if documents else 0
        ),
        "avg_chunks_per_doc": len(chunks) / len(documents) if documents else 0,
        "page_type_distribution": {},
        "product_distribution": {},
    }

    for doc in documents:
        pt = doc.page_type
        report["stats"]["page_type_distribution"][pt] = (
            report["stats"]["page_type_distribution"].get(pt, 0) + 1
        )
        prod = doc.product_name or "unknown"
        report["stats"]["product_distribution"][prod] = (
            report["stats"]["product_distribution"].get(prod, 0) + 1
        )

    output_path = Path(output_dir) / "ingestion_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, default=str))
    logger.info(f"Ingestion report saved to {output_path}")
    return report


def _build_bm25_from_vector_store(vector_store: VectorStore) -> BM25Index:
    """
    Reconstruct BM25Index from documents already stored in ChromaDB.

    Called on the fast path when ingestion is skipped. ChromaDB stores the
    full chunk text and metadata, so BM25 can be rebuilt without re-parsing
    any HTML files. This keeps BM25 and vector search in sync.
    """
    logger.info("Rebuilding BM25 index from existing ChromaDB documents...")

    collection  = vector_store.collection
    total       = vector_store.count()
    batch_size  = 500
    all_chunks  = []

    for offset in range(0, total, batch_size):
        result = collection.get(
            limit=batch_size,
            offset=offset,
            include=["documents", "metadatas"],
        )
        for doc_text, metadata in zip(result["documents"], result["metadatas"]):
            # Use the stored chunk_id if available; otherwise generate one.
            chunk_id = metadata.get("chunk_id", str(uuid.uuid4()))
            all_chunks.append(Chunk(
                chunk_id=chunk_id,
                text=doc_text,
                metadata=metadata,
            ))

    bm25_index = BM25Index()
    bm25_index.build_index(all_chunks)
    logger.info(f"BM25 index rebuilt from ChromaDB: {len(all_chunks)} chunks")
    return bm25_index


def run_ingestion_pipeline(
    data_dir: str | None = None,
    persist: bool = True,
    force: bool = False,
) -> tuple[VectorStore, BM25Index, list[Chunk]]:
    """
    Run the full ingestion pipeline.

    Args:
        data_dir: Path to directory containing HTML files.
        persist:  Whether to persist the vector store to disk.
        force:    If False (default), skip parsing/chunking/embedding when
                  ChromaDB already contains documents. Pass True to always
                  re-ingest — required after updating the HTML corpus or
                  modifying chunking/parsing logic.

    Returns:
        Tuple of (VectorStore, BM25Index, list of chunks).
        On the fast path the chunks list is empty — use vector_store.count()
        for the chunk count instead.
    """
    data_dir = data_dir or settings.raw_data_dir
    logger.info(f"Ingestion pipeline: data_dir={data_dir}, force={force}")

    vector_store = VectorStore()

    # ── Fast path: index already populated ───────────────────────────────
    if not force and vector_store.count() > 0:
        logger.info(
            f"ChromaDB already contains {vector_store.count()} chunks — "
            f"skipping ingestion. Use force=True (or --force-ingest) to re-ingest."
        )
        bm25_index = _build_bm25_from_vector_store(vector_store)
        return vector_store, bm25_index, []

    # ── Full ingestion path ───────────────────────────────────────────────
    logger.info(f"Starting full ingestion from: {data_dir}")
    start_time = time.time()

    logger.info("Step 1/4: Parsing HTML files...")
    documents = parse_directory(data_dir)
    if not documents:
        raise ValueError(f"No valid documents found in {data_dir}")

    logger.info("Step 2/4: Chunking documents...")
    chunks = chunk_documents(documents)

    logger.info("Step 3/4: Building vector store (embedding chunks)...")
    vector_store.add_chunks(chunks)

    logger.info("Step 4/4: Building BM25 index...")
    bm25_index = BM25Index()
    bm25_index.build_index(chunks)

    save_processing_report(documents, chunks, settings.processed_data_dir)

    elapsed = time.time() - start_time
    logger.info(
        f"Ingestion complete in {elapsed:.1f}s: "
        f"{len(documents)} docs → {len(chunks)} chunks"
    )

    return vector_store, bm25_index, chunks