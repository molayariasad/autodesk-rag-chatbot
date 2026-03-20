# src/ingestion/pipeline_sentence_window.py
"""
Ingestion pipeline variant: Sentence Window.

Identical to the optimised pipeline except it uses
SentenceWindowChunker instead of the standard Chunker.
This produces a separate ChromaDB collection
('autodesk_docs_sentence_window') and a separate BM25 pickle
so you can A/B between pipelines without re-ingesting everything.

Called by run_trulens_eval.py when --pipeline sentence_window is passed.
"""

from pathlib import Path
from loguru import logger

from src.config import settings
from src.ingestion.html_parser import parse_directory
from src.ingestion.sentence_window_chunker import SentenceWindowChunker
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_retriever import BM25Index

# Separate collection / index paths so existing runs are untouched
_SW_COLLECTION  = "autodesk_docs_sentence_window"
_SW_BM25_PATH   = Path(settings.processed_data_dir) / "bm25_sentence_window.pkl"
_SW_STATE_PATH  = Path(settings.processed_data_dir) / "sw_ingestion_done.flag"


def _safe_reset_collection(vector_store: VectorStore) -> None:
    """
    Delete the sentence-window ChromaDB collection only if it exists.

    VectorStore.reset() calls delete_collection() unconditionally, which
    raises chromadb.errors.NotFoundError on the very first --force-ingest
    run because the collection was never created yet.  This helper checks
    the live collection list before deleting so the first run never errors.
    """
    existing_names = [c.name for c in vector_store.client.list_collections()]
    if _SW_COLLECTION in existing_names:
        vector_store.client.delete_collection(_SW_COLLECTION)
        logger.info(f"Dropped existing ChromaDB collection '{_SW_COLLECTION}'")
    else:
        logger.info(f"Collection '{_SW_COLLECTION}' does not exist yet — skipping delete")
    # Force the VectorStore to re-create the collection on next access
    vector_store._collection = None


def run_sentence_window_pipeline(
    force:       bool = False,
    window_size: int  = 2,
):
    """
    Build or load the sentence-window index.

    Parameters
    ----------
    force       : re-ingest even if the flag file exists
    window_size : sentences on each side to include when expanding

    Returns
    -------
    (vector_store, bm25_index, chunks)
    """
    # --- Override collection name for isolation ---
    # We temporarily patch settings so VectorStore uses the SW collection.
    _orig_collection = settings.chroma_collection_name
    settings.chroma_collection_name = _SW_COLLECTION

    vector_store = VectorStore()
    bm25_index   = BM25Index()

    already_done = (
        _SW_STATE_PATH.exists()
        and _SW_BM25_PATH.exists()
        and vector_store.count() > 0
    )

    if already_done and not force:
        logger.info(
            f"Sentence-window index already built "
            f"({vector_store.count()} chunks). "
            "Pass force=True to re-ingest."
        )
        bm25_index.load(_SW_BM25_PATH)
        settings.chroma_collection_name = _orig_collection
        return vector_store, bm25_index, []

    logger.info("Running sentence-window ingestion pipeline...")

    # 1. Parse HTML (reuse the optimised parser)
    raw_dir = Path(settings.raw_data_dir)
    parsed_docs = parse_directory(raw_dir)
    logger.info(f"Parsed {len(parsed_docs)} documents")

    # 2. Chunk into sentences
    chunker = SentenceWindowChunker(window_size=window_size)
    chunks  = chunker.chunk_documents(parsed_docs)
    logger.info(f"Created {len(chunks)} sentence chunks")

    # 3. Wipe and rebuild vector store (safe: checks existence before deleting)
    _safe_reset_collection(vector_store)
    vector_store.add_chunks(chunks)

    # 4. Build BM25 index
    bm25_index.build_index(chunks)
    bm25_index.save(_SW_BM25_PATH)

    # 5. Write completion flag
    _SW_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _SW_STATE_PATH.write_text(
        f"sentence_window ingestion complete: {len(chunks)} chunks, window={window_size}"
    )
    logger.info("Sentence-window ingestion complete.")

    settings.chroma_collection_name = _orig_collection
    return vector_store, bm25_index, chunks