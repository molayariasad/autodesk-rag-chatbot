# src/ingestion/pipeline_baseline.py
"""
Baseline ingestion pipeline — used exclusively by Experiment 01_baseline.

Run via:
    python scripts/run_trulens_eval.py --pipeline baseline --experiment 01_baseline

Do not modify. This file intentionally preserves v1 parser and chunker
behaviour (chunk_size=512, trafilatura precision mode, no alias normalization)
to establish the lower bound that the optimized pipeline is measured against.
See html_parser_baseline.py and chunker_baseline.py for documented deficiencies.
"""

import json
import time
import uuid
from pathlib import Path

from loguru import logger

from src.config import settings
from src.ingestion.html_parser_baseline import parse_directory
from src.ingestion.chunker_baseline import chunk_documents, Chunk
from src.retrieval.bm25_retriever import BM25Index

# Isolated ChromaDB collection — does not touch the optimized index
BASELINE_COLLECTION = "autodesk_docs_baseline"


def _get_baseline_vector_store():
    """
    Return a VectorStore pointed at the baseline ChromaDB collection.
    Isolated from the optimized collection so both can coexist.
    """
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    from sentence_transformers import SentenceTransformer
    from src.config import settings as cfg

    class BaselineVectorStore:
        def __init__(self):
            persist_dir = Path(cfg.chroma_persist_dir)
            persist_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=str(persist_dir),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            self._collection = self._client.get_or_create_collection(
                name=BASELINE_COLLECTION,
                metadata={"hnsw:space": "cosine"},
            )
            self._embed_model = SentenceTransformer(cfg.embedding_model)

        def count(self) -> int:
            return self._collection.count()

        def add_chunks(self, chunks: list[Chunk], batch_size: int = 64) -> None:
            if not chunks:
                return
            texts     = [c.text for c in chunks]
            ids       = [c.chunk_id for c in chunks]
            metadatas = [c.metadata for c in chunks]
            for i in range(0, len(chunks), batch_size):
                embeddings = self._embed_model.encode(
                    texts[i:i+batch_size],
                    show_progress_bar=False,
                    normalize_embeddings=True,
                ).tolist()
                self._collection.upsert(
                    ids=ids[i:i+batch_size],
                    embeddings=embeddings,
                    documents=texts[i:i+batch_size],
                    metadatas=metadatas[i:i+batch_size],
                )
            logger.info(f"Baseline: added {len(chunks)} chunks to '{BASELINE_COLLECTION}'")

        def query(self, query_text: str, top_k: int = 10, filter_dict=None):
            embedding = self._embed_model.encode(
                [query_text], normalize_embeddings=True
            ).tolist()
            results = self._collection.query(
                query_embeddings=embedding,
                n_results=top_k,
                where=filter_dict,
                include=["documents", "metadatas", "distances"],
            )
            hits = []
            if results and results["ids"]:
                for i, chunk_id in enumerate(results["ids"][0]):
                    hits.append({
                        "chunk_id": chunk_id,
                        "text": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "score": 1.0 - results["distances"][0][i],
                        "source": "vector",
                    })
            return hits

        def reset(self):
            self._client.delete_collection(BASELINE_COLLECTION)
            self._collection = self._client.get_or_create_collection(
                name=BASELINE_COLLECTION,
                metadata={"hnsw:space": "cosine"},
            )

    return BaselineVectorStore()


def _build_bm25_from_baseline_store(vector_store) -> BM25Index:
    logger.info("Rebuilding baseline BM25 from ChromaDB...")
    collection = vector_store._collection
    total      = vector_store.count()
    all_chunks = []
    for offset in range(0, total, 500):
        result = collection.get(
            limit=500, offset=offset,
            include=["documents", "metadatas"],
        )
        for doc_text, metadata in zip(result["documents"], result["metadatas"]):
            all_chunks.append(Chunk(
                chunk_id=metadata.get("chunk_id", str(uuid.uuid4())),
                text=doc_text,
                metadata=metadata,
            ))
    bm25 = BM25Index()
    bm25.build_index(all_chunks)
    logger.info(f"Baseline BM25 rebuilt: {len(all_chunks)} chunks")
    return bm25


def run_baseline_pipeline(
    data_dir: str | None = None,
    force: bool = False,
):
    """
    Run the baseline ingestion pipeline.

    Returns (vector_store, bm25_index, chunks) using the baseline parser
    and chunker. Uses an isolated ChromaDB collection so it never touches
    the optimized index.
    """
    data_dir = data_dir or settings.raw_data_dir
    vector_store = _get_baseline_vector_store()

    if not force and vector_store.count() > 0:
        logger.info(
            f"Baseline index already contains {vector_store.count()} chunks — "
            f"skipping ingestion."
        )
        bm25_index = _build_bm25_from_baseline_store(vector_store)
        return vector_store, bm25_index, []

    logger.info(f"Running BASELINE ingestion from: {data_dir}")
    start = time.time()

    documents = parse_directory(data_dir)
    if not documents:
        raise ValueError(f"No documents found in {data_dir}")

    chunks = chunk_documents(documents)
    vector_store.add_chunks(chunks)

    bm25_index = BM25Index()
    bm25_index.build_index(chunks)

    logger.info(
        f"Baseline ingestion complete in {time.time()-start:.1f}s: "
        f"{len(documents)} docs → {len(chunks)} chunks"
    )
    return vector_store, bm25_index, chunks