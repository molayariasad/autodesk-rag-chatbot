"""
Vector store backed by ChromaDB with sentence-transformers embeddings.

Design decisions:
- BGE-small-en-v1.5: Best quality-to-size ratio for English retrieval tasks.
  Outperforms all-MiniLM-L6-v2 on MTEB benchmarks while still being fast on CPU.
- ChromaDB: Lightweight, embedded, no separate server process needed.
  Perfect for a PoC. In production, would migrate to Qdrant or Weaviate.
"""

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
from loguru import logger
from pathlib import Path

from src.config import settings
from src.ingestion.chunker import Chunk


class VectorStore:
    """ChromaDB-backed vector store with sentence-transformer embeddings."""

    def __init__(self):
        self._embed_model = None
        self._client = None
        self._collection = None

    @property
    def embed_model(self) -> SentenceTransformer:
        if self._embed_model is None:
            logger.info(f"Loading embedding model: {settings.embedding_model}")
            self._embed_model = SentenceTransformer(settings.embedding_model)
        return self._embed_model

    @property
    def client(self) -> chromadb.ClientAPI:
        if self._client is None:
            persist_dir = Path(settings.chroma_persist_dir)
            persist_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=str(persist_dir),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
        return self._client

    @property
    def collection(self) -> chromadb.Collection:
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=settings.chroma_collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def add_chunks(self, chunks: list[Chunk], batch_size: int = 64) -> None:
        """Embed chunks and add to ChromaDB."""
        if not chunks:
            logger.warning("No chunks to add")
            return

        logger.info(f"Embedding {len(chunks)} chunks with {settings.embedding_model}...")
        
        texts = [c.text for c in chunks]
        ids = [c.chunk_id for c in chunks]
        metadatas = [c.metadata for c in chunks]

        # Batch embedding + insertion
        for i in range(0, len(chunks), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_ids = ids[i : i + batch_size]
            batch_meta = metadatas[i : i + batch_size]

            embeddings = self.embed_model.encode(
                batch_texts,
                show_progress_bar=False,
                normalize_embeddings=True,
            ).tolist()

            self.collection.upsert(
                ids=batch_ids,
                embeddings=embeddings,
                documents=batch_texts,
                metadatas=batch_meta,
            )

            if (i // batch_size) % 5 == 0:
                logger.debug(f"  Embedded batch {i // batch_size + 1}")

        logger.info(f"Added {len(chunks)} chunks to ChromaDB (collection: {settings.chroma_collection_name})")

    def query(
        self,
        query_text: str,
        top_k: int | None = None,
        filter_dict: dict | None = None,
    ) -> list[dict]:
        """
        Query the vector store.
        
        Returns a list of dicts with keys: chunk_id, text, metadata, score.
        """
        top_k = top_k or settings.retrieval_top_k

        query_embedding = self.embed_model.encode(
            [query_text],
            normalize_embeddings=True,
        ).tolist()

        where_filter = filter_dict if filter_dict else None

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        # Convert ChromaDB results to a flat list of dicts
        hits = []
        if results and results["ids"]:
            for i, chunk_id in enumerate(results["ids"][0]):
                # ChromaDB returns cosine distance; convert to similarity
                distance = results["distances"][0][i]
                similarity = 1.0 - distance  # cosine distance → similarity

                hits.append({
                    "chunk_id": chunk_id,
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "score": similarity,
                    "source": "vector",
                })

        return hits

    def count(self) -> int:
        """Return total number of chunks in the collection."""
        return self.collection.count()

    def reset(self) -> None:
        """Delete and recreate the collection."""
        self.client.delete_collection(settings.chroma_collection_name)
        self._collection = None
        logger.info("Vector store reset.")
