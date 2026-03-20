"""
FastAPI application for the Autodesk RAG Chatbot.

Endpoints:
- POST /chat             — Main chat endpoint (conversational, supports follow-ups)
- GET  /health           — Health check
- GET  /stats            — System statistics
- GET  /files/{filename} — Serve raw source HTML files (for source citations)

Note: The /evaluate endpoint has been removed from the API surface.
Evaluation is run offline via scripts/run_trulens_eval.py, which gives
full experiment tracking, per-question scoring, and DB persistence.
Running evaluation through the API would block the server for several minutes
and bypass the experiment naming and pipeline-selection flags.
"""

import os
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from loguru import logger

from src.config import settings
from src.ingestion.pipeline import run_ingestion_pipeline
from src.retrieval.hybrid_retriever import HybridRetriever
from src.generation.llm import LLMClient
from src.generation.rag_chain import RAGChain


# ============================================================
# Global state
# ============================================================
app_state = {
    "rag_chain": None,
    "retriever": None,
    "ready": False,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the RAG pipeline on startup."""
    logger.info("Initializing RAG chatbot...")
    try:
        # force=False (default): skips re-ingestion if ChromaDB already
        # has documents. Startup goes from ~60s to ~3s on every restart.
        # To force re-ingestion, either:
        #   (a) Set env var:  FORCE_INGEST=true uvicorn src.api.main:app ...
        #   (b) Run manually: python scripts/ingest.py --reset
        force = os.getenv("FORCE_INGEST", "false").lower() == "true"
        if force:
            logger.info("FORCE_INGEST=true — running full re-ingestion.")

        vector_store, bm25_index, chunks = run_ingestion_pipeline(force=force)
        retriever = HybridRetriever(vector_store, bm25_index)
        llm = LLMClient()
        llm.warmup()

        rag_chain = RAGChain(retriever, llm)
        app_state["rag_chain"] = rag_chain
        app_state["retriever"] = retriever
        app_state["ready"] = True

        chunk_count = vector_store.count()
        logger.info(f"RAG chatbot ready. {chunk_count} chunks indexed.")
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        raise

    yield

    logger.info("Shutting down RAG chatbot...")


app = FastAPI(
    title="Autodesk RAG Chatbot API",
    description="RAG-based conversational chatbot for Autodesk product documentation",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve raw HTML files so source citations are clickable in the UI.
_raw_dir = settings.raw_data_dir
if os.path.isdir(_raw_dir):
    app.mount(
        "/files",
        StaticFiles(directory=_raw_dir),
        name="raw_files",
    )
    logger.info(f"Static file server: /files → {_raw_dir}")
else:
    logger.warning(
        f"raw_data_dir not found ({_raw_dir}); /files endpoint disabled."
    )


# ============================================================
# Request / Response Models
# ============================================================

class ChatMessage(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'")
    content: str


class ChatRequest(BaseModel):
    question: str = Field(..., description="User's question", min_length=1)
    mode: str = Field(
        default="corpus",
        description="'corpus' (docs only) or 'blended' (docs + web search)"
    )
    chat_history: list[ChatMessage] = Field(
        default_factory=list,
        description="Previous conversation turns for context"
    )
    top_k: int = Field(default=5, ge=1, le=20, description="Number of context chunks")


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]
    mode: str
    query: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    llm_available: bool
    vector_store_count: int
    ready: bool


# ============================================================
# Endpoints
# ============================================================

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint. Supports conversational follow-ups."""
    if not app_state["ready"]:
        raise HTTPException(
            status_code=503,
            detail="System not ready. Still initializing."
        )

    rag_chain: RAGChain = app_state["rag_chain"]

    start = time.time()
    response = rag_chain.query(
        question=request.question,
        mode=request.mode,
        chat_history=[m.model_dump() for m in request.chat_history],
        top_k=request.top_k,
    )
    latency_ms = (time.time() - start) * 1000

    return ChatResponse(
        answer=response.answer,
        sources=response.to_dict()["sources"],
        mode=response.mode,
        query=response.query,
        latency_ms=round(latency_ms, 1),
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    if not app_state["ready"]:
        return HealthResponse(
            status="initializing",
            llm_available=False,
            vector_store_count=0,
            ready=False,
        )

    rag_chain: RAGChain = app_state["rag_chain"]
    health_info = rag_chain.is_healthy()

    return HealthResponse(
        status="healthy" if health_info["llm_available"] else "degraded",
        llm_available=health_info["llm_available"],
        vector_store_count=health_info["vector_store_count"],
        ready=True,
    )


@app.get("/stats")
async def stats():
    """System statistics."""
    if not app_state["ready"]:
        return {"status": "not_ready"}

    rag_chain: RAGChain = app_state["rag_chain"]
    return {
        "vector_store_chunks": rag_chain.retriever.vector_store.count(),
        "llm_model": settings.llm_model,
        "embedding_model": settings.embedding_model,
        "reranker_model": settings.reranker_model,
    }