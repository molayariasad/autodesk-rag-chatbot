"""
Centralized configuration for the Autodesk RAG Chatbot.
Loads from environment variables with sensible defaults.
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from dotenv import load_dotenv

load_dotenv()

# Project root: two levels up from this file
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # --- LLM ---
    llm_provider: str = "ollama"
    # llm_model: str = "mistral:7b-instruct-v0.3-q4_K_M"
    llm_model: str = "gemma3:4b"          # Best speed/quality on Apple MACBook M-series
    llm_base_url: str = "http://localhost:11434"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1024
    llm_keep_alive: str = "60m"

    # --- Embedding ---
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dimension: int = 384

    # --- Reranker ---
    # baseline reranker
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- ChromaDB ---
    chroma_persist_dir: str = str(PROJECT_ROOT / "data" / "chroma_db")
    chroma_collection_name: str = "autodesk_docs"

    # --- Retrieval ---
    #baseline weights before tuning:
    # retrieval_top_k: int = 10
    # rerank_top_k: int = 5
    # bm25_weight: float = 0.3
    # vector_weight: float = 0.7
    
    # Exp: 05_retrieval_weights--(tuned for better recall and LLM context (more candidates, more BM25 emphasis):
    retrieval_top_k: int = 15   # wider candidate pool before reranking
    rerank_top_k: int = 5       # keep the same — LLM context window constraint
    bm25_weight: float = 0.4    # BM25 up — rewards exact product name matches
    vector_weight: float = 0.6  # vector down correspondingly

    # --- Chunking ---
    chunk_size: int = 1200
    chunk_overlap: int = 300

    # --- Data ---
    raw_data_dir: str = str(PROJECT_ROOT / "data" / "raw")
    processed_data_dir: str = str(PROJECT_ROOT / "data" / "processed")

    # --- API ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # --- UI ---
    streamlit_port: int = 8501

    class Settings(BaseSettings):
        model_config = ConfigDict( 
        env_file=".env",
        env_file_encoding="utf-8",
    )


settings = Settings()
