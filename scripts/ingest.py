#!/usr/bin/env python3
"""
Standalone ingestion script.
Run this to parse HTML files and build the vector index before starting the API.

Usage:
    python scripts/ingest.py --data-dir ./data/raw
    python scripts/ingest.py --data-dir /path/to/html/files
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger
from src.ingestion.pipeline import run_ingestion_pipeline
from src.config import settings


def main():
    parser = argparse.ArgumentParser(description="Ingest HTML files into the RAG pipeline")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=settings.raw_data_dir,
        help="Directory containing HTML files",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear existing vector store before ingesting",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        sys.exit(1)

    html_count = len(list(data_dir.glob("*.html")))
    logger.info(f"Found {html_count} HTML files in {data_dir}")

    if args.reset:
        from src.retrieval.vector_store import VectorStore
        vs = VectorStore()
        vs.reset()
        logger.info("Vector store cleared.")

    vector_store, bm25_index, chunks = run_ingestion_pipeline(data_dir=str(data_dir))

    logger.info(f"\nIngestion complete!")
    logger.info(f"  Chunks in vector store: {vector_store.count()}")
    logger.info(f"  BM25 index size: {len(chunks)}")
    logger.info(f"  Report saved to: {settings.processed_data_dir}/ingestion_report.json")


if __name__ == "__main__":
    main()
