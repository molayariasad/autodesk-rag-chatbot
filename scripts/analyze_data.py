#!/usr/bin/env python3
"""
Data Analysis / EDA script for the Autodesk HTML corpus.

Generates summary statistics and Plotly visualizations:
  - File size distribution
  - Extracted text length per document
  - Page type distribution
  - Top-15 product distribution
  - Chunk size distribution
  - Combined dashboard

Outputs saved as PNG to data/processed/corpus_analysis/
PNG requires kaleido: uv add kaleido

IMPORTANT: This script does NOT re-ingest or re-embed documents.
  - If the ChromaDB index is already populated, it reads chunk stats
    directly from ChromaDB (fast, no re-parsing).
  - If the index is empty, it parses HTML files to get doc stats,
    but does NOT write anything to ChromaDB.

Run:
    uv run python scripts/analyze_data.py
    uv run python scripts/analyze_data.py --data-dir ./data/raw
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from loguru import logger

from src.config import settings

# ── Output config ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR   = PROJECT_ROOT / "data" / "processed" / "corpus_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEMPLATE  = "plotly_white"
PNG_SCALE = 2
PNG_WIDTH = 1400


def _save(fig: go.Figure, stem: str, height: int = 500) -> None:
    """Save figure as PNG (git-friendly — no 92MB HTML files)."""
    out = OUTPUT_DIR / f"{stem}.png"
    fig.update_layout(height=height)
    fig.write_image(str(out), scale=PNG_SCALE, width=PNG_WIDTH, height=height)
    logger.info(f"Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Chunk stats — read from ChromaDB if available (no re-parsing needed)
# ══════════════════════════════════════════════════════════════════════════════

def _get_chunk_lengths_from_chroma() -> list[int] | None:
    """
    Read chunk text lengths directly from the existing ChromaDB index.
    Returns None if the index is empty or unavailable.
    This avoids re-parsing and re-embedding the HTML corpus just for EDA.
    """
    try:
        from src.retrieval.vector_store import VectorStore
        vs = VectorStore()
        total = vs.count()
        if total == 0:
            return None

        logger.info(f"Reading {total} chunk lengths from ChromaDB (no re-ingest)")
        lengths = []
        batch_size = 500
        for offset in range(0, total, batch_size):
            result = vs.collection.get(
                limit=batch_size,
                offset=offset,
                include=["documents"],
            )
            lengths.extend(len(doc) for doc in result["documents"])
        return lengths
    except Exception as e:
        logger.warning(f"Could not read from ChromaDB: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Main analysis
# ══════════════════════════════════════════════════════════════════════════════

def analyze_corpus(data_dir: str) -> dict:
    """
    Run EDA on the HTML corpus.

    Chunk lengths are read from ChromaDB if the index exists.
    Document-level stats require parsing the HTML files (read-only,
    no writes to ChromaDB or BM25 index).
    """
    from src.ingestion.html_parser import parse_directory

    data_dir  = Path(data_dir)
    html_files = sorted(data_dir.glob("*.html"))

    logger.info(f"Found {len(html_files)} HTML files in {data_dir}")

    # File-level stats (just stat() calls — instant)
    file_stats = [
        {"filename": fp.name, "size_kb": fp.stat().st_size / 1024}
        for fp in html_files
    ]
    file_df = pd.DataFrame(file_stats)

    # Document-level stats — parse HTML (read-only, no ChromaDB writes)
    logger.info("Parsing HTML files for document stats (read-only, no re-indexing)...")
    documents = parse_directory(data_dir)
    doc_stats = [
        {
            "filename":   doc.source_file,
            "title":      doc.title[:60],
            "chars":      doc.char_count,
            "product":    doc.product_name or "Unknown",
            "page_type":  doc.page_type,
            "num_tables": len(doc.tables_markdown),
        }
        for doc in documents
    ]
    doc_df = pd.DataFrame(doc_stats)

    # Chunk lengths — prefer ChromaDB (fast) over re-chunking (slow)
    chunk_lengths = _get_chunk_lengths_from_chroma()
    if chunk_lengths is None:
        logger.info("ChromaDB empty — computing chunk lengths from parsed docs...")
        from src.ingestion.chunker import chunk_documents
        # chunk_documents is read-only: it returns Chunk objects without
        # writing to any store. ChromaDB is only written via pipeline.py.
        chunks        = chunk_documents(documents)
        chunk_lengths = [len(c.text) for c in chunks]
        logger.info(f"Computed {len(chunk_lengths)} chunk lengths from parsed docs")
    else:
        logger.info(f"Loaded {len(chunk_lengths)} chunk lengths from ChromaDB")

    summary = {
        "total_html_files":     len(html_files),
        "files_with_content":   len(documents),
        "files_skipped":        len(html_files) - len(documents),
        "skip_rate":            f"{(len(html_files) - len(documents)) / max(len(html_files), 1):.1%}",
        "total_chunks":         len(chunk_lengths),
        "avg_chars_per_doc":    f"{doc_df['chars'].mean():.0f}" if len(doc_df) > 0 else "N/A",
        "median_chars_per_doc": f"{doc_df['chars'].median():.0f}" if len(doc_df) > 0 else "N/A",
        "avg_chunk_length":     f"{sum(chunk_lengths) / max(len(chunk_lengths), 1):.0f}",
        "chunk_size_setting":   settings.chunk_size,
        "chunk_overlap_setting": settings.chunk_overlap,
    }

    logger.info(f"\n{'='*50}")
    logger.info("CORPUS ANALYSIS SUMMARY")
    logger.info(f"{'='*50}")
    for k, v in summary.items():
        logger.info(f"  {k}: {v}")

    # Save skipped files list for audit
    parsed_names  = {doc.source_file for doc in documents}
    skipped_files = [fp.name for fp in html_files if fp.name not in parsed_names]
    skipped_path  = OUTPUT_DIR / "skipped_files.txt"
    skipped_path.write_text("\n".join(skipped_files))
    logger.info(f"Skipped files list saved to {skipped_path}")

    return {
        "file_df":      file_df,
        "doc_df":       doc_df,
        "chunk_lengths": chunk_lengths,
        "summary":      summary,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Plots
# ══════════════════════════════════════════════════════════════════════════════

def generate_plots(analysis: dict) -> None:
    file_df       = analysis["file_df"]
    doc_df        = analysis["doc_df"]
    chunk_lengths = analysis["chunk_lengths"]

    # 1. File size distribution
    if len(file_df) > 0:
        fig = px.histogram(
            file_df, x="size_kb", nbins=50,
            title="HTML File Size Distribution",
            labels={"size_kb": "File Size (KB)", "count": "Files"},
            color_discrete_sequence=["#065A82"],
        )
        fig.update_layout(template=TEMPLATE)
        _save(fig, "01_file_size_distribution")

    # 2. Extracted text length per document
    if len(doc_df) > 0:
        fig = px.histogram(
            doc_df, x="chars", nbins=60,
            title="Extracted Text Length per Document",
            labels={"chars": "Characters Extracted", "count": "Documents"},
            color_discrete_sequence=["#1C7293"],
        )
        fig.update_layout(template=TEMPLATE)
        _save(fig, "02_content_length_distribution")

    # 3. Page type distribution
    if len(doc_df) > 0:
        tc = doc_df["page_type"].value_counts().reset_index()
        tc.columns = ["page_type", "count"]
        fig = px.bar(
            tc, x="page_type", y="count",
            title="Document Page Type Distribution",
            labels={"page_type": "Page Type", "count": "Count"},
            color="page_type",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(template=TEMPLATE, showlegend=False)
        _save(fig, "03_page_type_distribution", height=420)

    # 4. Product distribution (top 15)
    if len(doc_df) > 0:
        pc = doc_df["product"].value_counts().head(15).reset_index()
        pc.columns = ["product", "count"]
        fig = px.bar(
            pc, x="count", y="product", orientation="h",
            title="Top 15 Products by Document Count",
            labels={"product": "Product", "count": "Documents"},
            color_discrete_sequence=["#02C39A"],
        )
        fig.update_layout(
            template=TEMPLATE,
            yaxis={"categoryorder": "total ascending"},
        )
        _save(fig, "04_product_distribution", height=500)

    # 5. Chunk length distribution
    if chunk_lengths:
        chunk_size   = analysis["summary"].get("chunk_size_setting", 1200)
        fig = px.histogram(
            x=chunk_lengths, nbins=50,
            title=(f"Chunk Length Distribution  "
                   f"(chunk_size={chunk_size}, "
                   f"n={len(chunk_lengths):,})"),
            labels={"x": "Chunk Length (chars)", "count": "Chunks"},
            color_discrete_sequence=["#065A82"],
        )
        fig.add_vline(
            x=chunk_size, line_dash="dash", line_color="red",
            annotation_text=f"chunk_size={chunk_size}",
            annotation_position="top right",
        )
        fig.update_layout(template=TEMPLATE)
        _save(fig, "05_chunk_length_distribution")

    # 6. Combined dashboard (2×2)
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "File Size Distribution",
            "Extracted Text per Document",
            "Page Types",
            "Chunk Lengths",
        ),
    )
    if len(file_df) > 0:
        fig.add_trace(
            go.Histogram(x=file_df["size_kb"], nbinsx=30,
                         marker_color="#065A82", name="Files"),
            row=1, col=1,
        )
    if len(doc_df) > 0:
        fig.add_trace(
            go.Histogram(x=doc_df["chars"], nbinsx=60,
                         marker_color="#1C7293", name="Docs"),
            row=1, col=2,
        )
        tc2 = doc_df["page_type"].value_counts()
        fig.add_trace(
            go.Bar(x=tc2.index, y=tc2.values,
                   marker_color="#02C39A", name="Types"),
            row=2, col=1,
        )
    if chunk_lengths:
        fig.add_trace(
            go.Histogram(x=chunk_lengths, nbinsx=40,
                         marker_color="#065A82", name="Chunks"),
            row=2, col=2,
        )
    fig.update_layout(
        title_text="Corpus Analysis Dashboard",
        template=TEMPLATE,
        showlegend=False,
    )
    _save(fig, "06_dashboard", height=700)

    # Save summary JSON
    summary_path = OUTPUT_DIR / "analysis_summary.json"
    summary_path.write_text(json.dumps(analysis["summary"], indent=2))
    logger.info(f"Saved: {summary_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Analyze the HTML corpus (read-only)")
    parser.add_argument(
        "--data-dir",
        default=settings.raw_data_dir,
        help="Path to HTML files (default: settings.raw_data_dir)",
    )
    args = parser.parse_args()

    logger.info(f"Output directory: {OUTPUT_DIR}")
    analysis = analyze_corpus(args.data_dir)
    generate_plots(analysis)
    logger.info("Analysis complete.")


if __name__ == "__main__":
    main()