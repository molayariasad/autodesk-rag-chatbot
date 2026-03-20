# scripts/run_trulens_eval.py
"""
Run TruLens RAG Triad evaluation for corpus and blended modes.

Experiment Tracking:
    --experiment controls app_name in TruLens (top-level grouping).
    --mode controls app_version ("corpus" vs "blended").

    One experiment run produces two TruLens app versions:
        app_name: "01_baseline"
          app_version: "corpus"   ← Task 1a (internal docs only)
          app_version: "blended"  ← Task 1b (docs + web search)

    --pipeline controls which ingestion stack is used:
        "_baseline"        → chunk_size=512, precision mode, no alias normalization
        "optimized"        → chunk_size=1200, recall mode, full normalization
        "sentence_window"  → sentence-level chunks + window expansion at retrieval

Usage:
    # Baseline (Experiment 01)
    python scripts/run_trulens_eval.py --mode both --experiment 01_baseline --pipeline baseline --force-ingest

    # Optimized (Experiment 02. 03, 04, 05)
    python scripts/run_trulens_eval.py --mode both --experiment *_optimized_html_parser --pipeline optimized

    # Sentence Window Retrieval (Experiment 06)
    python scripts/run_trulens_eval.py --mode both --experiment 06_sentence_window --pipeline sentence_window --force-ingest

    # Reset DB and re-run
    python scripts/run_trulens_eval.py --reset-db --mode both --experiment 07_sentence_window --pipeline sentence_window
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

from src.config import settings
from src.evaluation.db import get_session, DB_PATH
from src.evaluation.evaluator import run_trulens_evaluation, load_eval_questions
from src.ingestion.pipeline import run_ingestion_pipeline
from src.retrieval.hybrid_retriever import HybridRetriever
from src.generation.llm import LLMClient
from src.generation.rag_chain import RAGChain


def build_pipeline(
    force_ingest: bool = False,
    baseline:     bool = False,
    sentence_window: bool = False,
) -> RAGChain:
    """
    Build the RAG pipeline.

    baseline=True         → html_parser_baseline + chunker_baseline
                            (chunk_size=512, precision mode, no alias normalization)
                            ChromaDB collection: 'autodesk_docs_baseline'

    sentence_window=True  → optimized html_parser + SentenceWindowChunker
                            (sentence-level chunks, window expansion at retrieval)
                            ChromaDB collection: 'autodesk_docs_sentence_window'
                            Retriever: SentenceWindowRetriever (wraps HybridRetriever)

    baseline=False,
    sentence_window=False → optimized html_parser + standard chunker
                            (chunk_size=1200, recall mode, full alias normalization)
                            ChromaDB collection: 'autodesk_docs'
    """
    if sentence_window:
        from src.ingestion.pipeline_sentence_window import run_sentence_window_pipeline
        from src.retrieval.sentence_window_retriever import SentenceWindowRetriever

        vector_store, bm25_index, _ = run_sentence_window_pipeline(
            force=force_ingest,
            window_size=2,
        )
        retriever = SentenceWindowRetriever(
            vector_store=vector_store,
            bm25_index=bm25_index,
            window_size=2,          # override; also stored in chunk metadata
        )
        llm = LLMClient()
        return RAGChain(retriever, llm)

    elif baseline:
        from src.ingestion.pipeline_baseline import run_baseline_pipeline
        vector_store, bm25_index, _ = run_baseline_pipeline(force=force_ingest)
    else:
        vector_store, bm25_index, _ = run_ingestion_pipeline(force=force_ingest)

    retriever = HybridRetriever(vector_store, bm25_index)
    llm       = LLMClient()
    return RAGChain(retriever, llm)


def compare_results(corpus: dict, blended: dict) -> None:
    """Terminal diff table — mirrors the TruLens Compare tab."""
    print(f"\n{'='*72}")
    print(f"{'EXPERIMENT: ' + corpus['experiment_name']:^72}")
    print(f"{'CORPUS (Task 1a)  vs  BLENDED (Task 1b)':^72}")
    print(f"{'='*72}")
    print(f"{'Metric':<30} {'Corpus':>12} {'Blended':>12} {'Delta':>12}")
    print(f"{'-'*72}")

    cs = corpus["aggregate_scores"]
    bs = blended["aggregate_scores"]

    for label, key in [
        ("Context Relevance",   "context_relevance"),
        ("Groundedness",        "groundedness"),
        ("Answer Relevance",    "answer_relevance"),
        ("RAG Triad Composite", "rag_triad_composite"),
    ]:
        c, b = cs.get(key, 0.0), bs.get(key, 0.0)
        d    = b - c
        print(f"{label:<30} {c:>12.3f} {b:>12.3f} {'+' if d >= 0 else ''}{d:>11.3f}")

    print(f"{'-'*72}")
    ld = blended["avg_latency_ms"] - corpus["avg_latency_ms"]
    print(
        f"{'Avg Latency (ms)':<30} {corpus['avg_latency_ms']:>12.0f} "
        f"{blended['avg_latency_ms']:>12.0f} "
        f"{'+' if ld >= 0 else ''}{ld:>11.0f}"
    )
    print(f"{'='*72}\n")

    if bs["groundedness"] < cs["groundedness"] - 0.05:
        print("  ⚠  Groundedness dropped — web results introduced ungrounded claims.")
        print("     Mitigation: tighten blended prompt or increase corpus RRF weight.")
    if bs["context_relevance"] > cs["context_relevance"] + 0.05:
        print("  ✓  Context Relevance improved — web search adding relevant material.")
    if bs["answer_relevance"] > cs["answer_relevance"] + 0.05:
        print("  ✓  Answer Relevance improved — recency/out-of-corpus questions helped.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run TruLens RAG Triad evaluation"
    )
    parser.add_argument(
        "--pipeline",
        choices=["optimized", "baseline", "sentence_window"],
        default="optimized",
        help=(
            "'baseline':        chunk_size=512, overlap=64, trafilatura precision mode, "
            "                   no product alias normalization (Experiment 01). "
            "'optimized':       chunk_size=1200, overlap=200, recall mode, "
            "                   full normalization (Experiment 02+). "
            "'sentence_window': sentence-level chunks + window expansion at retrieval "
            "                   time (Experiment 03+)."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["corpus", "blended", "both"],
        default="both",
    )
    parser.add_argument(
        "--experiment",
        default="01_baseline",
        help=(
            "Experiment phase label → app_name in TruLens. "
            "Use numbered prefixes to keep Leaderboard sorted: "
            "'01_baseline', '02_optimized_html_parser', "
            "'03_sentence_window'."
        ),
    )
    parser.add_argument(
        "--force-ingest",
        action="store_true",
        help=(
            "Force full re-ingestion (parse + chunk + embed). "
            "Required when running a pipeline for the first time or after "
            "changing corpus/chunking logic. Skipped automatically if the "
            "ChromaDB collection already has documents."
        ),
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=2,
        help=(
            "Sentence Window only: number of sentences on each side to include "
            "when expanding a retrieved sentence chunk. Default=2 (5-sentence window). "
            "Increase to 3 for longer context; decrease to 1 for more precision."
        ),
    )
    parser.add_argument(
        "--reset-db",
        action="store_true",
        help="Wipe ALL TruLens evaluation records before running.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(settings.processed_data_dir) / "experiment_results"),
        help="Directory for JSON exports (read by Streamlit App 1).",
    )
    parser.add_argument(
        "--questions-file",
        default=None,
        help=(
            "Path to a custom eval questions JSON file. "
            "Defaults to data/eval/eval_questions.json."
        ),
    )
    args = parser.parse_args()

    logger.info(f"TruLens DB:   {DB_PATH}")
    logger.info(f"Experiment:   {args.experiment}  |  Mode: {args.mode}")
    logger.info(f"Pipeline:     {args.pipeline}  |  Force ingest: {args.force_ingest}")
    if args.pipeline == "sentence_window":
        logger.info(f"Window size:  {args.window_size} sentences on each side")

    if args.reset_db:
        logger.warning("--reset-db: wiping all TruLens records. Cannot be undone.")
        get_session(reset=True)

    questions = load_eval_questions(path=args.questions_file)
    rag_chain = build_pipeline(
        force_ingest    = args.force_ingest,
        baseline        = (args.pipeline == "baseline"),
        sentence_window = (args.pipeline == "sentence_window"),
    )

    corpus_result = blended_result = None

    if args.mode in ("corpus", "both"):
        logger.info("── Task 1a: Corpus-only evaluation ──")
        corpus_result = run_trulens_evaluation(
            rag_chain,
            mode="corpus",
            questions=questions,
            output_dir=args.output_dir,
            experiment_name=args.experiment,
        )

    if args.mode in ("blended", "both"):
        logger.info("── Task 1b: Blended (corpus + web) evaluation ──")
        blended_result = run_trulens_evaluation(
            rag_chain,
            mode="blended",
            questions=questions,
            output_dir=args.output_dir,
            experiment_name=args.experiment,
        )

    if corpus_result and blended_result:
        compare_results(corpus_result, blended_result)


if __name__ == "__main__":
    main()
