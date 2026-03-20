# src/evaluation/evaluator.py
"""
Evaluation Framework for the Autodesk RAG Chatbot (TruLens RAG Triad).

RAG Triad — three evaluation dimensions:
─────────────────────────────────────────────────────────────────────────────
  1. Context Relevance  — Are the retrieved chunks relevant to the query?
  2. Groundedness       — Is every claim in the answer supported by context?
                          Primary hallucination detection signal.
  3. Answer Relevance   — Does the generated answer address what was asked?

Composite = geometric mean of all three. A single weak leg collapses it.

Key architectural decisions:
─────────────────────────────────────────────────────────────────────────────
1. Test suite decoupled from code (data/eval/eval_questions.json).

2. Post-hoc scoring: all generation completes first, then judge LLM scores
   in a second pass — decouples generation latency from eval latency.

3. The Context Relevance scores based on chain-of-thought reasons as well.
"""
from pathlib import Path
import json
import time
from pathlib import Path
from typing import Literal

from loguru import logger
from trulens.apps.basic import TruBasicApp
from trulens.providers.litellm import LiteLLM

from src.config import settings
from src.evaluation.db import get_session
from src.generation.rag_chain import RAGChain


# ============================================================
# Test suite loader
# ============================================================

def load_eval_questions(path: str | Path | None = None) -> list[dict]:
    """
    Load the evaluation question suite from the JSON data store.

    Default path: data/eval/eval_questions.json (relative to project root).
    Non-engineers can add/modify questions without touching Python.
    Different evaluation profiles can be maintained as separate JSON files.
    """
    if path is None:
        project_root = Path(__file__).resolve().parent.parent.parent
        path = project_root / "data" / "eval" / "eval_questions.json"

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Evaluation question file not found: {path}\n"
            f"Create it at data/eval/eval_questions.json or pass a custom path."
        )

    questions = json.loads(path.read_text(encoding="utf-8"))
    logger.info(f"Loaded {len(questions)} eval questions from {path}")
    return questions


# ============================================================
# Provider
# ============================================================

def _build_provider() -> LiteLLM:
    """
    LiteLLM wrapping local Ollama. No external API key needed.

    Production: replace with a stronger separate judge to eliminate
    same-model generator/judge bias:
        LiteLLM(model_engine="gpt-4o")
        LiteLLM(model_engine="claude-3-5-sonnet-20241022")
    """
    return LiteLLM(model_engine=f"ollama/{settings.llm_model}")


# ============================================================
# FeedbackDefinition registration
# ============================================================

def _register_feedback_definitions(session) -> None:
    """
    Write FeedbackDefinition rows directly to the TruLens DB.

    TruLens 2.7.x with OTEL enabled rejects Feedback objects passed to
    TruBasicApp with "ValueError: missing selectors". Writing definitions
    directly to the DB registers the Leaderboard columns without triggering
    OTEL selector validation.
    """
    try:
        from trulens.core.schema.feedback import FeedbackDefinition

        for name in ("Context Relevance", "Groundedness", "Answer Relevance"):
            try:
                defn = FeedbackDefinition(
                    feedback_definition_id=name,
                    name=name,
                    implementation=None,
                    aggregator=None,
                )
                session.add_feedback_definition(defn)
                logger.debug(f"Registered FeedbackDefinition: {name}")
            except Exception:
                pass  # already exists from a previous run

    except Exception as e:
        logger.warning(
            f"FeedbackDefinition registration failed: {e}. "
            "Leaderboard columns may not appear, but scores are in JSON."
        )


# ============================================================
# Score unpacking + CoT extraction
# ============================================================

def _unpack_score(raw) -> tuple[float, str | None]:
    """
    Unpack a provider return value into (score, cot_reason).

      float            → (float, None)
      (float, str)     → (float, str)   — CoT variants always return this
      {"score": float} → (float, reason or None)
    """
    if isinstance(raw, tuple):
        return float(raw[0]), (str(raw[1]) if len(raw) > 1 else None)
    if isinstance(raw, dict):
        score  = float(raw.get("score", raw.get("value", 0.0)))
        reason = raw.get("reason") or raw.get("reasons")
        return score, str(reason) if reason else None
    return float(raw), None


def _cot_method(provider: LiteLLM, base_name: str):
    """Prefer _with_cot_reasons variant; fall back to standard."""
    for name in (f"{base_name}_with_cot_reasons", base_name):
        fn = getattr(provider, name, None)
        if fn is not None:
            logger.debug(f"Using scorer: provider.{name}")
            return fn
    raise AttributeError(f"No scorer found for '{base_name}' on provider.")


def _get_groundedness_fn(provider: LiteLLM):
    """Return the best available groundedness method on this provider."""
    for name in (
        "groundedness_measure_with_cot_reasons",
        "groundedness_measure",
        "groundedness",
    ):
        fn = getattr(provider, name, None)
        if fn is not None:
            logger.debug(f"Groundedness method: provider.{name}")
            return fn
    raise AttributeError("No groundedness method found on LiteLLM provider.")


# ============================================================
# Per-question scoring
# ============================================================

def _score_question(
    provider: LiteLLM,
    question: str,
    answer: str,
    context_chunks: list[str],
) -> dict:
    """
    Run all three RAG Triad scorers against (question, answer, context).

    context_chunks must contain ALL chunks the LLM saw — corpus + web.
    Passing corpus-only chunks to a blended answer gives incorrect scores.
    """
    out = {
        "context_relevance":        None,
        "groundedness":             None,
        "answer_relevance":         None,
        "context_relevance_reason": None,
        "groundedness_reason":      None,
        "answer_relevance_reason":  None,
    }

    # Context Relevance — scored per chunk, averaged
    try:
        cr_fn = _cot_method(provider, "context_relevance")
        chunk_scores, chunk_reasons = [], []
        for chunk in context_chunks:
            score, reason = _unpack_score(
                cr_fn(question=question, context=chunk)
            )
            chunk_scores.append(score)
            if reason:
                chunk_reasons.append(reason)
        if chunk_scores:
            out["context_relevance"] = round(
                sum(chunk_scores) / len(chunk_scores), 4
            )
            if chunk_reasons:
                out["context_relevance_reason"] = "\n---\n".join(chunk_reasons)
    except Exception as e:
        logger.warning(f"Context relevance scoring failed: {e}")

    # Groundedness — primary hallucination signal
    try:
        grd_fn = _get_groundedness_fn(provider)
        score, reason = _unpack_score(
            grd_fn(source="\n\n".join(context_chunks), statement=answer)
        )
        out["groundedness"]        = round(score, 4)
        out["groundedness_reason"] = reason
    except Exception as e:
        logger.warning(f"Groundedness scoring failed: {e}")

    # Answer Relevance
    try:
        ar_fn = _cot_method(provider, "relevance")
        score, reason = _unpack_score(ar_fn(prompt=question, response=answer))
        out["answer_relevance"]        = round(score, 4)
        out["answer_relevance_reason"] = reason
    except Exception as e:
        logger.warning(f"Answer relevance scoring failed: {e}")

    return out


# ============================================================
# RAG callable with response cache
# ============================================================

def _make_rag_callable(rag_chain: RAGChain, mode: str):
    """
    Return a closure that runs the full RAG pipeline and caches the response.

    Why the cache matters for blended mode:
        rag_chain.query(mode="blended") merges corpus chunks + web results
        into all_results before passing them to the LLM. The evaluator
        previously re-ran retriever.retrieve() to get context for scoring,
        which only returns corpus chunks — making the scorer blind to web
        results. With the cache, scoring uses the same context the LLM saw.

    """
    cache = {}

    def _run(question: str) -> str:
        response      = rag_chain.query(question=question, mode=mode, top_k=5)
        cache["last"] = response   # captured for context extraction below
        return response.answer

    _run.cache = cache   # expose cache so the eval loop can read it
    return _run


def _extract_context_chunks(rag_callable, fallback_retriever, question: str) -> list[str]:
    """
    Extract context chunks from the cached RAGResponse.

    RAGResponse.sources is the raw retrieval list passed to the LLM.
    Each entry is a dict with a top-level "text" key (confirmed from rag_chain.py).
    This contains both corpus and web chunks in blended mode.

    Falls back to corpus-only retrieval if the cache is empty, which can
    happen if tru_app.app() raised an exception before caching.
    """
    cached_response = rag_callable.cache.get("last")

    if cached_response is not None:
        sources = getattr(cached_response, "sources", []) or []
        chunks  = [s.get("text", "") for s in sources if s.get("text")]
        if chunks:
            logger.debug(
                f"Context from cache: {len(chunks)} chunks "
                f"(mode={cached_response.mode})"
            )
            return chunks

    # Fallback — corpus only
    logger.debug("Cache miss — falling back to corpus retrieval for scoring")
    results = fallback_retriever.retrieve(query=question, top_k=5)
    return [r.get("text", "") for r in results]


# ============================================================
# Main evaluation runner
# ============================================================

def run_trulens_evaluation(
    rag_chain: RAGChain,
    mode: Literal["corpus", "blended"] = "corpus",
    questions: list[dict] | None = None,
    output_dir: str | None = None,
    experiment_name: str = "01_baseline",
) -> dict:
    """
    Run RAG Triad evaluation and persist results to TruLens DB + JSON.

    TruLens dashboard hierarchy:
        app_name    = experiment_name  → Leaderboard grouping
        app_version = mode             → corpus vs blended (Task 1a vs 1b)
        Record Input  = question
        Record Output = LLM answer
        Feedback cols = Context Relevance, Groundedness, Answer Relevance
    """
    if questions is None:
        questions = load_eval_questions()

    output_dir = output_dir or str(
        Path(settings.processed_data_dir) / "experiment_results"
    )

    logger.info(
        f"RAG Triad — experiment='{experiment_name}' "
        f"version='{mode}'  questions={len(questions)}"
    )

    session  = get_session(reset=False)
    provider = _build_provider()

    # Register FeedbackDefinition rows so Leaderboard columns appear.
    # Done before TruBasicApp creation — OTEL mode rejects Feedback objects
    # with selectors at __init__ time so we never pass feedbacks=[...].
    _register_feedback_definitions(session)

    rag_callable = _make_rag_callable(rag_chain, mode)
    tru_app = TruBasicApp(
        text_to_text=rag_callable,
        app_name=experiment_name,
        app_version=mode,
        feedbacks=[],   # OTEL-safe: definitions registered directly above
    )

    results: list[dict] = []
    scored_records: list[tuple[str, dict]] = []

    with tru_app as recording:
        for i, q_data in enumerate(questions):
            question         = q_data["question"]
            category         = q_data.get("category", "unknown")
            expected_product = q_data.get("expected_product", "")

            logger.info(
                f"  [{i+1}/{len(questions)}] [{category}] {question[:65]}..."
            )

            start = time.time()
            try:
                # TruBasicApp intercepts this call:

                answer = tru_app.app(question)

                # Extract ALL context chunks the LLM actually saw.
                # In blended mode this includes web results — critical for
                # correct Context Relevance and Groundedness scores.
                context_chunks = _extract_context_chunks(
                    rag_callable=rag_callable,
                    fallback_retriever=rag_chain.retriever,
                    question=question,
                )

                latency_ms = (time.time() - start) * 1000
                error      = None

            except Exception as e:
                logger.error(f"Query failed: {e}")
                answer         = "ERROR"
                context_chunks = []
                latency_ms     = (time.time() - start) * 1000
                error          = str(e)

            # Capture record_id of the most recently committed record
            record_id = None
            try:
                record_id = recording.records[-1].record_id
            except Exception as e:
                logger.debug(f"Could not capture record_id: {e}")

            # RAG Triad scoring — uses full context (corpus + web)
            triad = (
                _score_question(provider, question, answer, context_chunks)
                if not error
                else {
                    "context_relevance": None, "groundedness": None,
                    "answer_relevance":  None,
                    "context_relevance_reason": None,
                    "groundedness_reason":      None,
                    "answer_relevance_reason":  None,
                }
            )

            if record_id and not error:
                scored_records.append((record_id, triad))

            result = {
                "question":         question,
                "category":         category,
                "expected_product": expected_product,
                "answer":           answer,
                "num_sources":      len(context_chunks),
                "latency_ms":       round(latency_ms, 1),
                "error":            error,
                "record_id":        record_id,
                **triad,
            }
            results.append(result)

            if not error:
                logger.info(
                    f"    ctx_rel={triad['context_relevance']}  "
                    f"grd={triad['groundedness']}  "
                    f"ans_rel={triad['answer_relevance']}  "
                    f"sources={len(context_chunks)}"
                )

    # Post-hoc feedback persistence
    _persist_feedback_scores(session, scored_records)

    # ── Aggregate ──────────────────────────────────────────────────────────
    valid = [r for r in results if not r.get("error")]

    def _mean(key: str) -> float:
        vals = [r[key] for r in valid if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else 0.0

    cr  = _mean("context_relevance")
    gr  = _mean("groundedness")
    ar  = _mean("answer_relevance")
    cmp = (cr * gr * ar) ** (1 / 3) if (cr * gr * ar) > 0 else 0.0

    # Per-category breakdown
    categories: dict[str, list] = {}
    for r in valid:
        categories.setdefault(r.get("category", "unknown"), []).append(r)

    category_summary: dict[str, dict] = {}
    for cat, cat_results in categories.items():
        def _cat_mean(key: str, _rs=cat_results) -> float:
            vals = [r[key] for r in _rs if r.get(key) is not None]
            return round(sum(vals) / len(vals), 4) if vals else 0.0
        category_summary[cat] = {
            "n":                 len(cat_results),
            "context_relevance": _cat_mean("context_relevance"),
            "groundedness":      _cat_mean("groundedness"),
            "answer_relevance":  _cat_mean("answer_relevance"),
        }

    summary = {
        "mode":            mode,
        "experiment_name": experiment_name,
        "app_version":     mode,
        "timestamp":       time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_questions":   len(questions),
        "num_errors":      sum(1 for r in results if r.get("error")),
        "aggregate_scores": {
            "context_relevance":   round(cr, 4),
            "groundedness":        round(gr, 4),
            "answer_relevance":    round(ar, 4),
            "rag_triad_composite": round(cmp, 4),
        },
        "category_scores":  category_summary,
        "avg_latency_ms":   round(_mean("latency_ms"), 1),
        "per_question":     results,
    }

    out_path = Path(output_dir) / f"trulens_{mode}_{int(time.time())}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    logger.info(f"JSON export: {out_path}")

    _log_summary(summary)
    return summary


def _persist_feedback_scores(
    session,
    scored_records: list[tuple[str, dict]],
) -> None:
    """
    Write pre-computed RAG Triad scores to the TruLens DB.

    Post-hoc pattern: scores computed after all generation completes,
    then batch-written. FeedbackResult.record_id links each score to
    the correct record row in the dashboard.
    """
    if not scored_records:
        return

    try:
        from trulens.core.schema.feedback import FeedbackResult

        metric_keys = [
            ("Context Relevance", "context_relevance", "context_relevance_reason"),
            ("Groundedness",      "groundedness",      "groundedness_reason"),
            ("Answer Relevance",  "answer_relevance",  "answer_relevance_reason"),
        ]

        feedback_results = []
        for record_id, triad in scored_records:
            for name, score_key, reason_key in metric_keys:
                score  = triad.get(score_key)
                reason = triad.get(reason_key)
                if score is None:
                    continue
                feedback_results.append(
                    FeedbackResult(
                        record_id=record_id,
                        feedback_definition_id=name,
                        name=name,
                        result=score,
                        reason=reason or "",
                        status="done",
                        multi_result=None,
                    )
                )

        if feedback_results:
            session.add_feedbacks(feedback_results)
            logger.info(
                f"Persisted {len(feedback_results)} feedback scores "
                f"across {len(scored_records)} records"
            )

    except Exception as e:
        logger.warning(
            f"Feedback persistence failed (scores safe in JSON): {e}"
        )


def _log_summary(summary: dict) -> None:
    scores = summary["aggregate_scores"]
    sep    = "=" * 65
    logger.info(f"\n{sep}")
    logger.info(
        f"RAG TRIAD — experiment='{summary['experiment_name']}' "
        f"version='{summary['mode']}'"
    )
    logger.info(sep)
    logger.info(f"Context Relevance:    {scores['context_relevance']:.3f}")
    logger.info(
        f"Groundedness:         {scores['groundedness']:.3f}"
        "  ← hallucination signal"
    )
    logger.info(f"Answer Relevance:     {scores['answer_relevance']:.3f}")
    logger.info(
        f"RAG Triad Composite:  {scores['rag_triad_composite']:.3f}"
        "  ← geometric mean (all three must be strong)"
    )
    logger.info(f"Avg Latency:          {summary['avg_latency_ms']:.0f}ms")
    logger.info(f"Errors:               {summary['num_errors']}/{summary['num_questions']}")

    if summary.get("category_scores"):
        logger.info(f"\n{'Category Breakdown':─<65}")
        logger.info(f"  {'Category':<28} {'N':>3}  {'CR':>6}  {'GRD':>6}  {'AR':>6}")
        for cat, s in sorted(summary["category_scores"].items()):
            logger.info(
                f"  {cat:<28} {s['n']:>3}  "
                f"{s['context_relevance']:>6.3f}  "
                f"{s['groundedness']:>6.3f}  "
                f"{s['answer_relevance']:>6.3f}"
            )
    logger.info(sep)