"""
Streamlit UI for the Autodesk RAG Chatbot (Customer-Facing UI at port 8501).

Features:
- Conversational chat with history
- Toggle between corpus-only and blended mode (with web search)
- Source inspection panel (transparency), easily viewed alongside the answer
- Evaluation dashboard with metrics visualization, for User transparency, viwed by interviewers and product stakeholders
    + Reads the two most recent corpus + blended JSON files from data/processed/trulens_*.json and renders them side-by-side.
- Drill-down link to admin dashboard for trace inspection (Admin dashboard runs on separate port 8502, so it never blocks or interferes with this Streamlit process)
- Human feedback collection (thumbs up/down)
"""

import json
import glob
from pathlib import Path

import httpx
import streamlit as st
import plotly.graph_objects as go
import pandas as pd

# ── Configuration ────────────────────────────────────────────────────────────
API_BASE      = "http://fastapi:8000"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
# Port where App 2 (admin dashboard) is served.
# Must match the port in scripts/run_admin_dashboard.py.
ADMIN_DASHBOARD_URL = "http://localhost:8502"


def get_api_url() -> str:
    for url in [API_BASE, "http://localhost:8000"]:
        try:
            if httpx.get(f"{url}/health", timeout=3).status_code == 200:
                return url
        except Exception:
            continue
    return "http://localhost:8000"


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Autodesk RAG Chatbot — Asad Molayari",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state ─────────────────────────────────────────────────────────────
if "messages"  not in st.session_state: st.session_state.messages  = []
if "feedback"  not in st.session_state: st.session_state.feedback  = {}
if "api_url"   not in st.session_state: st.session_state.api_url   = get_api_url()

API_URL = st.session_state.api_url


# ══════════════════════════════════════════════════════════════════════════════
# TruLens JSON helpers
# We read from the JSON files written by trulens_evaluator.py rather than
# importing TruLens directly. This keeps App 1 free of TruLens's OTEL
# initialisation overhead and avoids SQLite lock contention when App 2 is
# also open against the same database.
# ══════════════════════════════════════════════════════════════════════════════

def _load_latest(mode: str) -> dict | None:
    """Return the most recently written trulens_{mode}_*.json, or None."""
    matches = sorted(
        glob.glob(str(PROCESSED_DIR / "experiment_results" / f"trulens_{mode}_*.json")),
        reverse=True,
    )
    if not matches:
        return None
    try:
        return json.loads(Path(matches[0]).read_text())
    except Exception:
        return None


def _score(data: dict | None, key: str) -> float | None:
    if data is None:
        return None
    return data.get("aggregate_scores", {}).get(key)


def _fmt(val: float | None) -> str:
    return f"{val:.3f}" if val is not None else "—"


def _delta_str(b: float | None, c: float | None) -> str | None:
    """Delta label for st.metric: blended minus corpus, or None if either missing."""
    if b is None or c is None:
        return None
    d = b - c
    return f"{'+' if d >= 0 else ''}{d:.3f}"


# ══════════════════════════════════════════════════════════════════════════════
# Source rendering helper
# ══════════════════════════════════════════════════════════════════════════════

def _render_source(src: dict, api_url: str) -> None:
    title    = src.get("title", "Unknown")
    score    = src.get("score", 0.0)
    src_type = src.get("type", "")
    raw_url  = src.get("url", "")
    file     = src.get("file", "")
    icon     = "🌐" if src_type == "web_search" else "📄"

    if raw_url and raw_url.startswith("http"):
        link = f" | [View online]({raw_url})"
    elif raw_url and raw_url.startswith("/files/"):
        # Replace internal Docker hostname (fastapi:8000) with localhost:8000
        # so source document links open correctly in the host browser.
        browser_url = api_url.replace("http://fastapi:8000", "http://localhost:8000")
        link = f" | [View source document]({browser_url}{raw_url})"
    elif file:
        link = f" | `{file}`"
    else:
        link = ""

    st.markdown(
        f"**{icon} {title}** (score: {score:.3f})\n\nType: `{src_type}`{link}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("🏗️ Autodesk RAG Chatbot")
    st.caption("Built by Asad Molayari · Powered by Open-Source ML")
    st.divider()

    mode  = st.radio(
        "Retrieval Mode", ["corpus", "blended"], index=0,
        help="**Corpus**: internal docs only.\n\n**Blended**: docs + live web search.",
    )
    top_k = st.slider(
        "Context chunks (top-k)", 1, 15, 5,
        help=(
            "Number of retrieved document chunks passed to the LLM as context.\n\n"
            "**Higher** → more context, better coverage of complex questions, "
            "slightly slower and higher hallucination risk if noisy chunks are included.\n\n"
            "**Lower** → tighter context, faster responses, better for simple factual questions.\n\n"
            "Default of 5 is the evaluated sweet spot for this corpus."
        ),
    )
    st.divider()

    if st.button("🔍 Check System Health"):
        try:
            h = httpx.get(f"{API_URL}/health", timeout=5).json()
            if h.get("ready"):
                st.success(
                    f"✅ Healthy\n"
                    f"- LLM: {'🟢' if h['llm_available'] else '🔴'}\n"
                    f"- Chunks: {h['vector_store_count']}"
                )
            else:
                st.warning("⏳ Initializing…")
        except Exception as e:
            st.error(f"❌ {e}")

    st.divider()

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.feedback = {}
        st.rerun()

    st.subheader("📝 Sample Questions")
    for idx, q in enumerate([
        "What does Fusion 360 do?",
        "What's the difference between AutoCAD and Revit?",
        "Does AutoCAD LT do 3D?",
        "What's the latest release for Maya?",
        "Can I use Fusion 360 on a Mac?",
        "What is Autodesk Construction Cloud?",
        "What new features were added to Forma?",
        "What's the difference between f360 and Solidworks?",
        "What are the newest AI features added to Autodesk products in 2025?",
        "What is MotionBuilder used for?",
       ]):
        if st.button(q, key=f"sq_{idx}"):
            st.session_state.pending_question = q
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# Tabs
# App 1 exposes exactly two tabs to the customer/interviewer:
#   1. Chat          — the product experience
#   2. Eval Summary  — high-level quality signal (RAG Triad only)
# Detailed tracing lives exclusively in App 2 (Admin dashboard).
# ══════════════════════════════════════════════════════════════════════════════
tab_chat, tab_eval = st.tabs(["💬 Chat", "📊 Evaluation Summary — Current System Performance"])


# ══════════════════════════════════════════════════════════════════════════════
# Tab 1 — Chat
# ══════════════════════════════════════════════════════════════════════════════
with tab_chat:
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            if msg["role"] == "assistant":
                lat      = msg.get("latency_ms", 0)
                msg_mode = msg.get("mode", "")
                st.caption(f"⏱️ {lat:.0f}ms | Mode: {msg_mode}")

                sources = msg.get("sources", [])
                if sources:
                    with st.expander(f"📄 {len(sources)} source(s)", expanded=False):
                        for j, src in enumerate(sources):
                            _render_source(src, API_URL)
                            if j < len(sources) - 1:
                                st.divider()

                c1, c2, _ = st.columns([1, 1, 10])
                mid = f"msg_{i}"
                with c1:
                    if st.button("👍", key=f"up_{mid}"):
                        st.session_state.feedback[mid] = "positive"
                        st.toast("Thanks!")
                with c2:
                    if st.button("👎", key=f"dn_{mid}"):
                        st.session_state.feedback[mid] = "negative"
                        st.toast("Noted — we'll improve.")

    prompt = st.chat_input("Ask about Autodesk products…")
    if "pending_question" in st.session_state:
        prompt = st.session_state.pending_question
        del st.session_state.pending_question

    if prompt:
        st.session_state.messages.append({
            "role": "user", "content": prompt,
            "sources": [], "latency_ms": 0, "mode": "",
        })
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    history = [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages[:-1]
                    ]
                    r = httpx.post(
                        f"{API_URL}/chat",
                        json={"question": prompt, "mode": mode,
                              "chat_history": history, "top_k": top_k},
                        timeout=120,
                    )
                    r.raise_for_status()
                    data = r.json()

                    answer  = data["answer"]
                    srcs    = data.get("sources", [])
                    lat     = data.get("latency_ms", 0)
                    r_mode  = data.get("mode", mode)

                    st.markdown(answer)
                    st.caption(f"⏱️ {lat:.0f}ms | Mode: {r_mode}")

                    if srcs:
                        with st.expander(f"📄 {len(srcs)} source(s)", expanded=False):
                            for j, src in enumerate(srcs):
                                _render_source(src, API_URL)
                                if j < len(srcs) - 1:
                                    st.divider()

                    st.session_state.messages.append({
                        "role": "assistant", "content": answer,
                        "sources": srcs, "latency_ms": lat, "mode": r_mode,
                    })
                except httpx.ConnectError:
                    st.error("Cannot reach the API. Is FastAPI running?")
                except Exception as e:
                    st.error(f"Error: {e}")
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# Tab 2 — Evaluation Summary
#
# Design intent: show only the signal an interviewer or product stakeholder
# needs at a glance. No raw traces, no per-claim breakdowns — those live in
# App 2. This tab answers one question: "Is the system trustworthy, and does
# adding web search make it better or worse?"
# ══════════════════════════════════════════════════════════════════════════════
with tab_eval:
    st.header("📊 Evaluation Summary — RAG Triad")
    st.caption(
        "Scores are produced by RAG Triad approach using an LLM-as-judge methodology "
        "(Context Relevance · Groundedness · Answer Relevance). "
        "Run `python scripts/run_trulens_eval.py --mode both` to refresh results."
    )

    # ── Reload & Admin launch bar ─────────────────────────────────────────
    col_reload, col_admin = st.columns([2, 3])
    with col_reload:
        if st.button("🔄 Reload latest results"):
            for k in ("_tl_corpus", "_tl_blended"):
                st.session_state.pop(k, None)
            st.rerun()
    with col_admin:
        # The admin dashboard (App 2) runs on a separate port so it never
        # blocks or interferes with this Streamlit process.
        st.link_button(
            "🛠️ Open Advanced Admin Dashboard for tracing and drill-down analysis (TruLens)",
            url=ADMIN_DASHBOARD_URL,
            help=(
                "Opens the native TruLens dashboard for per-query traces, "
                "per-claim groundedness breakdowns, and historical run comparisons. "
                f"Served at {ADMIN_DASHBOARD_URL} — start it with: "
                "python scripts/run_admin_dashboard.py"
            ),
        )

    st.divider()

    # ── Load results (cached per session, invalidated by Reload button) ───
    if "_tl_corpus"  not in st.session_state:
        st.session_state["_tl_corpus"]  = _load_latest("corpus")
    if "_tl_blended" not in st.session_state:
        st.session_state["_tl_blended"] = _load_latest("blended")

    corpus_data  = st.session_state["_tl_corpus"]
    blended_data = st.session_state["_tl_blended"]

    if corpus_data is None and blended_data is None:
        st.info(
            "No evaluation results found yet.\n\n"
            "```bash\npython scripts/run_trulens_eval.py --mode both\n```"
        )
        st.stop()

    # ── Metric definitions with tooltips ─────────────────────────────────
    METRICS = [
        (
            "Context Relevance",
            "context_relevance",
            "Are retrieved chunks relevant to the query? "
            "Low = retrieval failure upstream.",
        ),
        (
            "Groundedness",
            "groundedness",
            "Fraction of answer claims entailed by retrieved context. "
            "This is the primary hallucination signal. "
            "Score < 0.5 = model invented content not in the corpus.",
        ),
        (
            "Answer Relevance",
            "answer_relevance",
            "Does the answer address the question that was asked? "
            "Catches grounded-but-off-topic responses.",
        ),
        (
            "RAG Triad Composite",
            "rag_triad_composite",
            "Geometric mean of all three scores. "
            "A single weak leg collapses this — the system must pass all three.",
        ),
    ]

    # ── Side-by-side st.metric cards ──────────────────────────────────────
    # Layout: label column | corpus value | blended value (with delta)
    st.subheader("Task 1a vs. Task 1b — Side-by-Side")
    st.caption(
        "**Task 1a** = internal corpus only &nbsp;|&nbsp; "
        "**Task 1b** = corpus + live web search (blended). "
        "Delta = Blended − Corpus. Green = improvement, red = degradation."
    )

    header_c, header_a, header_b = st.columns([2, 1, 1])
    header_c.markdown("**Metric**")
    header_a.markdown("**Task 1a · Corpus**")
    header_b.markdown("**Task 1b · Blended**")

    st.divider()

    for label, key, tooltip in METRICS:
        c_val = _score(corpus_data,  key)
        b_val = _score(blended_data, key)
        delta = _delta_str(b_val, c_val)

        col_label, col_corpus, col_blended = st.columns([2, 1, 1])
        col_label.markdown(f"**{label}**", help=tooltip)

        if corpus_data is not None:
            col_corpus.metric(
                label="",
                value=_fmt(c_val),
                label_visibility="collapsed",
            )
        else:
            col_corpus.markdown("*not run*")

        if blended_data is not None:
            col_blended.metric(
                label="",
                value=_fmt(b_val),
                delta=delta,
                delta_color="normal",
                label_visibility="collapsed",
            )
        else:
            col_blended.markdown("*not run*")

    # ── Run metadata ──────────────────────────────────────────────────────
    st.divider()
    meta_c, meta_b = st.columns(2)
    if corpus_data:
        meta_c.caption(
            f"Corpus run: {corpus_data.get('timestamp', '—')} · "
            f"{corpus_data.get('num_questions', 0)} questions · "
            f"{corpus_data.get('num_errors', 0)} errors"
        )
    if blended_data:
        meta_b.caption(
            f"Blended run: {blended_data.get('timestamp', '—')} · "
            f"{blended_data.get('num_questions', 0)} questions · "
            f"{blended_data.get('num_errors', 0)} errors"
        )

    # ── Radar comparison ──────────────────────────────────────────────────
    # Only render if both modes have been evaluated — a single-mode radar
    # conveys nothing useful about the assignment's comparison requirement.
    if corpus_data and blended_data:
        st.divider()
        st.subheader("Radar Comparison")

        radar_dims = [m[0] for m in METRICS]
        radar_keys = [m[1] for m in METRICS]
        c_vals     = [(_score(corpus_data,  k) or 0.0) for k in radar_keys]
        b_vals     = [(_score(blended_data, k) or 0.0) for k in radar_keys]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=c_vals + [c_vals[0]],
            theta=radar_dims + [radar_dims[0]],
            fill="toself", name="Task 1a · Corpus", opacity=0.75,
        ))
        fig.add_trace(go.Scatterpolar(
            r=b_vals + [b_vals[0]],
            theta=radar_dims + [radar_dims[0]],
            fill="toself", name="Task 1b · Blended", opacity=0.75,
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            legend=dict(orientation="h", y=-0.18),
            height=440,
            margin=dict(t=30, b=70),
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Automated interpretation callouts ─────────────────────────────
        grd_delta = (_score(blended_data, "groundedness") or 0) - (_score(corpus_data, "groundedness") or 0)
        cr_delta  = (_score(blended_data, "context_relevance") or 0) - (_score(corpus_data, "context_relevance") or 0)
        ar_delta  = (_score(blended_data, "answer_relevance") or 0) - (_score(corpus_data, "answer_relevance") or 0)

        if grd_delta < -0.05:
            st.warning(
                "⚠️ **Groundedness dropped in blended mode.** "
                "Web snippets introduced claims the corpus cannot verify. "
                "Mitigation: tighten the blended system prompt, or increase "
                "corpus weight in RRF fusion."
            )
        elif grd_delta > 0.02:
            st.success(
                "✅ **Groundedness held or improved in blended mode.** "
                "Web results are contributing grounded, verifiable content."
            )

        if cr_delta > 0.05:
            st.success(
                "✅ **Context Relevance improved in blended mode.** "
                "Web search is surfacing relevant material absent from the static corpus."
            )

        if ar_delta > 0.05:
            st.success(
                "✅ **Answer Relevance improved in blended mode.** "
                "Likely driven by recency probes or out-of-corpus questions."
            )