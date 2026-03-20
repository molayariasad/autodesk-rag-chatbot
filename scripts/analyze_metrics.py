# scripts/analyze_metrics.py
"""
Experiment metrics analysis for the Autodesk RAG Chatbot.

Reads all trulens_*.json files from data/processed/ and produces:
  1. Experiment progression line chart (composite score over iterations)
  2. RAG Triad radar chart (best corpus vs best blended)
  3. Per-category CR heatmap (retrieval weakness analysis)
  4. Corpus vs Blended side-by-side bar chart
  5. Latency vs quality scatter plot
  6. Category CR vs GRD gap analysis
  7. Three-metric grouped bars (corpus only)
  8. Summary CSV

All plots saved as PNG to data/processed/metrics_analysis/
PNG requires kaleido: uv add kaleido

Usage:
    uv run python scripts/analyze_metrics.py
"""

import json
import glob
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from loguru import logger

# ── Config ────────────────────────────────────────────────────────────────────
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "experiment_results"
OUTPUT_DIR    = PROJECT_ROOT / "data" / "processed" / "metrics_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# All experiments in chronological order
EXPERIMENT_ORDER = [
    "01_baseline",
    "02_optimized_html_parser",
    "03_optimized_MIN_TEXT_LENGTH",
    "04_retrieval_weights",
    "05_quality_filter",
    "06_sentence_window",
    "07_prompt_engineering",
    "08_quality_filter_MinTxt500_RetrWeights",
    "09_sw_min_sent_80",
    "10_quality_filter_MinTxt500_RetrWeights",
]

EXPERIMENT_LABELS = {
    "01_baseline":                           "01 Baseline\n(chunk=512, precision)",
    "02_optimized_html_parser":              "02 Optimized\nParser",
    "03_optimized_MIN_TEXT_LENGTH":          "03 Min Text\nFilter=400",
    "04_retrieval_weights":                  "04 Retrieval\nWeights",
    "05_quality_filter":                     "05 Quality\nFilter",
    "06_sentence_window":                    "06 Sentence\nWindow",
    "07_prompt_engineering":                 "07 Prompt\nEngineering",
    "08_quality_filter_MinTxt500_RetrWeights": "08 Quality\nFilter + Retrieval\nWeights",
    "09_sw_min_sent_80":                     "09 SW Min\nSent=80",
    "10_quality_filter_MinTxt500_RetrWeights": "10 Combined\nFilters",
}

METRICS = ["context_relevance", "groundedness", "answer_relevance", "rag_triad_composite"]
METRIC_LABELS = {
    "context_relevance":   "Context Relevance",
    "groundedness":        "Groundedness",
    "answer_relevance":    "Answer Relevance",
    "rag_triad_composite": "RAG Triad Composite",
}

COLORS = {
    "corpus":    "#1f77b4",   # blue
    "blended":   "#e55c00",   # dark orange (was #ff7f0e — too light on white)
    "cr":        "#d62728",   # red
    "grd":       "#2ca02c",   # green
    "ar":        "#7b3fa0",   # purple (was #9467bd — slightly deeper for white bg)
    "composite": "#5c3317",   # dark brown (was #8c564b)
}

TEMPLATE = "plotly_white"
PNG_SCALE = 2      # retina-quality PNG (2× pixel density)
PNG_WIDTH = 1400
PNG_HEIGHT = 700


def _save(fig: go.Figure, stem: str, height: int = PNG_HEIGHT) -> None:
    """Save figure as PNG only (git-friendly, no 92MB HTML files)."""
    out = OUTPUT_DIR / f"{stem}.png"
    fig.update_layout(height=height)
    fig.write_image(str(out), scale=PNG_SCALE, width=PNG_WIDTH, height=height)
    logger.info(f"Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

def load_all_results() -> pd.DataFrame:
    """Load all trulens_*.json files into a tidy DataFrame."""
    files = sorted(glob.glob(str(PROCESSED_DIR / "trulens_*.json")))
    if not files:
        logger.error(f"No trulens_*.json files found in {PROCESSED_DIR}")
        sys.exit(1)

    rows = []
    for fp in files:
        try:
            data = json.loads(Path(fp).read_text())
            exp  = data.get("experiment_name", "unknown")
            mode = data.get("mode", "unknown")
            agg  = data.get("aggregate_scores", {})
            cats = data.get("category_scores", {})

            row = {
                "experiment":          exp,
                "mode":                mode,
                "timestamp":           data.get("timestamp", ""),
                "num_questions":       data.get("num_questions", 0),
                "num_errors":          data.get("num_errors", 0),
                "avg_latency_ms":      data.get("avg_latency_ms", 0),
                "context_relevance":   agg.get("context_relevance"),
                "groundedness":        agg.get("groundedness"),
                "answer_relevance":    agg.get("answer_relevance"),
                "rag_triad_composite": agg.get("rag_triad_composite"),
                "source_file":         Path(fp).name,
            }
            for cat, cs in cats.items():
                row[f"cat_{cat}_cr"]  = cs.get("context_relevance")
                row[f"cat_{cat}_grd"] = cs.get("groundedness")
                row[f"cat_{cat}_ar"]  = cs.get("answer_relevance")

            rows.append(row)
        except Exception as e:
            logger.warning(f"Could not parse {fp}: {e}")

    df = pd.DataFrame(rows)
    order_map = {e: i for i, e in enumerate(EXPERIMENT_ORDER)}
    df["exp_order"] = df["experiment"].map(order_map).fillna(99)
    df = df.sort_values(["exp_order", "mode"]).reset_index(drop=True)
    logger.info(f"Loaded {len(df)} runs from {len(files)} files")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Plot 1: Experiment progression
# ══════════════════════════════════════════════════════════════════════════════

def plot_experiment_progression(df: pd.DataFrame) -> None:
    """Line chart — all four metrics across experiments, split by mode."""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Corpus Mode", "Blended Mode"),
        vertical_spacing=0.16,
    )

    metric_styles = {
        "context_relevance":   (COLORS["cr"],        "dot",   1.8),
        "groundedness":        (COLORS["grd"],        "dot",   1.8),
        "answer_relevance":    (COLORS["ar"],         "dot",   1.8),
        "rag_triad_composite": (COLORS["composite"],  "solid", 3.0),
    }

    for row_idx, mode in enumerate(["corpus", "blended"], start=1):
        sub = df[df["mode"] == mode].copy()
        # Keep only known experiments, in order
        sub = sub[sub["experiment"].isin(EXPERIMENT_ORDER)]
        sub = sub.set_index("experiment").reindex(EXPERIMENT_ORDER).reset_index()
        sub = sub.dropna(subset=["rag_triad_composite"])

        x_labels = [
            EXPERIMENT_LABELS.get(e, e).replace("\n", " ")
            for e in sub["experiment"]
        ]

        for metric, (color, dash, width) in metric_styles.items():
            fig.add_trace(
                go.Scatter(
                    x=x_labels, y=sub[metric],
                    mode="lines+markers",
                    name=METRIC_LABELS[metric],
                    line=dict(color=color, dash=dash, width=width),
                    marker=dict(size=7),
                    showlegend=(row_idx == 1),
                ),
                row=row_idx, col=1,
            )

    fig.update_yaxes(range=[0.3, 1.05], tickformat=".2f")
    fig.update_layout(
        template=TEMPLATE,
        title=dict(text="RAG Triad Metric Progression Across Experiments", font=dict(size=16)),
        legend=dict(orientation="h", y=-0.12),
        margin=dict(t=80, b=100, l=60, r=20),
    )
    fig.add_annotation(
        text="⚠ Context Relevance (red dashed) is the persistent weak leg",
        xref="paper", yref="paper", x=0.5, y=1.04,
        showarrow=False, font=dict(color=COLORS["cr"], size=11),
    )
    _save(fig, "01_experiment_progression", height=760)


# ══════════════════════════════════════════════════════════════════════════════
# Plot 2: Radar — best corpus vs best blended
# ══════════════════════════════════════════════════════════════════════════════

def plot_radar_best(df: pd.DataFrame) -> None:
    """Radar comparing the best-composite corpus and blended runs."""
    best_c = (df[df["mode"] == "corpus"]
              .sort_values("rag_triad_composite", ascending=False).iloc[0])
    best_b_df = df[df["mode"] == "blended"]
    if best_b_df.empty:
        logger.warning("No blended runs found — skipping radar plot")
        return
    best_b = best_b_df.sort_values("rag_triad_composite", ascending=False).iloc[0]

    dims = ["Context\nRelevance", "Groundedness", "Answer\nRelevance", "RAG Triad\nComposite"]
    keys = ["context_relevance", "groundedness", "answer_relevance", "rag_triad_composite"]

    c_vals = [best_c[k] for k in keys]
    b_vals = [best_b[k] for k in keys]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=c_vals + [c_vals[0]], theta=dims + [dims[0]],
        fill="toself", opacity=0.75,
        name=f"Best Corpus: {best_c['experiment']}  (composite={best_c['rag_triad_composite']:.3f})",
        line_color=COLORS["corpus"],
    ))
    fig.add_trace(go.Scatterpolar(
        r=b_vals + [b_vals[0]], theta=dims + [dims[0]],
        fill="toself", opacity=0.75,
        name=f"Best Blended: {best_b['experiment']}  (composite={best_b['rag_triad_composite']:.3f})",
        line_color=COLORS["blended"],
    ))
    fig.update_layout(
        template=TEMPLATE,
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title=dict(text="RAG Triad Radar — Best Corpus vs Best Blended", font=dict(size=16)),
        legend=dict(orientation="h", y=-0.18),
        margin=dict(t=80, b=100),
    )
    _save(fig, "02_radar_best_runs", height=520)


# ══════════════════════════════════════════════════════════════════════════════
# Plot 3: Per-category CR heatmap
# ══════════════════════════════════════════════════════════════════════════════

def plot_category_cr_heatmap(df: pd.DataFrame) -> None:
    """Heatmap: CR by (experiment × category) for corpus mode."""
    cat_cr_cols = sorted([c for c in df.columns
                          if c.startswith("cat_") and c.endswith("_cr")])
    categories  = [c.replace("cat_", "").replace("_cr", "") for c in cat_cr_cols]

    sub = (df[df["mode"] == "corpus"]
           .copy()
           .set_index("experiment")
           .reindex(EXPERIMENT_ORDER)
           .reset_index()
           .dropna(subset=["rag_triad_composite"]))

    sub["exp_label"] = (sub["experiment"]
                        .map(EXPERIMENT_LABELS)
                        .fillna(sub["experiment"])
                        .str.replace("\n", " "))

    z    = sub[cat_cr_cols].values.tolist()
    ylbl = sub["exp_label"].tolist()

    fig = go.Figure(go.Heatmap(
        z=z, x=categories, y=ylbl,
        colorscale="RdYlGn", zmin=0, zmax=1,
        text=[[f"{v:.2f}" if v is not None else "—" for v in row] for row in z],
        texttemplate="%{text}",
        hovertemplate="Exp: %{y}<br>Cat: %{x}<br>CR: %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        template=TEMPLATE,
        title=dict(
            text="Context Relevance by Category — Corpus Mode (Red = Retrieval Failure)",
            font=dict(size=14),
        ),
        xaxis=dict(title="Query Category", tickangle=-30),
        yaxis_title="Experiment",
        margin=dict(b=100, l=260, t=70, r=20),
    )
    _save(fig, "03_category_cr_heatmap", height=420)


# ══════════════════════════════════════════════════════════════════════════════
# Plot 4: Corpus vs Blended grouped bar
# ══════════════════════════════════════════════════════════════════════════════

def plot_corpus_vs_blended(df: pd.DataFrame) -> None:
    """Grouped bar: composite score per experiment, corpus vs blended."""
    corpus  = df[df["mode"] == "corpus" ].set_index("experiment")
    blended = df[df["mode"] == "blended"].set_index("experiment")

    # Only experiments that have corpus results
    exps    = [e for e in EXPERIMENT_ORDER if e in corpus.index]
    xlabels = [EXPERIMENT_LABELS.get(e, e).replace("\n", "<br>") for e in exps]
    c_vals  = [corpus.loc[e, "rag_triad_composite"] for e in exps]
    b_vals  = [blended.loc[e, "rag_triad_composite"] if e in blended.index else None
               for e in exps]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Task 1a · Corpus", x=xlabels, y=c_vals,
        marker_color=COLORS["corpus"],
        text=[f"{v:.3f}" for v in c_vals], textposition="outside",
    ))
    fig.add_trace(go.Bar(
        name="Task 1b · Blended", x=xlabels, y=b_vals,
        marker_color=COLORS["blended"],
        text=[f"{v:.3f}" if v else "—" for v in b_vals], textposition="outside",
    ))

    for i, (c, b) in enumerate(zip(c_vals, b_vals)):
        if b is not None:
            delta = b - c
            fig.add_annotation(
                x=xlabels[i], y=max(c, b) + 0.07,
                text=f"Δ{delta:+.3f}", showarrow=False,
                font=dict(size=10, color="#00cc00" if delta >= 0 else "#ff4444"),
            )

    fig.update_layout(
        template=TEMPLATE,
        barmode="group",
        title=dict(text="RAG Triad Composite — Task 1a vs Task 1b by Experiment",
                   font=dict(size=15)),
        yaxis=dict(range=[0, 1.12], tickformat=".2f", title="Composite Score"),
        legend=dict(orientation="h", y=-0.22),
        margin=dict(b=120, t=80, l=60, r=20),
    )
    _save(fig, "04_corpus_vs_blended", height=560)


# ══════════════════════════════════════════════════════════════════════════════
# Plot 5: Latency vs quality scatter
# ══════════════════════════════════════════════════════════════════════════════

def plot_latency_vs_quality(df: pd.DataFrame) -> None:
    """Scatter: latency vs composite — top-left = best trade-off."""
    d2 = df.copy()
    d2["label"] = (d2["experiment"]
                   .map(EXPERIMENT_LABELS)
                   .fillna(d2["experiment"])
                   .str.replace("\n", " ")
                   + " (" + d2["mode"] + ")")

    fig = px.scatter(
        d2, x="avg_latency_ms", y="rag_triad_composite",
        color="mode", text="label",
        color_discrete_map={"corpus": COLORS["corpus"], "blended": COLORS["blended"]},
        template=TEMPLATE,
        title="Latency vs RAG Triad Composite — Top-Left = Best Trade-off",
        labels={"avg_latency_ms": "Avg Latency (ms)",
                "rag_triad_composite": "Composite Score"},
    )
    fig.update_traces(textposition="top center", marker=dict(size=10))
    fig.update_layout(
        yaxis=dict(range=[0.5, 0.9], tickformat=".3f"),
        margin=dict(t=80, b=60),
    )
    _save(fig, "05_latency_vs_quality", height=500)


# ══════════════════════════════════════════════════════════════════════════════
# Plot 6: CR vs GRD gap by category
# ══════════════════════════════════════════════════════════════════════════════

def plot_cr_grd_gap(df: pd.DataFrame) -> None:
    """
    Bar chart: GRD - CR gap per category (corpus, averaged across experiments).
    Large gap = retrieved context is wrong despite good answer grounding.
    These categories are the prime targets for retrieval improvements.
    """
    cat_cr_cols  = sorted([c for c in df.columns
                           if c.startswith("cat_") and c.endswith("_cr")])
    cat_grd_cols = [c.replace("_cr", "_grd") for c in cat_cr_cols]
    categories   = [c.replace("cat_", "").replace("_cr", "") for c in cat_cr_cols]

    corpus_df = df[df["mode"] == "corpus"]
    mean_cr   = corpus_df[cat_cr_cols].mean()
    mean_grd  = corpus_df[cat_grd_cols].mean()
    gaps      = [mean_grd.iloc[i] - mean_cr.iloc[i] for i in range(len(categories))]

    # Sort by gap descending — biggest problems first
    sorted_data = sorted(
        zip(categories, mean_cr.tolist(), mean_grd.tolist(), gaps),
        key=lambda x: x[3], reverse=True,
    )
    cats  = [d[0] for d in sorted_data]
    cr_v  = [d[1] for d in sorted_data]
    grd_v = [d[2] for d in sorted_data]
    gap_v = [d[3] for d in sorted_data]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Context Relevance (avg)", x=cats, y=cr_v,
        marker_color=COLORS["cr"],
        text=[f"{v:.2f}" for v in cr_v], textposition="outside",
    ))
    fig.add_trace(go.Bar(
        name="Groundedness (avg)", x=cats, y=grd_v,
        marker_color=COLORS["grd"],
        text=[f"{v:.2f}" for v in grd_v], textposition="outside",
    ))
    for i, (cat, gap) in enumerate(zip(cats, gap_v)):
        fig.add_annotation(
            x=cat, y=max(cr_v[i], grd_v[i]) + 0.09,
            text=f"gap={gap:.2f}", showarrow=False,
            font=dict(size=10, color="#ffaa00"),
        )

    fig.update_layout(
        template=TEMPLATE,
        barmode="group",
        title=dict(
            text=("Groundedness vs Context Relevance Gap by Category "
                  "(Corpus, avg all experiments)<br>"
                  "<sup>Large gap = retrieval returning wrong chunks — "
                  "prime targets for sentence window / reranker upgrade</sup>"),
            font=dict(size=13),
        ),
        yaxis=dict(range=[0, 1.2], tickformat=".2f"),
        legend=dict(orientation="h", y=-0.2),
        margin=dict(b=120, t=100, l=60, r=20),
    )
    _save(fig, "06_cr_grd_gap_by_category", height=560)


# ══════════════════════════════════════════════════════════════════════════════
# Plot 7: Three-metric grouped bars — corpus only
# ══════════════════════════════════════════════════════════════════════════════

def plot_three_metric_bars(df: pd.DataFrame) -> None:
    """Grouped bars: CR / GRD / AR / Composite side-by-side per experiment."""
    sub = (df[df["mode"] == "corpus"]
           .copy()
           .set_index("experiment")
           .reindex(EXPERIMENT_ORDER)
           .reset_index()
           .dropna(subset=["rag_triad_composite"]))

    sub["exp_label"] = (sub["experiment"]
                        .map(EXPERIMENT_LABELS)
                        .fillna(sub["experiment"])
                        .str.replace("\n", "<br>"))

    fig = go.Figure()
    for metric, color in [
        ("context_relevance",   COLORS["cr"]),
        ("groundedness",        COLORS["grd"]),
        ("answer_relevance",    COLORS["ar"]),
        ("rag_triad_composite", COLORS["composite"]),
    ]:
        fig.add_trace(go.Bar(
            name=METRIC_LABELS[metric],
            x=sub["exp_label"], y=sub[metric],
            marker_color=color,
            text=[f"{v:.3f}" if v is not None else "" for v in sub[metric]],
            textposition="outside",
        ))

    fig.update_layout(
        template=TEMPLATE,
        barmode="group",
        title=dict(
            text=("All RAG Triad Metrics — Corpus Mode by Experiment<br>"
                  "<sup>Context Relevance (red) is the floor pulling composite below "
                  "Groundedness & Answer Relevance</sup>"),
            font=dict(size=13),
        ),
        yaxis=dict(range=[0, 1.18], tickformat=".2f"),
        legend=dict(orientation="h", y=-0.22),
        margin=dict(b=120, t=100, l=60, r=20),
    )
    _save(fig, "07_three_metric_bars_corpus", height=560)


# ══════════════════════════════════════════════════════════════════════════════
# Plot 8: Sentence window vs standard — direct comparison
# ══════════════════════════════════════════════════════════════════════════════

def plot_sentence_window_comparison(df: pd.DataFrame) -> None:
    """
    Direct comparison: standard chunking experiments vs sentence window
    experiments. Sentence window (06, 09) showed lower CR than expected —
    this chart makes that finding explicit and presents it as an honest
    negative result with discussion.
    """
    standard_exps = ["02_optimized_html_parser", "04_retrieval_weights",
                     "05_quality_filter", "07_prompt_engineering",
                     "10_quality_filter_MinTxt500_RetrWeights"]
    sw_exps       = ["06_sentence_window", "09_sw_min_sent_80"]

    corpus = df[df["mode"] == "corpus"].set_index("experiment")

    def _row(exp, label_prefix):
        if exp not in corpus.index:
            return None
        r = corpus.loc[exp]
        return {
            "label": f"{label_prefix}\n{EXPERIMENT_LABELS.get(exp, exp).replace(chr(10), ' ')}",
            "group": label_prefix,
            "context_relevance":   r["context_relevance"],
            "groundedness":        r["groundedness"],
            "answer_relevance":    r["answer_relevance"],
            "rag_triad_composite": r["rag_triad_composite"],
        }

    rows = (
        [r for e in standard_exps if (r := _row(e, "Standard"))] +
        [r for e in sw_exps       if (r := _row(e, "Sent.Window"))]
    )
    if not rows:
        return

    comp_df = pd.DataFrame(rows)

    fig = go.Figure()
    for metric, color in [
        ("context_relevance",   COLORS["cr"]),
        ("rag_triad_composite", COLORS["composite"]),
    ]:
        fig.add_trace(go.Bar(
            name=METRIC_LABELS[metric],
            x=comp_df["label"],
            y=comp_df[metric],
            marker_color=color,
            text=[f"{v:.3f}" if v is not None else "" for v in comp_df[metric]],
            textposition="outside",
        ))

    # Shade sentence window region
    sw_labels = comp_df[comp_df["group"] == "Sent.Window"]["label"].tolist()
    if sw_labels:
        fig.add_vrect(
            x0=sw_labels[0], x1=sw_labels[-1],
            fillcolor="rgba(255,100,100,0.08)",
            layer="below", line_width=0,
            annotation_text="Sentence Window experiments",
            annotation_position="top left",
            annotation_font=dict(color="#ff6666", size=11),
        )

    fig.update_layout(
        template=TEMPLATE,
        barmode="group",
        title=dict(
            text=("Sentence Window vs Standard Chunking — CR & Composite (Corpus)<br>"
                  "<sup>Sentence window produced lower CR — smaller retrieval units "
                  "increased noise without improving precision at this corpus scale</sup>"),
            font=dict(size=13),
        ),
        yaxis=dict(range=[0, 1.1], tickformat=".2f"),
        xaxis=dict(tickangle=-20),
        legend=dict(orientation="h", y=-0.22),
        margin=dict(b=120, t=100, l=60, r=20),
    )
    _save(fig, "08_sentence_window_vs_standard", height=520)


# ══════════════════════════════════════════════════════════════════════════════
# Summary CSV + terminal findings
# ══════════════════════════════════════════════════════════════════════════════

def save_summary_csv(df: pd.DataFrame) -> None:
    cols = ["experiment", "mode", "timestamp",
            "context_relevance", "groundedness", "answer_relevance",
            "rag_triad_composite", "avg_latency_ms",
            "num_questions", "num_errors"]
    out = OUTPUT_DIR / "experiment_summary.csv"
    df[cols].to_csv(out, index=False, float_format="%.4f")
    logger.info(f"Saved: {out}")


def print_findings(df: pd.DataFrame) -> None:
    sep = "=" * 72

    print(f"\n{sep}")
    print("EXPERIMENT ANALYSIS — Autodesk RAG Chatbot")
    print(sep)

    # Best runs
    best_c = (df[df["mode"] == "corpus"]
              .sort_values("rag_triad_composite", ascending=False).iloc[0])
    print(f"\n{'Best Corpus Run':─<72}")
    print(f"  Experiment  : {best_c['experiment']}")
    print(f"  Composite   : {best_c['rag_triad_composite']:.3f}")
    print(f"  CR/GRD/AR   : {best_c['context_relevance']:.3f} / "
          f"{best_c['groundedness']:.3f} / {best_c['answer_relevance']:.3f}")
    print(f"  Latency     : {best_c['avg_latency_ms']:.0f}ms")

    blended_df = df[df["mode"] == "blended"]
    if not blended_df.empty:
        best_b = blended_df.sort_values("rag_triad_composite", ascending=False).iloc[0]
        print(f"\n{'Best Blended Run':─<72}")
        print(f"  Experiment  : {best_b['experiment']}")
        print(f"  Composite   : {best_b['rag_triad_composite']:.3f}")
        print(f"  CR/GRD/AR   : {best_b['context_relevance']:.3f} / "
              f"{best_b['groundedness']:.3f} / {best_b['answer_relevance']:.3f}")
        print(f"  Latency     : {best_b['avg_latency_ms']:.0f}ms")

    # CR analysis
    print(f"\n{'Context Relevance — Weak Leg Analysis (corpus avg)':─<72}")
    corpus_df = df[df["mode"] == "corpus"]
    print(f"  Avg CR  : {corpus_df['context_relevance'].mean():.3f}")
    print(f"  Avg GRD : {corpus_df['groundedness'].mean():.3f}")
    print(f"  Avg AR  : {corpus_df['answer_relevance'].mean():.3f}")
    print(f"  CR deficit vs GRD : "
          f"{corpus_df['groundedness'].mean() - corpus_df['context_relevance'].mean():.3f}")

    # Per-category CR
    cat_cr_cols = sorted([c for c in df.columns
                          if c.startswith("cat_") and c.endswith("_cr")])
    categories  = [c.replace("cat_", "").replace("_cr", "") for c in cat_cr_cols]
    mean_cr     = corpus_df[cat_cr_cols].mean()
    print(f"\n{'Context Relevance by Category (worst→best)':─<72}")
    for cat, val in sorted(zip(categories, mean_cr), key=lambda x: x[1]):
        bar = "█" * max(1, int(val * 24))
        print(f"  {cat:<28} {val:.3f}  {bar}")

    # Sentence window finding
    print(f"\n{'Sentence Window Finding (negative result)':─<72}")
    sw_runs = df[(df["mode"] == "corpus") &
                 (df["experiment"].isin(["06_sentence_window", "09_sw_min_sent_80"]))]
    std_avg = corpus_df[~corpus_df["experiment"].isin(
        ["06_sentence_window", "09_sw_min_sent_80"]
    )]["context_relevance"].mean()
    sw_avg = sw_runs["context_relevance"].mean() if not sw_runs.empty else None

    if sw_avg is not None:
        print(f"  Standard chunking avg CR  : {std_avg:.3f}")
        print(f"  Sentence window avg CR    : {sw_avg:.3f}")
        print(f"  Delta                     : {sw_avg - std_avg:+.3f}")
        print("  Interpretation: Sentence-level retrieval units are too fine-grained")
        print("  for this corpus. Small chunks lack the context the judge needs to")
        print("  score them as relevant. Correct fix: reranker upgrade or larger")
        print("  embedding model rather than smaller retrieval granularity.")

    # Latency
    c_lat = corpus_df["avg_latency_ms"].mean()
    b_lat = df[df["mode"] == "blended"]["avg_latency_ms"].mean() if not blended_df.empty else 0
    print(f"\n{'Latency':─<72}")
    print(f"  Avg corpus latency  : {c_lat:.0f}ms")
    if b_lat:
        print(f"  Avg blended latency : {b_lat:.0f}ms  "
              f"(+{b_lat - c_lat:.0f}ms / {(b_lat/c_lat - 1)*100:.0f}% web overhead)")

    print(f"\n{'Recommended Next Experiments':─<72}")
    print("  08_larger_reranker  — swap to ms-marco-MiniLM-L-12-v2 (no re-ingest)")
    print("  11_large_embeddings — swap to bge-large-en-v1.5 (re-ingest required)")
    print("  Both target CR directly via better ranking and semantic matching.")
    print(f"\n{sep}\n")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    logger.info(f"Loading results from {PROCESSED_DIR}")
    df = load_all_results()

    logger.info("Generating plots (PNG)...")
    plot_experiment_progression(df)
    plot_radar_best(df)
    plot_category_cr_heatmap(df)
    plot_corpus_vs_blended(df)
    plot_latency_vs_quality(df)
    plot_cr_grd_gap(df)
    plot_three_metric_bars(df)
    plot_sentence_window_comparison(df)

    save_summary_csv(df)
    print_findings(df)

    logger.info(f"All PNG outputs saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()