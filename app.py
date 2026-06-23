import os
import multiprocessing

multiprocessing.freeze_support()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from src.ingestion import BioRAGVectorStore, load_embeddings, CHROMA_PERSIST
from src.pipeline import BioRAGPipeline
from src.retrievers import (
    NaiveRetriever, HyDERetriever, RerankerRetriever,
    HyDERerankerRetriever, MultiQueryRetriever, HybridRetriever,
)

load_dotenv()

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BioRAG — BioNLP 2026",
    page_icon="🔬",
    layout="wide",
)

# ── Constants ──────────────────────────────────────────────────────────────────
RESULTS_CSV      = "results/ablation_table_bioasq.csv"
SIGNIFICANCE_CSV = "results/significance_summary.csv"
K_SENS_CSV       = "results/k_sensitivity.csv"

VARIANT_LABELS = {
    "V1_Naive":         "V1 Naive",
    "V2_HyDE":          "V2 HyDE",
    "V3_Reranker":      "V3 Reranker",
    "V4_HyDE_Reranker": "V4 HyDE+Reranker",
    "V5_MultiQuery":    "V5 Multi-Query",
    "V6_SelfRAG":       "V6 Self-RAG†",
    "V7_Hybrid":        "V7 Hybrid",
}

VARIANT_COLORS = {
    "V1_Naive":         "#888780",  # poster gray (baseline)
    "V2_HyDE":          "#378add",  # poster blue
    "V3_Reranker":      "#5b9fd6",  # poster blue (lighter)
    "V4_HyDE_Reranker": "#2070bd",  # poster blue (darker)
    "V5_MultiQuery":    "#4d90d4",  # poster blue (mid)
    "V6_SelfRAG":       "#e24b4a",  # poster red (rate-limited)
    "V7_Hybrid":        "#0d2d5e",  # poster navy (winner)
}

# Demo retriever keys — used for ordering and color index in Live Demo tab
DEMO_KEYS = [
    "V1: Naive", "V2: HyDE", "V3: Reranker",
    "V4: HyDE + Reranker", "V5: Multi-Query",
    "V7: Hybrid BM25+Dense",
]

_PLOT_BG    = "#ffffff"
_GRID_COLOR = "#d0cfc8"
_FONT       = "IBM Plex Sans"


def _hex_rgba(hex_color: str, alpha: float = 0.12) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ── Cache functions (defined before CSS to avoid inspect.getsource tokenizer issues) ──
@st.cache_resource(show_spinner=False)
def _get_embeddings():
    return load_embeddings()

@st.cache_resource(show_spinner=False)
def _get_vectorstore(_emb):
    return BioRAGVectorStore(embeddings=_emb, persist_dir=CHROMA_PERSIST)

@st.cache_resource(show_spinner=False)
def _get_llm(api_key: str):
    return ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.1-8b-instant",
        temperature=0.1,
        max_tokens=512,
    )

@st.cache_resource(show_spinner="Loading retrieval models…")
def _get_retrievers(_store, _llm):
    # _-prefix parameters are excluded from Streamlit's hash check
    return {
        "V1: Naive":             NaiveRetriever(_store),
        "V2: HyDE":              HyDERetriever(_store, _llm),
        "V3: Reranker":          RerankerRetriever(_store),
        "V4: HyDE + Reranker":   HyDERerankerRetriever(_store, _llm),
        "V5: Multi-Query":       MultiQueryRetriever(_store, _llm),
        "V7: Hybrid BM25+Dense": HybridRetriever(_store, corpus_dir="data/corpus"),
    }


# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,wght@0,300;0,600;1,300&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
:root {
    --bg:      #f8f7f3;
    --surface: #ffffff;
    --raised:  #f0ede6;
    --border:  #d0cfc8;
    --navy:    #0d2d5e;
    --teal:    #0f6e56;
    --gold:    #b8860b;
    --red:     #e24b4a;
    --text:    #1a1a1a;
    --muted:   #555555;
}
html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    font-family: 'IBM Plex Sans', sans-serif;
    color: var(--text);
}
[data-testid="stSidebar"] {
    background: var(--navy) !important;
    border-right: none;
}
[data-testid="stSidebar"] * { color: rgba(255,255,255,0.88) !important; }
[data-testid="stSidebar"] .stTextInput input {
    background: rgba(255,255,255,0.12) !important;
    border-color: rgba(255,255,255,0.25) !important;
    color: #fff !important;
}
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.2) !important; }
[data-testid="stSidebar"] label { color: rgba(255,255,255,0.7) !important; }
h1, h2, h3 { font-family: 'Source Serif 4', serif; color: var(--navy); }
h1 { font-size: 1.3rem; }
h2 { font-size: 1.05rem; }
h3 { font-size: 0.9rem; }
.sec-hdr {
    display: inline-block;
    background: var(--navy);
    color: #fff;
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    padding: 3px 10px;
    border-radius: 2px;
    margin-bottom: 10px;
}
.paper-title {
    font-family: 'Source Serif 4', serif;
    font-size: 1.35rem;
    font-weight: 600;
    color: var(--navy);
    line-height: 1.35;
    margin-bottom: 6px;
}
.authors {
    font-size: 0.78rem;
    color: var(--muted);
    margin-bottom: 20px;
}
.stat-card {
    background: var(--surface);
    border: 0.5px solid var(--border);
    border-radius: 5px;
    padding: 12px 16px;
    text-align: center;
}
.stat-val {
    font-family: 'Source Serif 4', serif;
    font-size: 1.8rem;
    font-weight: 600;
    color: var(--navy);
}
.stat-lbl {
    font-size: 0.62rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-top: 2px;
}
.finding-card {
    background: var(--surface);
    border-top: 0.5px solid var(--border);
    border-right: 0.5px solid var(--border);
    border-bottom: 0.5px solid var(--border);
    border-left: 3px solid var(--navy);
    border-radius: 0 4px 4px 0;
    padding: 12px 16px;
    margin-bottom: 10px;
    font-size: 0.88rem;
    line-height: 1.65;
    color: var(--text);
}
.finding-tag {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.62rem;
    font-weight: 600;
    color: var(--navy);
    letter-spacing: 0.09em;
    text-transform: uppercase;
    margin-bottom: 5px;
}
.note-box {
    background: #e1f5ee;
    border: 0.5px solid #1d9e75;
    border-radius: 4px;
    padding: 9px 14px;
    font-size: 0.8rem;
    color: #04342c;
    margin: 8px 0 14px 0;
}
.warn-box {
    background: #faeeda;
    border: 0.5px solid #ef9f27;
    border-radius: 4px;
    padding: 9px 14px;
    font-size: 0.8rem;
    color: #412402;
    margin: 8px 0 14px 0;
}
.variant-card {
    background: var(--surface);
    border-top: 0.5px solid var(--border);
    border-right: 0.5px solid var(--border);
    border-bottom: 0.5px solid var(--border);
    border-left: 3px solid var(--navy);
    border-radius: 0 4px 4px 0;
    padding: 14px 16px;
    margin-bottom: 10px;
}
.variant-label {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.answer-text { font-size: 0.88rem; line-height: 1.7; color: var(--text); }
.latency-badge {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem;
    padding: 2px 7px;
    border-radius: 3px;
    background: #e6f1fb;
    color: #0c447c;
    margin-left: 8px;
}
</style>
""", unsafe_allow_html=True)


# ── Data loaders ───────────────────────────────────────────────────────────────
def _load_csv(path: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if "variant" in df.columns:
        df["label"] = df["variant"].map(VARIANT_LABELS).fillna(df["variant"])
    return df

def _plot_layout(**kwargs) -> dict:
    return dict(
        paper_bgcolor=_PLOT_BG, plot_bgcolor=_PLOT_BG,
        font=dict(color="#1a1a1a", family=_FONT),
        margin=dict(t=40, b=20, l=20, r=20),
        **kwargs,
    )


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='font-family:Source Serif 4,serif;font-size:1.3rem;font-weight:600;"
        "color:#fff;margin-bottom:2px;'>BioRAG</div>"
        "<div style='font-family:IBM Plex Sans,sans-serif;font-size:0.65rem;"
        "color:rgba(255,255,255,0.6);letter-spacing:0.08em;text-transform:uppercase;"
        "margin-bottom:4px;'>BioNLP 2026</div>"
        "<div style='font-size:0.76rem;color:rgba(255,255,255,0.75);margin-bottom:20px;"
        "line-height:1.5;'>7-variant ablation study on<br>BioASQ-13b · 4 RAGAs metrics</div>",
        unsafe_allow_html=True,
    )

    api_key = os.getenv("GROQ_API_KEY", "")
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    if not api_key:
        api_key = st.text_input(
            "Groq API Key",
            type="password",
            placeholder="gsk_… (required for Live Demo)",
        )

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.68rem;color:rgba(255,255,255,0.55);"
        "font-family:IBM Plex Mono,monospace;line-height:1.7;'>"
        "bhojank@sunypoly.edu<br>"
        "SUNY Polytechnic Institute, USA</div>",
        unsafe_allow_html=True,
    )


# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_paper, tab_demo, tab_explorer = st.tabs([
    "📄  Paper Results",
    "🔍  Live Demo",
    "📈  Ablation Explorer",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PAPER RESULTS
# ══════════════════════════════════════════════════════════════════════════════
with tab_paper:
    st.markdown(
        "<div class='paper-title'>"
        "BioRAG: A Systematic Ablation Study of Retrieval Strategies<br>"
        "for Biomedical Question Answering"
        "</div>"
        "<div class='authors'>"
        "Krushil Bhojani · Mayank Waghmare · Hima Bindu Nandyala · "
        "Krupali Hirpara · Shekhar Yadav · Aakash Malhan"
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Corpus stats banner ────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    for col, val, lbl in [
        (c1, "1,954",  "PubMed Articles"),
        (c2, "10,425", "Indexed Chunks"),
        (c3, "7",      "Retrieval Variants"),
        (c4, "100",    "BioASQ Questions"),
        (c5, "4",      "RAGAs Metrics"),
    ]:
        with col:
            st.markdown(
                f"<div class='stat-card'>"
                f"<div class='stat-val'>{val}</div>"
                f"<div class='stat-lbl'>{lbl}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Key findings ───────────────────────────────────────────────────────────
    st.markdown("### Key Findings")
    for tag, body in [
        (
            "Finding 1 — V7 Hybrid dominates",
            "Hybrid BM25+Dense retrieval achieves faithfulness <strong>0.534</strong> and context recall "
            "<strong>0.507</strong> — improvements of <strong>50%</strong> and <strong>85%</strong> "
            "over the V1 Naive baseline. Bootstrap confidence intervals are non-overlapping, "
            "confirming statistical significance.",
        ),
        (
            "Finding 2 — HyDE faithfulness-precision tradeoff",
            "HyDE (V2) improves faithfulness by <strong>+14%</strong> over V1 but reduces context "
            "precision by <strong>−52%</strong>. This previously undocumented tradeoff has direct "
            "implications for clinical RAG design: HyDE alone is counterproductive where precision "
            "is critical (e.g., drug dosage retrieval).",
        ),
        (
            "Finding 3 — No single strategy dominates all metrics",
            "V7 Hybrid leads faithfulness, context precision, and context recall. "
            "V4 HyDE+Reranker leads answer relevancy. "
            "Retrieval strategy selection must be driven by application-specific clinical requirements.",
        ),
    ]:
        st.markdown(
            f"<div class='finding-card'>"
            f"<div class='finding-tag'>{tag}</div>{body}</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Load results ───────────────────────────────────────────────────────────
    df = _load_csv(RESULTS_CSV)
    if df is None:
        st.warning(
            "Results not found. Run:\n"
            "```\npython run_eval.py --variant all --eval_set bioasq "
            "--bioasq_path data/BioASQ-training13b.json --n_questions 100 "
            "--output results/ablation_table_bioasq.csv --resume "
            "--ollama_model gemma3:12b\n```"
        )
    else:
        # ── Table 2 ────────────────────────────────────────────────────────────
        st.markdown("### Table 2 · RAGAs Evaluation Results &nbsp; *(seed 42, k = 5)*")
        st.markdown(
            "<div class='note-box'>"
            "†V6 Self-RAG scores reflect Groq free-tier rate-limiting constraints "
            "(48,730 ms latency), not algorithmic performance. See Appendix E.</div>",
            unsafe_allow_html=True,
        )

        tbl = df[["label", "faithfulness", "answer_relevancy",
                   "context_precision", "context_recall", "avg_latency_ms"]].copy()
        tbl.columns = ["Variant", "Faithfulness ↑", "Ans. Relevancy ↑",
                        "Ctx. Precision ↑", "Ctx. Recall ↑", "Latency (ms)"]
        tbl["Latency (ms)"] = tbl["Latency (ms)"].round(0).astype(int)

        st.dataframe(
            tbl.style
               .highlight_max(
                   subset=["Faithfulness ↑", "Ans. Relevancy ↑",
                            "Ctx. Precision ↑", "Ctx. Recall ↑"],
                   color="#e1f5ee",
               )
               .format({c: "{:.3f}" for c in ["Faithfulness ↑", "Ans. Relevancy ↑",
                                               "Ctx. Precision ↑", "Ctx. Recall ↑"]}),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        col_fig2, col_fig3 = st.columns([3, 2])

        # ── Figure 2 — Radar (V6 excluded per paper) ──────────────────────────
        with col_fig2:
            st.markdown("### Figure 2 · Metric Profiles *(V6 excluded — infrastructure confounding)*")
            metrics       = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
            metric_labels = ["Faithfulness", "Ans. Relevancy", "Ctx. Precision", "Ctx. Recall"]
            df_radar      = df[df["variant"] != "V6_SelfRAG"].reset_index(drop=True)

            fig2 = go.Figure()
            for _, row in df_radar.iterrows():
                color = VARIANT_COLORS.get(row["variant"], "#888888")
                vals  = [row[m] for m in metrics] + [row[metrics[0]]]
                fig2.add_trace(go.Scatterpolar(
                    r=vals,
                    theta=metric_labels + [metric_labels[0]],
                    name=row["label"],
                    line=dict(color=color, width=2),
                    fill="toself",
                    fillcolor=_hex_rgba(color, 0.10),
                ))
            fig2.update_layout(
                **_plot_layout(height=400),
                polar=dict(
                    bgcolor="rgba(248,247,243,0.5)",
                    radialaxis=dict(range=[0, 0.65], gridcolor=_GRID_COLOR,
                                    tickfont=dict(size=9)),
                    angularaxis=dict(gridcolor=_GRID_COLOR),
                ),
                legend=dict(bgcolor="#ffffff", bordercolor=_GRID_COLOR,
                            borderwidth=1, font=dict(size=10)),
            )
            st.plotly_chart(fig2, use_container_width=True)

        # ── Figure 3 — Faithfulness bar (all variants) ────────────────────────
        with col_fig3:
            st.markdown("### Figure 3 · Faithfulness by Variant *(all variants)*")
            baseline = float(df[df["variant"] == "V1_Naive"]["faithfulness"].iloc[0])
            fig3 = go.Figure([go.Bar(
                x=df["label"],
                y=df["faithfulness"],
                marker_color=[VARIANT_COLORS.get(v, "#888") for v in df["variant"]],
                text=[f"{v:.3f}" for v in df["faithfulness"]],
                textposition="outside",
                textfont=dict(size=10),
            )])
            fig3.add_hline(
                y=baseline, line_dash="dash", line_color="#888780",
                annotation_text="V1 baseline",
                annotation_font_size=9,
                annotation_font_color="#888780",
            )
            fig3.update_layout(
                **_plot_layout(height=400),
                yaxis=dict(range=[0, 0.68], gridcolor=_GRID_COLOR, title="Faithfulness"),
                xaxis=dict(gridcolor=_GRID_COLOR, tickangle=-35),
                showlegend=False,
            )
            st.plotly_chart(fig3, use_container_width=True)

        st.markdown("---")

        # ── Table 4 — Clinical deployment guidance ────────────────────────────
        st.markdown("### Table 4 · Clinical Deployment Guidance")
        st.markdown(
            "<div class='note-box'>"
            "Strategy selection should be driven by the primary clinical objective. "
            "Clinical validation is required before deployment.</div>",
            unsafe_allow_html=True,
        )
        clinical = pd.DataFrame([
            ("Clinical Decision Support",    "V7 Hybrid",         "Faithfulness",      "Maximises grounded answers; minimises hallucination risk"),
            ("Patient-Facing QA",            "V4 HyDE+Reranker",  "Answer Relevancy",  "Highest directness and relevance to patient questions"),
            ("Systematic Literature Review", "V7 Hybrid",         "Context Recall",    "Retrieves most gold-standard BioASQ evidence"),
            ("Real-Time Interface",          "V3 Reranker",       "Latency (388 ms)",  "Best accuracy–latency tradeoff for interactive QA"),
            ("High-Throughput Screening",    "V1 Naive",          "Latency (46 ms)",   "Fastest retrieval for large-scale batch workloads"),
        ], columns=["Clinical Use Case", "Recommended Variant", "Primary Metric", "Rationale"])
        st.dataframe(clinical, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — LIVE DEMO
# ══════════════════════════════════════════════════════════════════════════════
with tab_demo:
    st.markdown("## 🔍 Live Biomedical QA")

    if not api_key:
        st.info("Enter your Groq API key in the sidebar to use the live demo.")
    else:
        with st.spinner("Loading BioMedBERT + ChromaDB…"):
            emb   = _get_embeddings()
            vs    = _get_vectorstore(emb)
            llm   = _get_llm(api_key)
            store = vs._get_or_create_store()
            stats = vs.collection_stats()

        if stats["total_chunks"] == 0:
            st.error(
                "ChromaDB collection is empty.\n\n"
                "Run to fetch and index articles:\n"
                "```\npython data/fetch_pubmed.py --topic neurology --max_results 2000\n"
                "python src/ingestion.py\n```"
            )
        else:
            st.markdown(
                f"<div style='font-size:0.78rem;color:#555555;margin-bottom:16px;'>"
                f"Corpus: {stats['unique_pmids']:,} PubMed articles · "
                f"{stats['total_chunks']:,} indexed chunks · "
                f"Embeddings: BioMedBERT</div>",
                unsafe_allow_html=True,
            )

            retrievers = _get_retrievers(store, llm)

            # Controls
            col_mode, col_k, col_year = st.columns([3, 1, 1])
            with col_mode:
                mode = st.radio(
                    "Mode",
                    ["Compare All Variants", "Single Variant"],
                    horizontal=True,
                    label_visibility="collapsed",
                )
            with col_k:
                k = st.slider("k (chunks)", 3, 10, 5)
            with col_year:
                year_input = st.text_input("Year filter", placeholder="e.g. 2023")
            meta_filter = {"year": year_input} if year_input else None

            selected = None
            if mode == "Single Variant":
                selected = st.selectbox("Retriever variant", list(retrievers.keys()))

            # Sample questions
            SAMPLE_QS = [
                "What are the mechanisms by which tau hyperphosphorylation leads to neurofibrillary tangle formation in Alzheimer's disease?",
                "Which disease-modifying therapies have shown efficacy in relapsing-remitting multiple sclerosis?",
                "How does alpha-synuclein aggregation contribute to dopaminergic neuron loss in Parkinson's disease?",
                "What is the role of neuroinflammation in the progression of amyotrophic lateral sclerosis?",
                "What are the evidence-based treatment strategies for status epilepticus in adults?",
            ]

            st.markdown(
                "<div style='font-size:0.72rem;font-weight:600;color:#0d2d5e;"
                "letter-spacing:0.06em;text-transform:uppercase;margin-bottom:6px;'>"
                "Sample questions</div>",
                unsafe_allow_html=True,
            )
            sq_cols = st.columns(len(SAMPLE_QS))
            for i, (col, sq) in enumerate(zip(sq_cols, SAMPLE_QS)):
                with col:
                    label = sq[:38] + "…" if len(sq) > 38 else sq
                    if st.button(label, key=f"sq_{i}", use_container_width=True):
                        st.session_state["_demo_q"] = sq

            st.markdown(
                "<div class='warn-box'>"
                "⚠️ <strong>Scope disclaimer:</strong> This corpus contains 1,954 PubMed neurology "
                "abstracts (2015–2024). Questions outside neurology or this date range may return "
                "low-confidence or unrelated answers.</div>",
                unsafe_allow_html=True,
            )

            question = st.text_area(
                "question",
                value=st.session_state.get("_demo_q", ""),
                placeholder=(
                    "e.g. What are the mechanisms by which tau hyperphosphorylation "
                    "leads to neurofibrillary tangle formation in Alzheimer's disease?"
                ),
                height=80,
                label_visibility="collapsed",
            )

            if st.button("▶ Ask", type="primary") and question.strip():
                active = (
                    {selected: retrievers[selected]}
                    if mode == "Single Variant" and selected
                    else retrievers
                )
                pipeline = BioRAGPipeline(retriever=list(active.values())[0], llm=llm)
                n_cols   = min(len(active), 3)
                groups   = [
                    list(active.items())[i : i + n_cols]
                    for i in range(0, len(active), n_cols)
                ]

                for group in groups:
                    cols = st.columns(len(group))
                    for col, (name, retriever) in zip(cols, group):
                        with col:
                            with st.spinner(f"Querying {name}…"):
                                pipeline.switch_retriever(retriever)
                                result = pipeline.query(
                                    question, k=k, metadata_filter=meta_filter
                                )
                            color = VARIANT_COLORS.get(
                                result["variant"].replace(" ", "_"), "#00e5ff"
                            )
                            st.markdown(
                                f"<div class='variant-card' style='border-left-color:{color}'>"
                                f"<div class='variant-label' style='color:{color}'>"
                                f"{result['variant']}"
                                f"<span class='latency-badge'>{result['latency_ms']:.0f} ms</span>"
                                f"</div>"
                                f"<div class='answer-text'>{result['answer']}</div>"
                                f"</div>",
                                unsafe_allow_html=True,
                            )
                            inter = result.get("intermediate", {})
                            if "hypothesis" in inter:
                                with st.expander("💡 HyDE hypothesis"):
                                    st.code(inter["hypothesis"], language=None)
                            if "subqueries" in inter:
                                with st.expander("🔀 Decomposed sub-queries"):
                                    for i, sq in enumerate(inter["subqueries"], 1):
                                        st.markdown(f"`{i}.` {sq}")
                            if result["sources"]:
                                with st.expander(f"📎 {len(result['sources'])} sources"):
                                    for src in result["sources"]:
                                        st.markdown(
                                            f"**PMID {src['pmid']}** — {src['title'][:60]}  \n"
                                            f"*{src['journal']}, {src['year']}* · {src['section']}  \n"
                                            f"> {src['snippet'][:200]}…"
                                        )
                                        st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ABLATION EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with tab_explorer:
    st.markdown("## 📈 Ablation Explorer")

    df     = _load_csv(RESULTS_CSV)
    df_sig = _load_csv(SIGNIFICANCE_CSV)
    df_k   = _load_csv(K_SENS_CSV)

    if df is None:
        st.warning("Run evaluation first to generate results CSVs.")
    else:
        # ── Table 2 (interactive) ──────────────────────────────────────────────
        st.markdown("### Table 2 · Full Ablation Results")
        st.dataframe(
            df[["label", "faithfulness", "answer_relevancy",
                "context_precision", "context_recall",
                "avg_latency_ms", "n_questions"]]
            .rename(columns={
                "label": "Variant",
                "faithfulness": "Faithfulness",
                "answer_relevancy": "Ans. Relevancy",
                "context_precision": "Ctx. Precision",
                "context_recall": "Ctx. Recall",
                "avg_latency_ms": "Latency (ms)",
                "n_questions": "N",
            }),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Table 3 — Bootstrap CIs ────────────────────────────────────────────
        if df_sig is not None:
            st.markdown("### Table 3 · Bootstrap Confidence Intervals *(95%, 1000 resamples, 3 seeds)*")
            st.markdown(
                "<div class='note-box'>"
                "V6 Self-RAG excluded (infrastructure confounding). "
                "V7 Hybrid faithfulness CI [0.565, 0.607] does not overlap with any other variant — "
                "gains are statistically reliable.</div>",
                unsafe_allow_html=True,
            )

            sig_tbl = df_sig[["variant", "metric", "mean", "std", "ci_lower", "ci_upper"]].copy()
            sig_tbl["variant"] = sig_tbl["variant"].map(VARIANT_LABELS).fillna(sig_tbl["variant"])
            sig_tbl.columns = ["Variant", "Metric", "Mean", "Std", "CI Lower", "CI Upper"]

            col_tbl, col_ci = st.columns([2, 3])
            with col_tbl:
                faith_recall = sig_tbl[
                    sig_tbl["Metric"].isin(["faithfulness", "context_recall"])
                ]
                st.dataframe(
                    faith_recall.style.format({
                        "Mean": "{:.4f}", "Std": "{:.4f}",
                        "CI Lower": "{:.4f}", "CI Upper": "{:.4f}",
                    }),
                    use_container_width=True,
                    hide_index=True,
                )

            with col_ci:
                faith_sig = sig_tbl[sig_tbl["Metric"] == "faithfulness"].sort_values("Mean")
                fig_ci = go.Figure()
                for _, row in faith_sig.iterrows():
                    # find original variant key for color lookup
                    orig_key = next(
                        (k for k, v in VARIANT_LABELS.items() if v == row["Variant"]),
                        None,
                    )
                    color = VARIANT_COLORS.get(orig_key, "#888888") if orig_key else "#888888"
                    # error bar span
                    fig_ci.add_trace(go.Scatter(
                        x=[row["CI Lower"], row["CI Upper"]],
                        y=[row["Variant"], row["Variant"]],
                        mode="lines",
                        line=dict(color=color, width=2),
                        showlegend=False,
                    ))
                    # mean point
                    fig_ci.add_trace(go.Scatter(
                        x=[row["Mean"]],
                        y=[row["Variant"]],
                        mode="markers",
                        marker=dict(color=color, size=9, symbol="circle"),
                        name=row["Variant"],
                        showlegend=False,
                    ))
                fig_ci.update_layout(
                    **_plot_layout(height=300, title="Faithfulness — 95% Bootstrap CIs"),
                    xaxis=dict(title="Faithfulness", gridcolor=_GRID_COLOR, range=[0.27, 0.65]),
                    yaxis=dict(gridcolor=_GRID_COLOR),
                )
                st.plotly_chart(fig_ci, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Table 6 — k sensitivity ────────────────────────────────────────────
        if df_k is not None:
            st.markdown("### Table 6 · Hyperparameter Sensitivity *(k ∈ {3, 5, 10}, seed 123)*")
            st.markdown(
                "<div class='note-box'>"
                "V7 Hybrid maintains highest faithfulness at k=5 and k=10. "
                "Context recall improves from 0.368 → 0.658 as k increases. "
                "Relative rankings between variants are preserved across all tested k.</div>",
                unsafe_allow_html=True,
            )

            col_ktbl, col_kplot = st.columns([2, 3])

            with col_ktbl:
                st.dataframe(
                    df_k[["label", "k", "faithfulness",
                           "context_precision", "context_recall", "avg_latency_ms"]]
                    .rename(columns={
                        "label": "Variant", "faithfulness": "Faithfulness",
                        "context_precision": "Ctx. Precision",
                        "context_recall": "Ctx. Recall",
                        "avg_latency_ms": "Latency (ms)",
                    }),
                    use_container_width=True,
                    hide_index=True,
                )

            with col_kplot:
                metric_opt = st.selectbox(
                    "Metric", ["faithfulness", "context_recall", "context_precision"],
                    format_func=lambda x: x.replace("_", " ").title(),
                    key="k_metric",
                )
                fig_k = go.Figure()
                k_colors = {
                    "V1_Naive":    VARIANT_COLORS["V1_Naive"],
                    "V3_Reranker": VARIANT_COLORS["V3_Reranker"],
                    "V7_Hybrid":   VARIANT_COLORS["V7_Hybrid"],
                }
                for variant, grp in df_k.groupby("variant"):
                    grp = grp.sort_values("k")
                    fig_k.add_trace(go.Scatter(
                        x=grp["k"],
                        y=grp[metric_opt],
                        mode="lines+markers",
                        name=VARIANT_LABELS.get(variant, variant),
                        line=dict(color=k_colors.get(variant, "#888"), width=2),
                        marker=dict(size=9),
                    ))
                fig_k.update_layout(
                    **_plot_layout(
                        height=300,
                        title=f"{metric_opt.replace('_', ' ').title()} vs k",
                    ),
                    xaxis=dict(
                        title="k (chunks retrieved)",
                        tickvals=[3, 5, 10],
                        gridcolor=_GRID_COLOR,
                    ),
                    yaxis=dict(
                        title=metric_opt.replace("_", " ").title(),
                        gridcolor=_GRID_COLOR,
                    ),
                    legend=dict(
                        bgcolor="#ffffff",
                        bordercolor=_GRID_COLOR,
                        font=dict(size=10),
                    ),
                )
                st.plotly_chart(fig_k, use_container_width=True)
