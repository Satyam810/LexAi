"""LexAI v3.2 — Streamlit Dashboard.

STRICTLY display-only. All ML logic lives in src/search_pipeline.py.
"""
import streamlit as st
import plotly.express as px

import pandas as pd
import json, os, numpy as np

# Must be first Streamlit call
st.set_page_config(
    page_title="LexAI - Legal Judgment Analyzer",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Dark theme CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif; }

.main-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 2rem; border-radius: 16px; margin-bottom: 2rem;
    border: 1px solid rgba(255,255,255,0.1);
}
.main-header h1 {
    color: #e94560; font-size: 2.2rem; font-weight: 700; margin: 0;
}
.main-header p { color: #a8a8b3; margin: 0.5rem 0 0 0; }

.metric-card {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border: 1px solid rgba(233,69,96,0.3); border-radius: 12px;
    padding: 1.2rem; text-align: center;
}
.metric-card h3 { color: #e94560; font-size: 1.8rem; margin: 0; }
.metric-card p { color: #a8a8b3; font-size: 0.85rem; margin: 0.3rem 0 0 0; }

.result-card {
    background: rgba(26,26,46,0.8); border: 1px solid rgba(233,69,96,0.2);
    border-radius: 12px; padding: 1.2rem; margin-bottom: 1rem;
    transition: border-color 0.3s;
}
.result-card:hover { border-color: rgba(233,69,96,0.6); }

.gap-card {
    background: linear-gradient(135deg, rgba(233,69,96,0.1), rgba(15,52,96,0.3));
    border: 1px solid rgba(233,69,96,0.4); border-radius: 12px;
    padding: 1.2rem; margin-bottom: 1rem;
}

.tag {
    display: inline-block; background: rgba(233,69,96,0.2);
    color: #e94560; padding: 2px 8px; border-radius: 6px;
    font-size: 0.75rem; margin: 2px;
}
.tag-green {
    background: rgba(0,200,117,0.2); color: #00c875;
}
</style>
""", unsafe_allow_html=True)


def get_metrics():
    with open("data/processed/eval_metrics.json") as f:
        return json.load(f)

def get_cluster_data():
    with open("data/processed/cases.json") as f: cases = json.load(f)
    with open("data/processed/cluster_topics.json") as f: topics = json.load(f)
    labels = np.load("data/processed/cluster_labels.npy")
    coords = np.load("data/processed/coords_2d.npy")
    return {"cases": cases, "labels": labels, "coords": coords, "topics": topics}

def get_gaps():
    with open("data/processed/gaps.json") as f:
        return json.load(f)


def render_header():
    st.markdown("""
    <div class="main-header">
        <h1>LexAI v3.2</h1>
        <p>Indian Court Judgment Similarity Engine & Legal Gap Finder</p>
    </div>
    """, unsafe_allow_html=True)


def render_metrics(metrics):
    cols = st.columns(4)
    items = [
        (metrics.get("total_cases", 0), "Cases Analyzed"),
        (metrics.get("n_clusters", 0), "Judgment Clusters"),
        (f"{metrics.get('silhouette_score', 0):.3f}", "Silhouette Score"),
        (metrics.get("winner_algorithm", "N/A").upper(), "Clustering Algorithm"),
    ]
    for col, (val, label) in zip(cols, items):
        col.markdown(f"""
        <div class="metric-card">
            <h3>{val}</h3>
            <p>{label}</p>
        </div>
        """, unsafe_allow_html=True)


@st.cache_resource
def get_pipeline():
    from src.search_pipeline import SearchPipeline
    return SearchPipeline()


def render_search():
    st.markdown("### Search Similar Judgments")
    query = st.text_area(
        "Enter legal query or case description:",
        placeholder="e.g., bail application under IPC 302 murder where accused has no prior record",
        height=100,
        key="search_query"
    )

    col1, col2 = st.columns([3, 1])
    with col2:
        use_reranker = st.checkbox("Use cross-encoder reranker", value=True)

    if st.button("Search", type="primary", use_container_width=True):
        pipeline = get_pipeline()
        ok, msg = pipeline.health_check()
        if not ok:
            st.warning(msg)
            return

        with st.spinner("Searching..."):
            response = pipeline.search(query, top_k=5)

        if not response.success:
            st.warning(response.error)
            return

        st.success(
            f"Found {len(response.results)} candidates. "
            f"Latency: {response.latency_ms}ms"
        )

        for i, r in enumerate(response.results):
            exp = r.explanation
            score = r.score
            strength = "High" if score > 0.8 else "Medium" if score > 0.5 else "Low"

            with st.expander(
                f"#{r.rank} | Case {i+1} | "
                f"{strength} ({score:.3f})",
                expanded=(i == 0)
            ):
                c1, c2, c3 = st.columns(3)
                c1.metric("Court", r.case.get("court", "N/A"))
                c2.metric("Verdict", r.case.get("verdict", "N/A").replace("_", " ").title())
                c3.metric("Crime Type", r.case.get("case_type", "N/A"))

                # IPC tags
                ipc = r.case.get("ipc_sections", [])
                if ipc:
                    tags = " ".join(f'<span class="tag">IPC {s}</span>' for s in ipc[:8])
                    st.markdown(f"**IPC Sections:** {tags}", unsafe_allow_html=True)

                # Explanation reasons
                st.markdown("**Why similar:**")
                st.markdown(f"- {exp.get('similarity_reason')}")
                st.markdown(f"- {exp.get('key_differences')}")
                st.markdown(f"- {exp.get('verdict_analysis')}")

                # Case text
                st.markdown("**Judgment excerpt:**")
                st.markdown(f">{r.case['text'][:400]}...")


def render_cluster_map():
    st.markdown("### Judgment Cluster Map")

    with st.spinner("Loading cluster data..."):
        try:
            data = get_cluster_data()
        except Exception as e:
            st.error(f"Could not load cluster data: {e}")
            return

    cases = data["cases"]
    labels = data["labels"]
    coords = data["coords"]
    topics = data["topics"]

    df = pd.DataFrame({
        "x": [c[0] for c in coords],
        "y": [c[1] for c in coords],
        "cluster": [str(l) for l in labels],
        "title": [c.get("case_title", c.get("id", f"Case {i}"))[:50] for i, c in enumerate(cases)],
        "verdict": [c.get("verdict", "unknown") for c in cases],
        "court": [c.get("court", "unknown") for c in cases],
        "topic": [topics.get(str(l), f"Cluster {l}") for l in labels],
    })

    fig = px.scatter(
        df, x="x", y="y", color="cluster",
        hover_data=["title", "verdict", "court", "topic"],
        title="UMAP Projection of Judgment Embeddings",
        color_discrete_sequence=px.colors.qualitative.Set2,
        template="plotly_dark",
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter"),
        height=500,
    )
    fig.update_traces(marker=dict(size=6, opacity=0.7))
    st.plotly_chart(fig, use_container_width=True)


def render_verdict_distribution():
    st.markdown("### Verdict Distribution")
    try:
        data = get_cluster_data()
    except Exception as e:
        st.error(f"Data not found: {e}")
        return
        
    cases = data["cases"]

    verdicts = {}
    for c in cases:
        v = c.get("verdict", "unknown")
        verdicts[v] = verdicts.get(v, 0) + 1

    df = pd.DataFrame({
        "Verdict": list(verdicts.keys()),
        "Count": list(verdicts.values()),
    })
    df["Verdict"] = df["Verdict"].str.replace("_", " ").str.title()

    fig = px.bar(
        df, x="Verdict", y="Count", color="Verdict",
        color_discrete_sequence=["#e94560", "#0f3460", "#00c875", "#533483"],
        template="plotly_dark",
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter"),
        showlegend=False, height=350,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_gaps():
    st.markdown("### Legal Gaps & Inconsistencies")
    
    try:
        gaps = get_gaps()
    except Exception as e:
        st.error(f"Gaps data not found: {e}")
        return

    if not gaps:
        st.info("No verdict inconsistencies detected in the current dataset.")
        return

    st.markdown(f"**{len(gaps)} clusters** show opposing verdicts for similar cases:")

    for g in gaps[:10]:
        score_pct = int(g["inconsistency_score"] * 100)
        color = "#e94560" if score_pct > 30 else "#f39c12" if score_pct > 15 else "#00c875"

        st.markdown(f"""
        <div class="gap-card">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <strong style="color:{color};">Cluster {g['cluster_id']}</strong>
                    <span style="color:#a8a8b3;"> | {g['total_cases']} cases |
                    Inconsistency: {score_pct}%</span>
                </div>
                <div>
                    <span class="tag-green" style="display:inline-block;background:rgba(0,200,117,0.2);color:#00c875;padding:2px 8px;border-radius:6px;font-size:0.75rem;">
                        Granted: {g.get('bail_granted_count',0) + g.get('acquitted_count',0)}
                    </span>
                    <span class="tag" style="display:inline-block;background:rgba(233,69,96,0.2);color:#e94560;padding:2px 8px;border-radius:6px;font-size:0.75rem;">
                        Rejected: {g.get('bail_rejected_count',0) + g.get('convicted_count',0)}
                    </span>
                </div>
            </div>
            <p style="color:#a8a8b3;margin-top:8px;font-size:0.85rem;">
                {g.get('explanation', '')}
            </p>
        </div>
        """, unsafe_allow_html=True)


def render_eval_metrics():
    st.markdown("### Retrieval Evaluation Metrics")
    metrics_path = "data/processed/eval_metrics_retrieval.json"
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            rmetrics = json.load(f)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**FAISS Only**")
            fo = rmetrics.get("faiss_only", {})
            st.metric("MRR@5", f"{fo.get('MRR@5', 0):.4f}")
            st.metric("Precision@5", f"{fo.get('P@5', 0):.4f}")
            st.metric("NDCG@5", f"{fo.get('NDCG@5', 0):.4f}")

        with col2:
            st.markdown("**FAISS + Reranker**")
            fr = rmetrics.get("faiss_plus_reranker", {})
            imp = rmetrics.get("reranker_improvement", {})
            st.metric("MRR@5", f"{fr.get('MRR@5', 0):.4f}",
                      delta=f"{imp.get('MRR_delta', 0):+.4f}")
            st.metric("Precision@5", f"{fr.get('P@5', 0):.4f}",
                      delta=f"{imp.get('P5_delta', 0):+.4f}")
            st.metric("NDCG@5", f"{fr.get('NDCG@5', 0):.4f}",
                      delta=f"{imp.get('NDCG_delta', 0):+.4f}")
    else:
        st.info("Run `python -m src.eval_pipeline` to generate retrieval metrics.")


# ── Main App ──────────────────────────────────────────────────────────────
def main():
    render_header()

    # Sidebar
    with st.sidebar:
        st.markdown("## Navigation")
        page = st.radio(
            "Go to:",
            ["Search", "Cluster Map", "Legal Gaps", "Analytics"],
            label_visibility="collapsed"
        )
        st.markdown("---")
        st.markdown("**LexAI v3.2**")
        st.markdown("Built with LegalBERT + FAISS + DeBERTa")

    # Load metrics for header
    try:
        metrics = get_metrics()
        render_metrics(metrics)
    except Exception as e:
        st.warning(f"Run Phase 4 Colab notebook first to generate data. Error: {e}")
        return

    st.markdown("---")

    if page == "Search":
        render_search()
    elif page == "Cluster Map":
        render_cluster_map()
    elif page == "Legal Gaps":
        render_gaps()
    elif page == "Analytics":
        col1, col2 = st.columns(2)
        with col1:
            render_verdict_distribution()
        with col2:
            render_eval_metrics()


if __name__ == "__main__":
    main()
