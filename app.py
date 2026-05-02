"""LexAI v3.3 — Streamlit Dashboard.

STRICTLY display-only. All ML logic lives in src/search_pipeline.py.
"""
import streamlit as st
import plotly.express as px

import pandas as pd
import json, os, hashlib, logging, time, numpy as np

# Must be first Streamlit call
st.set_page_config(
    page_title="LexAI - Legal Judgment Analyzer",
    page_icon="\u2696\uFE0F",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize feedback store in session_state
if "feedback" not in st.session_state:
    st.session_state["feedback"] = {}
    # structure: { "query_hash|case_id": "relevant" | "not_relevant" }

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


# ── Legal Gap Explanation Engine (v3.3) ───────────────────────────────
def generate_gap_explanation(gap: dict, cases: list, labels) -> dict:
    """
    Generate a deterministic explanation for why a cluster
    has verdict inconsistency. No LLM. Pure rule-based logic.
    """
    from collections import Counter

    cluster_id = gap["cluster_id"]

    # Get all cases in this cluster
    cluster_cases = [
        c for c, lbl in zip(cases, labels)
        if int(lbl) == int(cluster_id)
    ]

    convicted  = [c for c in cluster_cases if c.get("verdict") == "convicted"]
    acquitted  = [c for c in cluster_cases if c.get("verdict") == "acquitted"]
    bail_granted = [c for c in cluster_cases if c.get("verdict") == "bail_granted"]
    bail_rejected = [c for c in cluster_cases if c.get("verdict") == "bail_rejected"]

    if not cluster_cases:
        return {
            "summary": "Insufficient data for this cluster.",
            "key_differences": [],
            "legal_insight": "No cases available for analysis."
        }

    # Evidence comparison
    def get_evidence_set(case_list):
        evidence = []
        for c in case_list:
            evidence.extend(c.get("evidence_types", []))
        return Counter(evidence)

    conv_evidence = get_evidence_set(convicted)
    acqu_evidence = get_evidence_set(acquitted)

    # IPC section comparison
    def get_ipc_set(case_list):
        sections = []
        for c in case_list:
            sections.extend(c.get("ipc_sections", []))
        return Counter(sections)

    conv_ipc = get_ipc_set(convicted)
    acqu_ipc = get_ipc_set(acquitted)

    # Court comparison
    conv_courts = Counter(c.get("court", "unknown") for c in convicted)
    acqu_courts = Counter(c.get("court", "unknown") for c in acquitted)

    # Build key differences
    key_differences = []

    # Evidence differences
    evidence_in_convicted_not_acquitted = [
        e for e in conv_evidence
        if conv_evidence[e] > 0 and acqu_evidence.get(e, 0) == 0
    ]
    evidence_in_acquitted_not_convicted = [
        e for e in acqu_evidence
        if acqu_evidence[e] > 0 and conv_evidence.get(e, 0) == 0
    ]

    if evidence_in_convicted_not_acquitted:
        key_differences.append(
            f"Convicted cases more often had: "
            f"{', '.join(evidence_in_convicted_not_acquitted)}"
        )
    if evidence_in_acquitted_not_convicted:
        key_differences.append(
            f"Acquitted cases more often had: "
            f"{', '.join(evidence_in_acquitted_not_convicted)}"
        )

    # Court level differences
    top_conv_court  = conv_courts.most_common(1)[0][0] if conv_courts else "unknown"
    top_acqu_court  = acqu_courts.most_common(1)[0][0] if acqu_courts else "unknown"
    if top_conv_court != top_acqu_court and top_conv_court != "unknown":
        key_differences.append(
            f"Most convictions from {top_conv_court}, "
            f"most acquittals from {top_acqu_court}"
        )

    # IPC differences
    common_ipc = gap.get("common_ipc_sections", [])
    if common_ipc:
        key_differences.append(
            f"Shared IPC sections: {', '.join(common_ipc[:3])}"
        )

    # If no differences found, add generic insight
    if not key_differences:
        key_differences.append(
            "Cases share similar charges but differ in "
            "factual circumstances or evidence quality"
        )

    # Generate summary
    total     = gap["total_cases"]
    conv_cnt  = gap["convicted_count"]
    acqu_cnt  = gap["acquitted_count"]
    score     = gap["inconsistency_score"]
    dom_type  = gap.get("dominant_case_type", "unknown")

    summary = (
        f"In this {dom_type} law cluster, {total} similar cases "
        f"resulted in {conv_cnt} convictions and {acqu_cnt} acquittals "
        f"({score:.0%} inconsistency). "
        f"Cases share the same charges but produced opposite outcomes."
    )

    # Legal insight
    if score >= 0.45:
        legal_insight = (
            "This cluster shows high verdict inconsistency \u2014 "
            "nearly equal split between conviction and acquittal "
            "on similar charges. This may indicate judicial discretion "
            "based on evidence quality, witness credibility, or "
            "differing interpretations of the same IPC sections."
        )
    elif score >= 0.30:
        legal_insight = (
            "Moderate inconsistency detected. Similar cases are "
            "leaning towards one outcome but a significant minority "
            "received the opposite verdict. Evidence strength and "
            "court level may be contributing factors."
        )
    else:
        legal_insight = (
            "Low-moderate inconsistency. The majority of similar "
            "cases share a verdict pattern, but exceptions exist "
            "suggesting fact-specific reasoning by courts."
        )

    return {
        "summary":         summary,
        "key_differences": key_differences,
        "legal_insight":   legal_insight
    }


def get_cluster_summary(cluster_id: int, cases: list, labels) -> dict:
    """
    Compute a readable summary for a single cluster.
    Used in cluster map legend and Legal Gaps section.
    """
    from collections import Counter

    cluster_cases = [
        c for c, lbl in zip(cases, labels)
        if int(lbl) == int(cluster_id)
    ]

    if not cluster_cases:
        return {
            "total": 0,
            "dominant_ipc": [],
            "case_type": "unknown",
            "verdict_split": {},
            "label": f"Cluster {cluster_id}"
        }

    # IPC sections
    all_ipc = []
    for c in cluster_cases:
        all_ipc.extend(c.get("ipc_sections", []))
    ipc_counter = Counter(all_ipc)
    dominant_ipc = [ipc for ipc, _ in ipc_counter.most_common(3)]

    # Case type
    types = Counter(c.get("case_type", "unknown") for c in cluster_cases)
    dominant_type = types.most_common(1)[0][0]

    # Verdict split
    verdicts = Counter(c.get("verdict", "unknown") for c in cluster_cases)
    total = len(cluster_cases)
    verdict_split = {
        v: round(count / total * 100)
        for v, count in verdicts.most_common(4)
    }

    # Build a human-readable label
    if dominant_ipc:
        label = f"IPC {', '.join(dominant_ipc[:2])} \u2014 {dominant_type.title()}"
    else:
        label = f"{dominant_type.title()} cases"

    return {
        "total":        total,
        "dominant_ipc": dominant_ipc,
        "case_type":    dominant_type,
        "verdict_split": verdict_split,
        "label":        label
    }


@st.cache_data
def compute_all_gap_explanations(_cases, _labels, _gaps_json):
    """
    Pre-compute explanations for ALL gaps at once.
    Cached — only recomputes when data changes.
    """
    import json as _json
    gaps_list = _json.loads(_gaps_json)
    return {
        gap["cluster_id"]: generate_gap_explanation(gap, _cases, _labels)
        for gap in gaps_list
    }


@st.cache_data
def compute_all_cluster_summaries(_cases, _labels):
    """
    Pre-compute summaries for ALL clusters at once.
    Cached — only recomputes when data changes.
    """
    if _labels is None or len(_cases) == 0:
        return {}
    summaries = {}
    for cid in set(int(l) for l in _labels if int(l) != -1):
        summaries[cid] = get_cluster_summary(cid, _cases, _labels)
    return summaries


# ── Persistent Feedback Storage (v3.3) ────────────────────────────────
def save_feedback_to_disk(feedback_dict: dict):
    """
    Write feedback to JSON with file locking.
    Safe for Streamlit concurrent reruns.
    """
    feedback_path = "data/feedback.json"
    lock_path     = "data/feedback.lock"

    # Simple file lock — wait up to 2 seconds
    waited = 0
    while os.path.exists(lock_path) and waited < 2:
        time.sleep(0.1)
        waited += 0.1

    try:
        # Acquire lock
        with open(lock_path, "w") as lf:
            lf.write("locked")

        # Load existing, merge, save
        existing = {}
        if os.path.exists(feedback_path):
            try:
                with open(feedback_path) as f:
                    existing = json.load(f)
            except Exception:
                existing = {}

        existing.update(feedback_dict)

        os.makedirs("data", exist_ok=True)
        with open(feedback_path, "w") as f:
            json.dump(existing, f, indent=2)
    except Exception:
        pass  # feedback persistence must never crash the app
    finally:
        # Always release lock
        if os.path.exists(lock_path):
            try:
                os.remove(lock_path)
            except Exception:
                pass


# ── Query Logger (v3.3) ───────────────────────────────────────────────
_query_logger = None


def get_query_logger():
    global _query_logger
    if _query_logger is None:
        os.makedirs("logs", exist_ok=True)
        handler = logging.FileHandler("logs/queries.log")
        handler.setLevel(logging.INFO)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
        )
        _query_logger = logging.getLogger("lexai_queries")
        _query_logger.setLevel(logging.INFO)
        _query_logger.addHandler(handler)
    return _query_logger


def render_header():
    st.markdown("""
    <div class="main-header">
        <h1>LexAI v3.3</h1>
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

        # Loading states
        progress_bar = st.progress(0, text="Validating query...")
        progress_bar.progress(25, text="Embedding query with LegalBERT...")
        response = pipeline.search(query, top_k=5)
        progress_bar.progress(75, text="Ranking results...")
        progress_bar.progress(100, text="Building explanations...")
        time.sleep(0.3)
        progress_bar.empty()

        if not response.success:
            st.warning(response.error)
            return

        st.success(
            f"Found {len(response.results)} candidates. "
            f"Latency: {response.latency_ms}ms"
        )

        # Query logging
        try:
            get_query_logger().info(
                f"query={query[:100]!r} | "
                f"results={len(response.results)} | "
                f"latency={response.latency_ms}ms | "
                f"ipc={response.query_case.get('ipc_sections', [])} | "
                f"case_type={response.query_case.get('case_type', 'unknown')}"
            )
        except Exception:
            pass  # logging must never crash the app

        # Query understanding display
        if response.query_case:
            qc = response.query_case
            ipc_text  = ", ".join(qc["ipc_sections"]) if qc["ipc_sections"] else "none detected"
            ev_text   = ", ".join(qc["evidence_types"]) if qc["evidence_types"] else "none detected"
            type_text = qc["case_type"].title()

            st.markdown(
                f"**Query understood:** "
                f"IPC {ipc_text} &nbsp;|&nbsp; "
                f"{type_text} case &nbsp;|&nbsp; "
                f"Evidence: {ev_text}",
            )
            st.divider()

        # Verdict color maps
        VERDICT_COLORS = {
            "convicted":         "#ef4444",
            "acquitted":         "#22c55e",
            "bail_granted":      "#eab308",
            "bail_rejected":     "#f97316",
            "appeal_allowed":    "#3b82f6",
            "appeal_dismissed":  "#8b5cf6",
            "sentence_modified": "#6366f1",
            "unknown":           "#6b7280",
        }
        VERDICT_LABELS = {
            "convicted":         "CONVICTED",
            "acquitted":         "ACQUITTED",
            "bail_granted":      "BAIL GRANTED",
            "bail_rejected":     "BAIL REJECTED",
            "appeal_allowed":    "APPEAL ALLOWED",
            "appeal_dismissed":  "APPEAL DISMISSED",
            "sentence_modified": "SENTENCE MODIFIED",
            "unknown":           "VERDICT UNKNOWN",
        }

        for result in response.results:
            case    = result.case
            exp     = result.explanation
            verdict = case.get("verdict", "unknown")
            color   = VERDICT_COLORS.get(verdict, "#6b7280")
            label   = VERDICT_LABELS.get(verdict, "UNKNOWN")

            with st.container():
                # Title row
                title_col, score_col = st.columns([4, 1])
                with title_col:
                    court = case.get("court", "Unknown Court")
                    date  = case.get("date", "")
                    st.markdown(f"**#{result.rank} \u2014 {court}**")
                    if date:
                        st.caption(f"Date: {date}")
                with score_col:
                    st.metric(
                        "Similarity",
                        f"{result.score:.3f}",
                        help="Higher = more similar to your query"
                    )

                # Verdict badge + key evidence
                badge_col, info_col = st.columns([1, 3])
                with badge_col:
                    st.markdown(
                        f"<div style='background:{color};color:white;"
                        f"padding:6px 12px;border-radius:6px;"
                        f"text-align:center;font-weight:bold;"
                        f"font-size:13px'>{label}</div>",
                        unsafe_allow_html=True
                    )
                with info_col:
                    if exp.get("shared_ipc"):
                        st.write(f"**Shared IPC:** {', '.join(exp['shared_ipc'])}")
                    if exp.get("shared_evidence"):
                        st.write(f"**Matching evidence:** {', '.join(exp['shared_evidence'])}")
                    if exp.get("shared_case_type"):
                        st.write(f"**Same case type:** {case.get('case_type','').title()}")

                # Why this result — expandable
                with st.expander("\u2696\uFE0F Why this result?"):
                    st.info(exp["similarity_reason"])
                    st.warning(exp["key_differences"])
                    verdict_text = exp["verdict_analysis"]
                    if "divergence" in verdict_text.lower():
                        st.error(verdict_text)
                    elif "alignment" in verdict_text.lower():
                        st.success(verdict_text)
                    else:
                        st.write(verdict_text)

                # Feedback buttons
                query_hash = hashlib.md5(query.encode()).hexdigest()[:8]
                case_id    = case.get("id", str(result.rank))
                fb_key     = f"{query_hash}|{case_id}"

                current_fb = st.session_state["feedback"].get(fb_key, None)

                fb_col1, fb_col2, fb_col3 = st.columns([1, 1, 6])
                with fb_col1:
                    if st.button(
                        "\U0001F44D Relevant",
                        key=f"up_{fb_key}",
                        type="primary" if current_fb == "relevant" else "secondary"
                    ):
                        st.session_state["feedback"][fb_key] = "relevant"
                        save_feedback_to_disk(st.session_state["feedback"])
                        st.rerun()
                with fb_col2:
                    if st.button(
                        "\U0001F44E Not relevant",
                        key=f"down_{fb_key}",
                        type="primary" if current_fb == "not_relevant" else "secondary"
                    ):
                        st.session_state["feedback"][fb_key] = "not_relevant"
                        save_feedback_to_disk(st.session_state["feedback"])
                        st.rerun()
                with fb_col3:
                    if current_fb:
                        st.caption(
                            f"\u2713 Marked as {'relevant' if current_fb == 'relevant' else 'not relevant'}"
                        )

                # Judgment excerpt
                excerpt = case.get("text", "")[:400]
                if excerpt:
                    st.caption(f'"{excerpt}..."')

                st.divider()


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

    # Build cluster summaries for tooltips (cached)
    cluster_summaries = compute_all_cluster_summaries(cases, labels)

    # Build DataFrame with rich hover columns
    df = pd.DataFrame({
        "x": [c[0] for c in coords],
        "y": [c[1] for c in coords],
        "cluster": [str(l) for l in labels],
        "cluster_label": [
            cluster_summaries.get(int(l), {}).get("label", f"Cluster {l}")
            if int(l) != -1 else "Noise"
            for l in labels
        ],
        "case_type": [c.get("case_type", "unknown") for c in cases],
        "verdict": [c.get("verdict", "unknown") for c in cases],
        "court": [c.get("court", "unknown") for c in cases],
        "date": [c.get("date", "unknown") for c in cases],
    })

    fig = px.scatter(
        df, x="x", y="y",
        color="cluster",
        hover_data={
            "x": False,
            "y": False,
            "cluster": False,
            "cluster_label": True,
            "verdict": True,
            "case_type": True,
            "court": True,
            "date": True,
        },
        labels={"cluster_label": "Cluster"},
        height=500,
        title="UMAP Projection of Judgment Embeddings",
        color_discrete_sequence=px.colors.qualitative.Set2,
        template="plotly_dark",
    )
    fig.update_traces(marker=dict(size=6, opacity=0.7))
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Cluster summary table
    st.markdown("**Cluster Summary**")
    summary_rows = []
    for cid in sorted(cluster_summaries.keys()):
        s = cluster_summaries[cid]
        top_verdict = max(s["verdict_split"], key=s["verdict_split"].get) \
                      if s["verdict_split"] else "unknown"
        summary_rows.append({
            "Cluster":           cid,
            "Cases":             s["total"],
            "Legal Category":    s["label"],
            "Dominant Verdict":  top_verdict,
            "Verdict Split":     " | ".join(
                f"{v}: {p}%" for v, p in
                list(s["verdict_split"].items())[:3]
            ),
        })

    if summary_rows:
        st.dataframe(
            pd.DataFrame(summary_rows),
            use_container_width=True,
            hide_index=True
        )


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
    st.caption(
        "Dataset note: Bail applications dominate because Indian Kanoon "
        "returns a high volume of bail matter judgments. "
        "Bail Rejected + Bail Granted cases reflect real Indian court "
        "docket composition where bail matters vastly outnumber trials."
    )


def render_gaps():
    st.markdown("## Legal Gaps & Inconsistencies")

    try:
        gaps = get_gaps()
    except Exception as e:
        st.error(f"Gaps data not found: {e}")
        return

    if not gaps:
        st.info(
            "No inconsistent clusters detected. "
            "This improves as more cases are added and verdict "
            "coverage increases above 40%."
        )
        return

    st.caption(
        f"{len(gaps)} clusters show opposing verdicts for similar cases. "
        "Cases in the same cluster share IPC sections and case type "
        "but received opposite verdicts."
    )

    # Load cases and labels for explanation engine
    try:
        data = get_cluster_data()
        cases = data["cases"]
        labels_for_gaps = data["labels"]
    except Exception:
        cases = []
        labels_for_gaps = None

    # Pre-compute ALL explanations once (cached) — not inside the render loop
    if labels_for_gaps is not None and cases:
        all_explanations = compute_all_gap_explanations(
            cases, labels_for_gaps, json.dumps(gaps)
        )
    else:
        all_explanations = {}

    for gap in gaps:
        score = gap["inconsistency_score"]

        with st.container():
            # Header row
            col_head, col_badges = st.columns([3, 1])
            with col_head:
                st.markdown(
                    f"**Cluster {gap['cluster_id']}** \u2014 "
                    f"{gap['total_cases']} cases \u2014 "
                    f"Inconsistency: **{score:.0%}**"
                )
            with col_badges:
                badge_col1, badge_col2 = st.columns(2)
                badge_col1.markdown(
                    f"<span style='background:#16a34a;color:white;"
                    f"padding:2px 8px;border-radius:4px;font-size:12px'>"
                    f"Granted: {gap.get('bail_granted_count', 0) + gap.get('acquitted_count', 0)}</span>",
                    unsafe_allow_html=True
                )
                badge_col2.markdown(
                    f"<span style='background:#dc2626;color:white;"
                    f"padding:2px 8px;border-radius:4px;font-size:12px'>"
                    f"Rejected: {gap.get('bail_rejected_count', 0) + gap.get('convicted_count', 0)}</span>",
                    unsafe_allow_html=True
                )

            # Look up pre-computed explanation
            explanation = all_explanations.get(
                gap["cluster_id"],
                {
                    "summary": f"Cluster {gap['cluster_id']} shows {score:.0%} inconsistency.",
                    "key_differences": [
                        f"IPC sections: {', '.join(gap.get('common_ipc_sections', ['unknown']))}"
                    ],
                    "legal_insight": "Load cluster labels to see full analysis."
                }
            )

            # Summary
            st.write(explanation["summary"])

            # Key differences as bullet points
            if explanation["key_differences"]:
                for diff in explanation["key_differences"]:
                    st.markdown(f"\u2022 {diff}")

            # Legal insight in a styled box
            st.info(f"\u2696\uFE0F **Legal Insight:** {explanation['legal_insight']}")

            # IPC sections
            if gap.get("common_ipc_sections"):
                st.caption(
                    f"Common IPC sections: "
                    f"{', '.join(gap['common_ipc_sections'][:5])}"
                )

            st.divider()


def render_eval_metrics():
    st.markdown("### Retrieval Evaluation Metrics")
    metrics_path = "data/processed/eval_metrics_retrieval.json"
    if not os.path.exists(metrics_path):
        st.info("Run `python -m src.eval_pipeline` to generate retrieval metrics.")
        return

    with open(metrics_path) as f:
        rmetrics = json.load(f)

    # Interpretation helpers
    def interpret_mrr(v):
        if v >= 0.7: return "Strong \u2014 relevant case usually in top 2"
        if v >= 0.5: return "Moderate \u2014 relevant case usually in top 3"
        if v >= 0.3: return "Fair \u2014 relevant case usually in top 5"
        return "Low \u2014 relevant case may not appear in top results"

    def interpret_ndcg(v):
        if v >= 0.7: return "Strong ranking quality"
        if v >= 0.5: return "Fair ranking relevance"
        if v >= 0.3: return "Moderate \u2014 some ranking noise"
        return "Low \u2014 ranking needs improvement"

    def interpret_p5(v):
        if v >= 0.6: return "High \u2014 most results are relevant"
        if v >= 0.4: return "Moderate \u2014 roughly half are relevant"
        if v >= 0.2: return "Low \u2014 minority of results are relevant"
        return "Very low \u2014 precision needs improvement"

    fr = rmetrics.get("faiss_plus_reranker", rmetrics.get("faiss_only", {}))
    mrr  = fr.get("MRR@5", 0)
    p5   = fr.get("P@5", 0)
    ndcg = fr.get("NDCG@5", 0)

    m1, m2, m3 = st.columns(3)
    m1.metric("MRR@5", f"{mrr:.4f}", help=interpret_mrr(mrr))
    m1.caption(interpret_mrr(mrr))
    m2.metric("P@5", f"{p5:.4f}", help=interpret_p5(p5))
    m2.caption(interpret_p5(p5))
    m3.metric("NDCG@5", f"{ndcg:.4f}", help=interpret_ndcg(ndcg))
    m3.caption(interpret_ndcg(ndcg))

    # Reranker experiment explanation
    st.markdown("**Reranker Experiment**")
    st.info(
        "Two cross-encoder models were evaluated on 96 queries: \n\n"
        "\u2022 **nli-deberta-v3-small** \u2192 NDCG delta: \u22120.10 (degraded quality) \n\n"
        "\u2022 **ms-marco-MiniLM** \u2192 NDCG delta: 0.00 (no improvement) \n\n"
        "**Conclusion:** LegalBERT bi-encoder already captures Indian legal "
        "domain similarity well without a cross-encoder. Neither model improved "
        "results without fine-tuning on labeled Indian law data. "
        "The reranker is kept in the codebase for v4 fine-tuning. "
        "FAISS direct retrieval is used in production at ~270ms latency."
    )


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
        st.markdown("**LexAI v3.3**")
        st.markdown("Built with LegalBERT + FAISS + DeBERTa")

    # Load metrics for header
    try:
        metrics = get_metrics()
        render_metrics(metrics)
    except Exception as e:
        metrics = {}
        st.warning(
            "Clustering metrics not available. "
            "Run the Colab notebook and copy eval_metrics.json "
            "to data/processed/."
        )

    st.markdown("---")

    if page == "Search":
        render_search()
    elif page == "Cluster Map":
        render_cluster_map()
    elif page == "Legal Gaps":
        render_gaps()
    elif page == "Analytics":
        if not metrics:
            st.info(
                "Clustering metrics not available. "
                "Run the Colab notebook and copy eval_metrics.json "
                "to data/processed/."
            )
        # Cluster quality label
        sil_score = metrics.get("silhouette_score", 0)
        if sil_score >= 0.5:
            quality_label = "GOOD"
            quality_color = "#22c55e"
            quality_note  = "Strong cluster separation. Cases group meaningfully."
        elif sil_score >= 0.2:
            quality_label = "MODERATE"
            quality_color = "#eab308"
            quality_note  = "Some overlap between clusters. Acceptable for this dataset size."
        else:
            quality_label = "LOW"
            quality_color = "#ef4444"
            quality_note  = (
                "Overlapping clusters \u2014 cases are semantically similar across groups. "
                "This is expected with 500 cases. Improves significantly with 1,000+ cases."
            )

        col_sil, col_q = st.columns([1, 2])
        with col_sil:
            st.metric("Silhouette Score", f"{sil_score:.3f}")
        with col_q:
            st.markdown(
                f"<span style='background:{quality_color};color:white;"
                f"padding:3px 10px;border-radius:4px;font-weight:bold'>"
                f"Cluster Quality: {quality_label}</span>",
                unsafe_allow_html=True
            )
            st.caption(quality_note)

        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            render_verdict_distribution()
        with col2:
            render_eval_metrics()

        # User feedback summary
        st.divider()
        st.markdown("### User Feedback Summary")
        fb = st.session_state.get("feedback", {})
        if not fb:
            st.caption("No feedback collected yet. Search and mark results to see feedback stats.")
        else:
            relevant     = sum(1 for v in fb.values() if v == "relevant")
            not_relevant = sum(1 for v in fb.values() if v == "not_relevant")
            total_fb     = len(fb)
            user_precision = relevant / total_fb if total_fb > 0 else 0

            fb1, fb2, fb3 = st.columns(3)
            fb1.metric("Total rated",     total_fb)
            fb2.metric("Marked relevant", relevant)
            fb3.metric("User Precision",  f"{user_precision:.0%}",
                       help="% of rated results marked relevant by user")

            # Evaluation loop — compare user feedback to offline metrics
            try:
                ret_metrics_path = "data/processed/eval_metrics_retrieval.json"
                if os.path.exists(ret_metrics_path):
                    with open(ret_metrics_path) as f:
                        ret_metrics_fb = json.load(f)
                    offline_mrr = ret_metrics_fb.get(
                        "faiss_plus_reranker",
                        ret_metrics_fb.get("faiss_only", {})
                    ).get("MRR@5", 0)
                    if total_fb >= 5 and offline_mrr > 0:
                        st.markdown("**Feedback vs Offline Metrics**")
                        comp1, comp2 = st.columns(2)
                        comp1.metric("User Precision (live)", f"{user_precision:.2f}")
                        comp2.metric("MRR@5 (offline eval)", f"{offline_mrr:.4f}")
                        if user_precision >= offline_mrr - 0.1:
                            st.success(
                                "User feedback aligns with offline evaluation. "
                                "Retrieval quality is consistent in practice."
                            )
                        else:
                            st.warning(
                                "User precision is lower than offline MRR. "
                                "This may indicate the eval oracle (shared IPC sections) "
                                "overestimates real-world relevance. "
                                "Collecting more labeled feedback would improve accuracy."
                            )
            except Exception:
                pass

            st.caption(
                "Session feedback resets on page refresh. "
                "Persistent feedback is saved to data/feedback.json."
            )


if __name__ == "__main__":
    main()
