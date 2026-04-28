"""Phase 6.6 — Search Pipeline Orchestrator.

SINGLE ENTRY POINT for all UI calls. No ML code in app.py.
"""
import json, numpy as np, faiss
from sentence_transformers import SentenceTransformer
from config import (
    EMBEDDING_MODEL, CASES_JSON_PATH, EMBEDDINGS_PATH,
    FAISS_INDEX_PATH, LABELS_PATH, TOPICS_PATH,
    GAPS_PATH, METRICS_PATH, COORDS_PATH,
    TOP_K_RETRIEVAL, TOP_K_RESULTS
)
from src.reranker import rerank
from src.explanation_engine import explain_similarity, explain_gap


_model = None
_index = None
_cases = None
_labels = None
_topics = None
_gaps = None
_metrics = None
_coords = None
_embeddings = None


def _load():
    global _model, _index, _cases, _labels, _topics
    global _gaps, _metrics, _coords, _embeddings

    if _cases is not None:
        return

    with open(CASES_JSON_PATH) as f:
        _cases = json.load(f)
    _embeddings = np.load(EMBEDDINGS_PATH).astype("float32")
    _index = faiss.read_index(FAISS_INDEX_PATH)
    _labels = np.load(LABELS_PATH)
    _model = SentenceTransformer(EMBEDDING_MODEL)

    try:
        with open(TOPICS_PATH) as f:
            _topics = json.load(f)
    except FileNotFoundError:
        _topics = {}

    try:
        with open(GAPS_PATH) as f:
            _gaps = json.load(f)
    except FileNotFoundError:
        _gaps = []

    try:
        with open(METRICS_PATH) as f:
            _metrics = json.load(f)
    except FileNotFoundError:
        _metrics = {}

    try:
        _coords = np.load(COORDS_PATH)
    except FileNotFoundError:
        _coords = np.zeros((len(_cases), 2))


def search(query: str, top_k: int = TOP_K_RESULTS,
           use_reranker: bool = True) -> dict:
    """
    Main search function. Returns results with explanations.
    """
    _load()

    # Encode query
    query_vec = _model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(query_vec)

    # FAISS retrieval
    distances, indices = _index.search(query_vec, TOP_K_RETRIEVAL)

    candidates = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(_cases):
            continue
        case = _cases[idx].copy()
        case["faiss_score"] = float(dist)
        case["cluster_id"] = int(_labels[idx])
        case["cluster_topic"] = _topics.get(str(int(_labels[idx])), "")
        candidates.append(case)

    # Rerank
    if use_reranker and candidates:
        results = rerank(query, candidates, top_k=top_k)
    else:
        results = sorted(candidates, key=lambda x: -x["faiss_score"])[:top_k]

    # Build a pseudo query case for explanation
    query_case = {
        "text": query, "ipc_sections": [], "court": "",
        "verdict": "", "case_type": "", "crime_type": "",
        "evidence_types": [], "entities": {},
    }

    # Add explanations
    for r in results:
        score = r.get("final_score", r.get("faiss_score", 0))
        r["explanation"] = explain_similarity(query_case, r, score)

    return {
        "query": query,
        "results": results,
        "total_candidates": len(candidates),
        "reranker_used": use_reranker,
    }


def get_gaps():
    _load()
    explained = []
    for g in _gaps:
        g_copy = g.copy()
        g_copy["explanation"] = explain_gap(g)
        explained.append(g_copy)
    return explained


def get_cluster_data():
    _load()
    return {
        "cases": _cases,
        "labels": _labels.tolist(),
        "coords": _coords.tolist(),
        "topics": _topics,
        "metrics": _metrics,
    }


def get_metrics():
    _load()
    return _metrics
