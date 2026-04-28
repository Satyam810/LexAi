"""Phase 6.5 — Cross-Encoder Reranker.

Takes FAISS top-K candidates and reranks with NLI-DeBERTa.
"""
from sentence_transformers import CrossEncoder
from config import RERANKER_MODEL


_reranker = None

def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


def rerank(query_text: str, candidates: list, top_k: int = 5) -> list:
    """
    Rerank candidates using cross-encoder.

    Args:
        query_text: The search query text
        candidates: List of dicts with at least 'text' and 'faiss_score'
        top_k: Number of results to return

    Returns:
        List of candidates with added 'rerank_score' and 'final_score', sorted
    """
    if not candidates:
        return []

    reranker = get_reranker()
    pairs = [(query_text[:500], c["text"][:500]) for c in candidates]
    import numpy as np
    scores = np.array(reranker.predict(pairs)).flatten().tolist()

    for c, score in zip(candidates, scores):
        c["rerank_score"] = score
        # Combined score: 0.4 * FAISS + 0.6 * reranker (reranker is more precise)
        c["final_score"] = 0.4 * c.get("faiss_score", 0) + 0.6 * score

    ranked = sorted(candidates, key=lambda x: x["final_score"], reverse=True)
    return ranked[:top_k]
