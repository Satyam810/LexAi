"""
Cross-encoder re-ranker — sits between FAISS and explanation engine.

Pipeline: FAISS top 50 → CrossEncoder rerank → top 5 → Explanation.

MODEL: cross-encoder/nli-deberta-v3-small
CRITICAL NOTE ON NLI MODEL SCORING:
  nli-deberta-v3-small is an NLI (Natural Language Inference) model.
  It outputs 3 scores per pair: [contradiction, neutral, entailment]
  We want the ENTAILMENT score (index 2) as our relevance signal.
  "This query entails this judgment" = relevant.
  If predict() returns a 1D array per pair, extract index 2.
  If predict() returns a single float (some versions), use directly.

WHY NLI OVER ms-marco:
  ms-marco trained on Bing web search clicks — wrong domain for law.
  NLI trained on entailment/contradiction — maps to legal reasoning.
  "Does this judgment's holding apply to this case?" = entailment.
"""

import numpy as np
from sentence_transformers import CrossEncoder
from config import RERANKER_MODEL

_model = None


def get_reranker() -> CrossEncoder:
    """Load cross-encoder once and reuse. Thread-safe for Streamlit."""
    global _model
    if _model is None:
        print(f"Loading cross-encoder: {RERANKER_MODEL}")
        print("(First run only — downloads and caches ~110MB)")
        _model = CrossEncoder(RERANKER_MODEL, max_length=512)
        print("Cross-encoder ready.")
    return _model


def _extract_relevance_score(raw_score) -> float:
    """
    Extract a single relevance float from the model's raw output.

    nli-deberta-v3-small outputs per pair:
      - numpy array of shape (3,): [contradiction, neutral, entailment]
      - OR a single float in some sentence-transformers versions

    We always want the ENTAILMENT score.
    If it's already a float, return it directly.
    """
    if isinstance(raw_score, (int, float)):
        return float(raw_score)
    arr = np.asarray(raw_score)
    if arr.ndim == 0:
        # Scalar wrapped in array
        return float(arr)
    if arr.shape == (3,):
        # Standard NLI output: [contradiction, neutral, entailment]
        return float(arr[2])  # index 2 = entailment score
    if arr.shape == (2,):
        # Some binary NLI variants: [not_entail, entail]
        return float(arr[1])
    # Fallback: take the max score
    return float(arr.max())


def rerank(query_text: str, candidates: list, top_k: int = 5) -> list:
    """
    Re-rank FAISS candidates using cross-encoder entailment scores.

    Args:
        query_text:  raw query string (NOT the embedding — raw text)
        candidates:  list of (case_dict, faiss_score) tuples
        top_k:       number of results to return

    Returns:
        list of (case_dict, entailment_score) sorted descending
    """
    if not candidates:
        return []

    model = get_reranker()

    # Build (query, candidate_text) pairs
    # Query gets 256 chars, candidate gets 400 chars
    pairs = [
        [query_text[:256], case["text"][:400]]
        for case, _ in candidates
    ]

    # Single forward pass over all pairs
    raw_scores = model.predict(pairs, show_progress_bar=False)

    # Extract entailment score for each pair
    scored = [
        (candidates[i][0], _extract_relevance_score(raw_scores[i]))
        for i in range(len(candidates))
    ]

    # Sort by relevance descending
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]
