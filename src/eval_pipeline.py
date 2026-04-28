"""Phase 5.5 — Self-supervised Retrieval Evaluation.

Compares FAISS-only vs FAISS+reranker using MRR@5, Precision@5, NDCG@5.
Self-supervised: uses each case as its own query, expects same-cluster
cases to be relevant (no labeled data needed).
"""
import json, numpy as np, faiss, math
from sentence_transformers import SentenceTransformer, CrossEncoder
from config import (
    EMBEDDING_MODEL, RERANKER_MODEL,
    CASES_JSON_PATH, EMBEDDINGS_PATH, FAISS_INDEX_PATH,
    LABELS_PATH, RETRIEVAL_METRICS_PATH,
    TOP_K_RETRIEVAL, EVAL_SAMPLE_SIZE, EVAL_TOP_K
)


def load_assets():
    with open(CASES_JSON_PATH) as f:
        cases = json.load(f)
    embeddings = np.load(EMBEDDINGS_PATH).astype("float32")
    labels = np.load(LABELS_PATH)
    index = faiss.read_index(FAISS_INDEX_PATH)
    return cases, embeddings, labels, index


def dcg(relevances, k):
    return sum(r / math.log2(i + 2) for i, r in enumerate(relevances[:k]))


def ndcg_at_k(relevances, k):
    actual = dcg(relevances, k)
    ideal = dcg(sorted(relevances, reverse=True), k)
    return actual / ideal if ideal > 0 else 0.0


def evaluate_retrieval(cases, embeddings, labels, index, use_reranker=False, k=EVAL_TOP_K):
    if use_reranker:
        reranker = CrossEncoder(RERANKER_MODEL)

    mrr_scores = []
    precision_scores = []
    ndcg_scores = []

    # Sample queries for speed
    step = max(1, len(cases) // EVAL_SAMPLE_SIZE)
    query_indices = list(range(0, len(cases), step))[:EVAL_SAMPLE_SIZE]

    for qi in query_indices:
        query_label = labels[qi]
        if query_label == -1:
            continue

        # FAISS search
        query_vec = embeddings[qi:qi+1].copy()
        faiss.normalize_L2(query_vec)
        distances, indices = index.search(query_vec, TOP_K_RETRIEVAL)
        retrieved_ids = [int(idx) for idx in indices[0] if idx != qi][:TOP_K_RETRIEVAL]

        if use_reranker:
            query_text = cases[qi]["text"][:500]
            pairs = [(query_text, cases[rid]["text"][:500]) for rid in retrieved_ids]
            scores = reranker.predict(pairs)
            scores = np.array(scores).flatten()
            score_list = scores.tolist()
            ranked = sorted(zip(retrieved_ids, score_list), key=lambda x: -x[1])
            retrieved_ids = [rid for rid, _ in ranked]

        top_k_ids = retrieved_ids[:k]

        # Relevance: same cluster = 1, different = 0
        relevances = [1 if labels[rid] == query_label else 0 for rid in top_k_ids]

        # MRR
        rr = 0.0
        for i, rel in enumerate(relevances):
            if rel == 1:
                rr = 1.0 / (i + 1)
                break
        mrr_scores.append(rr)

        # Precision@k
        precision_scores.append(sum(relevances) / k if k > 0 else 0)

        # NDCG@k
        ndcg_scores.append(ndcg_at_k(relevances, k))

    return {
        "mrr": round(np.mean(mrr_scores), 4),
        "precision": round(np.mean(precision_scores), 4),
        "ndcg": round(np.mean(ndcg_scores), 4),
        "n_queries": len(mrr_scores),
    }


if __name__ == "__main__":
    print("Loading assets...")
    cases, embeddings, labels, index = load_assets()

    print(f"\n{'='*55}")
    print("PHASE 5.5 — RETRIEVAL EVALUATION (self-supervised)")
    print(f"{'='*55}")

    print("\n[1/2] FAISS-only retrieval...")
    faiss_metrics = evaluate_retrieval(cases, embeddings, labels, index,
                                        use_reranker=False)
    print(f"  MRR@5:       {faiss_metrics['mrr']:.4f}")
    print(f"  Precision@5: {faiss_metrics['precision']:.4f}")
    print(f"  NDCG@5:      {faiss_metrics['ndcg']:.4f}")

    print("\n[2/2] FAISS + cross-encoder reranker...")
    reranked_metrics = evaluate_retrieval(cases, embeddings, labels, index,
                                           use_reranker=True)
    print(f"  MRR@5:       {reranked_metrics['mrr']:.4f}")
    print(f"  Precision@5: {reranked_metrics['precision']:.4f}")
    print(f"  NDCG@5:      {reranked_metrics['ndcg']:.4f}")

    # Improvement
    mrr_delta = reranked_metrics["mrr"] - faiss_metrics["mrr"]
    p_delta = reranked_metrics["precision"] - faiss_metrics["precision"]
    ndcg_delta = reranked_metrics["ndcg"] - faiss_metrics["ndcg"]

    print(f"\n--- IMPROVEMENT (reranker vs FAISS-only) ---")
    print(f"  MRR:       {mrr_delta:+.4f}")
    print(f"  Precision: {p_delta:+.4f}")
    print(f"  NDCG:      {ndcg_delta:+.4f}")

    results = {
        "faiss_only": faiss_metrics,
        "faiss_plus_reranker": reranked_metrics,
        "improvement": {
            "mrr_delta": round(mrr_delta, 4),
            "precision_delta": round(p_delta, 4),
            "ndcg_delta": round(ndcg_delta, 4),
        }
    }

    with open(RETRIEVAL_METRICS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {RETRIEVAL_METRICS_PATH}")
