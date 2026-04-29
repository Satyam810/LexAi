"""
Retrieval evaluation — self-supervised, no labeled data required.

Relevance oracle: two cases are "relevant" if they share
at least one IPC section AND the same case_type.

Metrics:
  MRR@5  — Mean Reciprocal Rank
  P@5    — Precision at 5
  NDCG@5 — Normalized Discounted Cumulative Gain

Output: data/processed/eval_metrics_retrieval.json (always complete)
Run:    python src/eval_pipeline.py
"""

import json, numpy as np, faiss, math, time, random, logging
from pathlib import Path
from sentence_transformers import SentenceTransformer, CrossEncoder

from config import (
    CASES_JSON_PATH, EMBEDDINGS_PATH, FAISS_INDEX_PATH,
    RETRIEVAL_METRICS_PATH, EMBEDDING_MODEL, RERANKER_MODEL,
    TOP_K_RETRIEVAL, TOP_K_RESULTS, EVAL_SAMPLE_SIZE, EVAL_TOP_K,
    MAX_TEXT_LENGTH
)
from src.reranker import _extract_relevance_score

log = logging.getLogger(__name__)


# ── Relevance oracle ─────────────────────────────────────────────────────────

def is_relevant(query_case: dict, candidate_case: dict) -> bool:
    """
    Self-supervised oracle: shared IPC section + same case type.
    Cases with no IPC sections cannot be evaluated and return False.
    """
    q_ipc  = set(query_case.get("ipc_sections", []))
    c_ipc  = set(candidate_case.get("ipc_sections", []))
    q_type = query_case.get("case_type", "")
    c_type = candidate_case.get("case_type", "")

    if not q_ipc or not c_ipc:
        return False
    return bool(q_ipc & c_ipc) and (q_type == c_type)


# ── IR metrics ───────────────────────────────────────────────────────────────

def reciprocal_rank(relevant_flags: list) -> float:
    """1/rank of the first relevant result. 0 if none found."""
    for i, flag in enumerate(relevant_flags):
        if flag:
            return 1.0 / (i + 1)
    return 0.0


def precision_at_k(relevant_flags: list, k: int) -> float:
    """Fraction of top-k results that are relevant."""
    return sum(relevant_flags[:k]) / k if k > 0 else 0.0


def ndcg_at_k(relevant_flags: list, k: int) -> float:
    """NDCG@k with binary relevance."""
    if k == 0:
        return 0.0
    dcg  = sum(flag / math.log2(i + 2) for i, flag in enumerate(relevant_flags[:k]))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(sum(relevant_flags[:k]), k)))
    return dcg / idcg if idcg > 0 else 0.0


# ── Main evaluation ──────────────────────────────────────────────────────────

def run_retrieval_eval() -> dict:
    """
    Run full self-supervised retrieval evaluation.
    Always writes a complete JSON even if evaluation finds 0 valid queries.

    Returns:
        dict with all evaluation results (also saved to RETRIEVAL_METRICS_PATH)
    """
    print("=" * 60)
    print("LEXAI RETRIEVAL EVALUATION")
    print("=" * 60)

    # ── Load assets ────────────────────────────────────────────────────────
    print("\nLoading cases and index...")
    with open(CASES_JSON_PATH, encoding="utf-8") as f:
        cases = json.load(f)

    embeddings  = np.load(EMBEDDINGS_PATH).astype("float32")
    index       = faiss.read_index(FAISS_INDEX_PATH)
    embed_model = SentenceTransformer(EMBEDDING_MODEL)
    rerank_model = CrossEncoder(RERANKER_MODEL, max_length=512)

    print(f"  Cases:      {len(cases)}")
    print(f"  Embeddings: {embeddings.shape}")
    print(f"  FAISS:      {index.ntotal} vectors")
    print(f"  Embed model: {EMBEDDING_MODEL}")
    print(f"  Rerank model: {RERANKER_MODEL}")

    # ── Select evaluable queries ───────────────────────────────────────────
    evaluable = [
        (i, c) for i, c in enumerate(cases)
        if c.get("ipc_sections")
    ]

    print(f"\n  Cases with IPC sections: {len(evaluable)} / {len(cases)}")

    # Build result skeleton BEFORE eval loop — ensures JSON is always complete
    results = {
        "evaluation_summary": {
            "total_cases":       len(cases),
            "evaluable_cases":   len(evaluable),
            "queries_evaluated": 0,
            "queries_skipped":   0,
            "eval_k":            EVAL_TOP_K,
            "relevance_oracle":  "shared_ipc_section_and_same_case_type",
        },
        "faiss_only": {
            "MRR@5": 0.0, "P@5": 0.0, "NDCG@5": 0.0
        },
        "faiss_plus_reranker": {
            "MRR@5": 0.0, "P@5": 0.0, "NDCG@5": 0.0
        },
        "reranker_improvement": {
            "MRR_delta": 0.0, "P5_delta": 0.0, "NDCG_delta": 0.0
        },
        "latency": {
            "avg_query_ms": 0.0, "p95_query_ms": 0.0
        },
        "model_info": {
            "embedding_model": EMBEDDING_MODEL,
            "reranker_model":  RERANKER_MODEL,
        }
    }

    if len(evaluable) < 10:
        print("\n⚠️  Fewer than 10 evaluable cases. Saving skeleton metrics.")
        _save_results(results)
        return results

    # ── Evaluation loop ────────────────────────────────────────────────────
    random.seed(42)
    sample_size = min(EVAL_SAMPLE_SIZE, len(evaluable))
    eval_sample = random.sample(evaluable, sample_size)
    print(f"  Eval sample: {sample_size} queries\n")

    faiss_mrr, faiss_p5, faiss_ndcg = [], [], []
    rerank_mrr, rerank_p5, rerank_ndcg = [], [], []
    latencies = []
    skipped = 0

    for eval_idx, (case_idx, query_case) in enumerate(eval_sample):
        if eval_idx % 20 == 0:
            print(f"  [{eval_idx}/{sample_size}] evaluating...")

        # Embed query
        q_words = query_case["text"].split()[:MAX_TEXT_LENGTH]
        q_emb = embed_model.encode(
            [" ".join(q_words)], convert_to_numpy=True
        ).astype("float32")
        faiss.normalize_L2(q_emb)

        # FAISS search — exclude the query case itself
        t0 = time.time()
        scores_arr, idxs = index.search(q_emb, TOP_K_RETRIEVAL + 1)
        faiss_candidates = [
            (cases[idx], float(sc))
            for idx, sc in zip(idxs[0], scores_arr[0])
            if idx != case_idx and 0 <= idx < len(cases)
        ][:TOP_K_RETRIEVAL]
        faiss_ms = (time.time() - t0) * 1000

        if not faiss_candidates:
            skipped += 1
            continue

        # Check if ANY candidate is relevant — if not, skip (unevaluable)
        all_flags = [is_relevant(query_case, c) for c, _ in faiss_candidates]
        if not any(all_flags):
            skipped += 1
            continue

        # FAISS-only metrics (top 5)
        faiss_flags = [is_relevant(query_case, c) for c, _ in faiss_candidates[:EVAL_TOP_K]]
        faiss_mrr.append(reciprocal_rank(faiss_flags))
        faiss_p5.append(precision_at_k(faiss_flags, EVAL_TOP_K))
        faiss_ndcg.append(ndcg_at_k(faiss_flags, EVAL_TOP_K))

        # Rerank
        pairs = [
            [query_case["text"][:256], c["text"][:400]]
            for c, _ in faiss_candidates
        ]
        t1 = time.time()
        raw_scores = rerank_model.predict(pairs, show_progress_bar=False)
        rerank_ms = (time.time() - t1) * 1000

        # Sort by entailment score (using _extract_relevance_score)
        reranked = sorted(
            [(faiss_candidates[i][0], _extract_relevance_score(raw_scores[i]))
             for i in range(len(faiss_candidates))],
            key=lambda x: x[1], reverse=True
        )

        rerank_flags = [is_relevant(query_case, c) for c, _ in reranked[:EVAL_TOP_K]]
        rerank_mrr.append(reciprocal_rank(rerank_flags))
        rerank_p5.append(precision_at_k(rerank_flags, EVAL_TOP_K))
        rerank_ndcg.append(ndcg_at_k(rerank_flags, EVAL_TOP_K))
        latencies.append(faiss_ms + rerank_ms)

    # ── Aggregate ──────────────────────────────────────────────────────────
    evaluated = len(faiss_mrr)
    print(f"\n  Evaluated: {evaluated} queries | Skipped: {skipped}")

    def avg(lst): return round(sum(lst) / len(lst), 4) if lst else 0.0

    if evaluated > 0:
        results["evaluation_summary"]["queries_evaluated"] = evaluated
        results["evaluation_summary"]["queries_skipped"]   = skipped

        results["faiss_only"] = {
            "MRR@5":  avg(faiss_mrr),
            "P@5":    avg(faiss_p5),
            "NDCG@5": avg(faiss_ndcg),
        }
        results["faiss_plus_reranker"] = {
            "MRR@5":  avg(rerank_mrr),
            "P@5":    avg(rerank_p5),
            "NDCG@5": avg(rerank_ndcg),
        }
        results["reranker_improvement"] = {
            "MRR_delta":  round(avg(rerank_mrr)  - avg(faiss_mrr),  4),
            "P5_delta":   round(avg(rerank_p5)   - avg(faiss_p5),   4),
            "NDCG_delta": round(avg(rerank_ndcg) - avg(faiss_ndcg), 4),
        }

        sorted_lat = sorted(latencies)
        results["latency"] = {
            "avg_query_ms": round(avg(latencies), 1),
            "p95_query_ms": round(sorted_lat[int(len(sorted_lat) * 0.95)], 1)
                            if sorted_lat else 0.0,
        }

    _save_results(results)
    _print_report(results)
    return results


def _save_results(results: dict):
    """Always save complete JSON — called even on early exit."""
    Path(RETRIEVAL_METRICS_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(RETRIEVAL_METRICS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved: {RETRIEVAL_METRICS_PATH}")


def _print_report(results: dict):
    """Print human-readable eval report."""
    fi    = results["faiss_only"]
    re    = results["faiss_plus_reranker"]
    delta = results["reranker_improvement"]
    lat   = results["latency"]
    ev    = results["evaluation_summary"]

    print(f"\n{'='*60}")
    print(f"RETRIEVAL EVALUATION RESULTS")
    print(f"Evaluated: {ev['queries_evaluated']} | Skipped: {ev['queries_skipped']}")
    print(f"Oracle: {ev['relevance_oracle']}")
    print(f"{'='*60}")
    print(f"\n  {'Metric':<12}  {'FAISS only':>12}  {'+ Reranker':>12}  {'Delta':>10}")
    print(f"  {'-'*50}")
    for metric in ["MRR@5", "P@5", "NDCG@5"]:
        f_val   = fi[metric]
        r_val   = re[metric]
        d_key   = metric.replace("@5","_delta").replace("P","P5")
        d_val   = delta.get(d_key, re[metric] - fi[metric])
        sign    = "+" if d_val >= 0 else ""
        print(f"  {metric:<12}  {f_val:>12.4f}  {r_val:>12.4f}  {sign}{d_val:>9.4f}")

    print(f"\n  Avg latency: {lat['avg_query_ms']}ms | P95: {lat['p95_query_ms']}ms")

    print(f"\n{'='*60}")
    print(f"*** SAVE THESE FOR RESUME ***")
    print(f"  MRR@5  (reranker): {re['MRR@5']:.4f}")
    print(f"  P@5    (reranker): {re['P@5']:.4f}")
    print(f"  NDCG@5 (reranker): {re['NDCG@5']:.4f}")
    print(f"  NDCG delta:        {delta['NDCG_delta']:+.4f}")
    print(f"  Avg latency:       {lat['avg_query_ms']}ms")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_retrieval_eval()
