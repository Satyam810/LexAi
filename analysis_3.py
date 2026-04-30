import json
from pathlib import Path

print("=== EVAL METRICS ANALYSIS ===")

path = "data/processed/eval_metrics_retrieval.json"
if not Path(path).exists():
    print("❌ eval_metrics_retrieval.json not found")
    print("   Run: python src/eval_pipeline.py first")
else:
    with open(path) as f:
        m = json.load(f)

    fi  = m.get("faiss_only", {})
    re  = m.get("faiss_plus_reranker", {})
    dlt = m.get("reranker_improvement", {})
    lat = m.get("latency", {})
    ev  = m.get("evaluation_summary", {})
    mod = m.get("model_info", {})

    print(f"Model used:          {mod.get('reranker_model', 'unknown')}")
    print(f"Queries evaluated:   {ev.get('queries_evaluated', '?')}")
    print()
    print(f"{'Metric':<12} {'FAISS only':>12} {'+ Reranker':>12} {'Delta':>10}")
    print("-" * 50)
    for key in ["MRR@5", "P@5", "NDCG@5"]:
        f_val = fi.get(key, 0)
        r_val = re.get(key, 0)
        delta = r_val - f_val
        sign = "+" if delta >= 0 else ""
        status = "✅" if delta >= 0 else "❌"
        print(f"{key:<12} {f_val:>12.4f} {r_val:>12.4f} {sign}{delta:>9.4f} {status}")
    print()
    print(f"Avg query latency:  {lat.get('avg_query_ms', '?')}ms")
    print(f"P95 query latency:  {lat.get('p95_query_ms', '?')}ms")
    print(f"Embed time:         {lat.get('avg_embed_ms', '?')}ms")
    print(f"Rerank time:        {lat.get('avg_rerank_ms', '?')}ms")
    print()

    # Diagnose
    avg_lat = lat.get("avg_query_ms", 9999)
    ndcg_delta = dlt.get("NDCG_delta", 0)

    print("=== AUTO DIAGNOSIS ===")
    if avg_lat > 1000:
        print(f"❌ LATENCY PROBLEM: {avg_lat}ms — reranker is too slow for production")
        rerank_ms = lat.get('avg_rerank_ms', 0)
        print(f"   Rerank alone: {rerank_ms}ms — this is where time is spent")
    else:
        print(f"✅ LATENCY OK: {avg_lat}ms")

    if ndcg_delta < 0:
        print(f"❌ QUALITY PROBLEM: Reranker degraded NDCG by {ndcg_delta:.4f}")
        print(f"   FAISS alone is better than FAISS + reranker on this dataset")
    else:
        print(f"✅ QUALITY OK: Reranker improved NDCG by +{ndcg_delta:.4f}")
