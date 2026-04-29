from src.reranker import rerank
import importlib, src.reranker
importlib.reload(src.reranker)
from src.reranker import rerank

query = "accused charged under IPC 302 murder with eyewitness and forensic evidence"

fake_candidates = [
    (
        {
            "text": "property dispute partition civil court decree injunction damages",
            "verdict": "acquitted", "ipc_sections": [],
            "case_type": "civil", "evidence_types": [],
            "court": "Delhi HC", "date": "2020"
        },
        0.82   # high FAISS score — wrong case
    ),
    (
        {
            "text": "accused convicted under IPC 302 murder eyewitness testimony "
                    "forensic report confirmed sessions court",
            "verdict": "convicted", "ipc_sections": ["302"],
            "case_type": "criminal", "evidence_types": ["eyewitness", "forensic"],
            "court": "Supreme Court", "date": "2019"
        },
        0.79   # lower FAISS score — correct case
    ),
]

results = rerank(query, fake_candidates, top_k=2)
print("Re-ranked order:")
for i, (case, score) in enumerate(results, 1):
    print(f"  #{i}  entailment={score:.4f}  case_type={case['case_type']}")

if results[0][0]["case_type"] == "criminal":
    print("\n✅ PASS: IPC 302 criminal case ranked #1 (entailment score fix working)")
else:
    print("\n❌ FAIL: Civil case still ranked #1")
    print("   Debug: print raw_scores in reranker.py to see the 3-column output")
