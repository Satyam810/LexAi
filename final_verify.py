# Quick import sanity check — all 3 previously broken imports
from src.explanation_engine import explain_similarity, explain_results
from src.search_pipeline import SearchPipeline, PipelineResponse, check_required_files
from src.eval_pipeline import run_retrieval_eval, is_relevant
print("✅ All previously broken imports now work")

# Validator spot check — 3 previously failing cases
from src.query_validator import validate_query
assert not validate_query("what is the weather like in Mumbai today please")[0], "Non-legal should fail"
assert not validate_query("आरोपी ने हत्या की थी")[0], "Hindi should fail"
assert not validate_query("accused " * 1000)[0], "Too long should fail"
print("✅ All 3 previously failing validator checks now pass")

# Reranker spot check
from src.reranker import rerank
r = rerank(
    "IPC 302 murder eyewitness forensic",
    [
        ({"text": "civil property dispute injunction", "ipc_sections": [], "case_type": "civil", "evidence_types": [], "court": "HC", "date": ""}, 0.82),
        ({"text": "IPC 302 murder convicted eyewitness forensic", "ipc_sections": ["302"], "case_type": "criminal", "evidence_types": ["eyewitness"], "court": "SC", "date": ""}, 0.79),
    ],
    top_k=2
)
assert r[0][0]["case_type"] == "criminal", "Reranker still wrong"
print("✅ Reranker: criminal case ranked #1 (entailment fix working)")

# File existence check
import os
missing = [f for f in ["test_reranker.py", "test_search_pipeline.py"] if not os.path.exists(f)]
assert not missing, f"Still missing: {missing}"
print("✅ test_reranker.py and test_search_pipeline.py exist")

print("\n🎉 All spot-checks passed. Run full diagnostic to confirm 8/8 sections.")
