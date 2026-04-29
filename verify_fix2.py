from src.explanation_engine import explain_similarity, explain_results

# Quick smoke test
case_a = {
    "text": "IPC 302 murder eyewitness forensic",
    "verdict": "convicted", "ipc_sections": ["302"],
    "case_type": "criminal", "evidence_types": ["eyewitness", "forensic"],
    "court": "Supreme Court", "date": "2019"
}
case_b = {
    "text": "IPC 302 murder circumstantial",
    "verdict": "acquitted", "ipc_sections": ["302"],
    "case_type": "criminal", "evidence_types": ["circumstantial"],
    "court": "Delhi HC", "date": "2021"
}

exp = explain_similarity(case_a, case_b, 0.87)
results_list = explain_results(case_a, [(case_b, 0.87)])

assert len(exp) == 10, f"Expected 10 keys, got {len(exp)}"
assert "divergence" in exp["verdict_analysis"].lower(), "Expected divergence"
assert isinstance(results_list, list) and len(results_list) == 1

print("✅ explain_similarity: OK")
print("✅ explain_results:    OK")
print(f"   Keys: {list(exp.keys())}")
print(f"   Verdict analysis: {exp['verdict_analysis']}")
