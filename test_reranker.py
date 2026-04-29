"""
Reranker test — verifies cross-encoder correctly reorders FAISS results.

The deliberately wrong civil case has a HIGHER FAISS score (0.82).
The correct IPC302 criminal case has a LOWER FAISS score (0.79).
After reranking, the criminal case MUST be ranked #1.

Run: python test_reranker.py
"""

import sys, time
import numpy as np
from src.reranker import rerank, _extract_relevance_score, get_reranker

print("=" * 55)
print("RERANKER TEST")
print("=" * 55)

all_passed = True

# ── Test 1: Score extraction logic ────────────────────────────────────────
print("\n[1] _extract_relevance_score extraction:")
test_cases = [
    (0.85,                      0.85,  "plain float"),
    (np.array([0.1, 0.2, 0.9]), 0.9,   "NLI array → entailment (idx 2)"),
    (np.array([0.8, 0.1, 0.1]), 0.1,   "NLI array → entailment (idx 2) low"),
    (np.array([0.3, 0.7]),      0.7,   "binary array → idx 1"),
]
for raw, expected, label in test_cases:
    result = _extract_relevance_score(raw)
    ok = abs(result - expected) < 0.001
    status = "✅ PASS" if ok else "❌ FAIL"
    print(f"  {status}  {label:<40}  result={result:.4f}  expected={expected:.4f}")
    if not ok:
        all_passed = False

# ── Test 2: Model loads ───────────────────────────────────────────────────
print("\n[2] Model loading:")
try:
    t0 = time.time()
    model = get_reranker()
    load_time = time.time() - t0
    print(f"  ✅ PASS  Loaded in {load_time:.1f}s")
except Exception as e:
    print(f"  ❌ FAIL  Model load error: {e}")
    all_passed = False
    print("\nCannot continue without model.")
    sys.exit(1)

# ── Test 3: Ranking order ───────────────────────────────────────────────
print("\n[3] Ranking order (civil case must drop, IPC302 must rise):")
query = "accused charged under IPC 302 murder with eyewitness and forensic evidence"
fake_candidates = [
    (
        {
            "text": "property dispute partition civil court decree injunction damages tort",
            "verdict": "acquitted", "ipc_sections": [],
            "case_type": "civil", "evidence_types": [],
            "court": "Delhi HC", "date": "2020"
        },
        0.82  # HIGH FAISS score — WRONG case
    ),
    (
        {
            "text": "accused convicted under IPC 302 murder eyewitness testimony "
                    "forensic report confirmed by sessions court",
            "verdict": "convicted", "ipc_sections": ["302"],
            "case_type": "criminal", "evidence_types": ["eyewitness", "forensic"],
            "court": "Supreme Court", "date": "2019"
        },
        0.79  # LOW FAISS score — CORRECT case
    ),
]

t0 = time.time()
results = rerank(query, fake_candidates, top_k=2)
elapsed = (time.time() - t0) * 1000

print(f"  Rerank completed in {elapsed:.0f}ms")
for i, (case, score) in enumerate(results, 1):
    print(f"  #{i}  score={score:.4f}  case_type={case['case_type']:<10}  verdict={case['verdict']}")

ok3 = (results[0][0]["case_type"] == "criminal")
status = "✅ PASS" if ok3 else "❌ FAIL"
print(f"\n  {status}  Criminal IPC302 ranked #1 despite lower FAISS score")
if not ok3:
    all_passed = False
    print("         Check _extract_relevance_score — NLI model may need index adjustment")

# ── Test 4: Speed ──────────────────────────────────────────────────────────
print("\n[4] Rerank speed:")
ok4 = elapsed < 1000
status = "✅ PASS" if ok4 else "⚠️  SLOW"
print(f"  {status}  {elapsed:.0f}ms for 2 pairs (limit 1000ms)")
if not ok4:
    all_passed = False

# ── Test 5: Returns correct count ────────────────────────────────────────
print("\n[5] Return count:")
ok5 = len(results) == 2
status = "✅ PASS" if ok5 else "❌ FAIL"
print(f"  {status}  Got {len(results)} results, expected 2")
if not ok5:
    all_passed = False

# ── Summary ───────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
if all_passed:
    print("✅ ALL TESTS PASSED — reranker is working correctly")
else:
    print("❌ SOME TESTS FAILED — paste this output for diagnosis")
print(f"{'='*55}")
