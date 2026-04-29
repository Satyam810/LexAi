"""
Search pipeline integration test.

Tests validation routing, result structure, explanation fields,
query_case population, and latency reporting.

Run: python test_search_pipeline.py
"""

import sys
from src.search_pipeline import SearchPipeline, PipelineResponse, SearchResult, check_required_files

print("=" * 60)
print("SEARCH PIPELINE INTEGRATION TEST")
print("=" * 60)

pipeline = SearchPipeline()
all_passed = True
results_from_valid = None

# ── Test 1: check_required_files ─────────────────────────────────────────
print("\n[1] check_required_files()...")
ok, msg = check_required_files()
status = "✅ PASS" if ok else "❌ FAIL"
print(f"  {status}  Files present: {ok}")
if not ok:
    print(f"         Missing: {msg}")
    all_passed = False

# ── Test 2: health_check ──────────────────────────────────────────────────
print("\n[2] pipeline.health_check()...")
health_ok, health_msg = pipeline.health_check()
status = "✅ PASS" if health_ok else "❌ FAIL"
print(f"  {status}  Health: {health_msg[:60]}")
if not health_ok:
    all_passed = False

# ── Test 3: Empty query → validation_error ────────────────────────────────
print("\n[3] Empty query rejected...")
resp = pipeline.search("")
ok3 = (not resp.success and resp.error_type == "validation_error")
status = "✅ PASS" if ok3 else "❌ FAIL"
print(f"  {status}  success={resp.success}  error_type={resp.error_type}")
if not ok3:
    all_passed = False

# ── Test 4: Short query → validation_error ────────────────────────────────
print("\n[4] Short query rejected...")
resp = pipeline.search("ipc")
ok4 = (not resp.success and resp.error_type == "validation_error")
status = "✅ PASS" if ok4 else "❌ FAIL"
print(f"  {status}  success={resp.success}  error_type={resp.error_type}")
if not ok4:
    all_passed = False

# ── Test 5: Non-legal query → validation_error ───────────────────────────
print("\n[5] Non-legal query rejected...")
resp = pipeline.search("what is the weather like in Mumbai today please tell me")
ok5 = (not resp.success and resp.error_type == "validation_error")
status = "✅ PASS" if ok5 else "❌ FAIL"
print(f"  {status}  success={resp.success}  error_type={resp.error_type}")
if not ok5:
    all_passed = False

# ── Test 6: Valid query executes ──────────────────────────────────────────
print("\n[6] Valid legal query returns results...")
if health_ok:
    resp = pipeline.search(
        "Accused charged under IPC Section 302 for murder. "
        "Prosecution relies on eyewitness testimony and forensic evidence. "
        "Sessions Court hearing in progress."
    )
    results_from_valid = resp
    ok6 = resp.success
    status = "✅ PASS" if ok6 else "❌ FAIL"
    n_results = len(resp.results) if resp.results else 0
    print(f"  {status}  success={resp.success}  results={n_results}")
    if not ok6:
        print(f"         Error: {resp.error}")
        all_passed = False
else:
    print("  ⏭️  SKIP (pipeline not healthy)")

# ── Test 7: SearchResult structure ───────────────────────────────────────
print("\n[7] SearchResult fields...")
if results_from_valid and results_from_valid.success and results_from_valid.results:
    r = results_from_valid.results[0]
    has_case  = hasattr(r, "case") and isinstance(r.case, dict)
    has_score = hasattr(r, "score") and isinstance(r.score, float)
    has_exp   = hasattr(r, "explanation") and isinstance(r.explanation, dict)
    has_rank  = hasattr(r, "rank") and r.rank == 1
    ok7 = all([has_case, has_score, has_exp, has_rank])
    status = "✅ PASS" if ok7 else "❌ FAIL"
    print(f"  {status}  case={has_case}  score={has_score}  explanation={has_exp}  rank={has_rank}")
    if not ok7:
        all_passed = False
else:
    print("  ⏭️  SKIP (no results)")

# ── Test 8: Explanation keys ─────────────────────────────────────────────
print("\n[8] Explanation dict keys...")
if results_from_valid and results_from_valid.success and results_from_valid.results:
    exp = results_from_valid.results[0].explanation
    required = ["similarity_reason", "key_differences", "verdict_analysis",
                "shared_ipc", "shared_evidence", "retrieved_verdict",
                "retrieved_court", "similarity_score"]
    missing = [k for k in required if k not in exp]
    ok8 = len(missing) == 0
    status = "✅ PASS" if ok8 else "❌ FAIL"
    print(f"  {status}  all required keys present")
    if missing:
        print(f"         Missing: {missing}")
        all_passed = False
else:
    print("  ⏭️  SKIP (no results)")

# ── Test 9: query_case populated ─────────────────────────────────────────
print("\n[9] query_case NLP fields...")
if results_from_valid and results_from_valid.success:
    qc = results_from_valid.query_case
    required = ["text", "verdict", "ipc_sections", "case_type", "evidence_types"]
    ok9 = qc is not None and all(k in qc for k in required)
    status = "✅ PASS" if ok9 else "❌ FAIL"
    print(f"  {status}  query_case has all fields")
    if ok9:
        print(f"         ipc_sections={qc['ipc_sections']}  case_type={qc['case_type']}")
    else:
        all_passed = False
else:
    print("  ⏭️  SKIP")

# ── Test 10: Latency reported ─────────────────────────────────────────────
print("\n[10] Latency reported...")
if results_from_valid and results_from_valid.success:
    ok10 = results_from_valid.latency_ms is not None and results_from_valid.latency_ms > 0
    status = "✅ PASS" if ok10 else "❌ FAIL"
    print(f"  {status}  latency_ms={results_from_valid.latency_ms}ms")
    if not ok10:
        all_passed = False
else:
    print("  ⏭️  SKIP")

# ── Summary ───────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
if all_passed:
    print("✅ ALL TESTS PASSED — search pipeline is production-ready")
else:
    print("❌ SOME TESTS FAILED — paste this output for diagnosis")
print(f"{'='*60}")
