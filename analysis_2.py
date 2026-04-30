import time, numpy as np
from config import RERANKER_MODEL
from src.reranker import rerank, get_reranker

print("=== RERANKER ANALYSIS ===")
print(f"Current model: {RERANKER_MODEL}")
print()

# Load model
print("Loading model...")
t0 = time.time()
model = get_reranker()
load_time = round((time.time() - t0) * 1000)
print(f"Load time: {load_time}ms")
print()

# Test with 50 pairs (same as real pipeline)
query = "accused charged under IPC 302 murder eyewitness forensic evidence sessions court"
dummy_case = {"text": "accused convicted IPC 302 murder eyewitness forensic report confirmed", 
              "ipc_sections": ["302"], "case_type": "criminal", 
              "evidence_types": ["eyewitness", "forensic"], "court": "SC", "date": "2019"}
dummy_civil = {"text": "property dispute partition civil court decree injunction damages",
               "ipc_sections": [], "case_type": "civil",
               "evidence_types": [], "court": "HC", "date": "2020"}

# Build 50 fake candidates like real pipeline does
candidates = []
for i in range(25):
    candidates.append((dummy_case, 0.79))
    candidates.append((dummy_civil, 0.82))

# Measure rerank time
t0 = time.time()
results = rerank(query, candidates, top_k=5)
rerank_time = round((time.time() - t0) * 1000)

print(f"Rerank time (50 pairs): {rerank_time}ms")
print(f"Target:                 < 300ms")
print(f"Status: {'✅ FAST ENOUGH' if rerank_time < 300 else '❌ TOO SLOW'}")
print()

# Check score shape (detect NLI vs single-score model)
raw = model.predict([[query[:256], dummy_case["text"][:400]]], show_progress_bar=False)
raw = np.array(raw)
print(f"Raw score shape: {raw.shape}")
if raw.ndim == 2:
    print(f"Model type: NLI (3-label) — extracts entailment column (index 2)")
    print(f"Raw scores: {raw[0]}")
else:
    print(f"Model type: Single-score — uses score directly")
    print(f"Raw score: {raw[0]:.4f}")
print()

# Check ordering (criminal should beat civil)
test_candidates = [(dummy_case, 0.79), (dummy_civil, 0.82)]
ordered = rerank(query, test_candidates, top_k=2)
top_type = ordered[0][0]["case_type"]
print(f"Ordering test:")
print(f"  #1: {ordered[0][0]['case_type']} (score={ordered[0][1]:.4f})")
print(f"  #2: {ordered[1][0]['case_type']} (score={ordered[1][1]:.4f})")
print(f"  Result: {'✅ CORRECT ORDER' if top_type == 'criminal' else '❌ WRONG ORDER'}")
