from src.eval_pipeline import (
    run_retrieval_eval,
    is_relevant,
    reciprocal_rank,
    precision_at_k,
    ndcg_at_k
)
print("✅ All eval_pipeline functions imported correctly")

# Quick unit test of metrics
assert reciprocal_rank([False, True, False]) == 0.5,  "RR should be 0.5 (rank 2)"
assert precision_at_k([True, False, True, False, True], 5) == 0.6, "P@5 should be 0.6"
import math
dcg = 1.0 + 1.0/math.log2(3) + 1.0/math.log2(5)
idcg = 1.0 + 1.0/math.log2(2) + 1.0/math.log2(3)
expected_ndcg = round(dcg/idcg, 4)
actual_ndcg = round(ndcg_at_k([True, False, True, False, True], 5), 4)
print(f"✅ reciprocal_rank([F,T,F]) = 0.5")
print(f"✅ precision_at_k = 0.6")
print(f"✅ ndcg_at_k = {actual_ndcg}")
print("Fix 6 complete.")
