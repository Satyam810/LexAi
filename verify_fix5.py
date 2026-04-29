from src.search_pipeline import SearchPipeline, PipelineResponse, check_required_files
from dataclasses import fields as dc_fields

# Check all exports present
print("✅ SearchPipeline imported")
print("✅ PipelineResponse imported")
print("✅ check_required_files imported")

# Check dataclass fields
sr_fields = [f.name for f in dc_fields(PipelineResponse)]
expected  = ["success", "query_case", "results", "error", "error_type", "latency_ms"]
missing   = [f for f in expected if f not in sr_fields]
assert not missing, f"PipelineResponse missing fields: {missing}"
print(f"✅ PipelineResponse fields: {sr_fields}")

# Check runtime safety
ok, msg = check_required_files()
print(f"✅ check_required_files() returned: ok={ok}, msg='{msg[:60]}'")

# Check pipeline instantiation
p = SearchPipeline()
h_ok, h_msg = p.health_check()
print(f"✅ SearchPipeline().health_check() = ok={h_ok}")

# Check validation routing
resp = p.search("")
assert resp.error_type == "validation_error", f"Expected validation_error, got {resp.error_type}"
print("✅ Empty query → validation_error")
print("\nFix 5 complete.")
