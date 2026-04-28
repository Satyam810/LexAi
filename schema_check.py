"""Phase 2.1 — Deep check of both datasets for text content."""
from datasets import load_dataset
import json

# Check KanoonGPT indexable_text length
print("=== KanoonGPT: checking text fields ===")
ds1 = load_dataset("KanoonGPT/indian-case-laws", split="train", streaming=True)
for i, row in enumerate(ds1):
    if i >= 3:
        break
    idx_text = str(row.get("indexable_text", ""))
    norm_json = str(row.get("normalized_record_json", ""))
    print(f"\nRow {i}:")
    print(f"  indexable_text length: {len(idx_text)} chars")
    print(f"  indexable_text first 200: {idx_text[:200]!r}")
    print(f"  court_name: {row.get('court_name')}")
    print(f"  decision_date: {row.get('decision_date')}")
    print(f"  disposition_text: {row.get('disposition_text')}")
    print(f"  case_title: {row.get('case_title')}")
    # Check if normalized_record_json has full text
    try:
        nj = json.loads(norm_json)
        # Look for text/body in the JSON
        full_text = nj.get("judgment_text", nj.get("text", nj.get("body", "")))
        order_text = ""
        if "case" in nj:
            order_text = str(nj["case"].get("order_text", ""))
        print(f"  order_text in normalized_json: {len(order_text)} chars")
        if order_text:
            print(f"    First 200: {order_text[:200]!r}")
    except:
        pass

print("\n\n=== SnehaDeshmukh: checking completeness ===")
ds2 = load_dataset("SnehaDeshmukh/IndianBailJudgments-1200", split="train", streaming=True)
count = 0
sample_facts_lengths = []
for i, row in enumerate(ds2):
    count += 1
    if i < 3:
        facts = str(row.get("facts", ""))
        reason = str(row.get("judgment_reason", ""))
        summary = str(row.get("summary", ""))
        combined = f"{facts} {reason} {summary}"
        sample_facts_lengths.append(len(combined))
        print(f"\nRow {i}:")
        print(f"  facts length: {len(facts)} chars")
        print(f"  judgment_reason length: {len(reason)} chars")
        print(f"  summary length: {len(summary)} chars")
        print(f"  combined: {len(combined)} chars")
        print(f"  court: {row.get('court')}")
        print(f"  ipc_sections: {row.get('ipc_sections')}")
        print(f"  bail_outcome: {row.get('bail_outcome')}")
        print(f"  crime_type: {row.get('crime_type')}")
        
print(f"\nTotal rows in SnehaDeshmukh: {count}")
print(f"Avg combined text length (first 3): {sum(sample_facts_lengths)//len(sample_facts_lengths)} chars")
