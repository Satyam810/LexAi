"""Phase 5 — Verify all Colab outputs."""
import numpy as np, json, os

files = {
    "embeddings.npy": "data/processed/embeddings.npy",
    "faiss.index": "data/processed/faiss.index",
    "cluster_labels.npy": "data/processed/cluster_labels.npy",
    "cluster_topics.json": "data/processed/cluster_topics.json",
    "coords_2d.npy": "data/processed/coords_2d.npy",
    "gaps.json": "data/processed/gaps.json",
    "eval_metrics.json": "data/processed/eval_metrics.json",
    "cases.json": "data/processed/cases.json",
}

print("=" * 55)
print("PHASE 5 — OUTPUT VERIFICATION")
print("=" * 55)

all_ok = True
for name, path in files.items():
    if not os.path.exists(path):
        print(f"  MISSING: {name}")
        all_ok = False
        continue
    size = os.path.getsize(path)
    print(f"  OK: {name:25s}  {size:>10,} bytes")

# Deep checks
print("\n--- Deep checks ---")

emb = np.load("data/processed/embeddings.npy")
print(f"Embeddings shape: {emb.shape}")
assert emb.shape == (500, 768), f"Bad shape: {emb.shape}"

labels = np.load("data/processed/cluster_labels.npy")
print(f"Cluster labels: {len(labels)} | unique: {len(set(labels))}")
assert len(labels) == 500

coords = np.load("data/processed/coords_2d.npy")
print(f"UMAP coords: {coords.shape}")
assert coords.shape[1] == 2

with open("data/processed/eval_metrics.json") as f:
    metrics = json.load(f)
print(f"Clustering: {metrics['winner_algorithm']} | k={metrics['n_clusters']} | sil={metrics['silhouette_score']}")
assert metrics["silhouette_score"] > 0

with open("data/processed/gaps.json") as f:
    gaps = json.load(f)
print(f"Gaps found: {len(gaps)}")

with open("data/processed/cluster_topics.json") as f:
    topics = json.load(f)
print(f"Cluster topics: {len(topics)}")

with open("data/processed/cases.json") as f:
    cases = json.load(f)
print(f"Cases: {len(cases)}")

print("\n" + "=" * 55)
if all_ok:
    print("PHASE 5 VERIFICATION: ALL PASSED")
else:
    print("PHASE 5 VERIFICATION: SOME FILES MISSING")
print("=" * 55)
