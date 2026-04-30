import sys
print(f"Python version: {sys.version}")
print()

packages = [
    ("hdbscan",      "import hdbscan; print(f'  version: {hdbscan.__version__}')"),
    ("bertopic",     "from bertopic import BERTopic; import bertopic; print(f'  version: {bertopic.__version__}')"),
    ("faiss",        "import faiss; print(f'  version: {faiss.__version__}')"),
    ("sentence_transformers", "import sentence_transformers; print(f'  version: {sentence_transformers.__version__}')"),
    ("spacy",        "import spacy; print(f'  version: {spacy.__version__}')"),
    ("streamlit",    "import streamlit; print(f'  version: {streamlit.__version__}')"),
    ("numpy",        "import numpy; print(f'  version: {numpy.__version__}')"),
    ("sklearn",      "import sklearn; print(f'  version: {sklearn.__version__}')"),
    ("umap",         "import umap; print('  ok')"),
    ("plotly",       "import plotly; print(f'  version: {plotly.__version__}')"),
]

print("=== PACKAGE ANALYSIS ===")
results = []
for name, stmt in packages:
    try:
        exec(stmt)
        print(f"  ✅ {name}")
        results.append((name, True, ""))
    except Exception as e:
        print(f"  ❌ {name} — {type(e).__name__}: {e}")
        results.append((name, False, str(e)))

failed = [r for r in results if not r[1]]
print(f"\nResult: {len(results)-len(failed)}/{len(results)} packages OK")
if failed:
    print("Failed:")
    for name, _, err in failed:
        print(f"  ❌ {name}: {err}")
