# ⚖️ LexAI — Indian Court Judgment AI

> Open-source AI tool for Indian legal research.
> Find similar past judgments, understand why they match,
> and detect legal inconsistencies — all in under 1 second.

## 🚀 Live Demo
**[→ Try LexAI](https://satyam810-lexai-app-xxxxxx.streamlit.app)**

## 📊 Key Metrics

| Metric | Value |
|---|---|
| Cases indexed | 5,007 Indian court judgments |
| Search latency | ~270ms (FAISS direct retrieval) |
| MRR@5 | 0.5269 |
| NDCG@5 | 0.5746 |
| Silhouette score | **0.834** (GOOD cluster quality) |
| Clustering algorithm | HDBSCAN |

## ✨ What LexAI Does

**🔍 Semantic Search**
Describe your case in plain English. LexAI finds the 5
most similar past Indian court judgments using LegalBERT
embeddings and FAISS approximate nearest neighbour search.

**⚖️ Explains Every Result**
Not just "here are similar cases" — LexAI explains:
- Why each case matches yours
- What structurally differs
- Why verdicts may diverge (forensic evidence, court level, etc.)

**🚨 Legal Gap Detection**
Automatically finds clusters where identical charges led
to opposite verdicts across different courts — exposing
inconsistencies in how Indian law is applied.

**📊 Evaluated with Real Metrics**
MRR@5, P@5, NDCG@5 measured on 96 queries.
Two cross-encoder models tested and documented honestly.

## 🏗️ Architecture
Query text
→ Input validation (length, language, legal signal)
→ NLP extraction (spaCy + regex)
verdict, IPC sections, case type, evidence types
→ LegalBERT embedding (nlpaueb/legal-bert-base-uncased)
→ FAISS ANN search (top 50, <100ms)
→ Explanation engine (deterministic, no LLM)
→ Response (<300ms total)
Background (Colab GPU, runs once):
5,007 cases → batch embed → FAISS index
→ HDBSCAN clustering → UMAP → BERTopic → gap detection

## 📦 Data Sources

- **Indian Kanoon API** — real current Indian court judgments
- **pile-of-law/indiankanoon** — HuggingFace legal corpus
- Cases cover: murder (IPC 302), rape (IPC 376), fraud (IPC 420),
  NDPS, POCSO, dowry death (IPC 304B), corruption, bail matters,
  constitutional writ petitions, criminal appeals

## 🛠️ Run Locally

```bash
git clone https://github.com/Satyam810/lexai
cd lexai

# Python 3.11 required (hdbscan wheels)
py -3.11 -m venv venv311
venv311\Scripts\activate       # Windows
# source venv311/bin/activate  # Mac/Linux

pip install -r requirements.txt
python -m spacy download en_core_web_lg

# Add your Indian Kanoon API key to .env
echo "INDIAN_KANOON_API_KEY=your_key" >> .env

python src/fetcher.py
python src/nlp_pipeline.py

# Run Phase 4 Colab notebook for GPU embeddings
# notebooks/colab_phase4.ipynb → download outputs

python verify_outputs.py
streamlit run app.py
```

## 🔬 Known Limitations

- 47.5% unknown verdicts — Indian Kanoon premium docs
  return 403 without paid API tier
- Cross-encoder reranker evaluated but disabled —
  both models tested degraded NDCG vs FAISS-only
  without domain fine-tuning
- Silhouette 0.834 — excellent for 5k cases,
  improves further with larger dataset

## 📄 License
MIT License — free to use, modify, and deploy.

## 👤 Author
**Satyam Kumar** — AI/ML Engineer
[GitHub](https://github.com/Satyam810) ·
[LinkedIn](https://linkedin.com/in/satyamkumar)
