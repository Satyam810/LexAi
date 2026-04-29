# LexAI v3.2 — Indian Court Judgment Similarity & Legal Gap Finder

A production-grade NLP system that analyzes Indian court bail judgments, finds similar cases, detects verdict inconsistencies, and visualizes judgment clusters.

## Features

- **Semantic Search** — Find similar judgments using LegalBERT embeddings + FAISS
- **Cross-Encoder Reranking** — NLI-DeBERTa reranker for precision
- **Legal Gap Detection** — Identifies clusters where similar cases have opposing verdicts
- **Deterministic Explanations** — Rule-based reasoning (no LLM hallucination)
- **Interactive Dashboard** — Streamlit UI with cluster maps, verdict analytics

## Tech Stack

| Component | Technology |
|---|---|
| Embeddings | LegalBERT (`nlpaueb/legal-bert-base-uncased`) |
| Vector Search | FAISS (cosine similarity) |
| Reranker | NLI-DeBERTa-v3-small |
| NER | spaCy `en_core_web_lg` |
| Clustering | KMeans (selected over HDBSCAN via silhouette) |
| Visualization | UMAP + Plotly |
| Dashboard | Streamlit |
| Dataset | `SnehaDeshmukh/IndianBailJudgments-1200` |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_lg

# Run data pipeline (if starting fresh)
python -m src.fetcher          # Fetch 500 cases
python -m src.nlp_pipeline     # Extract verdicts, IPC sections, entities

# Run GPU pipeline in Colab
# Upload notebooks/colab_phase4.ipynb to Google Colab
# Download the 7 output files to data/processed/

# Launch dashboard
set PYTHONPATH=.
python -m streamlit run app.py --server.port 8501
```

## Project Structure

```
lexai/
├── app.py                    # Streamlit dashboard
├── config.py                 # All system constants
├── src/
│   ├── fetcher.py            # Data collection from HuggingFace
│   ├── nlp_pipeline.py       # NLP: verdict, IPC, NER extraction
│   ├── embedder.py           # LegalBERT embedding wrapper
│   ├── retrieval.py          # FAISS search wrapper
│   ├── reranker.py           # Cross-encoder reranking
│   ├── explanation_engine.py # Deterministic explanation generation
│   ├── search_pipeline.py    # Search orchestrator (main entry point)
│   ├── inconsistency.py      # Gap detection module
│   ├── query_validator.py    # Input validation & sanitization
│   └── eval_pipeline.py      # MRR, Precision, NDCG evaluation
├── notebooks/
│   └── colab_phase4.ipynb    # GPU pipeline (embeddings, clustering)
├── data/processed/           # Generated data files
├── scripts/                  # Utility scripts
└── tests/                    # Unit tests
```

## Evaluation Results

| Metric | FAISS Only | FAISS + Reranker |
|---|---|---|
| MRR@5 | 0.8507 | 0.6840 |
| Precision@5 | 0.7120 | 0.5960 |
| NDCG@5 | 0.8787 | 0.7468 |

## Known Limitations

### Clustering note
Silhouette score = 0.049 on 500 cases. LegalBERT embeddings
on this dataset do not form tightly separated clusters — Indian
judgment texts share significant legal boilerplate across case
types. A larger, more diverse dataset (5,000+ cases) would
produce better cluster separation. This does not affect search
quality, which is evaluated separately with MRR/NDCG metrics.


- Clustering silhouette score is 0.0495 on 500 cases —
  below the 0.1 target. LegalBERT embeddings of 500 Indian
  judgments show low cluster separation, likely due to
  dataset size. Target: re-run Phase 4 with 1,000+ cases
  once Indian Kanoon API key is available.

## License

MIT
