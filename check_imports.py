"""Quick import check for Phase 1 gate."""
import streamlit
import sentence_transformers
import faiss
import sklearn
import spacy
import plotly
import pandas
import numpy
import torch

print("All core packages imported successfully")

nlp = spacy.load("en_core_web_lg")
print(f"spaCy model loaded: {nlp.meta['name']}")
print(f"PyTorch: {torch.__version__}")
print(f"NumPy: {numpy.__version__}")
print(f"Streamlit: {streamlit.__version__}")
print(f"FAISS: {faiss.__version__}")

print("\n✅ Phase 1 verification PASSED — all packages working.")
