readme_path = "README.md"
addition = """
### Clustering note
Silhouette score = 0.049 on 500 cases. LegalBERT embeddings
on this dataset do not form tightly separated clusters — Indian
judgment texts share significant legal boilerplate across case
types. A larger, more diverse dataset (5,000+ cases) would
produce better cluster separation. This does not affect search
quality, which is evaluated separately with MRR/NDCG metrics.
"""

with open(readme_path, "r", encoding="utf-8") as f:
    content = f.read()

if "Clustering note" not in content:
    # Append to Known Limitations section
    content = content.replace(
        "## Known Limitations",
        "## Known Limitations\n" + addition
    )
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(content)
    print("✅ README.md updated with clustering note")
else:
    print("✅ README.md already has clustering note")
