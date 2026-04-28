import json

nb = json.load(open("notebooks/colab_phase4.ipynb"))
print(f"Notebook cells: {len(nb['cells'])}")
for i, c in enumerate(nb["cells"]):
    ctype = c["cell_type"]
    src = c.get("source", [""])[0][:70].strip()
    print(f"  Cell {i}: [{ctype}] {src}")
