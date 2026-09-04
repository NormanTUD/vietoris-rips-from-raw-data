# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.26"]
# ///
import sys
from pathlib import Path
ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import csv
import numpy as np
from vrtda import datasets, pairwise_distances

dd = datasets._data_root()
layers = datasets.list_layers()

def token_key(layer):
    p = datasets.layer_path(dd, layer)
    with open(p, newline="") as fh:
        r = csv.reader(fh); h = next(r)
        pi, ti = h.index("prompt_idx"), h.index("token_pos")
        return frozenset((row[pi], row[ti]) for row in r if row)

ref = token_key(layers[0])
all_same = True
for L in layers:
    if token_key(L) != ref:
        all_same = False
        print(f"  layer {L} DIFFERS (n={len(token_key(L))} vs ref {len(ref)})")
print(f"layers checked: {len(layers)}")
print(f"ref token-set size: {len(ref)}")
print(f"ALL layers share the SAME 81 tokens: {all_same}")

# per-layer scale (mean nearest-neighbour) to see how embeddings spread with depth
scales = []
for L in [0, 8, 16, 32, 48, 64]:
    ps = datasets.load_token_cloud(layer=L)
    D = pairwise_distances(ps.data, "euclidean")
    d = D.copy(); np.fill_diagonal(d, np.inf)
    scales.append((L, round(float(d.min(1).mean()), 4)))
print("per-layer mean-nn (scale):", scales)
