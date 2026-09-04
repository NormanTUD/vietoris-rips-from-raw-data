# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.26"]
# ///
import sys
from pathlib import Path
ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import numpy as np
from vrtda.complexes import build_rips
from vrtda.distances import pairwise_distances
from vrtda.persistence import persistent_homology
from vrtda.cocycles import persistent_loops, loops_1skeleton
from vrtda.depth_persistence import depth_chains, LayerResult, _select_loops

base = np.array([[0,0],[1,0],[1,1],[0,1]], float)
extra = np.array([[10,10],[11,10],[10,11],[11,11]], float) + np.arange(4)[:,None]*3
X = np.vstack([base, extra])
D = pairwise_distances(X, "euclidean")
C = build_rips(X, D, 1.2, max_dim=2)
bc = persistent_homology(C)
print("n_simplices", C.n_simplices, "count dim1", C.count(1), "count dim2", C.count(2))
print("H1 intervals:", [(round(iv.birth,3), iv.death, iv.birth_simplex, iv.is_essential) for iv in bc.of_dim(1)])
print("basis loops:", [(lp.birth_simplex, lp.vertices) for lp in loops_1skeleton(C, 1.2)])
labels = [f"t{i}" for i in range(X.shape[0])]
lr = {L: LayerResult(layer=L, complex=C, barcode=bc, nn=1.0, eps_max=1.2, labels=labels) for L in [0,1,2]}
print("selected loops L0:", [ (lp.vertices,) for lp in _select_loops(lr[0], top_k=10)])
chains = depth_chains(lr, min_overlap=0.2, max_gap=1, top_k=10)
for c in chains:
    print("chain", c.span, "len", c.length, "tokens", c.tokens, c.per_layer_tokens)
