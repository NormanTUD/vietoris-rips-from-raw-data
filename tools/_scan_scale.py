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
from vrtda import datasets
from vrtda.distances import pairwise_distances
from vrtda.complexes import build_rips
from vrtda.persistence import persistent_homology

def nn(D):
    d = D.copy(); np.fill_diagonal(d, np.inf)
    return float(d.min(axis=1).mean())

for L in [0, 16, 32, 48, 64]:
    ps = datasets.load_token_cloud(layer=L)
    D = pairwise_distances(ps.data, "euclidean")
    n = nn(D)
    cap = 6.0 * n
    C = build_rips(ps.data, D, cap, max_dim=2)
    bc = persistent_homology(C)
    # H1 intervals
    ivs = [iv for iv in bc.of_dim(1)]
    ess = sum(1 for iv in ivs if iv.is_essential)
    tot = sum((min(iv.death, cap) - iv.birth) for iv in ivs)
    print(f"\nlayer {L}: nn={n:.1f}  #H1int={len(ivs)}  #essentialH1@{cap:.0f}={ess}  totalPERSIST_H1={tot:.1f}")
    # scan relative scale
    for f in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]:
        s = f * n
        b1 = bc.betti_at(s)[1]
        e = sum(1 for iv in ivs if iv.is_essential and iv.birth <= s)
        print(f"   frac={f:4.1f}  scale={s:8.1f}  beta1={b1:3d}  essentialH1={e}")
