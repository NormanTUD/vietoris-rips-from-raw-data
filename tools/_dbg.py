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
import vrtda as V


def nn(D):
    d = D.copy()
    np.fill_diagonal(d, np.inf)
    return float(d.min(1).mean())


pts = V.generators.circle(6, radius=1.0, seed=0)
ang = np.arctan2(pts[:, 1], pts[:, 0])
pts = pts[np.argsort(ang)]
D = V.pairwise_distances(pts)
side = D[0, 1]
diag = D[0, 2]
print("hexagon side=%.4f diag=%.4f" % (side, diag))
for eps in [side * 1.01, diag * 0.99, diag * 1.01]:
    C = V.build_rips(pts, D, eps, max_dim=2)
    bc = V.persistent_homology(C)
    b_bar = bc.betti_at(eps)
    b_rank = V.betti_at(C, eps)
    cnt = {d: C.count(d) for d in range(3)}
    print("eps=%.4f counts=%s betti_barcode=%s betti_rank=%s match=%s"
          % (eps, cnt, b_bar, b_rank, b_bar == b_rank))
