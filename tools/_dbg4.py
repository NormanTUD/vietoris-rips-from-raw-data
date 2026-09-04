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


pts = V.generators.circle(250, seed=0)
D = V.pairwise_distances(pts)
nn = nn(D)
print("250-circle nn=%.5f" % nn)
for f in [1.0, 2.0, 3.0, 5.0, 8.0]:
    eps = f * nn
    C = V.build_rips(pts, D, eps, max_dim=2)
    seen = set(C.simplexes)
    dup = len(seen) != C.n_simplices
    cnt = {d: C.count(d) for d in range(3)}
    bc = V.persistent_homology(C)
    b_bar = bc.betti_at(eps)
    b_rank = V.betti_at(C, eps)
    print("eps=%.5f cnt=%s rank=%s bar=%s dup=%s agree=%s"
          % (eps, cnt, b_rank, b_bar, dup, b_rank[:len(b_bar)] == b_bar))
