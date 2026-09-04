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

# evenly spaced octagon
n = 8
u = np.linspace(0, 2 * np.pi, n, endpoint=False)
pts = np.column_stack([np.cos(u), np.sin(u)])
D = V.pairwise_distances(pts)
side = D[0, 1]
print("octagon side=%.4f" % side)

# unique simplex check
def inspect(eps, max_dim=2):
    C = V.build_rips(pts, D, eps, max_dim=max_dim)
    # uniqueness
    seen = set(C.simplexes)
    assert len(seen) == C.n_simplices, "DUPLICATE simplices!"
    cnt = {d: C.count(d) for d in range(max_dim + 1)}
    bc = V.persistent_homology(C)
    b_bar = bc.betti_at(eps)
    b_rank = V.betti_at(C, eps)
    print("eps=%.4f cnt=%s rank=%s bar=%s same=%s" % (eps, cnt, b_rank, b_bar, b_rank[:len(b_bar)] == b_bar))

inspect(side * 1.01)      # just the 8 edges -> octagon cycle: expect H=(1,1)
inspect(side * 1.5)
inspect(side * 1.9)
inspect(D[0, 2] * 1.01)   # add near-diagonals
