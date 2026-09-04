# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.26"]
# ///
import sys, time
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


for (nu, nv) in [(24, 12), (32, 16)]:
    pts = V.generators.donut_grid(nu, nv)
    D = V.pairwise_distances(pts)
    nnn = nn(D)
    print(f"\ndonut_grid({nu},{nv}) n={len(pts)} nn={nnn:.4f}")
    for f in [1, 2, 3, 4, 5, 6, 8]:
        eps = f * nnn
        t0 = time.time()
        try:
            C = V.build_rips(pts, D, eps, max_dim=2, max_simplices=1_500_000)
            dt = time.time() - t0
            b = V.betti_at(C, eps)
            print(f"  f={f:2d} eps={eps:.4f} nsimp={C.n_simplices:7d} betti={b}  ({dt:.2f}s)")
        except V.TooLargeError as e:
            print(f"  f={f:2d} eps={eps:.4f} TOO LARGE ({time.time()-t0:.2f}s)")
