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


pts = V.generators.donut_grid(24, 12)
t0 = time.time()
D = V.pairwise_distances(pts)
print("pairwise: %.3fs" % (time.time() - t0), flush=True)
nnn = nn(D)
print("n=%d nn=%.4f" % (len(pts), nnn), flush=True)
for f in [1, 2, 3, 4]:
    eps = f * nnn
    A = (D <= eps + 1e-15).copy()
    np.fill_diagonal(A, False)
    ecount = int(A.sum() // 2)
    t0 = time.time()
    try:
        C = V.build_rips(pts, D, eps, max_dim=2, max_simplices=1_000_000)
        dt = time.time() - t0
        print("f=%d eps=%.4f edges=%d nsimp=%d (%.2fs)" % (f, eps, ecount, C.n_simplices, dt), flush=True)
    except V.TooLargeError:
        print("f=%d eps=%.4f edges=%d TOO BIG (%.2fs)" % (f, eps, ecount, time.time() - t0), flush=True)
