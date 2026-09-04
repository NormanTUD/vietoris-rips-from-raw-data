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
    d = D.copy(); np.fill_diagonal(d, np.inf)
    return float(d.min(1).mean())


def probe(name, pts, max_dim, n=60, lo=0.9, hi=2.2):
    D = V.pairwise_distances(pts); nnn = nn(D)
    C = V.build_rips(pts, D, hi * nnn, max_dim=max_dim)
    bc = V.persistent_homology(C)
    eps = np.linspace(lo * nnn, hi * nnn, n)
    maxb1 = 0; hit = None
    for e in eps:
        b = bc.betti_at(e)
        if b[0] == 1:
            maxb1 = max(maxb1, b[1])
            if hit is None and len(b) > 1 and b[1] == 3:
                hit = round(float(e), 4)
    print(f"{name:20s} n={len(pts):3d} nn={nnn:.4f} maxb1(conn)={maxb1:3d} hit_b1=3_at={hit}")


def main():
    for nper in [3, 4, 5, 6]:
        probe(f"T3 nper={nper}", V.generators.product_torus_grid(3, nper), 3)
    print("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
