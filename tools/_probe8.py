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
from vrtda.errors import TooLargeError


def nn(D):
    d = D.copy(); np.fill_diagonal(d, np.inf)
    return float(d.min(1).mean())


def probe(name, pts, max_dim, hi, n=40):
    D = V.pairwise_distances(pts); nnn = nn(D)
    try:
        C = V.build_rips(pts, D, hi * nnn, max_dim=max_dim)
    except TooLargeError as ex:
        print(f"{name:18s} n={len(pts):3d} TOO_LARGE ({ex})")
        return
    bc = V.persistent_homology(C)
    eps = np.linspace(0.9 * nnn, hi * nnn, n)
    seq = []
    for e in eps:
        b = bc.betti_at(e)
        if b[0] == 1:
            seq.append(b[1])
    maxb1 = max(seq) if seq else -1
    hit = 3 in seq
    print(f"{name:18s} n={len(pts):3d} nsimp={C.n_simplices:5d} nn={nnn:.4f} maxb1(conn)={maxb1:3d} hit_b1=3={hit}")


def main():
    for nper in [3, 4, 5]:
        probe(f"T3 nper={nper}", V.generators.product_torus_grid(3, nper), 3, 1.5)
    print("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
