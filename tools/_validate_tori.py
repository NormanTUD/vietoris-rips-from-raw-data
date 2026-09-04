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


def sweep(name, pts, max_dim, fracs, Fmax, target):
    D = V.pairwise_distances(pts)
    nnn = nn(D)
    eps_max = Fmax * nnn
    C = V.build_rips(pts, D, eps_max, max_dim=max_dim)
    bc = V.persistent_homology(C)
    print(f"\n== {name}  n={len(pts)} nn={nnn:.5f} eps_max={eps_max:.4f} nsimp={C.n_simplices} target={target} ==")
    hit = None
    for f in fracs:
        e = f * nnn
        b = bc.betti_at(e)
        ok = b[: len(target)] == target
        mark = "  <-- MATCH" if ok else ""
        if ok and hit is None:
            hit = e
        print(f"   eps={e:8.4f} betti={b}{mark}")
    print(f"   plateau found at eps={hit}")
    return hit


def main():
    # donut T^2: expect (1,2,1)
    sweep("donut_grid T^2", V.generators.donut_grid(40, 16), 2,
          [1, 1.5, 2, 3, 4, 5, 6, 8, 10], 14, [1, 2, 1])
    # product T^3 in R^6: expect (1,3,3,1)
    sweep("product T^3 grid", V.generators.product_torus_grid(3, 8), 3,
          [1, 1.5, 2, 3, 4, 5, 6], 10, [1, 3, 3, 1])
    # product T^2 in R^4: expect (1,2,1)
    sweep("product T^2 grid", V.generators.product_torus_grid(2, 20), 2,
          [1, 1.5, 2, 3, 4, 5], 8, [1, 2, 1])
    print("\nDONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
