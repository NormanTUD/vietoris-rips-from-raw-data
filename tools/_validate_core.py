# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy>=1.26",
# ]
# ///
import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import vrtda as V


def nn_mean(pts, D):
    d = D.copy()
    np.fill_diagonal(d, np.inf)
    return float(d.min(axis=1).mean())


def sweep(name, pts, max_dim, fracs, Fmax):
    D = V.pairwise_distances(pts)
    nn = nn_mean(pts, D)
    eps_max = Fmax * nn
    C = V.build_rips(pts, D, eps_max, max_dim=max_dim)
    bc = V.persistent_homology(C)
    grid = [f * nn for f in fracs]
    print(f"\n== {name}  n={len(pts)} nn={nn:.4f} eps_max={eps_max:.4f} nsimp={C.n_simplices} ==")
    for e in grid:
        b = bc.betti_at(e)
        print(f"   eps={e:7.4f}  betti={b}")
    ess = {d: len([iv for iv in bc.of_dim(d) if iv.is_essential]) for d in range(max_dim + 1)}
    print(f"   essential(infinite) counts per dim: {ess}")
    return bc


def main():
    V.sweep = None
    # circle: expect plateau H=(1,1)
    sweep("circle", V.generators.circle(250, seed=0), 2, [1, 1.5, 2, 3, 4, 6, 9, 14], 20)
    # donut T^2: expect plateau (1,2,1)
    sweep("donut T^2", V.generators.donut(500, seed=1), 2, [1.5, 2, 3, 4, 6, 9, 14, 22], 40)
    # product T^3: expect plateau (1,3,3,1)
    sweep("product T^3", V.generators.product_torus(3, 700, seed=2), 3, [2, 3, 4, 6, 8, 12], 20)
    print("\nDONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
