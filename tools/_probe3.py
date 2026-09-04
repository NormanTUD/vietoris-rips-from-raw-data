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


def sweep(name, pts, max_dim, fracs, Fmax, target):
    D = V.pairwise_distances(pts)
    nnn = nn(D)
    eps_max = Fmax * nnn
    t0 = time.time()
    C = V.build_rips(pts, D, eps_max, max_dim=max_dim)
    t_build = time.time() - t0
    t0 = time.time()
    bc = V.persistent_homology(C)
    t_pers = time.time() - t0
    print(f"\n== {name} n={len(pts)} nn={nnn:.4f} eps_max={eps_max:.4f} nsimp={C.n_simplices} "
          f"build={t_build:.2f}s pers={t_pers:.2f}s target={target} ==", flush=True)
    hit = None
    for f in fracs:
        e = f * nnn
        b = bc.betti_at(e)
        ok = b[: len(target)] == target
        if ok and hit is None:
            hit = e
        print(f"   eps={e:8.4f} betti={b}{'  <-- MATCH' if ok else ''}", flush=True)
    print(f"   plateau at eps={hit}", flush=True)
    return hit


def main():
    sweep("donut T^2", V.generators.donut_grid(28, 14), 2, [1, 2, 3, 4, 5, 6, 8], 8, [1, 2, 1])
    print("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
