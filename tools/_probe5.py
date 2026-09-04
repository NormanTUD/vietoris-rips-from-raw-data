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


def fine(name, pts, max_dim, Fmax, fracs, target):
    D = V.pairwise_distances(pts)
    nnn = nn(D)
    eps_max = Fmax * nnn
    C = V.build_rips(pts, D, eps_max, max_dim=max_dim)
    bc = V.persistent_homology(C)
    print(f"== {name} n={len(pts)} nn={nnn:.4f} eps_max={eps_max:.4f} nsimp={C.n_simplices} ==")
    hit = None
    for f in fracs:
        e = f * nnn
        b = bc.betti_at(e)
        ok = b[: len(target)] == target
        if ok and hit is None:
            hit = f
        print("   f=%.2f eps=%.4f betti=%s%s" % (f, e, b, "  <-- MATCH" if ok else ""))
    print("   first match f=%s" % hit)


def main():
    fracs = [round(x, 2) for x in np.arange(1.0, 2.01, 0.1)]
    fine("donut T^2", V.generators.donut_grid(24, 12), 2, 2.0, fracs, [1, 2, 1])
    print("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
