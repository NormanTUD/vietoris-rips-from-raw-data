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


def main():
    for k, cells, target in [(2, (3, 3), [1, 2, 1]), (2, (4, 5), [1, 2, 1]), (3, (3, 3, 3), [1, 3, 3, 1])]:
        C = V.complexes.make_torus_grid_complex(k, cells)
        eps = float(C.values.max())
        b_rank = V.betti_at(C, eps)
        bc = V.persistent_homology(C)
        b_bar = bc.betti_at(eps)
        b_coho = V.cohomology_at(C, eps)
        ok = b_rank == target and b_coho == target
        print(f"torus_grid k={k} cells={cells}: nsimp={C.n_simplices} rank={b_rank} bar={b_bar} coho={b_coho} expect={target} OK={ok}")
        assert ok, f"torus k={k} FAILED rank={b_rank} expect={target}"
        # persistent: essential counts per dim should equal target
        ess = {d: len([iv for iv in bc.of_dim(d) if iv.is_essential]) for d in range(k + 1)}
        ess_list = [ess.get(d, 0) for d in range(k + 1)]
        print(f"   essential per dim: {ess_list} (expect {target})")
        assert ess_list == target, f"essential mismatch {ess_list} != {target}"
    print("ALL ABSTRACT TORUS OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
