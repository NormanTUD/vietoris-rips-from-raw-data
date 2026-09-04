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


def check(name, simplices, expect, eps=None):
    C = V.FilteredComplex.from_explicit(simplices)
    bc = V.persistent_homology(C)
    if eps is None:
        eps = float(C.values.max())
    b_bar = bc.betti_at(eps)
    b_rank = V.betti_at(C, eps)
    b_coho = V.cohomology_at(C, eps)
    ok = b_rank == expect and b_bar[: len(expect)] == expect[: len(b_bar)] and b_coho == expect
    print(f"{name}: rank={b_rank} bar={b_bar} coho={b_coho} expect={expect} OK={ok}")
    assert ok, f"{name} FAILED: rank={b_rank} expect={expect}"
    return C


# 1) single triangle = disk: H=(1,0,0)
check(
    "triangle(disk)",
    [
        (0.0, 0, (0,)), (0.0, 0, (1,)), (0.0, 0, (2,)),
        (1.0, 1, (0, 1)), (1.0, 1, (1, 2)), (1.0, 1, (0, 2)),
        (2.0, 2, (0, 1, 2)),
    ],
    [1, 0, 0],
)

# 2) tetrahedron boundary = sphere S^2: H=(1,0,1)
check(
    "tetra-boundary(sphere)",
    [
        (0.0, 0, (0,)), (0.0, 0, (1,)), (0.0, 0, (2,)), (0.0, 0, (3,)),
        (1.0, 1, (0, 1)), (1.0, 1, (0, 2)), (1.0, 1, (0, 3)),
        (1.0, 1, (1, 2)), (1.0, 1, (1, 3)), (1.0, 1, (2, 3)),
        (2.0, 2, (0, 1, 2)), (2.0, 2, (0, 1, 3)), (2.0, 2, (0, 2, 3)), (2.0, 2, (1, 2, 3)),
    ],
    [1, 0, 1],
)

# 3) full tetrahedron = contractible: H=(1,0,0,0)
check(
    "tetra(solid)",
    [
        (0.0, 0, (0,)), (0.0, 0, (1,)), (0.0, 0, (2,)), (0.0, 0, (3,)),
        (1.0, 1, (0, 1)), (1.0, 1, (0, 2)), (1.0, 1, (0, 3)),
        (1.0, 1, (1, 2)), (1.0, 1, (1, 3)), (1.0, 1, (2, 3)),
        (2.0, 2, (0, 1, 2)), (2.0, 2, (0, 1, 3)), (2.0, 2, (0, 2, 3)), (2.0, 2, (1, 2, 3)),
        (3.0, 3, (0, 1, 2, 3)),
    ],
    [1, 0, 0, 0],
)

print("ALL ABSTRACT OK")
