from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from vrtda import homology as H
from vrtda.complexes import FilteredComplex


def _coboundary_columns(complex: FilteredComplex, keep: set[int], k: int) -> list[int]:
    if k < 0:
        return []
    col_dim = k
    row_dim = k + 1
    cols_list = [j for j in sorted(keep, key=lambda x: (complex.values[x], complex.dims[x]))
                 if complex.dims[j] == col_dim]
    col_pos = {old: new for new, old in enumerate(cols_list)}
    rows = [j for j in sorted(keep, key=lambda x: (complex.values[x], complex.dims[x]))
            if complex.dims[j] == row_dim]
    cols = [0] * len(cols_list)
    for r, row in enumerate(rows):
        rbit = 1 << r
        for face in complex.boundary_faces(row):
            if complex.dims[face] == col_dim and face in col_pos:
                cols[col_pos[face]] |= rbit
    return cols


def cohomology_at(complex: FilteredComplex, eps: float) -> list[int]:
    keep = {j for j in range(complex.n_simplices) if complex.values[j] <= eps + 1e-15}
    maxk = max((int(complex.dims[j]) for j in keep), default=-1)
    out = []
    for k in range(0, maxk + 1):
        n_k = sum(1 for j in keep if complex.dims[j] == k)
        rank_k = H.gf2_rank(_coboundary_columns(complex, keep, k))
        rank_km1 = H.gf2_rank(_coboundary_columns(complex, keep, k - 1))
        beta = (n_k - rank_k) - rank_km1
        assert beta >= 0, f"negative cohomology beta^{k}={beta} at eps={eps}"
        out.append(int(beta))
    return out


def cohomology_function(complex: FilteredComplex, epsilons: Sequence[float] | np.ndarray) -> np.ndarray:
    epsilons = list(epsilons)
    maxk = int(complex.dims.max()) if complex.n_simplices else 0
    arr = np.zeros((len(epsilons), maxk + 1), dtype=np.int64)
    for r, e in enumerate(epsilons):
        b = cohomology_at(complex, e)
        for c in range(len(b)):
            arr[r, c] = b[c]
    return arr


def assert_homology_cohomology_match(complex: FilteredComplex, epsilons: Sequence[float] | np.ndarray) -> None:
    h = H.betti_function(complex, epsilons)
    c = cohomology_function(complex, epsilons)
    assert h.shape == c.shape
    assert np.array_equal(h, c), f"homology {h.tolist()} != cohomology {c.tolist()}"


from vrtda.beartype_guard import beartype_module as _beartype_module

_beartype_module(__name__)
