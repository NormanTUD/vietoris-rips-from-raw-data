from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from vrtda import debug
from vrtda.complexes import FilteredComplex


def gf2_rank(cols: list[int]) -> int:
    reduced = list(cols)
    pivot_col: dict[int, int] = {}
    rank = 0
    for j in range(len(reduced)):
        c = reduced[j]
        while c:
            i = c.bit_length() - 1
            p = pivot_col.get(i)
            if p is not None:
                c ^= reduced[p]
            else:
                break
        reduced[j] = c
        if c:
            pivot_col[c.bit_length() - 1] = j
            rank += 1
    return rank


def _boundary_columns(complex: FilteredComplex, keep: set[int], pos: dict[int, int], k: int) -> list[int]:
    if k <= 0:
        return []
    row_dim = k - 1
    rows = [j for j in keep if complex.dims[j] == row_dim]
    row_pos = {old: new for new, old in enumerate(rows)}
    cols = []
    for j in sorted(keep, key=lambda x: (complex.values[x], complex.dims[x])):
        if complex.dims[j] != k:
            continue
        mask = 0
        for face in complex.boundary_faces(j):
            if complex.dims[face] == row_dim and face in row_pos:
                mask |= 1 << row_pos[face]
        cols.append(mask)
    return cols


def betti_at(complex: FilteredComplex, eps: float) -> list[int]:
    keep = {j for j in range(complex.n_simplices) if complex.values[j] <= eps + 1e-15}
    maxk = max((int(complex.dims[j]) for j in keep), default=-1)
    pos = {old: new for new, old in enumerate(sorted(keep))}
    out = []
    for k in range(0, maxk + 1):
        n_k = sum(1 for j in keep if complex.dims[j] == k)
        rank_k = gf2_rank(_boundary_columns(complex, keep, pos, k)) if k >= 1 else 0
        rank_k1 = gf2_rank(_boundary_columns(complex, keep, pos, k + 1))
        beta = (n_k - rank_k) - rank_k1
        assert beta >= 0, f"negative Betti beta_{k}={beta} at eps={eps}"
        out.append(int(beta))
    return out


def betti_function(complex: FilteredComplex, epsilons: Sequence[float] | np.ndarray) -> np.ndarray:
    epsilons = list(epsilons)
    maxk = int(complex.dims.max()) if complex.n_simplices else 0
    arr = np.zeros((len(epsilons), maxk + 1), dtype=np.int64)
    for r, e in enumerate(epsilons):
        b = betti_at(complex, e)
        for c in range(len(b)):
            arr[r, c] = b[c]
    return arr


def euler_characteristic(complex: FilteredComplex, eps: float) -> int:
    b = betti_at(complex, eps)
    return sum((-1) ** k * b[k] for k in range(len(b)))


from vrtda.beartype_guard import beartype_module as _beartype_module

_beartype_module(__name__)
