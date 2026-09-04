from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from vrtda import debug
from vrtda.complexes import FilteredComplex
from vrtda.errors import VrtdaError


@dataclass
class Interval:
    birth: float
    death: float  # np.inf for essential
    dim: int
    birth_simplex: int
    death_simplex: int = -1

    @property
    def is_essential(self) -> bool:
        return not np.isfinite(self.death)

    @property
    def length(self) -> float:
        return self.death - self.birth

    def as_tuple(self) -> tuple[int, float, float]:
        return (self.dim, self.birth, self.death)

    def alive_at(self, eps: float) -> bool:
        return bool(self.birth <= eps < self.death)


@dataclass
class Barcode:
    intervals: list[Interval] = field(default_factory=list)
    values: np.ndarray | None = None
    dims: np.ndarray | None = None

    def of_dim(self, dim: int) -> list[Interval]:
        return [iv for iv in self.intervals if iv.dim == dim]

    def max_dim(self) -> int:
        return max((iv.dim for iv in self.intervals), default=-1)

    def betti_at(self, eps: float) -> list[int]:
        md = self.max_dim()
        out = []
        for d in range(md + 1):
            out.append(sum(1 for iv in self.of_dim(d) if iv.alive_at(eps)))
        return out

    def betti_function(self, epsilons: Sequence[float] | np.ndarray) -> np.ndarray:
        epsilons = list(epsilons)
        md = self.max_dim()
        arr = np.zeros((len(epsilons), md + 1), dtype=np.int64)
        for col, d in enumerate(range(md + 1)):
            ivs = self.of_dim(d)
            for r, e in enumerate(epsilons):
                arr[r, col] = sum(1 for iv in ivs if iv.alive_at(e))
        return arr

    def summary(self) -> dict:
        md = self.max_dim()
        out = {"n_intervals": len(self.intervals), "dims": {}}
        for d in range(md + 1):
            ivs = self.of_dim(d)
            ess = [iv for iv in ivs if iv.is_essential]
            out["dims"][d] = {
                "n": len(ivs),
                "essential": len(ess),
                "max_length": max((iv.length for iv in ivs), default=0.0),
            }
        return out


def persistent_homology(complex: FilteredComplex) -> Barcode:
    n = complex.n_simplices
    cols: list[set[int]] = []
    for j in range(n):
        s = set()
        for i in complex.boundary_faces(j):
            assert i < j, f"boundary face {i} must precede simplex {j}"
            s.add(i)
        cols.append(s)

    pivot_col: dict[int, int] = {}
    pairs: list[tuple[int, int]] = []
    births: set[int] = set()
    with debug.timing(f"persistent_homology n_simplices={n}"):
        for j in range(n):
            c = cols[j]
            while c:
                i = max(c)
                p = pivot_col.get(i)
                if p is not None:
                    c = c ^ cols[p]
                else:
                    break
            cols[j] = c
            if c:
                i = max(c)
                pivot_col[i] = j
                pairs.append((i, j))
            else:
                births.add(j)

    used_birth = {i for (i, _) in pairs}
    infinite = [j for j in births if j not in used_birth]

    values = complex.values
    dims = complex.dims
    intervals: list[Interval] = []
    for (i, j) in pairs:
        intervals.append(
            Interval(
                birth=float(values[i]),
                death=float(values[j]),
                dim=int(dims[i]),
                birth_simplex=i,
                death_simplex=j,
            )
        )
    for j in infinite:
        intervals.append(
            Interval(
                birth=float(values[j]),
                death=np.inf,
                dim=int(dims[j]),
                birth_simplex=j,
            )
        )
    intervals.sort(key=lambda iv: (iv.dim, iv.birth, iv.death))
    return Barcode(intervals=intervals, values=values.copy(), dims=dims.copy())


from vrtda.beartype_guard import beartype_module as _beartype_module

_beartype_module(__name__)
