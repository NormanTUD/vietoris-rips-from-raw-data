from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from itertools import combinations, permutations

import numpy as np

from vrtda import debug
from vrtda import geometry as G
from vrtda.beartype_guard import beartype_function
from vrtda.errors import FiltrationError, TooLargeError


class FilteredComplex:
    def __init__(
        self,
        simplexes: list[tuple[int, ...]],
        values: np.ndarray,
        dims: np.ndarray,
        kind: str,
        params: dict[str, object],
    ) -> None:
        self.simplexes = simplexes
        self.values = np.asarray(values, dtype=np.float64)
        self.dims = np.asarray(dims, dtype=np.int64)
        self.kind = kind
        self.params = dict(params)
        n = len(simplexes)
        assert self.values.shape == (n,)
        assert self.dims.shape == (n,)
        self._index = {s: i for i, s in enumerate(simplexes)}
        self._face_cache: dict[int, list[int]] = {}
        self._validate_ordering()

    def _validate_ordering(self) -> None:
        for j, s in enumerate(self.simplexes):
            for face in self._faces_of(s):
                fi = self._index[face]
                if fi >= j:
                    raise FiltrationError(
                        f"face {face} (idx {fi}) of simplex {s} (idx {j}) must appear earlier "
                        f"(filtration not monotone). values: face={self.values[fi]}, sf={self.values[j]}"
                    )

    @staticmethod
    def _faces_of(s: tuple[int, ...]) -> list[tuple[int, ...]]:
        faces: list[tuple[int, ...]] = []
        for a in range(len(s)):
            face = s[:a] + s[a + 1:]
            if len(face) >= 1:
                faces.append(face)
        return faces

    @property
    def n_simplices(self) -> int:
        return len(self.simplexes)

    def count(self, dim: int) -> int:
        return int((self.dims == dim).sum())

    def max_dim(self) -> int:
        return int(self.dims.max()) if len(self.dims) else -1

    def index_of(self, s: tuple[int, ...]) -> int:
        return self._index[s]

    def boundary_faces(self, j: int) -> list[int]:
        cached = self._face_cache.get(j)
        if cached is not None:
            return cached
        faces = [self._index[face] for face in self._faces_of(self.simplexes[j])]
        self._face_cache[j] = faces
        return faces

    def boundary_columns(self) -> list[set[int]]:
        return [set(self.boundary_faces(j)) for j in range(self.n_simplices)]

    def summary(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "params": self.params,
            "n_simplices": self.n_simplices,
            "counts": {d: self.count(d) for d in range(self.max_dim() + 1)},
            "value_min": float(self.values.min()) if self.n_simplices else 0.0,
            "value_max": float(self.values.max()) if self.n_simplices else 0.0,
        }

    def __repr__(self) -> str:
        c = {d: self.count(d) for d in range(self.max_dim() + 1)}
        return f"FilteredComplex(kind={self.kind}, n={self.n_simplices}, counts={c})"

    @classmethod
    def from_explicit(
        cls,
        simplices: Iterable[tuple[int, ...]],
        kind: str = "explicit",
        params: dict[str, object] | None = None,
    ) -> "FilteredComplex":
        return _sort_and_build(list(simplices), kind, params or {})


def _sort_and_build(
    simplexes: list[tuple[float, int, tuple[int, ...]]],
    kind: str,
    params: dict[str, object],
) -> FilteredComplex:
    simplexes = sorted(simplexes, key=lambda t: (t[0], t[1], t[2]))
    values = np.array([t[0] for t in simplexes], dtype=np.float64)
    dims = np.array([t[1] for t in simplexes], dtype=np.int64)
    verts = [t[2] for t in simplexes]
    return FilteredComplex(verts, values, dims, kind, params)


def _enumerate_cliques(
    A: np.ndarray,
    D: np.ndarray,
    max_dim: int,
    max_simplices: int,
    value_fn: Callable[[tuple[int, ...]], float],
) -> list[tuple[float, int, tuple[int, ...]]]:
    n = A.shape[0]
    out: list[tuple] = [(0.0, 0, (i,)) for i in range(n)]
    if max_simplices and len(out) > max_simplices:
        raise TooLargeError(f"vertices {n} exceed max_simplices {max_simplices}")

    def add_batch(tri: np.ndarray, dim: int, vals: np.ndarray) -> None:
        nonlocal out
        T = tri.shape[0]
        if T == 0:
            return
        if max_simplices and len(out) + T > max_simplices:
            raise TooLargeError(
                f"simplices exceeded max_simplices={max_simplices}; lower eps_max or max_dim "
                f"(currently {len(out)}, adding {T} dim-{dim})"
            )
        for t, v in zip(tri, vals):
            out.append((float(v), dim, tuple(int(x) for x in t)))

    add_batch = beartype_function(add_batch)

    if max_dim >= 1:
        ii, jj = np.triu_indices(n, 1)
        m = A[ii, jj]
        if m.any():
            e = np.column_stack([ii[m], jj[m]])
            add_batch(e, 1, D[ii[m], jj[m]])

    if max_dim >= 2:
        for i in range(n - 1):
            S = np.where(A[i, i + 1:])[0] + (i + 1)
            if S.size < 2:
                continue
            sub = A[np.ix_(S, S)]
            jj, kk = np.triu_indices(S.size, 1)
            m = sub[jj, kk]
            if not m.any():
                continue
            a = S[jj[m]]
            b = S[kk[m]]
            tri = np.column_stack([np.full(a.size, i), a, b])
            vals = np.maximum(
                np.maximum(D[tri[:, 0], tri[:, 1]], D[tri[:, 0], tri[:, 2]]),
                D[tri[:, 1], tri[:, 2]],
            )
            add_batch(tri, 2, vals)

    if max_dim >= 3:
        triangles = np.array([s for (v, d, s) in out if d == 2], dtype=int)
        if triangles.size:
            for (i, j, k) in triangles:
                cand = A[i] & A[j] & A[k]
                cand[: k + 1] = False
                l = np.where(cand)[0]
                if l.size == 0:
                    continue
                tri = np.column_stack([
                    np.full(l.size, i), np.full(l.size, j), np.full(l.size, k), l
                ])
                vals = np.maximum(
                    np.maximum(
                        np.maximum(D[i, j], D[i, k]), np.maximum(D[j, k], D[i, l])
                    ),
                    np.maximum(D[j, l], D[k, l]),
                )
                add_batch(tri, 3, vals)

    if max_dim > 3:
        raise FiltrationError(f"max_dim={max_dim} > 3 not supported for clique enumeration")
    return out


def build_rips(
    X: np.ndarray,
    D: np.ndarray,
    eps_max: float,
    max_dim: int = 2,
    max_simplices: int = 2_000_000,
) -> FilteredComplex:
    if eps_max < 0:
        raise FiltrationError("eps_max must be >= 0")
    A = (D <= eps_max + 1e-15).copy()
    np.fill_diagonal(A, False)

    def value_fn(s: tuple[int, ...]) -> float:
        idx = list(s)
        v = 0.0
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                v = max(v, D[idx[a], idx[b]])
        return float(v)

    value_fn = beartype_function(value_fn)

    with debug.timing(f"build_rips n={D.shape[0]} eps={eps_max} dim<={max_dim}"):
        simplexes = _enumerate_cliques(A, D, max_dim, max_simplices, value_fn)
    return _sort_and_build(simplexes, "rips", {"eps_max": eps_max, "max_dim": max_dim})


def build_vietoris(
    X: np.ndarray,
    D: np.ndarray,
    r: float,
    max_dim: int = 2,
    max_simplices: int = 2_000_000,
) -> FilteredComplex:
    if r < 0:
        raise FiltrationError("r must be >= 0")
    # MEB radius <= r  =>  every pairwise distance <= 2*r, so an edge-adjacency
    # threshold of 2*r + 2*TOL is a guaranteed superset of every kept simplex's
    # faces. Using 2*TOL (not TOL) keeps the complex closed under faces.
    TOL = 1e-9 * max(1.0, r)
    A = (D <= 2.0 * r + 2.0 * TOL).copy()
    np.fill_diagonal(A, False)

    cand = _enumerate_cliques(
        A, D, max_dim, max_simplices, lambda s: 0.0
    )
    kept = []
    for v, d, s in cand:
        if d == 0:
            kept.append((0.0, 0, s))
            continue
        rad = G.min_enclosing_ball_radius(X[list(s)])
        if rad <= r + TOL:
            kept.append((rad, d, s))
    return _sort_and_build(kept, "vietoris", {"r": r, "max_dim": max_dim})


def make_torus_grid_complex(k: int, cells: Sequence[int], name: str = "torus_grid") -> FilteredComplex:
    if k < 2:
        raise FiltrationError("torus_grid requires k >= 2")
    cells = tuple(int(c) for c in cells)
    if len(cells) != k:
        raise FiltrationError(f"cells length {len(cells)} != k={k}")
    if any(c < 2 for c in cells):
        raise FiltrationError("each cell count must be >= 2")

    def vidx(coord: Sequence[int]) -> int:
        idx = 0
        stride = 1
        for a in range(k):
            idx += coord[a] * stride
            stride *= cells[a]
        return idx

    vidx = beartype_function(vidx)
    simp_set: set[tuple[int, ...]] = set()

    def add_simplex(verts: list[tuple[int, ...]]) -> None:
        if len(verts) < 1:
            return
        if len(verts) == 1:
            s = (vidx(verts[0]),)
        else:
            s = tuple(sorted(vidx(v) for v in verts))
        if len(set(s)) != len(s):
            return
        for r in range(1, len(s) + 1):
            for face in combinations(s, r):
                simp_set.add(face)

    add_simplex = beartype_function(add_simplex)

    from itertools import product as _product
    for o in _product(*[range(c) for c in cells]):
        o = list(o)
        for perm in permutations(range(k)):
            acc = list(o)
            verts = [tuple(acc)]
            for a in perm:
                acc = acc[:]
                acc[a] = (acc[a] + 1) % cells[a]
                verts.append(tuple(acc))
            add_simplex(verts)

    ordered = []
    for s in simp_set:
        d = len(s) - 1
        ordered.append((float(d), d, s))
    with debug.timing(f"make_torus_grid_complex k={k} cells={cells}"):
        return _sort_and_build(ordered, "torus_grid", {"k": k, "cells": cells, "name": name})


from vrtda.beartype_guard import beartype_module as _beartype_module

_beartype_module(__name__)
