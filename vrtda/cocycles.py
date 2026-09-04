from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from vrtda.beartype_guard import beartype_function
from vrtda.complexes import FilteredComplex
from vrtda.persistence import Barcode


@dataclass
class Loop:
    birth_simplex: int
    value: float
    vertices: list[int]
    edges: set[frozenset[int]] = field(default_factory=set)
    birth: float = 0.0
    death: float = np.inf

    @property
    def tokens(self) -> frozenset[int]:
        return frozenset(self.vertices)


def _edges_from_path(path: list[int], u: int, v: int) -> set[frozenset[int]]:
    es: set[frozenset[int]] = set()
    for a, b in zip(path, path[1:]):
        es.add(frozenset((a, b)))
    es.add(frozenset((u, v)))
    return es


def _tree_path(u: int, v: int, adj: dict[int, list[int]]) -> list[int]:
    from collections import deque

    prev = {u: None}
    q = deque([u])
    while q:
        x = q.popleft()
        if x == v:
            break
        for y in adj.get(x, ()):
            if y not in prev:
                prev[y] = x
                q.append(y)
    if v not in prev:
        return []
    path, cur = [], v
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


def loops_1skeleton(complex: FilteredComplex, eps_max: float | None = None) -> list[Loop]:
    """Fundamental cycle basis of the 1-skeleton at eps_max (spanning-forest).

    Each independent 1-cycle is represented by the unique tree path between the
    endpoints of the edge that closes it, plus that edge. The closing edge's index
    is the H_1 birth simplex, so this basis aligns with the persistence barcode.
    A real forest adjacency (not just union-find roots) is kept so the path is a
    genuine graph path of actual edges."""
    edges = []
    for j, s in enumerate(complex.simplexes):
        if complex.dims[j] == 1 and (eps_max is None or complex.values[j] <= eps_max + 1e-15):
            u, v = int(s[0]), int(s[1])
            edges.append((float(complex.values[j]), j, u, v))
    edges.sort(key=lambda e: (e[0], e[1]))

    parent: dict[int, int] = {}
    adj: dict[int, list[int]] = {}

    def root(n: int) -> int:
        while parent.get(n, n) != n:
            parent[n] = parent.get(parent[n], parent[n])
            n = parent.get(n, n)
        return n

    root = beartype_function(root)
    loops: list[Loop] = []
    for value, idx, u, v in edges:
        ru, rv = root(u), root(v)
        if ru == rv:
            path = _tree_path(u, v, adj)
            loops.append(Loop(birth_simplex=idx, value=value, vertices=path, edges=_edges_from_path(path, u, v)))
        else:
            parent[ru] = rv
            adj.setdefault(u, []).append(v)
            adj.setdefault(v, []).append(u)
    return loops


def persistent_loops(
    complex: FilteredComplex,
    barcode: Barcode,
    eps_max: float | None = None,
    essential_only: bool = True,
) -> list[Loop]:
    """Loops corresponding to the H_1 intervals of `barcode` (via birth_simplex)."""
    basis = {lp.birth_simplex: lp for lp in loops_1skeleton(complex, eps_max)}
    out: list[Loop] = []
    for iv in barcode.of_dim(1):
        if essential_only and not iv.is_essential:
            continue
        lp = basis.get(iv.birth_simplex)
        if lp is not None:
            lp = Loop(**{**lp.__dict__, "birth": float(iv.birth), "death": float(iv.death)})
            out.append(lp)
    return out


from vrtda.beartype_guard import beartype_module as _beartype_module

_beartype_module(__name__)
