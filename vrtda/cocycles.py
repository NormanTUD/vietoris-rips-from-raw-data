from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from vrtda.persistence import Barcode


@dataclass
class Loop:
    birth_simplex: int
    value: float
    vertices: list[int]
    edges: set = field(default_factory=set)
    birth: float = 0.0
    death: float = np.inf

    @property
    def tokens(self) -> frozenset:
        return frozenset(self.vertices)


def _path(u: int, v: int, parent: dict) -> list[int]:
    seen = {}
    cur = u
    while cur is not None:
        seen[cur] = True
        cur = parent.get(cur)
    lca = None
    cur = v
    while cur is not None and cur not in seen:
        cur = parent.get(cur)
    lca = cur
    left, cur = [], u
    while cur != lca:
        left.append(cur)
        cur = parent[cur]
    left.append(lca)
    right, cur = [], v
    while cur != lca:
        right.append(cur)
        cur = parent[cur]
    right.append(lca)
    return left + right[::-1][1:]


def _edges_from_path(path: list[int], u: int, v: int) -> set:
    es = set()
    for a, b in zip(path, path[1:]):
        es.add(frozenset((a, b)))
    es.add(frozenset((u, v)))
    return es


def loops_1skeleton(complex, eps_max: float | None = None) -> list[Loop]:
    """Fundamental cycle basis of the 1-skeleton at eps_max (spanning-forest).

    Each independent 1-cycle is represented by the unique tree path between the
    endpoints of the edge that closes it, plus that edge. The closing edge's index
    is the H_1 birth simplex, so this basis aligns with the persistence barcode."""
    edges = []
    for j, s in enumerate(complex.simplexes):
        if complex.dims[j] == 1 and (eps_max is None or complex.values[j] <= eps_max + 1e-15):
            u, v = int(s[0]), int(s[1])
            edges.append((float(complex.values[j]), j, u, v))
    edges.sort(key=lambda e: (e[0], e[1]))

    parent: dict[int, int] = {}

    def root(n: int) -> int:
        while parent.get(n) is not None:
            n = parent[n]
        return n

    loops: list[Loop] = []
    for value, idx, u, v in edges:
        ru, rv = root(u), root(v)
        if ru == rv:
            path = _path(u, v, parent)
            loops.append(Loop(birth_simplex=idx, value=value, vertices=path, edges=_edges_from_path(path, u, v)))
        else:
            parent[ru] = rv
    return loops


def persistent_loops(complex, barcode: Barcode, eps_max: float | None = None, essential_only: bool = True) -> list[Loop]:
    """Loops corresponding to the H_1 intervals of `barcode` (via birth_simplex)."""
    basis = {lp.birth_simplex: lp for lp in loops_1skeleton(complex, eps_max)}
    out = []
    for iv in barcode.of_dim(1):
        if essential_only and not iv.is_essential:
            continue
        lp = basis.get(iv.birth_simplex)
        if lp is not None:
            lp = Loop(**{**lp.__dict__, "birth": float(iv.birth), "death": float(iv.death)})
            out.append(lp)
    return out
