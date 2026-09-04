from __future__ import annotations

from collections import defaultdict, deque

import numpy as np

from vrtda.persistence import Barcode
from vrtda.persistence_metrics import persistence_diagram


def _points(bc: Barcode, dim: int = 1, top_k: int | None = None) -> np.ndarray:
    d = persistence_diagram(bc, dim)  # (n, 2) finite off-diagonal (birth, death)
    if d.size == 0:
        return np.zeros((0, 2))
    if top_k is not None and len(d) > top_k:
        pers = d[:, 1] - d[:, 0]
        idx = np.argsort(-pers)[:top_k]
        d = d[idx]
    return d


def _heights(P: np.ndarray, p: float) -> np.ndarray:
    """L_p distance of each point to the diagonal y=x."""
    b, d = P[:, 0], P[:, 1]
    return (2.0 * ((d - b) / 2.0) ** p) ** (1.0 / p)


def _lp_matrix(A: np.ndarray, B: np.ndarray, p: float) -> np.ndarray:
    d = np.abs(A[:, None, :] - B[None, :, :])
    return np.sum(d**p, axis=2) ** (1.0 / p)


def _linf_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.max(np.abs(A[:, None, :] - B[None, :, :]), axis=2)


def _edmonds_karp(N: int, adj: list, s: int, t: int) -> int:
    flow = 0
    while True:
        parent = [-1] * N  # parent[v] = (u, i) with i = index of forward edge in adj[u]
        parent[s] = s
        q = deque([s])
        while q and parent[t] == -1:
            u = q.popleft()
            for i, (v, cap, _rev) in enumerate(adj[u]):
                if cap > 0 and parent[v] == -1:
                    parent[v] = (u, i)
                    q.append(v)
        if parent[t] == -1:
            break
        path = []
        v = t
        while v != s:
            u, i = parent[v]
            path.append((u, i))
            v = u
        bot = min(adj[u][i][1] for (u, i) in path)
        for (u, i) in path:
            v = adj[u][i][0]
            adj[u][i][1] -= bot
            adj[v][adj[u][i][2]][1] += bot
        flow += bot
    return flow


def _add_edge(adj: list, u: int, v: int, c: int) -> None:
    if c <= 0:
        return
    adj[u].append([v, c, len(adj[v])])
    adj[v].append([u, 0, len(adj[u]) - 1])


def _bottleneck_feasible(P: np.ndarray, Q: np.ndarray, v: float) -> bool:
    """Is there a matching with every high point matched and low points allowed to
    go to the diagonal? Modelled as a lower-bounded s-t flow feasibility problem."""
    n1, n2 = len(P), len(Q)
    if n1 == 0 and n2 == 0:
        return True
    LINF = _linf_matrix(P, Q)
    hP = (P[:, 1] - P[:, 0]) / 2.0  # L_inf height
    hQ = (Q[:, 1] - Q[:, 0]) / 2.0
    S, T = n1 + n2, n1 + n2 + 1
    N = T + 1
    # edges with (cap, lb)
    edges = []
    for i in range(n1):
        edges.append((S, i, 1, 1 if hP[i] > v else 0))  # p must be accounted (high -> forced)
        for j in range(n2):
            if LINF[i, j] <= v + 1e-15:
                edges.append((i, n1 + j, 1, 0))
        if hP[i] <= v + 1e-15:
            edges.append((i, T, 1, 0))  # p -> diagonal
    for j in range(n2):
        if hQ[j] <= v + 1e-15:
            edges.append((S, n1 + j, 1, 0))  # q -> diagonal
        edges.append((n1 + j, T, 1, 1 if hQ[j] > v else 0))  # q must be accounted (high -> forced)
    # lower-bound transform -> circulation feasibility
    demand = [0] * N
    net = []
    for (u, w, cap, lb) in edges:
        demand[w] += lb
        demand[u] -= lb
        if cap - lb > 0:
            net.append((u, w, cap - lb))
    net.append((T, S, 10**9))
    SS, TT = N, N + 1
    M = N + 2
    adj = [[] for _ in range(M)]
    for (u, w, c) in net:
        _add_edge(adj, u, w, c)
    total = 0
    for x in range(N):
        if demand[x] > 0:
            _add_edge(adj, SS, x, demand[x])
            total += demand[x]
        elif demand[x] < 0:
            _add_edge(adj, x, TT, -demand[x])
    return _edmonds_karp(M, adj, SS, TT) == total


def bottleneck(bc1: Barcode, bc2: Barcode, dim: int = 1, top_k: int | None = None) -> float:
    """Bottleneck distance between the off-diagonal persistence diagrams of two
    barcodes (unmatched points may go to the diagonal). `top_k` restricts each
    diagram to its k most persistent points (faster, and compares the salient
    features rather than the ~thousands of short-lived ones)."""
    P, Q = _points(bc1, dim, top_k), _points(bc2, dim, top_k)
    if len(P) == 0 and len(Q) == 0:
        return 0.0
    cands = {0.0}
    if len(P):
        cands.update(((P[:, 1] - P[:, 0]) / 2.0).tolist())
    if len(Q):
        cands.update(((Q[:, 1] - Q[:, 0]) / 2.0).tolist())
    if len(P) and len(Q):
        cands.update(_linf_matrix(P, Q).ravel().tolist())
    for vv in sorted(cands):
        if _bottleneck_feasible(P, Q, vv):
            return float(vv)
    return float(max(cands))


def p_wasserstein(bc1: Barcode, bc2: Barcode, dim: int = 1, p: float = 2.0, top_k: int | None = None) -> float:
    """p-Wasserstein distance between two persistence diagrams (diagonal allowed).

    Uses the standard persdiagram cost matrix + Hungarian matching. `top_k`
    restricts each diagram to its k most persistent points. Requires scipy for
    larger diagrams; a brute-force fallback covers very small ones."""
    P, Q = _points(bc1, dim, top_k), _points(bc2, dim, top_k)
    n1, n2 = len(P), len(Q)
    if n1 == 0 and n2 == 0:
        return 0.0
    n = n1 + n2
    M = np.zeros((n, n))
    hP = _heights(P, p) ** p if n1 else np.zeros(0)
    hQ = _heights(Q, p) ** p if n2 else np.zeros(0)
    if n1 and n2:
        M[:n1, :n2] = _lp_matrix(P, Q, p) ** p
        M[n1:, n2:] = _lp_matrix(Q, P, p) ** p
    M[:n1, n2:] = 2.0 * hP[:, None]
    M[n1:, :n2] = 2.0 * hQ[None, :]
    if n <= 8:
        import itertools

        best = np.inf
        for perm in itertools.permutations(range(n)):
            tot = sum(M[i, perm[i]] for i in range(n))
            best = min(best, tot)
        w = best / 2.0
    else:
        try:
            from scipy.optimize import linear_sum_assignment
        except ImportError as e:  # pragma: no cover
            raise ImportError("scipy is required for p_wasserstein on larger diagrams") from e
        r, c = linear_sum_assignment(M)
        w = M[r, c].sum() / 2.0
    return float(w ** (1.0 / p))
