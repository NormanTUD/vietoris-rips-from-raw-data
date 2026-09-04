from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vrtda import datasets
from vrtda.complexes import build_rips
from vrtda.distances import pairwise_distances
from vrtda.persistence import persistent_homology


@dataclass
class MapperNode:
    interval: tuple[float, float]
    n_points: int
    beta0: int
    beta1: int
    point_indices: list[int]


@dataclass
class MapperGraph:
    nodes: list[MapperNode]
    edges: list[tuple[int, int, int]]  # (node_i, node_j, n_overlap)

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    def beta1_profile(self) -> list[int]:
        return [n.beta1 for n in self.nodes]


def _mean_nn(D: np.ndarray) -> float:
    d = D.copy()
    np.fill_diagonal(d, np.inf)
    return float(d.min(axis=1).mean()) if d.shape[0] > 1 else 0.0


def _betti(X: np.ndarray, eps: float, max_dim: int = 2) -> tuple[int, int]:
    n = len(X)
    if n < 2:
        return (1 if n else 0), 0
    D = pairwise_distances(X, "euclidean")
    C = build_rips(X, D, eps, max_dim=max_dim)
    bc = persistent_homology(C)
    return int(bc.betti_at(eps)[0]), int(sum(1 for iv in bc.of_dim(1) if iv.is_essential))


def mapper(
    X,
    phi,
    n_bins: int = 8,
    overlap: float = 0.5,
    eps: float | None = None,
    eps_frac: float = 2.0,
    max_dim: int = 2,
) -> MapperGraph:
    """1D Mapper (Carlsson et al.) with respect to the lens function `phi`.

    The range of `phi` is covered by `n_bins` overlapping intervals; each bin's
    points form a Rips complex (at `eps`, or `eps_frac * local_nn` if `eps` is
    None) whose beta_1 is stored on the node. Two adjacent bins are joined by an
    edge when their point overlap is non-empty and connected at the same scale."""
    X = np.asarray(X, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    lo, hi = float(phi.min()), float(phi.max())
    if hi <= lo:
        hi = lo + 1.0
    width = (1.0 + overlap) * (hi - lo) / n_bins
    starts = np.linspace(lo, hi - width, n_bins)
    intervals = [(float(starts[i]), float(starts[i] + width)) for i in range(n_bins)]

    def node_eps(sub: np.ndarray) -> float:
        if eps is not None:
            return float(eps)
        if len(sub) < 2:
            return 0.0
        return float(eps_frac) * _mean_nn(pairwise_distances(sub, "euclidean"))

    nodes: list[MapperNode] = []
    for (a, b) in intervals:
        idx = np.where((phi >= a) & (phi <= b))[0]
        Xi = X[idx]
        e = node_eps(Xi)
        b0, b1 = _betti(Xi, e, max_dim)
        nodes.append(MapperNode((a, b), len(idx), b0, b1, idx.tolist()))

    edges: list[tuple[int, int, int]] = []
    for i in range(n_bins - 1):
        a = max(intervals[i][0], intervals[i + 1][0])
        b = min(intervals[i][1], intervals[i + 1][1])
        if b <= a:
            continue
        ov = np.where((phi >= a) & (phi <= b))[0]
        if len(ov) < 2:
            continue
        Xo = X[ov]
        e = node_eps(Xo)
        b0, _ = _betti(Xo, e, max_dim=1)
        if b0 == 1:
            edges.append((i, i + 1, int(len(ov))))
    return MapperGraph(nodes, edges)


def mapper_residual(data_dir=None, layer: int = 0, n_bins: int = 8, overlap: float = 0.5,
                    eps: float | None = None, eps_frac: float = 2.0, max_dim: int = 2) -> MapperGraph:
    """Mapper on a layer's token cloud with the residual norm as the lens."""
    ps = datasets.load_token_cloud(data_dir, layer)
    norms, _labels = datasets.load_residual_matrix(data_dir, "norms")
    # norms is (n_tokens, n_layers); pick this layer's column
    L = int(layer)
    phi = norms[:, L]
    return mapper(ps.data, phi, n_bins=n_bins, overlap=overlap, eps=eps, eps_frac=eps_frac, max_dim=max_dim)
