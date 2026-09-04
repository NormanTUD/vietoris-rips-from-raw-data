from __future__ import annotations

from itertools import combinations

import numpy as np

from vrtda.errors import DataError


def _circumcenter(T: np.ndarray):
    """Center and radius of the unique sphere through the affinely independent
    points T, computed as the minimum-radius equidistant center in the affine
    hull. Returns (None, None) if T is affinely dependent."""
    T = np.asarray(T, dtype=np.float64)
    m = T.shape[0]
    if m == 1:
        return T[0].copy(), 0.0
    base = T[0]
    M = T[1:] - base  # (m-1) x d, the independent direction vectors
    if np.linalg.matrix_rank(M) < m - 1:
        return None, None
    b = 0.5 * np.einsum("ij,ij->i", M, M)  # (m-1,)
    AA = M @ M.T
    try:
        z = M.T @ np.linalg.solve(AA, b)
    except np.linalg.LinAlgError:
        return None, None
    c = base + z
    r = float(np.linalg.norm(z))
    return c, r


def min_enclosing_ball_radius(pts: np.ndarray) -> float:
    pts = np.asarray(pts, dtype=np.float64)
    if pts.ndim != 2:
        raise DataError(f"pts must be 2D, got {pts.shape}")
    k, d = pts.shape
    if k == 0:
        return 0.0
    if k == 1:
        return 0.0
    best = np.inf
    max_m = min(k, d + 1)
    for m in range(2, max_m + 1):
        for T in combinations(range(k), m):
            c, r = _circumcenter(pts[list(T)])
            if c is None or r >= best:
                continue
            dists = np.linalg.norm(pts - c, axis=1)
            if float(dists.max()) <= r + 1e-9 * max(1.0, r):
                best = r
    if not np.isfinite(best):
        # degenerate (all points (nearly) coincident): center-of-mass radius
        c = pts.mean(axis=0)
        best = float(np.linalg.norm(pts - c, axis=1).max())
    return best


def min_enclosing_ball(pts: np.ndarray) -> tuple[np.ndarray, float]:
    pts = np.asarray(pts, dtype=np.float64)
    k, d = pts.shape
    best_c, best_r = pts.mean(axis=0), float(np.linalg.norm(pts - pts.mean(axis=0), axis=1).max())
    max_m = min(k, d + 1)
    for m in range(2, max_m + 1):
        for T in combinations(range(k), m):
            c, r = _circumcenter(pts[list(T)])
            if c is None or r >= best_r:
                continue
            dists = np.linalg.norm(pts - c, axis=1)
            if float(dists.max()) <= r + 1e-9 * max(1.0, r):
                best_c, best_r = c, r
    return best_c, best_r
