from __future__ import annotations

from itertools import combinations

import numpy as np

from vrtda.errors import DataError


def _sphere_from_points(T: np.ndarray) -> tuple[np.ndarray, float]:
    T = np.asarray(T, dtype=np.float64)
    m = T.shape[0]
    if m == 0:
        raise DataError("empty point set for sphere")
    if m == 1:
        return T[0].copy(), 0.0
    base = T[0]
    B = 2.0 * (T[1:] - base)
    b = np.einsum("ij,ij->i", T[1:], T[1:]) - np.dot(base, base)
    c, *_ = np.linalg.lstsq(B, b, rcond=None)
    r = float(np.linalg.norm(c - base))
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
            c, r = _sphere_from_points(pts[list(T)])
            if r >= best:
                continue
            dists = np.linalg.norm(pts - c, axis=1)
            if float(dists.max()) <= r + 1e-9 * max(1.0, r):
                best = r
    if not np.isfinite(best):
        # fallback: center of mass radius (upper bound)
        c = pts.mean(axis=0)
        best = float(np.linalg.norm(pts - c, axis=1).max())
    return best


def min_enclosing_ball(pts: np.ndarray) -> tuple[np.ndarray, float]:
    pts = np.asarray(pts, dtype=np.float64)
    k, d = pts.shape
    best_c, best_r = pts.mean(axis=0), np.inf
    max_m = min(k, d + 1)
    for m in range(1, max_m + 1):
        for T in combinations(range(k), m):
            c, r = _sphere_from_points(pts[list(T)])
            if r >= best_r:
                continue
            dists = np.linalg.norm(pts - c, axis=1)
            if float(dists.max()) <= r + 1e-9 * max(1.0, r):
                best_c, best_r = c, r
    return best_c, best_r
