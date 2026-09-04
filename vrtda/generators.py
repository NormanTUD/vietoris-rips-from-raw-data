from __future__ import annotations

import numpy as np

from vrtda.errors import DataError


def _rng(seed):
    return np.random.default_rng(seed)


def circle(n: int, radius: float = 1.0, seed: int | None = None, noise: float = 0.0) -> np.ndarray:
    g = _rng(seed)
    u = g.uniform(0.0, 2.0 * np.pi, n)
    x = radius * np.cos(u)
    y = radius * np.sin(u)
    pts = np.column_stack([x, y])
    if noise:
        pts = pts + g.normal(0.0, noise, pts.shape)
    return pts


def product_torus(
    k: int,
    n: int,
    radius: float = 1.0,
    seed: int | None = None,
    noise: float = 0.0,
) -> np.ndarray:
    if k < 1:
        raise DataError("k must be >= 1")
    g = _rng(seed)
    u = g.uniform(0.0, 2.0 * np.pi, size=(n, k))
    pts = np.empty((n, 2 * k))
    for i in range(k):
        pts[:, 2 * i] = radius * np.cos(u[:, i])
        pts[:, 2 * i + 1] = radius * np.sin(u[:, i])
    if noise:
        pts = pts + g.normal(0.0, noise, pts.shape)
    return pts


def donut(
    n: int,
    R: float = 1.0,
    r: float = 0.35,
    seed: int | None = None,
    noise: float = 0.0,
) -> np.ndarray:
    g = _rng(seed)
    u = g.uniform(0.0, 2.0 * np.pi, n)
    v = g.uniform(0.0, 2.0 * np.pi, n)
    x = (R + r * np.cos(v)) * np.cos(u)
    y = (R + r * np.cos(v)) * np.sin(u)
    z = r * np.sin(v)
    pts = np.column_stack([x, y, z])
    if noise:
        pts = pts + g.normal(0.0, noise, pts.shape)
    return pts


def torus3d(
    n: int,
    R: float = 1.0,
    r: float = 0.35,
    s: float = 0.18,
    seed: int | None = None,
    noise: float = 0.0,
) -> np.ndarray:
    g = _rng(seed)
    u = g.uniform(0.0, 2.0 * np.pi, n)
    v = g.uniform(0.0, 2.0 * np.pi, n)
    w = g.uniform(0.0, 2.0 * np.pi, n)
    ring = R + r * np.cos(v)
    x = ring * np.cos(u)
    y = ring * np.sin(u)
    z = r * np.sin(v) * np.cos(w)
    t = r * np.sin(v) * np.sin(w)
    pts = np.column_stack([x, y, z, t])
    if noise:
        pts = pts + g.normal(0.0, noise, pts.shape)
    return pts


def sphere(n: int, dim: int, radius: float = 1.0, seed: int | None = None, noise: float = 0.0) -> np.ndarray:
    g = _rng(seed)
    if dim < 1:
        raise DataError("dim must be >= 1")
    vec = g.normal(size=(n, dim + 1))
    norms = np.linalg.norm(vec, axis=1, keepdims=True)
    pts = radius * vec / norms
    if noise:
        pts = pts + g.normal(0.0, noise, pts.shape)
    return pts


def two_blobs(n_each: int, sep: float = 4.0, seed: int | None = None) -> np.ndarray:
    g = _rng(seed)
    a = g.normal(0.0, 0.5, size=(n_each, 2))
    b = g.normal(0.0, 0.5, size=(n_each, 2)) + np.array([sep, 0.0])
    return np.vstack([a, b])


def gmm(
    n_clusters: int,
    n_per: int,
    dim: int,
    spread: float = 0.5,
    seed: int | None = None,
) -> np.ndarray:
    g = _rng(seed)
    centers = g.normal(0.0, 3.0, size=(n_clusters, dim))
    parts = [g.normal(c, spread, size=(n_per, dim)) for c in centers]
    return np.vstack(parts)


def binads(n: int, dim: int = 3, seed: int | None = None) -> np.ndarray:
    g = _rng(seed)
    return g.integers(0, 2, size=(n, dim)).astype(float)


# --- structured (grid) samplers: clean, deterministic topology for validation ---

def circle_grid(n: int, radius: float = 1.0) -> np.ndarray:
    u = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(u), radius * np.sin(u)])


def product_torus_grid(k: int, n_per: int, radius: float = 1.0) -> np.ndarray:
    if k < 1:
        raise DataError("k must be >= 1")
    us = [np.linspace(0.0, 2.0 * np.pi, n_per, endpoint=False) for _ in range(k)]
    mesh = np.meshgrid(*us, indexing="ij")
    u = np.stack([m.ravel() for m in mesh], axis=1)
    n = u.shape[0]
    pts = np.empty((n, 2 * k))
    for i in range(k):
        pts[:, 2 * i] = radius * np.cos(u[:, i])
        pts[:, 2 * i + 1] = radius * np.sin(u[:, i])
    return pts


def donut_grid(nu: int, nv: int, R: float = 1.0, r: float = 0.35) -> np.ndarray:
    U = np.linspace(0.0, 2.0 * np.pi, nu, endpoint=False)
    V = np.linspace(0.0, 2.0 * np.pi, nv, endpoint=False)
    UU, VV = np.meshgrid(U, V, indexing="ij")
    x = (R + r * np.cos(VV)) * np.cos(UU)
    y = (R + r * np.cos(VV)) * np.sin(UU)
    z = r * np.sin(VV)
    return np.column_stack([x.ravel(), y.ravel(), z.ravel()])


def torus3d_grid(nu: int, nv: int, nw: int, R: float = 1.0, r: float = 0.35) -> np.ndarray:
    U = np.linspace(0.0, 2.0 * np.pi, nu, endpoint=False)
    V = np.linspace(0.0, 2.0 * np.pi, nv, endpoint=False)
    W = np.linspace(0.0, 2.0 * np.pi, nw, endpoint=False)
    UU, VV, WW = np.meshgrid(U, V, W, indexing="ij")
    ring = R + r * np.cos(VV)
    x = ring * np.cos(UU)
    y = ring * np.sin(UU)
    z = r * np.sin(VV) * np.cos(WW)
    t = r * np.sin(VV) * np.sin(WW)
    return np.column_stack([x.ravel(), y.ravel(), z.ravel(), t.ravel()])
