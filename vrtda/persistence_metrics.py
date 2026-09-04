from __future__ import annotations

import numpy as np

from vrtda.persistence import Barcode


def persistence_values(bc: Barcode, dim: int | None = None, cap: float | None = None) -> list[float]:
    """Off-diagonal persistence amplitudes (death - birth), sorted descending.

    Finite intervals contribute their length; essential (infinite) intervals are
    included only if `cap` is given (capped at cap - birth), otherwise excluded.
    The persistence landscape/image are classically defined on finite points only.
    """
    ivs = bc.of_dim(dim) if dim is not None else bc.intervals
    vals = []
    for iv in ivs:
        if np.isfinite(iv.death):
            vals.append(max(0.0, float(iv.death - iv.birth)))
        elif cap is not None:
            vals.append(max(0.0, float(cap - iv.birth)))
    return sorted(vals, reverse=True)


def persistence_entropy(bc: Barcode, dim: int | None = None, cap: float | None = None, base: float = 2.0) -> float:
    """Persistence entropy (de Silva, Mémoli & Glaser 2011).

    H = -sum_i p_i log_b(p_i) with p_i = a_i / sum_j a_j. Lower = a few dominant
    features; higher (max log_b n) = many comparable features."""
    vals = persistence_values(bc, dim, cap)
    s = float(sum(vals))
    if s <= 0:
        return 0.0
    p = np.array([v for v in vals if v > 0]) / s
    return float(-np.sum(p * np.log(p) / np.log(base)))


def persistence_landscape(bc: Barcode, dim: int | None = None, resolution: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Cohen-Stein persistence landscape (finite points).

    Returns (xgrid, F): F has shape (n_levels, len(xgrid)); level j is
        F_j(x) = max_{i >= j} [ p_i - |x - i| ]_+
    with p_1 >= p_2 >= ... > 0 the sorted persistence values and i 1-indexed.
    F_0 is the top envelope; the sequence is non-increasing in j.
    """
    p = np.array(persistence_values(bc, dim), dtype=float)  # descending, finite only
    n = len(p)
    if n == 0:
        return np.array([0.0]), np.zeros((1, 1))
    xgrid = np.linspace(0.0, float(n), max(resolution, 10 * n))
    # tent for the i-th persistence (i 1-indexed) centered at x=i
    idx = np.arange(1, n + 1, dtype=float)
    tents = p[:, None] - np.abs(xgrid[None, :] - idx[:, None])
    tents = np.maximum(tents, 0.0)
    # F_j = max over tents i in [j, n)  -> cumulative max from the right
    F = np.maximum.accumulate(tents[::-1], axis=0)[::-1]
    return xgrid, F


def persistence_image(
    bc: Barcode,
    dim: int | None = None,
    n_grid: int = 32,
    eps_max: float | None = None,
    bandwidth: float | None = None,
    weights: str = "persistence",
) -> np.ndarray:
    """Bubenik persistence image: a fixed-size 2D array from the off-diagonal
    points (birth, death) of `dim`. Returns an (n_grid, n_grid) image (x=birth, y=death)."""
    ivs = bc.of_dim(dim) if dim is not None else bc.intervals
    pts = [(float(iv.birth), float(iv.death)) for iv in ivs if np.isfinite(iv.death)]
    img = np.zeros((n_grid, n_grid))
    if not pts:
        return img
    pts = np.array(pts)
    xmax = float(eps_max) if eps_max is not None else max(float(pts.max()), 1e-9)
    if bandwidth is None:
        bandwidth = max(xmax / (2.0 * n_grid), 1e-6)
    xs = np.linspace(0.0, xmax, n_grid)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    if weights == "persistence":
        w = (pts[:, 1] - pts[:, 0])
    else:
        w = np.ones(len(pts))
    inv = 1.0 / (2.0 * bandwidth * bandwidth)
    for (b, d), wi in zip(pts, w):
        img += wi * np.exp(-((X - b) ** 2 + (Y - d) ** 2) * inv)
    return img


def persistence_diagram(bc: Barcode, dim: int | None = None) -> np.ndarray:
    """Off-diagonal points (birth, death) of `dim` as an (n, 2) array."""
    ivs = bc.of_dim(dim) if dim is not None else bc.intervals
    return np.array([[float(iv.birth), float(iv.death)] for iv in ivs if np.isfinite(iv.death)])


from vrtda.beartype_guard import beartype_module as _beartype_module

_beartype_module(__name__)
