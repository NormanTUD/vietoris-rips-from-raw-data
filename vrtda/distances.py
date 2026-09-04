from __future__ import annotations

import numpy as np

from vrtda import debug
from vrtda import metrics as M
from vrtda.errors import DataError


def pairwise_distances(
    X: np.ndarray,
    metric: str = "euclidean",
    chunk: int | None = None,
    self_distances: bool = True,
) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise DataError(f"X must be 2D, got {X.shape}")
    n = X.shape[0]
    fn = M.get(metric)
    if chunk is None:
        chunk = max(1, min(n, int(2_000_000 // max(1, X.shape[1]))))
    out = np.empty((n, n), dtype=np.float64)
    with debug.timing(f"pairwise_distances n={n} d={X.shape[1]} metric={metric}"):
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            out[s:e, :] = fn(X[s:e], X)
    if self_distances:
        np.fill_diagonal(out, 0.0)
    _validate(out, metric)
    return out


def _validate(D: np.ndarray, metric: str) -> None:
    n = D.shape[0]
    if D.shape != (n, n):
        raise DataError(f"distance matrix shape {D.shape} != ({n},{n})")
    if not np.all(np.isfinite(D)):
        raise DataError("distance matrix contains NaN/Inf")
    if (D < -1e-9).any():
        raise DataError("distance matrix has negative entries")
    if (np.abs(D - D.T) > 1e-8 * max(1.0, float(np.abs(D).max()))).any():
        raise DataError("distance matrix is not symmetric (metric must be a true metric)")
    if (np.abs(np.diag(D)) > 1e-9).any():
        raise DataError("distance matrix diagonal is not zero")
    debug.assert_debug(bool(np.all(D >= 0)), "negative distances")


def distance_matrix_summary(D: np.ndarray) -> dict[str, int | float]:
    off = D[~np.eye(D.shape[0], dtype=bool)]
    return {
        "n": D.shape[0],
        "min": float(off.min()),
        "max": float(off.max()),
        "mean": float(off.mean()),
        "median": float(np.median(off)),
    }


from vrtda.beartype_guard import beartype_module as _beartype_module

_beartype_module(__name__)
