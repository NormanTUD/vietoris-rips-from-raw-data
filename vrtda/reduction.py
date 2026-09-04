from __future__ import annotations

import numpy as np

from vrtda.errors import DataError


def variance_of(X: np.ndarray) -> np.ndarray:
    """Per-dimension variance (population, ddof=0)."""
    X = np.asarray(X, dtype=np.float64)
    return np.var(X, axis=0)


def top_variance_dims(X: np.ndarray, k: int) -> list[int]:
    """Indices of the k dimensions with the largest variance (sorted ascending)."""
    X = np.asarray(X, dtype=np.float64)
    k = int(min(k, X.shape[1]))
    if k <= 0:
        return []
    order = np.argsort(variance_of(X))[::-1]
    return sorted(int(i) for i in order[:k])


def pca(
    X: np.ndarray,
    n_components: int | None = None,
    center: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pure-numpy PCA via SVD.

    Returns (scores, components, explained_variance_ratio, mean):
      scores  (n, k)  data projected onto the k principal axes
      components (k, d)  the k principal axes (unit, orthogonal)
      explained_variance_ratio (k,)
      mean (d,)  (zero if center=False)
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise DataError(f"pca input must be 2D, got {X.shape}")
    n, d = X.shape
    k = int(n_components) if n_components is not None else min(n, d)
    k = max(1, min(k, n, d))
    mean = X.mean(axis=0) if center else np.zeros(d)
    Xc = X - mean
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    scores = U[:, :k] * S[:k]  # (n, k)
    comps = Vt[:k]
    ss = S ** 2
    evr = ss / max(1e-300, float(ss.sum()))
    evr = evr[:k]
    return scores, comps, evr, mean


def explain_variance(X: np.ndarray, n_components: int | None = None) -> np.ndarray:
    _, _, evr, _ = pca(X, n_components=n_components)
    return evr


def umap_2d(X: np.ndarray, **kw: object) -> np.ndarray:
    try:
        import umap
    except ImportError as e:  # pragma: no cover - optional dep
        raise DataError(
            "umap-learn is not installed in this environment. Add 'umap-learn' to the "
            "PEP-723 script dependencies to use --method umap."
        ) from e
    kw.setdefault("n_components", 2)
    return umap.UMAP(**kw).fit_transform(np.asarray(X, dtype=np.float64))


def tsne_2d(X: np.ndarray, **kw: object) -> np.ndarray:
    try:
        from sklearn.manifold import TSNE
    except ImportError as e:  # pragma: no cover - optional dep
        raise DataError(
            "scikit-learn is not installed in this environment. Add 'scikit-learn' to the "
            "PEP-723 script dependencies to use --method tsne."
        ) from e
    kw.setdefault("n_components", 2)
    kw.setdefault("random_state", 0)
    return TSNE(**kw).fit_transform(np.asarray(X, dtype=np.float64))


def reduce(
    X: np.ndarray,
    method: str,
    n_components: int = 2,
    **kw: object,
) -> tuple[np.ndarray, dict[str, object]]:
    """Dispatch a reduction method. Returns (reduced (n,k), meta)."""
    method = method.lower()
    X = np.asarray(X, dtype=np.float64)
    if method == "pca":
        scores, comps, evr, mean = pca(X, n_components=n_components)
        return scores, {"method": "pca", "explained_variance_ratio": evr}
    if method == "umap":
        return umap_2d(X, n_components=n_components, **kw), {"method": "umap"}
    if method == "tsne":
        return tsne_2d(X, n_components=n_components, **kw), {"method": "tsne"}
    raise DataError(f"unknown reduction method {method!r}; use pca | umap | tsne")


from vrtda.beartype_guard import beartype_module as _beartype_module

_beartype_module(__name__)
