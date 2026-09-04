from __future__ import annotations

import numpy as np

from vrtda.errors import MetricError

_METRICS = {}


def register(name: str):
    def deco(fn):
        _METRICS[name] = fn
        fn.name = name
        return fn
    return deco


def names() -> list[str]:
    return sorted(_METRICS)


def get(name: str):
    if name not in _METRICS:
        raise MetricError(f"unknown metric {name!r}; available: {names()}")
    return _METRICS[name]


def _euclidean_block(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a2 = np.einsum("ij,ij->i", a, a)[:, None]
    b2 = np.einsum("ij,ij->i", b, b)[None, :]
    d2 = a2 + b2 - 2.0 * (a @ b.T)
    np.maximum(d2, 0.0, out=d2)
    return np.sqrt(d2)


@register("euclidean")
def euclidean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return _euclidean_block(a, b)


@register("squared")
def squared(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a2 = np.einsum("ij,ij->i", a, a)[:, None]
    b2 = np.einsum("ij,ij->i", b, b)[None, :]
    d2 = a2 + b2 - 2.0 * (a @ b.T)
    np.maximum(d2, 0.0, out=d2)
    return d2


@register("manhattan")
def manhattan(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(np.abs(a[:, None, :] - b[None, :, :]), axis=2)


@register("cosine")
def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    na = np.linalg.norm(a, axis=1, keepdims=True)
    nb = np.linalg.norm(b, axis=1, keepdims=True)
    na[na == 0] = 1.0
    nb[nb == 0] = 1.0
    sim = (a / na) @ (b / nb).T
    np.clip(sim, -1.0, 1.0, out=sim)
    return 1.0 - sim


@register("normalized_euclidean")
def normalized_euclidean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    na = np.linalg.norm(a, axis=1, keepdims=True)
    nb = np.linalg.norm(b, axis=1, keepdims=True)
    na[na == 0] = 1.0
    nb[nb == 0] = 1.0
    return _euclidean_block(a / na, b / nb)
