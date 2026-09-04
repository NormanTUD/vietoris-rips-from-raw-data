from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vrtda import datasets


@dataclass
class Convergence:
    layers: list[int]
    mean_pairwise: np.ndarray
    centroid_norm: np.ndarray
    mean_dist: np.ndarray
    max_dist: np.ndarray


def convergence(data_dir: str | Path | None = None) -> Convergence:
    """Final-token convergence statistics over depth (from convergence_analysis.csv)."""
    d = datasets.load_convergence(data_dir)
    return Convergence(
        layers=list(d["layer"]),
        mean_pairwise=np.asarray(d["mean_pairwise_distance_final_token"]),
        centroid_norm=np.asarray(d["centroid_norm"]),
        mean_dist=np.asarray(d["mean_distance_to_centroid"]),
        max_dist=np.asarray(d["max_distance_to_centroid"]),
    )


def _plateau_index(values: np.ndarray, rel: float = 0.2, after: int = 0) -> int:
    """First index >= after where the value stays within `rel` of its final value."""
    target = values[-1]
    span = values.max() - values.min()
    if span <= 0:
        return after
    for i in range(after, len(values)):
        if all(abs(values[j] - target) <= rel * span for j in range(i, len(values))):
            return i
    return len(values) - 1


def convergence_summary(conv: Convergence, rel: float = 0.2) -> dict[str, int | float]:
    """Characterise the depth-dynamics of the final (answer) tokens."""
    md = conv.mean_dist
    mp = conv.mean_pairwise
    peak_i = int(np.argmax(md))
    return {
        "n_layers": len(conv.layers),
        "peak_spread_layer": int(conv.layers[peak_i]),
        "peak_mean_pairwise": float(mp.max()),
        "final_mean_pairwise": float(mp[-1]),
        "final_centroid_norm": float(conv.centroid_norm[-1]),
        "final_mean_dist_to_centroid": float(md[-1]),
        "spread_shrink": float(1.0 - md[-1] / md[peak_i]) if md[peak_i] > 0 else 0.0,
        "pairwise_shrink": float(1.0 - mp[-1] / mp.max()) if mp.max() > 0 else 0.0,
        "converged_layer": int(conv.layers[_plateau_index(md, rel)]),
    }


def _final_token_cloud(data_dir: str | Path | None, layer: int) -> np.ndarray:
    """Positions of the final (answer) token of each prompt at a layer (n_prompts, D)."""
    ps = datasets.load_token_cloud(data_dir, layer)
    idx = datasets.final_token_indices(data_dir, layer)
    return ps.data[idx]


def per_language_final_token_distance(
    data_dir: str | Path | None = None,
    layers: list[int] | None = None,
) -> tuple[list[int], np.ndarray, list[object]]:
    """Per-layer, per-prompt distance of that prompt's answer token to the group
    centroid of all answer tokens. Returns (layers, matrix [n_prompts, n_layers],
    prompt_texts)."""
    layers = list(layers) if layers is not None else list(range(0, 65, 4))
    info = datasets.load_group_info(data_dir)
    prompts = info.get("prompts", [])
    mats = []
    for L in layers:
        ft = _final_token_cloud(data_dir, L)
        c = ft.mean(axis=0, keepdims=True)
        mats.append(np.linalg.norm(ft - c, axis=1))
    mat = np.stack(mats, axis=1)  # (n_prompts, n_layers)
    return layers, mat, prompts


def flow_svd(data_dir: str | Path | None = None, layers: list[int] | None = None, n_components: int = 5) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """SVD of the depth-trajectory of the answer-token centroid (the 'flow').

    Stacks the per-layer answer-token centroid vectors, centers them, and returns
    (components [n_components, D], explained_variance_ratio [n_components], layers).
    The dominant component is the direction the attractor centroid moves along."""
    layers = list(layers) if layers is not None else list(range(0, 65, 4))
    vecs = np.stack([_final_token_cloud(data_dir, L).mean(axis=0) for L in layers])
    vecs = vecs - vecs.mean(axis=0)
    U, S, Vt = np.linalg.svd(vecs, full_matrices=False)
    var = (S**2) / max(float((S**2).sum()), 1e-12)
    k = min(n_components, len(S))
    return Vt[:k], var[:k], layers


def attention_over_depth(data_dir: str | Path | None = None, metric: str = "to_self", n_heads: int = 40) -> tuple[list[int], np.ndarray, int]:
    """Mean attention metric over the answer tokens and all heads, per layer.

    Returns (layers, curve [n_layers], peak_layer). A rising self-attention on the
    answer token marks the layers where the model 'locks in' the attractor."""
    mat, _labels, cols = datasets.load_attention(data_dir, metric)
    n_layers = len(cols) // n_heads
    mat = mat[:, : n_layers * n_heads].reshape(mat.shape[0], n_layers, n_heads)
    # answer tokens = final token of each prompt
    idx = datasets.final_token_indices(data_dir, 0)
    layers = list(range(n_layers))
    curve = mat[idx].mean(axis=(0, 2))
    peak = int(np.argmax(curve))
    return layers, curve, layers[peak]


from vrtda.beartype_guard import beartype_module as _beartype_module

_beartype_module(__name__)
