from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from vrtda import datasets
from vrtda.complexes import FilteredComplex, build_rips
from vrtda.cocycles import Loop, persistent_loops
from vrtda.distances import pairwise_distances
from vrtda.persistence import Barcode, persistent_homology


@dataclass
class LayerResult:
    layer: int
    complex: FilteredComplex
    barcode: Barcode
    nn: float
    eps_max: float
    labels: list[str]
    texts: list[str] | None = None

    @property
    def scale(self) -> float:
        return self.nn


@dataclass
class AttractorChain:
    per_layer_tokens: dict[int, frozenset[str]] = field(default_factory=dict)

    def layers(self) -> list[int]:
        return sorted(self.per_layer_tokens)

    @property
    def span(self) -> tuple[int, int] | None:
        ls = self.layers()
        return (ls[0], ls[-1]) if ls else None

    @property
    def length(self) -> int:
        ls = self.layers()
        return ls[-1] - ls[0] + 1 if ls else 0

    @property
    def tokens(self) -> set[str]:
        return set().union(*self.per_layer_tokens.values()) if self.per_layer_tokens else set()

    def describe(self, texts: dict[str, str] | None = None) -> str:
        sp = self.span
        if sp is None:
            return "<empty>"
        toks = sorted(self.tokens)
        if texts:
            toks = [texts.get(t, t) for t in toks]
        return f"layers[{sp[0]}..{sp[1]}] len={self.length} tokens={toks}"


def _mean_nn(D: np.ndarray) -> float:
    d = D.copy()
    np.fill_diagonal(d, np.inf)
    return float(d.min(axis=1).mean())


def layer_barcodes(
    data_dir: str | Path | None = None,
    layers: list[int] | None = None,
    metric: str = "euclidean",
    eps_cap_frac: float = 4.0,
    max_dim: int = 2,
    normalize: bool = False,
    texts: bool = False,
) -> dict[int, LayerResult]:
    """For each layer, a capped Rips complex over the FIXED token set + its barcode.

    The 81 tokens share an identity (label) across layers, so vertex i is the same
    token in every layer (only its embedding changes). `eps_max = eps_cap_frac * nn`,
    where nn is the layer's mean nearest-neighbour distance, so the scale is
    comparable across the (widely) varying per-layer embedding spread."""
    dd = data_dir or datasets._data_root()
    Ls = list(layers) if layers is not None else datasets.list_layers(dd)
    out: dict[int, LayerResult] = {}
    for L in Ls:
        ps = datasets.load_token_cloud(dd, int(L), normalize=normalize)
        D = pairwise_distances(ps.data, metric)
        nn = _mean_nn(D)
        eps_max = float(eps_cap_frac) * nn
        C = build_rips(ps.data, D, eps_max, max_dim=max_dim)
        bc = persistent_homology(C)
        out[int(L)] = LayerResult(
            layer=int(L),
            complex=C,
            barcode=bc,
            nn=nn,
            eps_max=eps_max,
            labels=list(ps.labels),
            texts=(datasets.token_texts(dd, int(L)) if texts else None),
        )
    return out


def _essential_at(lr: LayerResult, scale: float, dim: int = 1) -> int:
    return sum(1 for iv in lr.barcode.of_dim(dim) if iv.is_essential and iv.birth <= scale + 1e-15)


def _total_persistence_at(lr: LayerResult, scale: float, dim: int = 1) -> float:
    s = 0.0
    for iv in lr.barcode.of_dim(dim):
        if iv.birth > scale + 1e-15:
            continue
        if np.isfinite(iv.death):
            s += max(0.0, min(iv.death, scale) - iv.birth)
        else:
            s += max(0.0, scale - iv.birth)
    return s


def betti_heatmap(
    layer_results: dict[int, LayerResult],
    scale_fracs: Sequence[float] | np.ndarray | None = None,
    dim: int = 1,
    metric: str = "betti",
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """2D array over (scale-fraction x layer).

    Returns (H, fracs, layers) with H[s, t] = the metric value at per-layer scale
    fracs[s] * nn(layer_t). `metric` in {"betti", "essential", "persistence"}.
    Per-layer relative scale makes the depth axis comparable across layers.
    (For point clouds `betti` is usually the informative default: `essential`
    is typically 0 because triangles fill the holes.)"""
    layers = sorted(layer_results)
    if scale_fracs is None:
        scale_fracs = np.linspace(0.25, 4.0, 16)
    fracs = np.asarray(scale_fracs, dtype=float)
    H = np.zeros((len(fracs), len(layers)), dtype=float)
    for s, f in enumerate(fracs):
        for t, L in enumerate(layers):
            lr = layer_results[L]
            scale = f * lr.nn
            if metric == "essential":
                H[s, t] = _essential_at(lr, scale, dim)
            elif metric == "betti":
                H[s, t] = lr.barcode.betti_at(scale)[dim]
            elif metric == "persistence":
                H[s, t] = _total_persistence_at(lr, scale, dim)
            else:
                raise ValueError(f"unknown metric {metric!r}")
    return H, fracs, layers


def _token_overlap(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _match_chains(
    per_layer: dict[int, list[frozenset]],
    min_overlap: float = 0.3,
    max_gap: int = 1,
) -> list[AttractorChain]:
    """Greedy forward matching of loop token-sets across layers into chains.

    A chain may skip up to `max_gap` consecutive sampled layers (a feature can be
    absent there) but must re-match within that window. Unmatched loops start new
    chains. `max_gap` is in sampled positions, not raw layer numbers."""
    layers = sorted(per_layer)
    pos = {L: i for i, L in enumerate(layers)}
    open_chains: list[AttractorChain] = []
    closed: list[AttractorChain] = []

    for L in layers:
        loops = per_layer[L]
        still = []
        for ch in open_chains:
            gap = pos[L] - pos[ch.layers()[-1]]
            (closed if gap > max_gap else still).append(ch)
        open_chains = still

        cands = []
        for ci, ch in enumerate(open_chains):
            last = ch.per_layer_tokens[ch.layers()[-1]]
            for j, B in enumerate(loops):
                sc = _token_overlap(last, B)
                if sc >= min_overlap:
                    cands.append((sc, ci, j))
        cands.sort(key=lambda x: -x[0])
        matched_chain, matched_loop = set(), set()
        for sc, ci, j in cands:
            if ci in matched_chain or j in matched_loop:
                continue
            matched_chain.add(ci)
            matched_loop.add(j)
            open_chains[ci].per_layer_tokens[L] = loops[j]

        for j, B in enumerate(loops):
            if j not in matched_loop:
                ch = AttractorChain()
                ch.per_layer_tokens[L] = B
                open_chains.append(ch)
    closed.extend(open_chains)
    out = [ch for ch in closed if ch.per_layer_tokens]
    out.sort(key=lambda ch: (-ch.length, ch.span[0] if ch.span else 0))
    return out


def _loop_persistence(lp: Loop) -> float:
    return float("inf") if lp.death is None or not np.isfinite(lp.death) else max(0.0, float(lp.death - lp.birth))


def _select_loops(
    lr: LayerResult,
    dim: int = 1,
    essential_only: bool = False,
    top_k: int | None = None,
    min_persistence: float | None = None,
    min_persistence_frac: float = 0.0,
) -> list[Loop]:
    """H_1 loops of a layer, optionally filtered to the significant ones.

    For point clouds almost all H_1 classes are short-lived (holes get filled by
    triangles), so tracking only *essential* (never-dying) loops usually yields
    nothing. Instead keep the longest-lived loops: `top_k` by persistence, and/or
    those with persistence >= `min_persistence` / `min_persistence_frac` * max."""
    loops = persistent_loops(lr.complex, lr.barcode, eps_max=lr.eps_max, essential_only=essential_only)
    if not loops:
        return []
    if min_persistence_frac > 0:
        cap = max(_loop_persistence(lp) for lp in loops)
        if np.isfinite(cap):
            loops = [lp for lp in loops if _loop_persistence(lp) >= min_persistence_frac * cap]
    if min_persistence is not None:
        loops = [lp for lp in loops if _loop_persistence(lp) >= min_persistence]
    if top_k is not None:
        loops = sorted(loops, key=lambda lp: -_loop_persistence(lp))[:top_k]
    return loops


def depth_chains(
    layer_results: dict[int, LayerResult],
    min_overlap: float = 0.3,
    max_gap: int = 1,
    dim: int = 1,
    essential_only: bool = False,
    top_k: int | None = 15,
    min_persistence: float | None = None,
    min_persistence_frac: float = 0.0,
) -> list[AttractorChain]:
    """Track significant H_1 loops across layers (by token-set overlap) into chains.

    Per layer we keep the `top_k` longest-lived loops (or those meeting a
    persistence threshold) and match them to consecutive layers by token-set
    overlap. A chain spanning many layers is an attractor persisting in depth."""
    per_layer: dict[int, list[frozenset]] = {}
    for L in sorted(layer_results):
        lr = layer_results[L]
        loops = _select_loops(
            lr,
            dim=dim,
            essential_only=essential_only,
            top_k=top_k,
            min_persistence=min_persistence,
            min_persistence_frac=min_persistence_frac,
        )
        per_layer[L] = [frozenset(lr.labels[v] for v in lp.vertices) for lp in loops]
    return _match_chains(per_layer, min_overlap, max_gap)


def depth_profile(layer_results: dict[int, LayerResult], dim: int = 1) -> dict[int, dict[str, int | float]]:
    """Per-layer attractor activity vs depth (scale-free summaries).

    For each layer: nn, number of H_d intervals, total H_d persistence, and the
    beta_d peak (value + the relative scale where it occurs). A quantity that is
    stable over depth is an attractor that persists across layers."""
    out: dict[int, dict] = {}
    for L in sorted(layer_results):
        lr = layer_results[L]
        ivs = lr.barcode.of_dim(dim)
        total = 0.0
        for iv in ivs:
            d = lr.eps_max if not np.isfinite(iv.death) else min(iv.death, lr.eps_max)
            total += max(0.0, d - iv.birth)
        fr = np.linspace(0.2, 6.0, 60)
        peak = 0
        peak_frac = 0.0
        for f in fr:
            b = lr.barcode.betti_at(f * lr.nn)[dim]
            if b > peak:
                peak, peak_frac = b, float(f)
        out[L] = {
            "nn": lr.nn,
            "n_intervals": len(ivs),
            "total_persistence": float(total),
            "essential": sum(1 for iv in ivs if iv.is_essential),
            "beta_peak": int(peak),
            "beta_peak_frac": peak_frac,
        }
    return out


def stable_core(
    chains: list[AttractorChain],
    min_layer_fraction: float = 0.5,
    total_layers: int | None = None,
) -> list[AttractorChain]:
    """Chains (attractors) present in at least a fraction of the depth range."""
    total = total_layers or max((c.length for c in chains), default=1)
    out = [c for c in chains if c.length >= max(1, round(min_layer_fraction * total))]
    return out


from vrtda.beartype_guard import beartype_module as _beartype_module

_beartype_module(__name__)
