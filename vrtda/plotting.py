from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from vrtda.dynamics import Convergence
from vrtda.errors import DataError
from vrtda.mapper import MapperGraph
from vrtda.persistence import Interval


def _mpl() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover - optional dep
        raise DataError(
            "matplotlib is not installed in this environment. Add 'matplotlib' to the "
            "PEP-723 script dependencies to enable plotting."
        ) from e
    return plt


def plot_betti_function(epsilons: Sequence[float] | np.ndarray, betti_arr: np.ndarray, path: str | Path, title: str = "Betti function") -> None:
    plt = _mpl()
    epsilons = np.asarray(epsilons)
    betti_arr = np.asarray(betti_arr)
    md = betti_arr.shape[1] - 1
    fig, ax = plt.subplots(figsize=(8, 5))
    for d in range(md + 1):
        ax.plot(epsilons, betti_arr[:, d], label=f"$\\beta_{d}$", marker="o", ms=3)
    ax.set_xlabel("epsilon")
    ax.set_ylabel("Betti number")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_barcode(intervals: list[Interval], path: str | Path, title: str = "Persistence barcode") -> None:
    """intervals: list of Interval (with .dim, .birth, .death)."""
    plt = _mpl()
    finite = [iv for iv in intervals if np.isfinite(iv.death)]
    essential = [iv for iv in intervals if not np.isfinite(iv.death)]
    xmax = max([iv.death for iv in finite] + [iv.birth for iv in intervals] + [0.0])
    fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(intervals))))
    y = 0
    for iv in sorted(intervals, key=lambda i: (i.dim, i.birth)):
        if np.isfinite(iv.death):
            ax.hlines(y, iv.birth, iv.death, color="tab:blue", lw=1.5)
        else:
            ax.hlines(y, iv.birth, xmax, color="tab:red", lw=1.5)
        y += 1
    ax.set_yticks([])
    ax.set_xlabel("epsilon")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_point_cloud_2d(pts: np.ndarray, path: str | Path, labels: list[str] | None = None, title: str = "Point cloud") -> None:
    plt = _mpl()
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape[1] < 2:
        raise DataError(f"need >=2 coordinates to plot, got {pts.shape[1]}")
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(pts[:, 0], pts[:, 1], s=18, c="tab:blue")
    if labels is not None:
        for i, lab in enumerate(labels):
            ax.annotate(str(lab), (pts[i, 0], pts[i, 1]), fontsize=6, alpha=0.7)
    ax.set_title(title)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_persistence_diagram(diagram: np.ndarray, path: str | Path, title: str = "Persistence diagram") -> None:
    """diagram: (n, 2) array of (birth, death) off-diagonal points."""
    plt = _mpl()
    diagram = np.asarray(diagram, dtype=np.float64)
    lim = float(max(diagram.max(), 1e-9)) if diagram.size else 1.0
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, lim], [0, lim], "k--", lw=0.8, alpha=0.5)
    if diagram.size:
        ax.scatter(diagram[:, 0], diagram[:, 1], s=14, c="tab:blue")
    ax.set_xlabel("birth")
    ax.set_ylabel("death")
    ax.set_title(title)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_persistence_landscape(xgrid: np.ndarray, F: np.ndarray, path: str | Path, title: str = "Persistence landscape") -> None:
    """F: (n_levels, len(xgrid)); plots the top few landscape levels."""
    plt = _mpl()
    xgrid = np.asarray(xgrid)
    F = np.asarray(F)
    k = min(F.shape[0], 8)
    fig, ax = plt.subplots(figsize=(8, 5))
    for j in range(k):
        ax.plot(xgrid, F[j], label=f"level {j}", lw=1.2)
    ax.set_xlabel("x")
    ax.set_ylabel("height")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_persistence_image(img: np.ndarray, path: str | Path, title: str = "Persistence image") -> None:
    plt = _mpl()
    img = np.asarray(img)
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(img, origin="lower", cmap="viridis")
    ax.set_xlabel("birth")
    ax.set_ylabel("death")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_depth_heatmap(H, fracs, layers, path, title="Betti heatmap (scale x layer)") -> None:
    """H: (n_fracs, n_layers) array of a metric over relative scale x layer."""
    plt = _mpl()
    H = np.asarray(H)
    fracs = np.asarray(fracs)
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(H, aspect="auto", origin="lower", cmap="magma", interpolation="nearest")
    ax.set_yticks(np.arange(len(fracs)), np.round(fracs, 2))
    ax.set_xticks(np.arange(len(layers)), [int(L) for L in layers])
    ax.set_xlabel("layer (depth)")
    ax.set_ylabel("scale = frac x nn(layer)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_betti_over_depth(profile, path, title="Topological activity over depth") -> None:
    """profile: dict layer -> {total_persistence, beta_peak, ...} from depth_profile."""
    plt = _mpl()
    layers = sorted(profile)
    tot = [profile[L]["total_persistence"] for L in layers]
    peak = [profile[L]["beta_peak"] for L in layers]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(layers, tot, "o-", color="tab:blue", label="total persistence")
    ax2 = ax.twinx()
    ax2.plot(layers, peak, "s--", color="tab:orange", label="peak $\\beta_1$")
    ax.set_xlabel("layer (depth)")
    ax.set_ylabel("total persistence")
    ax2.set_ylabel("peak $\\beta_1$")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_mapper(g, path, title="Mapper graph") -> None:
    """g: MapperGraph. Nodes placed at the midpoint of their lens interval, sized by beta_1."""
    plt = _mpl()
    mids = [0.5 * (n.interval[0] + n.interval[1]) for n in g.nodes]
    fig, ax = plt.subplots(figsize=(9, 4))
    for (i, j, nov) in g.edges:
        ax.plot([mids[i], mids[j]], [-0.35, -0.35], color="0.6", lw=1.5)
    for i, n in enumerate(g.nodes):
        size = 60 + 120 * n.beta1
        ax.scatter(mids[i], 0, s=size, c="tab:blue" if n.beta1 else "0.7",
                   edgecolor="k", linewidth=0.6, zorder=3)
        ax.annotate(str(n.beta1), (mids[i], 0.15), ha="center", fontsize=8)
    ax.set_xlabel("lens value (interval midpoint)")
    ax.set_yticks([])
    ax.set_title(title)
    ax.set_ylim(-0.8, 0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_attractor_overlay(pts, loops, path, title="Attractor overlay") -> None:
    """pts: (n, >=2) array; loops: list of vertex-index sequences (each a cycle)."""
    plt = _mpl()
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape[1] < 2:
        raise DataError(f"need >=2 coordinates to plot, got {pts.shape[1]}")
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(loops))))
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(pts[:, 0], pts[:, 1], s=16, c="0.5", zorder=2)
    for k, cyc in enumerate(loops):
        idx = list(cyc)
        if len(idx) < 2:
            continue
        ring = np.array(idx + [idx[0]])
        ax.plot(pts[ring, 0], pts[ring, 1], color=colors[k % len(colors)], lw=1.6, zorder=3)
        ax.scatter(pts[idx, 0], pts[idx, 1], s=22, color=colors[k % len(colors)], zorder=4)
    ax.set_title(title)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_convergence(conv, path, title="Answer-token convergence over depth") -> None:
    """conv: Convergence (from dynamics.convergence)."""
    plt = _mpl()
    L = [float(x) for x in conv.layers]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(L, conv.mean_pairwise, "o-", ms=3, label="mean pairwise dist")
    ax2 = ax.twinx()
    ax2.plot(L, conv.mean_dist, "s-", ms=3, color="tab:orange", label="mean dist to centroid")
    ax2.plot(L, conv.centroid_norm, ":", color="tab:green", label="centroid norm")
    ax.set_xlabel("layer (depth)")
    ax.set_ylabel("mean pairwise distance")
    ax2.set_ylabel("distance to centroid / norm")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
