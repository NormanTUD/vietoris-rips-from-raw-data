from __future__ import annotations

import numpy as np

from vrtda.errors import DataError


def _mpl():
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


def plot_betti_function(epsilons, betti_arr, path, title="Betti function") -> None:
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


def plot_barcode(intervals, path, title="Persistence barcode") -> None:
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


def plot_point_cloud_2d(pts, path, labels=None, title="Point cloud") -> None:
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
