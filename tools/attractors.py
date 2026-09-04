# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.26"]
# ///
"""Detect and compare persistent 'attractor' features across point clouds.

Compares persistent homology (essential + long-lived intervals, total persistence)
across a set of point clouds — e.g. the token cloud at each network layer, or a set
of CSVs — and writes a tabular report."""
import argparse
import csv
import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np

from vrtda import (
    PointSet,
    pairwise_distances,
    build_rips,
    persistent_homology,
    datasets,
    attractors,
)


def _nn_mean(D):
    d = D.copy(); np.fill_diagonal(d, np.inf)
    return float(d.min(1).mean())


def _barcode_for(pts: np.ndarray, metric: str, max_dim: int, frac: float):
    D = pairwise_distances(pts, metric)
    nn = _nn_mean(D)
    eps_max = frac * nn
    C = build_rips(pts, D, eps_max, max_dim=max_dim)
    return persistent_homology(C), eps_max, C.n_simplices


def collect_clouds(args):
    clouds = {}
    if args.layers is not None:
        for L in args.layers:
            ps = datasets.load_token_cloud(layer=L)
            clouds[f"layer_{int(L):03d}"] = ps.data
    if args.csvs:
        for c in args.csvs:
            ps = PointSet.from_csv(c)
            clouds[Path(c).stem] = ps.data
    if args.source == "residual_norms":
        mat, labels = datasets.load_residual_matrix(kind=args.kind)
        clouds["residual_norms"] = mat
    return clouds


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--layers", type=int, nargs="*", default=None, help="layer indices for token-cloud comparison")
    p.add_argument("--csvs", nargs="*", default=None, help="point-cloud CSVs to compare")
    p.add_argument("--source", choices=["residual_norms"], default=None)
    p.add_argument("--kind", default="norms", choices=["norms", "cosines", "deltas"])
    p.add_argument("--metric", default="euclidean")
    p.add_argument("--max-dim", type=int, default=2)
    p.add_argument("--frac", type=float, default=2.0, help="eps_max as fraction of mean-nn")
    p.add_argument("--min-fraction", type=float, default=0.1, help="long-lived threshold as fraction of eps_max")
    p.add_argument("--out", default=None, help="output CSV report")
    args = p.parse_args()

    clouds = collect_clouds(args)
    if not clouds:
        raise SystemExit("nothing to compare: give --layers and/or --csvs and/or --source")

    rows = []
    for name, pts in clouds.items():
        bc, eps_max, nsimp = _barcode_for(pts, args.metric, args.max_dim, args.frac)
        row = {"name": name, "n": pts.shape[0], "dim": pts.shape[1], "nsimp": nsimp, "eps_max": eps_max}
        for d, v in attractors.per_dim_summary(bc, eps_max, args.min_fraction).items():
            row[f"b{d}_essential"] = v["essential"]
            row[f"b{d}_long_lived"] = v["long_lived"]
            row[f"b{d}_persist"] = v["total_persistence"]
        rows.append(row)

    # column order: fixed leading cols, then per-dim blocks
    md = args.max_dim
    dims = sorted({k for r in rows for k in r if k.startswith("b")})
    cols = ["name", "n", "dim", "nsimp", "eps_max"] + dims
    hdr = "\t".join(cols)
    print(hdr)
    for r in rows:
        print("\t".join(_fmt(r[c], c) for c in cols))

    if args.out:
        with open(args.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({c: r[c] for c in cols})
        print(f"\nwrote {args.out}")
    return 0


def _fmt(v, c):
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


if __name__ == "__main__":
    raise SystemExit(main())
