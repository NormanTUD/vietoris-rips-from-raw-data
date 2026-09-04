# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.26", "beartype>=0.18"]
# ///
"""Compute persistent homology of a point cloud and write barcode + summary CSVs."""
import argparse
import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np

from vrtda import PointSet, pairwise_distances, build_rips, persistent_homology
from vrtda.barcodes import save_barcode_csv, persistence_summary_csv
from vrtda.beartype_guard import beartype_module


def _nn_mean(D: np.ndarray) -> float:
    d = D.copy()
    np.fill_diagonal(d, np.inf)
    return float(d.min(axis=1).mean())


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--points", required=True, help="point cloud CSV (one point per row)")
    p.add_argument("--value-cols", nargs="*", default=None)
    p.add_argument("--index-cols", nargs="*", default=None)
    p.add_argument("--metric", default="euclidean")
    p.add_argument("--max-dim", type=int, default=2)
    p.add_argument("--frac", type=float, default=2.0, help="eps_max as fraction of mean-nn")
    p.add_argument("--eps-max", type=float, default=None, help="explicit eps_max (overrides --frac)")
    p.add_argument("--out", required=True, help="output barcode CSV")
    p.add_argument("--summary-out", default=None, help="optional summary CSV")
    args = p.parse_args()

    ps = PointSet.from_csv(args.points, value_cols=args.value_cols, index_cols=args.index_cols)
    D = pairwise_distances(ps.data, args.metric)
    nn = _nn_mean(D)
    eps_max = args.eps_max if args.eps_max is not None else args.frac * nn

    C = build_rips(ps.data, D, eps_max, max_dim=args.max_dim)
    bc = persistent_homology(C)
    out = save_barcode_csv(bc, args.out)
    print(f"wrote {out}")
    if args.summary_out:
        so = persistence_summary_csv(bc, args.summary_out)
        print(f"wrote {so}")

    s = bc.summary()
    per_dim = " ".join(
        f"b{d}={s['dims'][d]['n']} (ess {s['dims'][d]['essential']})" for d in sorted(s["dims"])
    )
    print(f"n={ps.n} dim={ps.dim} metric={args.metric} eps_max={eps_max:.4g} nsimp={C.n_simplices} | {per_dim}")
    return 0


beartype_module(__name__)


if __name__ == "__main__":
    raise SystemExit(main())
