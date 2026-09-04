# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.26", "beartype>=0.18"]
# ///
"""One-command TDA analysis of a point-cloud CSV.

Reads a CSV (one point per row), builds the Vietoris-Rips filtration, computes
persistent homology, and writes a full analysis to --out-dir:
    barcode.csv         persistence barcode
    betti_function.csv  beta_k(eps) over eps
    attractors.csv      essential / long-lived / total-persistence per dimension
    report.md           human-readable summary
Add --plots to also save betti.png / barcode.png / cloud.png (needs matplotlib,
declared in the header automatically only when --plots is passed is NOT possible in
PEP-723, so for plots use:  uv run --with matplotlib tools/analyze.py ... --plots).

Quick start:
    uv run tools/analyze.py --points mydata.csv --out-dir out/
    uv run tools/analyze.py --points mydata.csv --value-cols x y --out-dir out/
    uv run --with matplotlib tools/analyze.py --points mydata.csv --plots --out-dir out/
"""
import argparse
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
    attractors,
    reports,
)
from vrtda.beartype_guard import beartype_module


def _nn(D: np.ndarray) -> float:
    d = D.copy(); np.fill_diagonal(d, np.inf)
    return float(d.min(1).mean())


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--points", required=True, help="point-cloud CSV (one point per row)")
    p.add_argument("--value-cols", nargs="*", default=None, help="coordinate columns (default: all non-index)")
    p.add_argument("--index-cols", nargs="*", default=None, help="label columns")
    p.add_argument("--metric", default="euclidean",
                   choices=["euclidean", "squared", "manhattan", "cosine", "normalized_euclidean"])
    p.add_argument("--max-dim", type=int, default=2)
    p.add_argument("--frac", type=float, default=1.6, help="eps_max as fraction of mean nearest-neighbour distance")
    p.add_argument("--min-fraction", type=float, default=0.1, help="long-lived threshold (fraction of eps_max)")
    p.add_argument("--out-dir", required=True, help="output directory")
    p.add_argument("--plots", action="store_true", help="also save PNG plots (requires matplotlib)")
    args = p.parse_args()

    ps = PointSet.from_csv(args.points, value_cols=args.value_cols, index_cols=args.index_cols)
    D = pairwise_distances(ps.data, args.metric)
    nn = _nn(D)
    eps_max = args.frac * nn
    C = build_rips(ps.data, D, eps_max, max_dim=args.max_dim)
    bc = persistent_homology(C)
    md = max(args.max_dim, bc.max_dim())

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    # barcode
    from vrtda.barcodes import save_barcode_csv, persistence_summary_csv
    save_barcode_csv(bc, out / "barcode.csv")
    persistence_summary_csv(bc, out / "attractors.csv")

    # betti function
    epsilons = np.linspace(0.9 * nn, eps_max, 20)
    arr = bc.betti_function(list(epsilons))
    with open(out / "betti_function.csv", "w") as fh:
        fh.write("eps," + ",".join(f"b{k}" for k in range(md + 1)) + "\n")
        for i, e in enumerate(epsilons):
            b = arr[i]
            fh.write(f"{e:.6g}," + ",".join(str(int(x)) for x in b) + "\n")

    # attractor summary + report
    s = attractors.per_dim_summary(bc, eps_max, args.min_fraction)
    empty = {"n": 0, "essential": 0, "long_lived": 0, "total_persistence": 0.0, "max_length": 0.0}
    rep = reports.Report(f"TDA report — {ps.name}")
    rep.section("Input", [
        f"file: {args.points}",
        f"points: {ps.n}, dims: {ps.dim}, metric: {args.metric}",
        f"mean nearest-neighbour distance: {nn:.4g}",
        f"eps_max: {eps_max:.4g} (= {args.frac} x nn), max_dim: {args.max_dim}",
        f"simplices in filtration: {C.n_simplices}",
    ])
    hdr = ["dim", "intervals", "essential", "long_lived", "total_persistence", "max_length"]
    rows = [[d, s.get(d, empty)["n"], s.get(d, empty)["essential"], s.get(d, empty)["long_lived"],
             s.get(d, empty)["total_persistence"], s.get(d, empty)["max_length"]] for d in range(md + 1)]
    rep.table("Persistent (attractor) features", hdr, rows)
    bhdr, brows = reports.betti_table(epsilons, arr)
    rep.table("Betti function beta_k(eps)", bhdr, brows)
    rep.write(out / "report.md", fmt="md")

    # optional plots
    if args.plots:
        try:
            from vrtda import plotting
            plotting.plot_betti_function(epsilons, arr, out / "betti.png", title="Betti function")
            plotting.plot_barcode(bc.intervals, out / "barcode.png", title="Persistence barcode")
            plotting.plot_point_cloud_2d(ps.data, out / "cloud.png", labels=ps.labels, title="Point cloud")
        except Exception as e:
            print(f"[warn] plotting skipped: {e}")

    # console summary
    print(f"n={ps.n} dim={ps.dim} metric={args.metric} nn={nn:.4g} eps_max={eps_max:.4g} nsimp={C.n_simplices}")
    for d in range(md + 1):
        v = s.get(d, empty)
        print(f"  b{d}: intervals={v['n']:4d}  essential={v['essential']}  "
              f"long_lived={v['long_lived']}  total_persistence={v['total_persistence']:.4g}")
    print(f"wrote: {out}/report.md, barcode.csv, betti_function.csv, attractors.csv"
          + (", betti.png, barcode.png, cloud.png" if args.plots else ""))
    return 0


beartype_module(__name__)


if __name__ == "__main__":
    raise SystemExit(main())
