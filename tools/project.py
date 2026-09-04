# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.26", "beartype>=0.18"]
# ///
"""Project a point cloud via feature-selection + dimensionality reduction.

Reads a point cloud (from a CSV or from the transformer dataset), optionally selects
the top-K variance dimensions, applies a reduction (pca/umap/tsne), and writes the
reduced cloud (2D/3D/...) as CSV plus a small text report."""
import argparse
import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np

from vrtda import PointSet, datasets, reduction
from vrtda.beartype_guard import beartype_module


def load_points(args: argparse.Namespace) -> PointSet:
    if args.points:
        return PointSet.from_csv(
            args.points,
            value_cols=args.value_cols,
            index_cols=args.index_cols,
            name=Path(args.points).stem,
        )
    if args.source == "token_cloud":
        return datasets.load_token_cloud(layer=args.layer, normalize=args.unit)
    if args.source == "layer_points":
        layers = [args.layer] if args.layer is not None else None
        return datasets.load_layer_points(layers=layers, normalize=args.unit)
    if args.source == "residual_norms":
        mat, labels = datasets.load_residual_matrix(kind=args.kind)
        return PointSet(mat, labels=labels, name=f"residual_{args.kind}_traj")
    raise SystemExit(f"unknown source {args.source!r}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--points", default=None, help="point cloud CSV")
    src.add_argument("--source", choices=["token_cloud", "layer_points", "residual_norms"], default=None)
    p.add_argument("--value-cols", nargs="*", default=None)
    p.add_argument("--index-cols", nargs="*", default=None)
    p.add_argument("--layer", type=int, default=None, help="layer index (token_cloud / single layer_points)")
    p.add_argument("--kind", default="norms", choices=["norms", "cosines", "deltas"])
    p.add_argument("--unit", action="store_true", help="unit-normalize each point before reduction")
    p.add_argument("--dims", default=None, help='explicit dims, e.g. "0,3,7" (overrides --top-k)')
    p.add_argument("--top-k", type=int, default=None, help="keep the top-K variance dims before reduction")
    p.add_argument("--method", default="pca", choices=["pca", "umap", "tsne"])
    p.add_argument("--components", type=int, default=2, help="number of reduced coordinates")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True, help="output reduced-cloud CSV")
    p.add_argument("--out-report", default=None, help="optional text report path")
    args = p.parse_args()

    ps = load_points(args)
    X = ps.data
    labels = ps.labels

    # feature selection
    sel = None
    if args.dims is not None:
        sel = [int(x) for x in args.dims.split(",")]
        X = X[:, sel]
    elif args.top_k is not None:
        sel = reduction.top_variance_dims(X, args.top_k)
        X = X[:, sel]

    if args.unit:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        X = X / norms

    reduced, meta = reduction.reduce(X, args.method, n_components=args.components)

    out_ps = PointSet(reduced, labels=labels, name=f"proj_{args.method}_{args.components}d")
    out_ps.meta["source"] = ps.name
    out_ps.meta["selected_dims"] = sel
    out_ps.to_csv(args.out)
    print(f"wrote {args.out}: n={out_ps.n} dim={out_ps.dim} method={args.method}")

    report = [
        f"source: {ps.name} (n={ps.n}, dim={ps.dim})",
        f"selected_dims: {'none' if sel is None else str(len(sel))} {'(' + str(sel[:12]) + ('...' if len(sel) > 12 else '') + ')' if sel else ''}".strip(),
        f"method: {args.method}, components={args.components}",
    ]
    if args.method == "pca":
        evr = meta["explained_variance_ratio"]
        report.append("explained_variance_ratio: " + " ".join(f"{v:.4f}" for v in evr))
        report.append(f"cumulative: {evr.sum():.4f}")
    report.append(f"output: {args.out}")
    txt = "\n".join(report) + "\n"
    if args.out_report:
        Path(args.out_report).write_text(txt)
        print(f"wrote {args.out_report}")
    print("\n".join(report))
    return 0


beartype_module(__name__)


if __name__ == "__main__":
    raise SystemExit(main())
