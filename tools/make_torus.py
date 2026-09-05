# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.26", "beartype>=0.18", "rich>=13"]
# ///
"""Generate a ground-truth point cloud CSV for TDA validation.

Supports n-tori (k = 2, 3, 4, 5, ...) via `--kind product --k K` (a grid of
points on T^k embedded in R^{2k}), a 1-torus (`circle`), a 3-D bagel (`donut`),
and a random S^k point cloud (`sphere`). `--verify` checks the ground-truth
topology (Betti numbers) against the exact cell complex, and -- for the low-dim
cases where a Vietoris-Rips complex on the cloud recovers it cleanly -- against
the cloud itself.
"""
import argparse
import sys
from math import comb
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[1])
TOOLS = str(Path(__file__).resolve().parent)
for _p in (ROOT, TOOLS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

from rich.console import Console

import _rich_ui
from vrtda import PointSet, generators as G
from vrtda import pairwise_distances, build_rips, persistent_homology, betti_at
from vrtda.complexes import make_torus_grid_complex, make_simplicial_sphere
from vrtda.beartype_guard import beartype_module


def build(args: argparse.Namespace) -> np.ndarray:
    if args.kind == "circle":
        if args.grid:
            return G.circle_grid(args.n, radius=args.radius)
        return G.circle(args.n, radius=args.radius, seed=args.seed, noise=args.noise)
    if args.kind == "product":
        if args.grid:
            return G.product_torus_grid(args.k, args.nper, radius=args.radius)
        return G.product_torus(args.k, n=args.n * args.nper, radius=args.radius, seed=args.seed, noise=args.noise)
    if args.kind == "donut":
        # IMPORTANT: this produces a bagel POINT CLOUD. If you later view that CSV via
        # Rips (e.g. `interactive.py --points out.csv`), a DENSE grid over-fills: the
        # 2-skeleton triangulates the torus void, so it reads beta_1 = ~n (the grid's
        # loops) and a huge beta_2 instead of the true [1, 2, 1]. To see a clean torus
        # use `interactive.py --shape donut` (exact T^2 cell complex), or keep the grid
        # sparse. See the IMPORTANT note block at the top of tools/interactive.py.
        if args.grid:
            return G.donut_grid(args.n, args.nper, R=args.radius, r=args.minor)
        return G.donut(args.n * args.nper, R=args.radius, r=args.minor, seed=args.seed, noise=args.noise)
    if args.kind == "sphere":
        return G.sphere(args.n, dim=args.k, radius=args.radius, seed=args.seed, noise=args.noise)
    raise SystemExit(f"unknown kind {args.kind!r}")


def _expected_betti(kind: str, k: int) -> tuple[list[int], str]:
    if kind == "circle":
        return [1, 1], "S^1 (1-torus)"
    if kind == "product":
        return [comb(k, d) for d in range(k + 1)], f"T^{k}"
    if kind == "donut":
        return [1, 2, 1], "T^2 (bagel)"
    if kind == "sphere":
        b = [0] * (k + 1)
        b[0] = 1
        b[k] = 1
        return b, f"S^{k}"
    raise SystemExit(f"unknown kind {kind!r}")


def _exact_betti(kind: str, k: int) -> list[int]:
    if kind == "circle":
        C = make_simplicial_sphere(1)
    elif kind == "product":
        C = make_torus_grid_complex(k, (3,) * k)
    elif kind == "donut":
        C = make_torus_grid_complex(2, (3, 3))
    elif kind == "sphere":
        C = make_simplicial_sphere(k)
    else:
        raise SystemExit(f"unknown kind {kind!r}")
    return betti_at(C, float(C.values.max()))


def _rips_betti(X: np.ndarray, max_dim: int) -> list[int]:
    D = pairwise_distances(X)
    d = D.copy()
    np.fill_diagonal(d, np.inf)
    eps = 1.6 * float(d.min(axis=1).mean())
    C = build_rips(X, D, eps, max_dim=max_dim)
    return persistent_homology(C).betti_at(eps)


def _feature_radius(kind: str, radius: float, minor: float) -> float:
    """Shape feature size that, blurred by this much noise, makes the shape
    unrecognizable: the tube radius for the bagel, the circle/sphere radius otherwise."""
    return minor if kind == "donut" else radius


def _apply_randomizer(data: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    """Jitter the cloud. randomizer=0 -> exact; randomizer=1 -> noise std = one
    feature radius (scrambled); in between -> proportional jitter."""
    if args.randomizer <= 0.0:
        return data
    std = args.randomizer * _feature_radius(args.kind, args.radius, args.minor)
    rng = np.random.default_rng(args.seed)
    return data + rng.normal(0.0, std, data.shape)


# Donut clouds denser than this trip the generation-time warning. A 12x12 grid is the
# largest bagel whose Rips complex stays feasible; beyond it the 2-skeleton over-fills
# (wrong beta_1, huge beta_2) and the slider's eps cap falls below the outer surface's
# formation scale (incomplete 3D view). See the IMPORTANT block in tools/interactive.py.
_DENSE_DONUT_POINTS: int = 144


def _warn_dense_donut(args: argparse.Namespace, n_points: int, console: Console) -> None:
    """SAFEGUARD (generation time): a donut POINT CLOUD is topologically a torus, but
    viewing it via Vietoris-Rips later (e.g. `interactive.py --points out.csv`) breaks
    on a DENSE grid -- beta_1 stays stuck at the grid's loop count, beta_2 explodes,
    and the 3D view is incomplete (the eps cap is below the outer surface's formation
    scale). Warn HERE, at the source, and hand the user the reliable command: the exact
    T^2 cell complex (`--shape donut`), which reads [1,2,1] instantly and fully."""
    if args.kind != "donut" or n_points <= _DENSE_DONUT_POINTS:
        return
    out_html = Path(args.out).with_suffix(".html").name
    console.print(
        f"[yellow][bold]NOTE — dense donut ({n_points} pts).[/bold] If you load this CSV "
        f"via Rips ([bold]interactive.py --points {args.out}[/bold]), it over-fills: beta_1 "
        f"stays at the grid's loop count, beta_2 explodes, and the 3D view is incomplete. "
        f"To SEE a clean torus, use the exact cell complex instead:\n"
        f"   [bold]uv run tools/interactive.py --shape donut --nper {args.nper} --out {out_html}[/bold][/yellow]"
    )


def verify(kind: str, k: int, cloud: np.ndarray, randomizer: float = 0.0,
           console: Console | None = None) -> None:
    c = console or Console()
    expected, label = _expected_betti(kind, k)
    rows: list[tuple[str, str]] = [("expected", _rich_ui.fmt_betti(expected))]
    with _rich_ui.timed(c, "Building exact complex + homology"):
        exact = _exact_betti(kind, k)
    rows.append(("exact complex", _rich_ui.fmt_betti(exact) + ("  [green]✓[/green]" if exact == expected else "  [red]✗[/red]")))
    assert exact == expected, f"exact complex beta {exact} != expected {expected}"
    if randomizer > 0.0:
        rows.append(("rips-on-cloud", "(skipped: --randomizer jitter)"))
    elif kind == "circle":
        rb = _rips_betti(cloud, max_dim=2)
        rows.append(("rips-on-cloud", _rich_ui.fmt_betti(rb)))
        assert (list(rb) + [0])[:2] == [1, 1], f"rips beta {rb} != [1,1]"
    elif kind == "product" and k <= 2:
        rb = _rips_betti(cloud, max_dim=k + 1)
        rows.append(("rips-on-cloud", _rich_ui.fmt_betti(rb)))
        assert (list(rb) + [0, 0])[: len(expected)] == expected, f"rips beta {rb} != {expected}"
    else:
        rows.append(("rips-on-cloud", "(not checked: combinatorial wall)"))
    _rich_ui.result_table(f"Verify {label}", rows, c)
    c.print(f"[bold green]OK:[/bold green] {label} topology verified")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kind", choices=["circle", "product", "donut", "sphere"], required=True)
    p.add_argument("--k", type=int, default=2, help="torus dimension (product) / sphere dimension (sphere)")
    p.add_argument("--n", type=int, default=24, help="points (or nu for grids)")
    p.add_argument("--nper", type=int, default=8, help="points per circle (grid) / nv (donut)")
    p.add_argument("--radius", type=float, default=1.0)
    p.add_argument("--minor", type=float, default=0.35, help="donut minor radius")
    p.add_argument("--grid", action="store_true", help="use deterministic grid sampling")
    p.add_argument("--randomizer", type=float, default=0.0,
                   help="jitter in [0,1]: 0=exact, ~0.2=slightly jittered, 1=scrambled/unrecognizable")
    p.add_argument("--noise", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--verify", action="store_true", help="check ground-truth Betti numbers")
    p.add_argument("--out", required=True, help="output CSV path")
    args = p.parse_args()

    console = Console()
    _rich_ui.params_table(p, args, console)

    if not (0.0 <= args.randomizer <= 1.0):
        raise SystemExit("--randomizer must be in [0, 1]")

    data = _apply_randomizer(build(args), args)
    ps = PointSet(data, name=args.kind)
    ps.to_csv(args.out)
    _warn_dense_donut(args, ps.n, console)

    expected, label = _expected_betti(args.kind, args.k)
    _rich_ui.result_table(f"Ground truth: {label}", [
        ("output", args.out),
        ("points", str(ps.n)),
        ("ambient dim", str(ps.dim)),
        ("expected β", _rich_ui.fmt_betti(expected)),
    ], console)

    if args.verify:
        if args.kind == "product" and args.k >= 5:
            console.print("[yellow]note:[/yellow] verifying k>=5 exact torus homology can take ~40s")
        verify(args.kind, args.k, data, args.randomizer, console)
    return 0


beartype_module(__name__)


if __name__ == "__main__":
    raise SystemExit(main())
