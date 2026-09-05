# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.26", "beartype>=0.18", "rich>=13"]
# ///
"""Generate a ground-truth bouquet (wedge) of n k-spheres for TDA validation.

The default (k=1) is the figure-of-n: n circles wedged at a common point, giving
beta_0 = 1 and beta_1 = n (a figure-8 for n=2). For k>=2 the summands are higher
spheres meeting only at the shared point, giving a free H_k of rank n. A point
cloud is written for k=1; for k>=2 only the exact complex is verified.
"""
import argparse
import sys
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
from vrtda.complexes import make_bouquet_complex
from vrtda.beartype_guard import beartype_module


def build(args: argparse.Namespace) -> np.ndarray:
    return G.bouquet_circles(args.n, n_per=args.nper, radius=args.radius)


def _expected_betti(n: int, k: int) -> list[int]:
    b = [0] * (k + 1)
    b[0] = 1
    b[k] = n
    return b


def _rips_betti(X: np.ndarray, max_dim: int) -> list[int]:
    D = pairwise_distances(X)
    d = D.copy()
    np.fill_diagonal(d, np.inf)
    eps = 1.6 * float(d.min(axis=1).mean())
    C = build_rips(X, D, eps, max_dim=max_dim)
    return persistent_homology(C).betti_at(eps)


def _apply_randomizer(data: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    """Jitter the cloud. randomizer=0 -> exact; randomizer=1 -> noise std = the circle
    radius (scrambled); in between -> proportional jitter."""
    if args.randomizer <= 0.0:
        return data
    rng = np.random.default_rng(args.seed)
    return data + rng.normal(0.0, args.randomizer * args.radius, data.shape)


def verify(n: int, k: int, cloud: np.ndarray | None, randomizer: float = 0.0,
           console: Console | None = None) -> None:
    c = console or Console()
    expected = _expected_betti(n, k)
    rows: list[tuple[str, str]] = [("expected", _rich_ui.fmt_betti(expected))]
    with _rich_ui.timed(c, f"Building exact bouquet of {n} S^{k} + homology"):
        C = make_bouquet_complex(n, k)
        exact = betti_at(C, float(C.values.max()))
    ok = exact == expected
    rows.append(("exact complex", _rich_ui.fmt_betti(exact) + ("  [green]✓[/green]" if ok else "  [red]✗[/red]")))
    assert ok, f"exact complex beta {exact} != expected {expected}"
    if randomizer > 0.0:
        rows.append(("rips-on-cloud", "(skipped: --randomizer jitter)"))
    elif k == 1 and cloud is not None:
        rb = _rips_betti(cloud, max_dim=2)
        rows.append(("rips-on-cloud (0,1)", _rich_ui.fmt_betti(list(rb[:2]))))
        assert (list(rb) + [0])[:2] == [1, n], f"rips beta {rb} != [1,{n}]"
    else:
        rows.append(("rips-on-cloud", "(not checked)"))
    _rich_ui.result_table(f"Verify bouquet of {n} S^{k}", rows, c)
    c.print(f"[bold green]OK:[/bold green] bouquet topology verified")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=3, help="number of summands (circles for k=1)")
    p.add_argument("--k", type=int, default=1, help="summand dimension (1 = circles / figure-of-n)")
    p.add_argument("--nper", type=int, default=16, help="points per circle (k=1 grid)")
    p.add_argument("--radius", type=float, default=1.0)
    p.add_argument("--randomizer", type=float, default=0.0,
                   help="jitter in [0,1]: 0=exact, ~0.2=slightly jittered, 1=scrambled/unrecognizable (k=1 cloud)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--verify", action="store_true", help="check ground-truth Betti numbers")
    p.add_argument("--out", default=None, help="output CSV path (k=1 point cloud)")
    args = p.parse_args()

    console = Console()
    _rich_ui.params_table(p, args, console)

    if not (0.0 <= args.randomizer <= 1.0):
        raise SystemExit("--randomizer must be in [0, 1]")

    expected = _expected_betti(args.n, args.k)
    _rich_ui.result_table(f"Ground truth: bouquet of {args.n} S^{args.k}", [
        ("summands", str(args.n)),
        ("dimension", f"S^{args.k}"),
        ("expected β", _rich_ui.fmt_betti(expected)),
        ("output", args.out or "(none -- exact complex only)"),
    ], console)

    if args.k != 1:
        if args.out:
            raise SystemExit("--out not available for k>=2 (no point cloud); omit --out")
        verify(args.n, args.k, None, 0.0, console)
        return 0

    data = _apply_randomizer(build(args), args)
    if args.out:
        ps = PointSet(data, name="bouquet")
        ps.to_csv(args.out)
        console.print(f"[bold green]wrote[/bold green] {args.out}: n={ps.n} dim={ps.dim}")
    else:
        console.print(f"bouquet cloud: n={data.shape[0]} dim={data.shape[1]} (use --out to write CSV)")
    if args.verify:
        verify(args.n, args.k, data, args.randomizer, console)
    return 0


beartype_module(__name__)


if __name__ == "__main__":
    raise SystemExit(main())
