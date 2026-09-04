import numpy as np
import pytest

from vrtda import FilteredComplex, betti_at, cohomology_at
from vrtda.cohomology import assert_homology_cohomology_match


def _sphere():
    return FilteredComplex.from_explicit([
        (0.0, 0, (0,)), (0.0, 0, (1,)), (0.0, 0, (2,)), (0.0, 0, (3,)),
        (1.0, 1, (0, 1)), (1.0, 1, (0, 2)), (1.0, 1, (0, 3)),
        (1.0, 1, (1, 2)), (1.0, 1, (1, 3)), (1.0, 1, (2, 3)),
        (2.0, 2, (0, 1, 2)), (2.0, 2, (0, 1, 3)), (2.0, 2, (0, 2, 3)), (2.0, 2, (1, 2, 3)),
    ])


def test_cohomology_equals_homology_sphere():
    C = _sphere()
    eps = float(C.values.max())
    assert cohomology_at(C, eps) == betti_at(C, eps) == [1, 0, 1]


def test_cohomology_match_helper_passes():
    from vrtda.complexes import make_torus_grid_complex
    C = make_torus_grid_complex(2, (3, 3))
    eps = float(C.values.max())
    assert_homology_cohomology_match(C, [0.0, 1.0, eps])  # should not raise


def test_cohomology_at_intermediate_eps():
    from vrtda import pairwise_distances, build_rips
    from vrtda.generators import circle_grid
    pts = circle_grid(24)
    D = pairwise_distances(pts)
    C = build_rips(pts, D, 1.2, max_dim=2)
    for eps in [0.0, 0.4, 0.8, 1.2]:
        h = betti_at(C, eps)
        c = cohomology_at(C, eps)
        assert h == c, f"eps={eps}: homology {h} != cohomology {c}"
