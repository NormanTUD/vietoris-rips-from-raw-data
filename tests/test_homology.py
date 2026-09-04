import numpy as np
import pytest

from vrtda import FilteredComplex, persistent_homology, betti_at, cohomology_at


def disk():
    return FilteredComplex.from_explicit([
        (0.0, 0, (0,)), (0.0, 0, (1,)), (0.0, 0, (2,)),
        (1.0, 1, (0, 1)), (1.0, 1, (1, 2)), (1.0, 1, (0, 2)),
        (2.0, 2, (0, 1, 2)),
    ])


def sphere():
    return FilteredComplex.from_explicit([
        (0.0, 0, (0,)), (0.0, 0, (1,)), (0.0, 0, (2,)), (0.0, 0, (3,)),
        (1.0, 1, (0, 1)), (1.0, 1, (0, 2)), (1.0, 1, (0, 3)),
        (1.0, 1, (1, 2)), (1.0, 1, (1, 3)), (1.0, 1, (2, 3)),
        (2.0, 2, (0, 1, 2)), (2.0, 2, (0, 1, 3)), (2.0, 2, (0, 2, 3)), (2.0, 2, (1, 2, 3)),
    ])


def solid_tetrahedron():
    return FilteredComplex.from_explicit(
        [
            (0.0, 0, (0,)), (0.0, 0, (1,)), (0.0, 0, (2,)), (0.0, 0, (3,)),
            (1.0, 1, (0, 1)), (1.0, 1, (0, 2)), (1.0, 1, (0, 3)),
            (1.0, 1, (1, 2)), (1.0, 1, (1, 3)), (1.0, 1, (2, 3)),
            (2.0, 2, (0, 1, 2)), (2.0, 2, (0, 1, 3)), (2.0, 2, (0, 2, 3)), (2.0, 2, (1, 2, 3)),
            (3.0, 3, (0, 1, 2, 3)),
        ]
    )


CASES = [
    ("disk", disk, [1, 0, 0]),
    ("sphere", sphere, [1, 0, 1]),
    ("solid_tetrahedron", solid_tetrahedron, [1, 0, 0, 0]),
]


@pytest.mark.parametrize("name,fn,target", CASES, ids=[c[0] for c in CASES])
def test_rank_betti(name, fn, target):
    C = fn()
    eps = float(C.values.max())
    assert betti_at(C, eps) == target


@pytest.mark.parametrize("name,fn,target", CASES, ids=[c[0] for c in CASES])
def test_cohomology_matches(name, fn, target):
    C = fn()
    eps = float(C.values.max())
    assert cohomology_at(C, eps) == target


def test_torus_grid_betti():
    from vrtda.complexes import make_torus_grid_complex
    C = make_torus_grid_complex(2, (3, 3))
    eps = float(C.values.max())
    assert betti_at(C, eps) == [1, 2, 1]
    assert cohomology_at(C, eps) == [1, 2, 1]


def test_3torus_grid_betti():
    from vrtda.complexes import make_torus_grid_complex
    C = make_torus_grid_complex(3, (3, 3, 3))
    eps = float(C.values.max())
    assert betti_at(C, eps) == [1, 3, 3, 1]
    assert cohomology_at(C, eps) == [1, 3, 3, 1]


def test_betti_function_shape():
    C = sphere()
    eps = float(C.values.max())
    arr = betti_at(C, eps)
    assert len(arr) == 3


def test_empty_at_zero_eps_disconnected():
    # two isolated vertices, eps below the edge
    C = FilteredComplex.from_explicit([
        (0.0, 0, (0,)), (0.0, 0, (1,)), (5.0, 1, (0, 1)),
    ])
    assert betti_at(C, 0.0) == [2]
    assert betti_at(C, 5.0) == [1, 0]
