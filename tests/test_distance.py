import math

import pytest

from vrtda.persistence import Barcode, Interval
from vrtda import distance as D


def bcd(intervals, dim=1):
    return Barcode(intervals=[Interval(b, d, dim, i) for i, (b, d) in enumerate(intervals)])


def test_empty_vs_empty():
    e = Barcode(intervals=[])
    assert D.bottleneck(e, e, dim=1) == 0.0
    assert D.p_wasserstein(e, e, dim=1, p=2) == 0.0


def test_identical_zero():
    a = bcd([(0.0, 2.0), (0.3, 5.0)])
    assert D.bottleneck(a, a, dim=1) == pytest.approx(0.0)
    assert D.p_wasserstein(a, a, dim=1, p=2) == pytest.approx(0.0)


def test_single_vs_empty():
    a = bcd([(0.0, 5.0)])
    e = Barcode(intervals=[])
    assert D.bottleneck(a, e, dim=1) == pytest.approx(2.5)  # L_inf height
    assert D.p_wasserstein(a, e, dim=1, p=2) == pytest.approx(5.0 / math.sqrt(2))
    assert D.p_wasserstein(a, e, dim=1, p=1) == pytest.approx(2.5)  # L1 height = (d-b)/2


def test_far_apart_go_to_diagonal():
    a = bcd([(0.0, 5.0)])
    b = bcd([(10.0, 15.0)])
    # both far apart -> each goes to its own diagonal (bottleneck = max height)
    assert D.bottleneck(a, b, dim=1) == pytest.approx(2.5)
    assert D.p_wasserstein(a, b, dim=1, p=2) == pytest.approx(5.0)


def test_moved_point_prefers_matching():
    a = bcd([(0.0, 1.0)])
    b = bcd([(0.0, 2.0)])
    assert D.bottleneck(a, b, dim=1) == pytest.approx(1.0)
    assert D.p_wasserstein(a, b, dim=1, p=2) == pytest.approx(1.0)


def test_symmetry():
    a = bcd([(0.0, 1.0), (0.5, 4.0)])
    b = bcd([(0.2, 2.0), (1.0, 5.0)])
    assert D.bottleneck(a, b, dim=1) == pytest.approx(D.bottleneck(b, a, dim=1))
    assert D.p_wasserstein(a, b, dim=1, p=2) == pytest.approx(D.p_wasserstein(b, a, dim=1, p=2))


def test_bottleneck_triangle_inequality():
    a = bcd([(0.0, 1.0), (0.5, 4.0)])
    b = bcd([(0.2, 2.0), (1.0, 5.0)])
    c = bcd([(0.1, 3.0)])
    d_ab = D.bottleneck(a, b, dim=1)
    d_bc = D.bottleneck(b, c, dim=1)
    d_ac = D.bottleneck(a, c, dim=1)
    assert d_ac <= d_ab + d_bc + 1e-12


def test_wasserstein_monotone_in_p():
    a = bcd([(0.0, 1.0), (0.5, 4.0), (2.0, 3.0)])
    b = bcd([(0.2, 2.0), (1.0, 5.0)])
    w1 = D.p_wasserstein(a, b, dim=1, p=1)
    w2 = D.p_wasserstein(a, b, dim=1, p=2)
    assert w1 <= w2 + 1e-9


def test_bottleneck_on_torus_grid():
    from vrtda.complexes import make_torus_grid_complex
    from vrtda.persistence import persistent_homology

    C = make_torus_grid_complex(2, (3, 3))
    bc = persistent_homology(C)
    assert D.bottleneck(bc, bc, dim=1) == pytest.approx(0.0)
    # empty vs torus: bottleneck = max L_inf height of the two essential loops
    e = Barcode(intervals=[])
    vals = [iv for iv in bc.of_dim(1)]
    assert D.bottleneck(bc, e, dim=1) > 0.0
