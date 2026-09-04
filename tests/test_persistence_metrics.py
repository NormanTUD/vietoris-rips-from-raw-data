from __future__ import annotations

import numpy as np
import pytest

from vrtda.beartype_guard import beartype_module
from vrtda.persistence import Barcode, Interval
from vrtda import persistence_metrics as M


def bc(*intervals: Interval) -> Barcode:
    return Barcode(intervals=list(intervals))


def test_persistence_values_sorted() -> None:
    b = bc(Interval(0.0, 1.0, 1, 0), Interval(0.0, 3.0, 1, 1), Interval(0.0, 2.0, 1, 2))
    assert M.persistence_values(b, dim=1) == [3.0, 2.0, 1.0]


def test_persistence_values_excludes_essential_by_default() -> None:
    b = bc(Interval(0.0, np.inf, 1, 0), Interval(0.0, 2.0, 1, 1))
    assert M.persistence_values(b, dim=1) == [2.0]
    # with cap, essential is included (capped)
    assert M.persistence_values(b, dim=1, cap=5.0) == [5.0, 2.0]


def test_entropy_single_is_zero() -> None:
    assert M.persistence_entropy(bc(Interval(0.0, 5.0, 1, 0))) == 0.0


def test_entropy_two_equal_is_log2_2() -> None:
    b = bc(Interval(0.0, 1.0, 1, 0), Interval(0.0, 1.0, 1, 1))
    assert M.persistence_entropy(b, dim=1) == pytest.approx(1.0)  # log2(2)


def test_entropy_equal_beats_unequal() -> None:
    eq = bc(Interval(0.0, 1.0, 1, 0), Interval(0.0, 1.0, 1, 1))
    un = bc(Interval(0.0, 1.0, 1, 0), Interval(0.0, 3.0, 1, 1))
    assert M.persistence_entropy(eq, dim=1) > M.persistence_entropy(un, dim=1)


def test_entropy_base() -> None:
    b = bc(Interval(0.0, 1.0, 1, 0), Interval(0.0, 1.0, 1, 1))
    assert M.persistence_entropy(b, dim=1, base=np.e) == pytest.approx(np.log(2.0))


def test_landscape_single_tent() -> None:
    b = bc(Interval(0.0, 2.0, 1, 0))
    xgrid, F = M.persistence_landscape(b, dim=1)
    assert F.shape == (1, len(xgrid))
    assert F[0].max() == pytest.approx(2.0)
    assert xgrid[np.argmax(F[0])] == pytest.approx(1.0, abs=0.1)  # tent centered at x=1


def test_landscape_monotone_in_level() -> None:
    b = bc(Interval(0.0, 3.0, 1, 0), Interval(0.0, 1.0, 1, 1))
    xgrid, F = M.persistence_landscape(b, dim=1)
    assert F.shape[0] == 2
    assert np.all(F[0] >= F[1] - 1e-12)  # top envelope >= second level
    assert F[1].max() == pytest.approx(1.0)  # second level = the smaller tent


def test_landscape_empty() -> None:
    xgrid, F = M.persistence_landscape(Barcode(intervals=[]), dim=1)
    assert F.max() == 0.0


def test_image_identical_barcodes_identical() -> None:
    def b() -> Barcode:
        return bc(Interval(0.1, 0.4, 1, 0), Interval(0.2, 0.9, 1, 1))
    i1 = M.persistence_image(b(), dim=1, eps_max=1.0)
    i2 = M.persistence_image(b(), dim=1, eps_max=1.0)
    np.testing.assert_allclose(i1, i2)
    assert i1.max() > 0


def test_image_mass_near_point() -> None:
    b = bc(Interval(0.25, 0.75, 1, 0))
    img = M.persistence_image(b, dim=1, eps_max=1.0, n_grid=40)
    ix, iy = np.unravel_index(np.argmax(img), img.shape)
    xs = np.linspace(0, 1.0, 40)
    assert abs(xs[ix] - 0.25) < 0.1
    assert abs(xs[iy] - 0.75) < 0.1


def test_image_empty() -> None:
    img = M.persistence_image(Barcode(intervals=[]), dim=1)
    assert img.max() == 0.0


def test_diagram_points() -> None:
    b = bc(Interval(0.1, 0.4, 1, 0), Interval(0.0, np.inf, 1, 1))
    d = M.persistence_diagram(b, dim=1)
    assert d.shape == (1, 2)  # essential excluded
    np.testing.assert_allclose(d[0], [0.1, 0.4])


beartype_module(__name__)
