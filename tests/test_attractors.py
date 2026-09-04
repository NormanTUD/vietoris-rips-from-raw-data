from __future__ import annotations

import numpy as np
import pytest

from vrtda import FilteredComplex, persistent_homology
from vrtda.beartype_guard import beartype_module
from vrtda.complexes import make_torus_grid_complex
from vrtda import attractors as A


def _torus2() -> FilteredComplex:
    return make_torus_grid_complex(2, (3, 3))


def _cycle() -> FilteredComplex:
    return FilteredComplex.from_explicit([
        (0.0, 0, (0,)), (0.0, 0, (1,)), (0.0, 0, (2,)),
        (1.0, 1, (0, 1)), (1.0, 1, (1, 2)), (1.0, 1, (2, 0)),
    ])


def test_essential_counts_torus() -> None:
    C = _torus2()
    bc = persistent_homology(C)
    assert len(A.essential_intervals(bc, 0)) == 1
    assert len(A.essential_intervals(bc, 1)) == 2
    assert len(A.essential_intervals(bc, 2)) == 1
    # all dims total
    assert len(A.essential_intervals(bc)) == 4


def test_long_lived_threshold() -> None:
    bc = persistent_homology(_cycle())
    eps_max = 1.0
    # intervals: 3 H0 (essential len=1.0, two finite len=1.0) + 1 H1 loop (born@1.0 -> capped len 0.0)
    assert len(A.long_lived_intervals(bc, 0.0, eps_max=eps_max)) == 4
    assert len(A.long_lived_intervals(bc, 0.0, eps_max=eps_max, dim=0)) == 3
    # threshold 0.5 excludes only the zero-length loop
    assert len(A.long_lived_intervals(bc, 0.5, eps_max=eps_max)) == 3
    # monotonic in the threshold
    lo = len(A.long_lived_intervals(bc, 0.2, eps_max=eps_max))
    hi = len(A.long_lived_intervals(bc, 0.8, eps_max=eps_max))
    assert lo >= hi


def test_total_persistence_nonneg() -> None:
    bc = persistent_homology(_torus2())
    eps_max = 2.0
    for d in range(3):
        assert A.total_persistence(bc, eps_max, d) >= 0.0


def test_per_dim_summary_keys() -> None:
    bc = persistent_homology(_torus2())
    s = A.per_dim_summary(bc, 2.0, min_fraction=0.1)
    assert set(s.keys()) == {0, 1, 2}
    for d, v in s.items():
        assert set(v.keys()) == {"n", "essential", "long_lived", "total_persistence", "max_length"}
        assert v["n"] >= v["essential"]
        assert v["n"] >= v["long_lived"]


def test_per_dim_summary_consistent_with_essential() -> None:
    bc = persistent_homology(_torus2())
    s = A.per_dim_summary(bc, 2.0)
    assert s[0]["essential"] == 1
    assert s[1]["essential"] == 2
    assert s[2]["essential"] == 1


def test_max_persistence_cycle() -> None:
    bc = persistent_homology(_cycle())
    # the loop interval is essential -> capped length = eps_max - birth = 1.0 - 1.0 = 0? no, birth=1
    # H1 loop born at 1.0, essential -> capped length = 1.0 (eps_max) - 1.0 = 0.0
    # H0 essentials: born 0, essential -> capped 1.0
    assert A.max_persistence(bc, 1.0, 0) == pytest.approx(1.0)
    assert A.max_persistence(bc, 1.0, 1) == pytest.approx(0.0)


def test_compare_rows() -> None:
    bc_a = persistent_homology(_torus2())
    bc_b = persistent_homology(_cycle())
    rows = A.compare({"torus": bc_a, "cycle": bc_b}, eps_max=2.0)
    assert len(rows) == 2
    torus_row = next(r for r in rows if r["name"] == "torus")
    cycle_row = next(r for r in rows if r["name"] == "cycle")
    assert torus_row["b1_essential"] == 2
    assert cycle_row["b1_essential"] == 1
    assert "b2_total_persistence" in torus_row


beartype_module(__name__)
