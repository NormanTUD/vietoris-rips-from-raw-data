from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vrtda import FilteredComplex, persistent_homology
from vrtda.barcodes import (
    barcode_to_rows,
    save_barcode_csv,
    load_barcode_csv,
    persistence_summary_csv,
)
from vrtda.beartype_guard import beartype_module


def _cycle() -> FilteredComplex:
    return FilteredComplex.from_explicit([
        (0.0, 0, (0,)), (0.0, 0, (1,)), (0.0, 0, (2,)),
        (1.0, 1, (0, 1)), (1.0, 1, (1, 2)), (1.0, 1, (2, 0)),
    ])


def test_barcode_to_rows() -> None:
    bc = persistent_homology(_cycle())
    rows = barcode_to_rows(bc)
    assert rows
    hdr = ("dim", "birth", "death", "birth_simplex", "death_simplex")
    assert len(rows[0]) == len(hdr)
    # the loop interval is essential -> death "inf"
    deaths = [r[2] for r in rows]
    assert "inf" in deaths


def test_csv_roundtrip(tmp_path: Path) -> None:
    bc = persistent_homology(_cycle())
    p = save_barcode_csv(bc, tmp_path / "bar.csv")
    back = load_barcode_csv(p)
    orig = sorted(iv.as_tuple() for iv in bc.intervals)
    got = sorted(iv.as_tuple() for iv in back.intervals)
    assert orig == got
    # betti functions agree
    for eps in [0.0, 1.0, 100.0]:
        assert bc.betti_at(eps) == back.betti_at(eps)


def test_csv_preserves_essential(tmp_path: Path) -> None:
    bc = persistent_homology(_cycle())
    p = save_barcode_csv(bc, tmp_path / "bar.csv")
    back = load_barcode_csv(p)
    assert len([iv for iv in back.of_dim(1) if iv.is_essential]) == 1


def test_summary_csv(tmp_path: Path) -> None:
    from vrtda.complexes import make_torus_grid_complex
    C = make_torus_grid_complex(2, (3, 3))
    bc = persistent_homology(C)
    p = persistence_summary_csv(bc, tmp_path / "sum.csv")
    import csv
    with open(p, newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert rows
    d1 = next(r for r in rows if r["dim"] == "1")
    assert int(d1["essential"]) == 2
    d2 = next(r for r in rows if r["dim"] == "2")
    assert int(d2["essential"]) == 1


def test_load_missing_file() -> None:
    with pytest.raises(Exception):
        load_barcode_csv("/nonexistent/bar.csv")


beartype_module(__name__)
