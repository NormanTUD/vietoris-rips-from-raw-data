from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vrtda import reports as R
from vrtda import FilteredComplex, persistent_homology
from vrtda.beartype_guard import beartype_module


def test_report_markdown_structure() -> None:
    rep = R.Report("My Report")
    rep.section("Intro", ["line one", "line two"])
    rep.table("Table", ["a", "b"], [[1, 2.5], [3, 4.0]])
    md = rep.to_markdown()
    assert "# My Report" in md
    assert "## Intro" in md
    assert "## Table" in md
    assert "| a | b |" in md
    assert "| 1 | 2.5 |" in md


def test_report_text_structure() -> None:
    rep = R.Report("T")
    rep.section("S", ["x"])
    txt = rep.to_text()
    assert "T" in txt
    assert "S" in txt
    assert "x" in txt
    assert "#" not in txt.split("T")[0]  # no markdown headers in text form


def test_report_write(tmp_path: Path) -> None:
    rep = R.Report("W")
    rep.section("S", ["hello"])
    p = rep.write(tmp_path / "out.md", fmt="md")
    assert p.exists()
    assert "# W" in p.read_text()


def test_report_float_formatting() -> None:
    rep = R.Report()
    rep.table("T", ["x"], [[0.123456789]])
    md = rep.to_markdown()
    assert "0.1235" in md  # %.4g


def test_betti_table_shape() -> None:
    eps = [0.0, 1.0, 2.0]
    arr = np.array([[3, 0, 0], [1, 1, 0], [1, 1, 1]])
    headers, rows = R.betti_table(eps, arr)
    assert headers == ["eps", "b0", "b1", "b2"]
    assert len(rows) == 3
    assert rows[0] == [0.0, 3, 0, 0]
    assert rows[2] == [2.0, 1, 1, 1]


def test_betti_table_from_real_complex() -> None:
    C = FilteredComplex.from_explicit([
        (0.0, 0, (0,)), (0.0, 0, (1,)), (0.0, 0, (2,)),
        (1.0, 1, (0, 1)), (1.0, 1, (1, 2)), (1.0, 1, (2, 0)),
    ])
    bc = persistent_homology(C)
    eps = [0.0, 1.0, 5.0]
    arr = bc.betti_function(eps)
    headers, rows = R.betti_table(eps, arr)
    # at eps=0: 3 components; at eps=1: 1 comp + 1 loop
    assert rows[0][1] == 3
    assert rows[1][1] == 1 and rows[1][2] == 1


beartype_module(__name__)
