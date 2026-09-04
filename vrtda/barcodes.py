from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from vrtda.persistence import Barcode, Interval


def barcode_to_rows(barcode: Barcode) -> list[tuple[int, str, str, int, int]]:
    rows: list[tuple[int, str, str, int, int]] = []
    for iv in barcode.intervals:
        death = "inf" if iv.is_essential else f"{iv.death:.9g}"
        rows.append((iv.dim, f"{iv.birth:.9g}", death, iv.birth_simplex, iv.death_simplex))
    return rows


def save_barcode_csv(barcode: Barcode, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dim", "birth", "death", "birth_simplex", "death_simplex"])
        for r in barcode_to_rows(barcode):
            w.writerow(r)
    return path


def load_barcode_csv(path: str | Path) -> Barcode:
    path = Path(path)
    intervals = []
    with open(path, newline="") as fh:
        r = csv.DictReader(fh)
        for row in r:
            death = row["death"].strip()
            death = np.inf if death == "inf" else float(death)
            intervals.append(
                Interval(
                    birth=float(row["birth"]),
                    death=death,
                    dim=int(row["dim"]),
                    birth_simplex=int(row["birth_simplex"]),
                    death_simplex=int(row["death_simplex"]),
                )
            )
    return Barcode(intervals=intervals)


def persistence_summary_csv(barcode: Barcode, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    s = barcode.summary()
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dim", "n_intervals", "essential", "max_length"])
        for d in sorted(s["dims"]):
            v = s["dims"][d]
            w.writerow([d, v["n"], v["essential"], f"{v['max_length']:.9g}"])
    return path


from vrtda.beartype_guard import beartype_module as _beartype_module

_beartype_module(__name__)
