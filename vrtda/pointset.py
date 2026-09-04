from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from vrtda import debug
from vrtda.errors import DataError, ShapeError


class PointSet:
    def __init__(
        self,
        data: np.ndarray,
        labels: list | None = None,
        meta: dict | None = None,
        name: str = "",
    ) -> None:
        data = np.asarray(data, dtype=np.float64)
        if data.ndim != 2:
            raise ShapeError(f"PointSet data must be 2D (N, D), got shape {data.shape}")
        if not np.all(np.isfinite(data)):
            raise DataError("PointSet data contains NaN/Inf; impute or remove first")
        self.data = data
        n = data.shape[0]
        if labels is None:
            labels = [str(i) for i in range(n)]
        if len(labels) != n:
            raise ShapeError(f"labels length {len(labels)} != N={n}")
        self.labels = list(labels)
        self.meta = dict(meta or {})
        self.name = name

    # ---- properties -------------------------------------------------------
    @property
    def n(self) -> int:
        return int(self.data.shape[0])

    @property
    def dim(self) -> int:
        return int(self.data.shape[1])

    def __len__(self) -> int:
        return self.n

    def __repr__(self) -> str:
        return f"PointSet(name={self.name!r}, N={self.n}, D={self.dim})"

    def __getitem__(self, idx):
        return self.data[idx]

    # ---- construction -----------------------------------------------------
    @classmethod
    def from_array(cls, data, labels=None, meta=None, name="") -> "PointSet":
        return cls(data, labels=labels, meta=meta, name=name)

    @classmethod
    def from_csv(
        cls,
        path,
        value_cols=None,
        index_cols=None,
        name=None,
    ) -> "PointSet":
        path = Path(path)
        if not path.exists():
            raise DataError(f"CSV not found: {path}")
        with open(path, newline="") as fh:
            reader = csv.reader(fh)
            header = next(reader)
            rows = [r for r in reader if r]
        ncol = len(header)
        if index_cols is None:
            index_cols = []
        if value_cols is None:
            value_cols = [c for c in header if c not in index_cols]
        try:
            vidx = [header.index(c) for c in value_cols]
        except ValueError as e:
            raise DataError(f"value column not in header: {e}; header={header[:10]}...") from e
        iidx = [header.index(c) for c in index_cols]
        arr = np.empty((len(rows), len(value_cols)), dtype=np.float64)
        for r, row in enumerate(rows):
            for c, ci in enumerate(vidx):
                try:
                    arr[r, c] = float(row[ci])
                except (ValueError, IndexError) as e:
                    raise DataError(
                        f"non-numeric value at row {r} col {header[ci]!r}: {row[ci]!r}"
                    ) from e
        labels = None
        if index_cols:
            labels = []
            for row in rows:
                parts = [str(row[ci]) for ci in iidx]
                labels.append("_".join(parts))
        meta = {"source": str(path), "value_cols": list(value_cols), "index_cols": list(index_cols)}
        name = name or path.stem
        return cls(arr, labels=labels, meta=meta, name=name)

    @classmethod
    def concat(cls, sets: list["PointSet"], name="concat") -> "PointSet":
        assert sets, "need at least one PointSet"
        d = sets[0].dim
        for s in sets[1:]:
            assert s.dim == d, f"dim mismatch: {s.dim} != {d}"
        data = np.vstack([s.data for s in sets])
        labels = [lb for s in sets for lb in s.labels]
        meta = {"parts": [s.name for s in sets]}
        return cls(data, labels=labels, meta=meta, name=name)

    # ---- transforms -------------------------------------------------------
    def select_dims(self, dims, name=None) -> "PointSet":
        dims = list(dims)
        for k in dims:
            assert 0 <= k < self.dim, f"dim index {k} out of range [0,{self.dim})"
        data = self.data[:, dims]
        return PointSet(
            data,
            labels=list(self.labels),
            meta={**self.meta, "selected_dims": dims},
            name=name or f"{self.name}[dims={dims}]",
        )

    def select_rows(self, idx, name=None) -> "PointSet":
        idx = list(idx)
        data = self.data[idx]
        labels = [self.labels[i] for i in idx]
        return PointSet(data, labels=labels, meta=dict(self.meta), name=name or self.name)

    def normalize(self, method: str = "none") -> "PointSet":
        x = self.data
        if method == "none":
            y = x
        elif method == "unit":
            norms = np.linalg.norm(x, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            y = x / norms
        elif method == "standard":
            mu = x.mean(axis=0)
            sd = x.std(axis=0)
            sd[sd == 0] = 1.0
            y = (x - mu) / sd
        else:
            raise DataError(f"unknown normalize method {method!r}")
        return PointSet(y, labels=list(self.labels), meta={**self.meta, "normalize": method},
                        name=f"{self.name}:{method}")

    # ---- io / stats -------------------------------------------------------
    def to_csv(self, path, header_prefix="dim_") -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        d = self.dim
        header = [f"{header_prefix}{i:04d}" for i in range(d)]
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh, lineterminator="\n")
            w.writerow(header)
            for row in self.data:
                w.writerow([f"{v:.9g}" for v in row])
        return path

    def stats(self) -> dict:
        x = self.data
        norms = np.linalg.norm(x, axis=1)
        return {
            "n": self.n,
            "dim": self.dim,
            "mean_abs": float(np.abs(x).mean()),
            "norm_min": float(norms.min()),
            "norm_mean": float(norms.mean()),
            "norm_max": float(norms.max()),
            "col_std_min": float(x.std(axis=0).min()),
            "col_std_max": float(x.std(axis=0).max()),
        }


def verify_pointset(ps: PointSet) -> None:
    assert ps.data.ndim == 2
    assert ps.n == len(ps.labels)
    assert np.all(np.isfinite(ps.data))
    debug.assert_debug(ps.dim >= 1, "PointSet has zero dims")
