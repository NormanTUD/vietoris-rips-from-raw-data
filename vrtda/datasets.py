from __future__ import annotations

from pathlib import Path

import numpy as np

from vrtda import PointSet
from vrtda.errors import DataError

INDEX_COLS = ["prompt_idx", "token_pos"]
NORM_KINDS = ("norms", "cosines", "deltas")


def _data_root() -> Path:
    return Path(__file__).resolve().parents[1] / "capital_berlin_multilingual"


def _resolve(data_dir):
    return Path(data_dir) if data_dir else _data_root()


def layer_path(data_dir, layer) -> Path:
    return _resolve(data_dir) / "all_token_streams" / f"layer_{int(layer):03d}.csv"


def _header(path) -> list[str]:
    import csv
    with open(path, newline="") as fh:
        return next(csv.reader(fh))


def _dim_cols(path) -> list[str]:
    return [c for c in _header(path) if c.startswith("dim_")]


def list_layers(data_dir=None) -> list[int]:
    d = _resolve(data_dir) / "all_token_streams"
    if not d.is_dir():
        raise DataError(f"layer directory not found: {d}")
    return sorted(int(p.stem.split("_")[1]) for p in d.glob("layer_*.csv"))


def load_token_cloud(
    data_dir=None,
    layer=0,
    value_cols=None,
    index_cols=None,
    normalize=False,
    name=None,
) -> PointSet:
    """The 81 token hidden states (5120-dim) at a single layer."""
    p = layer_path(data_dir, layer)
    ps = PointSet.from_csv(
        p,
        value_cols=value_cols or _dim_cols(p),
        index_cols=list(index_cols) if index_cols else list(INDEX_COLS),
        name=name or f"layer_{int(layer):03d}",
    )
    if normalize:
        ps = ps.normalize("unit")
    return ps


def load_layer_points(
    data_dir=None,
    layers=None,
    value_cols=None,
    index_cols=None,
    normalize=False,
    name="layer_points",
) -> PointSet:
    """Stack token hidden states across layers: (tokens x layers) points, each 5120-dim.

    Labels are '<prompt>_<pos>_L<layer>' so every (token, layer) is unique."""
    layers = list(layers) if layers is not None else list_layers(data_dir)
    parts = []
    for L in layers:
        ps = load_token_cloud(
            data_dir=data_dir,
            layer=L,
            value_cols=value_cols,
            index_cols=index_cols,
            normalize=normalize,
            name=f"L{int(L):03d}",
        )
        ps.labels = [f"{lbl}_L{int(L):03d}" for lbl in ps.labels]
        parts.append(ps)
    out = PointSet.concat(parts, name=name)
    out.meta["layers"] = list(layers)
    return out


def load_residual_matrix(data_dir=None, kind="norms"):
    """Load a per-token, per-layer scalar field (norms/cosines/deltas).

    Returns (matrix [n_tokens, n_layers], labels) where labels are '<prompt>_<pos>'."""
    if kind not in NORM_KINDS:
        raise DataError(f"kind must be one of {NORM_KINDS}, got {kind!r}")
    p = _resolve(data_dir) / "residual_norms" / f"{kind}_all.csv"
    if not p.exists():
        raise DataError(f"residual file not found: {p}")
    import csv
    with open(p, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        rows = [r for r in reader if r]
    iidx = [header.index(c) for c in INDEX_COLS]
    exclude = set(iidx)
    if "token_text" in header:
        exclude.add(header.index("token_text"))
    vidx = [i for i in range(len(header)) if i not in exclude]
    mat = np.empty((len(rows), len(vidx)), dtype=np.float64)
    labels = []
    for r, row in enumerate(rows):
        labels.append(f"{row[iidx[0]]}_{row[iidx[1]]}")
        for c, ci in enumerate(vidx):
            mat[r, c] = float(row[ci])
    return mat, labels


def token_texts(data_dir=None, layer=0) -> list[str]:
    """token_text per row of a layer file (aligned with the token cloud)."""
    import csv
    p = layer_path(data_dir, layer)
    with open(p, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        ti = header.index("token_text")
        return [row[ti] for row in reader if row]
