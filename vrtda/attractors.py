from __future__ import annotations

import numpy as np

from vrtda.persistence import Barcode


def _capped_length(iv, eps_max: float) -> float:
    death = iv.death if np.isfinite(iv.death) else eps_max
    return max(0.0, death - iv.birth)


def essential_intervals(bc: Barcode, dim: int | None = None) -> list:
    ivs = bc.of_dim(dim) if dim is not None else bc.intervals
    return [iv for iv in ivs if iv.is_essential]


def long_lived_intervals(bc: Barcode, min_length: float, dim: int | None = None, eps_max: float | None = None) -> list:
    if eps_max is None:
        eps_max = float(bc.values.max()) if bc.values is not None and len(bc.values) else 0.0
    ivs = bc.of_dim(dim) if dim is not None else bc.intervals
    return [iv for iv in ivs if _capped_length(iv, eps_max) >= min_length]


def total_persistence(bc: Barcode, eps_max: float, dim: int | None = None) -> float:
    ivs = bc.of_dim(dim) if dim is not None else bc.intervals
    return float(sum(_capped_length(iv, eps_max) for iv in ivs))


def max_persistence(bc: Barcode, eps_max: float, dim: int | None = None) -> float:
    ivs = bc.of_dim(dim) if dim is not None else bc.intervals
    return float(max((_capped_length(iv, eps_max) for iv in ivs), default=0.0))


def per_dim_summary(bc: Barcode, eps_max: float, min_fraction: float = 0.0) -> dict:
    """Per-dimension attractor metrics.

    - n:            total intervals in dim
    - essential:    intervals that never die (infinite)
    - long_lived:   intervals with (capped) length >= min_fraction * eps_max
    - total_persistence: sum of (capped) lengths  (persistence-landscape proxy)
    - max_length:   longest single interval
    """
    min_length = min_fraction * eps_max
    md = bc.max_dim()
    out = {}
    for d in range(md + 1):
        ivs = bc.of_dim(d)
        out[d] = {
            "n": len(ivs),
            "essential": len(essential_intervals(bc, d)),
            "long_lived": len(long_lived_intervals(bc, min_length, d, eps_max)),
            "total_persistence": total_persistence(bc, eps_max, d),
            "max_length": max_persistence(bc, eps_max, d),
        }
    return out


def compare(clouds: dict, eps_max: float, min_fraction: float = 0.0) -> list[dict]:
    """clouds: dict name -> Barcode. Returns a list of rows (one per name) with
    per-dimension attractor metrics flattened, for tabular comparison."""
    rows = []
    for name, bc in clouds.items():
        row = {"name": name}
        for d, v in per_dim_summary(bc, eps_max, min_fraction).items():
            row[f"b{d}_essential"] = v["essential"]
            row[f"b{d}_long_lived"] = v["long_lived"]
            row[f"b{d}_total_persistence"] = v["total_persistence"]
        rows.append(row)
    return rows
