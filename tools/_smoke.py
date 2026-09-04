# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy>=1.26",
#   "beartype>=0.18",
# ]
# ///
import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import vrtda
from vrtda.beartype_guard import beartype_module


def main() -> int:
    a = np.arange(6, dtype=float).reshape(3, 2)
    d = np.linalg.norm(a, axis=1)
    assert d.shape == (3,)
    assert abs(d[2] - np.sqrt(41.0)) < 1e-12
    print("OK", "vrtda", vrtda.__version__, "numpy", np.__version__)
    return 0


beartype_module(__name__)


if __name__ == "__main__":
    raise SystemExit(main())
