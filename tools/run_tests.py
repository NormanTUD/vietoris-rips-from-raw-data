# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.26", "pytest>=8", "beartype>=0.18", "rich>=13"]
# ///
import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# IMPORTANT (DO NOT REMOVE "rich" from the PEP-723 deps above): the tools under
# tools/ (interactive.py, make_torus.py, make_sphere.py, make_wedge.py, analyze.py,
# barcodes.py, betti_sweep.py, plot.py, attractors.py) import `rich` at module load.
# Tests load those tools via importlib, so the TEST environment must provide `rich`
# too -- otherwise collection fails with `ModuleNotFoundError: No module named
# 'rich'` (this broke CI). When you add a new top-level import to a tool, add its
# package here as well.
import pytest

from vrtda.beartype_guard import beartype_module


def main() -> int:
    args = sys.argv[1:] or ["-q"]
    return pytest.main([str(Path(ROOT) / "tests"), "-q", *args])


beartype_module(__name__)


if __name__ == "__main__":
    raise SystemExit(main())
