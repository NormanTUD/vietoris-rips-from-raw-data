# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.26", "pytest>=8", "beartype>=0.18"]
# ///
import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pytest

from vrtda.beartype_guard import beartype_module


def main() -> int:
    args = sys.argv[1:] or ["-q"]
    return pytest.main([str(Path(ROOT) / "tests"), "-q", *args])


beartype_module(__name__)


if __name__ == "__main__":
    raise SystemExit(main())
