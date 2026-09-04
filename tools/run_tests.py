# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.26", "pytest>=8"]
# ///
import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pytest


def main() -> int:
    args = sys.argv[1:] or ["-q"]
    return pytest.main([str(Path(ROOT) / "tests"), "-q", *args])


if __name__ == "__main__":
    raise SystemExit(main())
