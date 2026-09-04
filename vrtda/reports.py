from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np


class Report:
    """A tiny dependency-free structured report builder (markdown or plain text)."""

    def __init__(self, title: str = "") -> None:
        self.title = title
        self.sections: list[tuple[str, list[str]]] = []

    def section(self, heading: str, lines: list[str]) -> "Report":
        self.sections.append((heading, list(lines)))
        return self

    def table(self, heading: str, headers: list[str], rows: list[list[object]]) -> "Report":
        def cell(v: object) -> str:
            if isinstance(v, float):
                return f"{v:.4g}"
            return str(v)

        body = ["| " + " | ".join(headers) + " |",
                "| " + " | ".join("---" for _ in headers) + " |"]
        for r in rows:
            body.append("| " + " | ".join(cell(v) for v in r) + " |")
        self.sections.append((heading, body))
        return self

    def _render(self, fmt: str) -> str:
        out = []
        if self.title:
            out.append((f"# {self.title}" if fmt == "md" else f"{self.title}") + "\n")
        for heading, body in self.sections:
            if fmt == "md":
                out.append(f"\n## {heading}\n")
            else:
                out.append(f"\n{heading}\n" + "-" * len(heading) + "\n")
            out.append("\n".join(body) + "\n")
        return "".join(out)

    def to_markdown(self) -> str:
        return self._render("md")

    def to_text(self) -> str:
        return self._render("txt")

    def write(self, path: str | Path, fmt: str = "md") -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_markdown() if fmt == "md" else self.to_text())
        return path


def betti_table(epsilons: Sequence[float] | np.ndarray, betti_arr: np.ndarray) -> tuple[list[str], list[list[int | float]]]:
    """Turn a betti_function array (n_eps x (md+1)) into table headers/rows."""
    betti_arr = np.asarray(betti_arr)
    md = betti_arr.shape[1] - 1
    headers = ["eps"] + [f"b{k}" for k in range(md + 1)]
    rows: list[list[int | float]] = []
    for i, e in enumerate(epsilons):
        rows.append([float(e)] + [int(x) for x in betti_arr[i]])
    return headers, rows


from vrtda.beartype_guard import beartype_module as _beartype_module

_beartype_module(__name__)
