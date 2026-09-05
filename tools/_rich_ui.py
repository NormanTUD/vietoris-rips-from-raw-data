"""Shared rich terminal UI for the command-line tools.

Every tool prints (1) a parameters table at start (auto-derived from the argparse
parser, so the "meaning" column is always the live help text), (2) a result table,
and (3) a status spinner while doing slow work. Kept dependency-light: just `rich`.
"""
from __future__ import annotations

import argparse
import time

from rich.console import Console
from rich.table import Table


def params_table(p: argparse.ArgumentParser, args: argparse.Namespace, console: Console) -> None:
    """Print one table row per CLI flag: flag, current value, and meaning (the argparse help)."""
    t = Table(title="Parameters", title_justify="left", show_header=True)
    t.add_column("Flag", style="cyan", no_wrap=True)
    t.add_column("Value", style="bold", no_wrap=True)
    t.add_column("Meaning", style="dim", no_wrap=False)
    for a in p._actions:
        if a.dest == "help" or not a.option_strings:
            continue
        t.add_row(" ".join(a.option_strings), str(getattr(args, a.dest, a.default)), a.help or "")
    console.print(t)


def fmt_betti(b: list[int]) -> str:
    """Render a Betti vector as a compact '[1, 2, 1]' string."""
    return "[" + ", ".join(str(int(x)) for x in b) + "]"


def result_table(title: str, rows: list[tuple[str, str]], console: Console) -> None:
    t = Table(title=title, title_justify="left")
    t.add_column("Item", style="cyan", no_wrap=True)
    t.add_column("Value", style="bold")
    for k, v in rows:
        t.add_row(k, v)
    console.print(t)


class timed:
    """Context manager that shows a rich status spinner while a slow block runs,
    then prints how long it took. Example:  with timed(console, "Building Rips…"): ..."""

    def __init__(self, console: Console, message: str) -> None:
        self.console = console
        self.message = message
        self._status = None
        self._t0 = 0.0

    def __enter__(self) -> "timed":
        self._status = self.console.status(self.message)
        self._status.start()
        self._t0 = time.time()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self._status.stop()
        dt = time.time() - self._t0
        if exc_type is None:
            self.console.print(f"[dim]{self.message} … {dt:.1f}s[/dim]")
