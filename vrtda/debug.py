import os
import sys
import time
from contextlib import contextmanager

_RUNTIME = {"enabled": None}


def _env_enabled() -> bool:
    v = os.environ.get("VR_DEBUG", "").strip().lower()
    return v in ("1", "true", "yes", "on")


def enabled() -> bool:
    if _RUNTIME["enabled"] is not None:
        return _RUNTIME["enabled"]
    return _env_enabled()


def enable(on: bool | None = True) -> None:
    _RUNTIME["enabled"] = bool(on)


def disable() -> None:
    _RUNTIME["enabled"] = False


def is_debug() -> bool:
    return enabled()


def log(msg: str, *args) -> None:
    if enabled():
        if args:
            msg = msg % args
        print(f"[debug] {msg}", file=sys.stderr, flush=True)


def section(name: str) -> None:
    if enabled():
        bar = "=" * 8
        print(f"\n{bar} {name} {bar}", file=sys.stderr, flush=True)


def warn(msg: str, *args) -> None:
    if args:
        msg = msg % args
    print(f"[vrtda:warn] {msg}", file=sys.stderr, flush=True)


def dump(label: str, obj) -> None:
    if enabled():
        print(f"[debug] {label}: {obj!r}", file=sys.stderr, flush=True)


@contextmanager
def timing(label: str):
    if not enabled():
        yield
        return
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        print(f"[debug] timing {label}: {dt:.4f}s", file=sys.stderr, flush=True)


def assert_debug(cond: bool, msg: str) -> None:
    if enabled() and not cond:
        raise AssertionError(msg)
