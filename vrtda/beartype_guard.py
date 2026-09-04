"""Runtime type checking that is ALWAYS on, with no way to switch it off.

:func:`beartype_module` wraps every function defined in a module (module-level
functions and non-dunder class methods) with :func:`beartype.beartype`, so every
call is validated against its annotations at runtime. This is a hard contract:
type hints are enforced everywhere, without exception, and there is deliberately
no environment flag or argument to disable it.

Usage (bottom of each ``vrtda`` module)::

    from vrtda.beartype_guard import beartype_module
    beartype_module(__name__)

Because each module wraps itself right after its definitions, any other module
that imports one of its functions receives the already-wrapped (checked) version.
"""
from __future__ import annotations

import inspect
import sys
import types
from typing import Any

from beartype import beartype


def beartype_function(func: types.FunctionType) -> types.FunctionType:
    """Wrap a single function with beartype (idempotent via the marker below)."""
    if getattr(func, "__vrtda_beartyped__", False):
        return func
    wrapped = beartype(func)
    try:
        wrapped.__vrtda_beartyped__ = True
    except (AttributeError, TypeError):  # pragma: no cover - defensive
        pass
    return wrapped


def _beartype_property(prop: "property") -> "property":
    func = prop.fget
    if func is None:
        return prop
    return property(fget=beartype_function(func), fset=prop.fset, fdel=prop.fdel, doc=prop.__doc__)


def beartype_module(module: Any) -> None:
    """Wrap all functions, methods and properties of ``module`` with beartype.

    ``module`` may be a module object or its name (a string). Idempotent:
    already-wrapped functions (marked ``__vrtda_beartyped__``) are left alone.
    """
    if isinstance(module, str):
        module = sys.modules[module]
    for _name, obj in list(vars(module).items()):
        if isinstance(obj, types.FunctionType) and getattr(obj, "__module__", None) == module.__name__:
            setattr(module, _name, beartype_function(obj))
        elif inspect.isclass(obj) and getattr(obj, "__module__", None) == module.__name__:
            for mname, mobj in list(vars(obj).items()):
                if isinstance(mobj, property):
                    setattr(obj, mname, _beartype_property(mobj))
                elif isinstance(mobj, types.FunctionType) and not mname.startswith("__"):
                    setattr(obj, mname, beartype_function(mobj))


def is_wrapped(func: Any) -> bool:
    """True if ``func`` has been wrapped by :func:`beartype_function`."""
    return bool(getattr(func, "__vrtda_beartyped__", False))


if __name__ == "__main__":  # pragma: no cover - manual smoke
    sys.exit(0)
