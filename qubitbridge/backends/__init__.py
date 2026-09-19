"""Execution and code-generation backends for the Qubit VM."""

from __future__ import annotations

from .base import Backend, QState, VectorBackend
from .portable import PortableBackend

__all__ = [
    "Backend",
    "QState",
    "VectorBackend",
    "PortableBackend",
    "get_backend",
    "available_backends",
]


def available_backends() -> list[str]:
    """Names of the execution backends usable in this interpreter."""
    names = ["portable"]
    try:  # pragma: no cover - depends on the environment
        import importlib.util
        if importlib.util.find_spec("numpy") is not None:
            names.append("numpy")
    except (ImportError, ValueError):
        pass
    return names


def get_backend(name: str = "portable") -> Backend:
    """Instantiate an execution backend by name.

    ``"auto"`` prefers NumPy when it is importable and falls back to the
    portable backend otherwise.
    """
    if name == "auto":
        name = "numpy" if "numpy" in available_backends() else "portable"
    if name == "portable":
        return PortableBackend()
    if name == "numpy":
        from .numpy_backend import NumpyBackend
        return NumpyBackend()
    raise ValueError(f"unknown backend {name!r}; available: "
                     f"{', '.join(available_backends())}")
