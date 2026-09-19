"""Backend interface for the Qubit VM.

The QVM is an SPMD machine: one program, ``n`` independent lanes.  Every
register holds a *vector* of ``n`` values -- a scalar register one vector, a
pseudo-qubit register the pair ``(r, eta)`` -- and a backend is simply an
implementation of the APQB kernels over those vectors.

That is what makes the same program portable across the target column of the
architecture: :class:`~qubitbridge.backends.portable.PortableBackend` walks the
lanes in Python, :class:`~qubitbridge.backends.numpy_backend.NumpyBackend`
hands them to NumPy, and
:class:`~qubitbridge.backends.arm64.Arm64Backend` emits AArch64 NEON kernels
that chew two lanes per vector instruction.

A backend must be *numerically* interchangeable, not merely structurally: the
test-suite holds every backend to the scalar reference in
:mod:`qubitbridge.apqb` within a tight tolerance.
"""

from __future__ import annotations

from typing import Any, Sequence

__all__ = ["Backend", "QState", "VectorBackend"]

#: A pseudo-qubit register: the ``(r, eta)`` pair of lane-vectors.
QState = tuple[Any, Any]


class Backend:
    """Abstract execution backend.

    Subclasses implement the vector kernels below.  ``vec`` is whatever vector
    type the backend prefers (a Python list, an ``ndarray``, ...); the VM only
    moves them between registers and never inspects them.
    """

    #: Short identifier, e.g. ``"portable"``.
    name = "abstract"
    #: True if the backend can execute programs; False for pure code emitters.
    executes = True

    # -- lane plumbing ---------------------------------------------------
    def splat(self, value: float, n: int):
        """Broadcast a scalar to an ``n``-lane vector."""
        raise NotImplementedError

    def vector(self, values: Sequence[float]):
        """Build a lane vector from a sequence of floats."""
        raise NotImplementedError

    def to_list(self, v) -> list[float]:
        """Read a lane vector back out as a list of floats."""
        raise NotImplementedError

    # -- classical kernels -----------------------------------------------
    def add(self, x, y): raise NotImplementedError
    def sub(self, x, y): raise NotImplementedError
    def mul(self, x, y): raise NotImplementedError
    def div(self, x, y): raise NotImplementedError
    def neg(self, x): raise NotImplementedError
    def minimum(self, x, y): raise NotImplementedError
    def maximum(self, x, y): raise NotImplementedError
    def tanh(self, x): raise NotImplementedError
    def atanh(self, x): raise NotImplementedError
    def scale(self, x, c: float): raise NotImplementedError
    def offset(self, x, c: float): raise NotImplementedError

    # -- APQB kernels ----------------------------------------------------
    def qload(self, r: float, n: int) -> QState: raise NotImplementedError
    def qenc(self, x, mode: str) -> QState: raise NotImplementedError
    def qdec(self, s: QState, mode: str): raise NotImplementedError
    def qrot(self, s: QState, phi) -> QState: raise NotImplementedError
    def qint(self, a: QState, b: QState) -> QState: raise NotImplementedError
    def qmul(self, a: QState, b: QState) -> QState: raise NotImplementedError
    def qpow(self, s: QState, k: int) -> QState: raise NotImplementedError
    def qcorr(self, a: QState, b: QState): raise NotImplementedError
    def qunc(self, s: QState): raise NotImplementedError
    def qimag(self, s: QState): raise NotImplementedError
    def qent(self, s: QState): raise NotImplementedError
    def qgate(self, target: QState, source: QState, j) -> QState:
        raise NotImplementedError
    def qmeasure(self, s: QState, mode: str, uniforms): raise NotImplementedError
    def qnorm(self, s: QState) -> QState: raise NotImplementedError


class VectorBackend(Backend):
    """Marker base for backends whose vectors are contiguous lane arrays."""

    #: Lanes processed per hardware vector instruction (1 = scalar).
    lanes_per_vector = 1
