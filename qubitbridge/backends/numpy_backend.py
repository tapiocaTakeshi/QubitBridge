"""Vectorised backend built on NumPy.

Identical semantics to :class:`~qubitbridge.backends.portable.PortableBackend`,
but each kernel is one array expression over all lanes, which is what makes
batched APQB work (a QBNN layer, a bank of correlations) worth handing to the
QVM at all.  NumPy is an optional dependency: importing this module without it
raises, and the VM falls back to the portable backend.
"""

from __future__ import annotations

from typing import Sequence

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - exercised only without numpy
    raise ImportError(
        "the numpy backend requires numpy; install it or use backend='portable'"
    ) from exc

from ..apqb import R_MAX
from .base import QState, VectorBackend

__all__ = ["NumpyBackend"]

_DTYPE = np.float64


def _sech(a):
    """sech(a) without overflowing cosh(a)."""
    m = np.abs(a)
    e = np.exp(-m)
    return 2.0 * e / (1.0 + e * e)


def _eta_from_r(r):
    return np.sqrt(np.maximum(0.0, 1.0 - r * r))


class NumpyBackend(VectorBackend):
    """Lane vectors are 1-D ``float64`` arrays."""

    name = "numpy"
    lanes_per_vector = 1  # NumPy picks its own width internally

    # -- lane plumbing ---------------------------------------------------
    def splat(self, value: float, n: int):
        return np.full(n, float(value), dtype=_DTYPE)

    def vector(self, values: Sequence[float]):
        return np.asarray(values, dtype=_DTYPE)

    def to_list(self, v) -> list[float]:
        return [float(x) for x in np.asarray(v, dtype=_DTYPE).ravel()]

    # -- classical kernels -----------------------------------------------
    def add(self, x, y): return x + y
    def sub(self, x, y): return x - y
    def mul(self, x, y): return x * y
    def div(self, x, y): return x / y
    def neg(self, x): return -x
    def minimum(self, x, y): return np.minimum(x, y)
    def maximum(self, x, y): return np.maximum(x, y)
    def tanh(self, x): return np.tanh(x)

    def atanh(self, x):
        return np.arctanh(np.clip(x, -R_MAX, R_MAX))

    def scale(self, x, c: float): return x * c
    def offset(self, x, c: float): return x + c

    # -- APQB kernels ----------------------------------------------------
    def qload(self, r: float, n: int) -> QState:
        r = float(np.clip(r, -1.0, 1.0))
        eta = float(np.sqrt(max(0.0, 1.0 - r * r)))
        return self.splat(r, n), self.splat(eta, n)

    def qenc(self, x, mode: str) -> QState:
        if mode == "latent":
            return np.tanh(x), _sech(x)
        if mode == "linear":
            r = np.clip(x, -1.0, 1.0)
            return r, _eta_from_r(r)
        if mode == "angle":
            two = 2.0 * x
            return np.cos(two), np.sin(two)
        if mode == "prob":
            r = np.clip(2.0 * x - 1.0, -1.0, 1.0)
            return r, _eta_from_r(r)
        raise ValueError(f"unknown APQB encoding {mode!r}")

    def qdec(self, s: QState, mode: str):
        r, eta = s
        if mode == "latent":
            return self.atanh(r)
        if mode == "linear":
            return r
        if mode == "angle":
            return 0.5 * np.arctan2(eta, r)
        if mode == "prob":
            return 0.5 * (1.0 + r)
        raise ValueError(f"unknown APQB encoding {mode!r}")

    def qrot(self, s: QState, phi) -> QState:
        r, eta = s
        two = 2.0 * phi
        c, sn = np.cos(two), np.sin(two)
        return r * c - eta * sn, r * sn + eta * c

    def qint(self, a: QState, b: QState) -> QState:
        ra, ea = a
        rb, eb = b
        return ra * rb - ea * eb, ra * eb + ea * rb

    def qmul(self, a: QState, b: QState) -> QState:
        r = np.clip(a[0] * b[0], -1.0, 1.0)
        return r, _eta_from_r(r)

    def qpow(self, s: QState, k: int) -> QState:
        if k < 0:
            raise ValueError("APQB power requires k >= 0")
        n = np.shape(s[0])[0]
        acc: QState = (np.ones(n, dtype=_DTYPE), np.zeros(n, dtype=_DTYPE))
        for _ in range(k):
            acc = self.qint(acc, s)
        return acc

    def qcorr(self, a: QState, b: QState):
        return a[0] * b[0] + a[1] * b[1]

    def qunc(self, s: QState):
        return np.abs(s[1])

    def qimag(self, s: QState):
        return s[1]

    def qent(self, s: QState):
        r = np.clip(s[0], -1.0, 1.0)
        p0 = 0.5 * (1.0 + r)
        p1 = 0.5 * (1.0 - r)
        out = np.zeros_like(r)
        for p in (p0, p1):
            nz = p > 0.0
            out[nz] -= p[nz] * np.log2(p[nz])
        return out

    def qgate(self, target: QState, source: QState, j) -> QState:
        a = self.atanh(target[0]) + j * source[0]
        r, eta = np.tanh(a), _sech(a)
        identity = j == 0.0
        return (np.where(identity, target[0], r),
                np.where(identity, target[1], eta))

    def qmeasure(self, s: QState, mode: str, uniforms):
        if mode == "expect":
            return s[0]
        if mode == "sample":
            p0 = 0.5 * (1.0 + np.clip(s[0], -1.0, 1.0))
            return np.where(np.asarray(uniforms, dtype=_DTYPE) < p0, 1.0, -1.0)
        raise ValueError(f"unknown measurement mode {mode!r}")

    def qnorm(self, s: QState) -> QState:
        r, eta = s
        n = np.hypot(r, eta)
        safe = n == 0.0
        n = np.where(safe, 1.0, n)
        return (np.where(safe, 1.0, r / n), np.where(safe, 0.0, eta / n))
