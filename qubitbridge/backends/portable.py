"""Pure-Python reference backend.

Every kernel computes the same formulas as :mod:`qubitbridge.apqb` -- the
free functions there (``sech``, ``clamp_unit``, ``clamp_atanh``) are reused
directly, so this stays numerically identical to the scalar spec -- but
works on raw ``(r, eta)`` float pairs rather than constructing an
:class:`~qubitbridge.apqb.APQBState` per lane. A frozen dataclass built once
per element, per instruction, is real overhead an interpreter run over many
lanes pays every time (see ``benchmarks/apqb_pattern_bench.py``, where it
was the difference between the APQB pattern costing roughly the expected
2x-3x of the classical one and costing 20x); avoiding it here keeps this
backend fast enough to be a reasonable default, not just a correctness
oracle. This backend needs nothing but the standard library, which is what
lets the whole toolchain run anywhere Python does.
"""

from __future__ import annotations

import math
from typing import Sequence

from ..apqb import clamp_atanh, clamp_unit, sech
from .base import QState, VectorBackend

__all__ = ["PortableBackend"]


def _eta_from_r(r: float) -> float:
    return math.sqrt(max(0.0, 1.0 - r * r))


class PortableBackend(VectorBackend):
    """Scalar backend; lane vectors are plain ``list[float]``."""

    name = "portable"
    lanes_per_vector = 1

    # -- lane plumbing ---------------------------------------------------
    def splat(self, value: float, n: int) -> list[float]:
        return [float(value)] * n

    def vector(self, values: Sequence[float]) -> list[float]:
        return [float(v) for v in values]

    def to_list(self, v) -> list[float]:
        return list(v)

    # -- classical kernels -----------------------------------------------
    def add(self, x, y): return [a + b for a, b in zip(x, y)]
    def sub(self, x, y): return [a - b for a, b in zip(x, y)]
    def mul(self, x, y): return [a * b for a, b in zip(x, y)]
    def div(self, x, y): return [a / b for a, b in zip(x, y)]
    def neg(self, x): return [-a for a in x]
    def minimum(self, x, y): return [min(a, b) for a, b in zip(x, y)]
    def maximum(self, x, y): return [max(a, b) for a, b in zip(x, y)]
    def tanh(self, x): return [math.tanh(a) for a in x]

    def atanh(self, x):
        return [math.atanh(clamp_atanh(a)) for a in x]

    def scale(self, x, c: float): return [a * c for a in x]
    def offset(self, x, c: float): return [a + c for a in x]

    # -- APQB kernels ------------------------------------------------------
    # Each works directly on the (r, eta) lane-list pair; every formula below
    # is the same one the corresponding function in apqb.py applies to a
    # single APQBState, just without building one.

    def qload(self, r: float, n: int) -> QState:
        r = clamp_unit(r)
        return self.splat(r, n), self.splat(_eta_from_r(r), n)

    def qenc(self, x, mode: str) -> QState:
        if mode == "latent":
            return [math.tanh(a) for a in x], [sech(a) for a in x]
        if mode == "linear":
            rs = [clamp_unit(v) for v in x]
            return rs, [_eta_from_r(r) for r in rs]
        if mode == "angle":
            twos = [2.0 * v for v in x]
            return [math.cos(t) for t in twos], [math.sin(t) for t in twos]
        if mode == "prob":
            rs = [clamp_unit(2.0 * v - 1.0) for v in x]
            return rs, [_eta_from_r(r) for r in rs]
        raise ValueError(f"unknown APQB encoding {mode!r}")

    def qdec(self, s: QState, mode: str):
        r, eta = s
        if mode == "latent":
            return [math.atanh(clamp_atanh(v)) for v in r]
        if mode == "linear":
            return list(r)
        if mode == "angle":
            return [0.5 * math.atan2(e, v) for v, e in zip(r, eta)]
        if mode == "prob":
            return [0.5 * (1.0 + v) for v in r]
        raise ValueError(f"unknown APQB encoding {mode!r}")

    def qrot(self, s: QState, phi) -> QState:
        r, eta = s
        out_r, out_eta = [], []
        for rv, ev, p in zip(r, eta, phi):
            two = 2.0 * p
            c, sn = math.cos(two), math.sin(two)
            out_r.append(rv * c - ev * sn)
            out_eta.append(rv * sn + ev * c)
        return out_r, out_eta

    def qint(self, a: QState, b: QState) -> QState:
        ra, ea = a
        rb, eb = b
        out_r, out_eta = [], []
        for r1, e1, r2, e2 in zip(ra, ea, rb, eb):
            out_r.append(r1 * r2 - e1 * e2)
            out_eta.append(r1 * e2 + e1 * r2)
        return out_r, out_eta

    def qmul(self, a: QState, b: QState) -> QState:
        ra, _ = a
        rb, _ = b
        rs = [clamp_unit(r1 * r2) for r1, r2 in zip(ra, rb)]
        return rs, [_eta_from_r(r) for r in rs]

    def qpow(self, s: QState, k: int) -> QState:
        if k < 0:
            raise ValueError("APQB power requires k >= 0")
        n = len(s[0])
        acc: QState = ([1.0] * n, [0.0] * n)
        for _ in range(k):
            acc = self.qint(acc, s)
        return acc

    def qcorr(self, a: QState, b: QState):
        ra, ea = a
        rb, eb = b
        return [r1 * r2 + e1 * e2 for r1, e1, r2, e2 in zip(ra, ea, rb, eb)]

    def qunc(self, s: QState):
        return [abs(e) for e in s[1]]

    def qimag(self, s: QState):
        return list(s[1])

    def qent(self, s: QState):
        out = []
        for r in s[0]:
            rc = clamp_unit(r)
            p0, p1 = 0.5 * (1.0 + rc), 0.5 * (1.0 - rc)
            h = 0.0
            if p0 > 0.0:
                h -= p0 * math.log2(p0)
            if p1 > 0.0:
                h -= p1 * math.log2(p1)
            out.append(h)
        return out

    def qgate(self, target: QState, source: QState, j) -> QState:
        tr, te = target
        sr, _ = source
        out_r, out_eta = [], []
        for t, e, s_, jj in zip(tr, te, sr, j):
            if jj == 0.0:
                out_r.append(t)
                out_eta.append(e)
            else:
                a = math.atanh(clamp_atanh(t)) + jj * s_
                out_r.append(math.tanh(a))
                out_eta.append(sech(a))
        return out_r, out_eta

    def qmeasure(self, s: QState, mode: str, uniforms):
        if mode == "expect":
            return list(s[0])
        if mode == "sample":
            return [1.0 if u < 0.5 * (1.0 + clamp_unit(r)) else -1.0
                   for r, u in zip(s[0], uniforms)]
        raise ValueError(f"unknown measurement mode {mode!r}")

    def qnorm(self, s: QState) -> QState:
        out_r, out_eta = [], []
        for r, e in zip(s[0], s[1]):
            n = math.hypot(r, e)
            if n == 0.0:
                out_r.append(1.0)
                out_eta.append(0.0)
            else:
                out_r.append(r / n)
                out_eta.append(e / n)
        return out_r, out_eta
