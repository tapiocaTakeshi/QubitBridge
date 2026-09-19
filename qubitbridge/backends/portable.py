"""Pure-Python reference backend.

Every kernel is a lane-wise application of the scalar functions in
:mod:`qubitbridge.apqb`, so this backend *is* the specification: the other
backends are checked against it.  It needs nothing but the standard library,
which is what lets the whole toolchain run anywhere Python does.
"""

from __future__ import annotations

import math
from typing import Sequence

from .. import apqb
from .base import QState, VectorBackend

__all__ = ["PortableBackend"]


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
        return [math.atanh(apqb.clamp_atanh(a)) for a in x]

    def scale(self, x, c: float): return [a * c for a in x]
    def offset(self, x, c: float): return [a + c for a in x]

    # -- APQB kernels ----------------------------------------------------
    def _states(self, s: QState):
        return [apqb.APQBState(r, e) for r, e in zip(s[0], s[1])]

    @staticmethod
    def _unzip(states) -> QState:
        return [s.r for s in states], [s.eta for s in states]

    def qload(self, r: float, n: int) -> QState:
        state = apqb.APQBState.from_r(r)
        return [state.r] * n, [state.eta] * n

    def qenc(self, x, mode: str) -> QState:
        return self._unzip([apqb.encode(v, mode) for v in x])

    def qdec(self, s: QState, mode: str):
        return [apqb.decode(st, mode) for st in self._states(s)]

    def qrot(self, s: QState, phi) -> QState:
        return self._unzip([apqb.rotate(st, p)
                            for st, p in zip(self._states(s), phi)])

    def qint(self, a: QState, b: QState) -> QState:
        return self._unzip([apqb.interact(x, y)
                            for x, y in zip(self._states(a), self._states(b))])

    def qmul(self, a: QState, b: QState) -> QState:
        return self._unzip([apqb.state_mul(x, y)
                            for x, y in zip(self._states(a), self._states(b))])

    def qpow(self, s: QState, k: int) -> QState:
        return self._unzip([apqb.power(st, k) for st in self._states(s)])

    def qcorr(self, a: QState, b: QState):
        return [apqb.correlate(x, y)
                for x, y in zip(self._states(a), self._states(b))]

    def qunc(self, s: QState):
        return [apqb.uncertainty(st) for st in self._states(s)]

    def qimag(self, s: QState):
        return list(s[1])

    def qent(self, s: QState):
        return [apqb.entropy_z(st) for st in self._states(s)]

    def qgate(self, target: QState, source: QState, j) -> QState:
        return self._unzip([apqb.gate(t, s, jj) for t, s, jj
                            in zip(self._states(target), self._states(source), j)])

    def qmeasure(self, s: QState, mode: str, uniforms):
        states = self._states(s)
        if mode == "expect":
            return [apqb.measure_expect(st) for st in states]
        if mode == "sample":
            return [apqb.measure_sample(st, u) for st, u in zip(states, uniforms)]
        raise ValueError(f"unknown measurement mode {mode!r}")

    def qnorm(self, s: QState) -> QState:
        return self._unzip([st.normalized() for st in self._states(s)])
