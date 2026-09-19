"""The Qubit Virtual Machine: a virtual QPU over a classical processor.

The QVM does not simulate physical qubits -- there is no 2^n state vector here.
It executes the Qubit ISA over a register file of APQB coordinates, which costs
O(#registers) per lane rather than O(2^n), and hands the arithmetic to whatever
backend the host offers (Python, NumPy, NEON, ...).

Execution model
---------------
* **SPMD.** One program, ``lanes`` independent data lanes.  Registers hold lane
  vectors; ``lanes=1`` is the ordinary scalar case.
* **Straight line.** v1 has no branches: a module is a kernel, fully unrolled by
  the compiler, ending in ``HALT``.  Lane-divergent control flow is future work.
* **Deterministic.** ``QMEASURE ... sample`` draws from a seeded PRNG owned by
  the VM, not by the backend, so a run reproduces bit-for-bit on every target.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from .backends import Backend, get_backend
from .isa import (ENCODE_MODES, MEASURE_MODES, NUM_QREGS, NUM_RREGS, Instr, Op,
                  Program)

__all__ = ["QVM", "RunResult", "QVMError"]


class QVMError(RuntimeError):
    """Raised when a program faults (bad address, step limit, ...)."""


@dataclass
class RunResult:
    """Everything observable after a run."""

    #: Scalar register file, ``{index: [lane values]}``.
    r: dict[int, list[float]] = field(default_factory=dict)
    #: Pseudo-qubit register file, ``{index: ([r lanes], [eta lanes])}``.
    q: dict[int, tuple[list[float], list[float]]] = field(default_factory=dict)
    #: Data memory, one lane vector per address.
    memory: list[list[float]] = field(default_factory=list)
    #: Instructions retired.
    steps: int = 0
    #: Executed instructions, when the VM ran with ``trace=True``.
    trace: list[tuple[int, Instr]] = field(default_factory=list)

    def scalar(self, index: int, lane: int = 0) -> float:
        """One scalar register's value in one lane."""
        return self.r[index][lane]

    def state(self, index: int, lane: int = 0):
        """One pseudo-qubit register as an :class:`~qubitbridge.apqb.APQBState`."""
        from .apqb import APQBState
        r, eta = self.q[index]
        return APQBState(r[lane], eta[lane])


class QVM:
    """The virtual quantum processor."""

    def __init__(self, backend: Backend | str = "portable", seed: int = 0,
                 step_limit: int = 1_000_000):
        self.backend: Backend = (get_backend(backend) if isinstance(backend, str)
                                 else backend)
        if not self.backend.executes:
            raise QVMError(f"backend {self.backend.name!r} emits code but cannot "
                           f"execute it; use its emit API instead")
        self.seed = seed
        self.step_limit = step_limit

    # -- helpers ---------------------------------------------------------
    def _uniforms(self, rng: random.Random, n: int) -> list[float]:
        return [rng.random() for _ in range(n)]

    def run(self, program: Program, *,
            r_inputs: Mapping[int, Sequence[float] | float] | None = None,
            memory: Sequence[Sequence[float] | float] | None = None,
            lanes: int = 1, seed: int | None = None,
            trace: bool = False) -> RunResult:
        """Execute ``program`` and return the final machine state.

        ``r_inputs`` seeds scalar registers, each value either one float
        (broadcast to every lane) or a per-lane sequence.  ``memory`` seeds the
        data segment the same way.
        """
        program.validate()
        if lanes < 1:
            raise ValueError("lanes must be >= 1")
        be = self.backend
        rng = random.Random(self.seed if seed is None else seed)

        def _to_vec(value) -> list:
            if isinstance(value, (int, float)):
                return be.splat(float(value), lanes)
            values = list(value)
            if len(values) != lanes:
                raise ValueError(f"expected {lanes} lane values, got {len(values)}")
            return be.vector(values)

        rfile = [be.splat(0.0, lanes) for _ in range(NUM_RREGS)]
        qfile = [be.qload(1.0, lanes) for _ in range(NUM_QREGS)]
        mem = [be.splat(0.0, lanes) for _ in range(program.mem_size)]

        for index, value in (r_inputs or {}).items():
            if not 0 <= index < NUM_RREGS:
                raise ValueError(f"no scalar register R{index}")
            rfile[index] = _to_vec(value)
        for addr, value in enumerate(memory or ()):
            if addr >= len(mem):
                raise ValueError(f"memory seed at [{addr}] exceeds .mem "
                                 f"{program.mem_size}")
            mem[addr] = _to_vec(value)

        def _addr(instr: Instr, offset: int = 0) -> int:
            addr = instr.imm + offset
            if not 0 <= addr < len(mem):
                raise QVMError(f"{instr.op.name}: address [{addr}] outside the "
                               f"{len(mem)}-word data segment")
            return addr

        result = RunResult()
        pc = 0
        steps = 0
        code = program.code
        consts = program.consts

        while True:
            if pc >= len(code):
                raise QVMError("ran past the end of the program without HALT")
            instr = code[pc]
            if trace:
                result.trace.append((pc, instr))
            steps += 1
            if steps > self.step_limit:
                raise QVMError(f"step limit of {self.step_limit} exceeded")

            op = instr.op
            d, a, b, imm = instr.dst, instr.a, instr.b, instr.imm

            if op is Op.HALT:
                break
            elif op is Op.NOP:
                pass
            elif op is Op.MOV:
                rfile[d] = rfile[a]
            elif op is Op.LDI:
                rfile[d] = be.splat(consts[imm], lanes)
            elif op is Op.LOAD:
                rfile[d] = mem[_addr(instr)]
            elif op is Op.STORE:
                mem[_addr(instr)] = rfile[a]
            elif op is Op.ADD:
                rfile[d] = be.add(rfile[a], rfile[b])
            elif op is Op.SUB:
                rfile[d] = be.sub(rfile[a], rfile[b])
            elif op is Op.MUL:
                rfile[d] = be.mul(rfile[a], rfile[b])
            elif op is Op.DIV:
                rfile[d] = be.div(rfile[a], rfile[b])
            elif op is Op.NEG:
                rfile[d] = be.neg(rfile[a])
            elif op is Op.TANH:
                rfile[d] = be.tanh(rfile[a])
            elif op is Op.ATANH:
                rfile[d] = be.atanh(rfile[a])
            elif op is Op.SCALE:
                rfile[d] = be.scale(rfile[a], consts[imm])
            elif op is Op.ADDI:
                rfile[d] = be.offset(rfile[a], consts[imm])

            elif op is Op.QLOAD:
                qfile[d] = be.qload(consts[imm], lanes)
            elif op is Op.QSTORE:
                lo = _addr(instr)
                hi = _addr(instr, 1)
                mem[lo], mem[hi] = qfile[a][0], qfile[a][1]
            elif op is Op.QMOV:
                qfile[d] = qfile[a]
            elif op is Op.QENC:
                qfile[d] = be.qenc(rfile[a], ENCODE_MODES[imm])
            elif op is Op.QDEC:
                rfile[d] = be.qdec(qfile[a], ENCODE_MODES[imm])
            elif op is Op.QROT:
                qfile[d] = be.qrot(qfile[a], be.splat(consts[imm], lanes))
            elif op is Op.QROTR:
                qfile[d] = be.qrot(qfile[a], rfile[b])
            elif op is Op.QINT:
                qfile[d] = be.qint(qfile[a], qfile[b])
            elif op is Op.QMUL:
                qfile[d] = be.qmul(qfile[a], qfile[b])
            elif op is Op.QPOW:
                qfile[d] = be.qpow(qfile[a], imm)
            elif op is Op.QCORR:
                rfile[d] = be.qcorr(qfile[a], qfile[b])
            elif op is Op.QUNC:
                rfile[d] = be.qunc(qfile[a])
            elif op is Op.QIMAG:
                rfile[d] = be.qimag(qfile[a])
            elif op is Op.QENT:
                rfile[d] = be.qent(qfile[a])
            elif op is Op.QGATE:
                qfile[d] = be.qgate(qfile[a], qfile[b],
                                    be.splat(consts[imm], lanes))
            elif op is Op.QGATER:
                qfile[d] = be.qgate(qfile[a], qfile[b], rfile[imm])
            elif op is Op.QMEASURE:
                mode = MEASURE_MODES[imm]
                draws = (be.vector(self._uniforms(rng, lanes))
                         if mode == "sample" else None)
                rfile[d] = be.qmeasure(qfile[a], mode, draws)
            elif op is Op.QNORM:
                qfile[d] = be.qnorm(qfile[a])
            else:  # pragma: no cover - the table above is exhaustive
                raise QVMError(f"unimplemented opcode {op.name}")

            pc += 1

        result.steps = steps
        result.r = {i: be.to_list(v) for i, v in enumerate(rfile)}
        result.q = {i: (be.to_list(s[0]), be.to_list(s[1]))
                    for i, s in enumerate(qfile)}
        result.memory = [be.to_list(v) for v in mem]
        return result
