"""APQB IR -> Qubit ISA.

The lowering pass is a small straight-line code generator:

* **Liveness.** Each SSA value's last use is computed in one backward sweep.
* **Linear-scan allocation.** Registers are handed out from two free pools
  (``Q`` and ``R``) and returned the instant a value dies.  Because every QVM
  instruction reads its sources before writing its destination, a dying operand's
  register can be reused for the result of the very op that kills it.
* **Immediate folding.** An ``arith.const`` whose only uses are the angle of an
  ``apqb.rotate`` or the coupling of an ``apqb.gate`` never reaches a register:
  it is folded into the constant-immediate form (``QROT`` / ``QGATE``) instead of
  the register form (``QROTR`` / ``QGATER``).

The result is a :class:`~qubitbridge.isa.Program` plus the register assignment
of the function's arguments and results, which is the ABI the runner uses.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .ir import STATE, Func, IRError, Module, Op, Value
from .isa import (ENCODE_MODES, MEASURE_MODES, NUM_QREGS, NUM_RREGS, Op as I,
                  Program)

__all__ = ["Lowered", "LoweringError", "lower_func", "lower_module"]


class LoweringError(RuntimeError):
    """Raised when a function cannot be lowered (e.g. it runs out of registers)."""


@dataclass
class Lowered:
    """A lowered function: the module plus its register ABI."""

    name: str
    program: Program
    #: Per IR argument, the register it is passed in as ``("q"|"r", index)``.
    arg_regs: list[tuple[str, int]] = field(default_factory=list)
    #: Per IR result, the register it is produced in.
    result_regs: list[tuple[str, int]] = field(default_factory=list)

    def result_index(self, i: int = 0) -> int:
        return self.result_regs[i][1]


class _Allocator:
    """Two free pools, lowest register first so output stays readable."""

    def __init__(self):
        self.free = {"q": list(range(NUM_QREGS)), "r": list(range(NUM_RREGS))}
        self.assigned: dict[str, tuple[str, int]] = {}
        self.high_water = {"q": 0, "r": 0}

    def file_of(self, type_: str) -> str:
        return "q" if type_ == STATE else "r"

    def acquire(self, file: str) -> int:
        pool = self.free[file]
        if not pool:
            raise LoweringError(
                f"out of {file.upper()} registers: this kernel needs more than "
                f"{NUM_QREGS if file == 'q' else NUM_RREGS} live values; split it "
                f"into smaller functions")
        pool.sort()
        index = pool.pop(0)
        self.high_water[file] = max(self.high_water[file], index + 1)
        return index

    def bind(self, value: Value) -> tuple[str, int]:
        file = self.file_of(value.type)
        slot = (file, self.acquire(file))
        self.assigned[value.name] = slot
        return slot

    def release(self, value: Value) -> None:
        slot = self.assigned.pop(value.name, None)
        if slot is not None:
            self.free[slot[0]].append(slot[1])

    def temp(self, file: str):
        index = self.acquire(file)
        return index

    def give_back(self, file: str, index: int) -> None:
        self.free[file].append(index)

    def of(self, value: Value) -> tuple[str, int]:
        try:
            return self.assigned[value.name]
        except KeyError:
            raise LoweringError(f"%{value.name} has no register; it was never "
                                f"defined or already died") from None


_ARITH = {
    "arith.add": I.ADD,
    "arith.sub": I.SUB,
    "arith.mul": I.MUL,
    "arith.div": I.DIV,
    "arith.min": I.MIN,
    "arith.max": I.MAX,
}
_UNARY = {
    "arith.neg": I.NEG,
    "arith.tanh": I.TANH,
    "arith.atanh": I.ATANH,
}


def _const_value(op: Op) -> float:
    return float(op.attrs["value"])


def _mode_index(op: Op, table: tuple[str, ...], attr: str = "mode") -> int:
    mode = op.attrs[attr]
    if mode not in table:
        raise IRError(f"{op.name}: unknown {attr} {mode!r}, expected one of "
                      f"{', '.join(table)}")
    return table.index(mode)


def _int_attr(op: Op, attr: str) -> int:
    value = op.attrs[attr]
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise IRError(f"{op.name}: attribute {attr} must be a non-negative "
                      f"integer, got {value!r}")
    return value


def lower_func(func: Func, module_name: str | None = None) -> Lowered:
    """Lower one IR function to a runnable :class:`~qubitbridge.isa.Program`."""
    prog = Program(name=module_name or func.name)
    alloc = _Allocator()

    # -- liveness --------------------------------------------------------
    last_use: dict[str, int] = {}
    for i, op in enumerate(func.body):
        for operand in op.operands:
            last_use[operand.name] = i
    forever = len(func.body) + 1
    for value in func.results:
        last_use[value.name] = forever

    # -- which constants can stay immediates ------------------------------
    defs: dict[str, Op] = {op.result.name: op for op in func.body
                           if op.result is not None}
    folded: set[str] = set()
    for name, op in defs.items():
        if op.name != "arith.const":
            continue
        uses = [(other, pos) for other in func.body
                for pos, operand in enumerate(other.operands)
                if operand.name == name]
        if not uses or last_use.get(name) == forever:
            continue
        if all((other.name == "apqb.rotate" and pos == 1)
               or (other.name == "apqb.gate" and pos == 2)
               for other, pos in uses):
            folded.add(name)

    # -- arguments --------------------------------------------------------
    lowered = Lowered(name=func.name, program=prog)
    for arg in func.args:
        lowered.arg_regs.append(alloc.bind(arg))

    def _kill_dead(index: int, op: Op) -> None:
        for operand in op.operands:
            if operand.name in folded:
                continue
            if last_use.get(operand.name, -1) <= index:
                alloc.release(operand)

    # -- body -------------------------------------------------------------
    for i, op in enumerate(func.body):
        name = op.name
        if name == "arith.const" and op.result.name in folded:
            continue

        operand_regs = [alloc.of(o) for o in op.operands
                        if o.name not in folded]
        _kill_dead(i, op)
        dst_file, dst = alloc.bind(op.result)

        if name == "arith.const":
            prog.emit(I.LDI, dst=dst, imm=prog.const(_const_value(op)))
        elif name in _ARITH:
            prog.emit(_ARITH[name], dst=dst, a=operand_regs[0][1],
                      b=operand_regs[1][1])
        elif name in _UNARY:
            prog.emit(_UNARY[name], dst=dst, a=operand_regs[0][1])
        elif name == "apqb.encode":
            prog.emit(I.QENC, dst=dst, a=operand_regs[0][1],
                      imm=_mode_index(op, ENCODE_MODES))
        elif name == "apqb.decode":
            prog.emit(I.QDEC, dst=dst, a=operand_regs[0][1],
                      imm=_mode_index(op, ENCODE_MODES))
        elif name == "apqb.measure":
            prog.emit(I.QMEASURE, dst=dst, a=operand_regs[0][1],
                      imm=_mode_index(op, MEASURE_MODES))
        elif name == "apqb.rotate":
            phi = op.operands[1]
            if phi.name in folded:
                prog.emit(I.QROT, dst=dst, a=operand_regs[0][1],
                          imm=prog.const(_const_value(defs[phi.name])))
            else:
                prog.emit(I.QROTR, dst=dst, a=operand_regs[0][1],
                          b=operand_regs[1][1])
        elif name == "apqb.interact":
            prog.emit(I.QINT, dst=dst, a=operand_regs[0][1], b=operand_regs[1][1])
        elif name == "apqb.mul":
            prog.emit(I.QMUL, dst=dst, a=operand_regs[0][1], b=operand_regs[1][1])
        elif name == "apqb.pow":
            prog.emit(I.QPOW, dst=dst, a=operand_regs[0][1], imm=_int_attr(op, "k"))
        elif name == "apqb.correlate":
            prog.emit(I.QCORR, dst=dst, a=operand_regs[0][1], b=operand_regs[1][1])
        elif name == "apqb.uncertainty":
            prog.emit(I.QUNC, dst=dst, a=operand_regs[0][1])
        elif name == "apqb.imag":
            prog.emit(I.QIMAG, dst=dst, a=operand_regs[0][1])
        elif name == "apqb.entropy":
            prog.emit(I.QENT, dst=dst, a=operand_regs[0][1])
        elif name == "apqb.normalize":
            prog.emit(I.QNORM, dst=dst, a=operand_regs[0][1])
        elif name == "apqb.gate":
            j = op.operands[2]
            if j.name in folded:
                prog.emit(I.QGATE, dst=dst, a=operand_regs[0][1],
                          b=operand_regs[1][1],
                          imm=prog.const(_const_value(defs[j.name])))
            else:
                prog.emit(I.QGATER, dst=dst, a=operand_regs[0][1],
                          b=operand_regs[1][1], imm=operand_regs[2][1])
        elif name in ("apqb.cheb_t", "apqb.cheb_u"):
            k = _int_attr(op, "k")
            tmp = alloc.temp("q")
            prog.emit(I.QPOW, dst=tmp, a=operand_regs[0][1], imm=k)
            if name == "apqb.cheb_t":
                prog.emit(I.QDEC, dst=dst, a=tmp,
                          imm=ENCODE_MODES.index("linear"))
            else:
                prog.emit(I.QIMAG, dst=dst, a=tmp)
            alloc.give_back("q", tmp)
        else:  # pragma: no cover - verify() rejects unknown ops first
            raise LoweringError(f"no lowering for {name}")

    prog.emit(I.HALT)
    prog.validate()
    lowered.result_regs = [alloc.of(v) for v in func.results]
    return lowered


def lower_module(module: Module) -> dict[str, Lowered]:
    """Lower every function in a verified module."""
    module.verify()
    return {fn.name: lower_func(fn, module_name=f"{module.name}.{fn.name}")
            for fn in module.funcs}
