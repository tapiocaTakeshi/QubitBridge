"""The Qubit ISA: the instruction set the virtual QPU executes.

A classical CPU's basic verbs are ADD / MUL / MOV / CMP.  The Qubit processor's
are ENCODE / ROTATE / INTERACT / CORRELATE / UNCERTAINTY / MEASURE, operating on
a file of pseudo-qubit registers ``Q0..Q31`` -- each holding ``{r, eta, theta}``
-- next to an ordinary scalar file ``R0..R31``.

Every instruction is one fixed 8-byte word::

    +--------+--------+--------+--------+------------------+
    | opcode |  dst   |   a    |   b    |   imm (uint32)   |
    +--------+--------+--------+--------+------------------+
         1        1        1        1            4

``imm`` indexes the module's constant pool for float operands, or carries a
small integer/mode directly.  Which file ``dst``/``a``/``b`` name, and how
``imm`` is read, are fixed per opcode by the :class:`Spec` table below.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from enum import IntEnum

__all__ = [
    "Op",
    "Spec",
    "SPECS",
    "SPEC_BY_NAME",
    "Instr",
    "Program",
    "NUM_QREGS",
    "NUM_RREGS",
    "INSTR_SIZE",
    "MAGIC",
    "ENCODE_MODES",
    "MEASURE_MODES",
]

NUM_QREGS = 32
NUM_RREGS = 32
INSTR_SIZE = 8
MAGIC = b"QVM1"

#: Encoding-mode operands, matching :data:`qubitbridge.apqb.ENCODINGS`.
ENCODE_MODES = ("latent", "linear", "angle", "prob")
#: Measurement modes: deterministic expectation, or a seeded Bernoulli draw.
MEASURE_MODES = ("expect", "sample")


class Op(IntEnum):
    """Opcode numbers.  0x00-0x1f classical, 0x20+ quantum."""

    HALT = 0x00
    NOP = 0x01
    MOV = 0x02          # Rd <- Ra
    LDI = 0x03          # Rd <- const[imm]
    LOAD = 0x04         # Rd <- mem[imm]
    STORE = 0x05        # mem[imm] <- Ra
    ADD = 0x06          # Rd <- Ra + Rb
    SUB = 0x07          # Rd <- Ra - Rb
    MUL = 0x08          # Rd <- Ra * Rb
    DIV = 0x09          # Rd <- Ra / Rb
    NEG = 0x0A          # Rd <- -Ra
    TANH = 0x0B         # Rd <- tanh(Ra)
    ATANH = 0x0C        # Rd <- atanh(Ra)
    SCALE = 0x0D        # Rd <- Ra * const[imm]
    ADDI = 0x0E         # Rd <- Ra + const[imm]
    MIN = 0x0F          # Rd <- min(Ra, Rb)
    MAX = 0x10          # Rd <- max(Ra, Rb)

    QLOAD = 0x20        # Qd <- state(r = const[imm])
    QSTORE = 0x21       # mem[imm], mem[imm+1] <- r, eta of Qa
    QMOV = 0x22         # Qd <- Qa
    QENC = 0x23         # Qd <- encode(Ra, mode = imm)
    QDEC = 0x24         # Rd <- decode(Qa, mode = imm)
    QROT = 0x25         # Qd <- rotate(Qa, const[imm])
    QROTR = 0x26        # Qd <- rotate(Qa, Rb)
    QINT = 0x27         # Qd <- Qa (x) Qb          (z product / theta sum)
    QMUL = 0x28         # Qd <- Qa . Qb            (r product)
    QPOW = 0x29         # Qd <- Qa ^ imm           (z^k, Chebyshev degree k)
    QCORR = 0x2A        # Rd <- corr(Qa, Qb)
    QUNC = 0x2B         # Rd <- |eta(Qa)|
    QENT = 0x2C         # Rd <- H_Z(Qa)
    QGATE = 0x2D        # Qd <- gate(Qa, Qb, J = const[imm])
    QGATER = 0x2E       # Qd <- gate(Qa, Qb, J = R[imm])
    QMEASURE = 0x2F     # Rd <- measure(Qa, mode = imm)
    QNORM = 0x30        # Qd <- normalize(Qa)
    QIMAG = 0x31        # Rd <- eta(Qa), signed


# Operand kinds: which register file a slot names, or that it is unused.
Q = "q"
R = "r"
_ = None

# Immediate kinds.
IMM_NONE = "none"
IMM_CONST = "const"   # index into the float constant pool
IMM_INT = "int"       # small unsigned integer, used literally
IMM_ENC = "enc"       # index into ENCODE_MODES
IMM_MEAS = "meas"     # index into MEASURE_MODES
IMM_ADDR = "addr"     # data-memory address
IMM_RREG = "rreg"     # a scalar register index carried in the imm field


@dataclass(frozen=True)
class Spec:
    """Static signature of one opcode."""

    op: Op
    dst: str | None
    a: str | None
    b: str | None
    imm: str
    doc: str

    @property
    def name(self) -> str:
        return self.op.name


def _s(op, dst, a, b, imm, doc):
    return Spec(op, dst, a, b, imm, doc)


SPECS: dict[Op, Spec] = {s.op: s for s in (
    _s(Op.HALT, _, _, _, IMM_NONE, "stop the machine"),
    _s(Op.NOP, _, _, _, IMM_NONE, "do nothing"),
    _s(Op.MOV, R, R, _, IMM_NONE, "Rd <- Ra"),
    _s(Op.LDI, R, _, _, IMM_CONST, "Rd <- immediate constant"),
    _s(Op.LOAD, R, _, _, IMM_ADDR, "Rd <- mem[addr]"),
    _s(Op.STORE, _, R, _, IMM_ADDR, "mem[addr] <- Ra"),
    _s(Op.ADD, R, R, R, IMM_NONE, "Rd <- Ra + Rb"),
    _s(Op.SUB, R, R, R, IMM_NONE, "Rd <- Ra - Rb"),
    _s(Op.MUL, R, R, R, IMM_NONE, "Rd <- Ra * Rb"),
    _s(Op.DIV, R, R, R, IMM_NONE, "Rd <- Ra / Rb"),
    _s(Op.NEG, R, R, _, IMM_NONE, "Rd <- -Ra"),
    _s(Op.TANH, R, R, _, IMM_NONE, "Rd <- tanh(Ra)"),
    _s(Op.ATANH, R, R, _, IMM_NONE, "Rd <- atanh(Ra)"),
    _s(Op.SCALE, R, R, _, IMM_CONST, "Rd <- Ra * c"),
    _s(Op.ADDI, R, R, _, IMM_CONST, "Rd <- Ra + c"),
    _s(Op.MIN, R, R, R, IMM_NONE, "Rd <- min(Ra, Rb)"),
    _s(Op.MAX, R, R, R, IMM_NONE, "Rd <- max(Ra, Rb)"),

    _s(Op.QLOAD, Q, _, _, IMM_CONST, "Qd <- canonical state with r = c"),
    _s(Op.QSTORE, _, Q, _, IMM_ADDR, "mem[addr], mem[addr+1] <- r, eta"),
    _s(Op.QMOV, Q, Q, _, IMM_NONE, "Qd <- Qa"),
    _s(Op.QENC, Q, R, _, IMM_ENC, "Qd <- encode(Ra)"),
    _s(Op.QDEC, R, Q, _, IMM_ENC, "Rd <- decode(Qa)"),
    _s(Op.QROT, Q, Q, _, IMM_CONST, "Qd <- rotate(Qa, phi)"),
    _s(Op.QROTR, Q, Q, R, IMM_NONE, "Qd <- rotate(Qa, Rb)"),
    _s(Op.QINT, Q, Q, Q, IMM_NONE, "Qd <- Qa (x) Qb"),
    _s(Op.QMUL, Q, Q, Q, IMM_NONE, "Qd <- Qa . Qb"),
    _s(Op.QPOW, Q, Q, _, IMM_INT, "Qd <- Qa ^ k"),
    _s(Op.QCORR, R, Q, Q, IMM_NONE, "Rd <- corr(Qa, Qb)"),
    _s(Op.QUNC, R, Q, _, IMM_NONE, "Rd <- |eta(Qa)|"),
    _s(Op.QENT, R, Q, _, IMM_NONE, "Rd <- H_Z(Qa)"),
    _s(Op.QGATE, Q, Q, Q, IMM_CONST, "Qd <- gate(Qa, Qb, J = c)"),
    _s(Op.QGATER, Q, Q, Q, IMM_RREG, "Qd <- gate(Qa, Qb, J = R[imm])"),
    _s(Op.QMEASURE, R, Q, _, IMM_MEAS, "Rd <- measure(Qa)"),
    _s(Op.QNORM, Q, Q, _, IMM_NONE, "Qd <- normalize(Qa)"),
    _s(Op.QIMAG, R, Q, _, IMM_NONE, "Rd <- eta(Qa), signed"),
)}

SPEC_BY_NAME: dict[str, Spec] = {s.name: s for s in SPECS.values()}


def _check_reg(kind: str, value: int, op: Op, slot: str) -> None:
    limit = NUM_QREGS if kind == Q else NUM_RREGS
    if not 0 <= value < limit:
        raise ValueError(
            f"{op.name}: {slot} register {kind.upper()}{value} out of range "
            f"(0..{limit - 1})")


@dataclass(frozen=True)
class Instr:
    """A single decoded instruction word."""

    op: Op
    dst: int = 0
    a: int = 0
    b: int = 0
    imm: int = 0

    @property
    def spec(self) -> Spec:
        return SPECS[self.op]

    def validate(self, const_pool_size: int = 0) -> None:
        """Raise ValueError if any operand is out of range for this opcode."""
        spec = self.spec
        for slot, kind in (("dst", spec.dst), ("a", spec.a), ("b", spec.b)):
            value = getattr(self, slot)
            if kind is None:
                if value != 0:
                    raise ValueError(
                        f"{self.op.name}: unused {slot} slot must be 0, got {value}")
            else:
                _check_reg(kind, value, self.op, slot)
        if spec.imm == IMM_NONE and self.imm != 0:
            raise ValueError(f"{self.op.name}: takes no immediate, got {self.imm}")
        if spec.imm == IMM_CONST and self.imm >= const_pool_size:
            raise ValueError(
                f"{self.op.name}: constant index {self.imm} outside pool of "
                f"size {const_pool_size}")
        if spec.imm == IMM_ENC and self.imm >= len(ENCODE_MODES):
            raise ValueError(f"{self.op.name}: bad encoding mode {self.imm}")
        if spec.imm == IMM_MEAS and self.imm >= len(MEASURE_MODES):
            raise ValueError(f"{self.op.name}: bad measurement mode {self.imm}")
        if spec.imm == IMM_RREG and self.imm >= NUM_RREGS:
            raise ValueError(f"{self.op.name}: bad scalar register R{self.imm}")

    def pack(self) -> bytes:
        return struct.pack("<BBBBI", int(self.op), self.dst, self.a, self.b, self.imm)

    @staticmethod
    def unpack(word: bytes) -> "Instr":
        code, dst, a, b, imm = struct.unpack("<BBBBI", word)
        try:
            op = Op(code)
        except ValueError as exc:  # pragma: no cover - defensive
            raise ValueError(f"unknown opcode 0x{code:02x}") from exc
        return Instr(op, dst, a, b, imm)


@dataclass
class Program:
    """A loadable QVM module: code, constants and a data-segment size."""

    code: list[Instr] = field(default_factory=list)
    consts: list[float] = field(default_factory=list)
    mem_size: int = 0
    name: str = "module"

    def const(self, value: float) -> int:
        """Intern ``value`` in the constant pool and return its index."""
        value = float(value)
        for i, existing in enumerate(self.consts):
            # Bit-pattern comparison so that -0.0 and 0.0 stay distinct and
            # NaN interns exactly once.
            if struct.pack("<d", existing) == struct.pack("<d", value):
                return i
        self.consts.append(value)
        return len(self.consts) - 1

    def emit(self, op: Op, dst: int = 0, a: int = 0, b: int = 0, imm: int = 0) -> Instr:
        instr = Instr(op, dst, a, b, imm)
        instr.validate(len(self.consts))
        self.code.append(instr)
        return instr

    def validate(self) -> None:
        """Check every instruction, and that the module ends in HALT."""
        for pc, instr in enumerate(self.code):
            try:
                instr.validate(len(self.consts))
            except ValueError as exc:
                raise ValueError(f"at pc={pc}: {exc}") from exc
        if not self.code or self.code[-1].op is not Op.HALT:
            raise ValueError("program must end with HALT")

    # -- object file -----------------------------------------------------
    def pack(self) -> bytes:
        """Serialise to the ``.qvm`` object format."""
        name = self.name.encode("utf-8")
        head = struct.pack("<4sHHIII", MAGIC, 1, len(name), len(self.consts),
                           len(self.code), self.mem_size)
        body = name + b"".join(struct.pack("<d", c) for c in self.consts)
        return head + body + b"".join(i.pack() for i in self.code)

    @staticmethod
    def unpack(blob: bytes) -> "Program":
        if len(blob) < 20 or blob[:4] != MAGIC:
            raise ValueError("not a QVM object file")
        magic, version, name_len, n_consts, n_code, mem_size = struct.unpack(
            "<4sHHIII", blob[:20])
        if version != 1:
            raise ValueError(f"unsupported QVM object version {version}")
        off = 20
        name = blob[off:off + name_len].decode("utf-8")
        off += name_len
        consts = list(struct.unpack(f"<{n_consts}d", blob[off:off + 8 * n_consts]))
        off += 8 * n_consts
        need = off + INSTR_SIZE * n_code
        if len(blob) < need:
            raise ValueError("truncated QVM object file")
        code = [Instr.unpack(blob[p:p + INSTR_SIZE])
                for p in range(off, need, INSTR_SIZE)]
        return Program(code=code, consts=consts, mem_size=mem_size, name=name)
