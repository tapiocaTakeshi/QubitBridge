"""AArch64 / NEON code generation for the Qubit ISA.

This is the third layer of the ``APQB IR -> QVM -> ARM64`` stack.  It is a code
*emitter*, not an interpreter: it turns a :class:`~qubitbridge.isa.Program` into
an AArch64 assembly module made of

1. a **kernel library** -- one routine per APQB opcode, operating on lane arrays
   of ``double``.  The algebraic kernels (interact, correlate, normalize, ...)
   are NEON, chewing two lanes per ``.2d`` instruction with a scalar tail for an
   odd lane count.  The transcendental ones (latent encode/decode, rotate, gate,
   entropy) are scalar loops calling libm, because ``tanh``/``sin``/``log2`` have
   no single NEON instruction.
2. a **driver** ``qb_prog_<name>`` that walks the program once, computing each
   operand's address in a flat register file and calling the kernels.

Memory layout used by the driver, with ``stride = n * 8`` bytes::

    Q[i].r   at qbase + (2i)   * stride      R[i] at rbase + i * stride
    Q[i].eta at qbase + (2i+1) * stride      constants at cpool[imm]

Emitted assembly is checked with ``llvm-mc`` where available (see
:func:`assemble_text`), so the test-suite proves the output is real AArch64
rather than plausible-looking text.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass

from ..isa import (ENCODE_MODES, MEASURE_MODES, NUM_QREGS, NUM_RREGS,
                   Instr, Op, Program)
from .base import Backend

__all__ = [
    "Arm64Backend",
    "Arm64UnsupportedOp",
    "assemble_text",
    "have_assembler",
]

#: NEON registers the kernels are free to use (v16-v31 are all caller-saved).
_BODY_BASE = 16


class Arm64UnsupportedOp(NotImplementedError):
    """Raised for an opcode with no ARM64 lowering (see :data:`Arm64Backend.UNSUPPORTED`)."""


# ===========================================================================
# A tiny two-form instruction DSL: each body renders once as NEON .2d and
# once as scalar d, so a kernel is written exactly once.
# ===========================================================================

_BINARY = {"fmul", "fadd", "fsub", "fdiv", "fmax", "fmin"}
_UNARY = {"fabs", "fsqrt", "fneg"}


def _render(instr: tuple, vector: bool) -> str:
    def reg(i: int) -> str:
        return f"v{i}.2d" if vector else f"d{i}"

    kind = instr[0]
    if kind in _BINARY:
        _, d, n, m = instr
        return f"{kind} {reg(d)}, {reg(n)}, {reg(m)}"
    if kind in _UNARY:
        _, d, n = instr
        return f"{kind} {reg(d)}, {reg(n)}"
    if kind == "fmov_imm":
        _, d, imm = instr
        return f"fmov {reg(d)}, #{imm}"
    if kind == "copy":
        _, d, n = instr
        return f"mov v{d}.16b, v{n}.16b" if vector else f"fmov d{d}, d{n}"
    if kind == "dup":
        # Broadcast the scalar argument in d0 across both lanes.
        _, d, n = instr
        return f"dup v{d}.2d, v{n}.d[0]" if vector else f"fmov d{d}, d{n}"
    if kind == "zero":
        _, d = instr
        return f"movi v{d}.2d, #0000000000000000" if vector else f"fmov d{d}, xzr"
    raise ValueError(f"unknown DSL instruction {instr!r}")  # pragma: no cover


@dataclass
class _Kernel:
    """One lane-array routine.

    ``params`` is the ABI in order: ``"in"``/``"out"`` pointer arguments get
    consecutive x registers, ``"fp"`` a double in ``d0``; the lane count always
    lands in the x register after the last pointer.
    """

    name: str
    params: tuple[str, ...]
    loads: tuple[int, ...] = ()        # body reg <- params[i] for each in-ptr
    stores: tuple[int, ...] = ()       # body reg -> params[i] for each out-ptr
    body: tuple[tuple, ...] = ()
    doc: str = ""
    raw: str | None = None             # hand-written text (helper-call kernels)

    @property
    def ptr_params(self) -> list[int]:
        return [i for i, p in enumerate(self.params) if p in ("in", "out")]


def _emit_vector_kernel(k: _Kernel) -> str:
    ptrs = k.ptr_params
    n_reg = f"x{len(ptrs)}"
    ins = [i for i, p in enumerate(k.params) if p == "in"]
    outs = [i for i, p in enumerate(k.params) if p == "out"]
    slot = {p: idx for idx, p in enumerate(ptrs)}

    lines = [
        f"// {k.doc}",
        f".globl {k.name}",
        ".p2align 2",
        f"{k.name}:",
        f"    cbz   {n_reg}, .L{k.name}_done",
        f"    mov   x15, {n_reg}",
        "    lsr   x14, x15, #1",
        f"    cbz   x14, .L{k.name}_tail",
        f".L{k.name}_vec:",
    ]
    for reg, p in zip(k.loads, ins):
        lines.append(f"    ld1   {{v{reg}.2d}}, [x{slot[p]}], #16")
    for instr in k.body:
        lines.append("    " + _render(instr, vector=True))
    for reg, p in zip(k.stores, outs):
        lines.append(f"    st1   {{v{reg}.2d}}, [x{slot[p]}], #16")
    lines += [
        "    subs  x14, x14, #1",
        f"    b.ne  .L{k.name}_vec",
        f".L{k.name}_tail:",
        f"    tbz   x15, #0, .L{k.name}_done",
    ]
    for reg, p in zip(k.loads, ins):
        lines.append(f"    ldr   d{reg}, [x{slot[p]}]")
    for instr in k.body:
        lines.append("    " + _render(instr, vector=False))
    for reg, p in zip(k.stores, outs):
        lines.append(f"    str   d{reg}, [x{slot[p]}]")
    lines += [f".L{k.name}_done:", "    ret", ""]
    return "\n".join(lines)


# -- NEON kernels -----------------------------------------------------------
# Body registers: 16, 17, ... ; constants land in 30 (1.0) and 31 (-1.0).
_ONE, _NEG_ONE, _FP = 30, 31, 29
A_R, A_E, B_R, B_E, T0, T1, T2, T3 = (16, 17, 18, 19, 20, 21, 22, 23)

_CLAMP = (("fmov_imm", _ONE, "1.0"), ("fmov_imm", _NEG_ONE, "-1.0"))


def _eta_from_r(src: int, dst: int, tmp: int) -> tuple[tuple, ...]:
    """dst = sqrt(max(0, 1 - src^2))."""
    return (
        ("fmul", tmp, src, src),
        ("fmov_imm", _ONE, "1.0"),
        ("fsub", dst, _ONE, tmp),
        ("zero", _NEG_ONE),                   # 0.0 for the max below
        ("fmax", dst, dst, _NEG_ONE),
        ("fsqrt", dst, dst),
    )


NEON_KERNELS: tuple[_Kernel, ...] = (
    _Kernel(
        "qb_interact", ("in", "in", "in", "in", "out", "out"),
        loads=(A_R, A_E, B_R, B_E), stores=(T2, T3),
        body=(("fmul", T0, A_R, B_R), ("fmul", T1, A_E, B_E),
              ("fsub", T2, T0, T1),
              ("fmul", T0, A_R, B_E), ("fmul", T1, A_E, B_R),
              ("fadd", T3, T0, T1)),
        doc="QINT: (dr, de) = (ar*br - ae*be, ar*be + ae*br)"),
    _Kernel(
        "qb_mul", ("in", "in", "in", "in", "out", "out"),
        loads=(A_R, A_E, B_R, B_E), stores=(T2, T3),
        body=(("fmul", T2, A_R, B_R),) + _CLAMP +
             (("fmin", T2, T2, _ONE), ("fmax", T2, T2, _NEG_ONE)) +
             _eta_from_r(T2, T3, T0),
        doc="QMUL: dr = clamp(ar*br), de = sqrt(1 - dr^2)"),
    _Kernel(
        "qb_correlate", ("in", "in", "in", "in", "out"),
        loads=(A_R, A_E, B_R, B_E), stores=(T2,),
        body=(("fmul", T0, A_R, B_R), ("fmul", T1, A_E, B_E),
              ("fadd", T2, T0, T1)),
        doc="QCORR: out = ar*br + ae*be"),
    _Kernel(
        "qb_uncertainty", ("in", "out"), loads=(A_E,), stores=(T0,),
        body=(("fabs", T0, A_E),), doc="QUNC: out = |eta|"),
    _Kernel(
        "qb_copy", ("in", "out"), loads=(A_R,), stores=(A_R,),
        doc="MOV / QIMAG / QDEC linear / QMEASURE expect: out = in"),
    _Kernel(
        "qb_encode_linear", ("in", "out", "out"),
        loads=(A_R,), stores=(T2, T3),
        body=_CLAMP + (("fmin", T2, A_R, _ONE), ("fmax", T2, T2, _NEG_ONE)) +
             _eta_from_r(T2, T3, T0),
        doc="QENC linear: dr = clamp(x), de = sqrt(1 - dr^2)"),
    _Kernel(
        "qb_encode_prob", ("in", "out", "out"),
        loads=(A_R,), stores=(T2, T3),
        body=(("fadd", T2, A_R, A_R), ("fmov_imm", _ONE, "1.0"),
              ("fsub", T2, T2, _ONE), ("fmov_imm", _NEG_ONE, "-1.0"),
              ("fmin", T2, T2, _ONE), ("fmax", T2, T2, _NEG_ONE)) +
             _eta_from_r(T2, T3, T0),
        doc="QENC prob: dr = clamp(2p - 1), de = sqrt(1 - dr^2)"),
    _Kernel(
        "qb_normalize", ("in", "in", "out", "out"),
        loads=(A_R, A_E), stores=(T2, T3),
        body=(("fmul", T0, A_R, A_R), ("fmul", T1, A_E, A_E),
              ("fadd", T0, T0, T1), ("fsqrt", T0, T0),
              ("fdiv", T2, A_R, T0), ("fdiv", T3, A_E, T0)),
        doc="QNORM: re-project (r, eta) onto the unit circle"),
    _Kernel(
        "qb_add", ("in", "in", "out"), loads=(A_R, B_R), stores=(T0,),
        body=(("fadd", T0, A_R, B_R),), doc="ADD"),
    _Kernel(
        "qb_sub", ("in", "in", "out"), loads=(A_R, B_R), stores=(T0,),
        body=(("fsub", T0, A_R, B_R),), doc="SUB"),
    _Kernel(
        "qb_fmul", ("in", "in", "out"), loads=(A_R, B_R), stores=(T0,),
        body=(("fmul", T0, A_R, B_R),), doc="MUL"),
    _Kernel(
        "qb_fdiv", ("in", "in", "out"), loads=(A_R, B_R), stores=(T0,),
        body=(("fdiv", T0, A_R, B_R),), doc="DIV"),
    _Kernel(
        "qb_neg", ("in", "out"), loads=(A_R,), stores=(T0,),
        body=(("fneg", T0, A_R),), doc="NEG"),
    _Kernel(
        "qb_min", ("in", "in", "out"), loads=(A_R, B_R), stores=(T0,),
        body=(("fmin", T0, A_R, B_R),), doc="MIN"),
    _Kernel(
        "qb_max", ("in", "in", "out"), loads=(A_R, B_R), stores=(T0,),
        body=(("fmax", T0, A_R, B_R),), doc="MAX"),
    _Kernel(
        "qb_scale", ("in", "fp", "out"), loads=(A_R,), stores=(T0,),
        body=(("dup", _FP, 0), ("fmul", T0, A_R, _FP)),
        doc="SCALE: out = in * c (c in d0)"),
    _Kernel(
        "qb_offset", ("in", "fp", "out"), loads=(A_R,), stores=(T0,),
        body=(("dup", _FP, 0), ("fadd", T0, A_R, _FP)),
        doc="ADDI: out = in + c (c in d0)"),
    _Kernel(
        "qb_splat", ("fp", "out"), stores=(_FP,),
        body=(("dup", _FP, 0),), doc="LDI: out[i] = c (c in d0)"),
    _Kernel(
        "qb_qload", ("fp", "out", "out"), stores=(T2, T3),
        body=(("dup", _FP, 0),) + _CLAMP +
             (("fmin", T2, _FP, _ONE), ("fmax", T2, T2, _NEG_ONE)) +
             _eta_from_r(T2, T3, T0),
        doc="QLOAD: canonical state with r = c (c in d0)"),
    _Kernel(
        "qb_identity", ("out", "out"), stores=(T2, T3),
        body=(("fmov_imm", T2, "1.0"), ("zero", T3)),
        doc="QPOW seed: (r, eta) = (1, 0)"),
)


# -- libm-backed scalar kernels --------------------------------------------
# These keep the loop pointer in x19..x22 and the carried value in d8/d9,
# which survive the calls (callee-saved), per AAPCS64.

def _scalar_loop(name: str, doc: str, n_ptrs: int, body: str) -> _Kernel:
    saves = ["    stp   x29, x30, [sp, #-96]!", "    mov   x29, sp",
             "    stp   x19, x20, [sp, #16]", "    stp   x21, x22, [sp, #32]",
             "    stp   x23, x24, [sp, #48]", "    stp   d8, d9, [sp, #64]",
             "    stp   d10, d11, [sp, #80]"]
    restores = ["    ldp   d10, d11, [sp, #80]", "    ldp   d8, d9, [sp, #64]",
                "    ldp   x23, x24, [sp, #48]", "    ldp   x21, x22, [sp, #32]",
                "    ldp   x19, x20, [sp, #16]",
                "    ldp   x29, x30, [sp], #96", "    ret"]
    moves = [f"    mov   x{19 + i}, x{i}" for i in range(n_ptrs)]
    moves.append(f"    mov   x{19 + n_ptrs}, x{n_ptrs}")
    counter = f"x{19 + n_ptrs}"
    text = "\n".join([
        f"// {doc}", f".globl {name}", ".p2align 2", f"{name}:", *saves,
        f"    cbz   x{n_ptrs}, .L{name}_done", *moves, f".L{name}_loop:",
        body.rstrip(),
        f"    subs  {counter}, {counter}, #1",
        f"    b.ne  .L{name}_loop", f".L{name}_done:", *restores, ""])
    return _Kernel(name, (), doc=doc, raw=text)


HELPER_KERNELS: tuple[_Kernel, ...] = (
    _scalar_loop(
        "qb_encode_latent",
        "QENC latent: r = tanh(a), eta = sech(a) = 1/cosh(a)", 3,
        """    ldr   d0, [x19], #8
    fmov  d8, d0
    bl    tanh
    str   d0, [x20], #8
    fmov  d0, d8
    bl    cosh
    fmov  d1, #1.0
    fdiv  d0, d1, d0
    str   d0, [x21], #8"""),
    _scalar_loop(
        "qb_decode_latent", "QDEC latent: out = atanh(clamp(r, +-R_MAX))", 2,
        """    ldr   d0, [x19], #8
    mov   x9, #0xffff
    movk  x9, #0xffff, lsl #16
    movk  x9, #0xffff, lsl #32
    movk  x9, #0x3fef, lsl #48
    fmov  d1, x9
    fmin  d0, d0, d1
    fneg  d1, d1
    fmax  d0, d0, d1
    bl    atanh
    str   d0, [x20], #8"""),
    _scalar_loop(
        "qb_decode_angle", "QDEC angle: out = 0.5 * atan2(eta, r)", 3,
        """    ldr   d0, [x19], #8
    ldr   d1, [x20], #8
    bl    atan2
    fmov  d1, #0.5
    fmul  d0, d0, d1
    str   d0, [x21], #8"""),
    _scalar_loop(
        "qb_rotate",
        "QROT: z -> z * e^{i2phi}; args (ar, ae, phi, dr, de, n)", 5,
        """    ldr   d0, [x21], #8
    fadd  d0, d0, d0
    fmov  d8, d0
    bl    cos
    fmov  d9, d0
    fmov  d0, d8
    bl    sin
    ldr   d2, [x19], #8
    ldr   d3, [x20], #8
    fmul  d4, d2, d9
    fmul  d5, d3, d0
    fsub  d4, d4, d5
    fmul  d6, d2, d0
    fmul  d7, d3, d9
    fadd  d6, d6, d7
    str   d4, [x22], #8
    str   d6, [x23], #8"""),
    _scalar_loop(
        "qb_gate",
        "QGATE: a = atanh(tr) + j*sr; (dr, de) = (tanh a, 2/cosh a); "
        "args (tr, sr, j, dr, de, n)", 5,
        """    ldr   d0, [x19], #8
    mov   x9, #0xffff
    movk  x9, #0xffff, lsl #16
    movk  x9, #0xffff, lsl #32
    movk  x9, #0x3fef, lsl #48
    fmov  d1, x9
    fmin  d0, d0, d1
    fneg  d1, d1
    fmax  d0, d0, d1
    bl    atanh
    ldr   d1, [x20], #8
    ldr   d2, [x21], #8
    fmul  d1, d1, d2
    fadd  d0, d0, d1
    fmov  d8, d0
    bl    tanh
    str   d0, [x22], #8
    fmov  d0, d8
    bl    cosh
    fmov  d1, #1.0
    fdiv  d0, d1, d0
    str   d0, [x23], #8"""),
    _scalar_loop(
        "qb_entropy", "QENT: out = H_Z(r) in bits", 2,
        """    ldr   d0, [x19], #8
    fmov  d1, #1.0
    fadd  d2, d1, d0
    fsub  d3, d1, d0
    fmov  d1, #0.5
    fmul  d8, d2, d1
    fmul  d9, d3, d1
    fmov  d10, xzr
    fcmp  d8, #0.0
    b.le  1f
    fmov  d0, d8
    bl    log2
    fmul  d0, d0, d8
    fsub  d10, d10, d0
1:
    fcmp  d9, #0.0
    b.le  2f
    fmov  d0, d9
    bl    log2
    fmul  d0, d0, d9
    fsub  d10, d10, d0
2:
    str   d10, [x20], #8"""),
)


HELPER_KERNELS = HELPER_KERNELS + (
    _scalar_loop(
        "qb_tanh", "TANH: out = tanh(x)", 2,
        """    ldr   d0, [x19], #8
    bl    tanh
    str   d0, [x20], #8"""),
    _scalar_loop(
        "qb_encode_angle", "QENC angle: (dr, de) = (cos 2t, sin 2t)", 3,
        """    ldr   d0, [x19], #8
    fadd  d0, d0, d0
    fmov  d8, d0
    bl    cos
    str   d0, [x20], #8
    fmov  d0, d8
    bl    sin
    str   d0, [x21], #8"""),
)

NEON_KERNELS = NEON_KERNELS + (
    _Kernel(
        "qb_decode_prob", ("in", "out"), loads=(A_R,), stores=(T0,),
        body=(("fmov_imm", _ONE, "1.0"), ("fadd", T0, A_R, _ONE),
              ("fmov_imm", _ONE, "0.5"), ("fmul", T0, T0, _ONE)),
        doc="QDEC prob: out = (1 + r) / 2"),
)


def kernel_library() -> str:
    """The full AArch64 kernel library as assembly text."""
    parts = [
        "// Qubit VM -- AArch64/NEON kernel library",
        "// Generated by qubitbridge.backends.arm64; do not edit by hand.",
        "    .text",
        "",
    ]
    for k in NEON_KERNELS:
        parts.append(_emit_vector_kernel(k))
    for k in HELPER_KERNELS:
        parts.append(k.raw)
    return "\n".join(parts)


# ===========================================================================
# Program driver
# ===========================================================================

#: Base registers held across calls by the driver.
_QBASE, _RBASE, _CPOOL, _SCRATCH, _N, _STRIDE, _TMP = (
    "x19", "x20", "x21", "x22", "x23", "x24", "x25")


class _Driver:
    """Emits ``qb_prog_<name>``: one ``bl`` sequence for a whole program."""

    def __init__(self, program: Program):
        self.program = program
        self.lines: list[str] = []

    # -- address arithmetic ----------------------------------------------
    def _addr(self, dst: str, base: str, multiple: int) -> None:
        if multiple == 0:
            self.lines.append(f"    mov   {dst}, {base}")
        else:
            self.lines.append(f"    mov   {_TMP}, #{multiple}")
            self.lines.append(f"    madd  {dst}, {_TMP}, {_STRIDE}, {base}")

    def _operand(self, dst: str, ref: tuple[str, int]) -> None:
        kind, index = ref
        if kind == "qr":
            self._addr(dst, _QBASE, 2 * index)
        elif kind == "qe":
            self._addr(dst, _QBASE, 2 * index + 1)
        elif kind == "r":
            self._addr(dst, _RBASE, index)
        elif kind == "s":
            self._addr(dst, _SCRATCH, index)
        else:  # pragma: no cover - internal
            raise ValueError(f"bad operand reference {ref!r}")

    def call(self, kernel: str, ptr_args: list[tuple[str, int]],
             const: int | None = None) -> None:
        for slot, ref in enumerate(ptr_args):
            self._operand(f"x{slot}", ref)
        self.lines.append(f"    mov   x{len(ptr_args)}, {_N}")
        if const is not None:
            offset = 8 * const
            if offset > 32760:  # pragma: no cover - pools are small
                raise Arm64UnsupportedOp(
                    "constant pool too large for a single ldr offset")
            self.lines.append(f"    ldr   d0, [{_CPOOL}, #{offset}]")
        self.lines.append(f"    bl    {kernel}")

    def splat_const(self, const: int, slot: int = 0) -> tuple[str, int]:
        """Broadcast ``cpool[const]`` into scratch lane vector ``slot``."""
        self.call("qb_splat", [("s", slot)], const=const)
        return ("s", slot)


_ENC_KERNEL = {
    "latent": "qb_encode_latent",
    "linear": "qb_encode_linear",
    "prob": "qb_encode_prob",
    "angle": "qb_encode_angle",
}


def _emit_instr(drv: _Driver, instr: Instr) -> None:
    op, d, a, b, imm = instr.op, instr.dst, instr.a, instr.b, instr.imm

    if op in (Op.HALT, Op.NOP):
        return
    if op in Arm64Backend.UNSUPPORTED:
        raise Arm64UnsupportedOp(
            f"{op.name} has no ARM64 lowering: {Arm64Backend.UNSUPPORTED[op]}")

    if op is Op.MOV:
        drv.call("qb_copy", [("r", a), ("r", d)])
    elif op is Op.LDI:
        drv.call("qb_splat", [("r", d)], const=imm)
    elif op in (Op.ADD, Op.SUB, Op.MUL, Op.DIV, Op.MIN, Op.MAX):
        kernel = {Op.ADD: "qb_add", Op.SUB: "qb_sub",
                  Op.MUL: "qb_fmul", Op.DIV: "qb_fdiv",
                  Op.MIN: "qb_min", Op.MAX: "qb_max"}[op]
        drv.call(kernel, [("r", a), ("r", b), ("r", d)])
    elif op is Op.NEG:
        drv.call("qb_neg", [("r", a), ("r", d)])
    elif op is Op.TANH:
        drv.call("qb_tanh", [("r", a), ("r", d)])
    elif op is Op.ATANH:
        drv.call("qb_decode_latent", [("r", a), ("r", d)])
    elif op is Op.SCALE:
        drv.call("qb_scale", [("r", a), ("r", d)], const=imm)
    elif op is Op.ADDI:
        drv.call("qb_offset", [("r", a), ("r", d)], const=imm)

    elif op is Op.QLOAD:
        drv.call("qb_qload", [("qr", d), ("qe", d)], const=imm)
    elif op is Op.QMOV:
        drv.call("qb_copy", [("qr", a), ("qr", d)])
        drv.call("qb_copy", [("qe", a), ("qe", d)])
    elif op is Op.QENC:
        mode = ENCODE_MODES[imm]
        drv.call(_ENC_KERNEL[mode], [("r", a), ("qr", d), ("qe", d)])
    elif op is Op.QDEC:
        mode = ENCODE_MODES[imm]
        if mode == "linear":
            drv.call("qb_copy", [("qr", a), ("r", d)])
        elif mode == "latent":
            drv.call("qb_decode_latent", [("qr", a), ("r", d)])
        elif mode == "prob":
            drv.call("qb_decode_prob", [("qr", a), ("r", d)])
        else:
            drv.call("qb_decode_angle", [("qe", a), ("qr", a), ("r", d)])
    elif op in (Op.QROT, Op.QROTR):
        phi = drv.splat_const(imm) if op is Op.QROT else ("r", b)
        drv.call("qb_rotate", [("qr", a), ("qe", a), phi, ("qr", d), ("qe", d)])
    elif op is Op.QINT:
        drv.call("qb_interact",
                 [("qr", a), ("qe", a), ("qr", b), ("qe", b), ("qr", d), ("qe", d)])
    elif op is Op.QMUL:
        drv.call("qb_mul",
                 [("qr", a), ("qe", a), ("qr", b), ("qe", b), ("qr", d), ("qe", d)])
    elif op is Op.QPOW:
        # Stage the source in scratch first: the destination may alias it.
        drv.call("qb_copy", [("qr", a), ("s", 0)])
        drv.call("qb_copy", [("qe", a), ("s", 1)])
        drv.call("qb_identity", [("qr", d), ("qe", d)])
        for _ in range(imm):
            drv.call("qb_interact",
                     [("qr", d), ("qe", d), ("s", 0), ("s", 1),
                      ("qr", d), ("qe", d)])
    elif op is Op.QCORR:
        drv.call("qb_correlate",
                 [("qr", a), ("qe", a), ("qr", b), ("qe", b), ("r", d)])
    elif op is Op.QUNC:
        drv.call("qb_uncertainty", [("qe", a), ("r", d)])
    elif op is Op.QIMAG:
        drv.call("qb_copy", [("qe", a), ("r", d)])
    elif op is Op.QENT:
        drv.call("qb_entropy", [("qr", a), ("r", d)])
    elif op in (Op.QGATE, Op.QGATER):
        j = drv.splat_const(imm) if op is Op.QGATE else ("r", imm)
        drv.call("qb_gate",
                 [("qr", a), ("qr", b), j, ("qr", d), ("qe", d)])
    elif op is Op.QMEASURE:
        drv.call("qb_copy", [("qr", a), ("r", d)])
    elif op is Op.QNORM:
        drv.call("qb_normalize", [("qr", a), ("qe", a), ("qr", d), ("qe", d)])
    else:  # pragma: no cover - table above is exhaustive
        raise Arm64UnsupportedOp(f"no ARM64 lowering for {op.name}")


def _sanitize(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name)


def emit_driver(program: Program) -> str:
    """Emit ``qb_prog_<name>`` for one program."""
    program.validate()
    symbol = f"qb_prog_{_sanitize(program.name)}"
    drv = _Driver(program)
    for instr in program.code:
        _emit_instr(drv, instr)

    head = [
        f"// Driver for QVM module '{program.name}'",
        f"// void {symbol}(double *qbase, double *rbase, const double *cpool,",
        "//               double *scratch, uint64_t n);",
        f"//   qbase   : {2 * NUM_QREGS} lane vectors "
        f"(Q[i].r at 2i, Q[i].eta at 2i+1)",
        f"//   rbase   : {NUM_RREGS} lane vectors",
        f"//   cpool   : {len(program.consts)} doubles",
        "//   scratch : 2 lane vectors",
        "    .text",
        f"    .globl {symbol}",
        "    .p2align 2",
        f"{symbol}:",
        "    stp   x29, x30, [sp, #-80]!",
        "    mov   x29, sp",
        "    stp   x19, x20, [sp, #16]",
        "    stp   x21, x22, [sp, #32]",
        "    stp   x23, x24, [sp, #48]",
        "    str   x25, [sp, #64]",
        f"    mov   {_QBASE}, x0",
        f"    mov   {_RBASE}, x1",
        f"    mov   {_CPOOL}, x2",
        f"    mov   {_SCRATCH}, x3",
        f"    mov   {_N}, x4",
        f"    lsl   {_STRIDE}, {_N}, #3",
    ]
    tail = [
        "    ldr   x25, [sp, #64]",
        "    ldp   x23, x24, [sp, #48]",
        "    ldp   x21, x22, [sp, #32]",
        "    ldp   x19, x20, [sp, #16]",
        "    ldp   x29, x30, [sp], #80",
        "    ret",
        "",
    ]
    return "\n".join(head + drv.lines + tail)


# ===========================================================================
# Toolchain integration
# ===========================================================================

def have_assembler() -> str | None:
    """Path to an AArch64-capable assembler, or None."""
    return shutil.which("llvm-mc") or shutil.which("aarch64-linux-gnu-as")


def assemble_text(text: str) -> tuple[bool, str]:
    """Assemble AArch64 text, returning ``(ok, tool output)``.

    Uses ``llvm-mc``, which cross-assembles on any host, so emitted code can be
    validated even from an x86-64 machine.  Returns ``(False, reason)`` when no
    assembler is installed.
    """
    tool = have_assembler()
    if tool is None:
        return False, "no AArch64 assembler found (install llvm-mc or binutils)"
    if tool.endswith("llvm-mc"):
        cmd = [tool, "-triple=aarch64-unknown-linux-gnu", "-assemble",
               "-filetype=obj", "-o", "/dev/null"]
    else:  # pragma: no cover - depends on the host toolchain
        cmd = [tool, "-o", "/dev/null"]
    proc = subprocess.run(cmd, input=text, capture_output=True, text=True)
    return proc.returncode == 0, (proc.stderr or proc.stdout).strip()


class Arm64Backend(Backend):
    """AArch64/NEON code generator.

    It implements the emit half of the backend contract only: ``executes`` is
    False, so :class:`~qubitbridge.vm.QVM` refuses it and callers use
    :meth:`emit` instead.  Numerics are validated against the portable backend
    by assembling and running the output on an AArch64 host (see
    ``tests/test_arm64.py``).
    """

    name = "arm64"
    executes = False
    lanes_per_vector = 2

    #: Opcodes with no ARM64 lowering, and why.
    UNSUPPORTED = {
        Op.LOAD: "the driver has no data segment; keep values in registers",
        Op.STORE: "the driver has no data segment; keep values in registers",
        Op.QSTORE: "the driver has no data segment; keep values in registers",
    }

    def emit(self, program: Program, with_library: bool = True) -> str:
        """Emit assembly for ``program`` (kernel library + driver)."""
        for instr in program.code:
            if instr.op is Op.QMEASURE and MEASURE_MODES[instr.imm] == "sample":
                raise Arm64UnsupportedOp(
                    "QMEASURE sample needs the VM's seeded PRNG; draw the "
                    "uniforms on the host and feed them in as a scalar register")
        driver = emit_driver(program)
        return f"{kernel_library()}\n{driver}" if with_library else driver

    def verify(self, program: Program) -> tuple[bool, str]:
        """Emit and assemble, returning ``(ok, assembler output)``."""
        return assemble_text(self.emit(program))
