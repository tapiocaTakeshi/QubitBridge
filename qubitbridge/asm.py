"""Assembler and disassembler for the textual Qubit ISA.

The text form is one instruction per line, operands in the fixed order
``dst, a, b, imm`` with unused slots dropped::

    .module  dot2
    .mem     4
            QENC      Q0, R0, latent
            QENC      Q1, R1, latent
            QINT      Q2, Q0, Q1
            QCORR     R2, Q0, Q1
            QMEASURE  R3, Q2, expect
            HALT

Float immediates are written literally and interned into the constant pool;
memory addresses are written ``[12]``; encoding and measurement modes are
written by name.  ``;`` and ``#`` start a comment.
"""

from __future__ import annotations

from .isa import (ENCODE_MODES, IMM_ADDR, IMM_CONST, IMM_ENC, IMM_INT,
                  IMM_MEAS, IMM_NONE, IMM_RREG, MEASURE_MODES, SPEC_BY_NAME,
                  Instr, Program, Q, R)

__all__ = ["AsmError", "assemble", "disassemble", "format_instr"]


class AsmError(ValueError):
    """Raised for a malformed line, with the line number attached."""


def _parse_reg(token: str, kind: str, op: str) -> int:
    token = token.strip()
    want = "Q" if kind == Q else "R"
    if not token[:1].upper() == want or not token[1:].isdigit():
        raise AsmError(f"{op}: expected a {want} register, got {token!r}")
    return int(token[1:])


def _parse_imm(token: str, kind: str, op: str, prog: Program) -> int:
    token = token.strip()
    if kind == IMM_CONST:
        try:
            return prog.const(float(token))
        except ValueError:
            raise AsmError(f"{op}: expected a float constant, got {token!r}") from None
    if kind == IMM_ADDR:
        if not (token.startswith("[") and token.endswith("]")):
            raise AsmError(f"{op}: expected an address like [4], got {token!r}")
        inner = token[1:-1].strip()
        if not inner.isdigit():
            raise AsmError(f"{op}: bad address {token!r}")
        return int(inner)
    if kind == IMM_INT:
        if not token.isdigit():
            raise AsmError(f"{op}: expected a non-negative integer, got {token!r}")
        return int(token)
    if kind == IMM_ENC:
        if token not in ENCODE_MODES:
            raise AsmError(f"{op}: unknown encoding {token!r}, "
                           f"expected one of {', '.join(ENCODE_MODES)}")
        return ENCODE_MODES.index(token)
    if kind == IMM_RREG:
        return _parse_reg(token, R, op)
    if kind == IMM_MEAS:
        if token not in MEASURE_MODES:
            raise AsmError(f"{op}: unknown measurement mode {token!r}, "
                           f"expected one of {', '.join(MEASURE_MODES)}")
        return MEASURE_MODES.index(token)
    raise AsmError(f"{op}: unexpected operand {token!r}")


def _split_operands(rest: str) -> list[str]:
    return [t.strip() for t in rest.split(",") if t.strip()] if rest.strip() else []


def assemble(text: str, name: str = "module") -> Program:
    """Parse Qubit assembly into a validated :class:`~qubitbridge.isa.Program`."""
    prog = Program(name=name)
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.split(";", 1)[0].split("#", 1)[0].strip()
        if not line:
            continue
        try:
            if line.startswith("."):
                directive, _, arg = line.partition(" ")
                arg = arg.strip()
                if directive == ".module":
                    prog.name = arg
                elif directive == ".mem":
                    if not arg.isdigit():
                        raise AsmError(".mem expects a non-negative integer")
                    prog.mem_size = int(arg)
                else:
                    raise AsmError(f"unknown directive {directive!r}")
                continue

            mnemonic, _, rest = line.partition(" ")
            mnemonic = mnemonic.upper()
            if mnemonic not in SPEC_BY_NAME:
                raise AsmError(f"unknown instruction {mnemonic!r}")
            spec = SPEC_BY_NAME[mnemonic]
            operands = _split_operands(rest)

            slots = [k for k in (spec.dst, spec.a, spec.b) if k is not None]
            expected = len(slots) + (0 if spec.imm == IMM_NONE else 1)
            if len(operands) != expected:
                raise AsmError(f"{mnemonic}: expected {expected} operand(s), "
                               f"got {len(operands)}")

            values = {"dst": 0, "a": 0, "b": 0, "imm": 0}
            idx = 0
            for slot, kind in (("dst", spec.dst), ("a", spec.a), ("b", spec.b)):
                if kind is None:
                    continue
                values[slot] = _parse_reg(operands[idx], kind, mnemonic)
                idx += 1
            if spec.imm != IMM_NONE:
                values["imm"] = _parse_imm(operands[idx], spec.imm, mnemonic, prog)

            instr = Instr(spec.op, values["dst"], values["a"], values["b"],
                          values["imm"])
            instr.validate(len(prog.consts))
            prog.code.append(instr)
        except ValueError as exc:
            raise AsmError(f"line {lineno}: {exc}") from None

    try:
        prog.validate()
    except AsmError:
        raise
    except ValueError as exc:
        raise AsmError(str(exc)) from None
    return prog


def _fmt_const(value: float) -> str:
    text = repr(float(value))
    return text


def format_instr(instr: Instr, prog: Program) -> str:
    """Render one instruction back to assembly text."""
    spec = instr.spec
    parts: list[str] = []
    for slot, kind in (("dst", spec.dst), ("a", spec.a), ("b", spec.b)):
        if kind is None:
            continue
        parts.append(f"{'Q' if kind == Q else 'R'}{getattr(instr, slot)}")
    if spec.imm == IMM_CONST:
        parts.append(_fmt_const(prog.consts[instr.imm]))
    elif spec.imm == IMM_ADDR:
        parts.append(f"[{instr.imm}]")
    elif spec.imm == IMM_INT:
        parts.append(str(instr.imm))
    elif spec.imm == IMM_ENC:
        parts.append(ENCODE_MODES[instr.imm])
    elif spec.imm == IMM_MEAS:
        parts.append(MEASURE_MODES[instr.imm])
    elif spec.imm == IMM_RREG:
        parts.append(f"R{instr.imm}")
    if not parts:
        return spec.name
    return f"{spec.name:<9} {', '.join(parts)}"


def disassemble(prog: Program, addresses: bool = False) -> str:
    """Render a whole program as assembly text.

    The output re-assembles to an equivalent program, so
    ``assemble(disassemble(p))`` is a fixed point of the toolchain.
    """
    lines = [f".module {prog.name}"]
    if prog.mem_size:
        lines.append(f".mem {prog.mem_size}")
    lines.append("")
    for pc, instr in enumerate(prog.code):
        body = format_instr(instr, prog)
        lines.append(f"{pc:04d}: {body}" if addresses else f"        {body}")
    return "\n".join(lines) + "\n"
