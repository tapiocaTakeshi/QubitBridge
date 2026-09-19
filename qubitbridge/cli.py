"""``qvm`` -- command line driver for the Qubit toolchain.

    qvm compile kernel.py            # Python subset  -> APQB IR (+ partition report)
    qvm lower    kernel.ir           # APQB IR        -> Qubit assembly
    qvm asm      kernel.qasm -o k.qvm    # assembly   -> .qvm object
    qvm disasm   kernel.qvm          # object         -> assembly
    qvm run      kernel.py -i x=0.8 -i y=0.4
    qvm emit-arm64 kernel.ir --check # AArch64/NEON, optionally assembled
    qvm info                         # ISA, backends, toolchain
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .asm import assemble, disassemble
from .backends import available_backends
from .frontend import compile_source
from .ir import parse_module
from .isa import SPECS, Program
from .lower import Lowered, lower_module
from .vm import QVM

__all__ = ["main"]


def _load(path: Path) -> tuple[Program, Lowered | None, object]:
    """Load any supported input, returning (program, lowering, report)."""
    suffix = path.suffix.lower()
    text = None if suffix == ".qvm" else path.read_text()

    if suffix == ".py":
        module, report = compile_source(text, module_name=path.stem)
        lowered = lower_module(module)[module.funcs[0].name]
        return lowered.program, lowered, report
    if suffix == ".ir":
        module = parse_module(text)
        lowered = lower_module(module)[module.funcs[0].name]
        return lowered.program, lowered, None
    if suffix in (".qasm", ".s", ".asm"):
        return assemble(text, name=path.stem), None, None
    if suffix == ".qvm":
        return Program.unpack(path.read_bytes()), None, None
    raise SystemExit(f"qvm: unknown input type {suffix!r} (expected .py, .ir, "
                     f".qasm or .qvm)")


def _parse_inputs(pairs: list[str], lowered: Lowered | None) -> dict[int, list[float]]:
    """Turn ``--input x=0.8`` / ``--input R0=1,2`` into a register seed map."""
    seeds: dict[int, list[float]] = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"qvm: bad --input {pair!r}, expected name=value")
        name, _, raw = pair.partition("=")
        name = name.strip()
        try:
            values = [float(v) for v in raw.split(",")]
        except ValueError:
            raise SystemExit(f"qvm: bad --input value in {pair!r}") from None

        if name.upper().startswith("R") and name[1:].isdigit():
            index = int(name[1:])
        elif name.isdigit():
            index = int(name)
        elif lowered is not None:
            index = _arg_index(lowered, name)
        else:
            raise SystemExit(f"qvm: cannot resolve input {name!r} without IR; "
                             f"use R<n>=value")
        seeds[index] = values
    return seeds


def _arg_index(lowered: Lowered, name: str) -> int:
    raise SystemExit(f"qvm: input {name!r} is not a register; pass R<n>=value "
                     f"(argument order is {lowered.arg_regs})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="qvm", description="Qubit Virtual Machine toolchain")
    parser.add_argument("--version", action="version",
                        version=f"qubitbridge {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_input(p):
        p.add_argument("input", type=Path, help=".py, .ir, .qasm or .qvm")

    p = sub.add_parser("compile", help="Python subset -> APQB IR")
    p.add_argument("input", type=Path)
    p.add_argument("-o", "--output", type=Path)
    p.add_argument("--report", action="store_true",
                   help="also print the classical/APQB partition report")

    p = sub.add_parser("lower", help="-> Qubit assembly")
    add_input(p)
    p.add_argument("-o", "--output", type=Path)
    p.add_argument("--addresses", action="store_true")

    p = sub.add_parser("asm", help="assembly -> .qvm object")
    add_input(p)
    p.add_argument("-o", "--output", type=Path, required=True)

    p = sub.add_parser("disasm", help=".qvm object -> assembly")
    add_input(p)

    p = sub.add_parser("run", help="execute on the QVM")
    add_input(p)
    p.add_argument("-i", "--input-value", action="append", default=[],
                   metavar="R0=V", help="seed a scalar register (repeatable)")
    p.add_argument("--lanes", type=int, default=1)
    p.add_argument("--backend", default="auto",
                   help=f"one of: auto, {', '.join(available_backends())}")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--trace", action="store_true")

    p = sub.add_parser("emit-arm64", help="AArch64/NEON assembly")
    add_input(p)
    p.add_argument("-o", "--output", type=Path)
    p.add_argument("--driver-only", action="store_true")
    p.add_argument("--check", action="store_true",
                   help="assemble the result with llvm-mc")

    sub.add_parser("info", help="show the ISA and available backends")

    args = parser.parse_args(argv)

    if args.command == "info":
        return _cmd_info()

    program, lowered, report = _load(args.input)

    if args.command == "compile":
        module, rep = compile_source(args.input.read_text(),
                                     module_name=args.input.stem)
        text = module.to_text()
        _write(text, getattr(args, "output", None))
        if args.report:
            print(rep.summary(), file=sys.stderr)
        return 0

    if args.command == "lower":
        _write(disassemble(program, addresses=args.addresses), args.output)
        return 0

    if args.command == "asm":
        args.output.write_bytes(program.pack())
        print(f"wrote {args.output} ({len(program.code)} instructions, "
              f"{len(program.consts)} constants)", file=sys.stderr)
        return 0

    if args.command == "disasm":
        print(disassemble(program, addresses=True), end="")
        return 0

    if args.command == "run":
        return _cmd_run(args, program, lowered)

    if args.command == "emit-arm64":
        return _cmd_emit_arm64(args, program)

    parser.error(f"unhandled command {args.command}")  # pragma: no cover
    return 2


def _write(text: str, output) -> None:
    if output is None:
        print(text, end="" if text.endswith("\n") else "\n")
    else:
        output.write_text(text if text.endswith("\n") else text + "\n")


def _cmd_info() -> int:
    from .backends.arm64 import have_assembler
    print(f"qubitbridge {__version__}")
    print(f"backends: {', '.join(available_backends())} (+ arm64 code emitter)")
    print(f"AArch64 assembler: {have_assembler() or 'not found'}")
    print(f"\n{'opcode':<10} {'hex':<6} operands")
    for op, spec in sorted(SPECS.items(), key=lambda kv: int(kv[0])):
        slots = [k for k in (spec.dst, spec.a, spec.b) if k]
        shape = ", ".join(s.upper() for s in slots)
        if spec.imm != "none":
            shape = f"{shape}, <{spec.imm}>" if shape else f"<{spec.imm}>"
        print(f"{spec.name:<10} 0x{int(op):02x}   {shape:<22} {spec.doc}")
    return 0


def _cmd_run(args, program: Program, lowered: Lowered | None) -> int:
    seeds = _parse_inputs(args.input_value, lowered)
    lanes = args.lanes
    for index, values in seeds.items():
        if len(values) not in (1, lanes):
            raise SystemExit(f"qvm: R{index} has {len(values)} values but "
                             f"--lanes is {lanes}")
    seeds = {i: (v[0] if len(v) == 1 else v) for i, v in seeds.items()}

    vm = QVM(backend=args.backend, seed=args.seed)
    result = vm.run(program, r_inputs=seeds, lanes=lanes, trace=args.trace)

    if args.trace:
        from .asm import format_instr
        for pc, instr in result.trace:
            print(f"  {pc:04d}: {format_instr(instr, program)}", file=sys.stderr)

    print(f"backend={vm.backend.name} lanes={lanes} steps={result.steps}")
    if lowered is not None and lowered.result_regs:
        for i, (file, index) in enumerate(lowered.result_regs):
            if file == "r":
                print(f"result[{i}] = {result.r[index]}")
            else:
                r, eta = result.q[index]
                print(f"result[{i}] = state(r={r}, eta={eta})")
    else:
        touched = {instr.dst for instr in program.code
                   if instr.spec.dst == "r"}
        for index in sorted(touched):
            print(f"R{index} = {result.r[index]}")
    return 0


def _cmd_emit_arm64(args, program: Program) -> int:
    from .backends.arm64 import Arm64Backend, Arm64UnsupportedOp, assemble_text
    backend = Arm64Backend()
    try:
        text = backend.emit(program, with_library=not args.driver_only)
    except Arm64UnsupportedOp as exc:
        print(f"qvm: {exc}", file=sys.stderr)
        return 1
    _write(text, args.output)
    if args.check:
        ok, output = assemble_text(text)
        print(f"llvm-mc: {'ok' if ok else 'FAILED'}", file=sys.stderr)
        if output:
            print(output, file=sys.stderr)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
