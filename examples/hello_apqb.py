#!/usr/bin/env python3
"""The whole stack on one tiny program.

    x = 0.8
    y = 0.4
    z = x * y

Run it and watch the same computation descend from Python, through APQB IR and
the Qubit ISA, onto a virtual QPU -- and then onto AArch64 NEON.

    python3 examples/hello_apqb.py
"""

from qubitbridge import compile_source, disassemble, lower_module
from qubitbridge.backends.arm64 import Arm64Backend, have_assembler
from qubitbridge.vm import QVM

SOURCE = """
def mul2(x: unit, y: unit):
    return x * y
"""


def main() -> None:
    print(__doc__.splitlines()[0])

    print("\n=== 1. source (the 'unit' annotation promises x, y are in [-1,1])")
    print(SOURCE.strip())

    module, report = compile_source(SOURCE, module_name="hello")
    print("\n=== 2. APQB IR")
    print(module.to_text().rstrip())
    print("\n--- partition report")
    print(report.summary())

    lowered = lower_module(module)["mul2"]
    print("\n=== 3. Qubit ISA")
    print(disassemble(lowered.program, addresses=True).rstrip())

    print("\n=== 4. run on the QVM")
    seeds = {reg: value for (_, reg), value in zip(lowered.arg_regs, (0.8, 0.4))}
    result = QVM().run(lowered.program, r_inputs=seeds)
    got = result.r[lowered.result_index(0)][0]
    print(f"    z = {got!r}   (classical 0.8 * 0.4 = {0.8 * 0.4!r})")
    print(f"    exact match: {got == 0.8 * 0.4}")

    print("\n--- the pseudo-qubit registers at HALT\n    (Q0 was recycled by the allocator, so it holds the product)")
    for index in (0, 1):
        state = result.state(index)
        print(f"    Q{index}: r={state.r:+.6f}  eta={state.eta:.6f}  "
              f"theta={state.theta:.6f}  T={state.T:.6f}")

    print("\n=== 5. batch it: one program, four lanes")
    xs = [0.8, -0.5, 1.0, 0.0]
    ys = [0.4, 0.25, -1.0, 0.9]
    wide = QVM().run(lowered.program, r_inputs={0: xs, 1: ys}, lanes=4)
    print(f"    x    = {xs}")
    print(f"    y    = {ys}")
    print(f"    x*y  = {wide.r[lowered.result_index(0)]}")

    print("\n=== 6. AArch64 + NEON")
    asm = Arm64Backend().emit(lowered.program, with_library=False)
    print("\n".join(asm.splitlines()[8:24]))
    print("    ...")
    if have_assembler():
        ok, output = Arm64Backend().verify(lowered.program)
        print(f"\n    assembles with llvm-mc: {ok} {output}")
    else:
        print("\n    (install llvm-mc to assemble it here)")


if __name__ == "__main__":
    main()
