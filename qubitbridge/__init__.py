"""QubitBridge -- the Qubit Virtual Machine.

A three-layer stack that runs APQB (Adjustable Pseudo Quantum Bit) computation
on ordinary hardware::

    Python subset  --[ frontend ]-->  APQB IR
    APQB IR        --[ lower    ]-->  Qubit ISA
    Qubit ISA      --[ vm       ]-->  portable / numpy execution
                   --[ arm64    ]-->  AArch64 + NEON assembly

This is quantum-*inspired* classical computation, not quantum simulation: there
is no 2^n state vector anywhere in it.  The APQB state is the {r, eta, theta}
triple of the QBNN/APQB paper, and the VM's cost is linear in the number of
pseudo qubits.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .apqb import APQBState, decode, encode, interact
from .asm import assemble, disassemble
from .backends import available_backends, get_backend
from .frontend import compile_function, compile_source
from .ir import Builder, Module, parse_module
from .isa import Op, Program
from .lower import lower_func, lower_module
from .vm import QVM, RunResult

__all__ = [
    "__version__",
    "APQBState", "encode", "decode", "interact",
    "Module", "Builder", "parse_module",
    "Program", "Op", "assemble", "disassemble",
    "lower_func", "lower_module",
    "QVM", "RunResult",
    "get_backend", "available_backends",
    "compile_source", "compile_function",
    "run",
]


def run(source: str, *args: float, lanes: int = 1, backend: str = "auto",
        seed: int = 0) -> list:
    """Compile a Python-subset function and run it, in one call.

    >>> run("def f(x: unit, y: unit):\\n    return x * y", 0.8, 0.4)
    [[0.32000000000000006]]
    """
    module, _ = compile_source(source)
    lowered = lower_module(module)[module.funcs[0].name]
    seeds = {}
    for value, (file, index) in zip(args, lowered.arg_regs):
        if file != "r":
            raise ValueError("state-typed arguments cannot be seeded from run()")
        seeds[index] = value
    result = QVM(backend=backend, seed=seed).run(
        lowered.program, r_inputs=seeds, lanes=lanes)
    return [result.r[index] if file == "r" else result.q[index]
            for file, index in lowered.result_regs]
