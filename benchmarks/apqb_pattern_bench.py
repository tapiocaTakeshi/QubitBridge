#!/usr/bin/env python3
"""What does the APQB *pattern itself* cost, apart from backend choice?

qbnn_layer_bench.py and lean_kernel_bench.py both compare backends (naive
Python vs the portable and numpy QVM backends) on a fixed program. This
script asks a different question: for the *same* classical result -- a
chain product of k values in [-1, 1] -- how does the QVM instruction
pattern the frontend's partitioner emits for an APQB region

    QENC x0 -> Q0, QENC x1 -> Q1, ..., QMUL Q0 Q1 -> Q0, ..., QDEC Q0 -> out

compare to the plain classical pattern it would otherwise emit

    MUL x0 x1 -> out, MUL out x2 -> out, ...

Both compute bit-for-bit the same product (docs/architecture.md's "safe by
construction" argument: apqb.mul multiplies the r coordinates and [-1, 1] is
closed under multiplication) -- this only measures what encoding into pseudo
qubits costs on top of that, at the instruction level.

    python3 benchmarks/apqb_pattern_bench.py
    python3 benchmarks/apqb_pattern_bench.py --big     # also N=1,000,000

Findings on one x86-64 Linux host (4 cores, CPython 3.11, NumPy 2.4; min of
several reps per point). The APQB pattern for a chain of k factors is 2k+1
instructions (k encodes, k-1 muls, 1 decode) against the classical pattern's
k (k-1 muls, one fewer instruction since there is nothing to decode) -- a bit
more than double the instruction count at every k:

    backend    k    N        classical (ms)  apqb (ms)  apqb/classical
    numpy      4    10,000          77.7        79.7          1.03x
    numpy      4    500,000       5401.3      5962.0          1.10x
    numpy      16   10,000          83.9        85.3          1.02x
    numpy      16   500,000       8869.7      9337.7          1.05x
    portable   4    10,000          14.7        29.4          2.00x
    portable   16   10,000          23.4        90.5          3.87x

On numpy, the ratio sits close to 1.0 regardless of k or N: apqb.encode /
apqb.mul / apqb.decode are each one vectorized array pass, so the extra
encode/decode traffic is nearly free next to the array-sized work every
instruction already does. On portable it lands close to the *instruction*
ratio (9/4 = 2.25 at k=4, 33/16 = 2.06 at k=16) -- a bit above it, since
qenc/qmul do a `sech`/`sqrt` a plain `arith.mul` doesn't -- which is the
honest, un-inflated cost of the extra instructions: a first version of this
backend paid a much larger, avoidable tax here (up to 36x) from constructing
an APQBState object per lane per instruction; qubitbridge/backends/portable.py
now works on raw (r, eta) floats instead, so what's left is real arithmetic,
not object overhead.

Either way, this cost is the one the compiler's partitioner spends
correctness margin against -- worth paying only where a pseudo-qubit
register genuinely earns its place (a `QCORR`, `QUNC`, `QGATE`, or a
Chebyshev feature that has no cheaper classical equivalent), never for a
plain product on its own. See `qubitbridge/frontend.py`'s module docstring
for the partitioner's actual rule.
"""

from __future__ import annotations

import argparse
import random
import time

from qubitbridge.ir import Builder, Module
from qubitbridge.lower import Lowered, lower_module
from qubitbridge.vm import QVM


def classical_chain(k: int) -> Lowered:
    """out = x0 * x1 * ... * x{k-1}, as plain arith.mul."""
    module = Module("classical")
    b = Builder(module, "f")
    xs = [b.arg(f"x{i}") for i in range(k)]
    acc = xs[0]
    for x in xs[1:]:
        acc = b.emit("arith.mul", acc, x)
    b.ret(acc)
    module.verify()
    return lower_module(module)["f"]


def apqb_chain(k: int) -> Lowered:
    """The same product, routed through the APQB QENC/QMUL/QDEC pattern."""
    module = Module("apqb")
    b = Builder(module, "f")
    xs = [b.arg(f"x{i}") for i in range(k)]
    states = [b.encode(x, "linear") for x in xs]
    acc = states[0]
    for s in states[1:]:
        acc = b.emit("apqb.mul", acc, s)
    b.ret(b.decode(acc, "linear"))
    module.verify()
    return lower_module(module)["f"]


def bench(fn, reps: int) -> float:
    fn()
    return min(_time(fn) for _ in range(reps))


def _time(fn) -> float:
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def check_agreement(classical: Lowered, apqb: Lowered, k: int) -> None:
    rng = random.Random(0)
    xs = [rng.uniform(-1.0, 1.0) for _ in range(k)]
    seeds = {reg: xs[i] for i, (_, reg) in enumerate(classical.arg_regs)}
    c = QVM().run(classical.program, r_inputs=seeds).r[classical.result_index(0)][0]
    a = QVM().run(apqb.program, r_inputs=seeds).r[apqb.result_index(0)][0]
    assert abs(c - a) < 1e-12, f"k={k}: classical={c!r} apqb={a!r} disagree"


def run_case(k: int, sizes: list[int], backend: str) -> None:
    classical = classical_chain(k)
    apqb = apqb_chain(k)
    check_agreement(classical, apqb, k)

    print(f"\n=== k={k} factors -- classical: {len(classical.program.code)} instrs, "
          f"apqb: {len(apqb.program.code)} instrs -- backend={backend} ===")
    print(f"{'N':>9} {'classical (ms)':>15} {'apqb (ms)':>12} {'apqb/classical':>15}")

    rng = random.Random(1)
    for n in sizes:
        samples = [[rng.uniform(-1.0, 1.0) for _ in range(k)] for _ in range(n)]
        seeds = {reg: [s[i] for s in samples]
                 for i, (_, reg) in enumerate(classical.arg_regs)}
        reps = 3 if n < 100_000 else 1

        t_classical = bench(
            lambda: QVM(backend=backend).run(classical.program, r_inputs=seeds, lanes=n), reps)
        t_apqb = bench(
            lambda: QVM(backend=backend).run(apqb.program, r_inputs=seeds, lanes=n), reps)

        print(f"{n:>9} {t_classical*1000:>15.3f} {t_apqb*1000:>12.3f} "
              f"{t_apqb/t_classical:>14.2f}x")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--big", action="store_true",
                        help="also run N=1,000,000 (slow)")
    args = parser.parse_args()

    sizes = [100, 10_000, 500_000] + ([1_000_000] if args.big else [])
    for k in (4, 16):
        for backend in ("portable", "numpy"):
            run_case(k, sizes, backend)


if __name__ == "__main__":
    main()
