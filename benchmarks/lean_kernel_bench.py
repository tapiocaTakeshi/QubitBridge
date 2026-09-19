#!/usr/bin/env python3
"""The sharper version of qbnn_layer_bench.py's result: a *lean* kernel.

Nine instructions (encode, interact, rotate, gate, correlate, uncertainty,
entropy, decode) run at increasing lane counts N, compared against calling
the equivalent qubitbridge.apqb scalar functions directly in a Python loop.

    python3 benchmarks/lean_kernel_bench.py
    python3 benchmarks/lean_kernel_bench.py --big     # also try N=1,000,000

Findings on one x86-64 Linux host (4 cores, CPython 3.11, NumPy 2.4), min of
several reps per point:

    N            naive(ms)  portable(ms)  numpy(ms)  portable/naive  numpy/naive
    1,000            3.9          2.5          7.8        0.63x          2.01x
    100,000        409.4        311.6        810.5        0.76x          1.98x
    1,000,000     4208.7       6312.2      13446.2        1.50x          3.20x

For a kernel this short, **neither QVM backend beats a tight, already
optimized scalar Python loop** at any scale tested, up to a million lanes --
but the two backends are not interchangeable either: portable is 2x-3.2x
faster than numpy here, not the other way around. Each numpy instruction
still allocates a fresh N-element array with no fusion across instructions,
so nine instructions means nine full-array passes at numpy's per-call
overhead; portable's plain Python arithmetic on lists has no such per-call
tax for a kernel this size, so it wins. (Portable's own APQB kernels used to
add an *extra*, avoidable tax on top of that -- constructing an APQBState
object per lane per instruction -- fixed in
qubitbridge/backends/portable.py; that used to make numpy look better than
it is here by comparison.)

Read this together with qbnn_layer_bench.py's result, not instead of it:
there numpy wins clearly (2.7x-2.9x over naive Python, 2.5x-3x over portable)
because the 8x8 layer packs O(in_dim*out_dim) arithmetic into each
instruction, enough to amortize numpy's per-call cost. Here there isn't.
Batching pays off when there is enough work *per instruction*, not simply
because the lane count is large -- and which backend wins depends on that
too. Measure your own workload with these scripts rather than assuming
either backend, or `backend="auto"`'s numpy-if-available default, is faster.
"""

from __future__ import annotations

import argparse
import random
import time

from qubitbridge import apqb
from qubitbridge.asm import assemble
from qubitbridge.vm import QVM

PROGRAM = """
        QENC      Q0, R0, latent
        QENC      Q1, R1, latent
        QINT      Q2, Q0, Q1
        QROT      Q2, Q2, 0.25
        QGATE     Q3, Q2, Q0, 0.3
        QCORR     R2, Q2, Q3
        QUNC      R3, Q3
        QENT      R4, Q3
        QDEC      R5, Q3, linear
        HALT
"""


def scalar_reference(x: float, y: float) -> tuple[float, float, float, float]:
    """The same nine steps as PROGRAM, called directly against apqb.py."""
    q0 = apqb.encode(x, "latent")
    q1 = apqb.encode(y, "latent")
    q2 = apqb.rotate(apqb.interact(q0, q1), 0.25)
    q3 = apqb.gate(q2, q0, 0.3)
    return (apqb.correlate(q2, q3), apqb.uncertainty(q3),
           apqb.entropy_z(q3), apqb.decode(q3, "linear"))


def bench(fn, reps: int) -> float:
    fn()
    return min(_time(fn) for _ in range(reps))


def _time(fn) -> float:
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--big", action="store_true",
                        help="also run N=1,000,000 (slow: about a minute)")
    args = parser.parse_args()

    prog = assemble(PROGRAM)
    rng = random.Random(0)
    sizes = [1, 10, 100, 1_000, 10_000, 100_000] + ([1_000_000] if args.big else [])

    print(f"9-instruction kernel: {len(prog.code)} ops, "
          f"scalar Python loop vs QVM batched execution")
    print(f"{'N':>9} {'naive (ms)':>12} {'portable (ms)':>14} {'numpy (ms)':>12} "
          f"{'numpy/naive':>13} {'numpy/portable':>16}")
    for n in sizes:
        xs = [rng.uniform(-1.2, 1.2) for _ in range(n)]
        ys = [rng.uniform(-1.2, 1.2) for _ in range(n)]
        seeds = {0: xs, 1: ys}
        reps = 3 if n < 100_000 else 1

        t_naive = bench(lambda: [scalar_reference(x, y) for x, y in zip(xs, ys)], reps)
        t_portable = bench(
            lambda: QVM(backend="portable").run(prog, r_inputs=seeds, lanes=n), reps)
        t_numpy = bench(lambda: QVM(backend="numpy").run(prog, r_inputs=seeds, lanes=n), reps)

        print(f"{n:>9} {t_naive*1000:>12.3f} {t_portable*1000:>14.3f} {t_numpy*1000:>12.3f} "
              f"{t_naive/t_numpy:>12.2f}x {t_portable/t_numpy:>15.2f}x")


if __name__ == "__main__":
    main()
