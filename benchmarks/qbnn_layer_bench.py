#!/usr/bin/env python3
"""Does batching a QBNN-shaped layer onto the QVM actually run faster?

Builds a QBNN layer (Eq. 20-23 shape: classical affine + APQB latent-gate
coupling) directly with the IR Builder for a given (in_dim, out_dim), then
times three ways of computing the same forward pass over a batch of N
samples:

    naive     a plain Python loop calling a scalar reference once per sample
    portable  one QVM.run() call with lanes=N, the stdlib backend
    numpy     one QVM.run() call with lanes=N, the numpy backend

This isolates where any speedup actually comes from: batching plus
vectorization (numpy), not "quantum" anything -- the APQB encoding is the
same amount of floating point arithmetic as the classical formula, just
carried through pseudo-qubit registers. Run it yourself:

    python3 benchmarks/qbnn_layer_bench.py
    python3 benchmarks/qbnn_layer_bench.py --big     # also try N=1,000,000

Findings on one x86-64 Linux host (4 cores, CPython 3.11, NumPy 2.4), taken
as the min of several reps per point -- expect different numbers on your
machine, but the same *shape* of result:

    8x8 layer, 833 QVM instructions:
        N          naive(ms)  portable(ms)  numpy(ms)  numpy/naive  numpy/portable
        1,000          24.9          23.2        9.2         2.7x            2.5x
        10,000        253.1         233.1       87.7         2.9x            2.7x
        100,000     2,630.0       2,768.7      927.7         2.8x            3.0x

The numpy backend beats both the naive per-sample loop and the portable
backend once there is enough arithmetic per sample (larger in_dim/out_dim)
and enough lanes (>= ~1,000) to amortize each instruction's per-call
overhead: array allocation for numpy, CPython bytecode dispatch either way.
Below that -- a tiny layer, or a small batch -- numpy can be the *slowest*
of the three; see lean_kernel_bench.py for a kernel small enough that numpy
never wins, at any scale. Portable and naive stay close to each other
throughout (within about 20%) because both are, fundamentally, a Python
loop doing scalar arithmetic; numpy is the odd one out in both directions,
winning big when there's enough work per instruction and losing when there
isn't. There is no backend that is "the fast one" independent of the
workload -- see docs/backends.md's Performance section for the fuller
picture, including apqb_pattern_bench.py's result.
"""

from __future__ import annotations

import argparse
import math
import random
import time

from qubitbridge.ir import Builder, Module
from qubitbridge.lower import lower_module
from qubitbridge.vm import QVM

LAMBDA_R, LAMBDA_Q = 0.1, 0.1


def make_weights(in_dim: int, out_dim: int, seed: int = 0) -> dict:
    rng = random.Random(seed)

    def matrix(rows, cols, scale=0.5):
        return [[rng.uniform(-scale, scale) for _ in range(cols)] for _ in range(rows)]

    return {
        "W": matrix(out_dim, in_dim), "b": [rng.uniform(-0.1, 0.1) for _ in range(out_dim)],
        "P": matrix(in_dim, in_dim), "c": [rng.uniform(-0.1, 0.1) for _ in range(in_dim)],
        "J_r": matrix(in_dim, out_dim, 0.3), "J_q": matrix(in_dim, out_dim, 0.3),
    }


def build_ir(in_dim: int, out_dim: int, w: dict) -> Module:
    """The Eq. (20)-(23) layer shape, unrolled directly with the IR Builder."""
    module = Module("bench")
    b = Builder(module, "layer")
    h = [b.arg(f"h{i}") for i in range(in_dim)]

    def affine(row, bias, inputs):
        acc = b.const(bias)
        for weight, value in zip(row, inputs):
            acc = b.emit("arith.add", acc, b.emit("arith.mul", b.const(weight), value))
        return acc

    u = [affine(w["W"][j], w["b"][j], h) for j in range(out_dim)]
    states = [b.encode(affine(w["P"][i], w["c"][i], h), "latent") for i in range(in_dim)]
    r = [b.decode(s, "linear") for s in states]
    q = [b.emit("apqb.uncertainty", s) for s in states]

    outputs = []
    for j in range(out_dim):
        gate = b.const(1.0)
        for i in range(in_dim):
            gate = b.emit("arith.add", gate,
                          b.emit("arith.mul", b.const(LAMBDA_R * w["J_r"][i][j]), r[i]))
            gate = b.emit("arith.add", gate,
                          b.emit("arith.mul", b.const(LAMBDA_Q * w["J_q"][i][j]), q[i]))
        outputs.append(b.emit("arith.tanh", b.emit("arith.mul", u[j], gate)))

    b.ret(*outputs)
    module.verify()
    return module


def reference(w: dict, in_dim: int, out_dim: int, h: list[float]) -> list[float]:
    def affine(row, bias):
        return bias + sum(weight * value for weight, value in zip(row, h))

    u = [affine(w["W"][j], w["b"][j]) for j in range(out_dim)]
    a = [affine(w["P"][i], w["c"][i]) for i in range(in_dim)]
    r = [math.tanh(v) for v in a]
    q = [2.0 / (math.exp(v) + math.exp(-v)) for v in a]

    out = []
    for j in range(out_dim):
        gate = 1.0
        for i in range(in_dim):
            gate += LAMBDA_R * w["J_r"][i][j] * r[i]
            gate += LAMBDA_Q * w["J_q"][i][j] * q[i]
        out.append(math.tanh(u[j] * gate))
    return out


def bench(fn, reps: int) -> float:
    fn()  # warm up
    return min(_time(fn) for _ in range(reps))


def _time(fn) -> float:
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def run_case(in_dim: int, out_dim: int, label: str, sizes: list[int]) -> None:
    w = make_weights(in_dim, out_dim)
    lowered = lower_module(build_ir(in_dim, out_dim, w))["layer"]

    print(f"\n=== {label}: layer {in_dim} -> {out_dim}, "
          f"{len(lowered.program.code)} QVM instructions ===")
    print(f"{'N':>9} {'naive (ms)':>12} {'portable (ms)':>14} {'numpy (ms)':>12} "
          f"{'numpy/naive':>13} {'numpy/portable':>16}")

    rng = random.Random(1)
    for n in sizes:
        batch = [[rng.uniform(-1.5, 1.5) for _ in range(in_dim)] for _ in range(n)]
        seeds = {reg: [sample[i] for sample in batch]
                 for i, (_, reg) in enumerate(lowered.arg_regs)}
        reps = 3 if n < 100_000 else 1

        t_naive = bench(lambda: [reference(w, in_dim, out_dim, s) for s in batch], reps)
        t_portable = bench(
            lambda: QVM(backend="portable").run(lowered.program, r_inputs=seeds, lanes=n), reps)
        t_numpy = bench(
            lambda: QVM(backend="numpy").run(lowered.program, r_inputs=seeds, lanes=n), reps)

        print(f"{n:>9} {t_naive*1000:>12.3f} {t_portable*1000:>14.3f} {t_numpy*1000:>12.3f} "
              f"{t_naive/t_numpy:>12.2f}x {t_portable/t_numpy:>15.2f}x")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--big", action="store_true",
                        help="also run N=1,000,000 (slow: several minutes)")
    args = parser.parse_args()

    sizes = [1, 10, 100, 1_000, 10_000, 100_000] + ([1_000_000] if args.big else [])
    run_case(3, 2, "small", sizes)
    run_case(8, 8, "larger (AI/matrix-calc scale)", sizes)


if __name__ == "__main__":
    main()
