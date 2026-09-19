#!/usr/bin/env python3
"""A QBNN layer from the APQB paper, compiled to APQB IR and run on the QVM.

The layer of Eq. (20)-(23) is::

    u      = W h + b                          classical affine path
    a      = P h + c                          APQB latent coordinates
    r, q   = tanh(a), sech(a)                 one pseudo qubit per hidden unit
    c_r    = J_r^T r ,  c_q = J_q^T q         correlation coupling
    h_next = act( u * (1 + lam_r c_r + lam_q c_q) )

which is exactly the split the Qubit Compiler is built around: the affine and
gating arithmetic stays classical, while every ``(r, q)`` pair lives in a
pseudo-qubit register.  One ``apqb.encode`` in ``latent`` mode produces both
coordinates at once, because r and eta are two views of a single APQB state.

The batch dimension maps onto the VM's SPMD lanes: one program, B samples.

    python3 examples/qbnn_layer.py
"""

import math
import random

from qubitbridge import Builder, Module, disassemble, lower_module
from qubitbridge.backends.arm64 import Arm64Backend, have_assembler
from qubitbridge.vm import QVM

IN_DIM, OUT_DIM, BATCH = 3, 2, 6
LAMBDA_R, LAMBDA_Q = 0.1, 0.1


def make_weights(seed: int = 0):
    rng = random.Random(seed)
    def matrix(rows, cols, scale=0.5):
        return [[rng.uniform(-scale, scale) for _ in range(cols)]
                for _ in range(rows)]
    return {
        "W": matrix(OUT_DIM, IN_DIM), "b": [rng.uniform(-0.1, 0.1)
                                            for _ in range(OUT_DIM)],
        "P": matrix(IN_DIM, IN_DIM), "c": [rng.uniform(-0.1, 0.1)
                                           for _ in range(IN_DIM)],
        "J_r": matrix(IN_DIM, OUT_DIM, 0.3),
        "J_q": matrix(IN_DIM, OUT_DIM, 0.3),
    }


def build_ir(w) -> Module:
    """Emit the layer as APQB IR, unrolled over the (small) hidden dimensions."""
    module = Module("qbnn")
    b = Builder(module, "layer")
    h = [b.arg(f"h{i}") for i in range(IN_DIM)]

    def affine(row, bias):
        acc = b.const(bias)
        for weight, value in zip(row, h):
            acc = b.emit("arith.add", acc,
                         b.emit("arith.mul", b.const(weight), value))
        return acc

    # Classical path.
    u = [affine(w["W"][j], w["b"][j]) for j in range(OUT_DIM)]

    # APQB path: one pseudo qubit per hidden unit carries both r and q.
    states = [b.encode(affine(w["P"][i], w["c"][i]), "latent")
              for i in range(IN_DIM)]
    r = [b.decode(state, "linear") for state in states]
    q = [b.emit("apqb.uncertainty", state) for state in states]

    outputs = []
    for j in range(OUT_DIM):
        gate = b.const(1.0)
        for i in range(IN_DIM):
            gate = b.emit("arith.add", gate, b.emit(
                "arith.mul", b.const(LAMBDA_R * w["J_r"][i][j]), r[i]))
            gate = b.emit("arith.add", gate, b.emit(
                "arith.mul", b.const(LAMBDA_Q * w["J_q"][i][j]), q[i]))
        outputs.append(b.emit("arith.tanh", b.emit("arith.mul", u[j], gate)))

    b.ret(*outputs)
    module.verify()
    return module


def reference(w, h):
    """Plain-Python QBNN layer, to check the compiled one against."""
    def affine(row, bias):
        return bias + sum(weight * value for weight, value in zip(row, h))

    u = [affine(w["W"][j], w["b"][j]) for j in range(OUT_DIM)]
    a = [affine(w["P"][i], w["c"][i]) for i in range(IN_DIM)]
    r = [math.tanh(value) for value in a]
    q = [2.0 / (math.exp(value) + math.exp(-value)) for value in a]

    out = []
    for j in range(OUT_DIM):
        gate = 1.0
        for i in range(IN_DIM):
            gate += LAMBDA_R * w["J_r"][i][j] * r[i]
            gate += LAMBDA_Q * w["J_q"][i][j] * q[i]
        out.append(math.tanh(u[j] * gate))
    return out


def main() -> None:
    w = make_weights()
    module = build_ir(w)
    lowered = lower_module(module)["layer"]

    print(f"QBNN layer {IN_DIM} -> {OUT_DIM}, batch {BATCH}")
    print(f"  IR ops         : {len(module.func('layer').body)}")
    print(f"  QVM instructions: {len(lowered.program.code)}")
    print(f"  constants       : {len(lowered.program.consts)}")

    rng = random.Random(42)
    batch = [[rng.uniform(-1.5, 1.5) for _ in range(IN_DIM)]
             for _ in range(BATCH)]
    seeds = {reg: [sample[i] for sample in batch]
             for i, (_, reg) in enumerate(lowered.arg_regs)}

    result = QVM(backend="auto").run(lowered.program, r_inputs=seeds, lanes=BATCH)
    got = [[result.r[reg][lane] for _, reg in lowered.result_regs]
           for lane in range(BATCH)]

    print("\n  sample            QVM output                reference")
    worst = 0.0
    for lane, sample in enumerate(batch):
        want = reference(w, sample)
        worst = max(worst, max(abs(g - t) for g, t in zip(got[lane], want)))
        pretty = ", ".join(f"{v:+.4f}" for v in sample)
        print(f"  [{pretty}]  "
              f"[{', '.join(f'{v:+.6f}' for v in got[lane])}]  "
              f"[{', '.join(f'{v:+.6f}' for v in want)}]")
    print(f"\n  largest deviation from the reference: {worst:.3e}")

    print("\n  first instructions:")
    for line in disassemble(lowered.program, addresses=True).splitlines()[2:10]:
        print("   ", line)

    if have_assembler():
        ok, output = Arm64Backend().verify(lowered.program)
        print(f"\n  AArch64 codegen assembles: {ok} {output}")


if __name__ == "__main__":
    main()
