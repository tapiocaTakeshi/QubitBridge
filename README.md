# QubitBridge -- the Qubit Virtual Machine

A three-layer toolchain that runs **APQB** (Adjustable Pseudo Quantum Bit)
computation on hardware you already own.

```
  C / Rust / Python / Swift
             |
      Qubit Compiler          which parts are worth encoding as pseudo qubits?
        /         \
   classical      APQB IR     %q2 = apqb.interact %q0, %q1
       |             |
       |     Qubit Virtual Machine        virtual QPU: Q0..Q31 = {r, eta, theta}
       |             |
       +------+------+
              |
    +---------+---------+---------+
    |         |         |         |
  portable  numpy    AArch64    (future)
  (stdlib)  (SIMD)   + NEON      GPU / NPU / Q-NPU
```

## What this is, and what it is not

It is **not** a way to turn a PC into a quantum computer, and it is not a
state-vector simulator: there is no `2^n` amplitude array anywhere in it.  A
20-qubit state vector is 16 MiB, 30 qubits is 16 GiB, 40 qubits is 16 TiB --
that wall is exactly what this design avoids by not standing in front of it.

It **is** a compiler and virtual machine for the classical, quantum-*inspired*
APQB encoding of the QBNN/APQB paper.  An APQB is a point on a circle,

```
z = r + i*eta = e^{i*2*theta},     r^2 + eta^2 = 1
r     = cos(2 theta)     statistical correlation, in [-1, 1]
eta   = sin(2 theta)     coherence / uncertainty (the paper's q)
T     = |eta|            the AI-temperature analogue, with r^2 + T^2 = 1
```

so a pseudo qubit costs three doubles, not an exponential amplitude vector, and
the machine's cost is linear in the number of pseudo qubits.  No quantum
speedup is claimed or implied.

The honest caveat the paper itself makes applies here too: expanding *all*
`2^n` subset-product terms is exponential in either formalism.  The QVM's answer
is the same as the paper's -- bound the interaction degree (`apqb.pow`/`QPOW`
with a small `k`, chains of `QINT`), and keep the register file small.

## Install

```sh
pip install -e .            # stdlib only
pip install -e '.[numpy]'   # plus the vectorised backend
```

## Two minutes

```python
import qubitbridge as qb

# 'unit' promises the argument is in [-1, 1], which is what lets the
# compiler route the product through the APQB path.
qb.run("def f(x: unit, y: unit):\n    return x * y", 0.8, 0.4)
# [[0.32000000000000006]]   -- bit-identical to the classical product
```

```sh
qvm compile kernel.py --report     # source  -> APQB IR (+ what went where)
qvm lower   kernel.py              #         -> Qubit assembly
qvm run     kernel.py -i R0=0.8 -i R1=0.4
qvm emit-arm64 kernel.py --check   #         -> AArch64/NEON, assembled
qvm info                           # the ISA, the backends, the toolchain
```

Or drive the layers directly:

```python
from qubitbridge import compile_source, lower_module, QVM

module, report = compile_source(open("kernel.py").read())
print(module.to_text())            # APQB IR
print(report.summary())            # classical / APQB split, with reasons

lowered = lower_module(module)["kernel"]
result = QVM(backend="auto").run(lowered.program,
                                 r_inputs={0: [0.8, -0.5], 1: [0.4, 0.25]},
                                 lanes=2)                 # SPMD: 2 samples
print(result.r[lowered.result_index(0)])
```

## The three layers

| layer | module | what it is |
|---|---|---|
| **APQB IR** | `qubitbridge.ir` | SSA, two types (`f64`, `!apqb.state`), MLIR-style text that parses and prints round-trip |
| **Qubit ISA + QVM** | `qubitbridge.isa`, `.vm` | 32 pseudo-qubit + 32 scalar registers, fixed 8-byte instruction words, a `.qvm` object format, SPMD lanes |
| **Backends** | `qubitbridge.backends` | `portable` (stdlib reference), `numpy` (vectorised), `arm64` (AArch64 + NEON code emitter) |

Where a classical CPU's verbs are `ADD / MUL / MOV / CMP`, the virtual QPU's are:

```
QENC   QDEC     classical scalar <-> pseudo qubit
QROT            theta -> theta + phi
QINT            z_a * z_b -- the subset product, i.e. theta addition
QMUL            r_a * r_b -- correlation product, stays canonical
QCORR           cos(2(theta_a - theta_b))
QUNC   QENT     |eta| (the temperature T), and H_Z in bits
QGATE           the QBNN coupling J, applied in latent space
QMEASURE        expectation, or a seeded Bernoulli draw
```

`qvm info` prints the full table; `docs/isa.md` documents the encoding.

## Hybrid partitioning

Not everything should become a pseudo qubit.  Opening a file or printing a
string gains nothing from an APQB encoding, so the compiler splits the program
and says why:

```
APQB regions: 1
ops: 4 APQB / 3 classical (57% APQB)
  line 2: degree-3 chain -> QVM (all factors provably in [-1, 1])
  note: line 5: degree-2 product kept classical; factors are not provably in [-1, 1]
```

A value is *bounded* if it is a literal in `[-1, 1]`, an argument annotated
`unit`, a `tanh` result, or a product of bounded values.  A chain of at least
`min_degree` bounded factors becomes an APQB region.  Because `[-1, 1]` is
closed under multiplication and `apqb.mul` multiplies the `r` coordinates, the
APQB path returns **exactly** the classical product -- routing a region through
the QVM never changes the answer.  Override with `apqb(expr)` or
`classical(expr)`.

## Is the ARM64 backend real?

Yes, and the test-suite proves it rather than asserting it.  Emitted assembly is
assembled with `llvm-mc`, then -- on an ARM host, or through
`aarch64-linux-gnu-gcc` + `qemu-aarch64` -- linked against a C harness, executed,
and compared lane by lane against the portable Python backend.  Every opcode
agrees to within `3.3e-16`, and the QBNN layer in `examples/qbnn_layer.py`
matches its plain-Python reference to `5.6e-17` when run as real AArch64
machine code.

The algebraic kernels (`qb_interact`, `qb_correlate`, `qb_mul`, `qb_normalize`,
...) are NEON, two lanes per `.2d` instruction with a scalar tail.  The
transcendental ones (latent encode/decode, rotate, gate, entropy) are scalar
loops calling libm, because `tanh`/`sin`/`log2` have no single NEON instruction.

## Examples

```sh
python3 examples/hello_apqb.py    # x*y down all four layers, plus the emitted NEON
python3 examples/qbnn_layer.py    # a QBNN layer from the paper, batched over SPMD lanes
```

## Tests

```sh
python3 -m unittest discover -s tests -t . -v
```

155 tests, no third-party dependencies required.  Backend-dependent cases skip
cleanly when NumPy or an AArch64 toolchain is absent.

## Status and limitations

* **Straight-line only.** A module is a kernel ending in `HALT`; there are no
  branches yet, so the frontend unrolls.  Lane-divergent control flow is future
  work.
* **32 + 32 registers, no spilling.** Running out is a clear error asking you to
  split the kernel, not a silent miscompile.
* **The ARM64 driver is register-to-register.** `LOAD`/`STORE`/`QSTORE` and
  sampled `QMEASURE` have no ARM lowering and say so.
* Backends for Metal, CUDA and an eventual APQB ASIC ("Q-NPU") are not written;
  the point of stopping the IR above the register file is that they only need a
  new backend, not a new compiler.

## Relation to the Qubit repository

The APQB/QBNN mathematics comes from [`tapiocaTakeshi/Qubit`](
https://github.com/tapiocaTakeshi/Qubit) (`apqb_qbnn_v2.py`), which is the
paper's revision-2 formalisation in PyTorch.  This repository is the systems
half: the same algebra as an instruction set.  See `docs/architecture.md`.
