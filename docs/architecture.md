# Architecture

## Why a virtual QPU and not a simulator

A faithful `n`-qubit state vector needs `2^n` complex amplitudes.  At 16 bytes
each that is 16 MiB at 20 qubits, 16 GiB at 30, 64 GiB at 32, 16 TiB at 40.
Any design that starts there is finished before it begins on consumer hardware.

APQB does not reproduce a physical quantum state, so it does not pay that
price.  One pseudo qubit is a point on a circle -- `r`, `eta`, and the derived
`theta` -- which is three doubles.  A 32-register file is 768 bytes.  Cost is
linear in the number of pseudo qubits and in the number of instructions.

The honest boundary, which the paper states and this project repeats: expanding
*every* subset-product term `Phi_S(z) = prod_{i in S} z_i` over `n` states is
`2^n` terms in either formalism.  Truncating the interaction degree is not a
workaround bolted on afterwards, it is how the model is meant to be used:

| mode | meaning | cost |
|---|---|---|
| `K = 1` | independent pseudo qubits, no interaction | linear |
| `K = 2` | pairwise interaction (`QINT`, `QCORR`) | quadratic in the pairs used |
| `K = 3` | three-body terms | cubic in the terms used |
| `K = n` | the full subset basis | exponential -- special purpose only |

Nothing in the ISA forces a degree: `QPOW k` and chains of `QINT` build exactly
the terms you ask for and no others.

## The layers

```
source (Python subset today; C/Rust/Swift via LLVM or MLIR later)
   |
   |  qubitbridge.frontend      partition: classical vs APQB, with reasons
   v
APQB IR                          SSA, {f64, !apqb.state}, textual, verifiable
   |
   |  qubitbridge.lower         liveness, linear-scan allocation, immediate folding
   v
Qubit ISA                        Q0..Q31 / R0..R31, fixed 8-byte words, .qvm objects
   |
   +-- qubitbridge.vm           SPMD interpreter over a pluggable backend
   |      |
   |      +-- backends.portable stdlib reference -- the specification
   |      +-- backends.numpy    one array expression per kernel
   |
   +-- backends.arm64           AArch64 + NEON assembly (emit, not interpret)
```

Each boundary is a real interchange format, not just a Python object graph:
IR parses and prints, assembly assembles and disassembles, and programs
serialise to a `.qvm` object file with a magic header and a constant pool.
That is what makes it possible to add a backend without touching the compiler.

## Why the IR stops where it does

APQB IR knows about pseudo qubits and classical scalars.  It does **not** know
about registers, lanes, memory layout or instruction encodings.  Everything
target-specific lives below the `lower` pass, which is the reason a future
Metal, CUDA or dedicated APQB backend is an additive change: it consumes the
same ISA the portable interpreter does, and the differential tests that keep
the ARM64 backend honest apply to it unchanged.

If APQB silicon ever exists, the substitution is at the bottom of the stack:

```
APQB IR -> QVM -> {portable, numpy, NEON}        today
APQB IR -> QVM -> Q-NPU                          same IR, same ISA, new backend
```

## Execution model

* **SPMD.** One program, `n` independent lanes.  Every register holds a lane
  vector; a scalar register one vector, a pseudo-qubit register the `(r, eta)`
  pair.  The batch dimension of a neural-network layer maps straight onto lanes.
* **Straight line.** A module is a kernel that ends in `HALT`.  There are no
  branches in v1, so a kernel is fully unrolled by the compiler.  This is the
  same bargain a GPU shader makes, and it keeps the register allocator and every
  backend simple.  Lane-divergent control flow is the obvious next increment.
* **Deterministic randomness.** `QMEASURE ... sample` draws from a PRNG owned by
  the VM, not by the backend, so a seeded run reproduces bit-for-bit on every
  target.  This matters because the paper's own framing of APQB is as a
  *controllable* noise source: `r` near `+-1` is deterministic, `r` near 0 is
  maximally random, and `r^2 + T^2 = 1` is the trade-off between them.

## The classical/APQB split

The compiler's partitioner is deliberately conservative, and its rule is one
sentence: a chain of at least `min_degree` factors, each provably in `[-1, 1]`,
becomes an APQB region.

This is safe in a strong sense.  `apqb.mul` multiplies the `r` coordinates, and
`[-1, 1]` is closed under multiplication, so the APQB path returns bit-for-bit
the classical product.  Routing a region to the QVM can never change a result,
which is why the pass can run without asking.

Everything else -- addition, division, calls, anything the frontend cannot prove
bounded -- stays classical and is reported, with the line number and the reason.
`apqb(expr)` and `classical(expr)` override the decision per expression.

## Where the mathematics comes from

`qubitbridge/apqb.py` is the scalar reference, and it follows the revision-2
formalisation in [`tapiocaTakeshi/Qubit`](
https://github.com/tapiocaTakeshi/Qubit)'s `apqb_qbnn_v2.py`:

| paper | here |
|---|---|
| Eq. (4) `theta = (1/2) arccos r` | `APQBState.theta` |
| Eq. (6)-(7) `P(0) = (1+r)/2` | `probabilities`, encoding `prob` |
| Eq. (10) `H_Z(r)` | `entropy_z`, `QENT` |
| Eq. (12) `r = tanh a`, `q = sech a` | encoding `latent`, `QENC ... latent` |
| Eq. (14) `z = r + iq` | `APQBState.z`, the whole register model |
| Prop. 2 `Re(z^k) = T_k(r)` | `apqb.cheb_t` / `QPOW` |
| Eq. (17)-(18) `Phi_S(z)` | `apqb.interact` / `QINT` |
| Eq. (20)-(23) QBNN layer | `examples/qbnn_layer.py` |
| `T = |sin 2 theta|` | `uncertainty`, `QUNC` |

That correspondence is not decorative: `tests/test_apqb.py` checks the
identities directly, including `r^2 + T^2 = 1` and `Re(z^k) = T_k(r)`.
