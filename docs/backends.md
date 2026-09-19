# Backends

A backend implements the APQB kernels over lane vectors.  The VM only moves
those vectors between registers; it never inspects them, so a backend chooses
its own representation.

| backend | vectors | role |
|---|---|---|
| `portable` | `list[float]` | the reference semantics; stdlib only |
| `numpy` | `ndarray` (float64) | one array expression per kernel |
| `arm64` | -- | AArch64 + NEON **code emitter** (`executes = False`) |

```python
from qubitbridge.backends import get_backend, available_backends
available_backends()          # ['portable', 'numpy']
get_backend("auto")           # numpy if importable, else portable
```

`QVM(backend=...)` refuses a backend whose `executes` is False and says to use
its emit API instead.

## The contract

Interchangeability is *numerical*, not merely structural.
`tests/test_vm.py::TestBackendAgreement` runs a program touching every opcode on
four lanes through every execution backend and requires agreement to 11 decimal
places, and `tests/test_arm64.py` does the same against real machine code.

Two rules make that achievable:

* Randomness belongs to the VM, not the backend.  `QMEASURE ... sample` receives
  pre-drawn uniforms, so a seeded run is identical everywhere.
* Clamping bounds are exactly representable (`R_MAX = nextafter(1, 0)`), never
  "multiply by `1 - eps`", which no two implementations round alike.

## The ARM64 backend

### What it emits

1. A **kernel library**: one routine per opcode over lane arrays of `double`.
   The algebraic kernels are NEON, two lanes per `.2d` instruction plus a scalar
   tail for an odd lane count:

   ```
   qb_interact, qb_mul, qb_correlate, qb_uncertainty, qb_normalize,
   qb_encode_linear, qb_encode_prob, qb_decode_prob, qb_qload, qb_identity,
   qb_copy, qb_add, qb_sub, qb_fmul, qb_fdiv, qb_neg, qb_scale, qb_offset,
   qb_splat
   ```

   The transcendental kernels are scalar loops calling libm, because `tanh`,
   `sin`, `atanh` and `log2` have no single NEON instruction:

   ```
   qb_encode_latent, qb_encode_angle, qb_decode_latent, qb_decode_angle,
   qb_rotate, qb_gate, qb_entropy, qb_tanh
   ```

2. A **driver** `qb_prog_<name>`, which walks the program once and calls the
   kernels with computed addresses.

### Driver ABI

```c
void qb_prog_<name>(double *qbase, double *rbase, const double *cpool,
                    double *scratch, uint64_t n);
```

With `stride = n * 8` bytes:

```
Q[i].r    at qbase + (2i)   * stride        R[i] at rbase + i * stride
Q[i].eta  at qbase + (2i+1) * stride        cpool[imm] = the constant pool
scratch                                     2 lane vectors, for folded
                                            immediates and QPOW staging
```

`qbase` must hold `64` lane vectors, `rbase` `32`, `scratch` `2`.  Pseudo-qubit
registers should be initialised to `|0>` (`r = 1`, `eta = 0`) before the call.

The driver keeps its bases in `x19`-`x25`, all callee-saved and restored, and
passes `n` in the x register after the last pointer.

### Not lowered

| opcode | why |
|---|---|
| `LOAD`, `STORE`, `QSTORE` | the driver has no data segment; keep values in registers |
| `QMEASURE ... sample` | needs the VM's seeded PRNG; draw the uniforms on the host and pass them in a scalar register |

Both raise `Arm64UnsupportedOp` with that explanation rather than emitting
something that silently differs.

### How it is verified

```python
from qubitbridge.backends.arm64 import Arm64Backend
ok, output = Arm64Backend().verify(program)     # assembles with llvm-mc
```

and, when a toolchain is available, the stronger check: `tests/armrun.py`
compiles the emitted assembly with a C harness for aarch64 and runs it -- on an
ARM host directly, otherwise under `qemu-aarch64` -- then compares every
register, every lane, against the portable backend.

```sh
apt-get install -y gcc-aarch64-linux-gnu qemu-user   # to run those tests on x86
python3 -m unittest tests.test_arm64 -v
```

Measured agreement: every opcode within `3.3e-16`, and the QBNN layer example
within `5.6e-17`.

## Performance: does batching actually help?

`benchmarks/` answers this by measurement rather than assertion, the same way
`tests/test_arm64.py` answers "is the emitted code really AArch64" by actually
running it. Each script compares three ways of computing the same forward
pass over a batch of `N` samples: a plain Python loop calling a scalar
reference once per sample (`naive`), one `QVM.run()` call with `lanes=N` on
the `portable` backend, and the same on the `numpy` backend.

```sh
python3 benchmarks/qbnn_layer_bench.py     # a QBNN-shaped layer, two sizes
python3 benchmarks/lean_kernel_bench.py    # a lean 9-instruction kernel
```

The honest answer is **it depends on the shape of the workload, and there is
no backend that wins everywhere** -- not even numpy. The three scripts were
chosen to show that plainly rather than to lead to one recommendation.
Measured on one x86-64 Linux host (4 cores, CPython 3.11, NumPy 2.4; min of
several reps per point):

| workload | N | naive | portable | numpy | fastest |
|---|---|---|---|---|---|
| 8x8 layer, 833 instrs | 1,000 | 24.9 ms | 23.2 ms | 9.2 ms | **numpy**, 2.5-2.7x |
| 8x8 layer, 833 instrs | 100,000 | 2630.0 ms | 2768.7 ms | 927.7 ms | **numpy**, 2.8-3.0x |
| lean kernel, 9 instrs | 1,000 | 3.9 ms | 2.5 ms | 7.8 ms | **naive**, 1.6-3.2x |
| lean kernel, 9 instrs | 1,000,000 | 4208.7 ms | 6312.2 ms | 13446.2 ms | **naive**, 1.5-3.2x |
| APQB pattern (k=16 product), portable | 500,000 | -- | classical 2287 ms | apqb 7321 ms | classical, 3.2x |
| APQB pattern (k=16 product), numpy | 500,000 | -- | classical 8870 ms | apqb 9338 ms | classical, 1.05x |

Three things are true at once, and none of them is "always pick numpy":

* **Which backend wins depends on how much arithmetic is packed into each
  instruction, not on lane count.** The 8x8 layer does `O(in_dim * out_dim)`
  multiply-adds per sample; enough of that per instruction lets one
  vectorized numpy pass over `N` samples beat both a naive Python loop and
  the portable backend, once `N` is in the thousands. The lean kernel (few
  instructions, each cheap) doesn't have enough work per instruction to
  amortize numpy's per-instruction cost -- a fresh `N`-element array
  allocation and a full read/write pass, with no fusion across instructions
  -- so numpy *loses* to both alternatives here, at every scale tested, up to
  a million lanes. `benchmarks/apqb_pattern_bench.py` isolates the same
  effect one level down: it compares the APQB `QENC`/`QMUL`/`QDEC` pattern
  against the plain classical `MUL` pattern for the *same* product, and
  finds close to zero extra cost on numpy (every op is already a vectorized
  array pass) but a real, instruction-count-proportional cost on portable.
* **The portable backend is a legitimate choice for small or lean workloads,
  not just a slow reference implementation.** It beat numpy outright in the
  lean-kernel case above. It used to be worse than this table shows: an
  earlier version constructed an `APQBState` object per lane on every APQB
  instruction, which made the APQB pattern cost 8x-36x a plain classical one
  instead of the ~2x-4x its extra instructions actually cost; fixing that
  (working on raw `(r, eta)` floats instead, in
  `qubitbridge/backends/portable.py`) is what makes the comparison in this
  table fair.
* **Neither the APQB encoding nor "quantum" anything is the source of any
  number above.** Every comparison here is an ordinary vectorization-vs-
  interpreter-overhead trade-off, the same kind a plain NumPy or SIMD program
  faces. `backend="auto"` still prefers numpy when it's importable, because
  that is a reasonable default, not because it is always the fastest choice
  -- run the benchmarks against your own workload before assuming it is.

Neither the APQB encoding nor "quantum" anything is the source of any speedup
measured here -- it is exactly what Sec. "Why a virtual QPU and not a
simulator" in `docs/architecture.md` says it is: batching and vectorization,
the same kind a classical NumPy or SIMD program gets from processing an array
instead of a Python loop. Run the scripts yourself (`--big` adds the
1,000,000-lane row) before trusting any of these numbers on your own hardware.

## Writing a new backend

Subclass `qubitbridge.backends.base.Backend`, implement the lane plumbing
(`splat`, `vector`, `to_list`), the classical kernels and the APQB kernels, and
register it in `qubitbridge/backends/__init__.py`.  Then add it to
`TestBackendAgreement` -- if it agrees with `portable` on every opcode across
several lanes, it is correct by the same standard as the others.

For a code emitter rather than an interpreter, set `executes = False` and
provide `emit(program)`, as `Arm64Backend` does.
