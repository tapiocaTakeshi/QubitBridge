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

## Writing a new backend

Subclass `qubitbridge.backends.base.Backend`, implement the lane plumbing
(`splat`, `vector`, `to_list`), the classical kernels and the APQB kernels, and
register it in `qubitbridge/backends/__init__.py`.  Then add it to
`TestBackendAgreement` -- if it agrees with `portable` on every opcode across
several lanes, it is correct by the same standard as the others.

For a code emitter rather than an interpreter, set `executes = False` and
provide `emit(program)`, as `Arm64Backend` does.
