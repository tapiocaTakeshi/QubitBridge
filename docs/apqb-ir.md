# APQB IR

APQB IR is the target-independent middle layer.  It is SSA, it has two types,
and it round-trips through text so that it works as an interchange format
between tools rather than only as an in-memory graph.

## Types

| type | meaning |
|---|---|
| `f64` | an ordinary classical scalar |
| `!apqb.state` | one pseudo qubit: the `{r, eta, theta}` triple |

## Textual form

```
module @demo {
  func @mul2(%x: f64, %y: f64) -> (f64) {
    %q0 = apqb.encode %x {mode = "latent"} : !apqb.state
    %q1 = apqb.encode %y {mode = "latent"} : !apqb.state
    %q2 = apqb.interact %q0, %q1 : !apqb.state
    %z = apqb.decode %q2 {mode = "linear"} : f64
    return %z
  }
}
```

The type after `:` is the **result** type.  `//` starts a comment.  Attributes
are `{key = value}` with string, integer, float or boolean values.

```python
from qubitbridge import parse_module
module = parse_module(text)      # parses and verifies
print(module.to_text())          # prints back to the same text
```

## Operations

| operation | operands | result | attributes | meaning |
|---|---|---|---|---|
| `arith.const` | -- | `f64` | `value` | a literal scalar |
| `arith.add` | f64, f64 | `f64` | -- | x + y |
| `arith.sub` | f64, f64 | `f64` | -- | x - y |
| `arith.mul` | f64, f64 | `f64` | -- | x * y |
| `arith.div` | f64, f64 | `f64` | -- | x / y |
| `arith.neg` | f64 | `f64` | -- | -x |
| `arith.tanh` | f64 | `f64` | -- | tanh(x) |
| `arith.atanh` | f64 | `f64` | -- | atanh(x) |
| `apqb.encode` | f64 | `!apqb.state` | `mode` | classical scalar -> pseudo qubit |
| `apqb.decode` | !apqb.state | `f64` | `mode` | pseudo qubit -> classical scalar |
| `apqb.rotate` | !apqb.state, f64 | `!apqb.state` | -- | theta -> theta + phi |
| `apqb.interact` | !apqb.state, !apqb.state | `!apqb.state` | -- | z_a * z_b: the subset product, i.e. theta addition |
| `apqb.mul` | !apqb.state, !apqb.state | `!apqb.state` | -- | r_a * r_b |
| `apqb.pow` | !apqb.state | `!apqb.state` | `k` | z^k |
| `apqb.correlate` | !apqb.state, !apqb.state | `f64` | -- | cos(2(theta_a - theta_b)) |
| `apqb.uncertainty` | !apqb.state | `f64` | -- | T = |eta| |
| `apqb.imag` | !apqb.state | `f64` | -- | eta, signed |
| `apqb.entropy` | !apqb.state | `f64` | -- | H_Z in bits |
| `apqb.gate` | !apqb.state, !apqb.state, f64 | `!apqb.state` | -- | QBNN coupling in latent space |
| `apqb.measure` | !apqb.state | `f64` | `mode` | expectation or sample |
| `apqb.normalize` | !apqb.state | `!apqb.state` | -- | re-project onto the circle |
| `apqb.cheb_t` | !apqb.state | `f64` | `k` | Re(z^k) = T_k(r) |
| `apqb.cheb_u` | !apqb.state | `f64` | `k` | Im(z^k) = eta * U_{k-1}(r) |

### Encoding modes

`apqb.encode` / `apqb.decode` take `mode`:

| mode | encode | decode |
|---|---|---|
| `latent` | `r = tanh(a)`, `eta = sech(a)` (Eq. 12) | `a = atanh(r)` |
| `linear` | `r = clamp(x, -1, 1)` | `x = r` |
| `angle` | `r = cos 2t`, `eta = sin 2t` | `t = (1/2) atan2(eta, r)` |
| `prob` | `r = 2p - 1` (Eq. 6-7) | `p = (1 + r)/2` |

`apqb.measure` takes `mode = "expect"` (the deterministic `<Z> = r`) or
`"sample"` (a seeded Bernoulli draw returning `+1`/`-1`).

### Notes on individual operations

* **`apqb.interact`** is complex multiplication of the `z` coordinates, which is
  addition of angles.  Repeating it builds the degree-`k` subset products of
  Eq. (17)-(18); `apqb.pow {k}` is the self-interaction shorthand.
* **`apqb.mul`** multiplies the `r` coordinates instead, and re-derives
  `eta = +sqrt(1 - r^2)`.  Unlike `interact` it stays on the canonical half
  circle, and it is what the partitioner uses for bounded products.
* **`apqb.cheb_t {k}`** and **`apqb.cheb_u {k}`** are `Re(z^k) = T_k(r)` and
  `Im(z^k) = eta * U_{k-1}(r)`, the Chebyshev features of Prop. 2.  They lower
  to a `QPOW` into a scratch register followed by a read of one coordinate.
* **`apqb.gate`** applies the QBNN coupling in latent space,
  `a = atanh(r_t) + J * r_s`, `r' = tanh(a)`.  Working through `a` keeps the
  result in `[-1, 1]` for any `J`, and `J = 0` is exactly the identity -- the
  discrete counterpart of the `lambda = 0` reduction of `QBNNLayerV2`.
* **`apqb.uncertainty`** is `|eta|`, the paper's `T = |sin 2 theta|`.
  **`apqb.imag`** is the *signed* `eta`, which is what `cheb_u` needs.

## Building IR programmatically

```python
from qubitbridge import Builder, Module

module = Module("demo")
b = Builder(module, "mul2")
x, y = b.arg("x"), b.arg("y")
b.ret(b.decode(b.interact(b.encode(x), b.encode(y))))
module.verify()
```

`Builder.emit(name, *operands, **attrs)` reaches any operation;
`const`, `encode`, `decode`, `interact`, `correlate` and `gate` are shorthands.

## Verification

`Module.verify()` checks operand counts and types, that every use is dominated
by its definition, that attributes match the operation's signature exactly,
that results are SSA, and that returned values exist.  Errors name the function,
the operation index and the value, e.g.

```
IRError: @kernel op 4 (apqb.interact): %x has type f64, expected !apqb.state
```
