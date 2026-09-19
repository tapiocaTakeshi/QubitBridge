# The Qubit ISA

## Register files

| file | count | contents |
|---|---|---|
| `Q0..Q31` | 32 | one pseudo qubit each: `{r, eta, theta}`, `r^2 + eta^2 = 1` |
| `R0..R31` | 32 | ordinary `f64` scalars |

Under SPMD execution every register holds `n` lanes, so `Q5` is really two lane
vectors (`r` and `eta`) and `R5` is one.  All registers start zeroed: `R` at
`0.0`, `Q` at `|0>` (`r = 1`, `eta = 0`).

A module may also declare a data segment with `.mem N`, addressed by `LOAD`,
`STORE` and `QSTORE` (which writes `r` at `addr` and `eta` at `addr + 1`).

## Instruction word

One fixed 8-byte little-endian word:

```
+--------+--------+--------+--------+------------------+
| opcode |  dst   |   a    |   b    |   imm (uint32)   |
+--------+--------+--------+--------+------------------+
     1        1        1        1            4
```

`imm` either indexes the module's constant pool (`<const>`), or carries a small
value directly: a data address (`<addr>`), a non-negative integer (`<int>`), an
encoding mode (`<enc>`), a measurement mode (`<meas>`), or a scalar register
index (`<rreg>`, used by `QGATER` because its other three slots are full).

Which register file each slot names is fixed per opcode; `Instr.validate()`
rejects an out-of-range register, a non-zero unused slot, an immediate on an
opcode that takes none, a constant index outside the pool, and an unknown mode.

## Opcodes

| opcode | hex | operands | meaning |
|---|---|---|---|
| `HALT` | `0x00` | -- | stop the machine |
| `NOP` | `0x01` | -- | do nothing |
| `MOV` | `0x02` | Rd, Ra | Rd <- Ra |
| `LDI` | `0x03` | Rd, `<const>` | Rd <- immediate constant |
| `LOAD` | `0x04` | Rd, `<addr>` | Rd <- mem[addr] |
| `STORE` | `0x05` | Ra, `<addr>` | mem[addr] <- Ra |
| `ADD` | `0x06` | Rd, Ra, Rb | Rd <- Ra + Rb |
| `SUB` | `0x07` | Rd, Ra, Rb | Rd <- Ra - Rb |
| `MUL` | `0x08` | Rd, Ra, Rb | Rd <- Ra * Rb |
| `DIV` | `0x09` | Rd, Ra, Rb | Rd <- Ra / Rb |
| `NEG` | `0x0a` | Rd, Ra | Rd <- -Ra |
| `TANH` | `0x0b` | Rd, Ra | Rd <- tanh(Ra) |
| `ATANH` | `0x0c` | Rd, Ra | Rd <- atanh(Ra) |
| `SCALE` | `0x0d` | Rd, Ra, `<const>` | Rd <- Ra * c |
| `ADDI` | `0x0e` | Rd, Ra, `<const>` | Rd <- Ra + c |
| `MIN` | `0x0f` | Rd, Ra, Rb | Rd <- min(Ra, Rb) |
| `MAX` | `0x10` | Rd, Ra, Rb | Rd <- max(Ra, Rb) |
| `QLOAD` | `0x20` | Qd, `<const>` | Qd <- canonical state with r = c |
| `QSTORE` | `0x21` | Qa, `<addr>` | mem[addr], mem[addr+1] <- r, eta |
| `QMOV` | `0x22` | Qd, Qa | Qd <- Qa |
| `QENC` | `0x23` | Qd, Ra, `<enc>` | Qd <- encode(Ra) |
| `QDEC` | `0x24` | Rd, Qa, `<enc>` | Rd <- decode(Qa) |
| `QROT` | `0x25` | Qd, Qa, `<const>` | Qd <- rotate(Qa, phi) |
| `QROTR` | `0x26` | Qd, Qa, Rb | Qd <- rotate(Qa, Rb) |
| `QINT` | `0x27` | Qd, Qa, Qb | Qd <- Qa (x) Qb |
| `QMUL` | `0x28` | Qd, Qa, Qb | Qd <- Qa . Qb |
| `QPOW` | `0x29` | Qd, Qa, `<int>` | Qd <- Qa ^ k |
| `QCORR` | `0x2a` | Rd, Qa, Qb | Rd <- corr(Qa, Qb) |
| `QUNC` | `0x2b` | Rd, Qa | Rd <- |eta(Qa)| |
| `QENT` | `0x2c` | Rd, Qa | Rd <- H_Z(Qa) |
| `QGATE` | `0x2d` | Qd, Qa, Qb, `<const>` | Qd <- gate(Qa, Qb, J = c) |
| `QGATER` | `0x2e` | Qd, Qa, Qb, `<rreg>` | Qd <- gate(Qa, Qb, J = R[imm]) |
| `QMEASURE` | `0x2f` | Rd, Qa, `<meas>` | Rd <- measure(Qa) |
| `QNORM` | `0x30` | Qd, Qa | Qd <- normalize(Qa) |
| `QIMAG` | `0x31` | Rd, Qa | Rd <- eta(Qa), signed |

## Assembly syntax

One instruction per line, operands in the order `dst, a, b, imm` with unused
slots dropped.  `;` and `#` start a comment.

```
.module dot2
.mem 4
        QENC      Q0, R0, latent      ; mode by name
        QINT      Q2, Q0, Q1
        QROT      Q2, Q2, 0.125       ; float constants are interned
        QGATER    Q3, Q0, Q1, R4      ; coupling from a scalar register
        QPOW      Q4, Q2, 3           ; integer immediate
        QSTORE    Q2, [0]             ; addresses in brackets
        HALT
```

`assemble()` returns a validated `Program`; `disassemble()` prints one back.
The round trip is a fixed point, so `assemble(disassemble(p))` reproduces `p`.

Every program must end in `HALT`; v1 has no branches.

## Object format

`Program.pack()` writes a `.qvm` object:

```
magic "QVM1" | version u16 | name_len u16 | n_consts u32 | n_code u32 | mem_size u32
name bytes | consts (f64 each) | code (8 bytes each)
```

`Program.unpack()` checks the magic, the version and the length, so a truncated
or foreign file is rejected rather than misread.

## Semantics

Each opcode's meaning is the corresponding function in `qubitbridge/apqb.py`,
applied lane-wise.  The scalar Python there is the specification; the NumPy and
AArch64 backends are held to it by differential tests.

Notable details:

* `QINT` is `z_a * z_b` and may leave the canonical `eta >= 0` half circle --
  that is the point, since the subset products of Eq. (17) live on the full
  circle.  `QUNC` reports `|eta|`, so the paper's `T` is still correct.
* `QMUL` multiplies `r` and re-derives `eta = +sqrt(1 - r^2)`, staying canonical.
* `QGATE`/`QGATER` route through `atanh`, so any coupling strength stays in
  range, and `J = 0` is exactly the identity.
* `QMEASURE ... sample` consumes one uniform per lane from the VM's seeded PRNG,
  in instruction order.  `expect` consumes none.
* `ATANH` and the `latent` decode clamp to `+-R_MAX`, the largest double below
  1, rather than shrinking by a factor -- an exactly representable bound every
  backend can reproduce bit for bit.
