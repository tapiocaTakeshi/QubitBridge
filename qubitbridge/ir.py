"""APQB IR -- the target-independent middle layer of the Qubit toolchain.

APQB IR sits where LLVM IR sits in a classical toolchain: above it, a frontend
decides which parts of a program are worth expressing as pseudo qubits; below
it, :mod:`qubitbridge.lower` turns it into Qubit ISA that the QVM runs on a CPU,
a GPU, or -- eventually -- real APQB silicon.  Nothing in this module knows
about registers or backends.

The IR is SSA with two types::

    f64           an ordinary classical scalar
    !apqb.state   one pseudo qubit, the {r, eta, theta} triple

and a textual form in the MLIR idiom::

    module @demo {
      func @mul2(%x: f64, %y: f64) -> (f64) {
        %q0 = apqb.encode %x {mode = "latent"} : !apqb.state
        %q1 = apqb.encode %y {mode = "latent"} : !apqb.state
        %q2 = apqb.interact %q0, %q1 : !apqb.state
        %z = apqb.decode %q2 {mode = "linear"} : f64
        return %z
      }
    }

:func:`parse_module` and :meth:`Module.to_text` round-trip that form, so IR is a
real interchange format between tools and not just an in-memory graph.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Sequence

__all__ = [
    "F64",
    "STATE",
    "Value",
    "Op",
    "Func",
    "Module",
    "Builder",
    "OpSpec",
    "OP_SPECS",
    "IRError",
    "parse_module",
]

F64 = "f64"
STATE = "!apqb.state"
TYPES = (F64, STATE)


class IRError(ValueError):
    """Malformed IR: bad types, unknown op, undefined value, ..."""


@dataclass(frozen=True)
class OpSpec:
    """Signature of an IR operation."""

    name: str
    operands: tuple[str, ...]
    result: str
    attrs: tuple[str, ...] = ()
    doc: str = ""


def _spec(name, operands, result, attrs=(), doc=""):
    return OpSpec(name, tuple(operands), result, tuple(attrs), doc)


#: Every operation the IR admits, with its operand and result types.
OP_SPECS: dict[str, OpSpec] = {s.name: s for s in (
    _spec("arith.const", (), F64, ("value",), "a literal scalar"),
    _spec("arith.add", (F64, F64), F64, (), "x + y"),
    _spec("arith.sub", (F64, F64), F64, (), "x - y"),
    _spec("arith.mul", (F64, F64), F64, (), "x * y"),
    _spec("arith.div", (F64, F64), F64, (), "x / y"),
    _spec("arith.neg", (F64,), F64, (), "-x"),
    _spec("arith.tanh", (F64,), F64, (), "tanh(x)"),
    _spec("arith.atanh", (F64,), F64, (), "atanh(x)"),

    _spec("apqb.encode", (F64,), STATE, ("mode",),
          "classical scalar -> pseudo qubit"),
    _spec("apqb.decode", (STATE,), F64, ("mode",),
          "pseudo qubit -> classical scalar"),
    _spec("apqb.rotate", (STATE, F64), STATE, (), "theta -> theta + phi"),
    _spec("apqb.interact", (STATE, STATE), STATE, (),
          "z_a * z_b: the subset product, i.e. theta addition"),
    _spec("apqb.mul", (STATE, STATE), STATE, (), "r_a * r_b"),
    _spec("apqb.pow", (STATE,), STATE, ("k",), "z^k"),
    _spec("apqb.correlate", (STATE, STATE), F64, (),
          "cos(2(theta_a - theta_b))"),
    _spec("apqb.uncertainty", (STATE,), F64, (), "T = |eta|"),
    _spec("apqb.imag", (STATE,), F64, (), "eta, signed"),
    _spec("apqb.entropy", (STATE,), F64, (), "H_Z in bits"),
    _spec("apqb.gate", (STATE, STATE, F64), STATE, (),
          "QBNN coupling in latent space"),
    _spec("apqb.measure", (STATE,), F64, ("mode",), "expectation or sample"),
    _spec("apqb.normalize", (STATE,), STATE, (), "re-project onto the circle"),
    _spec("apqb.cheb_t", (STATE,), F64, ("k",), "Re(z^k) = T_k(r)"),
    _spec("apqb.cheb_u", (STATE,), F64, ("k",), "Im(z^k) = eta * U_{k-1}(r)"),
)}


@dataclass(frozen=True)
class Value:
    """An SSA value: a name and a type."""

    name: str
    type: str

    def __str__(self) -> str:
        return f"%{self.name}"


@dataclass
class Op:
    """One SSA operation producing exactly one result."""

    name: str
    operands: list[Value] = field(default_factory=list)
    result: Value | None = None
    attrs: dict[str, Any] = field(default_factory=dict)

    @property
    def spec(self) -> OpSpec:
        try:
            return OP_SPECS[self.name]
        except KeyError:
            raise IRError(f"unknown operation {self.name!r}") from None


@dataclass
class Func:
    """A function: typed arguments, a straight-line body, returned values."""

    name: str
    args: list[Value] = field(default_factory=list)
    body: list[Op] = field(default_factory=list)
    results: list[Value] = field(default_factory=list)

    def values(self) -> dict[str, Value]:
        """Every value in scope, by name."""
        out = {v.name: v for v in self.args}
        for op in self.body:
            if op.result is not None:
                out[op.result.name] = op.result
        return out


@dataclass
class Module:
    """A named collection of functions."""

    name: str = "module"
    funcs: list[Func] = field(default_factory=list)

    def func(self, name: str) -> Func:
        for f in self.funcs:
            if f.name == name:
                return f
        raise KeyError(f"no function @{name} in module @{self.name}")

    # -- verification ----------------------------------------------------
    def verify(self) -> None:
        """Type-check the whole module; raises :class:`IRError` on any fault."""
        seen_funcs: set[str] = set()
        for fn in self.funcs:
            if fn.name in seen_funcs:
                raise IRError(f"duplicate function @{fn.name}")
            seen_funcs.add(fn.name)
            defined: dict[str, Value] = {}
            for arg in fn.args:
                if arg.type not in TYPES:
                    raise IRError(f"@{fn.name}: argument %{arg.name} has unknown "
                                  f"type {arg.type!r}")
                if arg.name in defined:
                    raise IRError(f"@{fn.name}: duplicate argument %{arg.name}")
                defined[arg.name] = arg

            for i, op in enumerate(fn.body):
                spec = op.spec
                where = f"@{fn.name} op {i} ({op.name})"
                if len(op.operands) != len(spec.operands):
                    raise IRError(f"{where}: expects {len(spec.operands)} "
                                  f"operand(s), got {len(op.operands)}")
                for operand, want in zip(op.operands, spec.operands):
                    known = defined.get(operand.name)
                    if known is None:
                        raise IRError(f"{where}: %{operand.name} is not defined "
                                      f"before use")
                    if known.type != want:
                        raise IRError(f"{where}: %{operand.name} has type "
                                      f"{known.type}, expected {want}")
                missing = set(spec.attrs) - set(op.attrs)
                if missing:
                    raise IRError(f"{where}: missing attribute(s) "
                                  f"{', '.join(sorted(missing))}")
                extra = set(op.attrs) - set(spec.attrs)
                if extra:
                    raise IRError(f"{where}: unexpected attribute(s) "
                                  f"{', '.join(sorted(extra))}")
                if op.result is None:
                    raise IRError(f"{where}: must define a result")
                if op.result.type != spec.result:
                    raise IRError(f"{where}: result %{op.result.name} declared "
                                  f"{op.result.type}, op yields {spec.result}")
                if op.result.name in defined:
                    raise IRError(f"{where}: %{op.result.name} redefined; the IR "
                                  f"is SSA")
                defined[op.result.name] = op.result

            for value in fn.results:
                if value.name not in defined:
                    raise IRError(f"@{fn.name}: returns undefined %{value.name}")

    # -- printing --------------------------------------------------------
    def to_text(self) -> str:
        lines = [f"module @{self.name} {{"]
        for fn in self.funcs:
            args = ", ".join(f"%{a.name}: {a.type}" for a in fn.args)
            rets = ", ".join(r.type for r in fn.results)
            lines.append(f"  func @{fn.name}({args}) -> ({rets}) {{")
            for op in fn.body:
                lines.append("    " + _format_op(op))
            lines.append("    return " + ", ".join(f"%{r.name}" for r in fn.results))
            lines.append("  }")
        lines.append("}")
        return "\n".join(lines) + "\n"

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.to_text()


def _format_attr(value: Any) -> str:
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    return repr(float(value))


def _format_op(op: Op) -> str:
    text = f"%{op.result.name} = {op.name}"
    if op.operands:
        text += " " + ", ".join(f"%{o.name}" for o in op.operands)
    if op.attrs:
        inner = ", ".join(f"{k} = {_format_attr(v)}"
                          for k, v in sorted(op.attrs.items()))
        text += " {" + inner + "}"
    return text + f" : {op.result.type}"


class Builder:
    """Convenience API for constructing a :class:`Func` programmatically."""

    def __init__(self, module: Module, name: str):
        self.module = module
        self.func = Func(name=name)
        module.funcs.append(self.func)
        self._counter = 0

    def _fresh(self, hint: str, type_: str) -> Value:
        self._counter += 1
        return Value(f"{hint}{self._counter}", type_)

    def arg(self, name: str, type_: str = F64) -> Value:
        value = Value(name, type_)
        self.func.args.append(value)
        return value

    def emit(self, name: str, *operands: Value, result_hint: str | None = None,
             **attrs: Any) -> Value:
        """Append an operation and return its result value."""
        spec = OP_SPECS.get(name)
        if spec is None:
            raise IRError(f"unknown operation {name!r}")
        hint = result_hint or ("q" if spec.result == STATE else "v")
        result = self._fresh(hint, spec.result)
        self.func.body.append(Op(name, list(operands), result, dict(attrs)))
        return result

    def ret(self, *values: Value) -> None:
        self.func.results = list(values)

    # -- shorthands ------------------------------------------------------
    def const(self, value: float) -> Value:
        return self.emit("arith.const", value=float(value), result_hint="c")

    def encode(self, x: Value, mode: str = "latent") -> Value:
        return self.emit("apqb.encode", x, mode=mode)

    def decode(self, q: Value, mode: str = "linear") -> Value:
        return self.emit("apqb.decode", q, mode=mode)

    def interact(self, a: Value, b: Value) -> Value:
        return self.emit("apqb.interact", a, b)

    def correlate(self, a: Value, b: Value) -> Value:
        return self.emit("apqb.correlate", a, b)

    def gate(self, target: Value, source: Value, j: Value) -> Value:
        return self.emit("apqb.gate", target, source, j)


# ===========================================================================
# Textual parser
# ===========================================================================

_TOKEN = re.compile(r"""
      (?P<ws>\s+)
    | (?P<comment>//[^\n]*)
    | (?P<string>"(?:[^"\\]|\\.)*")
    | (?P<ssa>%[A-Za-z_][A-Za-z0-9_.]*)
    | (?P<sym>@[A-Za-z_][A-Za-z0-9_.]*)
    | (?P<type>![A-Za-z_][A-Za-z0-9_.]*)
    | (?P<number>-?(?:\d+\.\d*(?:[eE][-+]?\d+)?|\.\d+(?:[eE][-+]?\d+)?|\d+(?:[eE][-+]?\d+)?))
    | (?P<arrow>->)
    | (?P<ident>[A-Za-z_][A-Za-z0-9_.]*)
    | (?P<punct>[{}(),:=])
""", re.VERBOSE)


def _tokenize(text: str) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []
    pos = 0
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if not m:
            snippet = text[pos:pos + 20]
            raise IRError(f"cannot tokenize IR at {snippet!r}")
        kind = m.lastgroup
        if kind not in ("ws", "comment"):
            tokens.append((kind, m.group()))
        pos = m.end()
    return tokens


class _Parser:
    def __init__(self, tokens: Sequence[tuple[str, str]]):
        self.tokens = list(tokens)
        self.i = 0

    def peek(self) -> tuple[str, str]:
        return self.tokens[self.i] if self.i < len(self.tokens) else ("eof", "")

    def next(self) -> tuple[str, str]:
        token = self.peek()
        self.i += 1
        return token

    def expect(self, text: str) -> str:
        kind, value = self.next()
        if value != text:
            raise IRError(f"expected {text!r}, got {value!r}")
        return value

    def accept(self, text: str) -> bool:
        if self.peek()[1] == text:
            self.i += 1
            return True
        return False

    def parse_type(self) -> str:
        kind, value = self.next()
        if value not in TYPES:
            raise IRError(f"unknown type {value!r}")
        return value

    def parse_module(self) -> Module:
        self.expect("module")
        name = self.next()[1].lstrip("@")
        self.expect("{")
        module = Module(name=name)
        while not self.accept("}"):
            module.funcs.append(self.parse_func())
        return module

    def parse_func(self) -> Func:
        self.expect("func")
        fn = Func(name=self.next()[1].lstrip("@"))
        scope: dict[str, Value] = {}
        self.expect("(")
        while not self.accept(")"):
            arg_name = self.next()[1].lstrip("%")
            self.expect(":")
            value = Value(arg_name, self.parse_type())
            fn.args.append(value)
            scope[arg_name] = value
            self.accept(",")
        self.expect("->")
        self.expect("(")
        while not self.accept(")"):
            self.parse_type()  # declared result types, re-derived on return
            self.accept(",")
        self.expect("{")
        while not self.accept("}"):
            if self.accept("return"):
                while self.peek()[0] == "ssa":
                    ref = self.next()[1].lstrip("%")
                    if ref not in scope:
                        raise IRError(f"@{fn.name}: returns undefined %{ref}")
                    fn.results.append(scope[ref])
                    self.accept(",")
                continue
            op = self.parse_op(scope, fn.name)
            fn.body.append(op)
            scope[op.result.name] = op.result
        return fn

    def parse_op(self, scope: dict[str, Value], fn_name: str) -> Op:
        kind, value = self.next()
        if kind != "ssa":
            raise IRError(f"@{fn_name}: expected an SSA result name, got {value!r}")
        result_name = value.lstrip("%")
        self.expect("=")
        op_name = self.next()[1]
        spec = OP_SPECS.get(op_name)
        if spec is None:
            raise IRError(f"@{fn_name}: unknown operation {op_name!r}")

        operands: list[Value] = []
        while self.peek()[0] == "ssa":
            ref = self.next()[1].lstrip("%")
            if ref not in scope:
                raise IRError(f"@{fn_name}: %{ref} is not defined before use")
            operands.append(scope[ref])
            if not self.accept(","):
                break

        attrs: dict[str, Any] = {}
        if self.accept("{"):
            while not self.accept("}"):
                key = self.next()[1]
                self.expect("=")
                attrs[key] = self.parse_attr_value()
                self.accept(",")

        self.expect(":")
        result_type = self.parse_type()
        return Op(op_name, operands, Value(result_name, result_type), attrs)

    def parse_attr_value(self) -> Any:
        kind, value = self.next()
        if kind == "string":
            return value[1:-1]
        if kind == "number":
            if re.fullmatch(r"-?\d+", value):
                return int(value)
            return float(value)
        if value in ("true", "false"):
            return value == "true"
        raise IRError(f"unsupported attribute value {value!r}")


def parse_module(text: str) -> Module:
    """Parse the textual form and verify the result."""
    module = _Parser(_tokenize(text)).parse_module()
    module.verify()
    return module
