"""Qubit Compiler frontend: a Python subset -> APQB IR, with partitioning.

The point of the hybrid design is that *not everything should become a pseudo
qubit*.  Opening a file, copying memory or printing a string gains nothing from
an APQB encoding; products of bounded correlation-like quantities gain the
subset-product structure the APQB paper is built on.  So this frontend compiles
a whole function and decides, per subgraph, which side of the split it belongs
on::

        source expression
               |
        Qubit Compiler  --- bounded product chain of degree >= K ?
           /        \\
       classical    APQB
       arith.*      encode -> interact/mul chain -> decode

The rule is deliberately conservative and checkable rather than clever:

* A value is **bounded** if it is a literal in [-1, 1], an argument annotated
  ``unit``, the result of ``tanh``, or a product of bounded values.
* A **product chain** of at least ``min_degree`` bounded factors becomes an
  APQB region: each factor is encoded with ``mode = "linear"`` (so r = x
  exactly), the factors are combined with ``apqb.mul``, and the result is
  decoded back.  Because [-1, 1] is closed under multiplication and
  ``apqb.mul`` multiplies the r coordinates, **the APQB path returns exactly
  the classical product** -- routing a region through the QVM never changes the
  answer, which is what makes the partition safe to apply automatically.
* Everything else stays classical.

Anything the frontend cannot prove bounded stays classical too; wrap it in
``apqb(...)`` to force the APQB path (values are clamped into range), or in
``classical(...)`` to pin it to the classical one.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from dataclasses import dataclass, field
from typing import Callable

from .ir import F64, STATE, Builder, Module, Value

__all__ = [
    "CompileError",
    "Region",
    "PartitionReport",
    "compile_source",
    "compile_function",
    "INTRINSICS",
]


class CompileError(ValueError):
    """Source the frontend cannot compile, with a line number where possible."""


#: Intrinsics callable from source, mapped to their IR operation.
INTRINSICS = {
    "corr": ("apqb.correlate", 2),
    "unc": ("apqb.uncertainty", 1),
    "entropy": ("apqb.entropy", 1),
    "measure": ("apqb.measure", 1),
    "rot": ("apqb.rotate", 2),
    "gate": ("apqb.gate", 3),
    "cheb_t": ("apqb.cheb_t", 2),
    "cheb_u": ("apqb.cheb_u", 2),
    "tanh": ("arith.tanh", 1),
    "atanh": ("arith.atanh", 1),
}


# ===========================================================================
# Expression graph (built before any IR, so the partitioner can see shapes)
# ===========================================================================

@dataclass(eq=False)
class _Node:
    """One dataflow node.

    ``eq=False`` makes nodes hashable by identity, which is what the emitter's
    common-subexpression cache keys on; holding the node itself (rather than its
    ``id()``) also keeps it alive, so a freed node's address can never be
    mistaken for a live one.
    """

    kind: str                      # arg | const | binop | unary | call | force
    value: object = None           # name, literal, operator or intrinsic
    children: list["_Node"] = field(default_factory=list)
    bounded: bool = False
    line: int = 0


@dataclass
class Region:
    """One APQB region chosen by the partitioner."""

    #: Number of factors in the product chain.
    degree: int
    #: Source line the region came from.
    line: int
    #: Why it was routed to the QVM.
    reason: str


@dataclass
class PartitionReport:
    """What the compiler decided, and why -- the audit trail of the split."""

    regions: list[Region] = field(default_factory=list)
    classical_ops: int = 0
    apqb_ops: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def apqb_fraction(self) -> float:
        total = self.classical_ops + self.apqb_ops
        return self.apqb_ops / total if total else 0.0

    def summary(self) -> str:
        lines = [
            f"APQB regions: {len(self.regions)}",
            f"ops: {self.apqb_ops} APQB / {self.classical_ops} classical "
            f"({self.apqb_fraction:.0%} APQB)",
        ]
        for region in self.regions:
            lines.append(f"  line {region.line}: degree-{region.degree} chain "
                         f"-> QVM ({region.reason})")
        lines.extend(f"  note: {note}" for note in self.notes)
        return "\n".join(lines)


class _GraphBuilder(ast.NodeVisitor):
    """Turns the function body into a _Node dataflow graph."""

    def __init__(self, unit_args: set[str]):
        self.env: dict[str, _Node] = {}
        self.unit_args = unit_args

    def bind_arg(self, name: str, line: int) -> None:
        self.env[name] = _Node("arg", name, bounded=name in self.unit_args,
                               line=line)

    def expr(self, node: ast.AST) -> _Node:
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)) or isinstance(node.value, bool):
                raise CompileError(f"line {node.lineno}: only numeric literals "
                                   f"are supported")
            value = float(node.value)
            return _Node("const", value, bounded=-1.0 <= value <= 1.0,
                         line=node.lineno)

        if isinstance(node, ast.Name):
            if node.id not in self.env:
                raise CompileError(f"line {node.lineno}: {node.id!r} is not defined")
            return self.env[node.id]

        if isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, (ast.UAdd, ast.USub)):
                raise CompileError(f"line {node.lineno}: unsupported unary operator")
            operand = self.expr(node.operand)
            if isinstance(node.op, ast.UAdd):
                return operand
            return _Node("unary", "neg", [operand], operand.bounded, node.lineno)

        if isinstance(node, ast.BinOp):
            ops = {ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul", ast.Div: "div"}
            for kind, name in ops.items():
                if isinstance(node.op, kind):
                    left, right = self.expr(node.left), self.expr(node.right)
                    bounded = name == "mul" and left.bounded and right.bounded
                    return _Node("binop", name, [left, right], bounded, node.lineno)
            raise CompileError(f"line {node.lineno}: unsupported binary operator")

        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise CompileError(f"line {node.lineno}: only direct calls to "
                                   f"intrinsics are supported")
            name = node.func.id
            args = [self.expr(a) for a in node.args]
            if name in ("apqb", "classical"):
                if len(args) != 1:
                    raise CompileError(f"line {node.lineno}: {name}() takes one "
                                       f"argument")
                return _Node("force", name, args, True, node.lineno)
            if name not in INTRINSICS:
                raise CompileError(
                    f"line {node.lineno}: unknown function {name!r}; available: "
                    f"{', '.join(sorted(set(INTRINSICS) | {'apqb', 'classical'}))}")
            op_name, arity = INTRINSICS[name]
            if len(args) != arity:
                raise CompileError(f"line {node.lineno}: {name}() takes {arity} "
                                   f"argument(s), got {len(args)}")
            bounded = name in ("tanh", "corr", "unc", "measure", "cheb_t", "cheb_u")
            return _Node("call", name, args, bounded, node.lineno)

        raise CompileError(f"line {getattr(node, 'lineno', 0)}: unsupported "
                           f"expression {type(node).__name__}")


def _flatten_product(node: _Node) -> list[_Node]:
    """Collect the factors of a maximal multiplication chain."""
    if node.kind == "binop" and node.value == "mul":
        return _flatten_product(node.children[0]) + _flatten_product(node.children[1])
    return [node]


# ===========================================================================
# Emission
# ===========================================================================

class _Emitter:
    def __init__(self, builder: Builder, report: PartitionReport,
                 min_degree: int):
        self.b = builder
        self.report = report
        self.min_degree = min_degree
        self.cache: dict[_Node, Value] = {}
        self.encoded: dict[tuple[str, str], Value] = {}

    def _classical(self, name: str, *operands: Value) -> Value:
        self.report.classical_ops += 1
        return self.b.emit(name, *operands)

    def _apqb(self, name: str, *operands: Value, **attrs) -> Value:
        self.report.apqb_ops += 1
        return self.b.emit(name, *operands, **attrs)

    def _encode(self, value: Value, mode: str = "linear") -> Value:
        """Encode once per (value, mode): re-encoding a value is dead work."""
        key = (value.name, mode)
        if key not in self.encoded:
            self.encoded[key] = self._apqb("apqb.encode", value, mode=mode)
        return self.encoded[key]

    def emit(self, node: _Node, force: str | None = None) -> Value:
        if force is None and node in self.cache:
            return self.cache[node]

        if node.kind == "arg":
            value = self.b.func.args[[a.name for a in self.b.func.args]
                                     .index(node.value)]
        elif node.kind == "const":
            self.report.classical_ops += 1
            value = self.b.const(node.value)
        elif node.kind == "force":
            value = self.emit(node.children[0], force=node.value)
        elif node.kind == "unary":
            value = self._classical("arith.neg", self.emit(node.children[0]))
        elif node.kind == "binop":
            value = self._emit_binop(node, force)
        elif node.kind == "call":
            value = self._emit_call(node)
        else:  # pragma: no cover - the graph builder makes no other kinds
            raise CompileError(f"cannot emit node kind {node.kind!r}")

        if force is None:
            self.cache[node] = value
        return value

    def _emit_binop(self, node: _Node, force: str | None) -> Value:
        if node.value == "mul":
            factors = _flatten_product(node)
            eligible = all(f.bounded for f in factors)
            if force == "classical":
                pass
            elif force == "apqb" or (eligible and len(factors) >= self.min_degree):
                reason = ("forced with apqb()" if force == "apqb"
                          else "all factors provably in [-1, 1]")
                self.report.regions.append(
                    Region(len(factors), node.line, reason))
                return self._emit_product_region(factors)
            elif len(factors) >= self.min_degree:
                self.report.notes.append(
                    f"line {node.line}: degree-{len(factors)} product kept "
                    f"classical; factors are not provably in [-1, 1]")

        ops = {"add": "arith.add", "sub": "arith.sub",
               "mul": "arith.mul", "div": "arith.div"}
        left = self.emit(node.children[0])
        right = self.emit(node.children[1])
        return self._classical(ops[node.value], left, right)

    def _emit_product_region(self, factors: list[_Node]) -> Value:
        states = [self._encode(self.emit(f)) for f in factors]
        acc = states[0]
        for state in states[1:]:
            acc = self._apqb("apqb.mul", acc, state)
        return self._apqb("apqb.decode", acc, mode="linear")

    def _emit_call(self, node: _Node) -> Value:
        name = node.value
        op_name, _ = INTRINSICS[name]
        if op_name.startswith("arith."):
            return self._classical(op_name, *[self.emit(c) for c in node.children])

        spec_operands = {
            "corr": (STATE, STATE), "unc": (STATE,), "entropy": (STATE,),
            "measure": (STATE,), "rot": (STATE, F64), "gate": (STATE, STATE, F64),
            "cheb_t": (STATE,), "cheb_u": (STATE,),
        }[name]

        operands: list[Value] = []
        for child, want in zip(node.children, spec_operands):
            value = self.emit(child)
            if want == STATE:
                value = self._encode(value)
            operands.append(value)

        attrs: dict = {}
        if name in ("cheb_t", "cheb_u"):
            k_node = node.children[1]
            if k_node.kind != "const" or float(k_node.value) != int(k_node.value):
                raise CompileError(f"line {node.line}: {name}() needs a literal "
                                   f"integer degree")
            attrs["k"] = int(k_node.value)
        if name == "measure":
            attrs["mode"] = "expect"

        result = self._apqb(op_name, *operands, **attrs)
        if result.type == STATE:
            # The surface language is scalar-valued: an intrinsic that yields a
            # pseudo qubit (rot, gate) is decoded back before it escapes.
            result = self._apqb("apqb.decode", result, mode="linear")
        return result


# ===========================================================================
# Entry points
# ===========================================================================

def compile_source(source: str, *, module_name: str = "compiled",
                   min_degree: int = 2) -> tuple[Module, PartitionReport]:
    """Compile one Python function definition into a verified APQB IR module.

    Arguments annotated ``unit`` are assumed to lie in [-1, 1], which is what
    lets the partitioner route products through the APQB path.
    """
    # dedent, not cleandoc: cleandoc treats the first line specially and
    # would flatten the body of a `def` written at column zero.
    source = textwrap.dedent(source).strip("\n")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise CompileError(f"syntax error: {exc}") from None

    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if len(funcs) != 1:
        raise CompileError("expected exactly one function definition")
    fn = funcs[0]

    unit_args = {
        arg.arg for arg in fn.args.args
        if isinstance(arg.annotation, ast.Name) and arg.annotation.id == "unit"
    }
    if fn.args.vararg or fn.args.kwarg or fn.args.kwonlyargs:
        raise CompileError("only positional arguments are supported")

    module = Module(name=module_name)
    builder = Builder(module, fn.name)
    graph = _GraphBuilder(unit_args)
    for arg in fn.args.args:
        builder.arg(arg.arg)
        graph.bind_arg(arg.arg, fn.lineno)

    report = PartitionReport()
    if unit_args:
        report.notes.append(f"unit-annotated arguments: "
                            f"{', '.join(sorted(unit_args))}")
    emitter = _Emitter(builder, report, min_degree)

    returned: list[Value] = []
    for stmt in fn.body:
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
            continue  # docstring
        if isinstance(stmt, ast.Assign):
            if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
                raise CompileError(f"line {stmt.lineno}: only simple "
                                   f"'name = expr' assignments are supported")
            graph.env[stmt.targets[0].id] = graph.expr(stmt.value)
            continue
        if isinstance(stmt, ast.Return):
            if stmt.value is None:
                raise CompileError(f"line {stmt.lineno}: return needs a value")
            targets = (stmt.value.elts if isinstance(stmt.value, ast.Tuple)
                       else [stmt.value])
            returned = [emitter.emit(graph.expr(t)) for t in targets]
            break
        raise CompileError(f"line {stmt.lineno}: unsupported statement "
                           f"{type(stmt).__name__}; the frontend accepts "
                           f"assignments and a return")

    if not returned:
        raise CompileError("function has no return statement")
    for value in returned:
        if value.type != F64:
            raise CompileError(  # pragma: no cover - the emitter decodes states
                f"returned %{value.name} has type {value.type}; the frontend "
                f"only yields classical scalars")
    builder.ret(*returned)
    module.verify()
    return module, report


def compile_function(fn: Callable, **kwargs) -> tuple[Module, PartitionReport]:
    """Compile a live Python function by reading its source."""
    return compile_source(inspect.getsource(fn), **kwargs)
