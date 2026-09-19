"""A direct interpreter for APQB IR, used only as a test oracle.

It evaluates IR straight against :mod:`qubitbridge.apqb` with no registers, no
instruction encoding and no backends, so agreement between this and a
lowered program running on the QVM is evidence that the whole
IR -> ISA -> backend path preserves meaning.
"""

from __future__ import annotations

import math

from qubitbridge import apqb
from qubitbridge.ir import Func


def evaluate(func: Func, args: list[float]) -> list[float]:
    """Interpret ``func`` on ``args`` (measurement uses ``expect``)."""
    env: dict[str, object] = {}
    for value, given in zip(func.args, args):
        env[value.name] = given

    for op in func.body:
        ins = [env[o.name] for o in op.operands]
        name = op.name
        if name == "arith.const":
            out = float(op.attrs["value"])
        elif name == "arith.add":
            out = ins[0] + ins[1]
        elif name == "arith.sub":
            out = ins[0] - ins[1]
        elif name == "arith.mul":
            out = ins[0] * ins[1]
        elif name == "arith.div":
            out = ins[0] / ins[1]
        elif name == "arith.neg":
            out = -ins[0]
        elif name == "arith.tanh":
            out = math.tanh(ins[0])
        elif name == "arith.atanh":
            out = math.atanh(apqb.clamp_atanh(ins[0]))
        elif name == "apqb.encode":
            out = apqb.encode(ins[0], op.attrs["mode"])
        elif name == "apqb.decode":
            out = apqb.decode(ins[0], op.attrs["mode"])
        elif name == "apqb.rotate":
            out = apqb.rotate(ins[0], ins[1])
        elif name == "apqb.interact":
            out = apqb.interact(ins[0], ins[1])
        elif name == "apqb.mul":
            out = apqb.state_mul(ins[0], ins[1])
        elif name == "apqb.pow":
            out = apqb.power(ins[0], op.attrs["k"])
        elif name == "apqb.correlate":
            out = apqb.correlate(ins[0], ins[1])
        elif name == "apqb.uncertainty":
            out = apqb.uncertainty(ins[0])
        elif name == "apqb.imag":
            out = ins[0].eta
        elif name == "apqb.entropy":
            out = apqb.entropy_z(ins[0])
        elif name == "apqb.gate":
            out = apqb.gate(ins[0], ins[1], ins[2])
        elif name == "apqb.measure":
            out = apqb.measure_expect(ins[0])
        elif name == "apqb.normalize":
            out = ins[0].normalized()
        elif name == "apqb.cheb_t":
            out = apqb.power(ins[0], op.attrs["k"]).r
        elif name == "apqb.cheb_u":
            out = apqb.power(ins[0], op.attrs["k"]).eta
        else:
            raise AssertionError(f"reference has no rule for {name}")
        env[op.result.name] = out

    return [env[r.name] for r in func.results]
