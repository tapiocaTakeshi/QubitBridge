"""IR -> ISA lowering: allocation, folding and meaning preservation."""

import unittest

from qubitbridge.apqb import APQBState
from qubitbridge.ir import Builder, Module, parse_module
from qubitbridge.isa import Op
from qubitbridge.lower import LoweringError, lower_func, lower_module
from qubitbridge.vm import QVM

from .reference import evaluate

PROGRAMS = {
    "chain": """
        module @m {
          func @f(%x: f64, %y: f64) -> (f64) {
            %q1 = apqb.encode %x {mode = "latent"} : !apqb.state
            %q2 = apqb.encode %y {mode = "latent"} : !apqb.state
            %q3 = apqb.interact %q1, %q2 : !apqb.state
            %q4 = apqb.mul %q3, %q1 : !apqb.state
            %v5 = apqb.decode %q4 {mode = "linear"} : f64
            return %v5
          }
        }""",
    "rotate_and_gate": """
        module @m {
          func @f(%x: f64, %j: f64) -> (f64, f64) {
            %q1 = apqb.encode %x {mode = "linear"} : !apqb.state
            %c2 = arith.const {value = 0.3125} : f64
            %q3 = apqb.rotate %q1, %c2 : !apqb.state
            %q4 = apqb.gate %q3, %q1, %j : !apqb.state
            %v5 = apqb.correlate %q3, %q4 : f64
            %v6 = apqb.entropy %q4 : f64
            return %v5, %v6
          }
        }""",
    "chebyshev": """
        module @m {
          func @f(%x: f64) -> (f64, f64, f64) {
            %q1 = apqb.encode %x {mode = "linear"} : !apqb.state
            %v2 = apqb.cheb_t %q1 {k = 4} : f64
            %v3 = apqb.cheb_u %q1 {k = 4} : f64
            %q4 = apqb.pow %q1 {k = 2} : !apqb.state
            %v5 = apqb.uncertainty %q4 : f64
            return %v2, %v3, %v5
          }
        }""",
    "classical_mix": """
        module @m {
          func @f(%x: f64, %y: f64) -> (f64) {
            %v1 = arith.mul %x, %y : f64
            %v2 = arith.tanh %v1 : f64
            %q3 = apqb.encode %v2 {mode = "linear"} : !apqb.state
            %q4 = apqb.normalize %q3 : !apqb.state
            %v5 = apqb.measure %q4 {mode = "expect"} : f64
            %v6 = arith.sub %v5, %x : f64
            %v7 = arith.div %v6, %y : f64
            %v8 = arith.neg %v7 : f64
            return %v8
          }
        }""",
}

INPUTS = [(0.4, 0.7), (-0.9, 0.2), (0.0, 1.0), (1.3, -2.1)]


class TestMeaningIsPreserved(unittest.TestCase):
    def test_lowered_programs_match_the_ir_reference(self):
        for name, source in PROGRAMS.items():
            module = parse_module(source)
            func = module.func("f")
            lowered = lower_module(module)["f"]
            for args in INPUTS:
                args = list(args)[:len(func.args)]
                want = evaluate(func, args)
                seeds = {reg: value for (file, reg), value
                         in zip(lowered.arg_regs, args)}
                result = QVM().run(lowered.program, r_inputs=seeds)
                got = [result.r[reg][0] if file == "r"
                       else APQBState(*(v[0] for v in result.q[reg]))
                       for file, reg in lowered.result_regs]
                for g, w in zip(got, want):
                    self.assertAlmostEqual(g, w, places=10,
                                           msg=f"{name} on {args}")


class TestAllocation(unittest.TestCase):
    def _lower(self, source):
        return lower_module(parse_module(source))["f"]

    def test_registers_are_recycled(self):
        """A long chain of dead temporaries must not exhaust the file."""
        module = Module("deep")
        b = Builder(module, "f")
        x = b.arg("x")
        state = b.encode(x)
        for _ in range(200):
            state = b.interact(state, state)
        b.ret(b.decode(state))
        module.verify()
        lowered = lower_func(module.func("f"))
        used = {i.dst for i in lowered.program.code if i.spec.dst == "q"}
        self.assertLessEqual(len(used), 2)

    def test_constants_fold_into_immediates(self):
        lowered = self._lower(PROGRAMS["rotate_and_gate"])
        ops = [i.op for i in lowered.program.code]
        self.assertIn(Op.QROT, ops)          # folded form
        self.assertNotIn(Op.QROTR, ops)
        self.assertNotIn(Op.LDI, ops)        # the constant never took a register

    def test_non_constant_coupling_uses_the_register_form(self):
        lowered = self._lower(PROGRAMS["rotate_and_gate"])
        self.assertIn(Op.QGATER, [i.op for i in lowered.program.code])

    def test_constant_used_classically_is_materialised(self):
        lowered = self._lower("""
            module @m {
              func @f(%x: f64) -> (f64) {
                %c1 = arith.const {value = 2.0} : f64
                %v2 = arith.mul %x, %c1 : f64
                return %v2
              }
            }""")
        self.assertIn(Op.LDI, [i.op for i in lowered.program.code])

    def test_running_out_of_registers_is_a_clear_error(self):
        module = Module("wide")
        b = Builder(module, "f")
        x = b.arg("x")
        states = [b.encode(x) for _ in range(40)]
        b.ret(b.decode(states[0]), *[b.decode(s) for s in states[1:]])
        module.verify()
        with self.assertRaisesRegex(LoweringError, "out of Q registers"):
            lower_func(module.func("f"))

    def test_arg_and_result_registers_are_reported(self):
        lowered = self._lower(PROGRAMS["chebyshev"])
        self.assertEqual(lowered.arg_regs, [("r", 0)])
        self.assertEqual(len(lowered.result_regs), 3)
        self.assertEqual(lowered.result_index(0), lowered.result_regs[0][1])

    def test_bad_attribute_is_rejected(self):
        with self.assertRaises(Exception):
            self._lower("""
                module @m {
                  func @f(%x: f64) -> (f64) {
                    %q1 = apqb.encode %x {mode = "linear"} : !apqb.state
                    %v2 = apqb.cheb_t %q1 {k = -2} : f64
                    return %v2
                  }
                }""")

    def test_lowered_programs_always_validate(self):
        for source in PROGRAMS.values():
            self._lower(source).program.validate()


if __name__ == "__main__":
    unittest.main()
