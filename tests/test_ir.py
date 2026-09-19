"""APQB IR: construction, verification and the textual round trip."""

import unittest

from qubitbridge.ir import (F64, OP_SPECS, STATE, Builder, IRError, Module,
                            Value, parse_module)

SOURCE = """
module @demo {
  func @kernel(%x: f64, %y: f64) -> (f64, f64) {
    %q1 = apqb.encode %x {mode = "latent"} : !apqb.state
    %q2 = apqb.encode %y {mode = "linear"} : !apqb.state
    %c3 = arith.const {value = 0.25} : f64
    %q4 = apqb.rotate %q1, %c3 : !apqb.state
    %q5 = apqb.interact %q4, %q2 : !apqb.state
    %q6 = apqb.gate %q5, %q2, %c3 : !apqb.state
    %v7 = apqb.correlate %q5, %q6 : f64
    %v8 = apqb.cheb_t %q6 {k = 3} : f64
    return %v7, %v8
  }
}
"""


class TestParsing(unittest.TestCase):
    def test_round_trip(self):
        module = parse_module(SOURCE)
        self.assertEqual(module.name, "demo")
        again = parse_module(module.to_text())
        self.assertEqual(again.to_text(), module.to_text())

    def test_comments_are_ignored(self):
        module = parse_module(SOURCE.replace("module @demo {",
                                             "// header\nmodule @demo {"))
        self.assertEqual(module.name, "demo")

    def test_attributes_keep_their_python_type(self):
        op = parse_module(SOURCE).func("kernel").body[7]
        self.assertEqual(op.name, "apqb.cheb_t")
        self.assertIsInstance(op.attrs["k"], int)
        const = parse_module(SOURCE).func("kernel").body[2]
        self.assertIsInstance(const.attrs["value"], float)

    def test_unknown_op_is_rejected(self):
        with self.assertRaisesRegex(IRError, "unknown operation"):
            parse_module(SOURCE.replace("apqb.interact", "apqb.teleport"))

    def test_use_before_definition_is_rejected(self):
        with self.assertRaisesRegex(IRError, "not defined before use"):
            parse_module(SOURCE.replace("%q5 = apqb.interact %q4, %q2",
                                        "%q5 = apqb.interact %q9, %q2"))

    def test_unparsable_text(self):
        with self.assertRaises(IRError):
            parse_module("module @x { func @f() -> () { $ } }")


class TestVerifier(unittest.TestCase):
    def _module(self) -> Module:
        return parse_module(SOURCE)

    def test_type_mismatch(self):
        module = self._module()
        fn = module.func("kernel")
        fn.body[4].operands[0] = fn.args[0]          # f64 where a state is wanted
        with self.assertRaisesRegex(IRError, "expected !apqb.state"):
            module.verify()

    def test_operand_count(self):
        module = self._module()
        module.func("kernel").body[4].operands.pop()
        with self.assertRaisesRegex(IRError, "expects 2 operand"):
            module.verify()

    def test_missing_attribute(self):
        module = self._module()
        module.func("kernel").body[0].attrs.clear()
        with self.assertRaisesRegex(IRError, "missing attribute"):
            module.verify()

    def test_unexpected_attribute(self):
        module = self._module()
        module.func("kernel").body[4].attrs["spin"] = 1
        with self.assertRaisesRegex(IRError, "unexpected attribute"):
            module.verify()

    def test_ssa_redefinition(self):
        module = self._module()
        fn = module.func("kernel")
        fn.body[4].result = fn.body[3].result
        with self.assertRaisesRegex(IRError, "SSA"):
            module.verify()

    def test_wrong_result_type(self):
        module = self._module()
        op = module.func("kernel").body[6]           # apqb.correlate -> f64
        op.result = Value(op.result.name, STATE)
        with self.assertRaisesRegex(IRError, "op yields f64"):
            module.verify()

    def test_duplicate_function(self):
        module = self._module()
        module.funcs.append(module.funcs[0])
        with self.assertRaisesRegex(IRError, "duplicate function"):
            module.verify()

    def test_returning_an_undefined_value(self):
        module = self._module()
        module.func("kernel").results[0] = Value("nope", F64)
        with self.assertRaisesRegex(IRError, "undefined"):
            module.verify()


class TestBuilder(unittest.TestCase):
    def test_builds_a_verifiable_module(self):
        module = Module("built")
        b = Builder(module, "f")
        x, y = b.arg("x"), b.arg("y")
        q = b.interact(b.encode(x), b.encode(y, "linear"))
        b.ret(b.decode(q), b.emit("apqb.uncertainty", q))
        module.verify()
        self.assertEqual(len(module.func("f").body), 5)
        self.assertEqual(module.func("f").results[0].type, F64)

    def test_unknown_op_rejected_at_build_time(self):
        b = Builder(Module(), "f")
        with self.assertRaisesRegex(IRError, "unknown operation"):
            b.emit("apqb.nope")

    def test_missing_function_lookup(self):
        with self.assertRaises(KeyError):
            Module("m").func("absent")

    def test_every_spec_has_a_known_result_type(self):
        for spec in OP_SPECS.values():
            self.assertIn(spec.result, (F64, STATE))
            for operand in spec.operands:
                self.assertIn(operand, (F64, STATE))


if __name__ == "__main__":
    unittest.main()
