"""The Qubit Compiler frontend and its classical/APQB partitioning."""

import math
import unittest

import qubitbridge as qb
from qubitbridge.frontend import CompileError, compile_function, compile_source
from qubitbridge.lower import lower_module
from qubitbridge.vm import QVM

#: The frontend reads annotations from source text; this makes the
#: same source importable as a live function for compile_function().
unit = float


def _run(source, *args, min_degree=2):
    module, report = compile_source(source, min_degree=min_degree)
    lowered = lower_module(module)[module.funcs[0].name]
    seeds = {reg: value for (_, reg), value in zip(lowered.arg_regs, args)}
    result = QVM().run(lowered.program, r_inputs=seeds)
    return [result.r[reg][0] for _, reg in lowered.result_regs], report, module


class TestPartitioning(unittest.TestCase):
    def test_bounded_product_goes_to_the_apqb_path(self):
        _, report, module = _run("def f(x: unit, y: unit):\n    return x * y",
                                 0.8, 0.4)
        self.assertEqual(len(report.regions), 1)
        self.assertEqual(report.regions[0].degree, 2)
        self.assertIn("apqb.mul", module.to_text())
        self.assertNotIn("arith.mul", module.to_text())

    def test_unbounded_product_stays_classical(self):
        _, report, module = _run("def f(x, y):\n    return x * y", 3.0, 4.0)
        self.assertEqual(report.regions, [])
        self.assertIn("arith.mul", module.to_text())
        self.assertTrue(any("not provably in [-1, 1]" in n for n in report.notes))

    def test_the_apqb_path_returns_the_classical_answer(self):
        """Routing a region through the QVM must not change the result."""
        for x, y, z in ((0.8, 0.4, 0.5), (-0.9, 0.3, -1.0), (1.0, 1.0, -0.25)):
            got, report, _ = _run(
                "def f(x: unit, y: unit, z: unit):\n    return x * y * z", x, y, z)
            self.assertEqual(len(report.regions), 1)
            self.assertAlmostEqual(got[0], x * y * z, places=14)

    def test_min_degree_controls_the_split(self):
        _, report, _ = _run("def f(x: unit, y: unit):\n    return x * y",
                            0.5, 0.5, min_degree=3)
        self.assertEqual(report.regions, [])

    def test_literals_in_range_count_as_bounded(self):
        got, report, _ = _run("def f(x: unit):\n    return x * 0.5", 0.8)
        self.assertEqual(len(report.regions), 1)
        self.assertAlmostEqual(got[0], 0.4, places=14)

    def test_literal_out_of_range_is_not_bounded(self):
        _, report, _ = _run("def f(x: unit):\n    return x * 2.0", 0.8)
        self.assertEqual(report.regions, [])

    def test_tanh_results_are_bounded(self):
        got, report, _ = _run("def f(a, b):\n    return tanh(a) * tanh(b)",
                              1.5, -2.0)
        self.assertEqual(len(report.regions), 1)
        self.assertAlmostEqual(got[0], math.tanh(1.5) * math.tanh(-2.0), places=12)

    def test_apqb_forces_a_region(self):
        got, report, _ = _run("def f(a, b):\n    return apqb(a * b)", 0.5, 0.5)
        self.assertEqual(len(report.regions), 1)
        self.assertIn("forced", report.regions[0].reason)
        self.assertAlmostEqual(got[0], 0.25, places=14)

    def test_classical_pins_a_region(self):
        _, report, module = _run(
            "def f(x: unit, y: unit):\n    return classical(x * y)", 0.5, 0.5)
        self.assertEqual(report.regions, [])
        self.assertIn("arith.mul", module.to_text())

    def test_report_summary_is_informative(self):
        _, report, _ = _run("def f(x: unit, y: unit):\n    return x * y", 0.5, 0.5)
        summary = report.summary()
        self.assertIn("APQB regions: 1", summary)
        self.assertIn("degree-2", summary)
        self.assertGreater(report.apqb_fraction, 0.0)


class TestIntrinsics(unittest.TestCase):
    def test_correlation_and_uncertainty(self):
        got, _, _ = _run("def f(x: unit, y: unit):\n"
                         "    return corr(x, y), unc(x), entropy(x)", 0.6, -0.2)
        ex, ey = qb.encode(0.6, "linear"), qb.encode(-0.2, "linear")
        self.assertAlmostEqual(got[0], ex.r * ey.r + ex.eta * ey.eta, places=12)
        self.assertAlmostEqual(got[1], ex.eta, places=12)
        self.assertAlmostEqual(got[2], -sum(
            p * math.log2(p) for p in (0.8, 0.2)), places=12)

    def test_chebyshev_degree_must_be_literal(self):
        with self.assertRaisesRegex(CompileError, "literal integer degree"):
            compile_source("def f(x: unit, k):\n    return cheb_t(x, k)")

    def test_chebyshev_matches_the_closed_form(self):
        got, _, _ = _run("def f(x: unit):\n    return cheb_t(x, 3)", 0.4)
        self.assertAlmostEqual(got[0], math.cos(3 * math.acos(0.4)), places=10)

    def test_rotation_and_gate(self):
        got, _, _ = _run("def f(x: unit, p: unit, j):\n"
                         "    return rot(x, p), gate(x, p, j)", 0.5, 0.25, 0.3)
        rotated = qb.apqb.rotate(qb.encode(0.5, "linear"), 0.25)
        gated = qb.apqb.gate(qb.encode(0.5, "linear"),
                             qb.encode(0.25, "linear"), 0.3)
        self.assertAlmostEqual(got[0], rotated.r, places=12)
        self.assertAlmostEqual(got[1], gated.r, places=12)

    def test_measure_uses_the_expectation(self):
        got, _, _ = _run("def f(x: unit):\n    return measure(x)", 0.7)
        self.assertAlmostEqual(got[0], 0.7, places=12)


class TestSourceHandling(unittest.TestCase):
    def test_assignments_and_common_subexpressions(self):
        got, _, module = _run("def f(x: unit, y: unit):\n"
                              "    p = x * y\n"
                              "    return p + p, p", 0.5, 0.4)
        self.assertAlmostEqual(got[0], 0.4, places=14)
        self.assertEqual(module.to_text().count("apqb.mul"), 1)

    def test_docstrings_are_skipped(self):
        got, _, _ = _run('def f(x: unit):\n    "doc"\n    return x * x', 0.5)
        self.assertAlmostEqual(got[0], 0.25, places=14)

    def test_indented_source_is_accepted(self):
        def kernel(x: unit, y: unit):
            return x * y

        module, report = compile_function(kernel)
        self.assertEqual(module.funcs[0].name, "kernel")

    def test_live_function_via_run_helper(self):
        self.assertAlmostEqual(
            qb.run("def f(x: unit, y: unit):\n    return x * y", 0.8, 0.4)[0][0],
            0.32, places=14)


class TestErrors(unittest.TestCase):
    def _fails(self, source, pattern):
        with self.assertRaisesRegex(CompileError, pattern):
            compile_source(source)

    def test_syntax_error(self):
        self._fails("def f(x:\n", "syntax error")

    def test_two_functions(self):
        self._fails("def f(x):\n    return x\ndef g(x):\n    return x",
                    "exactly one function")

    def test_unknown_name(self):
        self._fails("def f(x):\n    return y", "not defined")

    def test_unknown_call(self):
        self._fails("def f(x):\n    return teleport(x)", "unknown function")

    def test_wrong_arity(self):
        self._fails("def f(x):\n    return corr(x)", "takes 2 argument")

    def test_unsupported_statement(self):
        self._fails("def f(x):\n    for i in x:\n        pass\n    return x",
                    "unsupported statement")

    def test_unsupported_operator(self):
        self._fails("def f(x):\n    return x ** 2", "unsupported binary operator")

    def test_missing_return(self):
        self._fails("def f(x):\n    y = x", "no return statement")

    def test_tuple_assignment(self):
        self._fails("def f(x):\n    a, b = x, x\n    return a", "simple")

    def test_varargs(self):
        self._fails("def f(*args):\n    return 1.0", "positional")

    def test_non_numeric_literal(self):
        self._fails("def f(x):\n    return x * 'a'", "numeric literals")

    def test_indirect_call(self):
        self._fails("def f(x):\n    return math.tanh(x)", "direct calls")


if __name__ == "__main__":
    unittest.main()
