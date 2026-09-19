"""QVM execution: SPMD lanes, memory, determinism, backend agreement."""

import unittest

from qubitbridge import apqb
from qubitbridge.asm import assemble
from qubitbridge.backends import available_backends, get_backend
from qubitbridge.backends.arm64 import Arm64Backend
from qubitbridge.vm import QVM, QVMError

ALL_OPS = """
.mem 2
        QENC      Q0, R0, latent
        QENC      Q1, R1, linear
        QENC      Q2, R0, prob
        QENC      Q3, R1, angle
        QINT      Q4, Q0, Q1
        QMUL      Q5, Q1, Q2
        QPOW      Q6, Q0, 3
        QROT      Q7, Q0, 0.375
        QROTR     Q8, Q0, R1
        QGATE     Q9, Q0, Q1, 0.5
        QGATER    Q10, Q0, Q1, R1
        QNORM     Q11, Q4
        QMOV      Q12, Q4
        QLOAD     Q13, -0.25
        QCORR     R2, Q0, Q1
        QUNC      R3, Q4
        QIMAG     R4, Q6
        QENT      R5, Q0
        QDEC      R6, Q4, linear
        QDEC      R7, Q0, latent
        QDEC      R8, Q4, angle
        QDEC      R9, Q4, prob
        QMEASURE  R10, Q4, expect
        ADD       R11, R0, R1
        SUB       R12, R0, R1
        MUL       R13, R0, R1
        DIV       R14, R0, R1
        NEG       R15, R0
        TANH      R16, R0
        ATANH     R17, R2
        SCALE     R18, R0, 3.5
        ADDI      R19, R0, -1.5
        LDI       R20, 7.25
        MOV       R21, R20
        QSTORE    Q4, [0]
        LOAD      R22, [1]
        HALT
"""


class TestExecution(unittest.TestCase):
    def test_matches_the_scalar_reference(self):
        prog = assemble(ALL_OPS)
        x, y = 0.6, -0.4
        result = QVM().run(prog, r_inputs={0: x, 1: y})

        q0 = apqb.encode(x, "latent")
        q1 = apqb.encode(y, "linear")
        q4 = apqb.interact(q0, q1)
        q6 = apqb.power(q0, 3)

        self.assertAlmostEqual(result.r[2][0], apqb.correlate(q0, q1), places=12)
        self.assertAlmostEqual(result.r[3][0], apqb.uncertainty(q4), places=12)
        self.assertAlmostEqual(result.r[4][0], q6.eta, places=12)
        self.assertAlmostEqual(result.r[5][0], apqb.entropy_z(q0), places=12)
        self.assertAlmostEqual(result.r[6][0], q4.r, places=12)
        self.assertAlmostEqual(result.r[7][0], x, places=8)
        self.assertAlmostEqual(result.r[8][0], q4.theta, places=12)
        self.assertAlmostEqual(result.r[9][0], 0.5 * (1 + q4.r), places=12)
        self.assertAlmostEqual(result.r[10][0], q4.r, places=12)
        self.assertAlmostEqual(result.r[13][0], x * y, places=12)
        self.assertAlmostEqual(result.r[18][0], x * 3.5, places=12)
        self.assertAlmostEqual(result.r[21][0], 7.25, places=12)
        self.assertAlmostEqual(result.state(4).r, q4.r, places=12)

    def test_qstore_writes_r_then_eta(self):
        prog = assemble(ALL_OPS)
        result = QVM().run(prog, r_inputs={0: 0.6, 1: -0.4})
        q4 = apqb.interact(apqb.encode(0.6, "latent"),
                           apqb.encode(-0.4, "linear"))
        self.assertAlmostEqual(result.memory[0][0], q4.r, places=12)
        self.assertAlmostEqual(result.memory[1][0], q4.eta, places=12)
        self.assertAlmostEqual(result.r[22][0], q4.eta, places=12)

    def test_registers_start_in_the_zero_state(self):
        result = QVM().run(assemble("HALT"))
        self.assertEqual(result.state(7), apqb.ZERO_STATE)
        self.assertEqual(result.r[7], [0.0])

    def test_step_count_and_trace(self):
        prog = assemble("QLOAD Q0, 0.5\nQUNC R0, Q0\nHALT")
        result = QVM().run(prog, trace=True)
        self.assertEqual(result.steps, 3)
        self.assertEqual([pc for pc, _ in result.trace], [0, 1, 2])


class TestLanes(unittest.TestCase):
    PROG = "QENC Q0, R0, latent\nQPOW Q1, Q0, 2\nQUNC R1, Q1\nHALT"

    def test_lanes_are_independent(self):
        values = [-1.0, 0.0, 0.5, 2.0]
        wide = QVM().run(assemble(self.PROG), r_inputs={0: values},
                         lanes=len(values))
        for lane, value in enumerate(values):
            single = QVM().run(assemble(self.PROG), r_inputs={0: value})
            self.assertAlmostEqual(wide.r[1][lane], single.r[1][0], places=14)

    def test_scalar_seed_broadcasts(self):
        result = QVM().run(assemble(self.PROG), r_inputs={0: 0.3}, lanes=4)
        self.assertEqual(len(set(result.r[1])), 1)

    def test_wrong_lane_count_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "expected 3 lane values"):
            QVM().run(assemble(self.PROG), r_inputs={0: [1.0, 2.0]}, lanes=3)

    def test_zero_lanes_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "lanes must be"):
            QVM().run(assemble(self.PROG), lanes=0)


class TestMeasurement(unittest.TestCase):
    PROG = "QENC Q0, R0, linear\nQMEASURE R1, Q0, sample\nHALT"

    def test_sampling_is_seeded_and_reproducible(self):
        a = QVM(seed=7).run(assemble(self.PROG), r_inputs={0: 0.5}, lanes=64)
        b = QVM(seed=7).run(assemble(self.PROG), r_inputs={0: 0.5}, lanes=64)
        self.assertEqual(a.r[1], b.r[1])

    def test_different_seeds_differ(self):
        a = QVM(seed=1).run(assemble(self.PROG), r_inputs={0: 0.0}, lanes=64)
        b = QVM(seed=2).run(assemble(self.PROG), r_inputs={0: 0.0}, lanes=64)
        self.assertNotEqual(a.r[1], b.r[1])

    def test_frequencies_follow_the_born_rule(self):
        result = QVM(seed=11).run(assemble(self.PROG), r_inputs={0: 0.5},
                                  lanes=4000)
        ones = sum(1 for v in result.r[1] if v > 0) / 4000.0
        self.assertAlmostEqual(ones, 0.75, places=1)

    def test_draws_are_only_consumed_when_sampling(self):
        prog = assemble("QENC Q0, R0, linear\nQMEASURE R1, Q0, expect\nHALT")
        a = QVM(seed=3).run(prog, r_inputs={0: 0.25})
        self.assertEqual(a.r[1], [0.25])


class TestFaults(unittest.TestCase):
    def test_address_out_of_range(self):
        prog = assemble(".mem 1\nQLOAD Q0, 0.5\nQSTORE Q0, [0]\nHALT")
        with self.assertRaisesRegex(QVMError, r"address \[1\]"):
            QVM().run(prog)

    def test_step_limit(self):
        prog = assemble("NOP\nNOP\nNOP\nHALT")
        with self.assertRaisesRegex(QVMError, "step limit"):
            QVM(step_limit=2).run(prog)

    def test_memory_seed_beyond_segment(self):
        with self.assertRaisesRegex(ValueError, "exceeds .mem"):
            QVM().run(assemble(".mem 1\nHALT"), memory=[1.0, 2.0])

    def test_unknown_register_seed(self):
        with self.assertRaisesRegex(ValueError, "no scalar register"):
            QVM().run(assemble("HALT"), r_inputs={99: 1.0})

    def test_code_emitters_cannot_execute(self):
        with self.assertRaisesRegex(QVMError, "cannot execute"):
            QVM(backend=Arm64Backend())

    def test_unknown_backend_name(self):
        with self.assertRaisesRegex(ValueError, "unknown backend"):
            get_backend("quantum-annealer")


class TestBackendAgreement(unittest.TestCase):
    def test_all_execution_backends_agree(self):
        names = available_backends()
        if len(names) < 2:
            self.skipTest(f"only one execution backend available: {names}")
        prog = assemble(ALL_OPS)
        seeds = {0: [0.6, -1.4, 0.0, 0.9], 1: [-0.4, 0.8, 1.0, -1.0]}
        runs = [QVM(backend=name, seed=5).run(prog, r_inputs=seeds, lanes=4)
                for name in names]
        for index in range(23):
            for other in runs[1:]:
                for got, want in zip(other.r[index], runs[0].r[index]):
                    self.assertAlmostEqual(got, want, places=11,
                                           msg=f"R{index} differs across backends")

    def test_auto_selects_an_executing_backend(self):
        self.assertTrue(get_backend("auto").executes)


if __name__ == "__main__":
    unittest.main()
