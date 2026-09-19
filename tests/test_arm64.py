"""AArch64 code generation: it assembles, and it computes the right numbers.

When a cross toolchain (or an ARM host) is present, every case here is compiled
and executed, and the results are compared against the portable backend.  On a
host without one, the execution assertions skip and the assemble-only checks
still run.
"""

import unittest

from qubitbridge.asm import assemble
from qubitbridge.backends.arm64 import (Arm64Backend, Arm64UnsupportedOp,
                                        assemble_text, emit_driver,
                                        have_assembler, kernel_library)
from qubitbridge.ir import parse_module
from qubitbridge.lower import lower_module
from qubitbridge.vm import QVM

from . import armrun

EVERY_OP = """
        QENC      Q0, R0, latent
        QENC      Q1, R1, linear
        QENC      Q2, R0, prob
        QENC      Q3, R1, angle
        QINT      Q4, Q0, Q1
        QMUL      Q5, Q0, Q1
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
        MIN       R22, R0, R1
        MAX       R23, R0, R1
        TANH      R16, R0
        ATANH     R17, R2
        SCALE     R18, R0, 3.5
        ADDI      R19, R0, -1.5
        LDI       R20, 7.25
        MOV       R21, R20
        HALT
"""

LANES = [0.6, -1.2, 0.0, 0.95, 2.5, -1.0]
OTHER = [-0.4, 0.9, 1.0, -1.0, 0.25, 1.0]


@unittest.skipUnless(have_assembler(), "no AArch64 assembler available")
class TestAssembles(unittest.TestCase):
    def test_kernel_library_assembles(self):
        ok, output = assemble_text(kernel_library())
        self.assertTrue(ok, output)

    def test_driver_assembles_for_every_opcode(self):
        ok, output = Arm64Backend().verify(assemble(EVERY_OP))
        self.assertTrue(ok, output)

    def test_driver_only_output_has_no_kernels(self):
        text = Arm64Backend().emit(assemble("QLOAD Q0, 0.5\nHALT"),
                                   with_library=False)
        self.assertNotIn("qb_interact:", text)
        self.assertIn("qb_prog_", text)

    def test_symbol_name_is_sanitised(self):
        prog = assemble(".module my.kernel-1\nHALT")
        self.assertIn("qb_prog_my_kernel_1", emit_driver(prog))


class TestUnsupported(unittest.TestCase):
    def test_memory_ops_are_rejected_with_a_reason(self):
        for source in (".mem 2\nQLOAD Q0, 0.5\nQSTORE Q0, [0]\nHALT",
                       ".mem 2\nLOAD R0, [0]\nHALT",
                       ".mem 2\nSTORE R0, [0]\nHALT"):
            with self.assertRaises(Arm64UnsupportedOp):
                Arm64Backend().emit(assemble(source))

    def test_sampling_is_rejected_with_a_reason(self):
        prog = assemble("QLOAD Q0, 0.5\nQMEASURE R0, Q0, sample\nHALT")
        with self.assertRaisesRegex(Arm64UnsupportedOp, "seeded PRNG"):
            Arm64Backend().emit(prog)

    def test_the_backend_refuses_to_execute(self):
        self.assertFalse(Arm64Backend().executes)


@unittest.skipUnless(armrun.available(),
                     "no AArch64 toolchain (need an ARM host, or "
                     "aarch64-linux-gnu-gcc + qemu-aarch64)")
class TestNumericsOnRealHardware(unittest.TestCase):
    """Differential test: emitted machine code vs the portable reference."""

    def _compare(self, program, inputs, lanes, places=12):
        arm_r, arm_q = armrun.run_program(program, inputs, lanes=lanes)
        ref = QVM().run(program, r_inputs=inputs, lanes=lanes)
        for index, values in ref.r.items():
            for lane, want in enumerate(values):
                self.assertAlmostEqual(arm_r[index][lane], want, places=places,
                                       msg=f"R{index} lane {lane}")
        for index, (rs, etas) in ref.q.items():
            for lane, (want_r, want_eta) in enumerate(zip(rs, etas)):
                self.assertAlmostEqual(arm_q[index][0][lane], want_r,
                                       places=places, msg=f"Q{index}.r lane {lane}")
                self.assertAlmostEqual(arm_q[index][1][lane], want_eta,
                                       places=places,
                                       msg=f"Q{index}.eta lane {lane}")

    def test_every_opcode_matches_the_portable_backend(self):
        self._compare(assemble(EVERY_OP), {0: LANES, 1: OTHER}, len(LANES))

    def test_odd_lane_count_exercises_the_scalar_tail(self):
        prog = assemble("QENC Q0, R0, latent\nQENC Q1, R1, linear\n"
                        "QINT Q2, Q0, Q1\nQCORR R2, Q0, Q1\nHALT")
        for lanes in (1, 2, 3, 5):
            self._compare(prog, {0: LANES[:lanes], 1: OTHER[:lanes]}, lanes)

    def test_destination_may_alias_a_source(self):
        prog = assemble("QENC Q0, R0, latent\nQINT Q0, Q0, Q0\n"
                        "QMUL Q0, Q0, Q0\nQNORM Q0, Q0\nQPOW Q0, Q0, 3\n"
                        "QROT Q0, Q0, 0.25\nQUNC R1, Q0\nHALT")
        self._compare(prog, {0: LANES}, len(LANES))

    def test_lowered_ir_runs_on_arm(self):
        module = parse_module("""
            module @qbnn {
              func @f(%x: f64, %j: f64) -> (f64, f64) {
                %q1 = apqb.encode %x {mode = "latent"} : !apqb.state
                %c2 = arith.const {value = 0.3} : f64
                %q3 = apqb.rotate %q1, %c2 : !apqb.state
                %q4 = apqb.gate %q3, %q1, %j : !apqb.state
                %v5 = apqb.correlate %q3, %q4 : f64
                %v6 = apqb.cheb_t %q4 {k = 4} : f64
                return %v5, %v6
              }
            }""")
        lowered = lower_module(module)["f"]
        self._compare(lowered.program, {0: LANES, 1: OTHER}, len(LANES))


if __name__ == "__main__":
    unittest.main()
