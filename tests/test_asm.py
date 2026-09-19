"""Textual assembly: parsing, printing and the round trip."""

import unittest

from qubitbridge.asm import AsmError, assemble, disassemble
from qubitbridge.isa import Op

SOURCE = """
.module demo
.mem 4
        QENC      Q0, R0, latent
        QENC      Q1, R1, prob
        QINT      Q2, Q0, Q1
        QROT      Q2, Q2, 0.125
        QROTR     Q3, Q2, R2
        QGATE     Q4, Q0, Q1, 0.5
        QGATER    Q5, Q0, Q1, R3
        QPOW      Q6, Q2, 4
        QCORR     R4, Q0, Q1
        QUNC      R5, Q2
        QIMAG     R6, Q2
        QENT      R7, Q2
        QMEASURE  R8, Q2, expect
        QSTORE    Q2, [0]
        LOAD      R9, [0]
        SCALE     R9, R9, -2.5
        MIN       R10, R4, R5
        MAX       R11, R4, R5
        HALT
"""


class TestRoundTrip(unittest.TestCase):
    def test_assemble_disassemble_is_a_fixed_point(self):
        first = assemble(SOURCE)
        text = disassemble(first)
        second = assemble(text)
        self.assertEqual(first.code, second.code)
        self.assertEqual(first.consts, second.consts)
        self.assertEqual(disassemble(second), text)

    def test_every_opcode_survives(self):
        prog = assemble(SOURCE)
        self.assertIn(Op.QGATER, [i.op for i in prog.code])
        self.assertEqual(prog.name, "demo")
        self.assertEqual(prog.mem_size, 4)

    def test_comments_and_blank_lines_are_ignored(self):
        prog = assemble("; a comment\n\n  QLOAD Q0, 0.5  # trailing\n  HALT\n")
        self.assertEqual(len(prog.code), 2)

    def test_addresses_option(self):
        text = disassemble(assemble("QLOAD Q0, 0.5\nHALT"), addresses=True)
        self.assertIn("0000:", text)


class TestErrors(unittest.TestCase):
    def _fails(self, source, pattern):
        with self.assertRaisesRegex(AsmError, pattern):
            assemble(source)

    def test_unknown_mnemonic(self):
        self._fails("FROBNICATE Q0\nHALT", "unknown instruction")

    def test_wrong_operand_count(self):
        self._fails("QINT Q0, Q1\nHALT", "expected 3 operand")

    def test_wrong_register_file(self):
        self._fails("QINT Q0, R1, Q2\nHALT", "expected a Q register")

    def test_bad_encoding_name(self):
        self._fails("QENC Q0, R0, teleport\nHALT", "unknown encoding")

    def test_bad_address_syntax(self):
        self._fails("QSTORE Q0, 4\nHALT", "expected an address")

    def test_unknown_directive(self):
        self._fails(".frobnicate 1\nHALT", "unknown directive")

    def test_missing_halt(self):
        self._fails("QLOAD Q0, 0.5", "HALT")

    def test_line_number_is_reported(self):
        self._fails("QLOAD Q0, 0.5\nQINT Q0\nHALT", "line 2")


if __name__ == "__main__":
    unittest.main()
