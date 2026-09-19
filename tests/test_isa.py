"""Instruction encoding, validation and the .qvm object format."""

import unittest

from qubitbridge.isa import (INSTR_SIZE, MAGIC, NUM_QREGS, SPECS, Instr, Op,
                             Program)


class TestInstructionWords(unittest.TestCase):
    def test_every_word_is_eight_bytes_and_round_trips(self):
        for op, spec in SPECS.items():
            instr = Instr(op,
                          dst=3 if spec.dst else 0,
                          a=4 if spec.a else 0,
                          b=5 if spec.b else 0,
                          imm=0)
            word = instr.pack()
            self.assertEqual(len(word), INSTR_SIZE)
            self.assertEqual(Instr.unpack(word), instr)

    def test_unknown_opcode_is_rejected(self):
        with self.assertRaises(ValueError):
            Instr.unpack(b"\xfe" + b"\x00" * 7)

    def test_every_opcode_has_a_spec(self):
        self.assertEqual(set(SPECS), set(Op))


class TestValidation(unittest.TestCase):
    def test_register_out_of_range(self):
        with self.assertRaisesRegex(ValueError, "out of range"):
            Instr(Op.QINT, dst=NUM_QREGS, a=0, b=0).validate()

    def test_unused_slot_must_be_zero(self):
        with self.assertRaisesRegex(ValueError, "unused"):
            Instr(Op.QUNC, dst=0, a=0, b=7).validate()

    def test_immediate_must_be_absent_when_unused(self):
        with self.assertRaisesRegex(ValueError, "takes no immediate"):
            Instr(Op.QINT, imm=3).validate()

    def test_constant_index_must_exist(self):
        with self.assertRaisesRegex(ValueError, "outside pool"):
            Instr(Op.QLOAD, dst=0, imm=2).validate(const_pool_size=1)

    def test_bad_modes_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "encoding mode"):
            Instr(Op.QENC, dst=0, a=0, imm=99).validate()
        with self.assertRaisesRegex(ValueError, "measurement mode"):
            Instr(Op.QMEASURE, dst=0, a=0, imm=99).validate()

    def test_program_must_end_with_halt(self):
        prog = Program()
        prog.emit(Op.NOP)
        with self.assertRaisesRegex(ValueError, "HALT"):
            prog.validate()

    def test_validate_reports_the_faulting_pc(self):
        prog = Program()
        prog.emit(Op.HALT)
        prog.code.insert(0, Instr(Op.QENC, dst=0, a=0, imm=99))
        with self.assertRaisesRegex(ValueError, r"at pc=0"):
            prog.validate()


class TestConstantPool(unittest.TestCase):
    def test_values_are_interned(self):
        prog = Program()
        self.assertEqual(prog.const(0.5), prog.const(0.5))
        self.assertNotEqual(prog.const(0.5), prog.const(0.25))
        self.assertEqual(len(prog.consts), 2)

    def test_signed_zero_stays_distinct(self):
        prog = Program()
        self.assertNotEqual(prog.const(0.0), prog.const(-0.0))


class TestObjectFile(unittest.TestCase):
    def _program(self) -> Program:
        prog = Program(name="obj", mem_size=4)
        prog.emit(Op.QLOAD, dst=1, imm=prog.const(0.25))
        prog.emit(Op.QPOW, dst=2, a=1, imm=3)
        prog.emit(Op.QCORR, dst=0, a=1, b=2)
        prog.emit(Op.HALT)
        return prog

    def test_round_trip(self):
        prog = self._program()
        again = Program.unpack(prog.pack())
        self.assertEqual(again.name, prog.name)
        self.assertEqual(again.consts, prog.consts)
        self.assertEqual(again.code, prog.code)
        self.assertEqual(again.mem_size, prog.mem_size)

    def test_magic_is_checked(self):
        with self.assertRaisesRegex(ValueError, "not a QVM object"):
            Program.unpack(b"XXXX" + b"\x00" * 32)

    def test_truncation_is_detected(self):
        blob = self._program().pack()
        with self.assertRaisesRegex(ValueError, "truncated"):
            Program.unpack(blob[:-4])

    def test_header_starts_with_the_magic(self):
        self.assertTrue(self._program().pack().startswith(MAGIC))


if __name__ == "__main__":
    unittest.main()
