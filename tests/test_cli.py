"""The qvm command line, end to end through every subcommand."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from qubitbridge.backends.arm64 import have_assembler
from qubitbridge.cli import main
from qubitbridge.isa import Program

KERNEL_PY = """
def kernel(x: unit, y: unit):
    z = x * y * x
    return z + corr(x, y), unc(x)
"""

KERNEL_ASM = """
.module hand
        QENC      Q0, R0, latent
        QPOW      Q1, Q0, 3
        QUNC      R1, Q1
        HALT
"""


@contextlib.contextmanager
def capture():
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        yield out, err


class CLITest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.py = self.dir / "kernel.py"
        self.py.write_text(KERNEL_PY)
        self.qasm = self.dir / "hand.qasm"
        self.qasm.write_text(KERNEL_ASM)

    def run_cli(self, *argv):
        with capture() as (out, err):
            code = main(list(argv))
        return code, out.getvalue(), err.getvalue()


class TestCommands(CLITest):
    def test_info(self):
        code, out, _ = self.run_cli("info")
        self.assertEqual(code, 0)
        self.assertIn("QINT", out)
        self.assertIn("backends:", out)

    def test_compile_prints_ir(self):
        code, out, _ = self.run_cli("compile", str(self.py))
        self.assertEqual(code, 0)
        self.assertIn("apqb.mul", out)
        self.assertIn("module @kernel", out)

    def test_compile_report_goes_to_stderr(self):
        _, out, err = self.run_cli("compile", str(self.py), "--report")
        self.assertIn("APQB regions", err)
        self.assertNotIn("APQB regions", out)

    def test_compile_to_file_then_lower(self):
        ir_path = self.dir / "kernel.ir"
        self.run_cli("compile", str(self.py), "-o", str(ir_path))
        code, out, _ = self.run_cli("lower", str(ir_path), "--addresses")
        self.assertEqual(code, 0)
        self.assertIn("QMUL", out)
        self.assertIn("0000:", out)

    def test_asm_then_disasm_round_trip(self):
        obj = self.dir / "hand.qvm"
        code, _, err = self.run_cli("asm", str(self.qasm), "-o", str(obj))
        self.assertEqual(code, 0)
        self.assertIn("instructions", err)
        self.assertTrue(Program.unpack(obj.read_bytes()).code)
        code, out, _ = self.run_cli("disasm", str(obj))
        self.assertEqual(code, 0)
        self.assertIn("QPOW", out)

    def test_run_from_python(self):
        code, out, _ = self.run_cli("run", str(self.py), "-i", "R0=0.8",
                                    "-i", "R1=0.4")
        self.assertEqual(code, 0)
        self.assertIn("result[0]", out)
        self.assertIn("backend=", out)

    def test_run_multiple_lanes(self):
        code, out, _ = self.run_cli("run", str(self.py), "-i", "R0=0.8,0.2",
                                    "-i", "R1=0.4,-0.5", "--lanes", "2")
        self.assertEqual(code, 0)
        self.assertIn("lanes=2", out)

    def test_run_assembly_reports_written_registers(self):
        code, out, _ = self.run_cli("run", str(self.qasm), "-i", "R0=0.5")
        self.assertEqual(code, 0)
        self.assertIn("R1 =", out)

    def test_run_trace(self):
        _, _, err = self.run_cli("run", str(self.qasm), "-i", "R0=0.5", "--trace")
        self.assertIn("QPOW", err)

    @unittest.skipUnless(have_assembler(), "no AArch64 assembler available")
    def test_emit_arm64_with_check(self):
        code, out, err = self.run_cli("emit-arm64", str(self.py), "--check")
        self.assertEqual(code, 0)
        self.assertIn("qb_prog_", out)
        self.assertIn("llvm-mc: ok", err)

    def test_emit_arm64_driver_only(self):
        code, out, _ = self.run_cli("emit-arm64", str(self.qasm), "--driver-only")
        self.assertEqual(code, 0)
        self.assertNotIn("qb_interact:", out)

    def test_emit_arm64_rejects_unsupported(self):
        source = self.dir / "mem.qasm"
        source.write_text(".mem 2\nQLOAD Q0, 0.5\nQSTORE Q0, [0]\nHALT\n")
        code, _, err = self.run_cli("emit-arm64", str(source))
        self.assertEqual(code, 1)
        self.assertIn("no ARM64 lowering", err)


class TestArgumentErrors(CLITest):
    def test_unknown_extension(self):
        bad = self.dir / "kernel.txt"
        bad.write_text("nope")
        with self.assertRaisesRegex(SystemExit, "unknown input type"):
            self.run_cli("lower", str(bad))

    def test_malformed_input_flag(self):
        with self.assertRaisesRegex(SystemExit, "bad --input"):
            self.run_cli("run", str(self.qasm), "-i", "R0")

    def test_non_numeric_input(self):
        with self.assertRaisesRegex(SystemExit, "bad --input value"):
            self.run_cli("run", str(self.qasm), "-i", "R0=abc")

    def test_lane_count_mismatch(self):
        with self.assertRaisesRegex(SystemExit, "--lanes"):
            self.run_cli("run", str(self.qasm), "-i", "R0=1,2,3", "--lanes", "2")

    def test_named_input_without_ir(self):
        with self.assertRaisesRegex(SystemExit, "without IR"):
            self.run_cli("run", str(self.qasm), "-i", "x=1.0")


if __name__ == "__main__":
    unittest.main()
