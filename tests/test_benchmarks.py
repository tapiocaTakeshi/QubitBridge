"""Smoke tests for benchmarks/: they must at least run and agree with the VM.

These are not performance assertions -- timing on a shared CI runner is not
a reliable signal, and benchmarks/*.py's own docstrings already carry the
measured numbers from a real run. This just guards against the scripts
bit-rotting: the reference math must still match what the QVM computes.
"""

import unittest

from benchmarks.apqb_pattern_bench import apqb_chain, check_agreement, classical_chain
from benchmarks.lean_kernel_bench import PROGRAM, scalar_reference
from benchmarks.qbnn_layer_bench import build_ir, make_weights, reference
from qubitbridge.asm import assemble
from qubitbridge.lower import lower_module
from qubitbridge.vm import QVM


class TestQbnnLayerBench(unittest.TestCase):
    def test_reference_matches_the_qvm(self):
        in_dim, out_dim = 4, 3
        w = make_weights(in_dim, out_dim, seed=7)
        lowered = lower_module(build_ir(in_dim, out_dim, w))["layer"]

        for h in ([0.4, -0.9, 0.2, 1.1], [0.0, 0.0, 0.0, 0.0], [-2.0, 2.0, -2.0, 2.0]):
            want = reference(w, in_dim, out_dim, h)
            seeds = {reg: value for (_, reg), value in zip(lowered.arg_regs, h)}
            result = QVM().run(lowered.program, r_inputs=seeds)
            got = [result.r[reg][0] for _, reg in lowered.result_regs]
            for g, t in zip(got, want):
                self.assertAlmostEqual(g, t, places=10)

    def test_batched_run_matches_the_per_sample_reference(self):
        in_dim, out_dim = 3, 2
        w = make_weights(in_dim, out_dim, seed=3)
        lowered = lower_module(build_ir(in_dim, out_dim, w))["layer"]

        batch = [[0.5, -0.5, 0.2], [1.0, 1.0, 1.0], [-1.0, 0.3, -0.7]]
        seeds = {reg: [sample[i] for sample in batch]
                 for i, (_, reg) in enumerate(lowered.arg_regs)}
        result = QVM(backend="auto").run(lowered.program, r_inputs=seeds, lanes=len(batch))

        for lane, sample in enumerate(batch):
            want = reference(w, in_dim, out_dim, sample)
            got = [result.r[reg][lane] for _, reg in lowered.result_regs]
            for g, t in zip(got, want):
                self.assertAlmostEqual(g, t, places=9)


class TestLeanKernelBench(unittest.TestCase):
    def test_scalar_reference_matches_the_qvm(self):
        prog = assemble(PROGRAM)
        for x, y in ((0.6, -0.4), (0.0, 0.0), (-1.0, 1.0), (0.99, -0.99)):
            want = scalar_reference(x, y)
            result = QVM().run(prog, r_inputs={0: x, 1: y})
            got = (result.r[2][0], result.r[3][0], result.r[4][0], result.r[5][0])
            for g, t in zip(got, want):
                self.assertAlmostEqual(g, t, places=10)


class TestApqbPatternBench(unittest.TestCase):
    def test_classical_and_apqb_chains_agree(self):
        """The two patterns compared must actually compute the same thing."""
        for k in (2, 3, 4, 8, 16):
            check_agreement(classical_chain(k), apqb_chain(k), k)

    def test_apqb_chain_has_roughly_double_the_instructions(self):
        for k in (2, 4, 8, 16):
            classical = classical_chain(k)
            apqb = apqb_chain(k)
            # k encodes + (k-1) muls + 1 decode + 1 halt, vs (k-1) muls + 1 halt.
            self.assertEqual(len(apqb.program.code), 2 * k + 1)
            self.assertEqual(len(classical.program.code), k)


if __name__ == "__main__":
    unittest.main()
