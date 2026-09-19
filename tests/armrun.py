"""Cross-compile and execute emitted AArch64 code, for differential testing.

Builds the emitter's assembly together with a small C harness, links it for
aarch64 and runs it -- natively on an ARM host, otherwise under ``qemu-aarch64``.
When neither is available the helpers report that, and the ARM64 tests fall back
to assemble-only checks.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import tempfile

from qubitbridge.backends.arm64 import Arm64Backend
from qubitbridge.isa import NUM_QREGS, NUM_RREGS, Program

_HARNESS = r"""
#include <stdio.h>
#include <stdlib.h>

void %(symbol)s(double *qbase, double *rbase, const double *cpool,
                double *scratch, unsigned long n);

static const double cpool[] = {%(consts)s};

int main(void) {
    const unsigned long n = %(lanes)luUL;
    double *qbase = calloc(%(nq)d * 2 * n, sizeof(double));
    double *rbase = calloc(%(nr)d * n, sizeof(double));
    double *scratch = calloc(2 * n, sizeof(double));
    if (!qbase || !rbase || !scratch) return 1;

    for (int i = 0; i < %(nq)d; i++)
        for (unsigned long l = 0; l < n; l++)
            qbase[(2 * i) * n + l] = 1.0;   /* |0>: r = 1, eta = 0 */

%(seeds)s
    %(symbol)s(qbase, rbase, cpool, scratch, n);

    for (int i = 0; i < %(nr)d; i++) {
        for (unsigned long l = 0; l < n; l++)
            printf("%%.17g ", rbase[i * n + l]);
        printf("\n");
    }
    for (int i = 0; i < %(nq)d; i++) {
        for (unsigned long l = 0; l < n; l++)
            printf("%%.17g %%.17g ", qbase[(2 * i) * n + l],
                   qbase[(2 * i + 1) * n + l]);
        printf("\n");
    }
    free(qbase); free(rbase); free(scratch);
    return 0;
}
"""


def toolchain() -> tuple[str, list[str]] | None:
    """Return ``(compiler, runner)`` for building and running aarch64 binaries."""
    native = platform.machine().lower() in ("aarch64", "arm64")
    if native:
        for cc in ("cc", "gcc", "clang"):
            if shutil.which(cc):
                return cc, []
    cross = shutil.which("aarch64-linux-gnu-gcc")
    runner = shutil.which("qemu-aarch64")
    if cross and runner:
        return cross, [runner]
    if cross and native:  # pragma: no cover - unusual host
        return cross, []
    return None


def available() -> bool:
    return toolchain() is not None


def run_program(program: Program, r_inputs: dict[int, list[float]],
                lanes: int = 1) -> tuple[dict[int, list[float]],
                                         dict[int, tuple[list[float], list[float]]]]:
    """Execute ``program`` as real AArch64 code; return its R and Q files."""
    tools = toolchain()
    if tools is None:  # pragma: no cover - guarded by available()
        raise RuntimeError("no aarch64 toolchain")
    cc, runner = tools

    asm = Arm64Backend().emit(program)
    symbol = "qb_prog_" + "".join(
        ch if ch.isalnum() else "_" for ch in program.name)

    seeds = []
    for index, values in r_inputs.items():
        lane_values = values if isinstance(values, list) else [values] * lanes
        for lane, value in enumerate(lane_values):
            seeds.append(f"    rbase[{index} * n + {lane}] = {value!r};")

    harness = _HARNESS % {
        "symbol": symbol,
        "consts": ", ".join(repr(c) for c in program.consts) or "0.0",
        "lanes": lanes,
        "nq": NUM_QREGS,
        "nr": NUM_RREGS,
        "seeds": "\n".join(seeds),
    }

    with tempfile.TemporaryDirectory() as tmp:
        asm_path = os.path.join(tmp, "prog.s")
        c_path = os.path.join(tmp, "harness.c")
        exe = os.path.join(tmp, "harness")
        with open(asm_path, "w") as fh:
            fh.write(asm)
        with open(c_path, "w") as fh:
            fh.write(harness)

        build = subprocess.run(
            [cc, "-static", "-O2", "-o", exe, c_path, asm_path, "-lm"],
            capture_output=True, text=True)
        if build.returncode != 0:
            raise RuntimeError(f"aarch64 build failed:\n{build.stderr}")

        run = subprocess.run(runner + [exe], capture_output=True, text=True,
                             timeout=120)
        if run.returncode != 0:
            raise RuntimeError(f"aarch64 run failed ({run.returncode}):\n"
                               f"{run.stderr}")

    lines = run.stdout.strip().splitlines()
    rfile = {i: [float(v) for v in lines[i].split()] for i in range(NUM_RREGS)}
    qfile = {}
    for i in range(NUM_QREGS):
        flat = [float(v) for v in lines[NUM_RREGS + i].split()]
        qfile[i] = (flat[0::2], flat[1::2])
    return rfile, qfile
