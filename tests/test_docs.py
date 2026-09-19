"""Doctests, and a check that the documentation's tables match the code."""

import doctest
import pathlib
import unittest

import qubitbridge
from qubitbridge.ir import OP_SPECS
from qubitbridge.isa import SPECS

DOCS = pathlib.Path(__file__).resolve().parent.parent / "docs"


def load_tests(loader, tests, ignore):
    for module in (qubitbridge, qubitbridge.apqb, qubitbridge.ir,
                   qubitbridge.isa, qubitbridge.asm, qubitbridge.lower,
                   qubitbridge.vm, qubitbridge.frontend):
        tests.addTests(doctest.DocTestSuite(module))
    return tests


class TestDocumentationIsCurrent(unittest.TestCase):
    """Cheap insurance against a table drifting away from the implementation."""

    def test_every_opcode_is_documented(self):
        text = (DOCS / "isa.md").read_text()
        for spec in SPECS.values():
            self.assertIn(f"`{spec.name}`", text,
                          f"{spec.name} is missing from docs/isa.md")

    def test_every_ir_operation_is_documented(self):
        text = (DOCS / "apqb-ir.md").read_text()
        for name in OP_SPECS:
            self.assertIn(f"`{name}`", text,
                          f"{name} is missing from docs/apqb-ir.md")

    def test_every_doc_page_exists(self):
        for page in ("architecture.md", "apqb-ir.md", "isa.md", "backends.md"):
            self.assertTrue((DOCS / page).is_file(), page)


if __name__ == "__main__":
    unittest.main()
