"""ARGUS-E01-T04: X0 reference table is generated from raw outputs with exact oracles."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from argus.cli import main
from argus.experiments.x0 import RAW_FILE, cases, export
from tools import x0_table

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "docs/verification/X0_raw.json"
DOC = ROOT / "docs/verification/X0.md"


def test_export_reproduces_committed_raw_outputs_bytewise(tmp_path: Path) -> None:
    path = export(tmp_path / "x0")
    assert path.name == RAW_FILE
    assert path.read_bytes() == RAW.read_bytes()


def test_cli_exports_and_refuses_to_overwrite(tmp_path: Path) -> None:
    target = tmp_path / "cli"
    assert main(["x0-reference", "--output", str(target)]) == 0
    assert (target / RAW_FILE).read_bytes() == RAW.read_bytes()
    assert main(["x0-reference", "--output", str(target)]) == 1


def test_table_in_document_is_generated_from_raw_outputs() -> None:
    raw = json.loads(RAW.read_text(encoding="utf-8"))
    block, all_ok = x0_table.table(raw)
    document = DOC.read_text(encoding="utf-8")
    assert x0_table.splice(document, block) == document
    assert all_ok


def test_every_case_passes_its_oracle() -> None:
    raw = json.loads(RAW.read_text(encoding="utf-8"))
    for case in raw["cases"]:
        _, ok = x0_table.evaluate(case)
        assert ok, case["id"]


def test_no_monte_carlo_value_is_used() -> None:
    kinds = {case["kind"] for case in cases()}
    assert kinds <= {"exact", "misspecified", "error"}
    for module in (
        "experiments/x0.py",
        "beliefs/exact_micro.py",
        "beliefs/joint_filter.py",
        "dynamics/binary_ctmc.py",
    ):
        tree = ast.parse((ROOT / "src/argus" / module).read_text(encoding="utf-8"))
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import | ast.ImportFrom)
            for alias in node.names
        } | {node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        assert not any("random" in name for name in imported), module


def test_misspecified_case_is_labelled_and_overconfident() -> None:
    by_id = {case["id"]: case for case in cases()}
    assert by_id["M2-04"]["kind"] == "misspecified"
    assert by_id["M2-04"]["output"] > by_id["M2-02"]["output"] == pytest.approx(325 / 362)


def test_independent_scipy_expm_comparison_is_part_of_the_table() -> None:
    block, _ = x0_table.table(json.loads(RAW.read_text(encoding="utf-8")))
    assert block.count("scipy") >= 2
    models = {case["model"] for case in cases()}
    assert {"M1", "M2/M3", "M3"} <= models
