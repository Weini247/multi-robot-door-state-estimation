"""ARGUS-E07-T04: the frozen port contract for the published low-bandwidth baseline."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs/baselines/PORT_CONTRACT.md"
CONTRACT = ROOT / "docs/baselines/crossrank_contract.json"


def contract() -> dict[str, object]:
    data: dict[str, object] = json.loads(CONTRACT.read_text(encoding="utf-8"))
    return data


def test_differences_are_explicit() -> None:
    c = contract()
    differences = c["differences"]
    assert isinstance(differences, list) and len(differences) >= 5
    for topic in ("task", "geometry", "waypoints", "timing", "channel", "results"):
        assert any(d.startswith(topic) for d in differences), topic
    assert "## 4. Mathematical and informational differences" in DOC.read_text(encoding="utf-8")


def test_original_code_status_is_documented_not_invented() -> None:
    c = contract()
    code = c["original_code"]
    assert isinstance(code, dict) and code["status"] == "not_found"
    assert c["port_kind"] == "adaptation_from_description"
    source = c["source_read"]
    assert isinstance(source, dict)
    assert len(str(source["sha256"])) == 64 and source["stored_in_repository"] is False
    assert c["journal_version_read"] is False


def test_no_reproduced_result_is_claimed() -> None:
    assert contract()["reproduction_claim"] is False
    text = DOC.read_text(encoding="utf-8").lower()
    assert "no result of the paper is claimed to be reproduced" in text
    for claim in ("we reproduce", "reproduced the results", "matches the paper"):
        assert claim not in text


def test_frozen_choice_is_bandwidth_compatible_and_tunable_only_on_validation() -> None:
    c = contract()
    assert c["variant"] == "CR-ESR_p"
    assert c["communication"] == "positions only, charged frames"
    assert set(c["tunable_on_validation"]) == set(c["parameters"])  # type: ignore[arg-type]
