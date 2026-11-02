"""ARGUS-E06-T03: pre-registered exact X2 micro ablation against theory."""

from __future__ import annotations

import hashlib
from pathlib import Path

from argus.experiments.x2_micro import load, run
from tools import x2_report

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/experiments/X2_micro.json"
REPORT = ROOT / "docs/results/X2_micro.md"
# Hash of the configuration as registered before the first computation.
PREREGISTERED_SHA256 = "8998a9ef5d21b5e920ed0ef94374cca2e284c391cddbafcb3c6e9e01fa088513"


def test_configuration_is_the_preregistered_one() -> None:
    assert hashlib.sha256(CONFIG.read_bytes()).hexdigest() == PREREGISTERED_SHA256
    config = load(CONFIG)
    assert [e["id"] for e in config["expectations"]] == ["E1", "E2", "E3", "E4", "E5"]


def test_every_exact_micro_result_matches_theory() -> None:
    config = load(CONFIG)
    checked = x2_report.compare(config, run(config))
    assert len(checked) == 16
    assert all(row["ok"] for row in checked), [r for r in checked if not r["ok"]]


def test_report_is_generated_from_current_results() -> None:
    config = load(CONFIG)
    text, all_ok = x2_report.render(config, x2_report.compare(config, run(config)))
    assert all_ok
    assert REPORT.read_text(encoding="utf-8") == text


def test_h3_is_not_claimed_and_disclaimer_is_present() -> None:
    text = REPORT.read_text(encoding="utf-8")
    assert "kein Befund zu H2 oder H3" in text
    lowered = text.lower()
    for claim in ("h3 supported", "h3 bestätigt", "h2 bestätigt", "proves", "beweist"):
        assert claim not in lowered


def test_negative_cases_and_unconfirmed_expectation_are_documented() -> None:
    config = load(CONFIG)
    checked = x2_report.compare(config, run(config))
    outcome = {i: ok for i, _, ok in x2_report.expectations(config, checked)}
    # E5 was registered as "positive value"; the exact theory gives 0, so it stays refuted.
    assert outcome == {"E1": True, "E2": True, "E3": True, "E4": True, "E5": False}
    text = REPORT.read_text(encoding="utf-8")
    assert "**nicht bestätigt**" in text
    negatives = text.split("## Negative und Null-Fälle")[1]
    assert negatives.count("\n- ") == sum(r["value"] <= x2_report.ZERO for r in checked) == 11


def test_mechanism_direction_under_pilot_losses() -> None:
    rows = {
        (r["kind"], r["bias"], r["losses"], r.get("costs") or r.get("receiver")): r["value"]
        for r in run(load(CONFIG))
    }
    persistent = rows[("reobserve", "persistent", "pilot", "zero")]
    decorrelated = rows[("reobserve", "decorrelated", "pilot", "zero")]
    assert decorrelated > persistent > 0
    # With pilot costs the persistent-bias re-observation is no longer worth it.
    assert rows[("reobserve", "persistent", "pilot", "pilot")] < 0
    assert rows[("reobserve", "decorrelated", "pilot", "pilot")] > 0
