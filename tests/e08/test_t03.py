"""ARGUS-E08-T03: preregistered inference, frozen before the test batch runs."""

from __future__ import annotations

import gzip
import json
import math
import shutil
from pathlib import Path

import pytest

from argus.evaluation.statistics import (
    AnalysisError,
    Interval,
    PairedBootstrap,
    building_values,
    complete_prefix,
    decision_loss,
    factorial_contrasts,
    family_weighted_mean,
    h2_verdict,
    holm,
    ratio_gain,
    repair_lags_d2,
    rescored_mean_loss,
)
from argus.experiments.batch import building_losses, plan_test_size
from argus.experiments.runset import ANALYSIS_CODE, FROZEN_PATH, text_sha256

ROOT = Path(__file__).resolve().parents[2]
PROBE = ROOT / "docs/results/test_probe_calib_v3"
CELLS = [
    f"{d}-{s}-{c}" for d in ("sta", "dyn") for s in ("ind", "cor") for c in ("direct", "relay")
]
FAMILY = {f"b{i}": ("loop" if i % 2 else "tree") for i in range(20)}


def rows_for(values: dict[str, dict[str, float]], method: str, reps: int = 1) -> list[dict]:
    return [
        {"method_id": method, "building_id": b, "cell_id": c, "replicate": k, "mean_loss": v}
        for b, per in values.items()
        for c, v in per.items()
        for k in range(reps)
    ]


# --- acceptance criteria of the backlog ---------------------------------------------------------


def test_a_constant_delta_gives_a_point_interval() -> None:
    ctrl = {b: 0.2 + 0.01 * i for i, b in enumerate(FAMILY)}
    cand = {b: 0.9 * v for b, v in ctrl.items()}
    gain = PairedBootstrap(list(FAMILY), FAMILY, draws=2000).ratio_gain(cand, ctrl)
    assert gain.estimate == pytest.approx(0.1)
    assert gain.low == pytest.approx(0.1) and gain.high == pytest.approx(0.1)


def test_more_rows_per_building_do_not_increase_n_and_duplicates_are_refused() -> None:
    values = {b: {c: 0.3 for c in CELLS} for b in FAMILY}
    one = building_values(rows_for(values, "m"), "m", "mean_loss", CELLS, 1)
    many = building_values(rows_for(values, "m", reps=5), "m", "mean_loss", CELLS, 5)
    assert len(one) == len(many) == len(FAMILY) and one == many
    duplicated = rows_for(values, "m") + rows_for({"b0": values["b0"]}, "m")
    with pytest.raises(AnalysisError, match="duplicate"):
        building_values(duplicated, "m", "mean_loss", CELLS, 1)
    incomplete = [
        r
        for r in rows_for(values, "m")
        if not (r["building_id"] == "b3" and r["cell_id"] == CELLS[0])
    ]
    with pytest.raises(AnalysisError, match="incomplete"):
        building_values(incomplete, "m", "mean_loss", CELLS, 1)


def test_interaction_signs_match_an_analytic_fixture() -> None:
    # L = 0.2 + 0.1*dyn + 0.05*cor + 0.03*dyn*cor - 0.02*cor*relay: only two interactions
    def loss(cell: str) -> float:
        d, s, c = (
            cell.split("-")[0] == "dyn",
            cell.split("-")[1] == "cor",
            cell.split("-")[2] == "relay",
        )
        return 0.2 + 0.1 * d + 0.05 * s + 0.03 * d * s - 0.02 * s * c

    got = factorial_contrasts({c: loss(c) for c in CELLS})
    assert got["dynamics x sensor"] == pytest.approx(0.03)
    assert got["sensor x circulation"] == pytest.approx(-0.02)
    assert got["dynamics x circulation"] == pytest.approx(0.0, abs=1e-15)


def test_h2_is_never_claimed_from_a_p_value_alone() -> None:
    tiny_p = Interval(0.06, 0.05, 0.07, 1e-4)  # significant but below the 0.10 threshold
    ok_cov = Interval(0.0, -0.01, 0.02)
    assert h2_verdict(tiny_p, ok_cov) == "not supported"
    big = Interval(0.2, 0.12, 0.28, 1e-4)
    assert h2_verdict(big, Interval(0.03, -0.01, 0.06)) == "not supported"  # coverage fails
    assert h2_verdict(big, ok_cov) == "supported"
    assert h2_verdict(Interval(0.2, 0.10, 0.3), Interval(0.0, -0.1, 0.05)) == "supported"


# --- estimator details --------------------------------------------------------------------------


def test_families_are_weighted_50_50_and_resampled_within_family() -> None:
    family = {"a": "loop", "b": "loop", "c": "loop", "d": "tree"}
    values = {"a": 1.0, "b": 1.0, "c": 1.0, "d": 3.0}
    assert family_weighted_mean(values, family) == pytest.approx(2.0)  # not 1.5
    means = PairedBootstrap(list(family), family, draws=500).means(values)
    assert means.min() == pytest.approx(2.0) and means.max() == pytest.approx(2.0)


def test_the_bootstrap_is_seeded_and_order_independent() -> None:
    ctrl = {b: 0.3 + 0.02 * (i % 5) for i, b in enumerate(FAMILY)}
    cand = {b: v - 0.01 * (i % 3) for i, (b, v) in enumerate(ctrl.items())}
    first = PairedBootstrap(list(FAMILY), FAMILY, draws=1000).ratio_gain(cand, ctrl)
    again = PairedBootstrap(sorted(FAMILY, reverse=True), FAMILY, draws=1000).ratio_gain(cand, ctrl)
    assert first == again
    assert first.low < first.estimate < first.high and 0 < first.p_value <= 1


def test_a_zero_control_mean_is_undefined_not_infinite() -> None:
    zero = {b: 0.0 for b in FAMILY}
    with pytest.raises(AnalysisError, match="undefined"):
        ratio_gain(zero, zero, FAMILY)


def test_holm_matches_a_hand_computed_example() -> None:
    got = holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.5})
    assert got == pytest.approx({"a": 0.04, "c": 0.09, "b": 0.09, "d": 0.5})


def test_the_prefix_rule_keeps_the_largest_even_complete_prefix() -> None:
    assert complete_prefix([True] * 80) == 80
    assert complete_prefix([True] * 63 + [False] + [True] * 10) == 62
    assert complete_prefix([True] * 61) == 60
    assert complete_prefix([True] * 59) is None


def test_the_pilot_gain_is_reproduced() -> None:
    rows = [
        json.loads(line) for line in (ROOT / "docs/results/calib_pilot_v2/results.jsonl").open()
    ]
    family = {r["building_id"]: r["family"] for r in rows}
    cand = building_losses(rows, "repair", 8)
    ctrl = building_losses(rows, "cross_rank", 8)
    assert ratio_gain(cand, ctrl, family) == pytest.approx(plan_test_size(cand, ctrl, family).gain)


# --- re-scoring and lags from raw records -----------------------------------------------------


def probe_rows() -> list[dict]:
    return [json.loads(line) for line in (PROBE / "results.jsonl").open(encoding="utf-8")]


def raw_of(row: dict) -> dict:
    return json.loads(gzip.decompress((PROBE / row["raw"]).read_bytes()))


def test_raw_rescoring_reproduces_every_committed_probe_row() -> None:
    for row in probe_rows():
        raw = raw_of(row)
        fo, fc = raw["config"]["losses"]
        assert rescored_mean_loss(raw, fo, fc) * fc == pytest.approx(row["mean_loss"], abs=1e-12)


def test_loss_ratios_and_the_m6_rule_on_defaulted_probes() -> None:
    assert decision_loss(0.8, 0, 4.0, 1.0) == 4.0  # threshold 0.8, tie opens
    assert decision_loss(0.79, 1, 4.0, 1.0) == 1.0
    assert decision_loss(0.67, 0, 2.0, 1.0) == 2.0  # threshold 2/3
    raw = {
        "key": "k",
        "probes": [[0, "r0", "D0"], [5, "r0", "D0"], [9, "r0", "D1"]],
        "answered": [[0, "r0", "D0", 0.9, 1, 1, 0.0, 0]],
        "defaulted": [[5, "r0", "D0", 1, 1.0], [9, "r0", "D1", 0, 0.0]],
    }
    assert rescored_mean_loss(raw, 4.0, 1.0) == pytest.approx(1 / 3)
    assert rescored_mean_loss(raw, 4.0, 1.0, failed_probe_loss=4.0) == pytest.approx(8 / 3)
    with pytest.raises(AnalysisError):
        rescored_mean_loss({**raw, "defaulted": raw["defaulted"][:1]}, 4.0, 1.0)


def test_repair_lags_follow_amendment_02() -> None:
    raw = {
        "failure_tick": None,
        "probes": [[0, "r0", "D0"], [0, "r1", "D0"]],
        "trace": {"passage_changes": {"D0": [100, 500]}},
        "answered": [
            # r0 after the change to closed (truth 0): p_open .1/.15/.3 -> 0.9, 0.85, 0.7
            [150, "r0", "D0", 0.1, 0, 0, 0.0, 0],
            [200, "r0", "D0", 0.15, 0, 0, 0.0, 0],
            [250, "r0", "D0", 0.1, 0, 0, 0.0, 0],
            [300, "r0", "D0", 0.1, 0, 0, 0.0, 0],
            [150, "r1", "D0", 0.5, 0, 0, 0.0, 0],
            # after the second change (truth 1) only two good probes before the horizon
            [600, "r0", "D0", 0.9, 1, 1, 0.0, 0],
            [650, "r0", "D0", 0.95, 1, 1, 0.0, 0],
        ],
    }
    lags = repair_lags_d2(raw, horizon_ticks=700)
    by = {(lag.change_tick, lag.lag_ticks, lag.censored) for lag in lags}
    assert (100, 50, False) in by  # r0 repaired at its first good probe of the run
    assert (100, 400, True) in by  # r1 never got 3 good probes: censored at the next change
    assert (500, 200, True) in by  # censored at the horizon
    assert len(lags) == 4
    failed = repair_lags_d2({**raw, "failure_tick": 240}, horizon_ticks=700)
    assert {(lag.change_tick, lag.censored) for lag in failed} == {(100, True)}


# --- driver ------------------------------------------------------------------------------------


def test_the_analysis_code_is_frozen_with_the_run_set() -> None:
    frozen = json.loads((ROOT / FROZEN_PATH).read_text(encoding="utf-8"))
    assert frozen["analysis"]["code"] == {p: text_sha256(ROOT / p) for p in ANALYSIS_CODE}


def test_the_dry_run_on_the_calib_probe_is_deterministic() -> None:
    from tools.analyze_x1x2 import analyze

    first = analyze(PROBE, dry_run=True, draws=500)
    assert first == analyze(PROBE, dry_run=True, draws=500)
    assert first["prefix_b"] == 2 and first["primary"]["list"] == "calib"
    assert first["primary"]["verdict"] in ("supported", "not supported")
    assert math.isfinite(first["primary"]["h2_loss"]["gain"]["estimate"])
    assert set(first["primary"]["gain_per_cell"]) == set(CELLS)
    assert first["primary"]["repair_lag"]["tau_ticks"] == 300


def test_the_driver_refuses_test_data_in_a_dry_run_and_interim_looks(tmp_path: Path) -> None:
    from tools.analyze_x1x2 import analyze

    out = tmp_path / "batch"
    shutil.copytree(PROBE, out)
    text = (out / "results.jsonl").read_text(encoding="utf-8").replace('"calib"', '"id_test"')
    (out / "results.jsonl").write_text(text, encoding="utf-8")
    with pytest.raises(AnalysisError, match="dry run must not read test splits"):
        analyze(out, dry_run=True)
    with pytest.raises(AnalysisError, match="needs id_test and ood_test"):
        analyze(out)
    rows = [json.loads(line) for line in text.splitlines() if line]
    for r in rows:
        if r["position"] == 1:
            r["split"] = "ood_test"
    lines = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
    (out / "results.jsonl").write_text(lines, encoding="utf-8")
    (out / "LOCK").write_text("", encoding="utf-8")
    with pytest.raises(AnalysisError, match="LOCK"):
        analyze(out)


# --- review fixes (amendment 03) ---------------------------------------------------------------


def test_a_zero_control_mean_reports_only_the_difference() -> None:
    zero = {b: 0.0 for b in FAMILY}
    cand = {b: 0.1 for b in FAMILY}
    got = PairedBootstrap(list(FAMILY), FAMILY, draws=200).gain_report(cand, zero)
    assert got["gain"] is None and "undefined" in got["gain_undefined"]
    assert got["difference"]["estimate"] == pytest.approx(0.1)


def test_a_replacement_takes_the_slot_of_the_building_it_replaced() -> None:
    from argus.experiments.runset import ImmutableRunSet
    from tools.analyze_x1x2 import slots

    runset = ImmutableRunSet.load(ROOT)
    calib = sorted((e for e in runset.entries if e.split == "calib"), key=lambda e: e.position)
    replaced, spare = calib[1], next(e for e in calib[2:] if e.family == calib[1].family)
    rows = [
        {"split": "calib", "building_id": calib[0].building_id},
        {"split": "calib", "building_id": spare.building_id},
    ]
    plan = {"replacements": {"calib": [[replaced.building_id, spare.building_id]]}}
    got = slots(rows, runset, plan)
    assert got[("calib", spare.building_id)] == 1 and got[("calib", calib[0].building_id)] == 0
    assert slots(rows, runset, None)[("calib", spare.building_id)] == spare.position


def test_a_killed_episode_is_analysed_not_a_crash(tmp_path: Path) -> None:
    from argus.experiments.runset import ImmutableRunSet, killed_episode
    from tools.analyze_x1x2 import analyze

    out = tmp_path / "batch"
    shutil.copytree(PROBE, out)
    rows = probe_rows()
    victim = next(r for r in rows if r["method_id"] == "repair" and r["cell_id"] == "dyn-cor-relay")
    runset = ImmutableRunSet.load(ROOT)
    job = next(
        j for j in runset.jobs(("calib",), 2, 1, out, "test", (0, 1)) if j.key == victim["key"]
    )
    killed = killed_episode({**job.payload, "out": str(out), "key": job.key})
    assert "coverage_closed_final" not in killed and killed["wall_s"] is None
    rows = [killed if r["key"] == victim["key"] else r for r in rows]
    lines = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
    (out / "results.jsonl").write_text(lines, encoding="utf-8")
    report = analyze(out, dry_run=True, draws=200)
    primary = report["primary"]
    assert primary["killed"]["repair"] == 1 and primary["status"]["repair"]["timeout"] == 1
    assert primary["sensitivity"]["pairs_without_failure"]["excluded_pairs"] == 1
    assert primary["runtime"]["repair"]["wall_s"]["n"] == 15.0
    assert primary["calibration_answered"]["repair"]["episodes_without_answers"] == 1


def test_only_the_holm_family_reports_p_values() -> None:
    from tools.analyze_x1x2 import analyze

    report = analyze(PROBE, dry_run=True, draws=200)
    family = report["primary"]["holm_family"]
    assert len(family["p_raw"]) == 11 and set(family["p_holm"]) == set(family["p_raw"])
    assert "p" not in report["primary"]["h2_loss"]["gain"]
    lag = report["primary"]["repair_lag"]["per_robot"]["repair"]
    assert lag["at_risk_at_tau"] <= lag["events"] and lag["kaplan_meier_to_tau"] is not None


def test_a_test_batch_is_analysed_only_when_finished(tmp_path: Path) -> None:
    from argus.experiments.runset import ImmutableRunSet, source_sha256
    from tools.analyze_x1x2 import guard

    runset = ImmutableRunSet.load(ROOT)
    splits = {"id_test", "ood_test"}
    plan = {"frozen_sha256": runset.sha256, "src_sha256": source_sha256(ROOT), "b": 80}
    (tmp_path / "run_plan.json").write_text(json.dumps(plan), encoding="utf-8")

    def sessions(reason: str | None) -> None:
        line = json.dumps({"stop_reason": reason}) + "\n"
        (tmp_path / "sessions.jsonl").write_text(line, encoding="utf-8")

    for reason in ("paused", "breaker: 4 failures in last 20"):
        sessions(reason)
        with pytest.raises(AnalysisError, match="resume it first"):
            guard(tmp_path, runset, False, splits)
    sessions("budget")
    assert guard(tmp_path, runset, False, splits)["terminal"] == "budget"
    sessions(None)
    assert guard(tmp_path, runset, False, splits)["terminal"] == "complete"
    (tmp_path / "PAUSE").write_text("", encoding="utf-8")
    with pytest.raises(AnalysisError, match="PAUSE"):
        guard(tmp_path, runset, False, splits)
    (tmp_path / "PAUSE").unlink()
    plan["src_sha256"] = "0" * 64
    (tmp_path / "run_plan.json").write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(AnalysisError, match="source tree"):
        guard(tmp_path, runset, False, splits)
