"""ARGUS-E09-T02: study 2 run set, manifest and frozen analysis (pre-run checks)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from argus.evaluation.statistics import AnalysisError
from argus.experiments.protocol import file_sha256
from argus.experiments.runset import ImmutableRunSet, RunSetError, _config, source_sha256
from argus.experiments.splits import SplitEntry
from argus.replay.interventions import InterventionSpec, open_window_s

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import analyze_s2  # noqa: E402
import s2_batch  # noqa: E402

PREREG = json.loads((ROOT / s2_batch.PREREG_PATH).read_text(encoding="utf-8"))
MANIFEST = "configs/splits/splits_v2.json"


def test_the_manifest_is_fresh_and_disjoint_from_v1() -> None:
    v1 = json.loads((ROOT / "configs/splits/splits_v1.json").read_text(encoding="utf-8"))
    v2 = json.loads((ROOT / MANIFEST).read_text(encoding="utf-8"))
    new = [SplitEntry(*e) for e in v2["entries"]]
    assert not {(e[2], e[3]) for e in v1["entries"]} & {(e.family, e.seed) for e in new}
    assert {e.scenario_hash for e in new}.isdisjoint({e[5] for e in v1["entries"]})
    assert v2["prior_buildings"] == len(v1["entries"])
    assert file_sha256(ROOT / MANIFEST) == PREREG["basis"]["split_manifest"]["sha256"]
    families = [e.family for e in sorted(new, key=lambda e: e.position)]
    assert families.count("tree") == families.count("loop") == 50
    assert families[:40].count("tree") == 20


def test_the_arms_reuse_the_frozen_configurations() -> None:
    runset = ImmutableRunSet.load(ROOT)
    selected = {m: (c, p) for m, c, p in runset.methods()}
    arms = PREREG["design"]["arms"]
    for arm in ("repair", "no_comm", "dynamics_ids", "age_revisit"):
        assert (arms[arm]["config_id"], arms[arm]["params"]) == selected[arm]
    for arm in ("repair_observe", "repair_send"):
        assert arms[arm]["params"] == selected["repair"][1]


@pytest.fixture(scope="module")
def frozen() -> tuple[dict, str]:
    return s2_batch.load_frozen(ROOT)


def test_the_frozen_set_binds_the_source_tree_and_driver(frozen: tuple) -> None:
    data, _ = frozen
    assert data["source_tree_sha256"] == source_sha256(ROOT)
    assert "driver_sha256" in data and "src/argus/replay/interventions.py" in data["bound_code"]


def test_jobs_cover_every_condition_arm_and_replicate(frozen: tuple, tmp_path: Path) -> None:
    data, sha = frozen
    runset = ImmutableRunSet.load(ROOT)
    entries = s2_batch.s2_entries(ROOT)[:1]
    jobs = s2_batch.make_jobs(runset, sha, data["design"], entries, 2, tmp_path, "c", "a")
    assert len(jobs) == 10 * 6 * 2
    assert len({j.key for j in jobs}) == len(jobs)
    assert {j.payload["loss_false_open"] for j in jobs} == {2.0, 4.0}
    static = [j for j in jobs if j.key.startswith("s0-")]
    assert all(j.payload["cell_id"] == "sta-ind-direct" for j in static)


def test_an_s2_episode_needs_the_approval_whatever_the_payload(
    frozen: tuple, tmp_path: Path
) -> None:
    data, sha = frozen
    runset = ImmutableRunSet.load(ROOT)
    entries = s2_batch.s2_entries(ROOT)[:1]
    job = s2_batch.make_jobs(runset, sha, data["design"], entries, 1, tmp_path, "c", "0" * 64)[0]
    payload = dict(job.payload)
    payload.pop("approval_path")
    with pytest.raises(RunSetError, match="approval"):
        _config(payload)


def test_the_cost_ratio_reaches_the_episode(frozen: tuple, tmp_path: Path) -> None:
    data, sha = frozen
    runset = ImmutableRunSet.load(ROOT)
    calib = runset.buildings("calib", 1)[0]
    jobs = s2_batch.make_jobs(runset, sha, data["design"], calib, 1, tmp_path, "c", None)
    job = next(j for j in jobs if j.key.startswith("s0.25-c2/") and "/repair_observe/" in j.key)
    _, config = _config(job.payload)
    assert (config.loss_false_open, config.loss_false_closed) == (2.0, 1.0)
    assert config.nominal.z_model.rate_01 == pytest.approx(0.0025)
    assert config.method_id == "repair_observe"


def test_family_wise_intervals_are_wider_than_95_percent() -> None:
    draws = np.random.default_rng(0).normal(0.0, 1.0, 10_000)
    wide = analyze_s2.interval(0.0, draws, 1 - 0.05 / 3)
    narrow = analyze_s2.interval(0.0, draws, 0.95)
    assert wide["low"] < narrow["low"] and wide["high"] > narrow["high"]


def test_outcomes_have_three_levels() -> None:
    assert analyze_s2.outcome("S2-H1", {"low": 0.01, "high": 0.2}) == "supported"
    assert analyze_s2.outcome("S2-H1", {"low": -0.05, "high": 0.02}) == "against"
    assert analyze_s2.outcome("S2-H1", {"low": -0.05, "high": 0.08}) == "inconclusive"
    assert analyze_s2.outcome("S2-H3", {"low": -0.02, "high": 0.02}) == "consistent"
    assert analyze_s2.outcome("S2-H3", {"low": 0.01, "high": 0.09}) == "against"
    assert analyze_s2.outcome("S2-H3", {"low": -0.06, "high": 0.06}) == "inconclusive"
    assert analyze_s2.outcome("S2-H2", {"undefined": True}) == "inconclusive"


def test_a_replacement_keeps_its_slot() -> None:
    entries = s2_batch.s2_entries(ROOT)
    old, new = entries[3].building_id, entries[70].building_id
    order = analyze_s2.slot_order(10, [[old, new]], entries)
    assert order[3] == new and old not in order and len(order) == 10


def test_the_analysis_refuses_interim_and_foreign_looks(tmp_path: Path) -> None:
    with pytest.raises(AnalysisError, match="dry run"):
        analyze_s2.guard(tmp_path, "x", {"s2_test"}, True)
    with pytest.raises(AnalysisError, match="s2_test only"):
        analyze_s2.guard(tmp_path, "x", {"calib"}, False)
    (tmp_path / "LOCK").touch()
    with pytest.raises(AnalysisError, match="LOCK"):
        analyze_s2.guard(tmp_path, "x", {"s2_test"}, False)


def test_no_p_values_decide() -> None:
    assert "p-values are not used" in PREREG["decision"]["family_wise"]
    assert "p_value" not in (ROOT / "tools/analyze_s2.py").read_text(encoding="utf-8")


def synthetic_gain(scale: float, ratio: float) -> float:
    """A pure window effect: the gain depends only on W, so matched pairs are equal."""
    window = open_window_s(scale, ratio)
    if window > 40:  # includes the static cells
        return 0.10
    if window > 20:
        return 0.05
    return 0.0


def test_end_to_end_on_a_budget_stopped_synthetic_batch(frozen: tuple, tmp_path: Path) -> None:
    data, sha = frozen
    design = data["design"]
    entries = s2_batch.s2_entries(ROOT)
    rng = np.random.default_rng(1)
    rows = []
    for slot, entry in enumerate(entries[:43]):  # 42 complete, the 43rd incomplete
        for scale, ratio in s2_batch.conditions(design):
            cid = InterventionSpec("repair", float(scale), float(ratio)).condition_id
            for arm in design["arms"]:
                for rep in range(design["replicates"]):
                    if slot == 42 and arm == "no_comm":
                        continue
                    base = 0.30 + rng.normal(0, 0.005)
                    gain = synthetic_gain(scale, ratio) if arm == "repair_observe" else 0.0
                    rows.append(
                        {
                            "key": f"{cid}/s2_test/{entry.building_id}/{arm}/x/{rep}",
                            "split": "s2_test",
                            "building_id": entry.building_id,
                            "family": entry.family,
                            "position": entry.position,
                            "cell_id": "dyn-ind-direct" if scale else "sta-ind-direct",
                            "method_id": arm,
                            "replicate": rep,
                            "status": "ok",
                            "mean_loss": base * (1 - gain),
                            "coverage_closed_final": 0.3,
                            "sent_bytes_total": 1000,
                        }
                    )
    (tmp_path / "results.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )
    plan = {"frozen_sha256": sha, "src_sha256": source_sha256(ROOT), "replacements": []}
    (tmp_path / "run_plan.json").write_text(json.dumps(plan), encoding="utf-8")
    (tmp_path / "sessions.jsonl").write_text(json.dumps({"stop_reason": "budget"}) + "\n", "utf-8")
    report = analyze_s2.analyze(tmp_path, draws=2000)
    assert report["prefix_b"] == 42 and report["terminal"] == "budget"
    h = report["hypotheses"]
    assert h["S2-H1"]["estimate"] == pytest.approx(0.05, abs=0.005)
    assert h["S2-H1"]["outcome"] == "supported"
    assert h["S2-H2"]["estimate"] == pytest.approx(0.10, abs=0.005)
    assert h["S2-H2"]["outcome"] == "supported"
    assert h["S2-H3"]["estimate"] == pytest.approx(0.0, abs=0.005)
    assert h["S2-H3"]["outcome"] == "consistent"
    assert report["descriptive"]["window_rank_correlation"] > 0
