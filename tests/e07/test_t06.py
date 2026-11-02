"""ARGUS-E07-T06: frozen splits, preregistration v1 and the selection rule."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from pathlib import Path

import pytest

from argus.experiments.cells import CELLS, PREREGISTERED
from argus.experiments.protocol import (
    ProtocolError,
    TuningResult,
    episode_config,
    episode_seeds,
    load_manifest_entries,
    load_preregistration,
    manifest_params,
    select,
    tuning_buildings,
    tuning_configs,
)
from argus.experiments.splits import (
    DEV_SEED_LIMIT,
    QUOTAS_V1,
    SPLIT_ORDER,
    DuplicateIndex,
    regenerate,
)
from argus.policies.baselines import METHOD_SPECS

ROOT = Path(__file__).resolve().parents[2]
PREREG = ROOT / "configs/experiments/preregistration_v1.json"
PROTOCOL = ROOT / "docs/EXPERIMENT_PROTOCOL.md"


@pytest.fixture(scope="module")
def prereg() -> dict:
    return load_preregistration(ROOT)


@pytest.fixture(scope="module")
def entries(prereg: dict) -> tuple:
    return load_manifest_entries(ROOT, prereg)


# --- hashes -----------------------------------------------------------------------


def test_configs_and_lists_are_hashed_and_the_document_names_the_hashes(prereg: dict) -> None:
    text = PROTOCOL.read_text(encoding="utf-8")
    prereg_sha = hashlib.sha256(PREREG.read_bytes()).hexdigest()
    manifest_sha = prereg["split_manifest"]["sha256"]
    assert prereg_sha in text and manifest_sha in text
    assert hashlib.sha256((ROOT / prereg["split_manifest"]["path"]).read_bytes()).hexdigest() == (
        manifest_sha
    )


def test_a_changed_manifest_is_refused(tmp_path: Path) -> None:
    (tmp_path / "configs/experiments").mkdir(parents=True)
    (tmp_path / "configs/splits").mkdir(parents=True)
    (tmp_path / "configs/experiments/preregistration_v1.json").write_bytes(PREREG.read_bytes())
    original = (ROOT / "configs/splits/splits_v1.json").read_bytes()
    (tmp_path / "configs/splits/splits_v1.json").write_bytes(
        original.replace(b'"val"', b'"vaL"', 1)
    )
    with pytest.raises(ProtocolError):
        load_preregistration(tmp_path)


def test_manifest_file_reproduces_from_the_generator() -> None:
    from tools.make_split_manifest import TARGET, render

    assert TARGET.read_bytes() == render().encode("utf-8")


# --- splits -----------------------------------------------------------------------


def test_quotas_order_and_development_seeds(entries: tuple) -> None:
    for split in SPLIT_ORDER:
        rows = [e for e in entries if e.split == split]
        for family, n in QUOTAS_V1[split].items():
            assert sum(e.family == family for e in rows) == n
        positions = sorted(e.position for e in rows)
        assert positions == list(range(len(rows)))
    assert all(e.seed >= DEV_SEED_LIMIT for e in entries)
    assert len({e.building_id for e in entries}) == len(entries)
    for split in ("id_test", "ood_test"):
        ordered = sorted((e for e in entries if e.split == split), key=lambda e: e.position)
        for b in range(2, len(ordered) + 1, 2):  # every even prefix is balanced
            fams = [e.family for e in ordered[:b]]
            assert len(set(fams)) == 2 and fams.count(fams[0]) == b // 2


def test_no_split_overlap_by_hash_isomorphism_or_raster(prereg: dict, entries: tuple) -> None:
    params = manifest_params(ROOT, prereg)
    index = DuplicateIndex(prereg_manifest_threshold(prereg))
    hashes = set()
    for entry in entries:
        plan = regenerate(entry, params)  # also checks scenario and graph hashes
        assert plan.spec.scenario_hash not in hashes
        hashes.add(plan.spec.scenario_hash)
        assert not index.conflicts(plan), entry.building_id
        index.add(plan)
    assert len(index) == len(entries)


def prereg_manifest_threshold(prereg: dict) -> float:
    data = json.loads((ROOT / prereg["split_manifest"]["path"]).read_text(encoding="utf-8"))
    return float(data["iou_threshold"])


# --- preregistration content -------------------------------------------------------


def test_preregistration_matches_the_code(prereg: dict, entries: tuple) -> None:
    assert prereg["cells"]["parameters"] == json.loads(
        json.dumps(dataclasses.asdict(PREREGISTERED))
    )
    assert prereg["cells"]["ids"] == [c.cell_id for c in CELLS]
    roles = {m: s.role for m, s in METHOD_SPECS.items()}
    assert set(prereg["methods"]["controls"]) == {m for m, r in roles.items() if r == "control"}
    assert prereg["methods"]["candidate"] == ["repair"]
    assert set(prereg["tuning"]["grid"]) == {m for m, r in roles.items() if r != "diagnostic"}
    sample = episode_config(
        entries[0], manifest_params(ROOT, prereg), CELLS[-1], "repair", "c", {}, 0
    )
    data = sample.to_json()
    for key in ("map_resolution_m", "request_period_ticks", "losses", "belief", "channel", "ge"):
        assert prereg["episode"][key] == json.loads(json.dumps(data[key]))
    assert prereg["episode"]["decision_threshold"] == sample.decision_threshold == 0.8


def test_tuning_design_fits_the_budget(prereg: dict, entries: tuple) -> None:
    tuning = prereg["tuning"]
    configs = tuning_configs(prereg)
    per_method: dict[str, int] = {}
    for method, _, _ in configs:
        per_method[method] = per_method.get(method, 0) + 1
    assert max(per_method.values()) <= tuning["max_configs_per_method"] == 4
    buildings = tuning_buildings(entries, tuning["buildings"])
    assert [b.position for b in buildings] == list(range(10))
    assert all(b.split == "val" for b in buildings)
    episodes = len(configs) * len(CELLS) * len(buildings) * tuning["replicates"]
    assert episodes == 2000
    assert tuning["budget"] == {"wall_hours": 2.0, "cpu_hours": 8.0, "workers": 5}


def test_novelty_and_claim_rules_are_frozen(prereg: dict) -> None:
    assert prereg["novelty"].startswith("pending")
    assert "pending" in PROTOCOL.read_text(encoding="utf-8")
    assert ">= 0.10" in prereg["claims"]["h2"] and "<= 0.05" in prereg["claims"]["h2"]
    assert re.search(r"never", prereg["claims"]["p_value_alone"])
    assert prereg["test"]["replicates"] == 2 and "inconclusive" in prereg["test"]["prefix_rule"]


# --- episodes ---------------------------------------------------------------------


def test_seeds_are_shared_across_methods_cells_and_configs(prereg: dict, entries: tuple) -> None:
    params = manifest_params(ROOT, prereg)
    entry = entries[0]
    configs = [
        episode_config(entry, params, cell, method, cid, values, 0)
        for cell in (CELLS[0], CELLS[-1])
        for method, cid, values in tuning_configs(prereg)[:3]
    ]
    seeds = {c.to_json()["seeds"].__repr__() for c in configs}
    assert len(seeds) == 1
    assert episode_seeds(entry.building_id, 0) != episode_seeds(entry.building_id, 1)
    assert episode_seeds(entries[1].building_id, 0) != episode_seeds(entry.building_id, 0)
    assert len({c.episode_id for c in configs}) == len(configs)


# --- selection rule ---------------------------------------------------------------


def results(table: dict[tuple[str, str], float], buildings: list[str]) -> list[TuningResult]:
    return [
        TuningResult(m, c, b, cell.cell_id, loss, 100.0, "ok")
        for (m, c), loss in table.items()
        for b in buildings
        for cell in CELLS
    ]


ROLES = {
    "repair": "candidate",
    "ttl_ids": "control",
    "age_revisit": "control",
    "wait": "diagnostic",
}


def test_selection_picks_the_best_config_and_the_strongest_control() -> None:
    table = {
        ("repair", "h5"): 0.20,
        ("repair", "h10"): 0.30,
        ("ttl_ids", "a"): 0.40,
        ("ttl_ids", "b"): 0.25,
        ("age_revisit", "x"): 0.35,
        ("wait", "w"): 0.10,  # diagnostics never become the H2 control
    }
    sel = select(results(table, ["b0", "b1", "b2"]), ROLES, 2)
    assert sel.best_config == {"repair": "h5", "ttl_ids": "b", "age_revisit": "x", "wait": "w"}
    assert sel.h2_control == "ttl_ids"
    assert sel.per_config[("ttl_ids", "b")][0] == pytest.approx(0.25)


def test_incomplete_buildings_are_ignored_and_too_few_are_inconclusive() -> None:
    table = {("ttl_ids", "a"): 0.4, ("age_revisit", "x"): 0.3}
    rows = results(table, ["b0", "b1"])
    rows += [TuningResult("ttl_ids", "a", "b2", CELLS[0].cell_id, 0.0, 0.0, "ok")]
    sel = select(rows, ROLES, 2)
    assert sel.complete_buildings == ("b0", "b1") and sel.h2_control == "age_revisit"
    assert select(rows, ROLES, 3).h2_control is None


def test_ties_are_broken_by_sent_bytes_then_id() -> None:
    rows = [
        TuningResult(m, "c", "b0", cell.cell_id, 0.3, sent, "ok")
        for m, sent in (("ttl_ids", 200.0), ("age_revisit", 100.0))
        for cell in CELLS
    ]
    assert select(rows, ROLES, 1).h2_control == "age_revisit"


# --- committed validation result --------------------------------------------------


def test_committed_tuning_result_reproduces_the_selection(prereg: dict) -> None:
    path = ROOT / "docs/results/val_tuning_v1.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    assert len(rows) == 2000 and all(r["status"] == "ok" for r in rows)
    roles = {m: s.role for m, s in METHOD_SPECS.items()}
    sel = select(
        [
            TuningResult(
                r["method_id"],
                r["config_id"],
                r["building_id"],
                r["cell_id"],
                r["mean_loss"],
                r["sent_bytes"],
                r["status"],
            )
            for r in rows
        ],
        roles,
        prereg["tuning"]["minimum_complete_buildings"],
    )
    assert len(sel.complete_buildings) == 10
    assert sel.h2_control == "dynamics_ids" and roles[sel.h2_control] == "control"
    assert sel.best_config["dynamics_ids"] == "p100"
    report = (ROOT / "docs/results/VAL_TUNING_v1.md").read_text(encoding="utf-8")
    assert hashlib.sha256(path.read_bytes()).hexdigest() in report
    assert "H2-Kontrolle: `dynamics_ids`" in report
