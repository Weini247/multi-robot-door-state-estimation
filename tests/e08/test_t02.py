"""ARGUS-E08-T02: frozen X1/X2 run set, failure rule, pause/resume and the calib probe."""

from __future__ import annotations

import dataclasses
import gzip
import hashlib
import json
from pathlib import Path

import pytest

from argus.experiments.batch import BatchCaps, BatchJob, BatchRunner, probe_worker
from argus.experiments.cells import CELLS
from argus.experiments.protocol import episode_config, episode_seeds, manifest_params
from argus.experiments.runner import ExperimentRunner, generate_requests
from argus.experiments.runset import (
    AMENDMENT_02_PATH,
    AMENDMENT_03_PATH,
    AMENDMENT_PATH,
    ANALYSIS_CODE,
    APPROVAL_PATH,
    FROZEN_PATH,
    PILOT_DIR,
    PROBE_DIR,
    VAL_RESULTS,
    ImmutableRunSet,
    RunSetError,
    _config,
    b_for_budget,
    cost_per_position,
    killed_episode,
    render_frozen,
    summarize,
)
from argus.world.scenario import ExogenousWorld

ROOT = Path(__file__).resolve().parents[2]
COST_CPU_H, COST_WALL_H = 62.9, 22.8  # B = 80, from the committed probe v3
CAPS = BatchCaps(wall_hours=1.0, cpu_hours=1.0, workers=1, episode_timeout_s=60.0)


@pytest.fixture(scope="module")
def runset() -> ImmutableRunSet:
    return ImmutableRunSet.load(ROOT)


def rows(out: Path) -> list[dict]:
    text = (out / "results.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line]


# --- frozen run set -------------------------------------------------------------------


def test_frozen_run_set_is_derived_from_committed_evidence(runset: ImmutableRunSet) -> None:
    assert (ROOT / FROZEN_PATH).read_text(encoding="utf-8") == render_frozen(ROOT)
    frozen = runset.frozen
    assert frozen["methods"]["h2_control"] == "cross_rank"  # v2 tuning (amendment 01)
    selected = frozen["methods"]["selected"]
    assert set(selected) == {"repair", *runset.prereg["methods"]["controls"]}
    assert not set(selected) & set(runset.prereg["methods"]["diagnostics"])
    assert selected["repair"]["config_id"] == "h5.0"
    assert selected["cross_rank"]["config_id"] == "m1.5f1.0"
    for method, info in selected.items():
        assert runset.prereg["tuning"]["grid"][method][info["config_id"]] == info["params"]
    assert frozen["test"]["b_planned"] == 80 and frozen["test"]["replicates"] == 2
    assert frozen["sources"]["amendment"]["path"] == AMENDMENT_PATH
    assert set(frozen["analysis"]["decisions"]) == {
        "O2_failure_rule",
        "O3_coverage",
        "O4_infrastructure",
        "O5_sensitivity",
        "O6_definitions",
        "O7_b_reading",
        "O8_budget",
        "D1_tuning_v2_resume",
        "D2_repair_lag",
        "A1_replacement_slot",
        "A2_failed_episode_coverage",
        "A3_terminal_batch",
        "A4_holm_family",
        "A5_zero_control",
        "A6_lags_scope",
        "A7_pairs_without_failure",
        "A8_calibration",
        "A9_ood",
        "A10_js",
        "A11_even_b",
        "A12_resource_limits",
    }
    assert frozen["sources"]["amendment_03"]["path"] == AMENDMENT_03_PATH
    assert frozen["analysis"]["decisions"]["O6_definitions"]["repair_lag"] == "see D2_repair_lag"
    assert frozen["sources"]["amendment_02"]["path"] == AMENDMENT_02_PATH
    assert frozen["test"]["lists"] == ["id_test", "ood_test"]
    assert frozen["cells"] == [c.cell_id for c in CELLS]


def test_a_changed_frozen_file_is_refused(tmp_path: Path) -> None:
    for rel in (
        "configs/experiments/preregistration_v1.json",
        "configs/splits/splits_v1.json",
        AMENDMENT_PATH,
        AMENDMENT_02_PATH,
        AMENDMENT_03_PATH,
        VAL_RESULTS,
        f"{PILOT_DIR}/results.jsonl",
        f"{PILOT_DIR}/usage.json",
        f"{PROBE_DIR}/results.jsonl",
        f"{PROBE_DIR}/usage.json",
        *ANALYSIS_CODE,
        FROZEN_PATH,
    ):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_bytes((ROOT / rel).read_bytes())
    ImmutableRunSet.load(tmp_path)  # an exact copy loads
    frozen = tmp_path / FROZEN_PATH
    frozen.write_text(
        frozen.read_text(encoding="utf-8").replace('"m1.5f1.0"', '"m2.5f0.5"', 1), "utf-8"
    )
    with pytest.raises(RunSetError):
        ImmutableRunSet.load(tmp_path)


def test_the_test_batch_is_refused_without_a_matching_approval(
    runset: ImmutableRunSet, tmp_path: Path
) -> None:
    assert not (ROOT / APPROVAL_PATH).exists()
    with pytest.raises(RunSetError, match="no approval"):
        runset.approval()
    other = dataclasses.replace(runset, root=tmp_path)
    (tmp_path / APPROVAL_PATH).parent.mkdir(parents=True)
    (tmp_path / APPROVAL_PATH).write_text(json.dumps({"frozen_sha256": "0" * 64}), "utf-8")
    with pytest.raises(RunSetError, match="another frozen run set"):
        other.approval()


def test_budget_caps_b_to_an_even_number_of_at_least_60(runset: ImmutableRunSet) -> None:
    cpu_h, wall_h = cost_per_position(runset.frozen)
    assert b_for_budget(runset.frozen, 1000.0, 1000.0) == 80
    assert b_for_budget(runset.frozen, 61.5 * cpu_h, 1000.0) == 60
    assert b_for_budget(runset.frozen, 1000.0, 75.2 * wall_h) == 74
    assert b_for_budget(runset.frozen, 59.9 * cpu_h, 1000.0) is None
    # cost model measured on the test worker (calib probe v3), not the pilot estimate
    assert runset.frozen["cost_model"]["source"].startswith(PROBE_DIR)
    assert 80 * cpu_h == pytest.approx(COST_CPU_H, abs=0.1)
    assert 80 * wall_h == pytest.approx(COST_WALL_H, abs=0.1)


def test_test_jobs_follow_the_frozen_order_and_share_seeds(runset: ImmutableRunSet) -> None:
    jobs = runset.jobs(("id_test", "ood_test"), 4, 2, Path("x"), "c")
    assert len(jobs) == 4 * 2 * 2 * 7 * 8 and len({j.key for j in jobs}) == len(jobs)
    groups = list(dict.fromkeys(j.group for j in jobs))
    assert [g.split("/")[0] for g in groups] == ["id_test", "ood_test"] * 4
    positions = [j.payload["entry"][1] for j in jobs]
    assert positions == sorted(positions)
    by_building: dict[tuple[str, int], set] = {}
    for j in jobs:
        b = f"{j.payload['entry'][2]}-{j.payload['entry'][3]}"
        by_building.setdefault((b, j.payload["replicate"]), set()).add(
            (j.payload["method_id"], j.payload["cell_id"])
        )
    assert all(len(v) == 7 * 8 for v in by_building.values())
    # seeds are a function of (building, replicate) only
    b0 = groups[0].split("/")[1]
    assert episode_seeds(b0, 0) != episode_seeds(b0, 1)


def test_invalid_scenarios_are_replaced_within_list_and_family(runset: ImmutableRunSet) -> None:
    prefix, _ = runset.buildings("id_test", 6)
    bad = {prefix[1].building_id}
    chosen, replaced = runset.buildings("id_test", 6, lambda e: e.building_id not in bad)
    assert replaced and replaced[0][0] == prefix[1].building_id
    new = next(e for e in chosen if e.building_id == replaced[0][1])
    assert new.family == prefix[1].family and new.position >= 6 and new.split == "id_test"
    assert [e.building_id for e in chosen if e.building_id not in {replaced[0][1]}] == [
        e.building_id for e in prefix if e.building_id not in bad
    ]


def test_the_worker_refuses_buildings_outside_the_allowed_splits(runset: ImmutableRunSet) -> None:
    job = runset.jobs(("id_test",), 1, 1, Path("x"), "c")[0]
    payload = {**job.payload, "allowed_splits": ["calib"]}
    with pytest.raises(RunSetError, match="not allowed"):
        _config(payload)


# --- failure rule ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def calib_payload(runset: ImmutableRunSet) -> dict:
    job = runset.jobs(("calib",), 1, 1, Path("unused"), "test")[-1]  # dyn-cor-relay, ttl_ids
    return {
        **job.payload,
        "method_id": "repair",
        "config_id": "h2.5",
        "method_params": {"horizon_s": 2.5},
    }


def short(payload: dict, horizon: int, **changes: object):
    entry, config = _config(payload)
    spec = dataclasses.replace(config.scenario, horizon_ticks=horizon)
    return entry, dataclasses.replace(config, scenario=spec, **changes)


def check_rule(config, row: dict, raw: dict) -> None:
    world = ExogenousWorld(config.scenario, config.seed_world)
    c10, c01 = config.loss_false_open, config.loss_false_closed
    ft = raw["failure_tick"]
    assert all(a[0] < ft for a in raw["answered"])
    assert all(d[0] >= ft for d in raw["defaulted"])
    for tick, _, entity, truth, loss in raw["defaulted"]:
        assert truth == world.state(entity, tick) and loss == (c01 if truth == 1 else 0.0)
    losses = [a[6] for a in raw["answered"]] + [d[4] for d in raw["defaulted"]]
    assert row["mean_loss"] == pytest.approx(sum(losses) / len(losses))
    assert row["requests"] == len(raw["probes"]) == row["answered"] + row["defaulted"]
    assert c10 == 4.0  # a false 'open' would cost 4; the default never does


def test_ok_episode_truths_match_the_exogenous_timeline_and_trace_is_passive(
    calib_payload: dict,
) -> None:
    entry, config = short(calib_payload, 400)
    plain = ExperimentRunner().run(config)
    traced = ExperimentRunner(trace=True).run(config)
    assert plain.status == traced.status == "ok" and plain.records == traced.records
    world = ExogenousWorld(config.scenario, config.seed_world)
    assert all(r.truth == world.state(r.entity, r.tick) for r in traced.records)
    row, raw = summarize(calib_payload, entry, config, traced, "ok", None)
    assert row["defaulted"] == 0 and row["answered"] == len(generate_requests(config))
    assert 0 < row["coverage_closed_auc"] <= row["coverage_closed_final"] <= 1
    trace = traced.trace
    for pid, ticks in trace["passage_changes"].items():
        expected = [k for k in range(1, 401) if world.state(pid, k) != world.state(pid, k - 1)]
        assert ticks == expected
    assert row["sent_bytes_total"] == sum(trace["channel_tx_bytes"].values()) > 0


def test_timeout_keeps_answered_probes_and_defaults_the_rest_to_closed(
    calib_payload: dict,
) -> None:
    entry, config = short(calib_payload, 3000, max_wall_seconds=1.5)
    result = ExperimentRunner(trace=True).run(config)
    assert result.status == "timeout" and result.failure_tick is not None
    row, raw = summarize(calib_payload, entry, config, result, result.status, result.error)
    assert row["answered"] > 0 and row["defaulted"] > 0
    check_rule(config, row, raw)


def test_an_error_mid_episode_uses_the_same_rule(
    calib_payload: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    import argus.experiments.runner as runner_module

    original = runner_module.scan_map
    calls = {"n": 0}

    def failing(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] > 4 * 60:  # 4 robots, scans every 5 ticks: fails near tick 300
            raise RuntimeError("injected failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(runner_module, "scan_map", failing)
    entry, config = short(calib_payload, 600)
    result = ExperimentRunner(trace=True).run(config)
    assert result.status == "error" and result.records == () and result.partial_records
    row, raw = summarize(calib_payload, entry, config, result, result.status, result.error)
    assert 250 <= row["failure_tick"] <= 350 and row["status"] == "error"
    check_rule(config, row, raw)


def test_an_invalid_episode_is_a_failure_under_the_same_rule(
    calib_payload: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Amendment 01, O2: runner status 'invalid' (ImpossibleEvidence) is an episode failure."""
    import argus.experiments.runner as runner_module
    from argus.beliefs.exact_micro import ImpossibleEvidence

    original = runner_module.scan_map
    calls = {"n": 0}

    def impossible(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] > 4 * 60:
            raise ImpossibleEvidence("injected")
        return original(*args, **kwargs)

    monkeypatch.setattr(runner_module, "scan_map", impossible)
    entry, config = short(calib_payload, 600)
    result = ExperimentRunner(trace=True).run(config)
    assert result.status == "invalid" and result.records == () and result.partial_records
    row, raw = summarize(calib_payload, entry, config, result, result.status, result.error)
    assert row["status"] == "invalid" and 250 <= row["failure_tick"] <= 350
    assert row["defaulted"] > 0 and row["mean_loss"] is not None
    check_rule(config, row, raw)


def test_the_replacement_validator_follows_amendment_01(
    runset: ImmutableRunSet, monkeypatch: pytest.MonkeyPatch
) -> None:
    """O2: not regenerable, not valid, or an empty coverage target set -> replaced."""
    import tools.test_batch as tb

    params = manifest_params(ROOT, runset.prereg)
    entry = next(e for e in runset.entries if e.split == "calib")
    assert tb.scenario_valid(entry, params)
    broken = dataclasses.replace(entry, scenario_hash="0" * 64)
    assert not tb.scenario_valid(broken, params)

    class Empty:
        def __init__(self, *args: object) -> None:
            self.targets = {"closed": frozenset(), "open": frozenset({(0, 0)})}

    monkeypatch.setattr(tb, "EpisodeTrace", Empty)
    assert not tb.scenario_valid(entry, params)


def test_a_killed_episode_defaults_every_probe(calib_payload: dict, tmp_path: Path) -> None:
    payload = {**calib_payload, "out": str(tmp_path), "key": "calib/x/k"}
    row = killed_episode(payload)
    raw = json.loads(gzip.decompress((tmp_path / row["raw"]).read_bytes()))
    assert row["status"] == "timeout" and row["answered"] == 0
    assert row["defaulted"] == row["requests"] > 0
    assert hashlib.sha256((tmp_path / row["raw"]).read_bytes()).hexdigest() == row["raw_sha256"]
    _, config = _config(payload)
    check_rule(config, row, raw)


# --- pause, resume, truncation ---------------------------------------------------------


def test_pause_file_stops_launching_and_a_rerun_completes_exactly_once(tmp_path: Path) -> None:
    pause = tmp_path / "PAUSE"
    jobs = [BatchJob("a/0", "a", {"touch": str(pause)})]
    jobs += [BatchJob(f"a/{i}", "a", {}) for i in range(1, 4)]
    jobs += [BatchJob(f"b/{i}", "b", {}) for i in range(2)]
    first = BatchRunner(tmp_path, CAPS, probe_worker).run(jobs)
    assert first.paused and first.ran == 1 and first.stopped_before == "a"
    with pytest.raises(RuntimeError, match="remove"):
        BatchRunner(tmp_path, CAPS, probe_worker).run(jobs)
    pause.unlink()
    rest = [BatchJob(j.key, j.group, {}) for j in jobs]  # same keys, no new pause
    second = BatchRunner(tmp_path, CAPS, probe_worker).run(rest)
    keys = [r["key"] for r in rows(tmp_path)]
    assert not second.paused and second.ran == 5 and second.skipped == 1
    assert sorted(keys) == sorted(j.key for j in jobs) and len(keys) == len(set(keys))
    assert second.usage.sessions == 2  # the refused start is not a session


def test_a_cut_off_last_line_is_quarantined_and_rerun(tmp_path: Path) -> None:
    jobs = [BatchJob(f"g/{i}", "g", {}) for i in range(3)]
    BatchRunner(tmp_path, CAPS, probe_worker).run(jobs[:2])
    results = tmp_path / "results.jsonl"
    text = results.read_text(encoding="utf-8")
    results.write_text(text[: len(text) - 20], encoding="utf-8")  # hard stop mid-write
    summary = BatchRunner(tmp_path, CAPS, probe_worker).run(jobs)
    keys = [r["key"] for r in rows(tmp_path)]
    assert summary.ran == 2 and sorted(keys) == ["g/0", "g/1", "g/2"]
    assert (tmp_path / "results.truncated").exists()
    results.write_text(results.read_text(encoding="utf-8") + '{"key": "g/0"}\n', "utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        BatchRunner(tmp_path, CAPS, probe_worker).run(jobs)


# --- committed calib probe --------------------------------------------------------------


def test_committed_probe_touched_calib_only(runset: ImmutableRunSet) -> None:
    from tools.test_batch import PROBE_POSITIONS

    out = ROOT / PROBE_DIR  # v3: after amendment 01, the cost model of the frozen run set
    data = rows(out)
    calib = {e.building_id for e in runset.entries if e.split == "calib"}
    probe_ids = {
        e.building_id
        for e in runset.entries
        if e.split == "calib" and e.position in PROBE_POSITIONS
    }
    assert {r["split"] for r in data} == {"calib"}
    assert {r["building_id"] for r in data} == probe_ids <= calib
    assert len(data) == len(PROBE_POSITIONS) * 7 * 8 and len({r["key"] for r in data}) == len(data)
    assert all(
        r["mean_loss"] is not None and r["requests"] == r["answered"] + r["defaulted"] for r in data
    )
    assert all(r["coverage_closed_final"] is not None for r in data)
    params = manifest_params(ROOT, runset.prereg)
    entry = next(e for e in runset.entries if e.building_id == data[0]["building_id"])
    cell = next(c for c in CELLS if c.cell_id == data[0]["cell_id"])
    row = data[0]
    values = runset.prereg["tuning"]["grid"][row["method_id"]][row["config_id"]]
    config = episode_config(entry, params, cell, row["method_id"], row["config_id"], values, 0)
    raw = json.loads(gzip.decompress((out / row["raw"]).read_bytes()))
    assert raw["config"]["transport_id"] == config.transport_id
    assert config.config_sha256 == row["config_sha256"]


# --- audit fixes: strict mode, breaker, lock, guards ---------------------------------------


def strict(out: Path, **kw: object) -> BatchRunner:
    return BatchRunner(out, CAPS, probe_worker, strict=True, **kw)  # type: ignore[arg-type]


def test_strict_worker_exceptions_are_infrastructure_and_stop_the_batch(tmp_path: Path) -> None:
    jobs = [BatchJob("g/0", "g", {}), BatchJob("g/1", "g", {"fail": True})]
    jobs += [BatchJob(f"g/{i}", "g", {}) for i in range(2, 5)]
    summary = strict(tmp_path).run(jobs)
    infra = [json.loads(x) for x in (tmp_path / "infra.jsonl").read_text("utf-8").splitlines()]
    assert summary.stop_reason and "infrastructure" in summary.stop_reason
    assert [r["key"] for r in infra] == ["g/1"] and "g/1" not in {r["key"] for r in rows(tmp_path)}
    fixed = [BatchJob(j.key, j.group, {}) for j in jobs]
    again = strict(tmp_path).run(fixed)
    assert again.stop_reason is None
    assert sorted(r["key"] for r in rows(tmp_path)) == sorted(j.key for j in jobs)


def test_strict_hard_kill_goes_to_infra_with_its_failure_row(tmp_path: Path) -> None:
    caps = BatchCaps(wall_hours=1.0, cpu_hours=1.0, workers=1, episode_timeout_s=2.0)
    runner = BatchRunner(
        tmp_path, caps, probe_worker, on_kill=lambda p: {"mean_loss": 0.5}, strict=True
    )
    summary = runner.run([BatchJob("g/0", "g", {"sleep_s": 60}), BatchJob("g/1", "g", {})])
    infra = json.loads((tmp_path / "infra.jsonl").read_text("utf-8"))
    assert infra["killed"] and infra["status"] == "timeout" and infra["mean_loss"] == 0.5
    assert summary.stop_reason and not (tmp_path / "results.jsonl").exists()


def test_breaker_stops_on_repeated_failures(tmp_path: Path) -> None:
    jobs = [BatchJob(f"g/{i}", "g", {"status": "error"}) for i in range(6)]
    jobs += [BatchJob(f"h/{i}", "h", {}) for i in range(3)]
    summary = strict(tmp_path, breaker_failures=2, breaker_window=5).run(jobs)
    assert summary.stop_reason and summary.stop_reason.startswith("breaker")
    assert all(r["status"] == "error" for r in rows(tmp_path))  # results stay results
    assert "h" not in summary.completed_groups


def test_a_lock_prevents_two_runners_and_a_missing_newline_is_repaired(tmp_path: Path) -> None:
    (tmp_path / "LOCK").write_text("123\n", "utf-8")
    with pytest.raises(RuntimeError, match="another runner"):
        strict(tmp_path).run([BatchJob("g/0", "g", {})])
    (tmp_path / "LOCK").unlink()
    strict(tmp_path).run([BatchJob("g/0", "g", {})])
    results = tmp_path / "results.jsonl"
    results.write_bytes(results.read_bytes().rstrip(b"\n"))  # cut exactly before the newline
    strict(tmp_path).run([BatchJob("g/0", "g", {}), BatchJob("g/1", "g", {})])
    assert [r["key"] for r in rows(tmp_path)] == ["g/0", "g/1"]
    assert not (tmp_path / "LOCK").exists()


def test_episodes_refuse_a_changed_source_tree_and_unapproved_test_buildings(
    runset: ImmutableRunSet,
) -> None:
    calib = runset.jobs(("calib",), 1, 1, Path("x"), "c")[0].payload
    with pytest.raises(RunSetError, match="source tree changed"):
        _config({**calib, "src_sha256": "0" * 64})
    test = runset.jobs(("id_test",), 1, 1, Path("x"), "c")[0].payload
    with pytest.raises(RunSetError, match="without a matching approval"):
        _config(test)
    with pytest.raises(RunSetError, match="without a matching approval"):
        _config({**test, "approval_sha256": "0" * 64})
    _, config = _config(calib)
    assert config.code_commit == "c"


def test_the_pilot_worker_never_runs_test_buildings(runset: ImmutableRunSet) -> None:
    from argus.experiments.batch import protocol_episode

    payload = runset.jobs(("ood_test",), 1, 1, Path("x"), "c")[0].payload
    with pytest.raises(ValueError, match="never runs test buildings"):
        protocol_episode({**payload, "failure_loss": 4.0})


def test_killed_raw_records_are_kept_apart_from_rerun_records(
    calib_payload: dict, tmp_path: Path
) -> None:
    payload = {**calib_payload, "out": str(tmp_path), "key": "calib/x/k"}
    first = killed_episode(payload)
    assert first["raw"].startswith("raw/infra/") and ".killed-" in first["raw"]
    assert runset_sources()["probe_results"]["path"] == f"{PROBE_DIR}/results.jsonl"


def runset_sources() -> dict:
    return json.loads((ROOT / FROZEN_PATH).read_text(encoding="utf-8"))["sources"]


# --- amendment 01 ----------------------------------------------------------------------


def test_transport_key_is_method_neutral_and_frames_carry_only_it(
    runset: ImmutableRunSet,
) -> None:
    from argus.communication.messages import Message

    jobs = runset.jobs(("calib",), 1, 2, Path("x"), "c")
    keys: dict[int, set[str]] = {0: set(), 1: set()}
    for job in jobs:
        _, config = _config(job.payload)
        keys[job.payload["replicate"]].add(config.transport_key)
        assert len(config.transport_key) == 16
    assert all(len(v) == 1 for v in keys.values()) and keys[0] != keys[1]
    sizes = {}
    for job in jobs:
        p = job.payload
        if p["replicate"] or p["cell_id"] != "dyn-ind-relay":
            continue
        if p["method_id"] not in ("repair", "dynamics_ids", "cross_rank"):
            continue
        _, config = _config(p)
        config = dataclasses.replace(
            config, scenario=dataclasses.replace(config.scenario, horizon_ticks=1500)
        )
        runner = ExperimentRunner()
        assert runner.run(config).status == "ok"
        frames = [a for a in runner.last_channel.attempts if isinstance(a.message, Message)]
        assert {a.message.observation.evidence_id.episode for a in frames} <= keys[0]
        assert all(p["method_id"] not in a.message.frame_id for a in frames)
        sizes[p["method_id"]] = {a.size - len(a.message.frame_id) for a in frames}
    assert sum(bool(v) for v in sizes.values()) >= 2  # at least two methods sent evidence


def test_a_second_kill_of_the_same_key_becomes_a_failure_row(tmp_path: Path) -> None:
    caps = BatchCaps(wall_hours=1.0, cpu_hours=1.0, workers=1, episode_timeout_s=2.0)

    def runner() -> BatchRunner:
        return BatchRunner(
            tmp_path, caps, probe_worker, on_kill=lambda p: {"mean_loss": 0.5}, strict=True
        )

    jobs = [BatchJob("g/0", "g", {"sleep_s": 60})]
    first = runner().run(jobs)
    assert first.stop_reason and not (tmp_path / "results.jsonl").exists()
    second = runner().run(jobs)
    got = rows(tmp_path)
    assert second.stop_reason is None and got[0]["key"] == "g/0" and got[0]["killed"]
    assert got[0]["status"] == "timeout" and got[0]["mean_loss"] == 0.5
