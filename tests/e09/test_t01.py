"""ARGUS-E09-T01: intervention schema and paired branching (change request 2026-11-07)."""

from __future__ import annotations

import dataclasses
import inspect
import math
from pathlib import Path

import pytest

from argus.experiments.cells import FactorCell
from argus.experiments.protocol import episode_config, manifest_params
from argus.experiments.runner import EpisodeConfig, ExperimentRunner
from argus.experiments.runset import ImmutableRunSet
from argus.policies.baselines import METHOD_SPECS
from argus.replay import interventions
from argus.replay.interventions import ARMS, InterventionSpec, open_window_s

ROOT = Path(__file__).resolve().parents[2]
HORIZON_TICKS = 600  # 60 s keeps the episodes short; the mechanics do not depend on it


def base(method: str) -> EpisodeConfig:
    runset = ImmutableRunSet.load(ROOT)
    entry = runset.buildings("calib", 1)[0][0]
    params = manifest_params(ROOT, runset.prereg)
    config = episode_config(entry, params, FactorCell(True, False, False), method, "h5.0", {}, 0)
    scenario = dataclasses.replace(config.scenario, horizon_ticks=HORIZON_TICKS)
    return dataclasses.replace(config, scenario=scenario)


@pytest.fixture(scope="module")
def runs() -> dict[str, object]:
    out = {}
    for arm in ("repair", "repair_observe", "repair_send"):
        config = InterventionSpec(arm, 1.0, 4.0).apply(base(arm))
        out[arm] = ExperimentRunner(trace=True).run(config)
    return out


def test_arms_share_seeds_transport_and_exogenous_world(runs: dict) -> None:
    configs = [InterventionSpec(a, 1.0, 4.0).apply(base(a)) for a in ("repair", "repair_observe")]
    first, second = (c.to_json() for c in configs)
    assert first["seeds"] == second["seeds"]
    assert first["transport_id"] == second["transport_id"]
    traces = [runs[a].trace for a in ("repair", "repair_observe", "repair_send")]
    assert all(t["passage_initial"] == traces[0]["passage_initial"] for t in traces)
    assert all(t["passage_changes"] == traces[0]["passage_changes"] for t in traces)


def test_switched_off_actions_are_never_taken(runs: dict) -> None:
    observe = runs["repair_observe"].trace["policy_actions"]
    send = runs["repair_send"].trace["policy_actions"]
    full = runs["repair"].trace["policy_actions"]
    assert all(c["send_frames"] == 0 for c in observe.values())
    assert all(c["observe_decisions"] == 0 for c in send.values())
    assert sum(c["observe_decisions"] for c in full.values()) > 0
    assert sum(c["send_frames"] for c in full.values()) > 0


def test_after_divergence_the_runs_differ(runs: dict) -> None:
    full, observe = runs["repair"], runs["repair_observe"]
    assert full.status == observe.status == "ok"
    probs = [[r.probability for r in x.records] for x in (full, observe)]
    assert probs[0] != probs[1]  # new trajectories produce new readings and beliefs
    assert full.trace["coverage_events"] != observe.trace["coverage_events"]


def test_door_tempo_scales_rates_and_keeps_the_open_share() -> None:
    for scale in (0.25, 0.5, 2.0):
        config = InterventionSpec("no_comm", scale, 4.0).apply(base("no_comm"))
        rates = config.nominal.z_model
        assert rates.rate_01 == pytest.approx(0.01 * scale)
        assert rates.rate_10 == pytest.approx(0.02 * scale)
        assert config.nominal.prior_open == pytest.approx(1 / 3)
    static = InterventionSpec("no_comm", 0.0, 4.0).apply(base("no_comm"))
    assert static.nominal.z_model.rate_01 == static.nominal.z_model.rate_10 == 0.0
    assert static.nominal.prior_open == pytest.approx(1 / 3)


def test_cost_ratio_moves_the_threshold() -> None:
    config = InterventionSpec("repair", 1.0, 2.0).apply(base("repair"))
    assert (config.loss_false_open, config.loss_false_closed) == (2.0, 1.0)
    assert config.loss_false_open / (config.loss_false_open + 1) == pytest.approx(2 / 3)


def test_the_open_window_matches_the_analytic_values() -> None:
    assert open_window_s(1.0, 4.0) == pytest.approx(11.889, abs=1e-3)
    assert open_window_s(0.25, 4.0) == pytest.approx(4 * 11.889, abs=1e-2)
    assert open_window_s(1.0, 2.0) == pytest.approx(23.105, abs=1e-3)
    assert math.isinf(open_window_s(0.0, 4.0))


def test_an_intervention_reruns_the_simulator_and_never_edits_a_log() -> None:
    source = inspect.getsource(interventions)
    assert "snapshot" not in source and "playback" not in source.split('"""')[2]
    with pytest.raises(ValueError, match="intervention arm"):
        InterventionSpec("repair_observe").apply(base("repair"))


def test_arms_are_registered_as_diagnostics_never_as_controls() -> None:
    assert set(ARMS) <= set(METHOD_SPECS)
    assert METHOD_SPECS["repair_observe"].role == "diagnostic"
    assert METHOD_SPECS["repair_send"].role == "diagnostic"
    with pytest.raises(ValueError):
        InterventionSpec("frontier")
    with pytest.raises(ValueError):
        InterventionSpec("repair", -1.0)
