"""ARGUS-E05-T03: inspection requests and the closed-loop episode runner."""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import pytest

from argus.beliefs.exact_micro import ImpossibleEvidence
from argus.core.types import AgentAction, AgentView
from argus.experiments import runner as runner_module
from argus.experiments.runner import (
    EpisodeConfig,
    ExperimentRunner,
    binary_loss,
    generate_requests,
    write_manifest,
)
from argus.policies import frontier as frontier_module
from argus.world.geometry import Geometry, Passage, Rect, Room
from argus.world.scenario import CtmcPassage, RobotStart, ScenarioSpec

HORIZON = 400


def scenario() -> ScenarioSpec:
    t = 0.1
    walls = (
        Rect(0.0, 0.0, 12.0, t),
        Rect(0.0, 4.0 - t, 12.0, 4.0),
        Rect(0.0, 0.0, t, 4.0),
        Rect(12.0 - t, 0.0, 12.0, 4.0),
        Rect(5.9, 0.0, 6.1, 1.4),
        Rect(5.9, 2.6, 6.1, 4.0),
    )
    geometry = Geometry(
        Rect(0.0, 0.0, 12.0, 4.0),
        walls,
        (Room("west", Rect(0.0, 0.0, 5.9, 4.0)), Room("east", Rect(6.1, 0.0, 12.0, 4.0))),
        (Passage("door", Rect(5.9, 1.4, 6.1, 2.6), ("west", "east")),),
    )
    starts = (
        RobotStart("r0", (1.0, 1.0, 0.0)),
        RobotStart("r1", (1.0, 3.0, 0.0)),
        RobotStart("r2", (11.0, 1.0, math.pi)),
        RobotStart("r3", (11.0, 3.0, math.pi)),
    )
    return ScenarioSpec(
        "runner-fixture",
        geometry,
        0.1,
        HORIZON,
        0.2,
        0.5,
        1.0,
        starts,
        (("door", CtmcPassage(0.05, 0.1)),),
    )


def config(method: str = "frontier", **changes: object) -> EpisodeConfig:
    base = EpisodeConfig("ep-1", "building-a", scenario(), method, 1, 2, 3, 4, 5)
    return dataclasses.replace(base, **changes) if changes else base  # type: ignore[arg-type]


def test_four_agents_run_to_the_horizon() -> None:
    runner = ExperimentRunner()
    result = runner.run(config())
    assert result.status == "ok" and result.error is None
    assert runner.last_world_tick == HORIZON
    assert {r.robot_id for r in result.records} == {"r0", "r1", "r2", "r3"}
    assert result.unanswered == 0 and result.manifest["status"] == "ok"
    reservations = [
        a for a in runner.last_channel.attempts if a.message.frame_id.startswith("res/")
    ]
    assert reservations, "the frontier agents coordinated through paid frames"


def test_request_plan_is_fixed_and_method_independent() -> None:
    plan = generate_requests(config())
    assert len(plan) == 4 * (HORIZON // 50)
    assert generate_requests(config("wait")) == plan
    frontier = ExperimentRunner().run(config())
    waiting = ExperimentRunner().run(config("wait"))
    key = [(r.tick, r.robot_id, r.entity) for r in frontier.records]
    assert key == [(r.tick, r.robot_id, r.entity) for r in waiting.records]
    assert [(r.tick, r.recipient, r.entity) for r in plan] == key


def test_unknown_entity_is_answered_with_the_prior() -> None:
    result = ExperimentRunner().run(config("wait"))
    # Waiting robots in the corners never see the door, so they answer with the prior.
    unseen = [r for r in result.records if r.robot_id in ("r0", "r1")]
    assert unseen and all(r.probability == pytest.approx(1 / 3, abs=1e-12) for r in unseen)
    assert all(r.decision == 0 for r in unseen)  # 1/3 < threshold .8


def test_losses_follow_the_pilot_costs() -> None:
    assert config().decision_threshold == pytest.approx(0.8)
    assert binary_loss(1, 0, 4, 1) == 4 and binary_loss(0, 1, 4, 1) == 1
    assert binary_loss(1, 1, 4, 1) == binary_loss(0, 0, 4, 1) == 0
    for record in ExperimentRunner().run(config()).records:
        assert record.decision == int(record.probability >= 0.8)
        assert record.decision_loss == binary_loss(record.decision, record.truth, 4, 1)


def test_future_requests_never_reach_agents(monkeypatch: pytest.MonkeyPatch) -> None:
    views: list[AgentView] = []
    original = frontier_module.FrontierPolicy.act

    def spy(self: object, view: AgentView, key: str) -> AgentAction:
        views.append(view)
        return original(self, view, key)  # type: ignore[arg-type]

    monkeypatch.setattr(frontier_module.FrontierPolicy, "act", spy)
    ExperimentRunner().run(config())
    seen = list(views)
    views.clear()
    ExperimentRunner().run(config(seed_requests=99))  # a different future request plan
    assert views == seen  # identical views: requests leave no trace in what agents see
    assert all("request" not in f.name for f in dataclasses.fields(AgentView))


class Explodes:
    def act(self, view: AgentView, key: str) -> AgentAction:
        raise RuntimeError("policy bug")


class Impossible:
    def act(self, view: AgentView, key: str) -> AgentAction:
        raise ImpossibleEvidence("zero likelihood")


@pytest.mark.parametrize(("policy", "status"), [(Explodes, "error"), (Impossible, "invalid")])
def test_failures_are_not_reported_as_success(
    monkeypatch: pytest.MonkeyPatch, policy: type, status: str
) -> None:
    monkeypatch.setitem(runner_module.METHODS, "frontier", lambda config, team: policy())
    result = ExperimentRunner().run(config())
    assert result.status == status and result.records == () and result.error
    assert result.manifest["status"] == status


def test_timeout_is_its_own_status() -> None:
    result = ExperimentRunner().run(config(max_wall_seconds=0.0))
    assert result.status == "timeout"
    assert result.unanswered == len(generate_requests(config())) - len(result.records)


def test_manifest_is_complete_and_written_atomically(tmp_path: Path) -> None:
    result = ExperimentRunner().run(config(code_commit="0" * 40))
    manifest = result.manifest
    for key in (
        "schema_version",
        "run_id",
        "code_commit",
        "config_sha256",
        "scenario_hash",
        "rng_schema",
        "method_components",
        "python",
        "numpy",
        "status",
        "requests",
    ):
        assert key in manifest
    assert manifest["config"]["seeds"] == {
        "world": 1,
        "sensor": 2,
        "channel": 3,
        "requests": 4,
        "policy": 5,
    }
    path = write_manifest(tmp_path / "run", manifest)
    assert (
        json.loads(path.read_text(encoding="utf-8"))["config_sha256"] == manifest["config_sha256"]
    )
    assert sorted(p.name for p in path.parent.iterdir()) == ["manifest.json"]
    with pytest.raises(FileExistsError):
        write_manifest(tmp_path / "run", manifest)
    text = json.dumps(manifest)
    assert ":\\\\" not in text and "Users" not in text


def test_episode_is_deterministic() -> None:
    a = ExperimentRunner().run(config())
    b = ExperimentRunner().run(config())
    assert a.records == b.records
    assert a.manifest["config_sha256"] == b.manifest["config_sha256"]


def test_memory_tracing_is_opt_in() -> None:
    assert ExperimentRunner().run(config("wait")).peak_traced_bytes is None
    traced = ExperimentRunner().run(config("wait", trace_memory=True))
    assert traced.peak_traced_bytes is not None and traced.peak_traced_bytes > 0
