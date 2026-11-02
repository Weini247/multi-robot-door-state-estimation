"""ARGUS-E05-T01: local agents, immutable views and the paired hidden-world canary."""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import pytest

from argus.agents.local import LocalAgent
from argus.beliefs.fixed_lag import FixedLagBelief, NominalPassageModel
from argus.core.events import KeyedRandom
from argus.core.types import AgentAction, AgentView, EvidenceId, Observation, Reservation, Send
from argus.dynamics.binary_ctmc import BinaryCTMC
from argus.sensors.binary import BinarySensor, IndependentNoise, SensorConfig
from argus.world.geometry import Geometry, Passage, Rect, Room
from argus.world.scenario import RobotStart, ScenarioSpec, ScriptedPassage, WorldBackend2D

MODEL = NominalPassageModel(BinaryCTMC(0.01, 0.02), BinaryCTMC(0.005, 0.045), 0.05, 1 / 3, 0.1)
SRC = Path(__file__).resolve().parents[2] / "src" / "argus"


def agent(robot: str = "r0") -> LocalAgent:
    belief = FixedLagBelief("ep", {}, 0.1, 300, 4096, default_model=MODEL)
    return LocalAgent(robot, belief, (2.0, 2.0, 0.0), 0.25)


def reading(origin: str, seq: int, tick: int, value: int = 1, entity: str = "door") -> Observation:
    return Observation(EvidenceId("ep", origin, seq), tick, entity, value, (2.0, 2.0, 0.0), "m")


def test_view_is_immutable_and_detached() -> None:
    a = agent()
    a.observe([reading("r0", 0, 0)], 0)
    a.update_map([(1, 1), (1, 2)], [(3, 3)])
    view = a.view(0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        view.pose = (0.0, 0.0, 0.0)  # type: ignore[misc]
    assert isinstance(view.free_map, tuple) and isinstance(view.known_evidence, tuple)
    a.update_map([(9, 9)], [])
    assert (9, 9) not in view.free_map  # the old view does not change afterwards


def test_cross_agent_mutation_is_impossible() -> None:
    a, b = agent("r0"), agent("r1")
    shared = reading("r0", 0, 0)
    a.observe([shared], 0)
    b.receive([shared], 2)
    before = b.view(2)
    a.update_map([(1, 1)], [])
    a.observe([reading("r0", 1, 3, value=0)], 3)
    a.receive_reservation(Reservation("r2", (5, 5), 50), 3)
    after = b.view(2)
    assert after == before
    assert a.view(3).belief != b.view(3).belief


def test_unknown_entity_reports_the_nominal_prior() -> None:
    a = agent()
    assert a.probability_open("never-seen", 0) == pytest.approx(1 / 3, abs=1e-15)
    a.observe([reading("r0", 0, 0, entity="door")], 0)
    assert a.probability_open("door", 0) > 1 / 3


def test_inbox_holds_only_newly_accepted_deliveries_until_decided() -> None:
    a = agent()
    msg = reading("r1", 0, 1)
    assert a.receive([msg, msg], 2) == ["accepted", "duplicate"]
    assert a.view(2).inbox == (msg,)

    class Wait:
        def act(self, view: AgentView, random_key: str) -> AgentAction:
            return AgentAction("wait", 1)

    a.decide(2, Wait(), "k")
    assert a.view(3).inbox == ()


def test_agent_only_sends_known_evidence_and_reservations_expire() -> None:
    a = agent()

    class Leak:
        def act(self, view: AgentView, random_key: str) -> AgentAction:
            return AgentAction("wait", 1, sends=(Send("r1", EvidenceId("ep", "r9", 0)),))

    with pytest.raises(ValueError):
        a.decide(0, Leak(), "k")
    a.receive_reservation(Reservation("r1", (4, 4), 10), 0)
    assert [r.robot_id for r in a.view(9).reservations] == ["r1"]
    assert a.view(10).reservations == ()
    with pytest.raises(ValueError):
        a.receive_reservation(Reservation("r0", (4, 4), 10), 0)
    with pytest.raises(ValueError):
        a.observe([reading("r1", 0, 0)], 0)  # not its own reading


def test_policy_and_belief_modules_import_no_world_or_evaluator() -> None:
    import ast

    forbidden = (
        "argus.world",
        "argus.sensors",
        "argus.experiments",
        "argus.evaluation",
        "argus.replay",
    )
    for package in ("agents", "policies", "beliefs"):
        for path in (SRC / package).glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            names = {n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
            names |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
            assert not any(name.startswith(forbidden) for name in names), path


# --- paired hidden-world canary --------------------------------------------------------


def canary_world(back: int) -> WorldBackend2D:
    geometry = Geometry(
        bounds=Rect(-2.0, 0.0, 10.0, 4.0),
        walls=(Rect(4.9, 0.0, 5.1, 1.4), Rect(4.9, 2.6, 5.1, 4.0)),
        rooms=(
            Room("hall", Rect(-2.0, 0.0, 0.0, 4.0)),
            Room("west", Rect(0.2, 0.0, 4.9, 4.0)),
            Room("east", Rect(5.1, 0.0, 10.0, 4.0)),
        ),
        passages=(
            Passage("door", Rect(4.9, 1.4, 5.1, 2.6), ("west", "east")),
            Passage("back", Rect(0.0, 1.4, 0.2, 2.6), ("west", "hall")),
        ),
    )
    spec = ScenarioSpec(
        scenario_id="canary",
        geometry=geometry,
        tick_seconds=0.1,
        horizon_ticks=500,
        robot_radius=0.2,
        v_max=0.5,
        omega_max=1.0,
        starts=(RobotStart("r0", (2.0, 2.0, 0.0)),),
        passages=(("door", ScriptedPassage(1, ((120, 0),))), ("back", ScriptedPassage(back))),
    )
    return WorldBackend2D(spec, 1)


class TurnWhenDoubtful:
    """Uses nothing but the view: slow turn while the door belief is uncertain."""

    def act(self, view: AgentView, random_key: str) -> AgentAction:
        p = dict(view.belief.passage_probabilities).get("door", 0.5)
        omega = 0.2 if 0.2 < p < 0.8 else 0.0
        return AgentAction("move", 10, v=0.0, omega=omega)


def run_canary(back: int) -> tuple[list[AgentView], list[AgentAction]]:
    world = canary_world(back)
    sensor = BinarySensor(
        SensorConfig(5.0, math.radians(120), 5, "binary-v1"),
        IndependentNoise(0.14, 0.14, KeyedRandom(4)),
        "ep",
    )
    a = agent()
    views, actions, action = [], [], AgentAction("wait", 1)
    for tick in range(0, 300):
        if tick:
            world.advance(tick, {"r0": (action.v, action.omega)})
            a.set_pose(world.poses()["r0"], False)
        if tick % 5 == 0:
            a.observe(sensor.observe(world, "r0", tick // 5), tick)
        if tick % 10 == 0:
            views.append(a.view(tick))
            action = a.decide(tick, TurnWhenDoubtful(), f"r0/{tick}")
            actions.append(action)
    return views, actions


def test_paired_hidden_world_canary() -> None:
    views_a, actions_a = run_canary(back=0)
    views_b, actions_b = run_canary(back=1)
    assert actions_a == actions_b
    assert views_a == views_b
    # Not vacuous: the visible door closes at tick 120 and the agent reacts to it.
    assert len({a.omega for a in actions_a}) > 1
    assert all("back" not in dict(v.belief.passage_probabilities) for v in views_a)
