"""ARGUS-E02-T03: exogenous passage timelines, scenario validation and 2D backend."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from argus.dynamics.binary_ctmc import BinaryCTMC
from argus.world.geometry import Geometry, Passage, Rect, Room
from argus.world.scenario import (
    CtmcPassage,
    ExogenousWorld,
    RobotStart,
    ScenarioError,
    ScenarioSpec,
    ScriptedPassage,
    WorldBackend2D,
)

R = 0.2


def geometry() -> Geometry:
    return Geometry(
        bounds=Rect(0.0, 0.0, 10.0, 4.0),
        walls=(Rect(4.9, 0.0, 5.1, 1.4), Rect(4.9, 2.6, 5.1, 4.0)),
        rooms=(Room("west", Rect(0.0, 0.0, 4.9, 4.0)), Room("east", Rect(5.1, 0.0, 10.0, 4.0))),
        passages=(Passage("door", Rect(4.9, 1.4, 5.1, 2.6), ("west", "east")),),
    )


STARTS = (
    RobotStart("r0", (1.0, 1.0, 0.0)),
    RobotStart("r1", (1.0, 3.0, 0.0)),
    RobotStart("r2", (8.0, 1.0, math.pi)),
    RobotStart("r3", (8.0, 3.0, math.pi)),
)


def spec(model: CtmcPassage | ScriptedPassage | None = None, **changes: object) -> ScenarioSpec:
    base = ScenarioSpec(
        scenario_id="two-rooms-v1",
        geometry=geometry(),
        tick_seconds=0.1,
        horizon_ticks=3000,
        robot_radius=R,
        v_max=0.5,
        omega_max=1.0,
        starts=STARTS,
        passages=(("door", model or CtmcPassage(0.5, 1.0)),),
    )
    return replace(base, **changes) if changes else base  # type: ignore[arg-type]


def timeline(
    world: WorldBackend2D, ticks: int, actions: dict[str, tuple[float, float]]
) -> list[int]:
    states = [world.passage_states()["door"]]
    for tick in range(1, ticks + 1):
        world.advance(tick, actions)
        states.append(world.passage_states()["door"])
    return states


def test_same_world_seed_gives_same_timeline_whatever_the_actions() -> None:
    idle = timeline(WorldBackend2D(spec(), 11), 400, {})
    busy = timeline(WorldBackend2D(spec(), 11), 400, {"r0": (0.5, 0.3), "r2": (0.4, -1.0)})
    assert idle == busy
    assert 0 < sum(idle) < len(idle)  # the passage actually changes
    other_seed = timeline(WorldBackend2D(spec(), 12), 400, {})
    assert other_seed != idle


def test_query_order_does_not_change_the_timeline() -> None:
    forward = ExogenousWorld(spec(), 5)
    sequential = [forward.state("door", k) for k in range(300)]
    backward = ExogenousWorld(spec(), 5)
    reverse = [backward.state("door", k) for k in reversed(range(300))][::-1]
    assert reverse == sequential


def test_ctmc_skeleton_matches_transition_probabilities() -> None:
    # Deterministic keys, so these 5-sigma checks cannot flake.
    model = CtmcPassage(0.5, 1.0)
    p01, p11 = (BinaryCTMC(0.5, 1.0).transition(0.1)[z][1] for z in (0, 1))
    from_closed = [0, 0]  # [count, opened]
    from_open = [0, 0]
    initial_open = 0
    seeds = range(40)
    for seed in seeds:
        world = ExogenousWorld(spec(model), seed)
        states = [world.state("door", k) for k in range(2001)]
        initial_open += states[0]
        for a, b in zip(states, states[1:], strict=False):
            bucket = from_open if a else from_closed
            bucket[0] += 1
            bucket[1] += b
    for (n, hits), p in ((from_closed, p01), (from_open, p11)):
        assert abs(hits / n - p) < 5 * math.sqrt(p * (1 - p) / n)
    pi = 0.5 / 1.5
    assert abs(initial_open / len(seeds) - pi) < 5 * math.sqrt(pi * (1 - pi) / len(seeds))


def test_scripted_passage_follows_script_only() -> None:
    script = ScriptedPassage(1, ((3, 0), (7, 1), (8, 0)))
    for seed in (1, 2, 3):
        world = ExogenousWorld(spec(script), seed)
        assert [world.state("door", k) for k in range(10)] == [1, 1, 1, 0, 0, 0, 0, 1, 0, 0]


@pytest.mark.parametrize(
    "build",
    [
        lambda: ScriptedPassage(1, ((3, 0), (3, 1))),
        lambda: ScriptedPassage(1, ((5, 0), (2, 1))),
        lambda: ScriptedPassage(1, ((0, 0),)),
        lambda: ScriptedPassage(2),
        lambda: spec(passages=()),
        lambda: spec(passages=(("door", CtmcPassage(0.1, 0.1)), ("door", ScriptedPassage(1)))),
        lambda: spec(passages=(("door", CtmcPassage(0.1, 0.1)), ("ghost", ScriptedPassage(1)))),
        lambda: spec(CtmcPassage(0.0, 0.0)),
        lambda: spec(CtmcPassage(0.1, 0.1, 1.5)),
    ],
)
def test_ctmc_and_script_cannot_both_drive_a_passage(build: object) -> None:
    with pytest.raises(ScenarioError):
        build()  # type: ignore[operator]


def test_zero_rate_ctmc_with_explicit_initial_state_is_valid() -> None:
    world = ExogenousWorld(spec(CtmcPassage(0.0, 0.0, 1.0)), 3)
    assert [world.state("door", k) for k in range(50)] == [1] * 50


@pytest.mark.parametrize(
    "starts",
    [
        (RobotStart("r0", (5.0, 0.5, 0.0)),),  # inside a wall
        (RobotStart("r0", (5.0, 2.0, 0.0)),),  # inside the passage
        (RobotStart("r0", (0.1, 1.0, 0.0)),),  # disc crosses the bounds
        (RobotStart("r0", (1.0, 1.0, 0.0)), RobotStart("r1", (1.3, 1.0, 0.0))),  # overlap
        (RobotStart("r0", (1.0, 1.0, 0.0)), RobotStart("r0", (3.0, 1.0, 0.0))),  # duplicate id
        (),
        (RobotStart("r0", (1.0, math.nan, 0.0)),),
    ],
)
def test_invalid_starts_rejected(starts: tuple[RobotStart, ...]) -> None:
    with pytest.raises(ScenarioError):
        spec(starts=starts)


def test_four_collision_free_starts_accepted_and_touching_allowed() -> None:
    assert len(spec().starts) == 4
    touching = (RobotStart("r0", (1.0, 1.0, 0.0)), RobotStart("r1", (1.4, 1.0, 0.0)))
    spec(starts=touching)


def test_scenario_hash_is_stable_and_sensitive() -> None:
    assert spec().scenario_hash == spec().scenario_hash
    assert len(spec().scenario_hash) == 64
    variants = [
        spec(tick_seconds=0.05),
        spec(CtmcPassage(0.5, 1.1)),
        spec(scenario_id="other"),
        spec(starts=STARTS[:3]),
    ]
    assert len({s.scenario_hash for s in variants} | {spec().scenario_hash}) == 5
    assert spec().to_json()["generator_version"] == "argus-scenario-v1"
    with pytest.raises(ScenarioError):
        spec(generator_version="argus-scenario-v0")


def test_motion_uses_previous_passage_state_and_closure_is_logged() -> None:
    # Door open for ticks 0-1, closed from tick 2 on.
    world = WorldBackend2D(spec(ScriptedPassage(1, ((2, 0),)), v_max=10.0), 1)
    world.restore(
        0,
        {
            "r0": (4.7, 2.0, 0.0),
            "r1": (1.0, 3.0, 0.0),
            "r2": (8.0, 1.0, math.pi),
            "r3": (8.0, 3.0, math.pi),
        },
    )
    step1 = world.advance(1, {"r0": (2.0, 0.0)})  # moves into the open door under z(0)=1
    assert step1.collisions == () and world.poses()["r0"][0] == pytest.approx(4.9)
    step2 = world.advance(2, {})
    assert step2.passage_changes == (("door", 0),)
    assert [(c.passage_id, c.robot_id) for c in step2.closure_conflicts] == [("door", "r0")]
    assert world.poses()["r0"][0] == pytest.approx(4.9)  # not teleported
    blocked = world.advance(3, {"r0": (2.0, 0.0)})
    assert [e.kind for e in blocked.collisions] == ["wall"]


def test_backend_rejects_wrong_ticks_robots_and_speeds() -> None:
    world = WorldBackend2D(spec(), 1)
    with pytest.raises(ValueError):
        world.advance(2, {})
    with pytest.raises(ValueError):
        world.advance(1, {"ghost": (0.1, 0.0)})
    with pytest.raises(ValueError):
        world.advance(1, {"r0": (0.6, 0.0)})
    with pytest.raises(ValueError):
        world.advance(1, {"r0": (math.inf, 0.0)})
    truth = world.truth_snapshot()
    assert truth.tick == 0 and dict(truth.robots)["r0"] == (1.0, 1.0, 0.0)
