"""ARGUS-E02-T04: snapshot resume reproduces the canonical event hash."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from argus.core.events import Phase, SimulationClock, SimulationEvent
from argus.replay.snapshot import (
    EventLog,
    SnapshotError,
    capture,
    load_snapshot,
    replay_events,
    resume,
    save_snapshot,
)
from argus.world.geometry import Geometry, Passage, Rect, Room
from argus.world.scenario import CtmcPassage, RobotStart, ScenarioSpec, WorldBackend2D

HORIZON = 120


def scenario(rate: float = 0.5) -> ScenarioSpec:
    geometry = Geometry(
        bounds=Rect(0.0, 0.0, 10.0, 4.0),
        walls=(Rect(4.9, 0.0, 5.1, 1.4), Rect(4.9, 2.6, 5.1, 4.0)),
        rooms=(Room("west", Rect(0.0, 0.0, 4.9, 4.0)), Room("east", Rect(5.1, 0.0, 10.0, 4.0))),
        passages=(Passage("door", Rect(4.9, 1.4, 5.1, 2.6), ("west", "east")),),
    )
    return ScenarioSpec(
        scenario_id="snapshot-fixture",
        geometry=geometry,
        tick_seconds=0.1,
        horizon_ticks=HORIZON,
        robot_radius=0.2,
        v_max=0.5,
        omega_max=1.0,
        starts=(
            RobotStart("r0", (1.0, 2.0, 0.0)),
            RobotStart("r1", (1.0, 3.0, 0.0)),
            RobotStart("r2", (8.0, 2.0, math.pi)),
            RobotStart("r3", (8.0, 1.0, math.pi)),
        ),
        passages=(("door", CtmcPassage(rate, 2 * rate)),),
    )


class Schedule:
    """Fixed open-loop action plan (a stand-in for policies); counts its calls."""

    calls = 0

    def __call__(self, tick: int) -> dict[str, list[float]]:
        Schedule.calls += 1
        turn = 0.6 if (tick // 15) % 2 else -0.6
        return {"r0": [0.5, 0.0], "r1": [0.3, turn], "r2": [0.5, 0.1], "r3": [0.2, -turn]}


def schedule_tick(clock: SimulationClock, tick: int, plan: Schedule) -> None:
    clock.schedule(SimulationEvent(tick, Phase.WORLD, f"world/{tick}", {"actions": plan(tick)}))
    clock.schedule(SimulationEvent(tick, Phase.EVALUATE, f"evaluate/{tick}"))


def run(
    clock: SimulationClock, world: WorldBackend2D, log: EventLog, plan: Schedule, until: int
) -> None:
    """Process events until the EVALUATE event of tick `until` (inclusive)."""
    while clock.peek() is not None:
        event = clock.pop()
        if event.phase is Phase.WORLD:
            assert isinstance(event.payload, dict)
            actions = {rid: (va[0], va[1]) for rid, va in event.payload["actions"].items()}
            step = world.advance(event.tick, actions)
            outcome: dict[str, object] = {
                "changes": [list(c) for c in step.passage_changes],
                "collisions": [[c.kind, list(c.robot_ids), c.fraction] for c in step.collisions],
                "closures": [[c.passage_id, c.robot_id] for c in step.closure_conflicts],
            }
        else:
            truth = world.truth_snapshot()
            outcome = {
                "passages": [list(p) for p in truth.passages],
                "poses": [[rid, list(pose)] for rid, pose in truth.robots],
            }
            if event.tick < HORIZON:
                schedule_tick(clock, event.tick + 1, plan)
        log.append(event, outcome)  # type: ignore[arg-type]
        if event.phase is Phase.EVALUATE and event.tick == until:
            return


def fresh(
    spec: ScenarioSpec, seed: int, plan: Schedule
) -> tuple[SimulationClock, WorldBackend2D, EventLog]:
    clock, world, log = SimulationClock(), WorldBackend2D(spec, seed), EventLog()
    schedule_tick(clock, 1, plan)
    return clock, world, log


def full_run(seed: int = 3) -> EventLog:
    clock, world, log = fresh(scenario(), seed, Schedule())
    run(clock, world, log, Schedule(), HORIZON)
    return log


def test_snapshot_resume_gives_identical_canonical_event_hash(tmp_path: Path) -> None:
    reference = full_run()
    spec, plan = scenario(), Schedule()
    clock, world, log = fresh(spec, 3, plan)
    run(clock, world, log, plan, 47)
    path = tmp_path / "episode.snapshot.json"
    save_snapshot(path, capture(clock, world, log))
    del clock, world, log

    clock2, world2, log2 = resume(load_snapshot(path, spec.scenario_hash), spec)
    run(clock2, world2, log2, plan, HORIZON)
    assert log2.head == reference.head
    assert len(log2) == len(reference) == 2 * HORIZON


def test_run_is_deterministic_and_seed_sensitive() -> None:
    assert full_run(3).head == full_run(3).head
    assert full_run(3).head != full_run(4).head


def test_absolute_time_and_phase_are_preserved(tmp_path: Path) -> None:
    spec, plan = scenario(), Schedule()
    clock, world, log = fresh(spec, 3, plan)
    run(clock, world, log, plan, 20)
    path = tmp_path / "s.json"
    save_snapshot(path, capture(clock, world, log))
    snapshot = load_snapshot(path)
    assert snapshot.world_tick == 20
    assert snapshot.last_key == (20, int(Phase.EVALUATE), "evaluate/20")
    clock2, world2, _ = resume(snapshot, spec)
    assert clock2.tick == 20 and world2.tick == 20
    assert world2.poses() == world.poses()
    assert world2.passage_states() == world.passage_states()
    with pytest.raises(ValueError):  # no event may be inserted into the past after resume
        clock2.schedule(SimulationEvent(20, Phase.POLICY, "late"))
    with pytest.raises(ValueError):  # used ids survive the snapshot
        clock2.schedule(SimulationEvent(30, Phase.WORLD, "world/5"))


def corrupt_variants(path: Path) -> dict[str, bytes]:
    raw = path.read_bytes()
    document = json.loads(raw)
    flipped = raw.replace(b'"world_tick":20', b'"world_tick":21')
    other_version = dict(document, snapshot=dict(document["snapshot"], schema_version=2))
    return {
        "flipped": flipped,
        "truncated": raw[: len(raw) // 2],
        "not utf8": b"\xff\xfe" + raw,
        "version without rehash": json.dumps(other_version).encode(),
    }


def test_corrupted_or_foreign_snapshots_rejected(tmp_path: Path) -> None:
    spec, plan = scenario(), Schedule()
    clock, world, log = fresh(spec, 3, plan)
    run(clock, world, log, plan, 20)
    path = tmp_path / "s.json"
    save_snapshot(path, capture(clock, world, log))
    for name, data in corrupt_variants(path).items():
        bad = tmp_path / f"{name}.json"
        bad.write_bytes(data)
        with pytest.raises(SnapshotError):
            load_snapshot(bad)
    with pytest.raises(SnapshotError):
        load_snapshot(path, scenario(rate=0.4).scenario_hash)
    with pytest.raises(SnapshotError):
        resume(load_snapshot(path), scenario(rate=0.4))


def test_other_snapshot_version_with_valid_hash_rejected(tmp_path: Path) -> None:
    from dataclasses import replace

    spec, plan = scenario(), Schedule()
    clock, world, log = fresh(spec, 3, plan)
    snapshot = replace(capture(clock, world, log), schema_version=2)
    path = tmp_path / "v2.json"
    save_snapshot(path, snapshot)
    with pytest.raises(SnapshotError, match="schema"):
        load_snapshot(path)
    save_snapshot(path, replace(snapshot, schema_version=1, rng_schema="argus-rng-v0"))
    with pytest.raises(SnapshotError, match="RNG"):
        load_snapshot(path)


def test_atomic_write_leaves_no_temporary_file(tmp_path: Path) -> None:
    clock, world, log = fresh(scenario(), 3, Schedule())
    path = tmp_path / "s.json"
    save_snapshot(path, capture(clock, world, log))
    save_snapshot(path, capture(clock, world, log))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["s.json"]


def test_replay_recomputes_hash_without_policy_calls() -> None:
    log = full_run()
    Schedule.calls = 0
    records = [json.loads(json.dumps(r)) for r in log.records]
    assert replay_events(records) == log.head
    assert Schedule.calls == 0
    tampered = [dict(r) for r in records]
    tampered[10] = {**tampered[10], "outcome": {"changes": [["door", 1]]}}
    assert replay_events(tampered) != log.head
    swapped = records[:]
    swapped[3], swapped[4] = swapped[4], swapped[3]
    with pytest.raises(SnapshotError, match="order"):
        replay_events(swapped)
