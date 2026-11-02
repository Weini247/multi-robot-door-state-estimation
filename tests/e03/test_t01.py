"""ARGUS-E03-T01: visible binary measurements, FOV boundaries and physical slots."""

from __future__ import annotations

import math

import pytest

from argus.core.events import KeyedRandom
from argus.sensors.binary import BinarySensor, IndependentNoise, SensorConfig, SensorModelError
from argus.world.geometry import Geometry, Passage, Rect, Room
from argus.world.scenario import RobotStart, ScenarioSpec, ScriptedPassage, WorldBackend2D

DOOR_CENTER = (5.0, 2.0)


def world(door: int = 1, starts: tuple[RobotStart, ...] | None = None, inner_wall: bool = False):
    walls = (Rect(4.9, 0.0, 5.1, 1.4), Rect(4.9, 2.6, 5.1, 4.0))
    if inner_wall:
        walls += (Rect(2.0, 1.0, 2.5, 3.0),)
    geometry = Geometry(
        bounds=Rect(0.0, 0.0, 10.0, 4.0),
        walls=walls,
        rooms=(Room("west", Rect(0.0, 0.0, 4.9, 4.0)), Room("east", Rect(5.1, 0.0, 10.0, 4.0))),
        passages=(Passage("door", Rect(4.9, 1.4, 5.1, 2.6), ("west", "east")),),
    )
    spec = ScenarioSpec(
        scenario_id="sensor-fixture",
        geometry=geometry,
        tick_seconds=0.1,
        horizon_ticks=1000,
        robot_radius=0.2,
        v_max=0.5,
        omega_max=1.0,
        starts=starts or (RobotStart("r0", (1.0, 2.0, 0.0)), RobotStart("r1", (8.0, 2.0, 0.0))),
        passages=(("door", ScriptedPassage(door)),),
    )
    return WorldBackend2D(spec, 1)


def sensor(fpr: float = 0.0, fnr: float = 0.0, fov: float = math.radians(120), seed: int = 9):
    config = SensorConfig(range_m=5.0, fov_rad=fov, slot_ticks=5, model_name="binary-v1")
    return BinarySensor(config, IndependentNoise(fpr, fnr, KeyedRandom(seed)), "ep-1")


def test_visible_open_and_closed_passage_are_measured_exactly_without_noise() -> None:
    for door in (0, 1):
        (obs,) = sensor().observe(world(door), "r0", 0)
        assert (obs.entity, obs.value, obs.observed_tick) == ("door", door, 0)
        assert obs.sensor_model == "binary-v1" and obs.observer_pose == (1.0, 2.0, 0.0)


def test_occluded_or_out_of_view_passage_produces_no_observation() -> None:
    assert sensor().observe(world(inner_wall=True), "r0", 0) == ()
    assert sensor().observe(world(), "r1", 0) == ()  # r1 faces away from the door
    far = world(starts=(RobotStart("r0", (0.5, 2.0, 0.0)),))  # 4.5 m: in range
    assert len(sensor().observe(far, "r0", 0)) == 1
    config = SensorConfig(range_m=4.49, fov_rad=math.pi, slot_ticks=5, model_name="binary-v1")
    short = BinarySensor(config, IndependentNoise(0, 0, KeyedRandom(1)), "ep-1")
    assert short.observe(far, "r0", 0) == ()


def test_fov_boundary_counts_as_visible() -> None:
    pose = (2.0, 0.5, 0.0)
    bearing = math.atan2(DOOR_CENTER[1] - pose[1], DOOR_CENTER[0] - pose[0])
    w = world(starts=(RobotStart("r0", pose),))
    assert len(sensor(fov=2 * bearing + 1e-12).observe(w, "r0", 0)) == 1
    assert sensor(fov=2 * bearing - 1e-9).observe(w, "r0", 0) == ()


def test_same_slot_is_the_same_measurement() -> None:
    s, w = sensor(fpr=0.3, fnr=0.3), world()
    first = s.observe(w, "r0", 0)
    for _ in range(5):
        sensor(fpr=0.3, fnr=0.3).observe(w, "r1", 0)  # other robots and sensors draw freely
    assert s.observe(w, "r0", 0) == first
    for tick in range(1, 6):
        w.advance(tick, {})
    later = s.observe(w, "r0", 1)
    assert later[0].evidence_id != first[0].evidence_id
    assert later[0].evidence_id.origin == "r0" and later[0].evidence_id.episode == "ep-1"


def test_slot_must_match_the_physical_tick() -> None:
    w = world()
    w.advance(1, {})
    with pytest.raises(SensorModelError):
        sensor().observe(w, "r0", 0)
    with pytest.raises(SensorModelError):
        sensor().observe(w, "r0", -1)


def test_zero_and_full_error_rates_are_deterministic() -> None:
    for door in (0, 1):
        for slot_seed in range(20):
            assert sensor(0.0, 0.0, seed=slot_seed).observe(world(door), "r0", 0)[0].value == door
            assert (
                sensor(1.0, 1.0, seed=slot_seed).observe(world(door), "r0", 0)[0].value == 1 - door
            )


def test_error_rates_apply_per_truth_state() -> None:
    # Fixed keys (seeds 0..1999), so these 5-sigma checks cannot flake.
    n = 2000
    for door, fpr, fnr, expected in ((1, 0.0, 0.2, 0.2), (0, 0.1, 0.0, 0.1)):
        flips = sum(
            sensor(fpr, fnr, seed=k).observe(world(door), "r0", 0)[0].value != door
            for k in range(n)
        )
        assert abs(flips / n - expected) < 5 * math.sqrt(expected * (1 - expected) / n)


@pytest.mark.parametrize(
    "build",
    [
        lambda: SensorConfig(0.0, 1.0, 5, "m"),
        lambda: SensorConfig(5.0, 7.0, 5, "m"),
        lambda: SensorConfig(5.0, 1.0, 0, "m"),
        lambda: SensorConfig(5.0, 1.0, 5, ""),
        lambda: IndependentNoise(-0.1, 0.0, KeyedRandom(1)),
        lambda: IndependentNoise(0.0, math.nan, KeyedRandom(1)),
    ],
)
def test_invalid_sensor_models_rejected_before_use(build: object) -> None:
    with pytest.raises(SensorModelError):
        build()  # type: ignore[operator]
