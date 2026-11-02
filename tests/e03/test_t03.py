"""ARGUS-E03-T03: hidden-world canary, observation order and marginal error rates."""

from __future__ import annotations

import json
import math

from argus.core.events import KeyedRandom
from argus.core.types import Observation
from argus.sensors.binary import BinarySensor, IndependentNoise, SensorConfig
from argus.sensors.correlated import BiasField, SharedBiasNoise, matched_private_error
from argus.world.geometry import Geometry, Passage, Rect, Room
from argus.world.scenario import RobotStart, ScenarioSpec, ScriptedPassage, WorldBackend2D

# Tolerances are fixed here, before any run, and not tuned afterwards.
MARGINAL_E = 0.14
REPLICATES = 4000
MARGINAL_TOL = 5 * math.sqrt(MARGINAL_E * (1 - MARGINAL_E) / REPLICATES)

CONFIG = SensorConfig(5.0, math.radians(120), 5, "binary-v1")


def geometry(order: tuple[str, str] = ("door", "back")) -> Geometry:
    passages = {
        "door": Passage("door", Rect(4.9, 1.4, 5.1, 2.6), ("west", "east")),
        # A second doorway in the west wall of the west room, behind the robot.
        "back": Passage("back", Rect(0.0, 1.4, 0.2, 2.6), ("west", "hall")),
    }
    return Geometry(
        bounds=Rect(-2.0, 0.0, 10.0, 4.0),
        walls=(Rect(4.9, 0.0, 5.1, 1.4), Rect(4.9, 2.6, 5.1, 4.0)),
        rooms=(
            Room("hall", Rect(-2.0, 0.0, 0.0, 4.0)),
            Room("west", Rect(0.2, 0.0, 4.9, 4.0)),
            Room("east", Rect(5.1, 0.0, 10.0, 4.0)),
        ),
        passages=tuple(passages[name] for name in order),
    )


def world(door: int, back: int, order: tuple[str, str] = ("door", "back")) -> WorldBackend2D:
    spec = ScenarioSpec(
        scenario_id="canary",
        geometry=geometry(order),
        tick_seconds=0.1,
        horizon_ticks=1000,
        robot_radius=0.2,
        v_max=0.5,
        omega_max=1.0,
        starts=(RobotStart("r0", (2.0, 2.0, 0.0)),),
        passages=(("door", ScriptedPassage(door)), ("back", ScriptedPassage(back))),
    )
    return WorldBackend2D(spec, 1)


def history(sensor: BinarySensor, w: WorldBackend2D, slots: int) -> list[Observation]:
    seen: list[Observation] = []
    for slot in range(slots):
        if slot:
            for tick in range((slot - 1) * 5 + 1, slot * 5 + 1):
                w.advance(tick, {})
        seen += sensor.observe(w, "r0", slot)
    return seen


def local_decision(observations: list[Observation]) -> list[str]:
    """Stand-in policy that may only use the local observation history."""
    actions, opened = [], 0
    for obs in observations:
        opened += 1 if obs.value else -1
        actions.append("go" if opened > 0 else "inspect")
    return actions


def test_hidden_passage_state_does_not_change_history_or_actions() -> None:
    sensor = BinarySensor(CONFIG, IndependentNoise(0.14, 0.14, KeyedRandom(4)), "ep")
    a = history(sensor, world(door=1, back=0), 40)
    b = history(sensor, world(door=1, back=1), 40)
    assert {o.entity for o in a} == {"door"}  # the back door is never visible
    assert [o.to_json() for o in a] == [o.to_json() for o in b]
    assert local_decision(a) == local_decision(b)
    # Not vacuous: a visible difference does change the history.
    c = history(sensor, world(door=0, back=0), 40)
    assert [o.value for o in c] != [o.value for o in a]


class BiasedExceptAt(BiasField):
    """Test field: g = 1 in every cell except one, which keeps g = 0."""

    def __init__(self, keep: tuple[float, float]) -> None:
        super().__init__(1.0, 0.0, 0.0, 0.1, KeyedRandom(7), "canary", initial_probability=0.0)
        object.__setattr__(self, "_keep", self.cell_of(keep))

    def state(self, cell: tuple[int, int], tick: int) -> int:
        return 0 if cell == self._keep else 1  # type: ignore[attr-defined]


def test_hidden_bias_cells_do_not_change_history() -> None:
    def sensor(field: BiasField) -> BinarySensor:
        return BinarySensor(CONFIG, SharedBiasNoise(field, 0.05, KeyedRandom(8)), "ep")

    unbiased = BiasField(1.0, 0.0, 0.0, 0.1, KeyedRandom(7), "canary", initial_probability=0.0)
    biased_elsewhere = BiasedExceptAt(keep=(5.0, 2.0))  # only the visible door's cell is clean
    a = history(sensor(unbiased), world(1, 0), 30)
    b = history(sensor(biased_elsewhere), world(1, 0), 30)
    assert [o.to_json() for o in a] == [o.to_json() for o in b]
    assert local_decision(a) == local_decision(b)
    # Not vacuous: biasing the door's own cell changes what the robot sees.
    door_biased = BiasField(1.0, 0.0, 0.0, 0.1, KeyedRandom(7), "canary", initial_probability=1.0)
    flipped = history(sensor(door_biased), world(1, 0), 30)
    assert [o.value for o in flipped] != [o.value for o in a]


def test_observation_order_and_ids_do_not_depend_on_geometry_order() -> None:
    sensor = BinarySensor(CONFIG, IndependentNoise(0.2, 0.2, KeyedRandom(3)), "ep")
    a = world(1, 1, ("door", "back"))
    b = world(1, 1, ("back", "door"))
    for w in (a, b):
        w.restore(0, {"r0": (2.5, 2.0, 0.0)})
    assert sensor.observe(a, "r0", 0) == sensor.observe(b, "r0", 0)


def test_serialized_observations_carry_only_public_fields() -> None:
    sensor = BinarySensor(CONFIG, IndependentNoise(0.1, 0.1, KeyedRandom(3)), "ep")
    text = json.dumps([o.to_json() for o in history(sensor, world(1, 0), 10)])
    for word in ("truth", "bias", "group", "latent", "cell", "back"):
        assert word not in text


def marginal_error_rate(make_sensor: object) -> float:
    errors = 0
    for replicate in range(REPLICATES):
        sensor = make_sensor(replicate)  # type: ignore[operator]
        (obs,) = sensor.observe(world(1, 0), "r0", 0)
        errors += obs.value != 1
    return errors / REPLICATES


def test_marginal_error_is_matched_across_independent_field_replicates() -> None:
    def independent(seed: int) -> BinarySensor:
        return BinarySensor(CONFIG, IndependentNoise(0.14, 0.14, KeyedRandom(seed)), "ep")

    def correlated(seed: int) -> BinarySensor:
        field = BiasField(4.0, 0.005, 0.045, 0.1, KeyedRandom(10_000 + seed), "canary")
        q = matched_private_error(MARGINAL_E, field.rho)
        return BinarySensor(CONFIG, SharedBiasNoise(field, q, KeyedRandom(seed)), "ep")

    # Each replicate is an independent field and noise draw, so n counts replicates.
    assert abs(marginal_error_rate(independent) - MARGINAL_E) < MARGINAL_TOL
    assert abs(marginal_error_rate(correlated) - MARGINAL_E) < MARGINAL_TOL
