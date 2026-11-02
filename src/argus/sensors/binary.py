"""Visible binary passage measurements with independent errors (M2, E03-T01).

A robot measures only passages that pass the range, field-of-view and
line-of-sight test against the current obstacles; invisible passages produce
no observation at all (not a negative reading). Each reading is

    y = z with probability 1 - fnr (z = 1) or 1 - fpr (z = 0), else the flip.

Measurements happen in physical sample slots every `slot_ticks` ticks. A
reading is a pure function of (sensor seed, robot, passage, slot), and its
EvidenceId is derived from the slot, so querying the same slot twice returns
the same observations: no second independent measurement exists.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from argus.core.events import KeyedRandom
from argus.core.types import EvidenceId, Observation
from argus.world.geometry import Point, visible_passages
from argus.world.scenario import WorldBackend2D


class SensorModelError(ValueError):
    """Invalid sensor parameters or a measurement outside a physical slot."""


class NoiseModel(Protocol):
    """Turns the true passage state into one reading; location is the passage centre."""

    def reading(
        self, robot_id: str, passage_id: str, slot: int, tick: int, truth: int, location: Point
    ) -> int: ...


@dataclass(frozen=True, slots=True)
class IndependentNoise:
    """Independent flips: P(y=1|z=0) = fpr, P(y=0|z=1) = fnr."""

    fpr: float
    fnr: float
    rng: KeyedRandom

    def __post_init__(self) -> None:
        for name in ("fpr", "fnr"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise SensorModelError(f"{name} must be a probability, got {value!r}")

    def reading(
        self, robot_id: str, passage_id: str, slot: int, tick: int, truth: int, location: Point
    ) -> int:
        flip = self.fnr if truth == 1 else self.fpr
        flipped = self.rng.bernoulli(flip, "sensor.private", robot_id, passage_id, slot)
        return truth ^ flipped


@dataclass(frozen=True, slots=True)
class SensorConfig:
    range_m: float
    fov_rad: float
    slot_ticks: int
    model_name: str

    def __post_init__(self) -> None:
        if not (math.isfinite(self.range_m) and self.range_m > 0):
            raise SensorModelError("range_m must be finite and > 0")
        if not (math.isfinite(self.fov_rad) and 0 < self.fov_rad <= 2 * math.pi):
            raise SensorModelError("fov_rad must be in (0, 2 pi]")
        if type(self.slot_ticks) is not int or self.slot_ticks < 1:
            raise SensorModelError("slot_ticks must be an int >= 1")
        if not self.model_name:
            raise SensorModelError("model_name must be a nonempty public identifier")


class BinarySensor:
    """Simulator-side sensor: reads world truth, emits only public observations."""

    def __init__(self, config: SensorConfig, noise: NoiseModel, episode_id: str) -> None:
        if not episode_id:
            raise SensorModelError("episode_id must be nonempty")
        self._config = config
        self._noise = noise
        self._episode = episode_id

    @property
    def config(self) -> SensorConfig:
        return self._config

    def observe(self, world: WorldBackend2D, robot_id: str, slot: int) -> tuple[Observation, ...]:
        """Observations of robot_id in physical sample slot `slot`.

        The slot is tied to time: world.tick must equal slot * slot_ticks.
        """
        if type(slot) is not int or slot < 0:
            raise SensorModelError(f"slot must be an int >= 0, got {slot!r}")
        if world.tick != slot * self._config.slot_ticks:
            raise SensorModelError(
                f"slot {slot} is at tick {slot * self._config.slot_ticks}, world is at {world.tick}"
            )
        pose = world.poses()[robot_id]
        states = world.passage_states()
        seen = visible_passages(
            world.spec.geometry, pose, self._config.range_m, self._config.fov_rad, states
        )
        order = sorted(p.passage_id for p in world.spec.geometry.passages)
        observations = []
        for passage_id in sorted(seen):
            location = world.spec.geometry.passage(passage_id).rect.center
            value = self._noise.reading(
                robot_id, passage_id, slot, world.tick, states[passage_id], location
            )
            sequence = slot * len(order) + order.index(passage_id)
            observations.append(
                Observation(
                    EvidenceId(self._episode, robot_id, sequence),
                    world.tick,
                    passage_id,
                    value,
                    pose,
                    self._config.model_name,
                )
            )
        return tuple(observations)
