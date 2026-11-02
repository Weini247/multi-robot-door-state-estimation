"""Scenario specification, exogenous passage dynamics and the 2D backend (E02-T03).

Each passage has exactly one model: a CTMC sampled as its exact skeleton on the
tick grid, or an explicit script of (tick, state) changes. A passage can
therefore never receive a CTMC draw and a scripted change for the same tick.

The passage timeline depends only on (world seed, scenario id, passage, tick):

    z_0 = [U("world.passage.initial", scenario, passage) < p0]
    z_k = [U("world.passage", scenario, passage, k) < P(tick_seconds)[z_{k-1}][1]]

so robot actions, sensor draws or the order of queries cannot change it.

Tick k of WorldBackend2D.advance: motion over (k-1, k] is resolved with the
passage states z(k-1); then z(k) applies. A passage that closes onto a robot
records a closure_conflict; the robot is not moved (no teleport). It stays
blocked until the passage reopens. This closure rule is a stated stress
assumption of the pilot, not a physical claim.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from argus.core.events import KeyedRandom, canonical_json
from argus.dynamics.binary_ctmc import BinaryCTMC
from argus.world.geometry import (
    CONTACT_TOL,
    CollisionEvent,
    Geometry,
    Motion,
    Pose,
    Rect,
    resolve_motions,
)

GENERATOR_VERSION = "argus-scenario-v1"


class ScenarioError(ValueError):
    """Invalid scenario specification or start configuration."""


@dataclass(frozen=True, slots=True)
class CtmcPassage:
    """Passage state follows a binary CTMC; initial_open None means stationary start."""

    rate_01: float
    rate_10: float
    initial_open: float | None = None

    def model(self) -> BinaryCTMC:
        return BinaryCTMC(self.rate_01, self.rate_10)

    def initial_probability(self) -> float:
        if self.initial_open is not None:
            p = self.initial_open
            if not math.isfinite(p) or not 0.0 <= p <= 1.0:
                raise ScenarioError(f"initial_open must be a probability, got {p!r}")
            return p
        stationary = self.model().stationary
        if stationary is None:
            raise ScenarioError("both rates are 0: initial_open must be given")
        return stationary

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": "ctmc",
            "rate_01": self.rate_01,
            "rate_10": self.rate_10,
            "initial_open": self.initial_open,
        }


@dataclass(frozen=True, slots=True)
class ScriptedPassage:
    """Passage state given by an initial state and absolute (tick, new_state) changes."""

    initial_state: int
    changes: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        if self.initial_state not in (0, 1):
            raise ScenarioError("initial_state must be 0 or 1")
        ticks = [tick for tick, _ in self.changes]
        if any(type(t) is not int or t < 1 for t in ticks) or ticks != sorted(set(ticks)):
            raise ScenarioError("script ticks must be strictly increasing ints >= 1")
        if any(state not in (0, 1) for _, state in self.changes):
            raise ScenarioError("scripted states must be 0 or 1")

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": "script",
            "initial_state": self.initial_state,
            "changes": [list(change) for change in self.changes],
        }


PassageModel = CtmcPassage | ScriptedPassage


@dataclass(frozen=True, slots=True)
class RobotStart:
    robot_id: str
    pose: Pose


def _rect_json(rect: Rect) -> list[float]:
    return [rect.xmin, rect.ymin, rect.xmax, rect.ymax]


def geometry_json(geometry: Geometry) -> dict[str, Any]:
    return {
        "bounds": _rect_json(geometry.bounds),
        "walls": [_rect_json(w) for w in geometry.walls],
        "rooms": [[r.room_id, _rect_json(r.rect)] for r in geometry.rooms],
        "passages": [[p.passage_id, _rect_json(p.rect), list(p.rooms)] for p in geometry.passages],
    }


@dataclass(frozen=True, slots=True)
class ScenarioSpec:
    scenario_id: str
    geometry: Geometry
    tick_seconds: float
    horizon_ticks: int
    robot_radius: float
    v_max: float
    omega_max: float
    starts: tuple[RobotStart, ...]
    passages: tuple[tuple[str, PassageModel], ...]
    generator_version: str = GENERATOR_VERSION

    def __post_init__(self) -> None:
        if not self.scenario_id:
            raise ScenarioError("scenario_id must be nonempty")
        if self.generator_version != GENERATOR_VERSION:
            raise ScenarioError(f"unsupported generator version {self.generator_version!r}")
        for name in ("tick_seconds", "robot_radius", "v_max", "omega_max"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ScenarioError(f"{name} must be finite and > 0, got {value!r}")
        if type(self.horizon_ticks) is not int or self.horizon_ticks < 1:
            raise ScenarioError("horizon_ticks must be an int >= 1")
        ids = [pid for pid, _ in self.passages]
        expected = [p.passage_id for p in self.geometry.passages]
        if len(ids) != len(set(ids)) or set(ids) != set(expected):
            raise ScenarioError("every geometry passage needs exactly one model")
        for _, model in self.passages:
            if isinstance(model, CtmcPassage):
                model.model()
                model.initial_probability()
        self._check_starts()

    def _check_starts(self) -> None:
        robots = [s.robot_id for s in self.starts]
        if not robots or len(robots) != len(set(robots)):
            raise ScenarioError("robot ids must be nonempty and unique")
        r, bounds = self.robot_radius, self.geometry.bounds
        # Starts must be clear of walls and of every passage, whatever its initial state.
        blocked = self.geometry.walls + tuple(p.rect for p in self.geometry.passages)
        for start in self.starts:
            x, y, theta = start.pose
            if not all(math.isfinite(v) for v in (x, y, theta)):
                raise ScenarioError(f"{start.robot_id}: pose must be finite")
            if not (
                bounds.xmin + r <= x <= bounds.xmax - r and bounds.ymin + r <= y <= bounds.ymax - r
            ):
                raise ScenarioError(f"{start.robot_id}: start outside the bounds")
            if any(rect.distance_to((x, y)) < r - CONTACT_TOL for rect in blocked):
                raise ScenarioError(f"{start.robot_id}: start overlaps a wall or passage")
        for i, a in enumerate(self.starts):
            for b in self.starts[i + 1 :]:
                if math.dist(a.pose[:2], b.pose[:2]) < 2 * r - CONTACT_TOL:
                    raise ScenarioError(f"starts of {a.robot_id} and {b.robot_id} overlap")

    def model(self, passage_id: str) -> PassageModel:
        return dict(self.passages)[passage_id]

    def to_json(self) -> dict[str, Any]:
        return {
            "generator_version": self.generator_version,
            "scenario_id": self.scenario_id,
            "geometry": geometry_json(self.geometry),
            "tick_seconds": self.tick_seconds,
            "horizon_ticks": self.horizon_ticks,
            "robot_radius": self.robot_radius,
            "v_max": self.v_max,
            "omega_max": self.omega_max,
            "starts": [[s.robot_id, list(s.pose)] for s in self.starts],
            "passages": [[pid, model.to_json()] for pid, model in sorted(self.passages)],
        }

    @property
    def scenario_hash(self) -> str:
        return hashlib.sha256(canonical_json(self.to_json()).encode("utf-8")).hexdigest()


class ExogenousWorld:
    """Passage timelines from the world seed; a pure function of (seed, scenario, passage, tick)."""

    def __init__(self, spec: ScenarioSpec, world_seed: int) -> None:
        self._spec = spec
        self._rng = KeyedRandom(world_seed)
        self._timelines: dict[str, list[int]] = {}

    @property
    def world_seed(self) -> int:
        return self._rng.seed

    def state(self, passage_id: str, tick: int) -> int:
        if type(tick) is not int or tick < 0:
            raise ValueError(f"tick must be an int >= 0, got {tick!r}")
        timeline = self._timelines.get(passage_id)
        if timeline is None:
            timeline = self._timelines[passage_id] = [self._initial(passage_id)]
        while len(timeline) <= tick:
            timeline.append(self._next(passage_id, len(timeline), timeline[-1]))
        return timeline[tick]

    def states(self, tick: int) -> dict[str, int]:
        return {p.passage_id: self.state(p.passage_id, tick) for p in self._spec.geometry.passages}

    def _initial(self, passage_id: str) -> int:
        model = self._spec.model(passage_id)
        if isinstance(model, ScriptedPassage):
            return model.initial_state
        p0 = model.initial_probability()
        return self._rng.bernoulli(p0, "world.passage.initial", self._spec.scenario_id, passage_id)

    def _next(self, passage_id: str, tick: int, previous: int) -> int:
        model = self._spec.model(passage_id)
        if isinstance(model, ScriptedPassage):
            return dict(model.changes).get(tick, previous)
        p_open = model.model().transition(self._spec.tick_seconds)[previous][1]
        return self._rng.bernoulli(
            p_open, "world.passage", self._spec.scenario_id, passage_id, tick
        )


@dataclass(frozen=True, slots=True)
class ClosureConflict:
    passage_id: str
    robot_id: str


@dataclass(frozen=True, slots=True)
class WorldStep:
    tick: int
    collisions: tuple[CollisionEvent, ...]
    closure_conflicts: tuple[ClosureConflict, ...]
    passage_changes: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class WorldTruth:
    """Evaluator-only view of the true world state."""

    tick: int
    passages: tuple[tuple[str, int], ...]
    robots: tuple[tuple[str, Pose], ...]


class WorldBackend2D:
    """Simulator-side world. Only backend, sensors and evaluator may read its truth."""

    def __init__(self, spec: ScenarioSpec, world_seed: int) -> None:
        self._spec = spec
        self._exogenous = ExogenousWorld(spec, world_seed)
        self._tick = 0
        self._poses: dict[str, Pose] = {s.robot_id: s.pose for s in spec.starts}

    @property
    def spec(self) -> ScenarioSpec:
        return self._spec

    @property
    def tick(self) -> int:
        return self._tick

    @property
    def world_seed(self) -> int:
        return self._exogenous.world_seed

    def passage_states(self) -> dict[str, int]:
        return self._exogenous.states(self._tick)

    def advance(self, tick: int, actions: Mapping[str, tuple[float, float]]) -> WorldStep:
        """Apply (v, omega) per robot over (tick-1, tick], then the passage states of tick."""
        if tick != self._tick + 1:
            raise ValueError(f"expected tick {self._tick + 1}, got {tick}")
        unknown = set(actions) - set(self._poses)
        if unknown:
            raise ValueError(f"unknown robots {sorted(unknown)}")
        spec = self._spec
        motions = []
        for rid, pose in sorted(self._poses.items()):
            v, omega = actions.get(rid, (0.0, 0.0))
            if not (math.isfinite(v) and math.isfinite(omega)):
                raise ValueError(f"{rid}: velocities must be finite")
            if abs(v) > spec.v_max or abs(omega) > spec.omega_max:
                raise ValueError(f"{rid}: velocity exceeds the limits")
            motions.append(Motion(rid, pose, spec.robot_radius, v, omega))
        before = self.passage_states()
        obstacles = spec.geometry.obstacles(before)
        self._poses, collisions = resolve_motions(motions, spec.tick_seconds, obstacles)
        self._tick = tick
        after = self.passage_states()
        changes = tuple((pid, after[pid]) for pid in sorted(after) if after[pid] != before[pid])
        conflicts = []
        for pid, state in changes:
            if state == 0:
                rect = spec.geometry.passage(pid).rect
                for rid, pose in sorted(self._poses.items()):
                    if rect.distance_to(pose[:2]) < spec.robot_radius - CONTACT_TOL:
                        conflicts.append(ClosureConflict(pid, rid))
        return WorldStep(tick, collisions, tuple(conflicts), changes)

    def truth_snapshot(self) -> WorldTruth:
        return WorldTruth(
            self._tick,
            tuple(sorted(self.passage_states().items())),
            tuple(sorted(self._poses.items())),
        )

    def poses(self) -> dict[str, Pose]:
        return dict(self._poses)

    def restore(self, tick: int, poses: Mapping[str, Pose]) -> None:
        """Set time and poses from a snapshot; passage states follow from the seed."""
        if set(poses) != set(self._poses):
            raise ValueError("snapshot robots do not match the scenario")
        self._tick = tick
        self._poses = {rid: (float(p[0]), float(p[1]), float(p[2])) for rid, p in poses.items()}
