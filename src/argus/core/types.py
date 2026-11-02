"""Unprivileged value types shared by sensors, beliefs, communication and agents.

Nothing here may carry simulator truth: no true passage states, no latent
error groups or bias values, no visibility scores of hidden objects. The
allowed public fields are pinned by tests/integration/test_sensor_boundary.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from argus.core.events import JsonValue

Pose2 = tuple[float, float, float]  # metres, metres, radians


@dataclass(frozen=True, slots=True, order=True)
class EvidenceId:
    """Transport identity of one physical measurement; never re-created by forwarding."""

    episode: str
    origin: str
    sequence: int

    def to_json(self) -> list[JsonValue]:
        return [self.episode, self.origin, self.sequence]


@dataclass(frozen=True, slots=True)
class Observation:
    """One binary reading of a visible, registered passage."""

    evidence_id: EvidenceId
    observed_tick: int
    entity: str
    value: int
    observer_pose: Pose2
    sensor_model: str  # public nominal model name, never the true error parameters

    def __post_init__(self) -> None:
        if self.value not in (0, 1):
            raise ValueError("observation value must be 0 or 1")
        if type(self.observed_tick) is not int or self.observed_tick < 0:
            raise ValueError("observed_tick must be an int >= 0")

    def to_json(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id.to_json(),
            "observed_tick": self.observed_tick,
            "entity": self.entity,
            "value": self.value,
            "observer_pose": list(self.observer_pose),
            "sensor_model": self.sensor_model,
        }


class EvidenceConflict(ValueError):
    """The same EvidenceId arrived with a different payload; never overwrite evidence."""


@dataclass(frozen=True, slots=True)
class BeliefState:
    """Immutable public belief snapshot: P(open) per entity at a tick."""

    tick: int
    passage_probabilities: tuple[tuple[str, float], ...]


GridIndex = tuple[int, int]  # local map index: floor(x / resolution), floor(y / resolution)


@dataclass(frozen=True, slots=True)
class Reservation:
    """A partner's announced exploration target, known only from a delivered message."""

    robot_id: str
    target: GridIndex
    expires_tick: int


@dataclass(frozen=True, slots=True)
class Sighting:
    """Latest known reading of an entity: where it was observed from, when and by whom."""

    entity: str
    observer_pose: Pose2
    observed_tick: int
    origin: str


@dataclass(frozen=True, slots=True)
class PositionReport:
    """A teammate's registered position as delivered in a position frame."""

    robot_id: str
    tick: int
    x: float
    y: float


@dataclass(frozen=True, slots=True)
class Send:
    """Request to transmit one known evidence item to one recipient."""

    recipient: str
    evidence_id: EvidenceId
    # The sender's own nominal estimate of the recipient's P(open) before this
    # message; logged against the true receiver belief by the runner, never shown back.
    assumed_receiver_open: float | None = None


@dataclass(frozen=True, slots=True)
class AgentAction:
    """A local decision; acts from the next tick on and lasts duration_ticks >= 1."""

    kind: str  # "move" or "wait"
    duration_ticks: int
    v: float = 0.0
    omega: float = 0.0
    sends: tuple[Send, ...] = ()
    reserve: GridIndex | None = None
    reserve_ttl_ticks: int = 0
    share_position: bool = False  # broadcast the own registered position (charged frames)

    def __post_init__(self) -> None:
        if self.kind not in ("move", "wait"):
            raise ValueError(f"unknown action kind {self.kind!r}")
        if type(self.duration_ticks) is not int or self.duration_ticks < 1:
            raise ValueError("duration_ticks must be an int >= 1")
        if self.kind == "wait" and (self.v != 0.0 or self.omega != 0.0):
            raise ValueError("a wait action has zero velocity")
        if self.reserve is not None and self.reserve_ttl_ticks < 1:
            raise ValueError("a reservation needs a positive TTL")


@dataclass(frozen=True, slots=True)
class AgentView:
    """Everything a policy may see. Built by the agent; contains no simulator handles."""

    robot_id: str
    tick: int
    pose: Pose2
    belief: BeliefState
    map_resolution_m: float
    free_map: tuple[GridIndex, ...]
    blocked_map: tuple[GridIndex, ...]
    inbox: tuple[Observation, ...]
    reservations: tuple[Reservation, ...]
    remaining_bytes: float
    known_evidence: tuple[EvidenceId, ...]
    last_collision: bool
    sightings: tuple[Sighting, ...] = ()
    new_own_evidence: tuple[Observation, ...] = ()
    position_reports: tuple[PositionReport, ...] = ()
    joint_beliefs: tuple[tuple[str, tuple[float, float, float, float]], ...] = ()
