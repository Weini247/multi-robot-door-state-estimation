"""Local agent with a private history and an immutable view for its policy (E05-T01).

An agent holds only what it observed itself or received: its own registered
pose and collision feedback, its own passage readings, delivered evidence,
delivered reservations, its local free/blocked map and its remaining send
budget. It owns one FixedLagBelief. It has no reference to the world, the
channel or other agents; the runner hands it plain values. A policy gets a
frozen AgentView whose containers are tuples, so neither the policy nor
another agent can mutate an agent's state through it.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from argus.core.types import (
    AgentAction,
    AgentView,
    BeliefState,
    EvidenceId,
    GridIndex,
    Observation,
    Pose2,
    PositionReport,
    Reservation,
    Sighting,
)


class BeliefModel(Protocol):
    """What an agent needs from its belief: FixedLagBelief or a control variant."""

    def ingest(self, observation: Observation, received_tick: int) -> str: ...
    def advance(self, tick: int) -> None: ...
    def snapshot(self) -> BeliefState: ...
    def joint(self, entity: str) -> tuple[float, float, float, float]: ...
    def probability_open(self, entity: str) -> float: ...


class Policy(Protocol):
    def act(self, view: AgentView, random_key: str) -> AgentAction: ...


class LocalAgent:
    def __init__(
        self,
        robot_id: str,
        belief: BeliefModel,
        pose: Pose2,
        map_resolution_m: float,
    ) -> None:
        self._id = robot_id
        self._belief = belief
        self._pose = pose
        self._resolution = map_resolution_m
        self._free: set[GridIndex] = set()
        self._blocked: set[GridIndex] = set()
        self._inbox: list[Observation] = []
        self._own_new: list[Observation] = []
        self._evidence: dict[EvidenceId, Observation] = {}
        self._reservations: dict[str, Reservation] = {}
        self._sightings: dict[str, Sighting] = {}
        self._positions: list[PositionReport] = []
        self._budget = 0.0
        self._collided = False

    @property
    def robot_id(self) -> str:
        return self._id

    def set_pose(self, pose: Pose2, collided: bool) -> None:
        self._pose = pose
        self._collided = collided

    def set_budget(self, remaining_bytes: float) -> None:
        self._budget = remaining_bytes

    def update_map(self, free: Iterable[GridIndex], blocked: Iterable[GridIndex]) -> None:
        free_now, blocked_now = set(free), set(blocked)
        self._free |= free_now
        self._free -= blocked_now
        self._blocked |= blocked_now
        self._blocked -= free_now  # a reopened passage is seen free again

    def _take(self, observation: Observation, tick: int) -> str:
        status = self._belief.ingest(observation, tick)
        if status == "accepted":
            self._evidence[observation.evidence_id] = observation
            known = self._sightings.get(observation.entity)
            if known is None or known.observed_tick <= observation.observed_tick:
                self._sightings[observation.entity] = Sighting(
                    observation.entity,
                    observation.observer_pose,
                    observation.observed_tick,
                    observation.evidence_id.origin,
                )
        return status

    def observe(self, observations: Iterable[Observation], tick: int) -> list[str]:
        """Own sensor readings of this tick."""
        statuses = []
        for obs in observations:
            if obs.evidence_id.origin != self._id:
                raise ValueError("own observations must originate from this agent")
            status = self._take(obs, tick)
            statuses.append(status)
            if status == "accepted":
                self._own_new.append(obs)
        return statuses

    def receive(self, observations: Iterable[Observation], tick: int) -> list[str]:
        """Evidence delivered by the channel at this tick."""
        statuses = []
        for obs in observations:
            status = self._take(obs, tick)
            statuses.append(status)
            if status == "accepted":
                self._inbox.append(obs)
        return statuses

    def receive_reservation(self, reservation: Reservation, tick: int) -> None:
        if reservation.robot_id == self._id:
            raise ValueError("an agent does not receive its own reservation")
        known = self._reservations.get(reservation.robot_id)
        if known is None or known.expires_tick <= reservation.expires_tick:
            self._reservations[reservation.robot_id] = reservation

    def receive_position(self, report: PositionReport) -> None:
        """A teammate's position delivered by the channel; kept until the next decision."""
        if report.robot_id == self._id:
            raise ValueError("an agent does not receive its own position")
        self._positions.append(report)

    def knows(self, evidence_id: EvidenceId) -> Observation | None:
        """The payload of evidence this agent holds (needed to transmit it)."""
        return self._evidence.get(evidence_id)

    def probability_open(self, entity: str, tick: int) -> float:
        """Decision probability; an entity never seen keeps the nominal prior."""
        self._belief.advance(tick)
        return self._belief.probability_open(entity)

    def view(self, tick: int) -> AgentView:
        self._belief.advance(tick)
        active = tuple(
            sorted(
                (r for r in self._reservations.values() if r.expires_tick > tick),
                key=lambda r: r.robot_id,
            )
        )
        return AgentView(
            robot_id=self._id,
            tick=tick,
            pose=self._pose,
            belief=self._belief.snapshot(),
            map_resolution_m=self._resolution,
            free_map=tuple(sorted(self._free)),
            blocked_map=tuple(sorted(self._blocked)),
            inbox=tuple(self._inbox),
            reservations=active,
            remaining_bytes=self._budget,
            known_evidence=tuple(sorted(self._evidence)),
            last_collision=self._collided,
            sightings=tuple(self._sightings[e] for e in sorted(self._sightings)),
            new_own_evidence=tuple(self._own_new),
            position_reports=tuple(self._positions),
            joint_beliefs=tuple(
                (entity, self._belief.joint(entity))
                for entity, _ in self._belief.snapshot().passage_probabilities
            ),
        )

    def decide(self, tick: int, policy: Policy, random_key: str) -> AgentAction:
        action = policy.act(self.view(tick), random_key)
        for send in action.sends:
            if send.evidence_id not in self._evidence:
                raise ValueError(f"{self._id} cannot send unknown evidence {send.evidence_id}")
            if send.recipient == self._id:
                raise ValueError("an agent cannot send to itself")
        self._inbox.clear()
        self._own_new.clear()
        self._positions.clear()
        return action
