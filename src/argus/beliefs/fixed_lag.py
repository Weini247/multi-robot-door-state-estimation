"""Identity-deduplicated fixed-lag belief with out-of-order reprocessing (M3, E04-T04).

For every entity the filter keeps a checkpoint: the joint (z, g) belief at the
floor tick, immediately before all events at the floor, floor = max(0, now - H).
Accepted measurements with observed_tick >= floor are kept as immutable
payloads indexed by EvidenceId. The posterior is recomputed from the
checkpoint by applying them in (observed_tick, EvidenceId) order and
predicting to now, so any arrival order of the same accepted set gives the
same posterior.

When the floor advances, measurements below it are folded into the
checkpoint and their payloads and ids are released. That is safe because a
measurement with observed_tick < floor is always rejected as too_old, so an
evicted id can never be counted twice. There is no LRU eviction inside the
window: when K payloads are held, new evidence is rejected (capacity_rejected)
instead of forgetting accepted evidence.

Entities may be unknown at the start (the agent does not know the building):
with a default_model they are added on first contact, starting from the
nominal prior at the episode start, so an entity never observed reports its
prior. Nominal model per entity: one passage CTMC z and one bias CTMC g shared by all
measurements of that entity (exact under that model; approximate when the
true error groups differ, e.g. cells spanning several passages).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from argus.beliefs.exact_micro import Joint, joint_prior
from argus.beliefs.joint_filter import JointFilter
from argus.core.types import BeliefState, EvidenceConflict, EvidenceId, Observation
from argus.dynamics.binary_ctmc import BinaryCTMC

IngestStatus = Literal["accepted", "duplicate", "too_old", "capacity_rejected"]


@dataclass(frozen=True, slots=True)
class NominalPassageModel:
    """The receiver's nominal model of one entity; never the generating truth."""

    z_model: BinaryCTMC
    g_model: BinaryCTMC
    private_error: float
    prior_open: float
    bias_probability: float


class FixedLagBelief:
    def __init__(
        self,
        episode: str,
        models: Mapping[str, NominalPassageModel],
        tick_seconds: float,
        window_ticks: int,
        capacity: int,
        start_tick: int = 0,
        default_model: NominalPassageModel | None = None,
    ) -> None:
        if type(window_ticks) is not int or window_ticks < 0:
            raise ValueError("window_ticks must be an int >= 0")
        if type(capacity) is not int or capacity < 1:
            raise ValueError("capacity must be an int >= 1")
        if not tick_seconds > 0:
            raise ValueError("tick_seconds must be > 0")
        self._episode = episode
        self._models = dict(models)
        self._tick_seconds = tick_seconds
        self._window = window_ticks
        self._capacity = capacity
        self._now = start_tick
        self._floor = max(0, start_tick - window_ticks)
        self._default = default_model
        self._checkpoint: dict[str, Joint] = {}
        for entity in list(self._models):
            self._add(entity)
        self._payloads: dict[EvidenceId, Observation] = {}

    def _add(self, entity: str) -> None:
        model = self._models[entity]
        filt = self._filter(entity, joint_prior(model.prior_open, model.bias_probability))
        filt.predict(self._floor * self._tick_seconds)  # the prior holds at episode tick 0
        self._checkpoint[entity] = filt.joint

    def _known(self, entity: str) -> bool:
        """True if the entity has a model; entities first seen later use default_model."""
        if entity in self._models:
            return True
        if self._default is None:
            return False
        self._models[entity] = self._default
        self._add(entity)
        return True

    @property
    def now(self) -> int:
        return self._now

    @property
    def floor(self) -> int:
        return self._floor

    @property
    def stored(self) -> int:
        """Number of payloads (and ids) currently held; never exceeds capacity."""
        return len(self._payloads)

    def _filter(self, entity: str, joint: Joint) -> JointFilter:
        m = self._models[entity]
        return JointFilter(m.z_model, m.g_model, m.private_error, joint)

    def _replay(self, entity: str, until: int, items: list[Observation]) -> Joint:
        filt = self._filter(entity, self._checkpoint[entity])
        current = self._floor
        for obs in sorted(items, key=lambda o: (o.observed_tick, o.evidence_id)):
            filt.predict((obs.observed_tick - current) * self._tick_seconds)
            current = obs.observed_tick
            filt.correct(obs.value)
        filt.predict((until - current) * self._tick_seconds)
        return filt.joint

    def advance(self, tick: int) -> None:
        """Move to `tick`; fold evidence that falls below the new floor into the checkpoint."""
        if type(tick) is not int or tick < self._now:
            raise ValueError(f"time cannot move backwards from {self._now} to {tick!r}")
        self._now = tick
        new_floor = max(0, tick - self._window)
        if new_floor <= self._floor:
            return
        folding = [o for o in self._payloads.values() if o.observed_tick < new_floor]
        for entity in self._models:
            items = [o for o in folding if o.entity == entity]
            self._checkpoint[entity] = self._replay(entity, new_floor, items)
        for obs in folding:
            del self._payloads[obs.evidence_id]
        self._floor = new_floor

    def ingest(self, observation: Observation, received_tick: int) -> IngestStatus:
        if observation.evidence_id.episode != self._episode:
            raise ValueError("observation belongs to another episode")
        if not self._known(observation.entity):
            raise ValueError(f"no nominal model for entity {observation.entity!r}")
        if received_tick < observation.observed_tick:
            raise ValueError("received before it was observed")
        self.advance(received_tick)
        if observation.observed_tick < self._floor:
            return "too_old"
        known = self._payloads.get(observation.evidence_id)
        if known is not None:
            if known != observation:
                raise EvidenceConflict(f"{observation.evidence_id} arrived with another payload")
            return "duplicate"
        if len(self._payloads) >= self._capacity:
            return "capacity_rejected"
        self._payloads[observation.evidence_id] = observation
        return "accepted"

    def joint(self, entity: str) -> Joint:
        if not self._known(entity):
            raise ValueError(f"no nominal model for entity {entity!r}")
        items = [o for o in self._payloads.values() if o.entity == entity]
        return self._replay(entity, self._now, items)

    def probability_open(self, entity: str) -> float:
        joint = self.joint(entity)
        return joint[2] + joint[3]

    def snapshot(self) -> BeliefState:
        return BeliefState(
            self._now,
            tuple((entity, self.probability_open(entity)) for entity in sorted(self._models)),
        )
