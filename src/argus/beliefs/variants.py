"""Control belief models for baselines (E07-T02).

IndependentLogOddsBelief treats every reading as an independent measurement
with the marginal error e (no shared bias). Two switches make the classic
controls:

* naive fusion: dedupe=False, no TTL, no dynamics — every received copy counts,
  evidence never ages (double counting and overconfidence by design);
* TTL + IDs: dedupe=True and readings older than ttl_ticks are ignored, still
  without dynamics.

It exposes the same interface as FixedLagBelief (ingest/advance/snapshot/joint/
probability_open) so the agent can hold either. joint() reports (z, g) with
g = 0 because this model has no bias variable.
"""

from __future__ import annotations

import math

from argus.core.types import BeliefState, EvidenceConflict, EvidenceId, Observation


class IndependentLogOddsBelief:
    def __init__(
        self,
        episode: str,
        prior_open: float,
        marginal_error: float,
        dedupe: bool,
        ttl_ticks: int | None,
        start_tick: int = 0,
    ) -> None:
        if not 0.0 < prior_open < 1.0:
            raise ValueError("prior_open must be in (0, 1)")
        if not 0.0 < marginal_error < 0.5:
            raise ValueError("marginal_error must be in (0, 0.5)")
        if ttl_ticks is not None and ttl_ticks < 0:
            raise ValueError("ttl_ticks must be >= 0")
        self._episode = episode
        self._prior = prior_open
        self._llr = math.log((1 - marginal_error) / marginal_error)
        self._dedupe = dedupe
        self._ttl = ttl_ticks
        self._now = start_tick
        self._items: list[Observation] = []
        self._ids: dict[EvidenceId, Observation] = {}

    @property
    def now(self) -> int:
        return self._now

    @property
    def stored(self) -> int:
        return len(self._items)

    def advance(self, tick: int) -> None:
        if tick < self._now:
            raise ValueError(f"time cannot move backwards from {self._now} to {tick}")
        self._now = tick
        if self._ttl is not None:
            floor = tick - self._ttl
            self._items = [o for o in self._items if o.observed_tick >= floor]
            self._ids = {o.evidence_id: o for o in self._items}

    def ingest(self, observation: Observation, received_tick: int) -> str:
        if observation.evidence_id.episode != self._episode:
            raise ValueError("observation belongs to another episode")
        if received_tick < observation.observed_tick:
            raise ValueError("received before it was observed")
        self.advance(received_tick)
        if self._ttl is not None and observation.observed_tick < received_tick - self._ttl:
            return "too_old"
        known = self._ids.get(observation.evidence_id)
        if known is not None and self._dedupe:
            if known != observation:
                raise EvidenceConflict(f"{observation.evidence_id} arrived with another payload")
            return "duplicate"
        self._items.append(observation)
        self._ids[observation.evidence_id] = observation
        return "accepted"

    def probability_open(self, entity: str) -> float:
        log_odds = math.log(self._prior / (1 - self._prior))
        for obs in self._items:
            if obs.entity == entity:
                log_odds += self._llr if obs.value == 1 else -self._llr
        if log_odds < -700.0:
            # exp(-log_odds) would overflow; 1 / (1 + e^-x) equals e^x to double precision.
            # Only this branch differs from the plain formula (fix 2026-11-02).
            return math.exp(log_odds)
        return 1 / (1 + math.exp(-log_odds))

    def joint(self, entity: str) -> tuple[float, float, float, float]:
        p = self.probability_open(entity)
        return (1 - p, 0.0, p, 0.0)

    def snapshot(self) -> BeliefState:
        entities = sorted({o.entity for o in self._items})
        return BeliefState(self._now, tuple((e, self.probability_open(e)) for e in entities))
