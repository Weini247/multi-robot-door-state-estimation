"""Integer simulation time, totally ordered events and keyed randomness (E02-T01).

Time is a nonnegative integer tick; no float timestamps are compared. Within a
tick six phases run in a fixed order (architecture §5). Every event has the
total key (tick, phase, event_id) with a semantic, episode-unique event_id, so
the processing order never depends on insertion order. Once an event with key
K has been popped, only events with keys strictly greater than K may be
scheduled: nothing can be inserted into the past or at the current key.

KeyedRandom derives each uniform from a hash of (schema, seed, domain, keys),
never from a call counter. Additional draws in one domain (an extra sensor
reading, say) therefore cannot shift any value in another domain or under
other keys. It is a simulation RNG, not a cryptographic one.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

RNG_SCHEMA = "argus-rng-v1"

JsonValue = None | bool | int | float | str | list[Any] | dict[str, Any]


class Phase(IntEnum):
    WORLD = 0
    DELIVERY = 1
    SENSOR = 2
    BELIEF = 3
    POLICY = 4
    EVALUATE = 5


EventKey = tuple[int, int, str]


def canonical_json(value: JsonValue) -> str:
    """Canonical JSON: sorted keys, compact separators, UTF-8 text, no NaN/Inf."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


@dataclass(frozen=True, slots=True)
class SimulationEvent:
    """One scheduled event. payload must be JSON-compatible (checked on scheduling)."""

    tick: int
    phase: Phase
    event_id: str
    payload: JsonValue = None

    @property
    def key(self) -> EventKey:
        return (self.tick, int(self.phase), self.event_id)

    def to_json(self) -> dict[str, JsonValue]:
        return {
            "tick": self.tick,
            "phase": self.phase.name,
            "event_id": self.event_id,
            "payload": self.payload,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> SimulationEvent:
        return cls(data["tick"], Phase[data["phase"]], data["event_id"], data["payload"])


class SimulationClock:
    """Priority queue of events ordered by (tick, phase, event_id)."""

    def __init__(self) -> None:
        self._queue: list[tuple[EventKey, SimulationEvent]] = []
        self._ids: set[str] = set()
        self._last: EventKey | None = None

    @property
    def last_key(self) -> EventKey | None:
        """Key of the most recently popped event, or None before the first pop."""
        return self._last

    @property
    def tick(self) -> int:
        return self._last[0] if self._last else 0

    def __len__(self) -> int:
        return len(self._queue)

    def schedule(self, event: SimulationEvent) -> None:
        if type(event.tick) is not int or event.tick < 0:
            raise ValueError(f"tick must be a nonnegative int, got {event.tick!r}")
        if not isinstance(event.phase, Phase):
            raise ValueError(f"phase must be a Phase, got {event.phase!r}")
        if not event.event_id:
            raise ValueError("event_id must be a nonempty semantic identifier")
        if self._last is not None and event.key <= self._last:
            raise ValueError(f"event {event.key} is not after the current key {self._last}")
        if event.event_id in self._ids:
            raise ValueError(f"duplicate event id {event.event_id!r}")
        canonical_json(event.payload)  # raises for non-JSON or non-finite payloads
        self._ids.add(event.event_id)
        heapq.heappush(self._queue, (event.key, event))

    def peek(self) -> SimulationEvent | None:
        return self._queue[0][1] if self._queue else None

    def pop(self) -> SimulationEvent:
        if not self._queue:
            raise IndexError("empty event queue")
        key, event = heapq.heappop(self._queue)
        self._last = key
        return event

    def pending(self) -> tuple[SimulationEvent, ...]:
        """All scheduled events in processing order (for snapshots)."""
        return tuple(event for _, event in sorted(self._queue))

    def used_ids(self) -> frozenset[str]:
        return frozenset(self._ids)

    @classmethod
    def restore(
        cls,
        last_key: EventKey | None,
        used_ids: frozenset[str],
        pending: tuple[SimulationEvent, ...],
    ) -> SimulationClock:
        """Rebuild a clock from snapshot fields; validates the same invariants."""
        clock = cls()
        clock._last = last_key
        for event in pending:
            clock.schedule(event)
        clock._ids |= set(used_ids)
        return clock


class KeyedRandom:
    """Deterministic U[0,1) from hashed semantic keys; independent of call order."""

    __slots__ = ("_seed",)

    def __init__(self, seed: int) -> None:
        if type(seed) is not int or seed < 0:
            raise ValueError(f"seed must be a nonnegative int, got {seed!r}")
        self._seed = seed

    @property
    def seed(self) -> int:
        return self._seed

    def uniform(self, domain: str, *keys: str | int) -> float:
        """U[0,1) with 53 random bits for (domain, keys)."""
        if not domain:
            raise ValueError("domain must be nonempty")
        for key in keys:
            if type(key) not in (str, int):
                raise ValueError(f"keys must be str or int, got {key!r}")
        payload = canonical_json([RNG_SCHEMA, self._seed, domain, *keys]).encode("utf-8")
        bits = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") >> 11
        return bits / 2**53

    def bernoulli(self, probability: float, domain: str, *keys: str | int) -> int:
        """1 with the given probability, else 0, using uniform(domain, *keys) < probability."""
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError(f"probability must be in [0, 1], got {probability!r}")
        return int(self.uniform(domain, *keys) < probability)
