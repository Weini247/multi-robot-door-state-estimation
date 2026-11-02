"""Episode snapshots and canonical event logs (E02-T04).

EventLog keeps a hash chain over processed records:

    h_0 = sha256("argus-eventlog-v1"),   h_n = sha256(h_{n-1} || canonical_json(record_n))

A record is the processed event plus its outcome. Equal hashes mean the same
records in the same order. replay_events recomputes the chain and checks the
event order from the records alone; it never calls a policy (playback, not a
counterfactual rerun).

A snapshot stores everything needed to continue an episode bit-identically in
the same lock: absolute tick, the last processed (tick, phase, id) key, used
event ids, pending events, robot poses, world seed, RNG schema, scenario hash
and the log hash. Passage states are not stored; they are a pure function of
seed and tick. Files carry a SHA-256 over their content and are written via a
temporary file and an atomic rename. Wrong versions, schemas, scenarios or
corrupted bytes are rejected.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from argus.core.events import (
    RNG_SCHEMA,
    EventKey,
    JsonValue,
    Phase,
    SimulationClock,
    SimulationEvent,
    canonical_json,
)
from argus.world.geometry import Pose
from argus.world.scenario import ScenarioSpec, WorldBackend2D

SNAPSHOT_SCHEMA = 1
LOG_SCHEMA = "argus-eventlog-v1"


class SnapshotError(ValueError):
    """Snapshot cannot be used: corrupted, other version, other scenario or RNG schema."""


def _genesis() -> str:
    return hashlib.sha256(LOG_SCHEMA.encode("utf-8")).hexdigest()


def chain(previous: str, record: JsonValue) -> str:
    return hashlib.sha256((previous + canonical_json(record)).encode("utf-8")).hexdigest()


class EventLog:
    """Append-only processed-event records with a running canonical hash."""

    def __init__(self, head: str | None = None, length: int = 0) -> None:
        self._head = head or _genesis()
        self._length = length
        self._records: list[dict[str, JsonValue]] = []

    @property
    def head(self) -> str:
        return self._head

    def __len__(self) -> int:
        return self._length

    @property
    def records(self) -> tuple[dict[str, JsonValue], ...]:
        """Records appended to this log object (not those before a resumed head)."""
        return tuple(self._records)

    def append(self, event: SimulationEvent, outcome: JsonValue) -> None:
        record: dict[str, JsonValue] = {"event": event.to_json(), "outcome": outcome}
        self._head = chain(self._head, record)
        self._length += 1
        self._records.append(record)


def replay_events(records: list[dict[str, Any]], head: str | None = None) -> str:
    """Recompute the log hash from records and check their event order; no policy calls."""
    current = head or _genesis()
    last: EventKey | None = None
    for record in records:
        event = SimulationEvent.from_json(record["event"])
        if last is not None and event.key <= last:
            raise SnapshotError(f"event order violated at {event.key}")
        last = event.key
        current = chain(current, record)
    return current


@dataclass(frozen=True, slots=True)
class EpisodeSnapshot:
    scenario_hash: str
    world_seed: int
    world_tick: int
    last_key: EventKey | None
    used_ids: tuple[str, ...]
    pending: tuple[SimulationEvent, ...]
    poses: tuple[tuple[str, Pose], ...]
    log_head: str
    log_length: int
    rng_schema: str = RNG_SCHEMA
    schema_version: int = SNAPSHOT_SCHEMA

    def to_json(self) -> dict[str, JsonValue]:
        return {
            "schema_version": self.schema_version,
            "rng_schema": self.rng_schema,
            "scenario_hash": self.scenario_hash,
            "world_seed": self.world_seed,
            "world_tick": self.world_tick,
            "last_key": None
            if self.last_key is None
            else [self.last_key[0], Phase(self.last_key[1]).name, self.last_key[2]],
            "used_ids": list(self.used_ids),
            "pending": [event.to_json() for event in self.pending],
            "poses": [[rid, list(pose)] for rid, pose in self.poses],
            "log_head": self.log_head,
            "log_length": self.log_length,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> EpisodeSnapshot:
        key = data["last_key"]
        return cls(
            scenario_hash=data["scenario_hash"],
            world_seed=data["world_seed"],
            world_tick=data["world_tick"],
            last_key=None if key is None else (key[0], int(Phase[key[1]]), key[2]),
            used_ids=tuple(data["used_ids"]),
            pending=tuple(SimulationEvent.from_json(e) for e in data["pending"]),
            poses=tuple((rid, (p[0], p[1], p[2])) for rid, p in data["poses"]),
            log_head=data["log_head"],
            log_length=data["log_length"],
            rng_schema=data["rng_schema"],
            schema_version=data["schema_version"],
        )


def capture(clock: SimulationClock, world: WorldBackend2D, log: EventLog) -> EpisodeSnapshot:
    return EpisodeSnapshot(
        scenario_hash=world.spec.scenario_hash,
        world_seed=world.world_seed,
        world_tick=world.tick,
        last_key=clock.last_key,
        used_ids=tuple(sorted(clock.used_ids())),
        pending=clock.pending(),
        poses=tuple(sorted(world.poses().items())),
        log_head=log.head,
        log_length=len(log),
    )


def resume(
    snapshot: EpisodeSnapshot, spec: ScenarioSpec
) -> tuple[SimulationClock, WorldBackend2D, EventLog]:
    """Rebuild clock, world and log head; the spec must be the snapshot's scenario."""
    if spec.scenario_hash != snapshot.scenario_hash:
        raise SnapshotError("snapshot belongs to another scenario")
    clock = SimulationClock.restore(
        snapshot.last_key, frozenset(snapshot.used_ids), snapshot.pending
    )
    world = WorldBackend2D(spec, snapshot.world_seed)
    world.restore(snapshot.world_tick, dict(snapshot.poses))
    return clock, world, EventLog(snapshot.log_head, snapshot.log_length)


def save_snapshot(path: Path, snapshot: EpisodeSnapshot) -> None:
    body = snapshot.to_json()
    document = {
        "snapshot": body,
        "sha256": hashlib.sha256(canonical_json(body).encode()).hexdigest(),
    }
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes((canonical_json(document) + "\n").encode("utf-8"))
    os.replace(temporary, path)


def load_snapshot(path: Path, expected_scenario_hash: str | None = None) -> EpisodeSnapshot:
    try:
        document = json.loads(path.read_bytes().decode("utf-8"))
        body, digest = document["snapshot"], document["sha256"]
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise SnapshotError(f"unreadable snapshot: {error}") from error
    if hashlib.sha256(canonical_json(body).encode()).hexdigest() != digest:
        raise SnapshotError("snapshot integrity hash mismatch")
    if body.get("schema_version") != SNAPSHOT_SCHEMA:
        raise SnapshotError(f"unsupported snapshot schema {body.get('schema_version')!r}")
    if body.get("rng_schema") != RNG_SCHEMA:
        raise SnapshotError(f"snapshot uses RNG schema {body.get('rng_schema')!r}")
    if expected_scenario_hash is not None and body["scenario_hash"] != expected_scenario_hash:
        raise SnapshotError("snapshot belongs to another scenario")
    return EpisodeSnapshot.from_json(body)
