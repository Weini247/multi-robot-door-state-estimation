"""Raw-evidence wire format v1 and provenance (M4, E04-T01).

A message carries exactly one raw Observation plus transport metadata. The
wire encoding is canonical UTF-8 JSON (sorted keys, compact separators, no
NaN); `size_bytes` is the length of that encoding, so payload and metadata are
both charged. Forwarding changes sender, recipient, sent tick and hop count
and nothing else: origin, EvidenceId and observation tick stay untouched.
There is no full route history in the packet; hop and age limits bound
circulation. Provenance is bookkeeping, not cryptographic authenticity.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

from argus.core.events import canonical_json
from argus.core.types import EvidenceConflict, EvidenceId, Observation

SCHEMA_VERSION = 1
MESSAGE_FIELDS = {"schema_version", "sender", "recipient", "sent_tick", "hop_count", "observation"}
OBSERVATION_FIELDS = {
    "evidence_id",
    "observed_tick",
    "entity",
    "value",
    "observer_pose",
    "sensor_model",
}


class MessageError(ValueError):
    """Malformed, unknown-version or inconsistent message."""


class ForwardLimit(ValueError):
    """Forwarding would exceed the hop limit or the maximum evidence age."""


def _int(value: Any, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise MessageError(f"{name} must be an int >= {minimum}, got {value!r}")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise MessageError(f"{name} must be a nonempty string")
    return value


@dataclass(frozen=True, slots=True)
class Message:
    sender: str
    recipient: str
    sent_tick: int
    hop_count: int
    observation: Observation
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise MessageError(f"unsupported message schema {self.schema_version!r}")
        _text(self.sender, "sender")
        _text(self.recipient, "recipient")
        if self.sender == self.recipient:
            raise MessageError("sender and recipient must differ")
        _int(self.sent_tick, "sent_tick")
        _int(self.hop_count, "hop_count")
        if self.observation.observed_tick > self.sent_tick:
            raise MessageError("a message cannot carry an observation from the future")

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "sender": self.sender,
            "recipient": self.recipient,
            "sent_tick": self.sent_tick,
            "hop_count": self.hop_count,
            "observation": self.observation.to_json(),
        }

    def encode(self) -> bytes:
        return canonical_json(self.to_json()).encode("utf-8")

    @property
    def size_bytes(self) -> int:
        return len(self.encode())

    @property
    def frame_id(self) -> str:
        # Same key format as in E04, so verified loss draws stay reproducible.
        eid = self.observation.evidence_id
        return f"{eid.episode}/{eid.origin}/{eid.sequence}"

    @classmethod
    def decode(cls, data: bytes) -> Message:
        try:
            raw = json.loads(data.decode("utf-8"), parse_constant=_reject_constant)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise MessageError(f"undecodable message: {error}") from error
        if not isinstance(raw, dict) or set(raw) != MESSAGE_FIELDS:
            raise MessageError("message fields do not match schema v1")
        if raw["schema_version"] != SCHEMA_VERSION:
            raise MessageError(f"unsupported message schema {raw['schema_version']!r}")
        obs = raw["observation"]
        if not isinstance(obs, dict) or set(obs) != OBSERVATION_FIELDS:
            raise MessageError("observation fields do not match schema v1")
        eid = obs["evidence_id"]
        pose = obs["observer_pose"]
        if not (isinstance(eid, list) and len(eid) == 3):
            raise MessageError("evidence_id must be [episode, origin, sequence]")
        if not (
            isinstance(pose, list)
            and len(pose) == 3
            and all(
                isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                for v in pose
            )
        ):
            raise MessageError("observer_pose must be three finite numbers")
        try:
            observation = Observation(
                EvidenceId(
                    _text(eid[0], "episode"), _text(eid[1], "origin"), _int(eid[2], "sequence")
                ),
                _int(obs["observed_tick"], "observed_tick"),
                _text(obs["entity"], "entity"),
                obs["value"] if type(obs["value"]) is int else -1,
                (float(pose[0]), float(pose[1]), float(pose[2])),
                _text(obs["sensor_model"], "sensor_model"),
            )
        except ValueError as error:
            raise MessageError(str(error)) from error
        return cls(
            _text(raw["sender"], "sender"),
            _text(raw["recipient"], "recipient"),
            _int(raw["sent_tick"], "sent_tick"),
            _int(raw["hop_count"], "hop_count"),
            observation,
        )


def _reject_constant(value: str) -> float:
    raise MessageError(f"non-finite number {value} in message")


def originate(observation: Observation, recipient: str, sent_tick: int) -> Message:
    """First transmission by the observing robot (hop 0)."""
    return Message(observation.evidence_id.origin, recipient, sent_tick, 0, observation)


def forward(
    message: Message,
    by: str,
    recipient: str,
    sent_tick: int,
    hop_limit: int,
    max_age_ticks: int,
) -> Message:
    """Re-send a received message; only transport metadata changes."""
    if by != message.recipient:
        raise MessageError(f"{by} cannot forward a message addressed to {message.recipient}")
    if sent_tick < message.sent_tick:
        raise MessageError("forwarding cannot happen before the message was sent")
    if message.hop_count + 1 > hop_limit:
        raise ForwardLimit(f"hop limit {hop_limit} reached")
    if sent_tick - message.observation.observed_tick > max_age_ticks:
        raise ForwardLimit(f"evidence older than {max_age_ticks} ticks")
    return Message(by, recipient, sent_tick, message.hop_count + 1, message.observation)


class EvidenceIndex:
    """Payload by EvidenceId: duplicates are no-ops, conflicting payloads raise."""

    def __init__(self) -> None:
        self._payloads: dict[EvidenceId, Observation] = {}

    def __len__(self) -> int:
        return len(self._payloads)

    def __contains__(self, evidence_id: object) -> bool:
        return evidence_id in self._payloads

    def add(self, observation: Observation) -> bool:
        """True when new, False for an identical duplicate; EvidenceConflict otherwise."""
        known = self._payloads.get(observation.evidence_id)
        if known is None:
            self._payloads[observation.evidence_id] = observation
            return True
        if known != observation:
            raise EvidenceConflict(f"{observation.evidence_id} arrived with another payload")
        return False

    def discard(self, evidence_id: EvidenceId) -> None:
        self._payloads.pop(evidence_id, None)


RESERVATION_FIELDS = {"schema_version", "sender", "recipient", "sent_tick", "reservation"}


@dataclass(frozen=True, slots=True)
class ReservationMessage:
    """Exploration target announcement; a separate frame type, charged like any frame."""

    sender: str
    recipient: str
    sent_tick: int
    target: tuple[int, int]
    expires_tick: int
    sequence: int
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise MessageError(f"unsupported message schema {self.schema_version!r}")
        _text(self.sender, "sender")
        _text(self.recipient, "recipient")
        if self.sender == self.recipient:
            raise MessageError("sender and recipient must differ")
        _int(self.sent_tick, "sent_tick")
        _int(self.sequence, "sequence")
        if self.expires_tick <= self.sent_tick:
            raise MessageError("a reservation must expire after it is sent")

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "sender": self.sender,
            "recipient": self.recipient,
            "sent_tick": self.sent_tick,
            "reservation": {
                "target": list(self.target),
                "expires_tick": self.expires_tick,
                "sequence": self.sequence,
            },
        }

    def encode(self) -> bytes:
        return canonical_json(self.to_json()).encode("utf-8")

    @property
    def size_bytes(self) -> int:
        return len(self.encode())

    @property
    def frame_id(self) -> str:
        return f"res/{self.sender}/{self.sequence}"


def decode_frame(data: bytes) -> Message | ReservationMessage | PoolMessage:
    """Decode either frame type; the field set decides, unknown shapes are rejected."""
    try:
        raw = json.loads(data.decode("utf-8"), parse_constant=_reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MessageError(f"undecodable frame: {error}") from error
    if isinstance(raw, dict) and set(raw) == POOL_FIELDS:
        body = raw["pool"]
        if (
            raw["schema_version"] != SCHEMA_VERSION
            or not isinstance(body, dict)
            or set(body) != {"pool_id", "entity", "probability_open", "as_of_tick"}
        ):
            raise MessageError("pool fields do not match schema v1")
        pid = body["pool_id"]
        if not (isinstance(pid, list) and len(pid) == 3):
            raise MessageError("pool_id must be [episode, origin, sequence]")
        probability = body["probability_open"]
        if type(probability) is int:
            probability = float(probability)
        return PoolMessage(
            _text(raw["sender"], "sender"),
            _text(raw["recipient"], "recipient"),
            _int(raw["sent_tick"], "sent_tick"),
            EvidenceId(_text(pid[0], "episode"), _text(pid[1], "origin"), _int(pid[2], "sequence")),
            _text(body["entity"], "entity"),
            probability,
            _int(body["as_of_tick"], "as_of_tick"),
        )
    if isinstance(raw, dict) and set(raw) == RESERVATION_FIELDS:
        body = raw["reservation"]
        if (
            raw["schema_version"] != SCHEMA_VERSION
            or not isinstance(body, dict)
            or set(body) != {"target", "expires_tick", "sequence"}
        ):
            raise MessageError("reservation fields do not match schema v1")
        target = body["target"]
        if not (
            isinstance(target, list) and len(target) == 2 and all(type(v) is int for v in target)
        ):
            raise MessageError("reservation target must be two ints")
        return ReservationMessage(
            _text(raw["sender"], "sender"),
            _text(raw["recipient"], "recipient"),
            _int(raw["sent_tick"], "sent_tick"),
            (target[0], target[1]),
            _int(body["expires_tick"], "expires_tick"),
            _int(body["sequence"], "sequence"),
        )
    return Message.decode(data)


POOL_FIELDS = {"schema_version", "sender", "recipient", "sent_tick", "pool"}


@dataclass(frozen=True, slots=True)
class PoolMessage:
    """Posterior-pool protocol frame: a marginal P(open), not raw evidence.

    A separate type on purpose: it never reuses the raw-evidence fields with a new
    meaning. Its pool_id identifies the pooling event so receivers apply it once.
    All fields, including metadata, are charged through size_bytes.
    """

    sender: str
    recipient: str
    sent_tick: int
    pool_id: EvidenceId
    entity: str
    probability_open: float
    as_of_tick: int
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise MessageError(f"unsupported message schema {self.schema_version!r}")
        _text(self.sender, "sender")
        _text(self.recipient, "recipient")
        _text(self.entity, "entity")
        if self.sender == self.recipient:
            raise MessageError("sender and recipient must differ")
        _int(self.sent_tick, "sent_tick")
        _int(self.as_of_tick, "as_of_tick")
        if self.as_of_tick > self.sent_tick:
            raise MessageError("a pool message cannot describe the future")
        p = self.probability_open
        if not (isinstance(p, float) and math.isfinite(p) and 0.0 <= p <= 1.0):
            raise MessageError("probability_open must be a finite probability")

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "sender": self.sender,
            "recipient": self.recipient,
            "sent_tick": self.sent_tick,
            "pool": {
                "pool_id": self.pool_id.to_json(),
                "entity": self.entity,
                "probability_open": self.probability_open,
                "as_of_tick": self.as_of_tick,
            },
        }

    def encode(self) -> bytes:
        return canonical_json(self.to_json()).encode("utf-8")

    @property
    def size_bytes(self) -> int:
        return len(self.encode())

    @property
    def frame_id(self) -> str:
        eid = self.pool_id
        return f"pool/{eid.episode}/{eid.origin}/{eid.sequence}"


POSITION_FIELDS = {"schema_version", "sender", "recipient", "sent_tick", "position"}


@dataclass(frozen=True, slots=True)
class PositionMessage:
    """Position-sharing frame (Cross-rank baseline): registered x, y of the sender."""

    sender: str
    recipient: str
    sent_tick: int
    x: float
    y: float
    sequence: int
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise MessageError(f"unsupported message schema {self.schema_version!r}")
        _text(self.sender, "sender")
        _text(self.recipient, "recipient")
        if self.sender == self.recipient:
            raise MessageError("sender and recipient must differ")
        _int(self.sent_tick, "sent_tick")
        _int(self.sequence, "sequence")
        if not (math.isfinite(self.x) and math.isfinite(self.y)):
            raise MessageError("position must be finite")

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "sender": self.sender,
            "recipient": self.recipient,
            "sent_tick": self.sent_tick,
            "position": {"x": self.x, "y": self.y, "sequence": self.sequence},
        }

    def encode(self) -> bytes:
        return canonical_json(self.to_json()).encode("utf-8")

    @property
    def size_bytes(self) -> int:
        return len(self.encode())

    @property
    def frame_id(self) -> str:
        return f"pos/{self.sender}/{self.sequence}"
