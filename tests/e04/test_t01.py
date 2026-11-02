"""ARGUS-E04-T01: wire format v1, byte accounting and provenance."""

from __future__ import annotations

import json

import pytest

from argus.communication.messages import (
    EvidenceIndex,
    ForwardLimit,
    Message,
    MessageError,
    forward,
    originate,
)
from argus.core.types import EvidenceConflict, EvidenceId, Observation


def obs(entity: str = "door", value: int = 1, tick: int = 3, seq: int = 0) -> Observation:
    return Observation(
        EvidenceId("ep-1", "r0", seq), tick, entity, value, (1.0, 2.5, -0.25), "binary-v1"
    )


def test_roundtrip_is_identical_in_object_and_bytes() -> None:
    message = originate(obs(), "r1", 4)
    data = message.encode()
    assert Message.decode(data) == message
    assert Message.decode(data).encode() == data
    assert json.loads(data)["observation"]["evidence_id"] == ["ep-1", "r0", 0]


def test_encoding_is_canonical_and_counts_every_utf8_byte() -> None:
    plain = originate(obs("door"), "r1", 4)
    umlaut = originate(obs("Tür-Ü"), "r1", 4)
    text = umlaut.encode().decode("utf-8")
    assert "Tür-Ü" in text  # not \\u-escaped: the real UTF-8 bytes are charged
    assert umlaut.size_bytes - plain.size_bytes == len("Tür-Ü".encode()) - len("door")
    assert umlaut.size_bytes == len(text.encode("utf-8")) > len(text)
    # Metadata is charged too: a longer recipient name costs exactly its extra bytes.
    longer = originate(obs("door"), "robot-1", 4)
    assert longer.size_bytes - plain.size_bytes == len("robot-1") - len("r1")
    assert b" " not in plain.encode()


def test_forward_keeps_origin_and_evidence_but_updates_transport() -> None:
    first = originate(obs(), "r1", 4)
    second = forward(first, by="r1", recipient="r2", sent_tick=6, hop_limit=3, max_age_ticks=300)
    assert second.observation == first.observation
    assert second.observation.evidence_id.origin == "r0"
    assert (second.sender, second.recipient, second.sent_tick, second.hop_count) == (
        "r1",
        "r2",
        6,
        1,
    )
    third = forward(second, by="r2", recipient="r0", sent_tick=6, hop_limit=3, max_age_ticks=300)
    assert third.hop_count == 2 and third.observation.evidence_id == first.observation.evidence_id


def test_hop_and_age_limits() -> None:
    message = originate(obs(tick=0), "r1", 0)
    hops = message
    for k, (by, to) in enumerate([("r1", "r2"), ("r2", "r3"), ("r3", "r0")], start=1):
        hops = forward(hops, by, to, k, hop_limit=3, max_age_ticks=300)
    with pytest.raises(ForwardLimit):
        forward(hops, "r0", "r1", 4, hop_limit=3, max_age_ticks=300)
    forward(message, "r1", "r2", 300, hop_limit=3, max_age_ticks=300)  # exactly at the limit
    with pytest.raises(ForwardLimit):
        forward(message, "r1", "r2", 301, hop_limit=3, max_age_ticks=300)


@pytest.mark.parametrize(
    "call",
    [
        lambda: forward(originate(obs(), "r1", 4), "r2", "r3", 5, 3, 300),  # not the recipient
        lambda: forward(originate(obs(), "r1", 4), "r1", "r2", 3, 3, 300),  # back in time
        lambda: Message("r0", "r1", 2, 0, obs(tick=3)),  # future observation
        lambda: Message("r0", "r0", 4, 0, obs()),
        lambda: Message("r0", "r1", 4, -1, obs()),
        lambda: Message("r0", "r1", 4, 0, obs(), schema_version=2),
    ],
)
def test_inconsistent_messages_rejected(call: object) -> None:
    with pytest.raises(MessageError):
        call()  # type: ignore[operator]


def mutate(transform: object) -> bytes:
    raw = json.loads(originate(obs(), "r1", 4).encode())
    transform(raw)  # type: ignore[operator]
    return json.dumps(raw).encode()


@pytest.mark.parametrize(
    "data",
    [
        b"\xff\xfe",
        b"{not json",
        mutate(lambda m: m.update(schema_version=2)),
        mutate(lambda m: m.update(extra=1)),
        mutate(lambda m: m.pop("hop_count")),
        mutate(lambda m: m["observation"].update(bias=1)),
        mutate(lambda m: m["observation"].update(value=2)),
        mutate(lambda m: m["observation"].update(value=True)),
        mutate(lambda m: m["observation"].update(evidence_id=["ep", "r0"])),
        mutate(lambda m: m.update(sent_tick=1.5)),
        b'{"schema_version":1,"sender":"r0","recipient":"r1","sent_tick":4,"hop_count":0,'
        b'"observation":{"evidence_id":["ep","r0",0],"observed_tick":3,"entity":"d","value":1,'
        b'"observer_pose":[NaN,0,0],"sensor_model":"m"}}',
    ],
)
def test_unknown_or_malformed_wire_data_rejected(data: bytes) -> None:
    with pytest.raises(MessageError):
        Message.decode(data)


def test_colliding_id_with_other_payload_is_a_conflict() -> None:
    index = EvidenceIndex()
    assert index.add(obs()) is True
    for _ in range(10):
        assert index.add(obs()) is False  # identical duplicate: no-op
    assert len(index) == 1
    with pytest.raises(EvidenceConflict):
        index.add(obs(value=0))  # same id, other bit
    with pytest.raises(EvidenceConflict):
        index.add(obs(tick=4))
    assert index.add(obs(seq=1)) is True and len(index) == 2
