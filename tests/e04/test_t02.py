"""ARGUS-E04-T02: rolling sender budget, FIFO links, delivery timing and queue limits."""

from __future__ import annotations

import pytest

from argus.communication.messages import Message, originate
from argus.communication.queue import Channel, ChannelConfig, Link
from argus.core.types import EvidenceId, Observation

W = 10


def msg(sender: str, recipient: str, tick: int, seq: int, entity: str = "door") -> Message:
    obs = Observation(EvidenceId("ep", sender, seq), tick, entity, 1, (0.0, 0.0, 0.0), "m")
    return originate(obs, recipient, tick)


SIZE = msg("r0", "r1", 0, 0).size_bytes


def config(
    budget_frames: float = 2.5, queue_frames: int = 10, link_rate: float = 1e9, propagation: int = 2
) -> ChannelConfig:
    # Budget of `budget_frames` frames per window; the half frame absorbs the few extra
    # bytes of multi-digit ticks, while one more frame still does not fit.
    return ChannelConfig(
        tick_seconds=0.1,
        sender_bytes_per_second=budget_frames * SIZE / (W * 0.1),
        window_ticks=W,
        link_bytes_per_second=link_rate,
        propagation_ticks=propagation,
        queue_capacity_bytes=queue_frames * SIZE,
    )


class DropAll:
    def drops(self, link: Link, tick: int, message_key: str) -> bool:
        return True


def test_budget_is_charged_for_lost_frames() -> None:
    channel = Channel(config(), DropAll())
    assert channel.submit(msg("r0", "r1", 0, 0), 0) == "queued"
    assert channel.submit(msg("r0", "r2", 0, 1), 0) == "queued"
    assert channel.submit(msg("r0", "r3", 0, 2), 0) == "budget_rejected"
    assert channel.stats.tx_attempt_bytes["r0"] == 2 * SIZE
    assert channel.stats.dropped_bytes["r0"] == 2 * SIZE
    assert all(channel.deliveries(t) == () for t in range(20))
    assert channel.stats.rx_bytes == {}


def test_window_boundary_k_equals_w() -> None:
    channel = Channel(config())
    # Starts at ticks 1 and W via separate links (each send starts at now + 1).
    assert channel.submit(msg("r0", "r1", 0, 0), 0) == "queued"  # start 1
    assert channel.submit(msg("r0", "r2", W - 1, 1), W - 1) == "queued"  # start W
    # Window (0, W] already holds two frames; a start at W is rejected ...
    assert channel.submit(msg("r0", "r3", W - 1, 2), W - 1) == "budget_rejected"
    # ... but at W + 1 the frame started at tick 1 has left the window (W+1-W, W+1].
    assert channel.submit(msg("r0", "r3", W, 3), W) == "queued"
    assert 0 <= channel.remaining_budget("r0", W + 1) < SIZE  # no further frame fits
    assert channel.remaining_budget("r0", 2 * W + 1) == pytest.approx(config().budget_bytes)


def test_new_start_cannot_break_a_later_window() -> None:
    channel = Channel(config())
    assert channel.submit(msg("r0", "r1", 4, 0), 4) == "queued"  # start 5
    assert channel.submit(msg("r0", "r2", 6, 1), 6) == "queued"  # start 7
    # A start at 1 fits window (-9, 1] but would overfill window (0, 10] later on.
    late = Channel(config())
    late.submit(msg("r0", "r1", 4, 0), 4)
    late.submit(msg("r0", "r2", 6, 1), 6)
    assert late.submit(msg("r0", "r3", 0, 2), 0) == "budget_rejected"


def test_budget_rejection_costs_nothing() -> None:
    channel = Channel(config(budget_frames=1.5))
    assert channel.submit(msg("r0", "r1", 0, 0), 0) == "queued"
    assert channel.submit(msg("r0", "r2", 0, 1), 0) == "budget_rejected"
    assert channel.stats.tx_attempt_bytes["r0"] == SIZE
    assert channel.stats.rejects[("r0", "budget_rejected")] == 1
    assert channel.submit(msg("r0", "r2", W + 1, 2), W + 1) == "queued"


def test_large_frames_rejected() -> None:
    big = msg("r0", "r1", 0, 0, entity="x" * 400)
    assert Channel(config(budget_frames=1.5)).submit(big, 0) == "frame_too_large"
    assert Channel(config(budget_frames=50, queue_frames=1)).submit(big, 0) == "frame_too_large"


def test_queue_memory_limit_and_fifo_service() -> None:
    # Slow link: one frame per 3 ticks, queue holds 2 frames waiting.
    slow = config(budget_frames=50, queue_frames=2, link_rate=SIZE / 0.25, propagation=0)
    channel = Channel(slow)
    statuses = [channel.submit(msg("r0", "r1", 0, k), 0) for k in range(4)]
    assert statuses == ["queued", "queued", "queue_rejected", "queue_rejected"]
    starts = [a.start for a in channel.attempts]
    assert starts == [1, 4]  # FIFO, service ceil(SIZE / (rate * .1)) = 3 ticks
    assert channel.deliveries(4) == (channel.attempts[0].message,)
    assert channel.deliveries(7) == (channel.attempts[1].message,)
    # Once the first frame is on the wire, memory for one more waiting frame frees up.
    assert channel.submit(msg("r0", "r1", 1, 9), 1) == "queued"


def test_first_delivery_is_never_in_the_submission_tick() -> None:
    channel = Channel(config(propagation=0))
    channel.submit(msg("r0", "r1", 5, 0), 5)
    assert channel.deliveries(5) == ()
    assert len(channel.deliveries(7)) == 1  # start 6, one service tick, no propagation
    assert channel.stats.rx_bytes["r1"] == SIZE
    assert channel.deliveries(7) == ()  # delivered exactly once


def test_deliveries_are_deterministic_across_links() -> None:
    def run(order: list[tuple[str, str]]) -> list[tuple[str, str]]:
        channel = Channel(config(budget_frames=50))
        for k, (s, r) in enumerate(order):
            channel.submit(msg(s, r, 0, k), 0)
        return [(m.sender, m.recipient) for m in channel.deliveries(4)]

    pairs = [("r2", "r0"), ("r0", "r1"), ("r1", "r2")]
    assert run(pairs) == run(pairs[::-1]) == sorted(pairs)


def test_submission_tick_must_match_and_config_validated() -> None:
    with pytest.raises(ValueError):
        Channel(config()).submit(msg("r0", "r1", 3, 0), 4)
    for bad in ({"window_ticks": 0}, {"propagation_ticks": -1}, {"tick_seconds": 0.0}):
        values = dict(
            tick_seconds=0.1,
            sender_bytes_per_second=100.0,
            window_ticks=10,
            link_bytes_per_second=100.0,
            propagation_ticks=1,
            queue_capacity_bytes=100,
        )
        values.update(bad)
        with pytest.raises(ValueError):
            ChannelConfig(**values)  # type: ignore[arg-type]
