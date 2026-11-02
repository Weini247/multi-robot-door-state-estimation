"""Delayed fusion under stress: ring forwarding through a lossy, budgeted channel (E04-T05).

DelayedFusionFixture runs three agents that observe the same passage. Each
sends its own evidence to both other agents (a broadcast as two unicasts) and
forwards every newly accepted item once to the next agent in a ring, so the
same evidence can arrive twice over different paths. Messages pass a
Gilbert-Elliott channel with propagation delay, a tight sender budget and
small link queues. Each agent fuses with a FixedLagBelief whose window and
capacity are small enough to reject evidence. The fixture records every
channel attempt and every ingest status, so the tests can audit identities,
bytes and the final posterior.
"""

from __future__ import annotations

import tracemalloc
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from argus.beliefs.exact_micro import joint_prior
from argus.beliefs.fixed_lag import FixedLagBelief, NominalPassageModel
from argus.beliefs.joint_filter import JointFilter
from argus.communication.loss import GilbertElliott
from argus.communication.messages import ForwardLimit, Message, forward, originate
from argus.communication.queue import Channel, ChannelConfig
from argus.core.events import KeyedRandom
from argus.core.types import EvidenceId, Observation
from argus.dynamics.binary_ctmc import BinaryCTMC

AGENTS = ("r0", "r1", "r2")
TICK = 0.1
MODEL = NominalPassageModel(BinaryCTMC(0.01, 0.02), BinaryCTMC(0.005, 0.045), 0.05, 1 / 3, 0.1)


def next_agent(agent: str) -> str:
    return AGENTS[(AGENTS.index(agent) + 1) % len(AGENTS)]


@dataclass
class DelayedFusionFixture:
    seed: int = 1
    horizon: int = 600
    slot_ticks: int = 5
    window_ticks: int = 15
    capacity: int = 6
    hop_limit: int = 3
    max_age_ticks: int = 15
    link_bytes_per_second: float = 600.0
    sender_bytes_per_second: float = 900.0
    queue_frames: int = 3
    statuses: dict[str, Counter[str]] = field(default_factory=dict)
    accepted: dict[str, list[Observation]] = field(default_factory=dict)
    received_ids: dict[str, set[EvidenceId]] = field(default_factory=dict)
    max_stored: int = 0
    forward_limits: int = 0
    peak_bytes: int = 0

    def run(self) -> DelayedFusionFixture:
        rng = KeyedRandom(self.seed)
        size = originate(self._observation("r0", 0, 0, rng), "r1", 0).size_bytes
        config = ChannelConfig(
            TICK,
            self.sender_bytes_per_second,
            50,
            self.link_bytes_per_second,
            2,
            self.queue_frames * size,
        )
        loss = GilbertElliott(0.2, 0.5, 0.05, 0.6, TICK, KeyedRandom(self.seed + 1), "stress")
        self.channel = Channel(config, loss)
        self.beliefs = {
            a: FixedLagBelief("ep", {"door": MODEL}, TICK, self.window_ticks, self.capacity)
            for a in AGENTS
        }
        for agent in AGENTS:
            self.statuses[agent] = Counter()
            self.accepted[agent] = []
            self.received_ids[agent] = set()
        tracemalloc.start()
        try:
            for tick in range(self.horizon + 1):
                for message in self.channel.deliveries(tick):
                    self._receive(message, tick)
                if tick % self.slot_ticks == 0:
                    for agent in AGENTS:
                        obs = self._observation(agent, tick, tick // self.slot_ticks, rng)
                        if self._ingest(agent, obs, tick):
                            # Broadcast = N-1 unicasts, so evidence also reaches agents twice.
                            for other in AGENTS:
                                if other != agent:
                                    self.channel.submit(originate(obs, other, tick), tick)
                self.max_stored = max(self.max_stored, *(b.stored for b in self.beliefs.values()))
            self.peak_bytes = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        for belief in self.beliefs.values():
            belief.advance(self.horizon)
        return self

    def _observation(self, agent: str, tick: int, slot: int, rng: KeyedRandom) -> Observation:
        value = rng.bernoulli(0.7, "stress.reading", agent, slot)
        return Observation(
            EvidenceId("ep", agent, slot), tick, "door", value, (0.0, 0.0, 0.0), "binary-v1"
        )

    def _ingest(self, agent: str, obs: Observation, tick: int) -> bool:
        status = self.beliefs[agent].ingest(obs, tick)
        self.statuses[agent][status] += 1
        if status == "accepted":
            self.accepted[agent].append(obs)
        return status == "accepted"

    def _receive(self, message: Message, tick: int) -> None:
        agent = message.recipient
        self.received_ids[agent].add(message.observation.evidence_id)
        if self._ingest(agent, message.observation, tick):
            try:
                onward = forward(
                    message, agent, next_agent(agent), tick, self.hop_limit, self.max_age_ticks
                )
            except ForwardLimit:
                self.forward_limits += 1
                return
            if onward.recipient != message.observation.evidence_id.origin:
                self.channel.submit(onward, tick)

    def reference_joint(self, agent: str) -> np.ndarray:
        """Chronological full-history filter over exactly the accepted evidence."""
        filt = JointFilter(
            MODEL.z_model,
            MODEL.g_model,
            MODEL.private_error,
            joint_prior(MODEL.prior_open, MODEL.bias_probability),
        )
        current = 0
        for o in sorted(self.accepted[agent], key=lambda o: (o.observed_tick, o.evidence_id)):
            filt.predict((o.observed_tick - current) * TICK)
            filt.correct(o.value)
            current = o.observed_tick
        filt.predict((self.horizon - current) * TICK)
        return np.array(filt.joint)

    def report(self) -> dict[str, object]:
        attempts = self.channel.attempts
        stats = self.channel.stats
        sent = sum(stats.tx_attempt_bytes.values())
        return {
            "attempts": len(attempts),
            "channel_drop_rate": sum(a.dropped for a in attempts) / max(1, len(attempts)),
            "tx_attempt_bytes": sent,
            "dropped_bytes": sum(stats.dropped_bytes.values()),
            "rx_bytes": sum(stats.rx_bytes.values()),
            "rejects": dict(Counter({reason: n for (_, reason), n in stats.rejects.items()})),
            "ingest": {a: dict(c) for a, c in self.statuses.items()},
            "forward_limits": self.forward_limits,
            "max_stored": self.max_stored,
            "peak_traced_bytes": self.peak_bytes,
        }


def test_fixture_is_deterministic() -> None:
    a, b = DelayedFusionFixture(seed=3).run(), DelayedFusionFixture(seed=3).run()
    assert a.report()["ingest"] == b.report()["ingest"]
    assert [x.key for x in a.channel.attempts] == [x.key for x in b.channel.attempts]
    np.testing.assert_array_equal(a.beliefs["r1"].joint("door"), b.beliefs["r1"].joint("door"))


def test_every_forward_keeps_the_original_identity() -> None:
    fixture = DelayedFusionFixture().run()
    for attempt in fixture.channel.attempts:
        message = attempt.message
        assert message.observation.evidence_id.origin in AGENTS
        assert message.hop_count <= fixture.hop_limit
        if message.hop_count == 0:
            assert message.sender == message.observation.evidence_id.origin
        else:
            assert message.recipient == next_agent(message.sender)
