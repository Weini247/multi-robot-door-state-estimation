"""ARGUS-E04-T04: ID deduplication and fixed-lag reprocessing against full history."""

from __future__ import annotations

import random
from itertools import permutations

import numpy as np
import pytest

from argus.beliefs.exact_micro import joint_prior
from argus.beliefs.fixed_lag import FixedLagBelief, NominalPassageModel
from argus.beliefs.joint_filter import JointFilter
from argus.core.types import EvidenceConflict, EvidenceId, Observation
from argus.dynamics.binary_ctmc import BinaryCTMC

TICK = 0.1
MODEL = NominalPassageModel(BinaryCTMC(0.01, 0.02), BinaryCTMC(0.005, 0.045), 0.05, 1 / 3, 0.1)
STATIC = NominalPassageModel(BinaryCTMC(0.0, 0.0), BinaryCTMC(0.0, 0.0), 0.05, 0.5, 0.1)


def belief(
    window: int = 300,
    capacity: int = 4096,
    model: NominalPassageModel = MODEL,
    entities: tuple[str, ...] = ("door", "hall"),
) -> FixedLagBelief:
    return FixedLagBelief("ep", {e: model for e in entities}, TICK, window, capacity)


def obs(tick: int, value: int, entity: str = "door", origin: str = "r0", seq: int | None = None):
    sequence = tick * 10 if seq is None else seq
    return Observation(
        EvidenceId("ep", origin, sequence), tick, entity, value, (0.0, 0.0, 0.0), "m"
    )


def reference(model: NominalPassageModel, items: list[Observation], until: int) -> np.ndarray:
    """Chronological full-history filter: independent of the fixed-lag bookkeeping."""
    filt = JointFilter(
        model.z_model,
        model.g_model,
        model.private_error,
        joint_prior(model.prior_open, model.bias_probability),
    )
    current = 0
    for o in sorted(items, key=lambda o: (o.observed_tick, o.evidence_id)):
        filt.predict((o.observed_tick - current) * TICK)
        filt.correct(o.value)
        current = o.observed_tick
    filt.predict((until - current) * TICK)
    return np.array(filt.joint)


EVIDENCE = [
    obs(5, 1),
    obs(12, 1, origin="r1"),
    obs(12, 0, origin="r2"),
    obs(20, 1),
    obs(31, 0, origin="r1"),
    obs(33, 1, entity="hall"),
]


def test_ten_duplicates_are_noops() -> None:
    b = belief()
    assert b.ingest(obs(5, 1), 6) == "accepted"
    before = b.snapshot()
    assert [b.ingest(obs(5, 1), 6 + k) for k in range(10)] == ["duplicate"] * 10
    b.advance(16)
    after = belief()
    after.ingest(obs(5, 1), 6)
    after.advance(16)
    assert b.snapshot() == after.snapshot() and b.stored == 1
    assert before.tick == 6


def test_arrival_permutations_give_the_reference_posterior() -> None:
    door_items = [o for o in EVIDENCE if o.entity == "door"]
    expected = reference(MODEL, door_items, 40)
    rng = random.Random(0)
    orders = list(permutations(EVIDENCE))
    rng.shuffle(orders)
    for order in orders[:200]:
        b = belief()
        # Each item arrives no earlier than observed; arrival ticks follow the order.
        arrival = max(o.observed_tick for o in EVIDENCE)
        for o in order:
            assert b.ingest(o, arrival) == "accepted"
        b.advance(40)
        np.testing.assert_allclose(b.joint("door"), expected, atol=1e-12, rtol=0)


def test_interleaved_late_arrivals_match_reference() -> None:
    b = belief()
    schedule = [
        (obs(20, 1), 21),
        (obs(5, 1), 22),
        (obs(12, 0, origin="r2"), 25),
        (obs(12, 1, origin="r1"), 30),
        (obs(31, 0, origin="r1"), 33),
    ]
    for o, received in schedule:
        b.ingest(o, received)
    b.advance(40)
    expected = reference(MODEL, [o for o, _ in schedule], 40)
    np.testing.assert_allclose(b.joint("door"), expected, atol=1e-12, rtol=0)


def test_floor_is_inclusive() -> None:
    b = belief(window=30)
    b.advance(100)
    assert b.floor == 70
    assert b.ingest(obs(70, 1), 100) == "accepted"
    assert b.ingest(obs(69, 1), 100) == "too_old"


def test_capacity_k_plus_one_is_rejected_not_forgotten() -> None:
    b = belief(capacity=3)
    statuses = [b.ingest(obs(t, 1, seq=t), 10) for t in (1, 2, 3, 4)]
    assert statuses == ["accepted", "accepted", "accepted", "capacity_rejected"]
    expected = reference(MODEL, [obs(t, 1, seq=t) for t in (1, 2, 3)], 10)
    np.testing.assert_allclose(b.joint("door"), expected, atol=1e-12, rtol=0)
    assert b.ingest(obs(1, 1, seq=1), 11) == "duplicate"  # the accepted ones are still known
    # After the floor passes them, the slots free up again.
    b.advance(400)
    assert b.stored == 0
    assert b.ingest(obs(390, 1, seq=390), 400) == "accepted"


def test_no_lru_recount_after_eviction() -> None:
    b = belief(window=30)
    assert b.ingest(obs(10, 1), 11) == "accepted"
    b.advance(50)
    assert b.stored == 0  # folded into the checkpoint and released
    snapshot = b.snapshot()
    for _ in range(5):
        assert b.ingest(obs(10, 1), 50) == "too_old"
    assert b.snapshot() == snapshot


def test_checkpoint_advance_equals_full_history() -> None:
    b = belief(window=10)
    accepted = []
    for o in EVIDENCE:
        received = o.observed_tick + 2
        if b.ingest(o, received) == "accepted":
            accepted.append(o)
    b.advance(80)
    assert b.floor == 70 and b.stored == 0
    for entity in ("door", "hall"):
        expected = reference(MODEL, [o for o in accepted if o.entity == entity], 80)
        np.testing.assert_allclose(b.joint(entity), expected, atol=1e-12, rtol=0)


def test_joint_z_g_structure_is_kept_across_messages() -> None:
    b = belief(model=STATIC, entities=("door",))
    b.ingest(obs(3, 1, origin="r0"), 4)
    b.ingest(obs(3, 1, origin="r1"), 9)  # distinct evidence with the same shared bias
    assert b.probability_open("door") == pytest.approx(325 / 362, abs=1e-14)


def test_conflicts_and_invalid_arrivals_raise() -> None:
    b = belief()
    b.ingest(obs(5, 1), 6)
    with pytest.raises(EvidenceConflict):
        b.ingest(obs(5, 0), 7)
    with pytest.raises(ValueError):
        b.ingest(obs(9, 1), 8)  # received before observed
    with pytest.raises(ValueError):
        b.ingest(obs(1, 1, seq=99), 3)  # time moves backwards
    with pytest.raises(ValueError):
        b.ingest(Observation(EvidenceId("other", "r0", 1), 7, "door", 1, (0, 0, 0), "m"), 8)
    with pytest.raises(ValueError):
        b.ingest(obs(7, 1, entity="ghost"), 8)
