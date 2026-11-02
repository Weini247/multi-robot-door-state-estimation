"""ARGUS-E07-T03: dependence heuristics — one reading per public bin and log pooling."""

from __future__ import annotations

import math

import pytest

from argus.beliefs.conservative import (
    POOL_PROTOCOL,
    RAW_PROTOCOL,
    PoolLedger,
    logarithmic_pool,
    one_per_bin,
    space_time_bin,
)
from argus.beliefs.exact_micro import ImpossibleEvidence, shared_bias_posterior
from argus.communication.messages import MessageError, PoolMessage, decode_frame, originate
from argus.core.types import EvidenceId, Observation


def obs(origin: str, seq: int, tick: int, pose: tuple[float, float, float], value: int = 1):
    return Observation(EvidenceId("ep", origin, seq), tick, "door", value, pose, "m")


def test_equal_pools_stay_equal() -> None:
    for p in (0.0, 0.1, 1 / 3, 0.5, 0.86, 1.0):
        for w in (0.0, 0.3, 0.5, 1.0):
            assert logarithmic_pool(p, p, w) == pytest.approx(p, abs=1e-15)


def test_weight_endpoints_return_one_input() -> None:
    assert logarithmic_pool(0.2, 0.9, 1.0) == 0.2
    assert logarithmic_pool(0.2, 0.9, 0.0) == 0.9
    assert logarithmic_pool(0.0, 1.0, 1.0) == 0.0  # endpoints never mix supports


def test_disjoint_supports_raise() -> None:
    with pytest.raises(ImpossibleEvidence):
        logarithmic_pool(0.0, 1.0, 0.5)
    with pytest.raises(ImpossibleEvidence):
        logarithmic_pool(1.0, 0.0, 0.5)
    with pytest.raises(ValueError):
        logarithmic_pool(0.5, 1.2, 0.5)


def test_pool_is_a_log_odds_average() -> None:
    def logit(p: float) -> float:
        return math.log(p / (1 - p))

    pooled = logarithmic_pool(0.2, 0.9, 0.25)
    assert logit(pooled) == pytest.approx(0.25 * logit(0.2) + 0.75 * logit(0.9), abs=1e-12)


def test_no_guarantee_pool_loses_information_and_depends_on_repetition() -> None:
    # Two agents, each holding one distinct reading under a shared bias (exact joint 325/362).
    single = shared_bias_posterior(0.5, 0.1, 0.05, [1])
    exact = shared_bias_posterior(0.5, 0.1, 0.05, [1, 1])
    pooled = logarithmic_pool(single, single)
    assert pooled < exact  # information lost: no exactness, no calibration guarantee
    # Re-pooling a pooled result with one input again shifts the answer.
    a, b = 0.6, 0.95
    once = logarithmic_pool(a, b)
    again = logarithmic_pool(once, b)
    assert again != pytest.approx(once)


def test_pool_events_are_deduplicated_by_their_own_id() -> None:
    ledger = PoolLedger()
    pid = EvidenceId("ep", "pool-r1", 3)
    assert ledger.admit(pid) is True
    assert [ledger.admit(pid) for _ in range(5)] == [False] * 5
    assert ledger.admit(EvidenceId("ep", "pool-r1", 4)) is True


def test_one_per_bin_uses_public_pose_and_time_only() -> None:
    near_a = obs("r0", 1, 10, (2.1, 2.2, 0.0))
    near_b = obs("r1", 1, 12, (2.4, 2.9, 1.0))  # same 1 m cell, same 5 s slice
    far = obs("r2", 1, 12, (5.5, 2.0, 0.0))  # other cell
    later = obs("r0", 2, 70, (2.1, 2.2, 0.0))  # other time slice
    kept = one_per_bin([near_b, near_a, far, later], cell_m=1.0, period_ticks=50)
    assert near_a in kept and near_b not in kept  # smallest EvidenceId wins, deterministic
    assert far in kept and later in kept and len(kept) == 3
    assert space_time_bin(near_a, 1.0, 50) == ("door", 2, 2, 0)
    with pytest.raises(ValueError):
        space_time_bin(near_a, 0.0, 50)


def test_raw_and_pool_protocols_are_distinct_frames_with_full_metadata_bytes() -> None:
    reading = obs("r0", 1, 10, (2.0, 2.0, 0.0))
    raw = originate(reading, "r1", 11)
    pool = PoolMessage("r0", "r1", 11, EvidenceId("ep", "pool-r0", 1), "door", 0.86, 10)
    assert RAW_PROTOCOL != POOL_PROTOCOL
    assert decode_frame(pool.encode()) == pool
    assert decode_frame(raw.encode()) == raw
    assert pool.size_bytes == len(pool.encode()) and raw.size_bytes == len(raw.encode())
    assert pool.frame_id.startswith("pool/") and not raw.frame_id.startswith("pool/")
    # A comparison must account for both payload kinds: the frames differ in bytes.
    assert pool.size_bytes != raw.size_bytes
    with pytest.raises(MessageError):
        PoolMessage("r0", "r1", 5, EvidenceId("ep", "p", 1), "door", 0.5, 6)  # future
    with pytest.raises(MessageError):
        PoolMessage("r0", "r1", 5, EvidenceId("ep", "p", 1), "door", 1.5, 5)
