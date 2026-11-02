"""ARGUS-E04-T05: communication and fixed-lag fusion stressed together."""

from __future__ import annotations

import json

import numpy as np

from tests.integration.test_delayed_fusion import AGENTS, DelayedFusionFixture


def stressed() -> DelayedFusionFixture:
    return DelayedFusionFixture(seed=1).run()


def test_ids_and_filter_events_are_consistent() -> None:
    fixture = stressed()
    for agent in AGENTS:
        accepted = [o.evidence_id for o in fixture.accepted[agent]]
        assert len(accepted) == len(set(accepted))  # nothing counted twice
        for eid in accepted:
            assert eid.origin == agent or eid in fixture.received_ids[agent]
        counts = fixture.statuses[agent]
        assert counts["accepted"] == len(accepted)
        total = sum(counts.values())
        own = fixture.horizon // fixture.slot_ticks + 1
        delivered = sum(
            1
            for a in fixture.channel.attempts
            if a.message.recipient == agent and not a.dropped and a.delivery <= fixture.horizon
        )
        assert total == own + delivered  # every own reading and every delivery was ingested once


def test_raw_bytes_are_complete() -> None:
    fixture = stressed()
    attempts, stats = fixture.channel.attempts, fixture.channel.stats
    for attempt in attempts:
        assert attempt.size == len(attempt.message.encode())
    assert sum(stats.tx_attempt_bytes.values()) == sum(a.size for a in attempts)
    assert sum(stats.dropped_bytes.values()) == sum(a.size for a in attempts if a.dropped)
    delivered = [a for a in attempts if not a.dropped and a.delivery <= fixture.horizon]
    assert sum(stats.rx_bytes.values()) == sum(a.size for a in delivered)


def test_cache_stays_bounded() -> None:
    fixture = stressed()
    assert 0 < fixture.max_stored <= fixture.capacity
    assert fixture.peak_bytes > 0


def test_final_posterior_equals_chronological_reference() -> None:
    fixture = stressed()
    for agent in AGENTS:
        np.testing.assert_allclose(
            fixture.beliefs[agent].joint("door"), fixture.reference_joint(agent), atol=1e-12, rtol=0
        )


def test_drop_rates_are_reported_and_every_stress_path_occurs() -> None:
    report = stressed().report()
    print(json.dumps(report, indent=2, sort_keys=True))
    assert 0 < report["channel_drop_rate"] < 1  # type: ignore[operator]
    rejects = report["rejects"]
    assert isinstance(rejects, dict) and rejects.get("budget_rejected", 0) > 0
    ingest = report["ingest"]
    assert isinstance(ingest, dict)
    totals = {
        k: sum(agent.get(k, 0) for agent in ingest.values())
        for k in ("accepted", "duplicate", "too_old", "capacity_rejected")
    }
    assert all(n > 0 for n in totals.values()), totals
