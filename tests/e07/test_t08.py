"""ARGUS-E07-T08: factor switches in the runner and the eight X1 cells."""

from __future__ import annotations

import dataclasses
import hashlib
import json

import pytest

from argus.communication.messages import Message
from argus.experiments.cells import CELLS, PREREGISTERED, FactorCell, apply_cell
from argus.experiments.runner import (
    EpisodeConfig,
    ExperimentRunner,
    SensorBias,
    build_noise,
)
from argus.world.floorplan import FloorplanParams, generate_floorplan
from argus.world.scenario import WorldBackend2D
from tests.e05.test_t03 import config

# Digests of the runner fixture recorded at commit 69387cf, before the factor switches.
BEFORE = {
    "frontier": "140920eb3260bd893f560d23cb90f24a01d068a5759260db9e305046a9e1a60c",
    "naive_gossip": "ea0aa470153a3cb122359864f6a843d289409779b253d8d6041689ebfc9d4d86",
    "repair": "140920eb3260bd893f560d23cb90f24a01d068a5759260db9e305046a9e1a60c",
    "ttl_ids": "32b11171f9d53c4567e64dfadcccc8df4e479565c7601409b62bf80d2a17fbaa",
    "cross_rank": "ff20b1760ebd00387735e5c55d38842b77a1417b0f0fb9e42506a85b04e6edbb",
}


def digest(method: str) -> str:
    runner = ExperimentRunner()
    result = runner.run(config(method))
    rows = [
        [r.tick, r.robot_id, r.entity, repr(r.probability), r.truth, r.decision]
        + [r.decision_loss, r.sent_bytes]
        for r in result.records
    ]
    attempts = [
        [a.message.frame_id, a.size, a.dropped, a.delivery] for a in runner.last_channel.attempts
    ]
    return hashlib.sha256(json.dumps([result.status, rows, attempts]).encode()).hexdigest()


@pytest.mark.parametrize("method", sorted(BEFORE))
def test_defaults_reproduce_the_previous_runner(method: str) -> None:
    assert digest(method) == BEFORE[method]


# --- sensor factor ----------------------------------------------------------------


def readings(cfg: EpisodeConfig, n: int = 6000) -> list[tuple[int, int]]:
    """Errors of two robots reading the same passage in the same slot, open truth."""
    _, noise = build_noise(cfg)
    out = []
    for k in range(n):
        tick = 5 * k
        location = (float(k % 7) * 2.3, float(k % 5) * 3.1)
        a = noise.reading("r0", "door", k, tick, 1, location) ^ 1
        b = noise.reading("r1", "door", k, tick, 1, location) ^ 1
        out.append((a, b))
    return out


def test_marginal_error_matches_and_only_the_correlated_sensor_shares_errors() -> None:
    independent = apply_cell(config(), FactorCell(True, False, True))
    correlated = apply_cell(config(), FactorCell(True, True, True))
    e = PREREGISTERED.marginal_error
    for cfg in (independent, correlated):
        pairs = readings(cfg)
        rate = sum(a + b for a, b in pairs) / (2 * len(pairs))
        assert abs(rate - e) < 0.015, rate
    both_ind = sum(a & b for a, b in readings(independent)) / 6000
    both_cor = sum(a & b for a, b in readings(correlated)) / 6000
    # independent: e^2 = .0196; correlated: rho (1-q)^2 + (1-rho) q^2 = .049
    assert abs(both_ind - e * e) < 0.01
    assert both_cor > 0.035 and both_cor > 2 * both_ind


def test_bias_parameters_are_validated() -> None:
    for bad in (0.0, 0.5, -0.1):
        with pytest.raises(ValueError):
            SensorBias(bad)
    with pytest.raises(ValueError):
        dataclasses.replace(config(), sensor_bias=SensorBias(0.2))  # rho above e = .14
    with pytest.raises(ValueError):
        dataclasses.replace(config(), hop_limit=-1)


# --- circulation factor ------------------------------------------------------------


def run_gossip(relay: bool) -> tuple[ExperimentRunner, list[Message]]:
    runner = ExperimentRunner()
    cfg = apply_cell(config("naive_gossip"), FactorCell(True, False, relay))
    assert runner.run(cfg).status == "ok"
    messages = [a.message for a in runner.last_channel.attempts if isinstance(a.message, Message)]
    return runner, messages


def test_hop_limit_zero_sends_only_own_evidence() -> None:
    runner, messages = run_gossip(relay=False)
    assert messages
    assert all(m.sender == m.observation.evidence_id.origin and m.hop_count == 0 for m in messages)
    assert runner.hop_limited > 0  # gossip tried to forward and was refused


def test_relay_counts_hops_and_respects_the_limit() -> None:
    runner, messages = run_gossip(relay=True)
    forwarded = [m for m in messages if m.sender != m.observation.evidence_id.origin]
    assert forwarded and all(1 <= m.hop_count <= PREREGISTERED.relay_hop_limit for m in forwarded)
    first = {}
    for m in messages:  # a forward never claims fewer hops than its sender's copy
        first.setdefault((m.recipient, m.observation.evidence_id), m.hop_count)
    for m in forwarded:
        held = first.get((m.sender, m.observation.evidence_id))
        assert held is not None and m.hop_count >= held + 1


# --- dynamics factor and cells -------------------------------------------------------


def test_static_passages_never_change() -> None:
    cfg = apply_cell(config(), FactorCell(False, False, False))
    world = WorldBackend2D(cfg.scenario, 11)
    first = world.passage_states()
    for tick in range(1, 400):
        world.advance(tick, {})
        assert world.passage_states() == first


def test_eight_distinct_model_matched_cells() -> None:
    assert len(CELLS) == 8 and len({c.cell_id for c in CELLS}) == 8
    configs = {c.cell_id: apply_cell(config(), c) for c in CELLS}
    assert len({json.dumps(c.to_json(), sort_keys=True) for c in configs.values()}) == 8
    cor = configs["dyn-cor-relay"].nominal
    assert cor.bias_probability == 0.05 and cor.private_error == pytest.approx(0.1)
    ind = configs["sta-ind-direct"]
    assert ind.nominal.bias_probability == 0.0 and ind.nominal.private_error == 0.14
    assert ind.nominal.z_model.rate_01 == 0.0 and ind.nominal.prior_open == pytest.approx(1 / 3)
    assert configs["dyn-ind-direct"].hop_limit == 0 and configs["dyn-ind-relay"].hop_limit == 3


@pytest.mark.parametrize("cell", CELLS, ids=lambda c: c.cell_id)
def test_every_cell_runs_on_a_generated_building(cell: FactorCell) -> None:
    plan = generate_floorplan("loop", 2, FloorplanParams(horizon_ticks=300))
    base = EpisodeConfig("e", "b", plan.spec, "repair", 1, 2, 3, 4, 5)
    result = ExperimentRunner().run(apply_cell(base, cell))
    assert result.status == "ok", result.error
    assert result.manifest["config"]["hop_limit"] == (3 if cell.relay else 0)


def test_naive_belief_survives_hundreds_of_duplicate_readings() -> None:
    """Regression (2026-11-02): naive gossip crashed with OverflowError in validation tuning."""
    import math

    from argus.beliefs.variants import IndependentLogOddsBelief
    from argus.core.types import EvidenceId, Observation

    belief = IndependentLogOddsBelief("ep", 1 / 3, 0.14, dedupe=False, ttl_ticks=None)
    obs = Observation(EvidenceId("ep", "r0", 0), 0, "door", 0, (0.0, 0.0, 0.0), "m")
    llr = math.log(0.86 / 0.14)
    x = math.log((1 / 3) / (1 - 1 / 3))
    for n in range(1, 500):
        belief.ingest(obs, n)
        p = belief.probability_open("door")
        x += -llr  # same summation order as the belief
        expected = math.exp(x) if x < -700 else 1 / (1 + math.exp(-x))
        assert p == expected and 0.0 <= p < 1.0
