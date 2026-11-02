"""ARGUS-E07-T02: simple control methods with fixed components and equal resources."""

from __future__ import annotations

import pytest

from argus.beliefs.variants import IndependentLogOddsBelief
from argus.core.types import EvidenceId, Observation
from argus.experiments.runner import ExperimentRunner
from argus.policies.baselines import METHOD_SPECS
from argus.policies.controller import DEFAULT_CONTROLLER
from tests.e05.test_t03 import config


def reading(origin: str, seq: int, tick: int, value: int = 1) -> Observation:
    return Observation(EvidenceId("ep", origin, seq), tick, "door", value, (0.0, 0.0, 0.0), "m")


def test_ttl_and_provenance_control_is_registered_and_naive_is_not_alone() -> None:
    controls = {mid for mid, spec in METHOD_SPECS.items() if spec.role == "control"}
    assert {"ttl_ids", "dynamics_ids", "no_comm", "naive_gossip", "age_revisit"} <= controls
    assert "ttl-id-belief" in METHOD_SPECS["ttl_ids"].components
    assert len(controls - {"naive_gossip"}) >= 4
    assert METHOD_SPECS["repair"].role == "candidate"


def test_every_method_declares_components_and_tunables() -> None:
    for spec in METHOD_SPECS.values():
        assert len(spec.components) == 3  # movement, belief, communication
        assert all(isinstance(t, str) for t in spec.tunable)


@pytest.mark.parametrize(
    "method", ["no_comm", "naive_gossip", "ttl_ids", "dynamics_ids", "age_revisit", "repair"]
)
def test_methods_run_with_identical_resources_and_list_components(method: str) -> None:
    runner = ExperimentRunner()
    cfg = config(method)
    result = runner.run(cfg)
    assert result.status == "ok", result.error
    manifest = result.manifest
    assert manifest["method_components"][:3] == list(METHOD_SPECS[method].components)
    # Same sensor, channel, losses and budget for every method: only method_id differs.
    reference = config("no_comm").to_json()
    assert {k: v for k, v in manifest["config"].items() if k != "method_id"} == {
        k: v for k, v in reference.items() if k != "method_id"
    }
    assert not any(r == "budget_rejected" for _, r in runner.last_channel.stats.rejects)


def test_all_policies_share_the_controller() -> None:
    from argus.experiments.runner import method_context

    ctx = method_context(config(), ("r0", "r1"))
    for spec in METHOD_SPECS.values():
        policy = spec.policy(ctx)
        inner = getattr(policy, "movement", policy)
        controller = getattr(inner, "controller", None) or getattr(
            getattr(inner, "frontier", None), "controller", None
        )
        if spec.method_id != "wait":
            assert controller is DEFAULT_CONTROLLER, spec.method_id


def test_same_marginal_error_in_every_belief() -> None:
    from argus.experiments.runner import method_context

    ctx = method_context(config(), ("r0", "r1"))
    assert ctx.marginal_error == pytest.approx(0.14)
    nominal = ctx.nominal
    e = (
        nominal.private_error
        + nominal.bias_probability
        - 2 * nominal.private_error * nominal.bias_probability
    )
    assert e == pytest.approx(ctx.marginal_error)  # joint model and controls share e = 0.14


def test_naive_belief_double_counts_and_ttl_belief_does_not() -> None:
    naive = IndependentLogOddsBelief("ep", 0.5, 0.14, dedupe=False, ttl_ticks=None)
    ttl = IndependentLogOddsBelief("ep", 0.5, 0.14, dedupe=True, ttl_ticks=50)
    copy = reading("r1", 0, 0)
    for belief in (naive, ttl):
        belief.ingest(copy, 1)
        belief.ingest(copy, 2)
    assert ttl.probability_open("door") == pytest.approx(0.86)
    assert naive.probability_open("door") == pytest.approx(0.86**2 / (0.86**2 + 0.14**2))
    ttl.advance(60)
    assert ttl.probability_open("door") == pytest.approx(0.5)  # expired evidence ignored
    assert ttl.ingest(reading("r1", 1, 5), 60) == "too_old"
    naive.advance(10_000)
    assert naive.probability_open("door") > 0.97  # never ages


def test_variant_rejects_conflicts_and_bad_parameters() -> None:
    from argus.core.types import EvidenceConflict

    ttl = IndependentLogOddsBelief("ep", 0.5, 0.14, dedupe=True, ttl_ticks=None)
    ttl.ingest(reading("r1", 0, 0, 1), 1)
    with pytest.raises(EvidenceConflict):
        ttl.ingest(reading("r1", 0, 0, 0), 2)
    for kwargs in ({"prior_open": 0.0}, {"marginal_error": 0.5}, {"ttl_ticks": -1}):
        args = {
            "episode": "ep",
            "prior_open": 0.5,
            "marginal_error": 0.14,
            "dedupe": True,
            "ttl_ticks": None,
            **kwargs,
        }
        with pytest.raises(ValueError):
            IndependentLogOddsBelief(**args)  # type: ignore[arg-type]


def test_timed_sends_respect_the_period_and_send_each_item_once() -> None:
    runner = ExperimentRunner()
    runner.run(config("dynamics_ids"))
    frames = [a for a in runner.last_channel.attempts if not a.message.frame_id.startswith("res/")]
    assert frames, "timed sends happened"
    pairs = [(a.message.frame_id, a.message.recipient) for a in frames]
    assert len(pairs) == len(set(pairs))  # no evidence sent twice to the same teammate
    assert all(a.message.observation.evidence_id.origin == a.message.sender for a in frames)
    sends_ticks = sorted({a.submitted for a in frames})
    assert all(b - a >= 50 for a, b in zip(sends_ticks, sends_ticks[1:], strict=False))
