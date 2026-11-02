"""ARGUS-E06-T02: re-observation and send candidates under a nominal receiver model."""

from __future__ import annotations

import ast
import dataclasses
import math
from pathlib import Path

import pytest

from argus.beliefs.exact_micro import correct_joint, joint_prior
from argus.beliefs.fixed_lag import NominalPassageModel
from argus.communication.messages import Message, originate
from argus.core.types import AgentView, BeliefState, EvidenceId, Observation, Sighting
from argus.dynamics.binary_ctmc import BinaryCTMC
from argus.experiments.runner import EpisodeConfig, ExperimentRunner
from argus.policies.repair import FRAME_OVERHEAD_BYTES, RepairPolicy, estimate_frame_bytes
from argus.policies.voi import Losses
from tests.e05.test_t03 import scenario

MODEL = NominalPassageModel(BinaryCTMC(0.05, 0.1), BinaryCTMC(0.005, 0.045), 0.05, 1 / 3, 0.1)
SRC = Path(__file__).resolve().parents[2] / "src" / "argus"


def policy(**kw: object) -> RepairPolicy:
    values: dict[str, object] = dict(
        team=("r0", "r1", "r2"),
        nominal=MODEL,
        losses=Losses(),
        tick_seconds=0.1,
        window_ticks=300,
        delivery_probability=0.9,
    )
    values.update(kw)
    return RepairPolicy(**values)  # type: ignore[arg-type]


def reading(origin: str, seq: int, tick: int, value: int) -> Observation:
    return Observation(EvidenceId("ep", origin, seq), tick, "door", value, (2.0, 2.0, 0.0), "m")


def view(
    tick: int,
    pose: tuple[float, float, float],
    own: tuple[Observation, ...] = (),
    p_joint: tuple[float, float, float, float] | None = None,
    remaining: float = 1e6,
    sighting_pose: tuple[float, float, float] = (2.0, 2.0, 0.0),
) -> AgentView:
    joint = p_joint or joint_prior(1 / 3, 0.1)
    free = tuple((i, j) for i in range(1, 40) for j in range(1, 14))
    return AgentView(
        "r0",
        tick,
        pose,
        BeliefState(tick, (("door", joint[2] + joint[3]),)),
        0.25,
        free,
        (),
        (),
        (),
        remaining,
        tuple(o.evidence_id for o in own),
        False,
        sightings=(
            Sighting("door", sighting_pose, max((o.observed_tick for o in own), default=0), "r0"),
        ),
        new_own_evidence=own,
        joint_beliefs=(("door", joint),),
    )


def test_policy_imports_no_remote_state_or_simulator() -> None:
    tree = ast.parse((SRC / "policies/repair.py").read_text(encoding="utf-8"))
    modules = {n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert not any(
        m.startswith(
            (
                "argus.world",
                "argus.agents",
                "argus.experiments",
                "argus.communication",
                "argus.sensors",
            )
        )
        for m in modules
    )
    assert {f.name for f in dataclasses.fields(AgentView)}.isdisjoint({"remote", "partner_beliefs"})


def test_receiver_model_uses_only_what_the_sender_holds() -> None:
    p = policy()
    p.act(view(10, (2.0, 2.0, 0.0), own=(reading("r0", 1, 5, 1),)), "k")
    prior_like = p.receiver_joint("r1", "door", 10)
    # The recipient's own evidence, once seen by the sender, enters the receiver model.
    inbox = (reading("r1", 7, 8, 0),)
    v = dataclasses.replace(view(12, (2.0, 2.0, 0.0)), inbox=inbox)
    p.act(v, "k")
    assert p.receiver_joint("r1", "door", 12) != prior_like
    assert p.receiver_joint("r2", "door", 12)[2] + p.receiver_joint("r2", "door", 12)[
        3
    ] == pytest.approx(prior_like[2] + prior_like[3], abs=0.05)


def test_sending_new_evidence_has_value_and_repeats_decay() -> None:
    own = (reading("r0", 1, 9, 1), reading("r0", 2, 9, 1))
    # The sender's own belief after its two readings (as its agent would report it).
    mine = correct_joint(correct_joint(joint_prior(1 / 3, 0.1), 1, 0.05), 1, 0.05)
    p = policy(max_sends_per_decision=4, losses=Losses(1.0, 1.0))
    first = p.act(view(10, (2.0, 2.0, 0.0), own=own, p_joint=mine), "k")
    assert first.sends and all(s.recipient in ("r1", "r2") for s in first.sends)
    assert all(s.assumed_receiver_open is not None for s in first.sends)
    values_before = {str(c.send): c.value for c in p.last_candidates if c.kind == "send"}
    p.act(view(11, (2.0, 2.0, 0.0), p_joint=mine), "k")
    for c in p.last_candidates:
        if c.kind == "send" and c.send in first.sends:
            assert c.value < values_before[str(dataclasses.replace(c.send))] + 1e-12


def test_additional_observations_cost_time() -> None:
    p = policy()  # 5 s horizon: a 2 m detour (4.5 s) is still feasible
    near = p._observe_candidates(view(0, (2.5, 2.0, 0.0)))[0]  # noqa: SLF001
    far = p._observe_candidates(view(0, (4.0, 2.0, 0.0)))[0]  # noqa: SLF001
    assert far.value < near.value
    gap = math.dist((4.0, 2.0), (2.0, 2.0)) - 0.5  # path difference (both on the grid)
    assert near.value - far.value >= 0.01 * gap / 0.5  # at least the time cost of the detour


def test_send_respects_byte_cap_with_a_conservative_size_bound() -> None:
    obs = reading("r0", 1, 5, 1)
    real = originate(obs, "r1", 10).size_bytes
    forwarded = Message("r0", "longer-robot-name", 10, 3, obs).size_bytes
    assert estimate_frame_bytes(obs) >= max(real, forwarded)
    assert estimate_frame_bytes(obs) - FRAME_OVERHEAD_BYTES < real
    tight = policy().act(view(10, (2.0, 2.0, 0.0), own=(obs,), remaining=real - 1), "k")
    assert tight.sends == ()


def test_falls_back_to_frontier_without_positive_value() -> None:
    certain = correct_joint(correct_joint(joint_prior(0.999, 0.0), 1, 0.05), 1, 0.05)
    p = policy(team=("r0",))
    action = p.act(view(0, (8.0, 2.0, 0.0), p_joint=certain), "k")
    assert all(c.value <= 0 for c in p.last_candidates)
    assert action.sends == ()
    assert action == dataclasses.replace(
        p.frontier.act(view(0, (8.0, 2.0, 0.0), p_joint=certain), "k"), sends=()
    )


def runner_config() -> EpisodeConfig:
    # Spec sensitivity 1:1: with the pilot 4:1 losses and shared bias the receiver can
    # hardly reach the 0.8 threshold, so sends almost never change a decision.
    config = EpisodeConfig("ep-1", "building-a", scenario(), "repair", 1, 2, 3, 4, 5)
    return dataclasses.replace(config, loss_false_open=1.0, loss_false_closed=1.0)


def test_shared_bias_caps_confidence_below_the_pilot_threshold() -> None:
    joint = joint_prior(1 / 3, 0.1)
    for _ in range(20):
        joint = correct_joint(joint, 1, 0.05)
    assert joint[2] + joint[3] < 0.82  # g may flip every reading: certainty is impossible


def test_runner_logs_receiver_model_errors_and_never_exceeds_the_byte_cap() -> None:
    runner = ExperimentRunner()
    result = runner.run(runner_config())
    assert result.status == "ok"
    log = runner.recipient_model_log
    assert log, "repair agents sent evidence"
    assert all(0 <= r.assumed_open <= 1 and 0 <= r.true_open <= 1 for r in log)
    assert any(abs(r.error) > 1e-6 for r in log)  # the nominal model is not the truth
    rejects = runner.last_channel.stats.rejects
    assert not any(reason in ("budget_rejected", "frame_too_large") for _, reason in rejects)
    obs_frames = [a for a in runner.last_channel.attempts if isinstance(a.message, Message)]
    assert obs_frames and all(
        a.size <= estimate_frame_bytes(a.message.observation) for a in obs_frames
    )
