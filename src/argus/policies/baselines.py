"""Simple control methods and the method registry (E07-T02).

Every method is a fixed combination of a belief model, a movement rule and a
communication rule, declared in METHOD_SPECS with its component list and the
parameters that may be tuned (on validation buildings only, see E07-T06). All
methods share the sensor, the channel, the byte budget and the navigation
controller; only the declared components differ.

Controls (no method sees other agents' beliefs or any ground truth):

* no_comm       frontier exploration, dynamic belief with IDs, never sends
* naive_gossip  frontier, naive belief (no IDs, no ageing), periodically sends all
                held evidence to every teammate
* ttl_ids       frontier, TTL belief with IDs (no dynamics), periodic own sends
* dynamics_ids  frontier, dynamic belief with IDs, periodic own sends
* age_revisit   revisits the stalest sighting older than a fixed age, else frontier;
                dynamic belief with IDs, no sends
* cross_rank    published low-bandwidth baseline (ARGUS adaptation of Cross-rank CR-ESR_p,
                E07-T05): shares positions only
* repair        the VOI repair policy (E06-T02), dynamic belief with IDs
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from argus.beliefs.fixed_lag import FixedLagBelief, NominalPassageModel
from argus.beliefs.variants import IndependentLogOddsBelief
from argus.core.types import AgentAction, AgentView, EvidenceId, Observation, Send
from argus.policies.crossrank_adapter import CrossRankPolicy
from argus.policies.frontier import FrontierPolicy
from argus.policies.repair import RepairPolicy, estimate_frame_bytes
from argus.policies.simple import WaitPolicy
from argus.policies.voi import Losses


@dataclass(frozen=True, slots=True)
class MethodContext:
    """Public, method-independent episode facts a factory may use."""

    episode_id: str
    team: tuple[str, ...]
    nominal: NominalPassageModel
    marginal_error: float
    losses: Losses
    tick_seconds: float
    window_ticks: int
    capacity: int
    delivery_probability: float
    params: tuple[tuple[str, float], ...] = ()  # validation-tuned values (E07-T06)

    def param(self, name: str, default: float) -> float:
        return dict(self.params).get(name, default)


@dataclass(slots=True)
class TimedSendPolicy:
    """Wraps a movement policy; every period sends evidence to all teammates.

    own_only=True sends only the agent's own readings (timed sends); own_only=False
    forwards everything it holds (gossip). Each item goes to each teammate at most
    once, within the remaining byte budget.
    """

    movement: Any
    team: tuple[str, ...]
    period_ticks: int = 50
    own_only: bool = True
    _memory: dict[EvidenceId, Observation] = field(default_factory=dict, repr=False)
    _sent: set[tuple[EvidenceId, str]] = field(default_factory=set, repr=False)
    _last: int = field(default=-(10**9), repr=False)

    def act(self, view: AgentView, random_key: str) -> AgentAction:
        for obs in (*view.new_own_evidence, *view.inbox):
            self._memory[obs.evidence_id] = obs
        base = self.movement.act(view, random_key)
        sends: list[Send] = []
        if view.tick - self._last >= self.period_ticks:
            self._last = view.tick
            budget = view.remaining_bytes
            for eid in sorted(self._memory):
                obs = self._memory[eid]
                if self.own_only and eid.origin != view.robot_id:
                    continue
                for recipient in self.team:
                    if recipient in (view.robot_id, eid.origin) or (eid, recipient) in self._sent:
                        continue
                    size = estimate_frame_bytes(obs)
                    if size > budget:
                        continue
                    sends.append(Send(recipient, eid))
                    self._sent.add((eid, recipient))
                    budget -= size
        return AgentAction(
            base.kind,
            base.duration_ticks,
            base.v,
            base.omega,
            sends=tuple(sends),
            reserve=base.reserve,
            reserve_ttl_ticks=base.reserve_ttl_ticks,
        )


@dataclass(slots=True)
class AgeRevisitPolicy:
    """Go back to the stalest sighting once it is older than max_age_ticks."""

    max_age_ticks: int = 300
    frontier: FrontierPolicy = field(default_factory=FrontierPolicy)
    arrival_m: float = 0.3

    def act(self, view: AgentView, random_key: str) -> AgentAction:
        stale = [s for s in view.sightings if view.tick - s.observed_tick > self.max_age_ticks]
        if not stale:
            return self.frontier.act(view, random_key)
        target = min(stale, key=lambda s: (s.observed_tick, s.entity))
        controller = self.frontier.controller
        if math.dist(view.pose[:2], target.observer_pose[:2]) > self.arrival_m:
            v, omega = controller.steer(view.pose, target.observer_pose[:2])
        else:
            error = (target.observer_pose[2] - view.pose[2] + math.pi) % (2 * math.pi) - math.pi
            v, omega = (
                0.0,
                max(-controller.omega_max, min(controller.omega_max, controller.turn_gain * error)),
            )
        return AgentAction(
            "move" if (v, omega) != (0.0, 0.0) else "wait", self.frontier.decision_ticks, v, omega
        )


BeliefFactory = Callable[[MethodContext], Any]
PolicyFactory = Callable[[MethodContext], Any]


def _dynamic(ctx: MethodContext) -> FixedLagBelief:
    return FixedLagBelief(
        ctx.episode_id,
        {},
        ctx.tick_seconds,
        ctx.window_ticks,
        ctx.capacity,
        default_model=ctx.nominal,
    )


def _naive(ctx: MethodContext) -> IndependentLogOddsBelief:
    return IndependentLogOddsBelief(
        ctx.episode_id, ctx.nominal.prior_open, ctx.marginal_error, dedupe=False, ttl_ticks=None
    )


def _ttl(ctx: MethodContext) -> IndependentLogOddsBelief:
    ttl_ticks = int(ctx.param("ttl_ticks", 300))
    return IndependentLogOddsBelief(
        ctx.episode_id, ctx.nominal.prior_open, ctx.marginal_error, dedupe=True, ttl_ticks=ttl_ticks
    )


def _period(ctx: MethodContext) -> int:
    return int(ctx.param("send_period_ticks", 50))


def _repair(ctx: MethodContext) -> RepairPolicy:
    return RepairPolicy(
        team=ctx.team,
        nominal=ctx.nominal,
        losses=ctx.losses,
        tick_seconds=ctx.tick_seconds,
        window_ticks=ctx.window_ticks,
        delivery_probability=ctx.delivery_probability,
        horizon_s=ctx.param("horizon_s", 5.0),
    )


@dataclass(frozen=True, slots=True)
class MethodSpec:
    method_id: str
    components: tuple[str, ...]
    belief: BeliefFactory
    policy: PolicyFactory
    tunable: tuple[str, ...]  # tuned on validation only (E07-T06)
    role: str  # "candidate", "control" or "diagnostic"


METHOD_SPECS: dict[str, MethodSpec] = {
    spec.method_id: spec
    for spec in (
        MethodSpec(
            "no_comm",
            ("frontier-movement", "fixed-lag-joint-belief", "no-communication"),
            _dynamic,
            lambda ctx: FrontierPolicy(),
            (),
            "control",
        ),
        MethodSpec(
            "naive_gossip",
            ("frontier-movement", "naive-independent-belief", "gossip-all-evidence"),
            _naive,
            lambda ctx: TimedSendPolicy(FrontierPolicy(), ctx.team, _period(ctx), own_only=False),
            ("send_period_ticks",),
            "control",
        ),
        MethodSpec(
            "ttl_ids",
            ("frontier-movement", "ttl-id-belief", "timed-own-sends"),
            _ttl,
            lambda ctx: TimedSendPolicy(FrontierPolicy(), ctx.team, _period(ctx)),
            ("ttl_ticks", "send_period_ticks"),
            "control",
        ),
        MethodSpec(
            "dynamics_ids",
            ("frontier-movement", "fixed-lag-joint-belief", "timed-own-sends"),
            _dynamic,
            lambda ctx: TimedSendPolicy(FrontierPolicy(), ctx.team, _period(ctx)),
            ("send_period_ticks",),
            "control",
        ),
        MethodSpec(
            "age_revisit",
            ("age-revisit-movement", "fixed-lag-joint-belief", "no-communication"),
            _dynamic,
            lambda ctx: AgeRevisitPolicy(int(ctx.param("max_age_ticks", 300))),
            ("max_age_ticks",),
            "control",
        ),
        MethodSpec(
            "cross_rank",
            ("cross-rank-movement", "fixed-lag-joint-belief", "position-sharing"),
            _dynamic,
            lambda ctx: CrossRankPolicy(ctx.param("m_r", 1.5), ctx.param("position_filter_m", 0.5)),
            ("m_r", "position_filter_m"),
            "control",
        ),
        MethodSpec(
            "repair",
            ("voi-repair-movement", "fixed-lag-joint-belief", "voi-sends"),
            _dynamic,
            _repair,
            ("horizon_s",),
            "candidate",
        ),
        MethodSpec(
            "frontier",
            ("frontier-movement", "fixed-lag-joint-belief", "reservations-only"),
            _dynamic,
            lambda ctx: FrontierPolicy(),
            (),
            "diagnostic",
        ),
        MethodSpec(
            "wait",
            ("no-movement", "fixed-lag-joint-belief", "no-communication"),
            _dynamic,
            lambda ctx: WaitPolicy(),
            (),
            "diagnostic",
        ),
    )
}
