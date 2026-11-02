"""Binary decision risk and exact one-step value of information (M5, E06-T01).

Decision d in {0, 1} about one passage with P(open) = p and losses
c10 (decide open, truly closed) and c01 (decide closed, truly open):

    R(p) = min(c10 (1 - p), c01 p),   d = 1 iff p >= c10 / (c10 + c01).

An information action delivers m distinct measurements of the entity at time
t_y (seconds from now) and is compared with doing nothing on the same decision
horizon T >= t_y:

    VOI(a; b) = R(Predict(b, T)) - sum_y P(y | a, b) R(Predict(b^{a,y}, T - t_y)) - c(a),

where b^{a,y} is b predicted to t_y and then corrected with y. Each interval is
predicted exactly once. Beliefs are the exact joint (z, g) of the nominal model,
so repeated measurements under a shared bias are valued correctly. Costs are
c(a) = w_d distance + w_t duration + w_s measurements. Evidence the decider
already holds carries no information: its gain is exactly 0.

This is the exact myopic reference for one entity, not a Dec-POMDP solver.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from itertools import product

from argus.beliefs.exact_micro import JOINT_ORDER, Joint, correct_joint, marginal_open
from argus.beliefs.fixed_lag import NominalPassageModel
from argus.beliefs.joint_filter import JointFilter
from argus.core.types import EvidenceId


@dataclass(frozen=True, slots=True)
class Losses:
    false_open: float = 4.0  # c10
    false_closed: float = 1.0  # c01

    def __post_init__(self) -> None:
        for value in (self.false_open, self.false_closed):
            if not math.isfinite(value) or value < 0:
                raise ValueError("losses must be finite and >= 0")
        if self.false_open + self.false_closed == 0:
            raise ValueError("at least one loss must be positive")

    @property
    def threshold(self) -> float:
        return self.false_open / (self.false_open + self.false_closed)


def binary_risk(p: float, losses: Losses) -> float:
    """Expected loss of the best decision at P(open) = p."""
    if not math.isfinite(p) or not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be a probability, got {p!r}")
    return min(losses.false_open * (1 - p), losses.false_closed * p)


def decide(p: float, losses: Losses) -> int:
    """Bayes decision; ties at the threshold resolve deterministically to 1."""
    return int(p >= losses.threshold)


@dataclass(frozen=True, slots=True)
class CostWeights:
    """Pre-registered unit costs in loss points (pilot defaults from the spec)."""

    per_metre: float = 0.05
    per_second: float = 0.01
    per_measurement: float = 0.01

    def cost(self, distance_m: float, duration_s: float, measurements: int) -> float:
        for value in (distance_m, duration_s):
            if not math.isfinite(value) or value < 0:
                raise ValueError("distance and duration must be finite and >= 0")
        if measurements < 0:
            raise ValueError("measurements must be >= 0")
        return (
            self.per_metre * distance_m
            + self.per_second * duration_s
            + self.per_measurement * measurements
        )


@dataclass(frozen=True, slots=True)
class ObserveAction:
    """Go and measure: m distinct readings of one entity at t_y seconds from now."""

    distance_m: float
    duration_s: float
    measurements: int = 1

    def __post_init__(self) -> None:
        if not math.isfinite(self.duration_s) or self.duration_s < 0:
            raise ValueError("duration_s must be finite and >= 0")
        if type(self.measurements) is not int or self.measurements < 1:
            raise ValueError("an observation action takes at least one measurement")


@dataclass(frozen=True, slots=True)
class ValueEstimate:
    risk_without: float
    expected_risk_with: float
    cost: float

    @property
    def gain(self) -> float:
        return self.risk_without - self.expected_risk_with

    @property
    def voi(self) -> float:
        return self.gain - self.cost


def _predict(model: NominalPassageModel, joint: Sequence[float], seconds: float) -> Joint:
    filt = JointFilter(model.z_model, model.g_model, model.private_error, joint)
    filt.predict(seconds)
    return filt.joint


def _outcome_probability(joint: Sequence[float], ys: Sequence[int], q: float) -> float:
    """P(y_1..y_m) under the joint (z, g); readings share g, private errors independent."""
    total = 0.0
    for weight, (z, g) in zip(joint, JOINT_ORDER, strict=True):
        likelihood = 1.0
        for y in ys:
            likelihood *= (1 - q) if y == (z ^ g) else q
        total += weight * likelihood
    return total


def expected_value(
    action: ObserveAction,
    joint: Sequence[float],
    model: NominalPassageModel,
    horizon_s: float,
    losses: Losses,
    costs: CostWeights,
) -> ValueEstimate:
    """Exact one-step VOI of an observation action against waiting, on the same horizon."""
    if not math.isfinite(horizon_s) or horizon_s < action.duration_s:
        raise ValueError("the decision horizon must not end before the observation")
    q = model.private_error
    without = binary_risk(marginal_open(_predict(model, joint, horizon_s)), losses)
    at_observation = _predict(model, joint, action.duration_s)
    expected = 0.0
    for ys in product((0, 1), repeat=action.measurements):
        probability = _outcome_probability(at_observation, ys, q)
        if probability == 0.0:
            continue
        posterior: Sequence[float] = at_observation
        for y in ys:
            posterior = correct_joint(posterior, y, q)
        decided = _predict(model, posterior, horizon_s - action.duration_s)
        expected += probability * binary_risk(marginal_open(decided), losses)
    cost = costs.cost(action.distance_m, action.duration_s, action.measurements)
    return ValueEstimate(without, expected, cost)


def delivery_value(
    joint: Sequence[float],
    model: NominalPassageModel,
    reading: int,
    evidence_id: EvidenceId,
    known: Collection[EvidenceId],
    horizon_s: float,
    losses: Losses,
) -> float:
    """Risk reduction from receiving one known reading now; 0 for evidence already held."""
    if evidence_id in known:
        return 0.0
    before = binary_risk(marginal_open(_predict(model, joint, horizon_s)), losses)
    corrected = correct_joint(joint, reading, model.private_error)
    after = binary_risk(marginal_open(_predict(model, corrected, horizon_s)), losses)
    return before - after


def send_value(
    receiver_joint: Sequence[float],
    sender_joint: Sequence[float],
    readings: Sequence[int],
    model: NominalPassageModel,
    horizon_s: float,
    losses: Losses,
) -> float:
    """Decision value of delivering distinct readings to a receiver, judged by the sender.

    The receiver decides with its own (nominal) belief, with and without the
    readings; both decisions are scored with the sender's belief about z at the
    horizon. Zero when the receiver's decision does not change. Readings must be
    evidence the receiver does not already hold.
    """
    before = _predict(model, receiver_joint, horizon_s)
    after: Sequence[float] = receiver_joint
    for y in readings:
        after = correct_joint(after, y, model.private_error)
    after = _predict(model, after, horizon_s)
    p_sender = marginal_open(_predict(model, sender_joint, horizon_s))

    def loss(decision: int) -> float:
        if decision == 1:
            return losses.false_open * (1 - p_sender)
        return losses.false_closed * p_sender

    return loss(decide(marginal_open(before), losses)) - loss(decide(marginal_open(after), losses))
