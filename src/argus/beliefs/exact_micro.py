"""Exact finite-state Bayes correction and the static shared-bias micro case (M2/M3).

Joint latent state x = (z, g) in the fixed order (0,0), (0,1), (1,0), (1,1):
z = 1 means "passable", g = 1 a shared sensor error. A binary measurement is

    y = z XOR g XOR eps,   g ~ Bernoulli(rho),   eps ~ Bernoulli(q) i.i.d.,

so measurements that share g are conditionally independent given (z, g) only.
Keeping the joint state is what makes repeated measurements exact; the
marginal error rate is e = q + rho - 2 q rho.

Inputs must be distinct evidence: identity deduplication happens upstream.
Zero marginal evidence probability raises ImpossibleEvidence instead of
silently returning a prior. Normalization runs in log space with math.fsum.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

JOINT_ORDER: tuple[tuple[int, int], ...] = ((0, 0), (0, 1), (1, 0), (1, 1))
Joint = tuple[float, float, float, float]


class ImpossibleEvidence(ValueError):
    """The evidence has probability zero under the specified model and prior."""


def _probability(name: str, value: float) -> float:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a finite probability in [0, 1], got {value!r}")
    return value


def bayes_update(prior: Sequence[float], likelihood: Sequence[float]) -> tuple[float, ...]:
    """Posterior proportional to prior * likelihood over a finite state space."""
    if not prior or len(prior) != len(likelihood):
        raise ValueError("prior and likelihood must be nonempty and of equal length")
    for value in (*prior, *likelihood):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"entries must be finite and >= 0, got {value!r}")
    if not math.isclose(math.fsum(prior), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("prior must sum to 1 within 1e-12")
    logs = [
        -math.inf if p == 0 or lik == 0 else math.log(p) + math.log(lik)
        for p, lik in zip(prior, likelihood, strict=True)
    ]
    peak = max(logs)
    if peak == -math.inf:
        raise ImpossibleEvidence("marginal probability of the evidence is zero")
    scaled = [math.exp(w - peak) for w in logs]
    total = math.fsum(scaled)
    return tuple(w / total for w in scaled)


def joint_prior(prior_open: float, bias_probability: float) -> Joint:
    """Independent prior over (z, g): P(z=1) = prior_open, P(g=1) = bias_probability."""
    p = _probability("prior_open", prior_open)
    rho = _probability("bias_probability", bias_probability)
    return ((1 - p) * (1 - rho), (1 - p) * rho, p * (1 - rho), p * rho)


def measurement_likelihood(y: int, private_error: float) -> Joint:
    """P(y | z, g) in JOINT_ORDER for y = z XOR g XOR eps."""
    if y not in (0, 1):
        raise ValueError(f"y must be 0 or 1, got {y!r}")
    q = _probability("private_error", private_error)
    a, b, c, d = (1 - q if y == (z ^ g) else q for z, g in JOINT_ORDER)
    return (a, b, c, d)


def correct_joint(joint: Sequence[float], y: int, private_error: float) -> Joint:
    """Condition the joint (z, g) belief on one distinct measurement y."""
    if len(joint) != 4:
        raise ValueError("joint belief over (z, g) needs exactly four entries")
    a, b, c, d = bayes_update(joint, measurement_likelihood(y, private_error))
    return (a, b, c, d)


def marginal_open(joint: Sequence[float]) -> float:
    """P(z=1) of a joint (z, g) belief."""
    return math.fsum((joint[2], joint[3]))


def shared_bias_posterior(
    prior_open: float,
    bias_probability: float,
    private_error: float,
    observations: Iterable[int],
) -> float:
    """P(z=1 | y_1..y_m) when one shared g affects every measurement of the group."""
    joint: Sequence[float] = joint_prior(prior_open, bias_probability)
    _probability("private_error", private_error)
    for y in observations:
        joint = correct_joint(joint, y, private_error)
    return marginal_open(joint)
