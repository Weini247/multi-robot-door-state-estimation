"""Exact propagation of a binary continuous-time Markov chain (M1, ARGUS-E01-T01).

State z=1 means "passable". With generator Q = [[-r01, r01], [r10, -r10]] and
k = r01 + r10 > 0, the marginal p(t) = P(z=1) solves dp/dt = r01 (1-p) - r10 p:

    p(t + d) = pi + (p(t) - pi) exp(-k d),   pi = r01 / k.

It is evaluated as p + (pi - p) * (-expm1(-k d)) so that tiny steps keep their
increment instead of cancelling to zero. For k = 0 the state never changes.

Scope: values are exact at the requested times. A later grid-based world
simulation samples the CTMC skeleton on ticks; it does not resolve exact jump
times between ticks. No IO, no global state, no world access.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

Matrix2 = tuple[tuple[float, float], tuple[float, float]]


def _finite_nonnegative(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and >= 0, got {value!r}")


@dataclass(frozen=True, slots=True)
class BinaryCTMC:
    """Two-state chain with transition rates in s^-1.

    rate_01: rate of 0 -> 1 (closed -> passable)
    rate_10: rate of 1 -> 0 (passable -> closed)
    """

    rate_01: float
    rate_10: float

    def __post_init__(self) -> None:
        _finite_nonnegative("rate_01", self.rate_01)
        _finite_nonnegative("rate_10", self.rate_10)
        if not math.isfinite(self.rate_01 + self.rate_10):
            raise ValueError("rate_01 + rate_10 exceeds the float range")

    @property
    def total_rate(self) -> float:
        """k = rate_01 + rate_10 in s^-1."""
        return self.rate_01 + self.rate_10

    @property
    def stationary(self) -> float | None:
        """pi = P(z=1) in the stationary distribution; None when both rates are 0."""
        k = self.total_rate
        return self.rate_01 / k if k > 0 else None

    def propagate(self, p: float, dt_seconds: float) -> float:
        """P(z=1) after dt_seconds, starting from P(z=1) = p."""
        if not math.isfinite(p) or not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be a finite probability in [0, 1], got {p!r}")
        _finite_nonnegative("dt_seconds", dt_seconds)
        k = self.total_rate
        if k == 0 or dt_seconds == 0:
            return p
        pi = self.rate_01 / k
        # k*dt may overflow to +inf; expm1(-inf) = -1 gives the exact limit pi.
        weight = -math.expm1(-k * dt_seconds)
        # Rounding guard only: valid inputs already yield a value in [0, 1].
        return min(1.0, max(0.0, p + (pi - p) * weight))

    def transition(self, dt_seconds: float) -> Matrix2:
        """Row-stochastic matrix ((P00, P01), (P10, P11)) for an interval dt_seconds."""
        p01 = self.propagate(0.0, dt_seconds)
        p11 = self.propagate(1.0, dt_seconds)
        return ((1.0 - p01, p01), (1.0 - p11, p11))
