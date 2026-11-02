"""Exact forward filter over the joint latent state (z, g) (M3, ARGUS-E01-T03).

Both components evolve as independent binary CTMCs, so the transition over an
interval is the Kronecker product P_z(dt) (x) P_g(dt). Measurements follow
y = z XOR g XOR eps (see exact_micro). After a measurement z and g are coupled
in the posterior; this filter keeps that coupling. Marginalizing g and
restarting it from its prior would be wrong and is deliberately impossible here.

This is the exact micro reference, not the scalable filter: the state space is
enumerated, and the nominal model is the only information the filter uses.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from argus.beliefs.exact_micro import JOINT_ORDER, Joint, correct_joint, joint_prior
from argus.dynamics.binary_ctmc import BinaryCTMC


class JointFilter:
    """Forward filter b(z, g) in JOINT_ORDER with predict/correct steps."""

    __slots__ = ("_joint", "_private_error", "_z_model", "_g_model")

    def __init__(
        self,
        z_model: BinaryCTMC,
        g_model: BinaryCTMC,
        private_error: float,
        initial: Sequence[float],
    ) -> None:
        if len(initial) != 4:
            raise ValueError("initial joint belief needs exactly four entries")
        for value in initial:
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"joint entries must be finite and >= 0, got {value!r}")
        if not math.isclose(math.fsum(initial), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("initial joint belief must sum to 1 within 1e-12")
        if not math.isfinite(private_error) or not 0.0 <= private_error <= 1.0:
            raise ValueError(f"private_error must be a probability, got {private_error!r}")
        self._z_model = z_model
        self._g_model = g_model
        self._private_error = private_error
        a, b, c, d = initial
        self._joint: Joint = (a, b, c, d)

    @classmethod
    def from_independent_prior(
        cls,
        z_model: BinaryCTMC,
        g_model: BinaryCTMC,
        private_error: float,
        prior_open: float,
        bias_probability: float,
    ) -> JointFilter:
        """Start from P(z=1) = prior_open and an independent P(g=1) = bias_probability."""
        return cls(z_model, g_model, private_error, joint_prior(prior_open, bias_probability))

    @property
    def joint(self) -> Joint:
        """Copy of the current joint belief in JOINT_ORDER."""
        return self._joint

    def predict(self, dt_seconds: float) -> None:
        """Propagate the joint belief over dt_seconds with P_z (x) P_g."""
        pz = self._z_model.transition(dt_seconds)
        pg = self._g_model.transition(dt_seconds)
        old = self._joint
        a, b, c, d = (
            math.fsum(old[2 * z + g] * pz[z][z_next] * pg[g][g_next] for z, g in JOINT_ORDER)
            for z_next, g_next in JOINT_ORDER
        )
        # No renormalization: both factors are row-stochastic, so mass is kept up to
        # rounding, and a mass test can still detect a wrong transition.
        self._joint = (a, b, c, d)

    def correct(self, y: int) -> None:
        """Condition on one distinct measurement y at the current time."""
        self._joint = correct_joint(self._joint, y, self._private_error)

    def marginal_open(self) -> float:
        """P(z=1)."""
        return math.fsum((self._joint[2], self._joint[3]))

    def marginal_bias(self) -> float:
        """P(g=1)."""
        return math.fsum((self._joint[1], self._joint[3]))
