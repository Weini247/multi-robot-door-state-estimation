"""ARGUS-E01-T03: joint (z, g) forward filter against brute-force path enumeration."""

from __future__ import annotations

from itertools import product

import numpy as np
import pytest
from scipy.linalg import expm

from argus.beliefs.exact_micro import correct_joint, shared_bias_posterior
from argus.beliefs.joint_filter import JointFilter
from argus.dynamics.binary_ctmc import BinaryCTMC

TOL = 1e-12


def generator(rate_01: float, rate_10: float) -> np.ndarray:
    return np.array([[-rate_01, rate_01], [rate_10, -rate_10]])


def enumerate_paths(
    prior: tuple[float, ...],
    rates_z: tuple[float, float],
    rates_g: tuple[float, float],
    q: float,
    ys: tuple[int, ...],
    gaps: tuple[float, ...],
) -> np.ndarray:
    """Oracle: sum over all (z, g) state paths, transitions from scipy expm.

    Measurement k is taken at time t_k; gaps[k-1] is the interval before it.
    """
    weights = np.zeros(4)
    for path in product(range(4), repeat=len(ys)):
        weight = prior[path[0]]
        for k, state in enumerate(path):
            z, g = divmod(state, 2)
            if k:
                zp, gp = divmod(path[k - 1], 2)
                pz = expm(generator(*rates_z) * gaps[k - 1])
                pg = expm(generator(*rates_g) * gaps[k - 1])
                weight *= pz[zp, z] * pg[gp, g]
            weight *= (1 - q) if ys[k] == (z ^ g) else q
        weights[path[-1]] += weight
    return weights / weights.sum()


def run_filter(
    prior: tuple[float, float, float, float],
    rates_z: tuple[float, float],
    rates_g: tuple[float, float],
    q: float,
    ys: tuple[int, ...],
    gaps: tuple[float, ...],
) -> JointFilter:
    filt = JointFilter(BinaryCTMC(*rates_z), BinaryCTMC(*rates_g), q, prior)
    for k, y in enumerate(ys):
        if k:
            filt.predict(gaps[k - 1])
        filt.correct(y)
    return filt


CASES = [
    ((0.45, 0.05, 0.45, 0.05), (0.2, 0.3), (0.01, 0.09), 0.05, (0.7, 0.7)),
    ((0.25, 0.25, 0.25, 0.25), (1.0, 0.1), (0.5, 0.5), 0.2, (0.1, 3.0)),
    ((0.6, 0.0, 0.3, 0.1), (0.0, 0.0), (0.0, 0.0), 0.1, (1.0, 1.0)),
    ((0.1, 0.2, 0.3, 0.4), (0.02, 0.0), (0.005, 0.045), 0.01, (10.0, 0.0)),
]


@pytest.mark.parametrize(("prior", "rates_z", "rates_g", "q", "gaps"), CASES)
def test_forward_posterior_matches_path_enumeration(
    prior: tuple[float, float, float, float],
    rates_z: tuple[float, float],
    rates_g: tuple[float, float],
    q: float,
    gaps: tuple[float, float],
) -> None:
    for ys in product((0, 1), repeat=3):
        oracle = enumerate_paths(prior, rates_z, rates_g, q, ys, gaps)
        filt = run_filter(prior, rates_z, rates_g, q, ys, gaps)
        np.testing.assert_allclose(filt.joint, oracle, atol=TOL, rtol=0)


def test_mass_stays_one_and_entries_valid() -> None:
    filt = JointFilter.from_independent_prior(
        BinaryCTMC(0.01, 0.02), BinaryCTMC(0.005, 0.045), 0.05, 1 / 3, 0.1
    )
    for step in range(300):
        filt.predict(0.1 * (step % 7))
        filt.correct(int(step % 3 == 0))
        assert sum(filt.joint) == pytest.approx(1.0, abs=TOL)
        assert all(0.0 <= value <= 1.0 for value in filt.joint)


def test_static_limit_equals_shared_bias_micro_case() -> None:
    filt = JointFilter.from_independent_prior(
        BinaryCTMC(0.0, 0.0), BinaryCTMC(0.0, 0.0), 0.05, 0.5, 0.1
    )
    filt.correct(1)
    filt.predict(5.0)
    filt.correct(1)
    assert filt.marginal_open() == pytest.approx(325 / 362, abs=1e-14)
    assert filt.marginal_open() == pytest.approx(
        shared_bias_posterior(0.5, 0.1, 0.05, [1, 1]), abs=1e-14
    )


def test_bias_belief_is_not_reset_after_measurement() -> None:
    filt = JointFilter.from_independent_prior(
        BinaryCTMC(0.0, 0.0), BinaryCTMC(0.0, 0.0), 0.05, 0.5, 0.1
    )
    filt.correct(1)
    joint = filt.joint
    # Posterior couples z and g: the joint is no longer a product of its marginals.
    p_open, p_bias = filt.marginal_open(), filt.marginal_bias()
    assert joint[3] != pytest.approx(p_open * p_bias, abs=1e-6)
    # The wrong "marginalize g and restart from its prior" update gives another answer.
    restarted = correct_joint(
        ((1 - p_open) * 0.9, (1 - p_open) * 0.1, p_open * 0.9, p_open * 0.1), 1, 0.05
    )
    filt.correct(1)
    assert filt.marginal_open() == pytest.approx(325 / 362, abs=1e-14)
    assert restarted[2] + restarted[3] > filt.marginal_open() + 0.05


def test_long_horizon_forgets_to_stationary_product() -> None:
    filt = JointFilter.from_independent_prior(
        BinaryCTMC(0.2, 0.3), BinaryCTMC(0.1, 0.9), 0.05, 0.5, 0.1
    )
    filt.correct(1)
    filt.correct(1)
    filt.predict(1000.0)
    np.testing.assert_allclose(filt.joint, (0.6 * 0.9, 0.6 * 0.1, 0.4 * 0.9, 0.4 * 0.1), atol=TOL)


def test_zero_interval_prediction_is_identity() -> None:
    filt = JointFilter(BinaryCTMC(0.2, 0.3), BinaryCTMC(0.1, 0.9), 0.05, (0.1, 0.2, 0.3, 0.4))
    filt.predict(0.0)
    np.testing.assert_allclose(filt.joint, (0.1, 0.2, 0.3, 0.4), atol=1e-15)


@pytest.mark.parametrize(
    "initial",
    [(0.5, 0.5, 0.0), (0.6, 0.6, 0.0, 0.0), (1.2, -0.2, 0.0, 0.0), (float("nan"), 0, 0, 1)],
)
def test_invalid_initial_belief_rejected(initial: tuple[float, ...]) -> None:
    with pytest.raises(ValueError):
        JointFilter(BinaryCTMC(0.1, 0.1), BinaryCTMC(0.1, 0.1), 0.05, initial)


def test_invalid_error_and_time_rejected() -> None:
    with pytest.raises(ValueError):
        JointFilter(BinaryCTMC(0.1, 0.1), BinaryCTMC(0.1, 0.1), 1.5, (0.25,) * 4)
    filt = JointFilter(BinaryCTMC(0.1, 0.1), BinaryCTMC(0.1, 0.1), 0.05, (0.25,) * 4)
    with pytest.raises(ValueError):
        filt.predict(-1.0)
    with pytest.raises(ValueError):
        filt.correct(2)
