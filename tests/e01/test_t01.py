"""ARGUS-E01-T01: binary CTMC against independent closed-form and matrix oracles."""

from __future__ import annotations

import math
from itertools import product

import numpy as np
import pytest
from scipy.linalg import expm

from argus.dynamics import BinaryCTMC

ATOL = 1e-14


def test_analytic_value_from_spec() -> None:
    # Oracle: pi=.4, k=.5, d=2 -> .4 + (.9-.4) e^-1, written as the closed number.
    expected = 0.4 + 0.5 / math.e
    assert expected == pytest.approx(0.5839397205857212, abs=1e-16)
    assert BinaryCTMC(0.2, 0.3).propagate(0.9, 2.0) == pytest.approx(expected, abs=ATOL)


def test_stationary_value_is_fixed_and_attracting() -> None:
    model = BinaryCTMC(0.2, 0.3)
    assert model.stationary == pytest.approx(0.4, abs=ATOL)
    assert model.propagate(0.4, 7.3) == pytest.approx(0.4, abs=ATOL)
    assert model.propagate(0.9, 1000.0) == pytest.approx(0.4, abs=ATOL)


@pytest.mark.parametrize("p", [0.0, 0.25, 0.73, 1.0])
def test_zero_rates_and_zero_time_are_identity(p: float) -> None:
    assert BinaryCTMC(0.0, 0.0).propagate(p, 100.0) == p
    assert BinaryCTMC(0.2, 0.3).propagate(p, 0.0) == p
    assert BinaryCTMC(0.0, 0.0).stationary is None


def test_absorbing_states() -> None:
    assert BinaryCTMC(0.0, 0.3).propagate(0.0, 9.0) == 0.0
    assert BinaryCTMC(0.3, 0.0).propagate(1.0, 9.0) == 1.0
    # One-sided absorption pulls every start towards the absorbing state.
    assert BinaryCTMC(0.0, 0.3).propagate(1.0, 1e4) == pytest.approx(0.0, abs=ATOL)


def test_semigroup_on_marginals_and_matrices() -> None:
    model = BinaryCTMC(0.2, 0.3)
    stepped = model.propagate(model.propagate(0.9, 0.7), 1.3)
    assert stepped == pytest.approx(model.propagate(0.9, 2.0), abs=ATOL)
    composed = np.array(model.transition(0.7)) @ np.array(model.transition(1.3))
    np.testing.assert_allclose(composed, np.array(model.transition(2.0)), atol=1e-12, rtol=0)


def test_tiny_step_keeps_its_increment() -> None:
    # dp/dt at p=0 equals rate_01, so p(dt) ~ 0.2e-12; naive 1-exp would lose digits.
    value = BinaryCTMC(0.2, 0.3).propagate(0.0, 1e-12)
    assert value > 0.0
    assert value == pytest.approx(0.2e-12, rel=1e-9)


def test_overflowing_exponent_reaches_stationary_limit() -> None:
    model = BinaryCTMC(1e200, 1e200)
    assert model.propagate(0.0, 1e200) == pytest.approx(0.5, abs=ATOL)


@pytest.mark.parametrize("dt", [0.0, 1e-12, 0.1, 2.0, 1e3])
def test_transition_is_row_stochastic(dt: float) -> None:
    matrix = np.array(BinaryCTMC(0.2, 0.3).transition(dt))
    assert (matrix >= 0).all()
    np.testing.assert_allclose(matrix.sum(axis=1), [1.0, 1.0], atol=ATOL, rtol=0)


def test_zero_time_transition_is_identity() -> None:
    assert BinaryCTMC(0.2, 0.3).transition(0.0) == ((1.0, 0.0), (0.0, 1.0))


@pytest.mark.parametrize(
    "rates",
    [(-1.0, 0.0), (0.0, -1e-300), (math.nan, 1.0), (1.0, math.inf), (1e308, 1e308)],
)
def test_invalid_rates_rejected(rates: tuple[float, float]) -> None:
    with pytest.raises(ValueError):
        BinaryCTMC(*rates)


@pytest.mark.parametrize(
    ("p", "dt"),
    [
        (-0.1, 1.0),
        (1.1, 1.0),
        (math.nan, 1.0),
        (math.inf, 1.0),
        (0.5, -1.0),
        (0.5, math.nan),
        (0.5, math.inf),
    ],
)
def test_invalid_probability_or_time_rejected(p: float, dt: float) -> None:
    with pytest.raises(ValueError):
        BinaryCTMC(0.2, 0.3).propagate(p, dt)


def test_matches_independent_matrix_exponential() -> None:
    worst = 0.0
    for a, b, dt in product((0.0, 1e-6, 0.2, 2.0), (0.0, 0.3, 3.0), (0.0, 1e-12, 0.1, 2.0, 100.0)):
        generator = np.array([[-a, a], [b, -b]])
        expected = expm(generator * dt)
        actual = np.array(BinaryCTMC(a, b).transition(dt))
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-10)
        worst = max(worst, float(np.max(np.abs(actual - expected))))
    print(f"max |transition - expm| over 60 cases: {worst:.3e}")
