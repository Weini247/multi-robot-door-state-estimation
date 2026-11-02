"""ARGUS-E01-T02: static shared-bias micro case against exact rational enumeration."""

from __future__ import annotations

from fractions import Fraction
from itertools import product

import pytest

from argus.beliefs.exact_micro import (
    ImpossibleEvidence,
    bayes_update,
    correct_joint,
    joint_prior,
    measurement_likelihood,
    shared_bias_posterior,
)

ATOL = 1e-14


def rational_posterior(p: Fraction, rho: Fraction, q: Fraction, ys: tuple[int, ...]) -> Fraction:
    """Independent oracle: P(z=1 | ys) by summing g out of the exact group likelihood."""

    def group_likelihood(z: int) -> Fraction:
        total = Fraction(0)
        for g, weight in ((0, 1 - rho), (1, rho)):
            term = weight
            for y in ys:
                term *= (1 - q) if y == (z ^ g) else q
            total += term
        return total

    open_mass = p * group_likelihood(1)
    return open_mass / (open_mass + (1 - p) * group_likelihood(0))


def test_two_agreeing_measurements_give_325_over_362() -> None:
    exact = rational_posterior(Fraction(1, 2), Fraction(1, 10), Fraction(1, 20), (1, 1))
    assert exact == Fraction(325, 362)
    assert shared_bias_posterior(0.5, 0.1, 0.05, [1, 1]) == pytest.approx(325 / 362, abs=ATOL)


def test_one_measurement_gives_marginal_accuracy_086() -> None:
    # e = q + rho - 2 q rho = .14, so a single measurement yields 1 - e at prior .5.
    assert rational_posterior(Fraction(1, 2), Fraction(1, 10), Fraction(1, 20), (1,)) == Fraction(
        43, 50
    )
    assert shared_bias_posterior(0.5, 0.1, 0.05, [1]) == pytest.approx(0.86, abs=ATOL)


def test_naive_independence_is_overconfident() -> None:
    naive = bayes_update((0.5, 0.5), (0.14**2, 0.86**2))[1]
    assert naive == pytest.approx(0.86**2 / (0.86**2 + 0.14**2), abs=ATOL)
    assert naive > shared_bias_posterior(0.5, 0.1, 0.05, [1, 1]) + 0.07


def test_conflicting_pair_returns_to_prior_by_symmetry() -> None:
    assert shared_bias_posterior(0.5, 0.1, 0.05, [1, 0]) == pytest.approx(0.5, abs=ATOL)


def test_order_of_distinct_measurements_does_not_matter() -> None:
    forward = shared_bias_posterior(0.3, 0.2, 0.1, [1, 0, 1, 1])
    backward = shared_bias_posterior(0.3, 0.2, 0.1, [1, 1, 0, 1])
    assert forward == pytest.approx(backward, abs=ATOL)


@pytest.mark.parametrize(
    ("p", "rho", "q"),
    [
        (Fraction(1, 2), Fraction(0), Fraction(7, 50)),
        (Fraction(1, 3), Fraction(1, 10), Fraction(1, 20)),
        (Fraction(9, 10), Fraction(1, 4), Fraction(1, 5)),
        (Fraction(1, 100), Fraction(3, 10), Fraction(1, 1000)),
    ],
)
def test_matches_rational_enumeration_for_all_short_sequences(
    p: Fraction, rho: Fraction, q: Fraction
) -> None:
    for length in range(0, 5):
        for ys in product((0, 1), repeat=length):
            oracle = float(rational_posterior(p, rho, q, ys))
            value = shared_bias_posterior(float(p), float(rho), float(q), ys)
            assert value == pytest.approx(oracle, abs=ATOL), ys


def test_impossible_evidence_raises() -> None:
    with pytest.raises(ImpossibleEvidence):
        bayes_update((1.0, 0.0), (0.0, 1.0))
    # No shared bias and a perfect sensor: contradicting readings are impossible.
    with pytest.raises(ImpossibleEvidence):
        shared_bias_posterior(0.5, 0.0, 0.0, [1, 0])


def test_certain_evidence_is_exact_not_clipped() -> None:
    assert bayes_update((0.5, 0.5), (0.0, 1.0)) == (0.0, 1.0)


def test_tiny_likelihoods_stay_finite() -> None:
    posterior = bayes_update((0.5, 0.5), (1e-300, 2e-300))
    assert posterior[1] == pytest.approx(2 / 3, abs=1e-13)
    subnormal = bayes_update((0.5, 0.5), (1e-320, 3e-320))
    assert subnormal[1] == pytest.approx(0.75, rel=1e-3)  # subnormal inputs carry few digits


def test_long_sequence_stays_normalized() -> None:
    joint = joint_prior(0.5, 0.1)
    for y in [1, 0] * 500:
        joint = correct_joint(joint, y, 0.05)
    assert sum(joint) == pytest.approx(1.0, abs=1e-12)
    assert all(0.0 <= value <= 1.0 for value in joint)


def test_likelihood_follows_xor_model() -> None:
    assert measurement_likelihood(1, 0.05) == (0.05, 0.95, 0.95, 0.05)
    assert measurement_likelihood(0, 0.05) == (0.95, 0.05, 0.05, 0.95)


@pytest.mark.parametrize(
    "call",
    [
        lambda: bayes_update((), ()),
        lambda: bayes_update((0.5, 0.5), (1.0,)),
        lambda: bayes_update((0.6, 0.6), (1.0, 1.0)),
        lambda: bayes_update((0.5, 0.5), (-1.0, 1.0)),
        lambda: bayes_update((0.5, 0.5), (float("nan"), 1.0)),
        lambda: joint_prior(1.5, 0.1),
        lambda: measurement_likelihood(2, 0.1),
        lambda: measurement_likelihood(1, float("inf")),
        lambda: correct_joint((1.0, 0.0, 0.0), 1, 0.1),
    ],
)
def test_invalid_inputs_raise_value_error(call: object) -> None:
    with pytest.raises(ValueError):
        call()  # type: ignore[operator]
