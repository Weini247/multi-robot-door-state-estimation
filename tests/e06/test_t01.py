"""ARGUS-E06-T01: binary risk and exact one-step VOI against an independent oracle."""

from __future__ import annotations

from itertools import product

import numpy as np
import pytest
from scipy.linalg import expm

from argus.beliefs.exact_micro import joint_prior
from argus.beliefs.fixed_lag import NominalPassageModel
from argus.core.types import EvidenceId
from argus.dynamics.binary_ctmc import BinaryCTMC
from argus.policies.voi import (
    CostWeights,
    Losses,
    ObserveAction,
    binary_risk,
    decide,
    delivery_value,
    expected_value,
)

SYMMETRIC = Losses(1.0, 1.0)
FREE = CostWeights(0.0, 0.0, 0.0)


def static(q: float, rho: float = 0.0) -> NominalPassageModel:
    return NominalPassageModel(BinaryCTMC(0.0, 0.0), BinaryCTMC(0.0, 0.0), q, 0.5, rho)


def test_perfect_free_static_measurement_at_half_has_value_half() -> None:
    est = expected_value(
        ObserveAction(0.0, 0.0), joint_prior(0.5, 0.0), static(0.0), 0.0, SYMMETRIC, FREE
    )
    assert est.risk_without == pytest.approx(0.5, abs=1e-15)
    assert est.expected_risk_with == pytest.approx(0.0, abs=1e-15)
    assert est.voi == pytest.approx(0.5, abs=1e-15)


def test_uninformative_sensor_has_value_minus_cost() -> None:
    costs = CostWeights(0.05, 0.01, 0.01)
    action = ObserveAction(3.0, 8.0)
    for p in (0.1, 0.5, 0.9):
        est = expected_value(action, joint_prior(p, 0.0), static(0.5), 8.0, Losses(), costs)
        assert est.gain == pytest.approx(0.0, abs=1e-15)
        assert est.voi == pytest.approx(-(0.05 * 3 + 0.01 * 8 + 0.01), abs=1e-15)


def test_known_evidence_has_zero_information_gain() -> None:
    eid = EvidenceId("ep", "r1", 4)
    joint = joint_prior(0.5, 0.1)
    args = (joint, static(0.05, 0.1), 1, eid)
    assert delivery_value(*args, known={eid}, horizon_s=5.0, losses=SYMMETRIC) == 0.0
    assert delivery_value(*args, known=set(), horizon_s=5.0, losses=SYMMETRIC) > 0.0


def test_risk_at_p_08_with_pilot_losses_is_equal_for_both_decisions() -> None:
    losses = Losses(4.0, 1.0)
    assert losses.false_open * (1 - 0.8) == pytest.approx(0.8)
    assert losses.false_closed * 0.8 == pytest.approx(0.8)
    assert binary_risk(0.8, losses) == pytest.approx(0.8)
    assert decide(0.8, losses) == 1 and decide(0.7999, losses) == 0


def oracle(
    action: ObserveAction,
    joint: tuple[float, ...],
    model: NominalPassageModel,
    horizon: float,
    losses: Losses,
    costs: CostWeights,
) -> float:
    """Independent VOI: joint law of (x at t_y, readings, x at T) from scipy expm."""

    def transition(t: float) -> np.ndarray:
        def gen(c: BinaryCTMC) -> np.ndarray:
            return np.array([[-c.rate_01, c.rate_01], [c.rate_10, -c.rate_10]])

        return np.kron(expm(gen(model.z_model) * t), expm(gen(model.g_model) * t))

    def risk(p: float) -> float:
        return min(losses.false_open * (1 - p), losses.false_closed * p)

    b = np.array(joint)
    without = risk(float((b @ transition(horizon))[2:].sum()))
    at_y = b @ transition(action.duration_s)
    rest = transition(horizon - action.duration_s)
    q = model.private_error
    expected = 0.0
    for ys in product((0, 1), repeat=action.measurements):
        weights = np.array(
            [
                at_y[s] * np.prod([(1 - q) if y == (z ^ g) else q for y in ys])
                for s, (z, g) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1)))
            ]
        )
        mass = weights.sum()
        if mass == 0:
            continue
        expected += mass * risk(float((weights / mass @ rest)[2:].sum()))
    cost = costs.cost(action.distance_m, action.duration_s, action.measurements)
    return without - expected - cost


@pytest.mark.parametrize("m", [1, 2, 3])
@pytest.mark.parametrize("duration", [0.0, 1.5, 6.0])
def test_dynamic_voi_matches_the_independent_oracle(m: int, duration: float) -> None:
    model = NominalPassageModel(BinaryCTMC(0.05, 0.1), BinaryCTMC(0.005, 0.045), 0.05, 1 / 3, 0.1)
    costs = CostWeights()
    for p in (0.2, 1 / 3, 0.6, 0.9):
        joint = joint_prior(p, 0.1)
        action = ObserveAction(2.0, duration, m)
        est = expected_value(action, joint, model, 10.0, Losses(), costs)
        assert est.voi == pytest.approx(
            oracle(action, joint, model, 10.0, Losses(), costs), abs=1e-12
        )


def test_no_double_prediction() -> None:
    # An uninformative observation at t_y must leave exactly Predict(b, T): predicting
    # to t_y and then to T - t_y equals one prediction to T, never t_y + T.
    model = NominalPassageModel(BinaryCTMC(0.2, 0.3), BinaryCTMC(0.0, 0.0), 0.5, 0.9, 0.0)
    joint = joint_prior(0.9, 0.0)
    est = expected_value(ObserveAction(0.0, 4.0), joint, model, 5.0, SYMMETRIC, FREE)
    p_at_t = 0.4 + 0.5 * np.exp(-0.5 * 5.0)  # one prediction over the full horizon
    assert est.risk_without == pytest.approx(min(1 - p_at_t, p_at_t), abs=1e-14)
    assert est.expected_risk_with == pytest.approx(est.risk_without, abs=1e-14)


def test_static_free_information_is_never_harmful() -> None:
    for p, rho, q, m in product((0.05, 0.3, 0.5, 0.85), (0.0, 0.1, 0.3), (0.0, 0.05, 0.3), (1, 2)):
        model = static(q, rho)
        est = expected_value(
            ObserveAction(0.0, 0.0, m), joint_prior(p, rho), model, 0.0, Losses(), FREE
        )
        assert est.voi >= -1e-15


def test_persistent_fully_shared_bias_is_not_resolved_by_repetition() -> None:
    # rho = .5 with q = 0: every repeated reading equals z XOR g, so repetitions add nothing.
    model = static(0.0, 0.5)
    one = expected_value(
        ObserveAction(0.0, 0.0, 1), joint_prior(0.5, 0.5), model, 0.0, SYMMETRIC, FREE
    )
    many = expected_value(
        ObserveAction(0.0, 0.0, 4), joint_prior(0.5, 0.5), model, 0.0, SYMMETRIC, FREE
    )
    assert one.gain == pytest.approx(0.0, abs=1e-15)
    assert many.gain == pytest.approx(0.0, abs=1e-15)


def test_invalid_inputs_rejected() -> None:
    with pytest.raises(ValueError):
        Losses(-1.0, 1.0)
    with pytest.raises(ValueError):
        Losses(0.0, 0.0)
    with pytest.raises(ValueError):
        binary_risk(1.2, Losses())
    with pytest.raises(ValueError):
        ObserveAction(1.0, 1.0, 0)
    with pytest.raises(ValueError):
        expected_value(
            ObserveAction(0.0, 5.0), joint_prior(0.5, 0.1), static(0.1), 4.0, Losses(), FREE
        )
    with pytest.raises(ValueError):
        CostWeights().cost(-1.0, 0.0, 1)
