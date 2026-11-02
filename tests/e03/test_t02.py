"""ARGUS-E03-T02: shared sensor error field with matched marginal error."""

from __future__ import annotations

import dataclasses
import math
from fractions import Fraction

import pytest

from argus.core.events import KeyedRandom
from argus.core.types import Observation
from argus.sensors.binary import BinarySensor, SensorConfig, SensorModelError
from argus.sensors.correlated import (
    BiasField,
    SharedBiasNoise,
    marginal_error,
    matched_private_error,
)
from argus.world.geometry import Geometry, Passage, Rect, Room
from argus.world.scenario import RobotStart, ScenarioSpec, ScriptedPassage, WorldBackend2D

MU01, MU10, TICK = 0.005, 0.045, 0.1  # pilot defaults: rho = .1


def field(seed: int = 1, cell: float = 4.0, **overrides: object) -> BiasField:
    values: dict[str, object] = dict(
        cell_size_m=cell,
        rate_01=MU01,
        rate_10=MU10,
        tick_seconds=TICK,
        rng=KeyedRandom(seed),
        scenario_id="field-fixture",
    )
    values.update(overrides)
    return BiasField(**values)  # type: ignore[arg-type]


def test_independent_and_correlated_conditions_have_exactly_the_same_marginal() -> None:
    e = Fraction(14, 100)
    for rho in (Fraction(0), Fraction(1, 10)):
        q = (e - rho) / (1 - 2 * rho)
        assert q + rho - 2 * q * rho == e  # exact rational identity
    assert matched_private_error(0.14, 0.0) == pytest.approx(0.14, abs=1e-16)
    assert matched_private_error(0.14, 0.1) == pytest.approx(0.05, abs=1e-16)
    for rho in (0.0, 0.1):
        q = matched_private_error(0.14, rho)
        assert marginal_error(q, rho) == pytest.approx(0.14, abs=1e-16)


def test_unmatched_correlation_would_raise_the_error_rate() -> None:
    # Guard against the forbidden design: raising rho while keeping q.
    assert marginal_error(0.14, 0.1) == pytest.approx(0.212, abs=1e-15)  # .14 + .1 - 2*.014


@pytest.mark.parametrize(("e", "rho"), [(0.14, 0.2), (0.5, 0.1), (0.14, -0.01), (math.nan, 0.0)])
def test_matching_requires_valid_range(e: float, rho: float) -> None:
    with pytest.raises(SensorModelError):
        matched_private_error(e, rho)


def test_autocovariance_matches_analytic_formula() -> None:
    f = field()
    rho, k = f.rho, MU01 + MU10
    assert rho == pytest.approx(0.1, abs=1e-15)
    model = f.model()
    for d_ticks in (0, 1, 10, 100, 1000):
        d = d_ticks * TICK
        joint_one = rho * model.transition(d)[1][1]  # P(g_t = 1, g_t+d = 1), stationary start
        cov = joint_one - rho * rho
        assert cov == pytest.approx(rho * (1 - rho) * math.exp(-k * d), abs=1e-14)


def test_sampled_field_reproduces_the_autocovariance() -> None:
    # Tolerances fixed before the run: 5 standard errors over independent replicates.
    replicates, lag = 4000, 20
    rho = 0.1
    expected = rho * (1 - rho) * math.exp(-(MU01 + MU10) * lag * TICK)
    pairs = [
        (field(seed=s).state((0, 0), 0), field(seed=s).state((0, 0), lag))
        for s in range(replicates)
    ]
    mean0 = sum(a for a, _ in pairs) / replicates
    mean1 = sum(b for _, b in pairs) / replicates
    cov = sum(a * b for a, b in pairs) / replicates - mean0 * mean1
    assert abs(mean0 - rho) < 5 * math.sqrt(rho * (1 - rho) / replicates)
    assert abs(cov - expected) < 5 * math.sqrt(rho * (1 - rho) / replicates)
    other = [
        (field(seed=s).state((0, 0), 0), field(seed=s).state((1, 0), 0)) for s in range(replicates)
    ]
    cross = sum(a * b for a, b in other) / replicates - (sum(a for a, _ in other) / replicates) * (
        sum(b for _, b in other) / replicates
    )
    assert abs(cross) < 5 * math.sqrt(rho * (1 - rho) / replicates)  # cells independent


def test_field_is_a_pure_function_of_keys() -> None:
    a, b = field(seed=3), field(seed=3)
    forward = [a.state((2, -1), t) for t in range(200)]
    for t in reversed(range(200)):
        b.state((5, 5), t)
    assert [b.state((2, -1), t) for t in range(200)] == forward
    points = [(0.5, 0.5), (3.9, 0.5), (8.0, 0.5)]
    cells = {a.cell_of(p) for p in points}
    assert len(cells) >= 2  # 4 m cells split a 7.5 m span


def test_zero_rate_field_needs_and_keeps_its_initial_state() -> None:
    with pytest.raises(SensorModelError):
        field(rate_01=0.0, rate_10=0.0)
    frozen = field(rate_01=0.0, rate_10=0.0, initial_probability=1.0)
    assert {frozen.state((0, 0), t) for t in range(100)} == {1}


def world(door: int) -> WorldBackend2D:
    geometry = Geometry(
        bounds=Rect(0.0, 0.0, 10.0, 4.0),
        walls=(Rect(4.9, 0.0, 5.1, 1.4), Rect(4.9, 2.6, 5.1, 4.0)),
        rooms=(Room("west", Rect(0.0, 0.0, 4.9, 4.0)), Room("east", Rect(5.1, 0.0, 10.0, 4.0))),
        passages=(Passage("door", Rect(4.9, 1.4, 5.1, 2.6), ("west", "east")),),
    )
    spec = ScenarioSpec(
        scenario_id="sensor-fixture",
        geometry=geometry,
        tick_seconds=TICK,
        horizon_ticks=100,
        robot_radius=0.2,
        v_max=0.5,
        omega_max=1.0,
        starts=(RobotStart("r0", (1.0, 2.0, 0.0)), RobotStart("r1", (2.0, 1.0, 0.3))),
        passages=(("door", ScriptedPassage(door)),),
    )
    return WorldBackend2D(spec, 1)


def test_shared_bias_flips_all_robots_together_and_is_hidden() -> None:
    config = SensorConfig(5.0, math.radians(120), 5, "binary-shared-v1")
    always_biased = field(rate_01=0.0, rate_10=0.0, initial_probability=1.0)
    sensor = BinarySensor(config, SharedBiasNoise(always_biased, 0.0, KeyedRandom(2)), "ep")
    w = world(door=1)
    readings = [sensor.observe(w, rid, 0)[0].value for rid in ("r0", "r1")]
    assert readings == [0, 0]  # the shared g flips both readings the same way
    names = {f.name for f in dataclasses.fields(Observation)}
    assert not names & {"bias", "group", "g", "cell", "latent", "truth"}


def test_private_noise_is_independent_between_robots() -> None:
    config = SensorConfig(5.0, math.radians(120), 5, "binary-shared-v1")
    unbiased = field(rate_01=0.0, rate_10=0.0, initial_probability=0.0)
    agree = 0
    n = 2000
    for seed in range(n):
        sensor = BinarySensor(config, SharedBiasNoise(unbiased, 0.3, KeyedRandom(seed)), "ep")
        w = world(door=1)
        a, b = (sensor.observe(w, rid, 0)[0].value for rid in ("r0", "r1"))
        agree += a == b
    expected = 0.7**2 + 0.3**2
    assert abs(agree / n - expected) < 5 * math.sqrt(expected * (1 - expected) / n)


def test_invalid_field_parameters_rejected() -> None:
    for overrides in (
        {"cell_size_m": 0.0},
        {"tick_seconds": -1.0},
        {"rate_01": -0.1},
        {"rate_01": 0.0, "rate_10": 0.0, "initial_probability": 2.0},
    ):
        with pytest.raises(ValueError):
            field(**overrides)
    with pytest.raises(SensorModelError):
        SharedBiasNoise(field(), 1.5, KeyedRandom(1))
    f = field()
    f.advance(5)
    with pytest.raises(ValueError):
        f.advance(4)
