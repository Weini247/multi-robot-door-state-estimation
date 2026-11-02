"""Persistent shared sensor error field (M2, E03-T02).

The geometry is covered by hidden square disturbance cells of side L with a
fixed random grid offset. Each cell c carries a binary error state g_c(t) that
follows its own CTMC with rates (mu_01, mu_10), stationary at the start:

    P(g = 1) = rho = mu_01 / (mu_01 + mu_10),
    Cov(g_c(t), g_c(t + d)) = rho (1 - rho) exp(-(mu_01 + mu_10) d),

and different cells are independent. A reading of a passage in cell c is

    y = z XOR g_c(t) XOR eps,   eps ~ Bernoulli(q) private to (robot, passage, slot),

so the marginal error is e = q + rho - 2 q rho. To vary dependence without
changing sensor quality, q is chosen as (e - rho) / (1 - 2 rho) for a fixed e
with 0 <= rho <= e < 0.5; raising rho with q unchanged would confound
correlation with a higher error rate.

The field is simulator-internal: cells, offsets and g never enter an
Observation. `evaluator_truth` exists for evaluation and debugging only.
The hard cell boundary is a model assumption, not calibrated visibility physics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from argus.core.events import KeyedRandom
from argus.dynamics.binary_ctmc import BinaryCTMC
from argus.sensors.binary import SensorModelError
from argus.world.geometry import Point

Cell = tuple[int, int]


def marginal_error(private_error: float, bias_probability: float) -> float:
    """e = q + rho - 2 q rho: probability that y differs from z."""
    q, rho = private_error, bias_probability
    return q + rho - 2 * q * rho


def matched_private_error(marginal: float, bias_probability: float) -> float:
    """q such that marginal_error(q, rho) == marginal; requires 0 <= rho <= e < 0.5."""
    e, rho = marginal, bias_probability
    if not (math.isfinite(e) and math.isfinite(rho) and 0.0 <= rho <= e < 0.5):
        raise SensorModelError(f"need 0 <= rho <= e < 0.5, got rho={rho!r}, e={e!r}")
    return (e - rho) / (1 - 2 * rho)


@dataclass(slots=True)
class BiasField:
    """Hidden cell field g_c(t); a pure function of (seed, scenario, cell, tick)."""

    cell_size_m: float
    rate_01: float
    rate_10: float
    tick_seconds: float
    rng: KeyedRandom
    scenario_id: str
    initial_probability: float | None = None
    _timelines: dict[Cell, list[int]] = field(default_factory=dict, repr=False)
    _offset: Point = field(default=(0.0, 0.0), repr=False)
    _tick: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        if not (math.isfinite(self.cell_size_m) and self.cell_size_m > 0):
            raise SensorModelError("cell_size_m must be finite and > 0")
        if not (math.isfinite(self.tick_seconds) and self.tick_seconds > 0):
            raise SensorModelError("tick_seconds must be finite and > 0")
        self.model()  # validates rates
        _ = self.rho  # validates the initial distribution
        self._offset = (
            self.rng.uniform("sensor.bias.offset", self.scenario_id, "x") * self.cell_size_m,
            self.rng.uniform("sensor.bias.offset", self.scenario_id, "y") * self.cell_size_m,
        )

    def model(self) -> BinaryCTMC:
        return BinaryCTMC(self.rate_01, self.rate_10)

    @property
    def rho(self) -> float:
        """Initial (and, for positive rates, stationary) P(g = 1)."""
        if self.initial_probability is not None:
            p = self.initial_probability
            if not math.isfinite(p) or not 0.0 <= p <= 1.0:
                raise SensorModelError("initial_probability must be a probability")
            return p
        stationary = self.model().stationary
        if stationary is None:
            raise SensorModelError("both rates are 0: initial_probability must be given")
        return stationary

    def cell_of(self, point: Point) -> Cell:
        ox, oy = self._offset
        return (
            math.floor((point[0] - ox) / self.cell_size_m),
            math.floor((point[1] - oy) / self.cell_size_m),
        )

    def state(self, cell: Cell, tick: int) -> int:
        if type(tick) is not int or tick < 0:
            raise ValueError(f"tick must be an int >= 0, got {tick!r}")
        timeline = self._timelines.get(cell)
        if timeline is None:
            first = self.rng.bernoulli(self.rho, "sensor.bias.initial", self.scenario_id, *cell)
            timeline = self._timelines[cell] = [first]
        transition = self.model().transition(self.tick_seconds)
        while len(timeline) <= tick:
            k = len(timeline)
            p_one = transition[timeline[-1]][1]
            timeline.append(
                self.rng.bernoulli(p_one, "sensor.bias", self.scenario_id, cell[0], cell[1], k)
            )
        return timeline[tick]

    def advance(self, tick: int) -> None:
        """Move the field clock; readings use the current tick."""
        if type(tick) is not int or tick < self._tick:
            raise ValueError(f"field time cannot move backwards to {tick!r}")
        self._tick = tick

    def bias_at(self, point: Point, tick: int) -> int:
        return self.state(self.cell_of(point), tick)

    def evaluator_truth(self, points: list[Point], tick: int) -> list[tuple[Cell, int]]:
        """Debug/evaluator view of the latent field. Never pass this to agents."""
        return [(self.cell_of(p), self.bias_at(p, tick)) for p in points]


def sample_private_noise(
    rng: KeyedRandom, private_error: float, robot_id: str, passage_id: str, slot: int
) -> int:
    """eps ~ Bernoulli(q), private to (robot, passage, physical slot)."""
    return rng.bernoulli(private_error, "sensor.private", robot_id, passage_id, slot)


@dataclass(frozen=True, slots=True)
class SharedBiasNoise:
    """y = z XOR g_cell(t) XOR eps with the field's g and private error q."""

    bias: BiasField
    private_error: float
    rng: KeyedRandom

    def __post_init__(self) -> None:
        q = self.private_error
        if not math.isfinite(q) or not 0.0 <= q <= 1.0:
            raise SensorModelError(f"private_error must be a probability, got {q!r}")

    def reading(
        self, robot_id: str, passage_id: str, slot: int, tick: int, truth: int, location: Point
    ) -> int:
        g = self.bias.bias_at(location, tick)
        eps = sample_private_noise(self.rng, self.private_error, robot_id, passage_id, slot)
        return truth ^ g ^ eps
