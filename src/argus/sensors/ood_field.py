"""Second sensor family for X3: a thresholded Gaussian field (M2, E08-T04).

A smooth random field u(x, t) replaces the hard square cells of correlated.py:

    u(x, t) = M^(-1/2) sum_m [a_m(t) cos(w_m . x) + b_m(t) sin(w_m . x)],
    g(x, t) = 1[u(x, t) > c],   c = Phi^-1(1 - rho).

The frequencies w_m ~ N(0, I / length^2) are fixed per episode; a_m and b_m are
independent stationary AR(1) processes with unit variance and time constant tau. For
any fixed frequencies u(x, t) is exactly N(0, 1) at every point, so P(g = 1) = rho holds
exactly, while the covariance of u tends to exp(-r^2 / (2 length^2)) exp(-d / tau) as M
grows. The covariance of the thresholded g is smaller than that of u and is not claimed
to equal it (Spec M2).

Agents keep the cell model as their nominal model, so this field is a deliberate
misspecification: its boundaries are soft, its g has another space-time correlation, and
no scalable exact filter is claimed for it. The field stays simulator-internal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import NormalDist

import numpy as np
from numpy.typing import NDArray

from argus.core.events import KeyedRandom
from argus.sensors.binary import SensorModelError
from argus.world.geometry import Point

FEATURES = 32
STEP_TICKS = 5  # the field moves once per sensor slot


@dataclass(slots=True)
class GaussianThresholdField:
    """Hidden field g(x, t); a pure function of (seed, scenario, point, tick)."""

    bias_probability: float
    length_m: float
    time_constant_s: float
    tick_seconds: float
    rng: KeyedRandom
    scenario_id: str
    _threshold: float = field(default=0.0, repr=False)
    _frequencies: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0), repr=False)
    _generator: np.random.Generator | None = field(default=None, repr=False)
    _states: list[NDArray[np.float64]] = field(default_factory=list, repr=False)
    _decay: float = field(default=0.0, repr=False)

    def __post_init__(self) -> None:
        if not 0.0 < self.bias_probability < 0.5:
            raise SensorModelError("bias_probability must be in (0, 0.5)")
        for name in ("length_m", "time_constant_s", "tick_seconds"):
            value = getattr(self, name)
            if not (math.isfinite(value) and value > 0):
                raise SensorModelError(f"{name} must be finite and > 0")
        self._threshold = NormalDist().inv_cdf(1.0 - self.bias_probability)
        self._decay = math.exp(-STEP_TICKS * self.tick_seconds / self.time_constant_s)
        seed = int(self.rng.uniform("sensor.gauss.seed", self.scenario_id) * 2**63)
        self._generator = np.random.Generator(np.random.PCG64(seed))
        self._frequencies = self._generator.normal(0.0, 1.0 / self.length_m, (FEATURES, 2))
        self._states = [self._generator.normal(0.0, 1.0, 2 * FEATURES)]

    def _state(self, step: int) -> NDArray[np.float64]:
        assert self._generator is not None
        noise_scale = math.sqrt(1.0 - self._decay**2)
        while len(self._states) <= step:
            innovation = self._generator.normal(0.0, 1.0, 2 * FEATURES)
            self._states.append(self._decay * self._states[-1] + noise_scale * innovation)
        return self._states[step]

    def value(self, point: Point, tick: int) -> float:
        """u(x, t), standard normal at every point. Evaluator and tests only."""
        if type(tick) is not int or tick < 0:
            raise ValueError(f"tick must be an int >= 0, got {tick!r}")
        state = self._state(tick // STEP_TICKS)
        phase = self._frequencies @ np.array(point[:2], dtype=np.float64)
        total = state[:FEATURES] @ np.cos(phase) + state[FEATURES:] @ np.sin(phase)
        return float(total) / math.sqrt(FEATURES)

    def bias_at(self, point: Point, tick: int) -> int:
        return int(self.value(point, tick) > self._threshold)
