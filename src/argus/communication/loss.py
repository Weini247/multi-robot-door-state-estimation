"""Gilbert-Elliott burst loss per directed link (M4, E04-T03).

Each link has a latent state c in {good, bad} that evolves per tick as the
exact skeleton of a CTMC with rates a (good -> bad) and b (bad -> good), drawn
from keyed (seed, scenario, link, tick) uniforms. It is a property of the
environment: sending more or fewer messages cannot change it. At transmission
start, a frame is dropped with probability eps_good or eps_bad depending on
c(tick), using a separate key per frame.

    P(bad) = a / (a + b),   stationary drop rate = eps_g (1 - P(bad)) + eps_b P(bad).

With a = b = 0 the initial state must be given explicitly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from argus.core.events import KeyedRandom
from argus.dynamics.binary_ctmc import BinaryCTMC

Link = tuple[str, str]
GOOD, BAD = 0, 1


@dataclass(slots=True)
class GilbertElliott:
    rate_good_to_bad: float
    rate_bad_to_good: float
    eps_good: float
    eps_bad: float
    tick_seconds: float
    rng: KeyedRandom
    scenario_id: str
    initial_bad: float | None = None
    _timelines: dict[Link, list[int]] = field(default_factory=dict, repr=False)
    _tick: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        for name in ("eps_good", "eps_bad"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be a probability, got {value!r}")
        if not (math.isfinite(self.tick_seconds) and self.tick_seconds > 0):
            raise ValueError("tick_seconds must be finite and > 0")
        self.model()
        _ = self.p_bad_initial

    def model(self) -> BinaryCTMC:
        """CTMC over c with state 1 = bad."""
        return BinaryCTMC(self.rate_good_to_bad, self.rate_bad_to_good)

    @property
    def p_bad_initial(self) -> float:
        if self.initial_bad is not None:
            p = self.initial_bad
            if not math.isfinite(p) or not 0.0 <= p <= 1.0:
                raise ValueError("initial_bad must be a probability")
            return p
        stationary = self.model().stationary
        if stationary is None:
            raise ValueError("both rates are 0: initial_bad must be given")
        return stationary

    def stationary_drop_rate(self) -> float:
        p_bad = self.model().stationary
        if p_bad is None:
            p_bad = self.p_bad_initial
        return self.eps_good * (1 - p_bad) + self.eps_bad * p_bad

    def state(self, link: Link, tick: int) -> int:
        if type(tick) is not int or tick < 0:
            raise ValueError(f"tick must be an int >= 0, got {tick!r}")
        name = f"{link[0]}>{link[1]}"
        timeline = self._timelines.get(link)
        if timeline is None:
            first = self.rng.bernoulli(
                self.p_bad_initial, "channel.state.initial", self.scenario_id, name
            )
            timeline = self._timelines[link] = [first]
        transition = self.model().transition(self.tick_seconds)
        while len(timeline) <= tick:
            k = len(timeline)
            p_bad = transition[timeline[-1]][1]
            timeline.append(self.rng.bernoulli(p_bad, "channel.state", self.scenario_id, name, k))
        return timeline[tick]

    def advance(self, tick: int) -> None:
        """Move the channel clock forward; the state field itself is precomputed by key."""
        if type(tick) is not int or tick < self._tick:
            raise ValueError(f"channel time cannot move backwards to {tick!r}")
        self._tick = tick

    def drops(self, link: Link, tick: int, message_key: str) -> bool:
        eps = self.eps_bad if self.state(link, tick) == BAD else self.eps_good
        name = f"{link[0]}>{link[1]}"
        return bool(
            self.rng.bernoulli(eps, "channel.loss", self.scenario_id, name, tick, message_key)
        )
