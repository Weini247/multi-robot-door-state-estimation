"""Controlled interventions for the mechanism study (E09-T01, change request 2026-11-07).

An intervention changes one declared part of an episode and reruns the simulator; it
never edits a recorded log (unchanged logs are playback, not counterfactuals). Three
kinds exist:

* arm: which information actions the unchanged `repair` policy may use (re-observe,
  send, both) or an anchor control (no_comm, age_revisit, dynamics_ids);
* door tempo s: both passage rates are multiplied by s, so the long-run open share
  pi = 1/3 stays fixed while beliefs age s times faster (s = 0 is the static cell);
* cost ratio: the false-open cost on the policy level, which moves the decision
  threshold c / (c + 1).

All arms of one building and replicate share the world, sensor, channel, request and
policy seeds. Their histories are identical until the first differing action; after
that the sensor draws follow the new trajectory because they are keyed by physical
sample slots (Spec M2).
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

from argus.experiments.cells import PREREGISTERED, CellParameters, FactorCell, apply_cell
from argus.experiments.runner import EpisodeConfig

ARMS = ("repair", "repair_observe", "repair_send", "no_comm", "age_revisit", "dynamics_ids")
BASE_RATES = PREREGISTERED.passage_rates


@dataclass(frozen=True, slots=True)
class InterventionSpec:
    arm: str
    rate_scale: float = 1.0
    cost_ratio: float = 4.0

    def __post_init__(self) -> None:
        if self.arm not in ARMS:
            raise ValueError(f"unknown arm {self.arm!r}")
        if not (math.isfinite(self.rate_scale) and self.rate_scale >= 0):
            raise ValueError("rate_scale must be finite and >= 0")
        if not (math.isfinite(self.cost_ratio) and self.cost_ratio > 0):
            raise ValueError("cost_ratio must be finite and > 0")

    @property
    def condition_id(self) -> str:
        return f"s{self.rate_scale:g}-c{self.cost_ratio:g}"

    @property
    def cell(self) -> FactorCell:
        return FactorCell(dynamic=self.rate_scale > 0, correlated=False, relay=False)

    @property
    def cell_params(self) -> CellParameters:
        rate_01, rate_10 = BASE_RATES
        scale = self.rate_scale
        return dataclasses.replace(PREREGISTERED, passage_rates=(rate_01 * scale, rate_10 * scale))

    def apply(self, config: EpisodeConfig) -> EpisodeConfig:
        """The intervened episode: same seeds and transport key, changed declared parts."""
        if config.method_id != self.arm:
            raise ValueError(f"config runs {config.method_id!r}, intervention arm is {self.arm!r}")
        changed = apply_cell(config, self.cell, self.cell_params)
        return dataclasses.replace(changed, loss_false_open=self.cost_ratio, loss_false_closed=1.0)


def open_window_s(rate_scale: float, cost_ratio: float) -> float:
    """Longest time after certainty that a model-matched belief still says "open":
    (1/k) ln((1 - pi) / (theta - pi)) with theta = c / (c + 1); infinite if static."""
    rate_01, rate_10 = (r * rate_scale for r in BASE_RATES)
    k = rate_01 + rate_10
    threshold = cost_ratio / (cost_ratio + 1.0)
    if k == 0:
        return math.inf
    pi = rate_01 / k
    if threshold <= pi:
        return math.inf
    return math.log((1.0 - pi) / (threshold - pi)) / k
