"""The eight cells of the 2^3 mechanism design (X1) as episode configurations (E07-T08).

Factors (Audit §8.3):

* dynamics: static passages (rates 0, same initial prior) or dynamic CTMC passages;
* sensor: independent flips with marginal error e, or the shared-bias field with
  P(g=1) = rho and the private error matched to the same e (Spec M2);
* circulation: no relaying (hop_limit 0, only own evidence leaves a robot) or
  multi-hop relaying with possible duplicates (hop_limit 3).

`apply_cell` sets the generating world and sensor and a model-matched nominal
model for the agents (the controlled X1/X2 condition); X3 varies the truth
without changing the nominal model. Channel, losses and everything else stay
as in the base configuration.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from argus.beliefs.fixed_lag import NominalPassageModel
from argus.dynamics.binary_ctmc import BinaryCTMC
from argus.experiments.runner import EpisodeConfig, SensorBias
from argus.sensors.correlated import matched_private_error
from argus.world.scenario import CtmcPassage, ScenarioSpec


@dataclass(frozen=True, slots=True)
class FactorCell:
    dynamic: bool
    correlated: bool
    relay: bool

    @property
    def cell_id(self) -> str:
        return "-".join(
            (
                "dyn" if self.dynamic else "sta",
                "cor" if self.correlated else "ind",
                "relay" if self.relay else "direct",
            )
        )


CELLS = tuple(
    FactorCell(d, c, r) for d in (False, True) for c in (False, True) for r in (False, True)
)


@dataclass(frozen=True, slots=True)
class CellParameters:
    passage_rates: tuple[float, float] = (0.01, 0.02)
    static_prior_open: float = 1 / 3
    marginal_error: float = 0.14
    rho: float = 0.05
    bias_rate_sum: float = 0.05
    bias_cell_m: float = 4.0
    relay_hop_limit: int = 3


PREREGISTERED = CellParameters()  # values approved 2026-11-02 (rho .05, matched q .10)


def with_passage_model(spec: ScenarioSpec, model: CtmcPassage) -> ScenarioSpec:
    return dataclasses.replace(spec, passages=tuple((pid, model) for pid, _ in spec.passages))


def apply_cell(
    config: EpisodeConfig, cell: FactorCell, params: CellParameters = PREREGISTERED
) -> EpisodeConfig:
    if cell.dynamic:
        rate_01, rate_10 = params.passage_rates
        passage = CtmcPassage(rate_01, rate_10)
        prior = rate_01 / (rate_01 + rate_10)
        z_model = BinaryCTMC(rate_01, rate_10)
    else:
        passage = CtmcPassage(0.0, 0.0, initial_open=params.static_prior_open)
        prior = params.static_prior_open
        z_model = BinaryCTMC(0.0, 0.0)
    bias: SensorBias | None = None
    if cell.correlated:
        bias = SensorBias(params.rho, params.bias_rate_sum, params.bias_cell_m)
        nominal = NominalPassageModel(
            z_model,
            BinaryCTMC(*bias.rates),
            matched_private_error(params.marginal_error, params.rho),
            prior,
            params.rho,
        )
    else:
        nominal = NominalPassageModel(
            z_model, BinaryCTMC(0.0, 0.0), params.marginal_error, prior, 0.0
        )
    return dataclasses.replace(
        config,
        scenario=with_passage_model(config.scenario, passage),
        sensor_error=params.marginal_error,
        sensor_bias=bias,
        nominal=nominal,
        hop_limit=params.relay_hop_limit if cell.relay else 0,
    )
