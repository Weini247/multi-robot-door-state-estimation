"""Heuristics for unknown dependence between evidence sources (M3, E07-T03).

Two controls, neither with a covariance-intersection-style guarantee:

* one_per_bin (raw-evidence protocol): keep at most one reading per entity and per
  public space-time bin, the bin built from the observer's registered pose and the
  observation tick. Readings in the same bin are treated as likely dependent and
  thinned; true error groups are never used.
* logarithmic_pool (posterior-pool protocol): log p_f(z) = w log p_i(z) +
  (1 - w) log p_j(z) - log Z for two posteriors of the same entity at the same time.
  The pool is a convex combination in log-odds, so it is never more extreme than the
  more extreme input; it can lose information (underconfidence) and it depends on
  how often and in which order results are re-pooled, which is why pool events carry
  their own ids and are deduplicated.

Raw evidence remains the main protocol; pooling is a separate, bounded baseline.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

from argus.beliefs.exact_micro import ImpossibleEvidence
from argus.core.types import EvidenceId, Observation

RAW_PROTOCOL = "raw-evidence-v1"
POOL_PROTOCOL = "posterior-pool-v1"


def space_time_bin(
    observation: Observation, cell_m: float, period_ticks: int
) -> tuple[str, int, int, int]:
    """Public bin of a reading: entity, observer grid cell and time slice."""
    if cell_m <= 0 or period_ticks < 1:
        raise ValueError("bin size must be positive")
    x, y, _ = observation.observer_pose
    return (
        observation.entity,
        math.floor(x / cell_m),
        math.floor(y / cell_m),
        observation.observed_tick // period_ticks,
    )


def one_per_bin(
    observations: Iterable[Observation], cell_m: float = 1.0, period_ticks: int = 50
) -> tuple[Observation, ...]:
    """Thin readings to one per public space-time bin; the smallest EvidenceId wins."""
    kept: dict[tuple[str, int, int, int], Observation] = {}
    for obs in sorted(observations, key=lambda o: o.evidence_id):
        kept.setdefault(space_time_bin(obs, cell_m, period_ticks), obs)
    return tuple(sorted(kept.values(), key=lambda o: (o.observed_tick, o.evidence_id)))


def logarithmic_pool(p_i: float, p_j: float, weight: float = 0.5) -> float:
    """Pooled P(open) of two posteriors about the same entity and time."""
    for value in (p_i, p_j, weight):
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"inputs must be in [0, 1], got {value!r}")
    if weight == 1.0:
        return p_i
    if weight == 0.0:
        return p_j
    if (p_i == 0.0 and p_j == 1.0) or (p_i == 1.0 and p_j == 0.0):
        raise ImpossibleEvidence("disjoint supports cannot be pooled")
    if p_i in (0.0, 1.0):
        return p_i
    if p_j in (0.0, 1.0):
        return p_j
    log_open = weight * math.log(p_i) + (1 - weight) * math.log(p_j)
    log_closed = weight * math.log(1 - p_i) + (1 - weight) * math.log(1 - p_j)
    peak = max(log_open, log_closed)
    a, b = math.exp(log_open - peak), math.exp(log_closed - peak)
    return a / (a + b)


class PoolLedger:
    """Applies each pool event at most once (pool events have their own ids)."""

    def __init__(self) -> None:
        self._seen: set[EvidenceId] = set()

    def admit(self, pool_id: EvidenceId) -> bool:
        if pool_id in self._seen:
            return False
        self._seen.add(pool_id)
        return True
