"""Preregistered inference for the X1/X2 test batch (ARGUS-E08-T03).

Written and tested before any test episode ran, then frozen together with the run set.
It implements the preregistration (configs/experiments/preregistration_v1.json) plus
amendments 01 and 02.

* Unit of inference: the building. Each building first gets its mean over cells (equal
  weights) and replicates, so more rows per building never increase n.
* Families are weighted 50/50: a family-weighted mean is the mean of the family means.
* Primary H2 estimand: gain = 1 - M(L_candidate) / M(L_control), where M is the
  family-weighted mean of the building means.
* Coverage condition (amendment 01, O3): relative coverage loss
  1 - M(Cov_candidate) / M(Cov_control) on the target set "closed" at the horizon.
* Paired cluster bootstrap: buildings are resampled with replacement within each family
  (stratified, n_f kept). Every quantity of one analysis uses the same draws. The interval
  is the 2.5/97.5 percentile, with 10 000 draws and seed 20261103.
* H2 claim: lower gain bound >= 0.10 AND upper coverage-loss bound <= 0.05. A p-value is
  never part of the decision.
* Secondary p-values are two-sided bootstrap p-values, Holm-adjusted. The bootstrap CIs
  themselves are not adjusted.

All functions are pure; the driver tools/analyze_x1x2.py reads the batch.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from argus.evaluation.metrics import LagEvent

ANALYSIS_SEED = 20261103
DRAWS = 10_000
GAIN_THRESHOLD = 0.10
COVERAGE_LOSS_LIMIT = 0.05
B_MINIMUM = 60
FACTORS = ("dynamics", "sensor", "circulation")
LEVELS = {"dynamics": ("sta", "dyn"), "sensor": ("ind", "cor"), "circulation": ("direct", "relay")}


class AnalysisError(ValueError):
    """The data do not support the preregistered analysis (missing, duplicated, undefined)."""


# --- aggregation ------------------------------------------------------------------------------


def building_values(
    rows: Iterable[Mapping[str, Any]],
    method: str,
    field: str,
    cells: Sequence[str],
    replicates: int,
    buildings: Iterable[str] | None = None,
) -> dict[str, float]:
    """Mean of `field` per building: the mean over cells (equal weights) of the mean over
    replicates. Every building must have each (cell, replicate) exactly once."""
    seen: dict[str, dict[tuple[str, int], float]] = defaultdict(dict)
    for r in rows:
        if r["method_id"] != method:
            continue
        key = (r["cell_id"], int(r["replicate"]))
        if key in seen[r["building_id"]]:
            raise AnalysisError(f"duplicate row {r['building_id']} {key} for {method}")
        if r[field] is None:
            raise AnalysisError(f"{field} missing in {r['building_id']} {key} for {method}")
        seen[r["building_id"]][key] = float(r[field])
    wanted = sorted(seen) if buildings is None else sorted(buildings)
    out = {}
    for b in wanted:
        values = seen.get(b, {})
        expected = {(c, k) for c in cells for k in range(replicates)}
        if set(values) != expected:
            raise AnalysisError(f"building {b} is incomplete for {method}")
        out[b] = math.fsum(
            math.fsum(values[(c, k)] for k in range(replicates)) / replicates for c in cells
        ) / len(cells)
    return out


def family_weighted_mean(values: Mapping[str, float], family: Mapping[str, str]) -> float:
    by: dict[str, list[float]] = defaultdict(list)
    for b, v in values.items():
        by[family[b]].append(v)
    if not by:
        raise AnalysisError("no buildings")
    return math.fsum(math.fsum(v) / len(v) for v in by.values()) / len(by)


def ratio_gain(
    candidate: Mapping[str, float], control: Mapping[str, float], family: Mapping[str, str]
) -> float:
    """1 - M(candidate) / M(control); undefined (raises) if the control mean is zero."""
    _paired(candidate, control)
    denominator = family_weighted_mean(control, family)
    if denominator == 0:
        raise AnalysisError("control mean is zero: the relative gain is undefined")
    return 1 - family_weighted_mean(candidate, family) / denominator


def _paired(*values: Mapping[str, float]) -> list[str]:
    keys = set(values[0])
    if any(set(v) != keys for v in values[1:]):
        raise AnalysisError("paired quantities need the same buildings")
    return sorted(keys)


# --- bootstrap ------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Interval:
    estimate: float
    low: float
    high: float
    p_value: float | None = None  # two-sided bootstrap p for H0: value = 0

    def to_json(self) -> dict[str, float | None]:
        """Estimate and percentile interval. The p-value is not exported here: only the
        preregistered Holm family reports p (amendment 03, A4)."""
        return {"estimate": self.estimate, "low": self.low, "high": self.high}


class PairedBootstrap:
    """Stratified resampling of buildings; one set of draws shared by all quantities."""

    def __init__(
        self,
        buildings: Sequence[str],
        family: Mapping[str, str],
        draws: int = DRAWS,
        seed: int = ANALYSIS_SEED,
    ) -> None:
        if len(set(buildings)) != len(buildings):
            raise AnalysisError("buildings must be unique")
        self.buildings = sorted(buildings)
        self.family = {b: family[b] for b in self.buildings}
        self.draws = draws
        rng = np.random.Generator(np.random.PCG64(seed))
        self._strata: list[tuple[NDArray[np.int64], NDArray[np.int64]]] = []
        index = {b: i for i, b in enumerate(self.buildings)}
        for f in sorted(set(self.family.values())):
            members = np.array([index[b] for b in self.buildings if self.family[b] == f])
            picks = rng.integers(0, len(members), size=(draws, len(members)))
            self._strata.append((members, picks))

    def _vector(self, values: Mapping[str, float]) -> NDArray[np.float64]:
        if set(values) != set(self.buildings):
            raise AnalysisError("values must cover exactly the bootstrap buildings")
        return np.array([values[b] for b in self.buildings], dtype=np.float64)

    def means(self, values: Mapping[str, float]) -> NDArray[np.float64]:
        """Family-weighted mean in every draw."""
        x = self._vector(values)
        per_family = [x[members][picks].mean(axis=1) for members, picks in self._strata]
        return np.mean(np.vstack(per_family), axis=0)

    def ratio_gain(self, candidate: Mapping[str, float], control: Mapping[str, float]) -> Interval:
        point = ratio_gain(candidate, control, self.family)
        ctrl = self.means(control)
        if np.any(ctrl == 0):
            raise AnalysisError("a bootstrap draw has a zero control mean")
        return _interval(point, 1 - self.means(candidate) / ctrl)

    def mean(self, values: Mapping[str, float]) -> Interval:
        return _interval(family_weighted_mean(values, self.family), self.means(values))

    def difference(self, candidate: Mapping[str, float], control: Mapping[str, float]) -> Interval:
        """M(candidate) - M(control) with the same draws."""
        _paired(candidate, control)
        point = family_weighted_mean(candidate, self.family) - family_weighted_mean(
            control, self.family
        )
        return _interval(point, self.means(candidate) - self.means(control))

    def gain_report(
        self, candidate: Mapping[str, float], control: Mapping[str, float]
    ) -> dict[str, Any]:
        """Ratio gain and the absolute difference. If the control mean is zero in the data
        or in any draw, the gain is undefined and only the difference is reported
        (preregistration claims.zero_control_loss)."""
        out: dict[str, Any] = {"difference": self.difference(candidate, control).to_json()}
        try:
            out["gain"] = self.ratio_gain(candidate, control).to_json()
        except AnalysisError as undefined:
            out["gain"] = None
            out["gain_undefined"] = str(undefined)
        return out


def _interval(point: float, draws: NDArray[np.float64]) -> Interval:
    low, high = np.percentile(draws, [2.5, 97.5])
    # two-sided bootstrap p for H0: value = 0, with the +1 correction (never exactly 0)
    below = (int(np.sum(draws <= 0)) + 1) / (len(draws) + 1)
    above = (int(np.sum(draws >= 0)) + 1) / (len(draws) + 1)
    p = min(1.0, 2 * min(below, above))
    return Interval(float(point), float(low), float(high), p)


def single_family(family: Mapping[str, str], name: str) -> list[str]:
    return sorted(b for b, f in family.items() if f == name)


# --- decisions ------------------------------------------------------------------------------


def h2_verdict(gain: Interval, coverage_loss: Interval) -> str:
    """Preregistered H2 rule: both CI conditions; p-values play no role."""
    if gain.low >= GAIN_THRESHOLD and coverage_loss.high <= COVERAGE_LOSS_LIMIT:
        return "supported"
    return "not supported"


def holm(p_values: Mapping[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values (monotone, capped at 1)."""
    order = sorted(p_values, key=lambda k: (p_values[k], k))
    m = len(order)
    out, running = {}, 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, (m - i) * p_values[k]))
        out[k] = running
    return out


def complete_prefix(complete: Sequence[bool], minimum: int = B_MINIMUM) -> int | None:
    """Amendment 01 (O8): the largest even prefix of positions that is complete in both lists.
    `complete[p]` says whether position p is complete in both lists. Below `minimum` the
    result is inconclusive (None)."""
    k = 0
    while k < len(complete) and complete[k]:
        k += 1
    k -= k % 2
    return k if k >= minimum else None


# --- factorial contrasts (H1) --------------------------------------------------------------------


def cell_levels(cell_id: str) -> dict[str, int]:
    parts = cell_id.split("-")
    if len(parts) != 3:
        raise AnalysisError(f"bad cell id {cell_id}")
    return {f: LEVELS[f].index(part) for f, part in zip(FACTORS, parts, strict=True)}


def factorial_contrasts(cell_values: Mapping[str, float]) -> dict[str, float]:
    """Per building I = L11 - L10 - L01 + L00 for each factor pair, averaged over the
    third factor (preregistration, estimands.h1). Keys look like 'dynamics x sensor'."""
    if len(cell_values) != 8:
        raise AnalysisError("contrasts need all 8 cells")
    table = {tuple(cell_levels(c)[f] for f in FACTORS): v for c, v in cell_values.items()}
    out = {}
    for i in range(3):
        for j in range(i + 1, 3):
            third = 3 - i - j
            values = []
            for t in (0, 1):

                def at(a: int, b: int, t: int = t) -> float:
                    key = [0, 0, 0]
                    key[i], key[j], key[third] = a, b, t
                    return table[(key[0], key[1], key[2])]

                values.append(at(1, 1) - at(1, 0) - at(0, 1) + at(0, 0))
            out[f"{FACTORS[i]} x {FACTORS[j]}"] = math.fsum(values) / 2
    return out


def building_cell_values(
    rows: Iterable[Mapping[str, Any]],
    method: str,
    field: str,
    cells: Sequence[str],
    replicates: int,
) -> dict[str, dict[str, float]]:
    """Per building and cell: mean over replicates (complete buildings only, as above)."""
    per: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["method_id"] == method:
            per[r["building_id"]][r["cell_id"]].append(float(r[field]))
    out = {}
    for b, by_cell in per.items():
        if set(by_cell) != set(cells) or any(len(v) != replicates for v in by_cell.values()):
            raise AnalysisError(f"building {b} is incomplete for {method}")
        out[b] = {c: math.fsum(v) / replicates for c, v in by_cell.items()}
    return out


# --- re-scoring from raw records (sensitivities) ------------------------------------------------


def decision_loss(p: float, truth: int, false_open: float, false_closed: float) -> float:
    """Bayes decision at threshold fo/(fo+fc), ties to 'open' as in the runner."""
    decision = int(p >= false_open / (false_open + false_closed))
    if decision == 1 and truth == 0:
        return false_open
    if decision == 0 and truth == 1:
        return false_closed
    return 0.0


def rescored_mean_loss(
    raw: Mapping[str, Any],
    false_open: float,
    false_closed: float,
    failed_probe_loss: float | None = None,
) -> float:
    """Mean loss per request of one raw episode under another loss ratio (amendment 01,
    O5). Answered probes are re-decided from the recorded probability. Defaulted probes
    keep the default decision 'closed', or get `failed_probe_loss` (spec M6 rule) if given.
    Losses are divided by false_closed, so every ratio is on the scale 'wrong closed = 1'."""
    losses = [decision_loss(a[3], a[4], false_open, false_closed) for a in raw["answered"]]
    for d in raw["defaulted"]:
        truth = d[3]
        if failed_probe_loss is not None:
            losses.append(failed_probe_loss)
        else:
            losses.append(false_closed if truth == 1 else 0.0)
    if len(losses) != len(raw["probes"]):
        raise AnalysisError(f"{raw['key']}: answered + defaulted != probes")
    return math.fsum(losses) / len(losses) / false_closed


# --- repair lag (amendment 02, D2) -------------------------------------------------------------


def repair_lags_d2(
    raw: Mapping[str, Any],
    horizon_ticks: int,
    threshold: float = 0.8,
    consecutive: int = 3,
    pooled: bool = False,
) -> list[LagEvent]:
    """Lags per (robot, passage, true change), or per (passage, change) with all robots'
    probes if `pooled`. Probability given to the true state = p if open, 1 - p if closed.
    Censored at the next change, the horizon or the failure tick."""
    end = horizon_ticks if raw["failure_tick"] is None else min(horizon_ticks, raw["failure_tick"])
    changes = raw["trace"]["passage_changes"]
    probes: dict[tuple[str, str], list[tuple[int, str, float]]] = defaultdict(list)
    for tick, robot, passage, p, truth, *_ in raw["answered"]:
        group = "*" if pooled else robot
        probes[(group, passage)].append((tick, robot, p if truth == 1 else 1 - p))
    robots = ["*"] if pooled else sorted({probe[1] for probe in raw["probes"]})
    lags = []
    for passage, ticks in sorted(changes.items()):
        ordered = sorted(t for t in ticks if t < end)
        for index, start in enumerate(ordered):
            stop = ordered[index + 1] if index + 1 < len(ordered) else end
            for robot in robots:
                window = sorted(
                    (t, r, q) for t, r, q in probes.get((robot, passage), []) if start <= t < stop
                )
                run, first, found = 0, 0, None
                for t, _, q in window:
                    if q >= threshold:
                        first = t if run == 0 else first
                        run += 1
                        if run >= consecutive:
                            found = first
                            break
                    else:
                        run = 0
                lags.append(
                    LagEvent(start, (stop if found is None else found) - start, found is None)
                )
    return lags
