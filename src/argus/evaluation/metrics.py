"""Proper scores, decision loss, coverage, repair lag and runtime metrics (M6, E07-T01).

All functions are pure and evaluator-side. Conventions from the specification:

* Brier: mean (p - z)^2, binary, no factor 2, no clipping.
* NLL: -mean[z log p_eps + (1 - z) log(1 - p_eps)], p_eps = clip(p, 1e-12, 1 - 1e-12);
  the clipping rate is reported alongside.
* Jensen-Shannon between Bernoulli(p) and Bernoulli(q), natural log, range [0, ln 2].
* Reliability: 10 fixed bins [j/10, (j+1)/10), the last one includes 1; empty bins are
  None (not zero accuracy).
* Every planned probe must be scored. Probes answered with the prior because an
  entity was never seen are ordinary probes; a missing probe is an error.
* Coverage over an empty target set is undefined and raises, it is never 100 %.
* Repair lag: from an exogenous change until p(z_true) >= threshold for k consecutive
  probes; the next change or the episode end censors it. Censored lags are kept and
  summarised by Kaplan-Meier and the restricted mean, never by averaging only successes.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

EPS = 1e-12


class MetricError(ValueError):
    """Inputs cannot be scored: incomplete probes, undefined denominators, bad values."""


def _check(p: float, z: int) -> None:
    if not math.isfinite(p) or not 0.0 <= p <= 1.0:
        raise MetricError(f"probability must be in [0, 1], got {p!r}")
    if z not in (0, 1):
        raise MetricError(f"truth must be 0 or 1, got {z!r}")


def brier(probabilities: Sequence[float], truths: Sequence[int]) -> float:
    if not probabilities or len(probabilities) != len(truths):
        raise MetricError("need equally many probabilities and truths, at least one")
    for p, z in zip(probabilities, truths, strict=True):
        _check(p, z)
    return math.fsum((p - z) ** 2 for p, z in zip(probabilities, truths, strict=True)) / len(
        probabilities
    )


def nll(probabilities: Sequence[float], truths: Sequence[int]) -> tuple[float, float]:
    """Mean negative log likelihood and the fraction of clipped probabilities."""
    if not probabilities or len(probabilities) != len(truths):
        raise MetricError("need equally many probabilities and truths, at least one")
    terms, clipped = [], 0
    for p, z in zip(probabilities, truths, strict=True):
        _check(p, z)
        q = min(1 - EPS, max(EPS, p))
        clipped += q != p
        terms.append(-(z * math.log(q) + (1 - z) * math.log(1 - q)))
    return math.fsum(terms) / len(terms), clipped / len(terms)


def _xlogx_ratio(a: float, b: float) -> float:
    return 0.0 if a == 0 else a * math.log(a / b)


def js_divergence(p: float, q: float) -> float:
    """Jensen-Shannon divergence between Bernoulli(p) and Bernoulli(q), natural log."""
    for value in (p, q):
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise MetricError(f"probability must be in [0, 1], got {value!r}")
    m1, m0 = (p + q) / 2, ((1 - p) + (1 - q)) / 2
    kl_p = _xlogx_ratio(p, m1) + _xlogx_ratio(1 - p, m0)
    kl_q = _xlogx_ratio(q, m1) + _xlogx_ratio(1 - q, m0)
    return 0.5 * kl_p + 0.5 * kl_q


@dataclass(frozen=True, slots=True)
class ReliabilityBin:
    lower: float
    upper: float
    count: int
    mean_probability: float | None
    mean_truth: float | None


def reliability(
    probabilities: Sequence[float], truths: Sequence[int]
) -> tuple[ReliabilityBin, ...]:
    groups: list[list[tuple[float, int]]] = [[] for _ in range(10)]
    for p, z in zip(probabilities, truths, strict=True):
        _check(p, z)
        groups[min(9, int(p * 10))].append((p, z))
    bins = []
    for j, group in enumerate(groups):
        if group:
            mean_p = math.fsum(p for p, _ in group) / len(group)
            mean_z = sum(z for _, z in group) / len(group)
            bins.append(ReliabilityBin(j / 10, (j + 1) / 10, len(group), mean_p, mean_z))
        else:
            bins.append(ReliabilityBin(j / 10, (j + 1) / 10, 0, None, None))
    return tuple(bins)


# --- decision records -------------------------------------------------------------------


def score_records(records: Sequence[Any], plan: Sequence[Any]) -> dict[str, Any]:
    """Score evaluation records against the fixed probe plan; every probe must be present.

    Records need tick, robot_id, entity, probability, truth and decision_loss; plan items
    need tick, recipient and entity.
    """
    planned = [(r.tick, r.recipient, r.entity) for r in plan]
    answered = [(r.tick, r.robot_id, r.entity) for r in records]
    if sorted(planned) != sorted(answered):
        missing = sorted(set(planned) - set(answered))
        extra = sorted(set(answered) - set(planned))
        raise MetricError(
            f"records do not match the probe plan: missing {missing[:3]}, extra {extra[:3]}"
        )
    if not records:
        raise MetricError("an empty probe plan cannot be scored")
    probabilities = [r.probability for r in records]
    truths = [r.truth for r in records]
    nll_value, clip_rate = nll(probabilities, truths)
    total_loss = math.fsum(r.decision_loss for r in records)
    return {
        "probes": len(records),
        "brier": brier(probabilities, truths),
        "nll": nll_value,
        "nll_clip_rate": clip_rate,
        "decision_loss_total": total_loss,
        "decision_loss_mean": total_loss / len(records),
        "reliability": reliability(probabilities, truths),
    }


def summarize_episode(result: Any, plan: Sequence[Any], max_request_loss: float) -> dict[str, Any]:
    """Episode summary with explicit status; failures yield no metrics.

    A scientifically valid timeout keeps its answered probes and charges the maximal
    request loss for every unanswered one. Invalid or failed episodes are reported
    as such and are not scored.
    """
    status = result.status
    summary: dict[str, Any] = {
        "episode_id": result.episode_id,
        "status": status,
        "unanswered": result.unanswered,
        "wall_seconds": result.wall_seconds,
    }
    if status in ("invalid", "error"):
        summary["metrics"] = None
        summary["error"] = result.error
        return summary
    if status == "timeout":
        answered = {(r.tick, r.robot_id, r.entity) for r in result.records}
        plan_answered = [p for p in plan if (p.tick, p.recipient, p.entity) in answered]
        metrics = score_records(result.records, plan_answered) if result.records else None
        summary["metrics"] = metrics
        summary["timeout_loss_total"] = (
            math.fsum(r.decision_loss for r in result.records)
            + max_request_loss * result.unanswered
        )
        return summary
    summary["metrics"] = score_records(result.records, plan)
    return summary


# --- coverage -------------------------------------------------------------------------------


def coverage(
    targets: set[tuple[int, int]],
    seen_by_tick: Sequence[tuple[int, Iterable[tuple[int, int]]]],
    horizon_ticks: int,
) -> tuple[float, float]:
    """Final covered fraction of fixed target cells and its time-average (AUC in [0, 1])."""
    if not targets:
        raise MetricError("coverage over an empty target set is undefined")
    if horizon_ticks < 1:
        raise MetricError("horizon must be at least one tick")
    covered: set[tuple[int, int]] = set()
    area = 0.0
    last_tick, last_fraction = 0, 0.0
    for tick, cells in sorted(seen_by_tick, key=lambda item: item[0]):
        if tick > horizon_ticks:
            break
        area += last_fraction * (tick - last_tick)
        covered |= set(cells) & targets
        last_tick, last_fraction = tick, len(covered) / len(targets)
    area += last_fraction * (horizon_ticks - last_tick)
    return len(covered) / len(targets), area / horizon_ticks


# --- repair lag -------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LagEvent:
    change_tick: int
    lag_ticks: int
    censored: bool


def repair_lags(
    change_ticks: Sequence[int],
    probes: Sequence[tuple[int, float]],
    end_tick: int,
    threshold: float = 0.8,
    consecutive: int = 3,
) -> tuple[LagEvent, ...]:
    """Lag after each change until p(z_true) >= threshold on `consecutive` probes in a row.

    probes are (tick, probability assigned to the true state). The lag ends at the
    first probe of the successful run. The next change or end_tick censors the event.
    """
    changes = sorted(change_ticks)
    events = []
    for index, start in enumerate(changes):
        stop = changes[index + 1] if index + 1 < len(changes) else end_tick
        window = [(t, p) for t, p in sorted(probes) if start <= t < stop]
        run_start, run = None, 0
        found = None
        for t, p in window:
            if p >= threshold:
                run_start = t if run == 0 else run_start
                run += 1
                if run >= consecutive:
                    found = run_start
                    break
            else:
                run = 0
        if found is None:
            events.append(LagEvent(start, stop - start, True))
        else:
            events.append(LagEvent(start, found - start, False))
    return tuple(events)


def kaplan_meier(events: Sequence[LagEvent]) -> tuple[tuple[int, float], ...]:
    """Survival S(t) of 'not yet repaired' at each observed repair time."""
    at_risk = len(events)
    survival = 1.0
    curve = []
    for t in sorted({e.lag_ticks for e in events}):
        repaired = sum(1 for e in events if e.lag_ticks == t and not e.censored)
        if repaired:
            survival *= 1 - repaired / at_risk
            curve.append((t, survival))
        at_risk -= sum(1 for e in events if e.lag_ticks == t)
    return tuple(curve)


def restricted_mean_lag(events: Sequence[LagEvent], tau: int) -> float:
    """Area under the Kaplan-Meier curve up to tau (restricted mean time to repair)."""
    if not events:
        raise MetricError("no lag events")
    area, last_t, survival = 0.0, 0, 1.0
    for t, s in kaplan_meier(events):
        if t >= tau:
            break
        area += survival * (t - last_t)
        last_t, survival = t, s
    return area + survival * (tau - last_t)


def censoring_rate(events: Sequence[LagEvent]) -> float:
    if not events:
        raise MetricError("no lag events")
    return sum(e.censored for e in events) / len(events)


# --- runtime ------------------------------------------------------------------------------------


def runtime_quantiles(seconds: Sequence[float]) -> dict[str, float]:
    """Nearest-rank p50/p95/p99 of measured durations (deterministic, no interpolation)."""
    if not seconds:
        raise MetricError("no runtime samples")
    ordered = sorted(seconds)

    def rank(q: float) -> float:
        return ordered[max(0, math.ceil(q * len(ordered)) - 1)]

    return {"p50": rank(0.50), "p95": rank(0.95), "p99": rank(0.99), "n": float(len(ordered))}
