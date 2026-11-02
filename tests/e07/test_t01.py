"""ARGUS-E07-T01: proper scores, probe completeness, coverage, censored lags, runtime."""

from __future__ import annotations

import dataclasses
import math

import pytest

from argus.evaluation.metrics import (
    LagEvent,
    MetricError,
    brier,
    censoring_rate,
    coverage,
    js_divergence,
    kaplan_meier,
    nll,
    reliability,
    repair_lags,
    restricted_mean_lag,
    runtime_quantiles,
    score_records,
    summarize_episode,
)
from argus.experiments.runner import ExperimentRunner, generate_requests
from tests.e05.test_t03 import config


def test_brier_nll_js_reference_values() -> None:
    assert brier([0.5], [1]) == 0.25
    assert brier([1.0, 0.0], [0, 1]) == 1.0
    value, clipped = nll([0.5], [1])
    assert value == pytest.approx(math.log(2), abs=1e-15) and clipped == 0.0
    assert nll([0.5], [0])[0] == pytest.approx(math.log(2), abs=1e-15)
    assert js_divergence(0.0, 1.0) == pytest.approx(math.log(2), abs=1e-15)
    assert js_divergence(0.3, 0.3) == 0.0
    assert 0 < js_divergence(0.2, 0.7) < math.log(2)


def test_nll_clips_and_reports_the_rate() -> None:
    value, clipped = nll([1.0, 0.0, 0.5], [0, 0, 1])
    assert clipped == pytest.approx(2 / 3)
    assert value == pytest.approx((-math.log(1e-12) + -math.log(1 - 1e-12) + math.log(2)) / 3)


def test_reliability_bins_include_one_in_the_last_bin_and_leave_empty_bins_na() -> None:
    bins = reliability([1.0, 0.95, 0.05, 0.1], [1, 1, 0, 0])
    assert bins[9].count == 2 and bins[9].mean_truth == 1.0
    assert bins[0].count == 1 and bins[1].count == 1  # 0.1 belongs to [0.1, 0.2)
    assert bins[5].count == 0 and bins[5].mean_probability is None and bins[5].mean_truth is None


@pytest.mark.parametrize("bad", [([1.2], [1]), ([0.5], [2]), ([math.nan], [0]), ([], [])])
def test_invalid_score_inputs_raise(bad: tuple[list[float], list[int]]) -> None:
    with pytest.raises(MetricError):
        brier(*bad)


def test_records_are_scored_against_the_full_probe_plan() -> None:
    cfg = config("wait")
    result = ExperimentRunner().run(cfg)
    plan = generate_requests(cfg)
    metrics = score_records(result.records, plan)
    assert metrics["probes"] == len(plan) == 32
    # Unknown entities answered with the prior are scored, not skipped.
    priors = [r for r in result.records if r.probability == pytest.approx(1 / 3)]
    assert priors and metrics["probes"] == len(result.records)
    with pytest.raises(MetricError, match="missing"):
        score_records(result.records[1:], plan)


def test_censored_timeouts_and_failures_are_explicit() -> None:
    cfg = config("wait")
    plan = generate_requests(cfg)
    ok = ExperimentRunner().run(cfg)
    timed_out = dataclasses.replace(
        ok, status="timeout", records=ok.records[:10], unanswered=len(plan) - 10
    )
    summary = summarize_episode(timed_out, plan, max_request_loss=4.0)
    assert summary["status"] == "timeout" and summary["unanswered"] == 22
    expected = math.fsum(r.decision_loss for r in ok.records[:10]) + 4.0 * 22
    assert summary["timeout_loss_total"] == pytest.approx(expected)
    failed = dataclasses.replace(ok, status="error", records=(), error="boom")
    assert summarize_episode(failed, plan, 4.0)["metrics"] is None
    assert summarize_episode(ok, plan, 4.0)["metrics"]["probes"] == 32


def test_coverage_and_auc() -> None:
    targets = {(0, 0), (1, 0), (2, 0), (3, 0)}
    final, auc = coverage(targets, [(0, [(0, 0)]), (5, [(1, 0), (9, 9)]), (10, [(2, 0)])], 20)
    assert final == 0.75
    assert auc == pytest.approx((0.25 * 5 + 0.5 * 5 + 0.75 * 10) / 20)
    with pytest.raises(MetricError):
        coverage(set(), [], 10)


def test_repair_lag_is_censored_at_the_next_change_and_end() -> None:
    probes = [(t, 0.9 if t >= 30 else 0.4) for t in range(0, 100, 5)]
    events = repair_lags([10, 60], probes, end_tick=100, threshold=0.8, consecutive=3)
    assert events[0] == LagEvent(10, 20, False)  # first of three consecutive at tick 30
    assert events[1] == LagEvent(60, 0, False)  # already confident when the change happens
    never = repair_lags([10, 50], [(t, 0.5) for t in range(0, 100, 5)], 100)
    assert [e.censored for e in never] == [True, True]
    assert [e.lag_ticks for e in never] == [40, 50]
    assert censoring_rate(never) == 1.0


def test_kaplan_meier_and_restricted_mean_do_not_drop_censored_events() -> None:
    events = [LagEvent(0, 10, False), LagEvent(0, 20, True), LagEvent(0, 30, False)]
    assert kaplan_meier(events) == ((10, pytest.approx(2 / 3)), (30, pytest.approx(0.0)))
    # Area: 1 * 10 + 2/3 * 20 = 23.33 up to tau 30.
    assert restricted_mean_lag(events, 30) == pytest.approx(10 + 2 / 3 * 20)
    success_only_mean = (10 + 30) / 2
    assert restricted_mean_lag(events, 30) != pytest.approx(success_only_mean)


def test_runtime_quantiles_nearest_rank() -> None:
    q = runtime_quantiles([float(x) for x in range(1, 101)])
    assert (q["p50"], q["p95"], q["p99"], q["n"]) == (50.0, 95.0, 99.0, 100.0)
    with pytest.raises(MetricError):
        runtime_quantiles([])
