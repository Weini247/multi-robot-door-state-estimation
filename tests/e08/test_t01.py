"""ARGUS-E08-T01: batch runner with resource caps and sample-size planning."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

from argus.experiments.batch import (
    BatchCaps,
    BatchJob,
    BatchRunner,
    BatchUsage,
    building_losses,
    peak_rss_bytes,
    plan_cost,
    plan_test_size,
    probe_worker,
)
from argus.experiments.protocol import load_manifest_entries, load_preregistration

ROOT = Path(__file__).resolve().parents[2]
CAPS = BatchCaps(wall_hours=1.0, cpu_hours=1.0, workers=3, episode_timeout_s=60.0)


def rows(out: Path) -> list[dict]:
    text = (out / "results.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line]


# --- runner ---------------------------------------------------------------------


def test_one_process_per_episode_with_cpu_and_rss_measured(tmp_path: Path) -> None:
    jobs = [
        BatchJob("small", "g0", {"alloc_mb": 1}),
        BatchJob("large", "g0", {"alloc_mb": 64, "cpu_s": 0.2}),
        BatchJob("broken", "g0", {"fail": True}),
    ]
    summary = BatchRunner(tmp_path, CAPS, probe_worker).run(jobs)
    got = {r["key"]: r for r in rows(tmp_path)}
    assert summary.ran == 3 and summary.completed_groups == ("g0",)
    assert got["small"]["status"] == got["large"]["status"] == "ok"
    assert got["broken"]["status"] == "error" and "probe failure" in got["broken"]["error"]
    assert got["large"]["cpu_s"] >= 0.2
    assert got["large"]["peak_rss_bytes"] - got["small"]["peak_rss_bytes"] > 50 * 2**20
    usage = BatchUsage.load(tmp_path / "usage.json")
    assert usage.cpu_s == pytest.approx(sum(r["cpu_s"] for r in got.values()), abs=1e-6)
    assert peak_rss_bytes() is not None


def test_hard_timeout_kills_and_keeps_the_row(tmp_path: Path) -> None:
    caps = BatchCaps(wall_hours=1.0, cpu_hours=1.0, workers=2, episode_timeout_s=2.0)
    BatchRunner(tmp_path, caps, probe_worker).run(
        [BatchJob("slow", "g0", {"sleep_s": 60}), BatchJob("fast", "g0", {})]
    )
    got = {r["key"]: r for r in rows(tmp_path)}
    assert got["slow"]["status"] == "timeout" and got["fast"]["status"] == "ok"
    assert 2.0 <= got["slow"]["process_wall_s"] < 20.0
    assert got["slow"]["cpu_s"] >= 2.0  # charged as an upper bound


def test_resume_skips_done_keys_and_accumulates_usage(tmp_path: Path) -> None:
    jobs = [BatchJob(f"j{i}", "g0", {}) for i in range(2)]
    BatchRunner(tmp_path, CAPS, probe_worker).run(jobs[:1])
    first = BatchUsage.load(tmp_path / "usage.json")
    summary = BatchRunner(tmp_path, CAPS, probe_worker).run(jobs)
    usage = BatchUsage.load(tmp_path / "usage.json")
    assert summary.ran == 1 and summary.skipped == 1
    assert [r["key"] for r in rows(tmp_path)] == ["j0", "j1"]
    assert usage.sessions == 2 and usage.wall_s > first.wall_s and usage.cpu_s >= first.cpu_s
    with pytest.raises(ValueError):
        BatchRunner(tmp_path, CAPS, probe_worker).run(jobs + jobs[:1])


def test_a_group_starts_only_if_the_projection_fits_across_sessions(tmp_path: Path) -> None:
    caps = BatchCaps(wall_hours=1.0, cpu_hours=0.5 / 3600, workers=2, episode_timeout_s=60.0)
    jobs = [BatchJob(f"{g}/{i}", g, {"cpu_s": 0.3}) for g in ("a", "b") for i in range(1)]
    summary = BatchRunner(tmp_path, caps, probe_worker).run(jobs)
    assert summary.completed_groups == ("a",) and summary.stopped_before == "b"
    # a new session sees the persisted usage and still refuses
    again = BatchRunner(tmp_path, caps, probe_worker).run(jobs)
    assert again.ran == 0 and again.stopped_before == "b"


# --- planning -------------------------------------------------------------------


FAMILY = {f"b{i}": ("tree" if i % 2 else "loop") for i in range(20)}


def test_constant_delta_gives_zero_sd_and_the_minimum_b() -> None:
    ctrl = {b: 0.3 + 0.01 * i for i, b in enumerate(FAMILY)}
    cand = {b: 0.9 * v for b, v in ctrl.items()}  # constant ratio: influence is zero
    size = plan_test_size(cand, ctrl, FAMILY)
    assert size.gain == pytest.approx(0.1)
    assert size.sd_used == pytest.approx(0.0, abs=1e-12) and size.b_planned == 60


def test_b_follows_the_preregistered_formula() -> None:
    ctrl = {b: 0.3 for b in FAMILY}
    cand = {b: 0.27 + (0.06 if i % 4 < 2 else -0.06) for i, b in enumerate(FAMILY)}
    size = plan_test_size(cand, ctrl, FAMILY)
    sd = math.sqrt(sum((0.06 / 0.3) ** 2 for _ in FAMILY) / (len(FAMILY) - 1))
    assert size.sd_pooled == pytest.approx(sd)
    assert size.b_formula == pytest.approx((2 * 1.96 * size.sd_used / 0.05) ** 2)
    assert size.b_planned == max(60, math.ceil(size.b_formula)) + (
        max(60, math.ceil(size.b_formula)) % 2
    )
    assert size.b_planned % 2 == 0 and size.b_planned > 60
    assert size.ci_half_width_at_b <= 0.025 + 1e-12


def test_incomplete_buildings_are_left_out_of_the_pilot() -> None:
    data = [
        {"method_id": "m", "building_id": b, "mean_loss": 0.1 * k}
        for b, n in (("x", 4), ("y", 2))
        for k in range(n)
    ]
    assert building_losses(data, "m", 2) == {"x": pytest.approx(0.15)}
    with pytest.raises(ValueError):
        building_losses(data[:3], "m", 2)


def test_cost_comes_from_measured_cpu_seconds() -> None:
    cost = plan_cost(60, {"a": 10.0, "b": 20.0}, 8, 2, 2, 0.25)
    assert cost["episodes"] == 60 * 8 * 2 * 2 * 2
    assert cost["cpu_hours"] == pytest.approx(60 * 8 * 2 * 2 * 30 / 3600)
    assert cost["wall_hours"] == pytest.approx(cost["cpu_hours"] / 4)


# --- pilot design and committed result --------------------------------------------


def test_pilot_runs_calib_only_with_the_test_design() -> None:
    from tools.calib_pilot import jobs, pair

    CANDIDATE, CONTROL = pair("v1")  # the committed v1 pilot (before amendment 01)

    prereg = load_preregistration(ROOT)
    entries = {e.building_id: e for e in load_manifest_entries(ROOT, prereg)}
    planned = jobs(prereg, "v1")
    assert len(planned) == 20 * 2 * 2 * 8 and len({j.key for j in planned}) == len(planned)
    assert {entries[j.group].split for j in planned} == {"calib"}
    assert {(j.payload["method_id"], j.payload["config_id"]) for j in planned} == {
        CANDIDATE,
        CONTROL,
    }
    assert {j.payload["replicate"] for j in planned} == {0, 1}


def test_committed_pilot_reproduces_the_plan() -> None:
    from tools.calib_pilot import plan

    out = ROOT / "docs/results/calib_pilot_v1"
    report = (ROOT / "docs/results/CALIB_PILOT_v1.md").read_text(encoding="utf-8")
    assert hashlib.sha256((out / "results.jsonl").read_bytes()).hexdigest() in report
    result = plan(out, None)
    assert result["pilot_episodes"] == 640
    assert f"**B = {result['size']['b_planned']}**" in report
    assert result["usage"]["wall_h"] <= 1.0 and result["usage"]["cpu_h"] <= 3.0
