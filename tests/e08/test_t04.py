"""ARGUS-E08-T04: second sensor family and the X3 run set (amendment 04)."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from argus.core.events import KeyedRandom
from argus.evaluation.statistics import AnalysisError
from argus.experiments.cells import PREREGISTERED, CellParameters, FactorCell, apply_cell
from argus.experiments.protocol import episode_config, manifest_params
from argus.experiments.runner import EpisodeConfig, SensorBias, build_noise
from argus.experiments.runset import ImmutableRunSet, RunSetError
from argus.sensors.correlated import SharedBiasNoise, matched_private_error
from argus.sensors.ood_field import GaussianThresholdField

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import analyze_x3  # noqa: E402
import x3_batch  # noqa: E402

RHO = 0.05


def field_for(seed: int, length: float = 2.0, tau: float = 20.0) -> GaussianThresholdField:
    return GaussianThresholdField(RHO, length, tau, 0.1, KeyedRandom(seed), "scenario")


def test_the_marginal_bias_rate_is_rho_over_independent_fields() -> None:
    n = 4000
    hits = sum(field_for(s).bias_at((3.0, 5.0), 700) for s in range(n))
    assert abs(hits / n - RHO) <= 5 * math.sqrt(RHO * (1 - RHO) / n)


def test_the_field_value_is_standard_normal() -> None:
    values = np.array([field_for(s).value((1.0, 2.0), 0) for s in range(4000)])
    assert abs(values.mean()) < 0.08
    assert abs(values.std() - 1.0) < 0.05


def test_the_matched_reading_keeps_the_marginal_error() -> None:
    q = matched_private_error(0.14, RHO)
    errors, n = 0, 6000
    for seed in range(n):
        noise = SharedBiasNoise(field_for(seed), q, KeyedRandom(seed))
        errors += noise.reading("r0", "p0", 3, 15, 1, (4.0, 4.0)) != 1
    assert abs(errors / n - 0.14) <= 5 * math.sqrt(0.14 * 0.86 / n)


def test_the_field_does_not_depend_on_the_query_order() -> None:
    a, b = field_for(7), field_for(7)
    late, early = a.value((2.0, 2.0), 900), a.value((2.0, 2.0), 10)
    assert b.value((2.0, 2.0), 10) == early
    assert b.value((2.0, 2.0), 900) == late


def test_correlation_falls_with_distance_and_time() -> None:
    fields = [field_for(s) for s in range(1500)]
    here = np.array([f.value((5.0, 5.0), 0) for f in fields])

    def corr(point: tuple[float, float], tick: int) -> float:
        other = np.array([f.value(point, tick) for f in fields])
        return float(np.corrcoef(here, other)[0, 1])

    assert corr((6.0, 5.0), 0) > 0.75  # exp(-1/8) = .88
    assert abs(corr((13.0, 5.0), 0)) < 0.12
    assert abs(corr((5.0, 5.0), 200) - math.exp(-1.0)) < 0.08  # 20 s = tau


def test_the_gauss_condition_changes_the_truth_but_not_the_nominal_model() -> None:
    cell = FactorCell(True, True, False)
    config = _config()
    cells = apply_cell(config, cell)
    gauss = apply_cell(config, cell, CellParameters(sensor_family="gauss"))
    assert gauss.nominal == cells.nominal
    assert gauss.sensor_error == cells.sensor_error
    assert build_noise(cells)[0] == "binary-shared-bias-v1"
    assert build_noise(gauss)[0] == "binary-gauss-threshold-v1"
    assert isinstance(build_noise(gauss)[1].bias, GaussianThresholdField)  # type: ignore[union-attr]


def test_the_default_family_keeps_the_historical_configuration() -> None:
    config = apply_cell(_config(), FactorCell(True, True, False), PREREGISTERED)
    assert config.to_json()["sensor_bias"] == [0.05, 0.05, 4.0]
    with pytest.raises(ValueError):
        SensorBias(0.05, family="smoke")


def test_rho_10_matches_the_marginal_error() -> None:
    assert matched_private_error(0.14, 0.1) == pytest.approx(0.05)


def _config() -> EpisodeConfig:
    runset = ImmutableRunSet.load(ROOT)
    entry = runset.buildings("calib", 1)[0][0]
    params = manifest_params(ROOT, runset.prereg)
    return episode_config(entry, params, FactorCell(False, False, False), "no_comm", "c0", {}, 0)


@pytest.fixture(scope="module")
def frozen() -> tuple[ImmutableRunSet, dict, str]:
    runset = ImmutableRunSet.load(ROOT)
    x3, sha = x3_batch.load_frozen(ROOT, runset)
    return runset, x3, sha


def test_x3_uses_only_buildings_x1x2_never_ran(frozen: tuple) -> None:
    runset, data, _ = frozen
    x3 = data["x3"]
    used = {e.building_id for e in runset.buildings("id_test", 80)[0]}
    used |= {e.building_id for e in runset.buildings("ood_test", 80)[0]}
    for split in ("id_test", "ood_test"):
        entries, _ = x3_batch.fresh_entries(runset, x3, split, None)
        assert len(entries) == x3["b"]
        assert not used & {e.building_id for e in entries}
        assert min(e.position for e in entries) == 80


def test_x3_jobs_are_paired_and_keep_the_frozen_configurations(
    frozen: tuple, tmp_path: Path
) -> None:
    runset, data, sha = frozen
    x3 = data["x3"]
    entries = {
        s: x3_batch.fresh_entries(runset, x3, s, None)[0][:1] for s in ("id_test", "ood_test")
    }
    jobs = x3_batch.make_jobs(
        runset, sha, x3, entries, x3["conditions"], x3["replicates"], tmp_path, "c", "a"
    )
    assert len(jobs) == 5 * 4 * 7 * 2
    assert len({j.key for j in jobs}) == len(jobs)
    assert len({j.group for j in jobs}) == 2
    selected = {m: c for m, c, _ in runset.methods()}
    assert all(j.payload["config_id"] == selected[j.payload["method_id"]] for j in jobs)
    assert {j.payload["cell_id"] for j in jobs} == set(x3["cells"])
    assert all(j.payload["approval_path"] == x3_batch.X3_APPROVAL_PATH for j in jobs)


def test_x3_refuses_to_run_without_an_approval(frozen: tuple) -> None:
    _, _, sha = frozen
    if (ROOT / x3_batch.X3_APPROVAL_PATH).exists():
        with pytest.raises(RunSetError, match="another frozen"):
            x3_batch.approval("0" * 64)
    else:
        with pytest.raises(RunSetError, match="no approval"):
            x3_batch.approval(sha)


def test_the_analysis_is_descriptive_and_reports_no_p_values() -> None:
    amendment = json.loads((ROOT / x3_batch.AMENDMENT_PATH).read_text(encoding="utf-8"))
    assert "no p-values" in amendment["decisions"]["B7 status"]
    assert amendment["x3"]["first_position"] == 80


def test_the_analysis_guard_refuses_interim_and_test_dry_runs(tmp_path: Path) -> None:
    with pytest.raises(AnalysisError, match="dry run"):
        analyze_x3.guard(tmp_path, "x", {"id_test"}, True)
    with pytest.raises(AnalysisError, match="test lists"):
        analyze_x3.guard(tmp_path, "x", {"calib"}, False)
    (tmp_path / "PAUSE").touch()
    with pytest.raises(AnalysisError, match="PAUSE"):
        analyze_x3.guard(tmp_path, "x", {"id_test"}, False)
