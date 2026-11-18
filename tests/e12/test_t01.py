"""ARGUS-E12-T01: reproduction check from a clean checkout."""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import repro_compare as rc  # noqa: E402


def _batch(out: Path, episodes: dict[str, dict], meta: dict) -> Path:
    (out / "raw").mkdir(parents=True)
    with (out / "results.jsonl").open("w", encoding="utf-8") as handle:
        for key, values in episodes.items():
            raw = f"raw/{key}.json.gz"
            record = {
                "code_commit": meta["commit"],
                "trace": values["trace"],
                "manifest": {
                    "run_id": meta["run_id"],
                    "platform": meta["platform"],
                    "created_utc": meta["created"],
                    "code_commit": meta["commit"],
                    "python": "3.12",
                },
            }
            (out / raw).write_bytes(gzip.compress(json.dumps(record).encode("utf-8")))
            row = {
                "key": key,
                "raw": raw,
                "brier": values["brier"],
                "wall_s": meta["wall"],
                "cpu_s": meta["wall"],
                "raw_sha256": meta["run_id"],
            }
            handle.write(json.dumps(row) + "\n")
    return out


A = {"commit": "a" * 40, "run_id": "r1", "platform": "Windows-26200", "created": "t1", "wall": 1.0}
B = {"commit": "b" * 40, "run_id": "r2", "platform": "Windows-26300", "created": "t2", "wall": 2.0}
EPISODES = {"e1": {"trace": [1, 2], "brier": 0.2}, "e2": {"trace": [3], "brier": 0.1}}


def test_run_metadata_and_timing_do_not_count_as_differences(tmp_path: Path) -> None:
    result = rc.compare(_batch(tmp_path / "a", EPISODES, A), _batch(tmp_path / "b", EPISODES, B))
    assert result["passed"] and result["reproduced_episodes"] == 2


def test_result_and_trace_changes_are_reported(tmp_path: Path) -> None:
    changed = {"e1": {"trace": [1, 9], "brier": 0.2}, "e2": {"trace": [3], "brier": 0.15}}
    result = rc.compare(_batch(tmp_path / "a", EPISODES, A), _batch(tmp_path / "b", changed, A))
    assert not result["passed"]
    assert result["raw_differences"] == ["e1"]
    assert result["row_differences"] == {"e2": ["brier"]}
    assert result["reproduced_episodes"] == 0


def test_missing_episodes_fail_unless_a_sample_is_declared(tmp_path: Path) -> None:
    a = _batch(tmp_path / "a", EPISODES, A)
    b = _batch(tmp_path / "b", {"e2": EPISODES["e2"]}, B)
    assert rc.compare(a, b)["missing_count"] == 1 and not rc.compare(a, b)["passed"]
    assert rc.compare(a, b, sample=True)["passed"]


def test_recorded_reproductions_passed() -> None:
    s2 = json.loads((ROOT / "docs/results/REPRO_S2.json").read_text(encoding="utf-8"))
    assert s2["episodes"]["passed"] and not s2["episodes"].get("sample")
    assert s2["episodes"]["reproduced_episodes"] == 12000 and s2["analysis"]["identical"]
    for name in ("REPRO_X1X2_SAMPLE.json", "REPRO_X3_SAMPLE.json"):
        sample = json.loads((ROOT / "docs/results" / name).read_text(encoding="utf-8"))["episodes"]
        assert sample["passed"] and sample["sample"] and sample["reproduced_episodes"] >= 30
