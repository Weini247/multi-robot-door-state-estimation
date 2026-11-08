"""X3 robustness batch (ARGUS-E08-T04, amendment 04).

    python tools/x3_batch.py freeze [--check]
    python tools/x3_batch.py probe --out runs/x3_probe_calib_vN
    python tools/x3_batch.py run --out runs/x3_v1          # needs the approval file
    python tools/x3_batch.py status --out <dir>

Runs the selected X1/X2 configurations unchanged on fresh test buildings under the
conditions of amendment 04. `probe` uses calib buildings only. `run` refuses unless
configs/experiments/X3_approval.json names the frozen hash, the source tree and a budget.
Stop with a PAUSE file in the output directory; a rerun continues.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from test_batch import (
    PROBE_CAPS,
    _summary_line,
    code_commit,
    log_session,
    scenario_valid,
    status,
)

from argus.experiments.batch import BatchCaps, BatchJob, BatchRunner
from argus.experiments.cells import CELLS
from argus.experiments.protocol import file_sha256, manifest_params
from argus.experiments.runset import (
    HARD_TIMEOUT_S,
    ImmutableRunSet,
    RunSetError,
    killed_episode,
    source_sha256,
    test_episode,
    text_sha256,
)
from argus.experiments.splits import SplitEntry

ROOT = Path(__file__).resolve().parents[1]
AMENDMENT_PATH = "configs/experiments/preregistration_v1_amendment_04.json"
X3_FROZEN_PATH = "configs/experiments/X3_frozen.json"
X3_APPROVAL_PATH = "configs/experiments/X3_approval.json"
X3_ANALYSIS_CODE = ("src/argus/evaluation/statistics.py", "tools/analyze_x3.py")
SENSOR_CODE = "src/argus/sensors/ood_field.py"
PROBE_POSITIONS = (0, 1)
WORKERS = 5


def render_frozen(root: Path, base_sha256: str) -> str:
    """The X3 run set: amendment 04 bound to the X1/X2 run set and the analysis code."""
    amendment = json.loads((root / AMENDMENT_PATH).read_text(encoding="utf-8"))
    frozen = {
        "schema": "argus-x3-frozen-v1",
        "status": "frozen before any X3 test episode; a run also needs the approval file",
        "base_frozen_sha256": base_sha256,
        "amendment_sha256": text_sha256(root / AMENDMENT_PATH),
        "sensor_code_sha256": text_sha256(root / SENSOR_CODE),
        "analysis_code": {path: text_sha256(root / path) for path in X3_ANALYSIS_CODE},
        "x3": amendment["x3"],
    }
    return json.dumps(frozen, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def load_frozen(root: Path, runset: ImmutableRunSet) -> tuple[dict[str, Any], str]:
    text = (root / X3_FROZEN_PATH).read_text(encoding="utf-8")
    if text != render_frozen(root, runset.sha256):
        raise RunSetError(f"{X3_FROZEN_PATH} does not match the committed sources")
    return json.loads(text), hashlib.sha256(text.encode("utf-8")).hexdigest()


def fresh_entries(
    runset: ImmutableRunSet,
    x3: Mapping[str, Any],
    split: str,
    valid: Callable[[SplitEntry], bool] | None,
) -> tuple[list[SplitEntry], list[tuple[str, str]]]:
    """Slots first_position .. first_position + b - 1 of a list. Positions before that
    belong to X1/X2 and are never used; a replacement comes from behind the slots."""
    first = x3["first_position"]
    chosen, replaced = runset.buildings(split, first + x3["b"], valid)
    return chosen[first:], replaced


def make_jobs(
    runset: ImmutableRunSet,
    frozen_sha256: str,
    x3: Mapping[str, Any],
    entries: Mapping[str, Sequence[SplitEntry]],
    conditions: Mapping[str, Mapping[str, Any]],
    replicates: int,
    out: Path,
    commit: str,
    approval_sha256: str | None,
) -> list[BatchJob]:
    """Per slot each list; per building the conditions, replicate, method and cell. A group
    is one building with all its conditions, so a stop leaves complete paired buildings."""
    params = manifest_params(runset.root, runset.prereg).to_json()
    src = source_sha256(runset.root)
    cells = [c for c in CELLS if c.cell_id in x3["cells"]]
    jobs = []
    slots = max(len(v) for v in entries.values())
    for slot in range(slots):
        for split, listed in entries.items():
            entry = listed[slot]
            for name, condition in conditions.items():
                if condition["list"] != split:
                    continue
                for replicate in range(replicates):
                    for method, config_id, values in runset.methods():
                        for cell in cells:
                            key = (
                                f"{name}/{split}/{entry.building_id}/{cell.cell_id}/"
                                f"{method}/{config_id}/{replicate}"
                            )
                            payload = {
                                "entry": list(entry.to_json()),
                                "allowed_splits": [split],
                                "params": params,
                                "cell_id": cell.cell_id,
                                "cell_params": dict(condition["cell_params"]),
                                "method_id": method,
                                "config_id": config_id,
                                "method_params": values,
                                "replicate": replicate,
                                "key": key,
                                "out": str(out),
                                "frozen_sha256": frozen_sha256,
                                "code_commit": commit,
                                "root": str(runset.root),
                                "src_sha256": src,
                                "approval_sha256": approval_sha256,
                                "approval_path": X3_APPROVAL_PATH,
                            }
                            jobs.append(BatchJob(key, f"{split}/{entry.building_id}", payload))
    return jobs


def probe(out: Path) -> None:
    runset = ImmutableRunSet.load(ROOT)
    frozen, sha = load_frozen(ROOT, runset)
    x3 = frozen["x3"]
    calib, _ = runset.buildings("calib", max(PROBE_POSITIONS) + 1)
    conditions = {k: {**v, "list": "calib"} for k, v in x3["conditions"].items()}
    jobs = make_jobs(runset, sha, x3, {"calib": calib}, conditions, 1, out, code_commit(), None)
    caps = dataclasses.replace(PROBE_CAPS, wall_hours=1.0, cpu_hours=3.0)
    summary = BatchRunner(out, caps, test_episode, killed_episode, strict=True).run(jobs)
    log_session(out, summary, code_commit())
    print(_summary_line(summary))


def approval(sha: str) -> dict[str, Any]:
    path = ROOT / X3_APPROVAL_PATH
    if not path.exists():
        raise RunSetError(f"no approval: {X3_APPROVAL_PATH} is missing; X3 is not run")
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if data.get("frozen_sha256") != sha:
        raise RunSetError("the approval refers to another frozen run set")
    for key in ("approved_by", "date", "cpu_hours", "wall_hours", "src_sha256"):
        if key not in data:
            raise RunSetError(f"approval lacks {key}")
    if data["src_sha256"] != source_sha256(ROOT):
        raise RunSetError("the approval names another source tree")
    return data


def run(out: Path) -> None:
    runset = ImmutableRunSet.load(ROOT)
    frozen, sha = load_frozen(ROOT, runset)
    x3 = frozen["x3"]
    commit = code_commit()
    granted = approval(sha)
    params = manifest_params(ROOT, runset.prereg)
    lists = sorted({c["list"] for c in x3["conditions"].values()})
    picked = {s: fresh_entries(runset, x3, s, lambda e: scenario_valid(e, params)) for s in lists}
    approval_sha = file_sha256(ROOT / X3_APPROVAL_PATH)
    plan = {
        "frozen_sha256": sha,
        "approval_sha256": approval_sha,
        "src_sha256": granted["src_sha256"],
        "b": x3["b"],
        "replacements": json.loads(json.dumps({s: picked[s][1] for s in lists})),
    }
    plan_path = out / "run_plan.json"
    if plan_path.exists():
        first = json.loads(plan_path.read_text(encoding="utf-8"))
        changed = {k for k in plan if first.get(k) != plan[k]}
        if changed:
            raise RunSetError(f"resume differs from the first session in {sorted(changed)}")
    else:
        out.mkdir(parents=True, exist_ok=True)
        record = {**plan, "first_commit": commit, "approval": granted}
        plan_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    jobs = make_jobs(
        runset,
        sha,
        x3,
        {s: picked[s][0] for s in lists},
        x3["conditions"],
        x3["replicates"],
        out,
        commit,
        approval_sha,
    )
    caps = BatchCaps(granted["wall_hours"], granted["cpu_hours"], WORKERS, HARD_TIMEOUT_S)
    summary = BatchRunner(out, caps, test_episode, killed_episode, strict=True).run(jobs)
    log_session(out, summary, commit)
    print(_summary_line(summary))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["freeze", "probe", "run", "status"])
    parser.add_argument("--out", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.command == "freeze":
        text = render_frozen(ROOT, ImmutableRunSet.load(ROOT).sha256)
        target = ROOT / X3_FROZEN_PATH
        if args.check:
            if target.read_text(encoding="utf-8") != text:
                raise SystemExit(f"{X3_FROZEN_PATH} is stale")
            print(f"{X3_FROZEN_PATH} current")
        else:
            target.write_bytes(text.encode("utf-8"))
            print(f"wrote {X3_FROZEN_PATH}")
    elif args.command == "status":
        status(args.out)
    elif args.out is None:
        raise SystemExit(f"{args.command} needs --out")
    elif args.command == "probe":
        probe(args.out)
    else:
        run(args.out)


if __name__ == "__main__":
    main()
