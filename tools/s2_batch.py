"""Study 2 batch: mechanism of targeted re-observation (ARGUS-E09-T02).

    python tools/s2_batch.py freeze [--check]
    python tools/s2_batch.py probe --out runs/s2_probe_calib_vN
    python tools/s2_batch.py run --out runs/s2_v1          # needs the approval file
    python tools/s2_batch.py status --out <dir>

Arms of the unchanged repair policy and two anchors run on fresh buildings
(configs/splits/splits_v2.json) under door tempo and cost-ratio interventions
(src/argus/replay/interventions.py). `probe` uses calib buildings of splits_v1 only.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from test_batch import PROBE_CAPS, _summary_line, code_commit, log_session, scenario_valid, status

from argus.experiments.batch import BatchCaps, BatchJob, BatchRunner
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
from argus.replay.interventions import InterventionSpec

ROOT = Path(__file__).resolve().parents[1]
PREREG_PATH = "configs/experiments/preregistration_s2.json"
S2_FROZEN_PATH = "configs/experiments/S2_frozen.json"
S2_APPROVAL_PATH = "configs/experiments/S2_approval.json"
S2_ANALYSIS_CODE = ("src/argus/evaluation/statistics.py", "tools/analyze_s2.py")
BOUND_CODE = ("src/argus/replay/interventions.py", "src/argus/policies/repair.py")
PROBE_POSITIONS = (0, 1)
WORKERS = 5


def render_frozen(root: Path) -> str:
    prereg = json.loads((root / PREREG_PATH).read_text(encoding="utf-8"))
    manifest = prereg["basis"]["split_manifest"]
    if file_sha256(root / manifest["path"]) != manifest["sha256"]:
        raise RunSetError("the split manifest differs from the preregistered hash")
    frozen = {
        "schema": "argus-s2-frozen-v1",
        "status": "frozen before any s2_test episode; a run also needs the approval file",
        "preregistration_sha256": text_sha256(root / PREREG_PATH),
        "split_manifest_sha256": manifest["sha256"],
        "x1x2_frozen_sha256": prereg["basis"]["x1x2_frozen_sha256"],
        "bound_code": {path: text_sha256(root / path) for path in BOUND_CODE},
        "source_tree_sha256": source_sha256(root),
        "driver_sha256": text_sha256(root / "tools/s2_batch.py"),
        "analysis_code": {path: text_sha256(root / path) for path in S2_ANALYSIS_CODE},
        "design": prereg["design"],
    }
    return json.dumps(frozen, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def load_frozen(root: Path) -> tuple[dict[str, Any], str]:
    text = (root / S2_FROZEN_PATH).read_text(encoding="utf-8")
    if text != render_frozen(root):
        raise RunSetError(f"{S2_FROZEN_PATH} does not match the committed sources")
    return json.loads(text), hashlib.sha256(text.encode("utf-8")).hexdigest()


def conditions(design: Mapping[str, Any]) -> list[tuple[float, float]]:
    return [(s, c) for s in design["rate_scales"] for c in design["cost_ratios"]]


def s2_entries(root: Path) -> list[SplitEntry]:
    prereg = json.loads((root / PREREG_PATH).read_text(encoding="utf-8"))
    data = json.loads((root / prereg["basis"]["split_manifest"]["path"]).read_text("utf-8"))
    entries = [SplitEntry(*row) for row in data["entries"]]
    return sorted((e for e in entries if e.split == "s2_test"), key=lambda e: e.position)


def pick(
    entries: Sequence[SplitEntry], b: int, valid: Any | None
) -> tuple[list[SplitEntry], list[tuple[str, str]]]:
    """The first b entries; an invalid one is replaced from behind by the same family."""
    chosen, spare, replaced = list(entries[:b]), list(entries[b:]), []
    if valid is not None:
        for i, entry in enumerate(chosen):
            if valid(entry):
                continue
            while True:
                candidates = [e for e in spare if e.family == entry.family]
                if not candidates:
                    raise RunSetError(f"no replacement left for {entry.building_id}")
                spare.remove(candidates[0])
                if valid(candidates[0]):
                    break
            chosen[i] = candidates[0]
            replaced.append((entry.building_id, candidates[0].building_id))
    return chosen, replaced


def make_jobs(
    runset: ImmutableRunSet,
    frozen_sha256: str,
    design: Mapping[str, Any],
    entries: Sequence[SplitEntry],
    replicates: int,
    out: Path,
    commit: str,
    approval_sha256: str | None,
) -> list[BatchJob]:
    """Per building all conditions, replicates and arms; a group is one building."""
    params = manifest_params(runset.root, runset.prereg).to_json()
    src = source_sha256(runset.root)
    jobs = []
    for entry in entries:
        for scale, ratio in conditions(design):
            for replicate in range(replicates):
                for arm, chosen in sorted(design["arms"].items()):
                    spec = InterventionSpec(arm, float(scale), float(ratio))
                    key = (
                        f"{spec.condition_id}/{entry.split}/{entry.building_id}/{arm}/"
                        f"{chosen['config_id']}/{replicate}"
                    )
                    payload = {
                        "entry": list(entry.to_json()),
                        "allowed_splits": [entry.split],
                        "params": params,
                        "cell_id": spec.cell.cell_id,
                        "cell_params": {"passage_rates": list(spec.cell_params.passage_rates)},
                        "loss_false_open": spec.cost_ratio,
                        "method_id": arm,
                        "config_id": chosen["config_id"],
                        "method_params": dict(chosen["params"]),
                        "replicate": replicate,
                        "key": key,
                        "out": str(out),
                        "frozen_sha256": frozen_sha256,
                        "code_commit": commit,
                        "root": str(runset.root),
                        "src_sha256": src,
                        "approval_sha256": approval_sha256,
                        "approval_path": S2_APPROVAL_PATH,
                    }
                    jobs.append(BatchJob(key, f"{entry.split}/{entry.building_id}", payload))
    return jobs


def probe(out: Path, buildings: int = len(PROBE_POSITIONS)) -> None:
    """calib only, one replicate; more buildings give the spread pilot for the MDE."""
    runset = ImmutableRunSet.load(ROOT)
    frozen, sha = load_frozen(ROOT)
    calib, _ = runset.buildings("calib", buildings)
    jobs = make_jobs(runset, sha, frozen["design"], calib, 1, out, code_commit(), None)
    caps = dataclasses.replace(PROBE_CAPS, wall_hours=2.0, cpu_hours=6.0)
    summary = BatchRunner(out, caps, test_episode, killed_episode, strict=True).run(jobs)
    log_session(out, summary, code_commit())
    print(_summary_line(summary))


def approval(sha: str) -> dict[str, Any]:
    path = ROOT / S2_APPROVAL_PATH
    if not path.exists():
        raise RunSetError(f"no approval: {S2_APPROVAL_PATH} is missing; study 2 is not run")
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
    frozen, sha = load_frozen(ROOT)
    design = frozen["design"]
    commit = code_commit()
    granted = approval(sha)
    params = manifest_params(ROOT, runset.prereg)
    entries, replaced = pick(s2_entries(ROOT), design["b"], lambda e: scenario_valid(e, params))
    approval_sha = file_sha256(ROOT / S2_APPROVAL_PATH)
    plan = {
        "frozen_sha256": sha,
        "approval_sha256": approval_sha,
        "src_sha256": granted["src_sha256"],
        "b": design["b"],
        "replacements": json.loads(json.dumps(replaced)),
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
    jobs = make_jobs(runset, sha, design, entries, design["replicates"], out, commit, approval_sha)
    caps = BatchCaps(granted["wall_hours"], granted["cpu_hours"], WORKERS, HARD_TIMEOUT_S)
    summary = BatchRunner(out, caps, test_episode, killed_episode, strict=True).run(jobs)
    log_session(out, summary, commit)
    print(_summary_line(summary))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["freeze", "probe", "run", "status"])
    parser.add_argument("--out", type=Path)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--buildings", type=int, default=len(PROBE_POSITIONS))
    args = parser.parse_args()
    if args.command == "freeze":
        text = render_frozen(ROOT)
        target = ROOT / S2_FROZEN_PATH
        if args.check:
            if target.read_text(encoding="utf-8") != text:
                raise SystemExit(f"{S2_FROZEN_PATH} is stale")
            print(f"{S2_FROZEN_PATH} current")
        else:
            target.write_bytes(text.encode("utf-8"))
            print(f"wrote {S2_FROZEN_PATH}")
    elif args.command == "status":
        status(args.out)
    elif args.out is None:
        raise SystemExit(f"{args.command} needs --out")
    elif args.command == "probe":
        probe(args.out, args.buildings)
    else:
        run(args.out)


if __name__ == "__main__":
    main()
