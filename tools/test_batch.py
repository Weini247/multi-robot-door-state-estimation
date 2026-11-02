"""Frozen X1/X2 test batch (ARGUS-E08-T02).

    python tools/test_batch.py freeze [--check]
    python tools/test_batch.py probe --out runs/test_probe_calib_vN   # --out required
    python tools/test_batch.py run --out runs/x1x2_v1          # needs the approval file
    python tools/test_batch.py status --out <dir>

`freeze` writes configs/experiments/X1_X2_frozen.json from committed evidence.
`probe` runs a small batch on calib buildings only (positions 0-1, 1 replicate,
all methods and cells) through the same worker as the test. `run` refuses
unless configs/experiments/X1_X2_approval.json names the frozen hash and a
budget; B is then capped by that budget. Stop with a PAUSE file in the output
directory; a rerun continues. Every run requires a clean tracked tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from argus.experiments.batch import BatchCaps, BatchRunner, BatchUsage
from argus.experiments.protocol import file_sha256, manifest_params
from argus.experiments.runner import TARGET_SETS, EpisodeConfig, EpisodeTrace
from argus.experiments.runset import (
    APPROVAL_PATH,
    FROZEN_PATH,
    HARD_TIMEOUT_S,
    ImmutableRunSet,
    RunSetError,
    b_for_budget,
    killed_episode,
    render_frozen,
    source_sha256,
    test_episode,
)
from argus.experiments.splits import regenerate

ROOT = Path(__file__).resolve().parents[1]
PROBE_SPLITS = ("calib",)
PROBE_POSITIONS = (0, 1)
PROBE_CAPS = BatchCaps(wall_hours=0.5, cpu_hours=1.5, workers=5, episode_timeout_s=HARD_TIMEOUT_S)


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def code_commit() -> str:
    """HEAD, refusing modified tracked files and untracked files where code or configs live."""
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise RunSetError("tracked files are modified; commit before running")
    if _git("status", "--porcelain", "--untracked-files=all", "--", "src", "tools", "configs"):
        raise RunSetError("untracked files in src/, tools/ or configs/; remove or commit them")
    return _git("rev-parse", "HEAD")


def probe(out: Path) -> None:
    runset = ImmutableRunSet.load(ROOT)
    jobs = runset.jobs(
        PROBE_SPLITS, max(PROBE_POSITIONS) + 1, 1, out, code_commit(), PROBE_POSITIONS
    )
    summary = BatchRunner(out, PROBE_CAPS, test_episode, killed_episode, strict=True).run(jobs)
    print(_summary_line(summary))


def run(out: Path) -> None:
    runset = ImmutableRunSet.load(ROOT)
    commit = code_commit()  # also proves the approval file is committed and unmodified
    approval = runset.approval()
    if _git("ls-files", APPROVAL_PATH) != APPROVAL_PATH:
        raise RunSetError("the approval file must be committed")
    if approval["src_sha256"] != source_sha256(ROOT):
        raise RunSetError("the approval names another source tree")
    b = b_for_budget(runset.frozen, approval["cpu_hours"], approval["wall_hours"])
    if b is None:
        raise RunSetError("the approved budget does not allow B = 60: inconclusive by design")
    params_valid = _validator(runset)
    caps = BatchCaps(
        approval["wall_hours"],
        approval["cpu_hours"],
        runset.frozen["cost_model"]["workers"],
        HARD_TIMEOUT_S,
    )
    lists = tuple(runset.frozen["test"]["lists"])
    approval_sha = file_sha256(ROOT / APPROVAL_PATH)
    plan = {
        "frozen_sha256": runset.sha256,
        "approval_sha256": approval_sha,
        "src_sha256": approval["src_sha256"],
        "b": b,
        "tool_sha256": tool_sha256(),
        # JSON round trip, so a resume compares like with like
        "replacements": json.loads(
            json.dumps({s: runset.buildings(s, b, params_valid)[1] for s in lists})
        ),
    }
    plan_path = out / "run_plan.json"
    if plan_path.exists():  # a resume must continue exactly the first session's plan
        first = json.loads(plan_path.read_text(encoding="utf-8"))
        changed = {k for k in plan if first.get(k) != plan[k]}
        if changed:
            raise RunSetError(f"resume differs from the first session in {sorted(changed)}")
    if not plan_path.exists():
        out.mkdir(parents=True, exist_ok=True)
        record = {**plan, "first_commit": commit, "approval": approval}
        plan_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    jobs = runset.jobs(
        lists,
        b,
        runset.frozen["test"]["replicates"],
        out,
        commit,
        None,
        params_valid,
        approval_sha,
    )
    summary = BatchRunner(out, caps, test_episode, killed_episode, strict=True).run(jobs)
    log_session(out, summary, commit)
    print(_summary_line(summary))


def log_session(out: Path, summary: Any, commit: str) -> None:
    """One line per session; the analysis accepts only a complete or budget-stopped batch."""
    entry = {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "commit": commit,
        "stop_reason": summary.stop_reason,
        "stopped_before": summary.stopped_before,
        "completed_groups": len(summary.completed_groups),
        "ran": summary.ran,
        "cpu_h": summary.usage.cpu_s / 3600,
        "wall_h": summary.usage.wall_s / 3600,
    }
    with (out / "sessions.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")


def tool_sha256() -> str:
    """Hash of this driver (line endings normalised); a resume must use the same one."""
    text = Path(__file__).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(text).hexdigest()


def scenario_valid(entry: Any, params: Any) -> bool:
    """Amendment 01 (O2): an entry is invalid if it cannot be regenerated to its manifest
    hashes, fails scenario validation, or has an empty coverage target set."""
    try:
        plan = regenerate(entry, params)
        resolution = EpisodeConfig.__dataclass_fields__["map_resolution_m"].default
        trace = EpisodeTrace(plan.spec.geometry, resolution)
    except Exception:  # noqa: BLE001 - any failure makes the scenario invalid
        return False
    return all(trace.targets[name] for name in TARGET_SETS)


def _validator(runset: ImmutableRunSet) -> Any:
    params = manifest_params(ROOT, runset.prereg)
    return lambda entry: scenario_valid(entry, params)


def _summary_line(summary: Any) -> str:
    return (
        f"groups {len(summary.completed_groups)}, ran {summary.ran}, skipped {summary.skipped}, "
        f"stopped before {summary.stopped_before}, reason {summary.stop_reason}, wall "
        f"{summary.usage.wall_s / 3600:.2f}h, cpu {summary.usage.cpu_s / 3600:.2f}h, "
        f"sessions {summary.usage.sessions}"
    )


def status(out: Path) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in (out / "results.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    usage = BatchUsage.load(out / "usage.json")
    per: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for field in ("mean_loss", "sent_bytes_total", "cpu_s", "coverage_closed_final"):
            if r.get(field) is not None:
                per[r["method_id"]][field].append(float(r[field]))
    report = {
        "episodes": len(rows),
        "splits": dict(Counter(r.get("split") for r in rows)),
        "status": dict(Counter(r["status"] for r in rows)),
        "missing_loss": sum(r.get("mean_loss") is None for r in rows),
        "usage": {"wall_h": usage.wall_s / 3600, "cpu_h": usage.cpu_s / 3600},
        "sessions": usage.sessions,
        "per_method": {
            m: {f: math.fsum(v) / len(v) for f, v in sorted(fields.items())}
            for m, fields in sorted(per.items())
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["freeze", "probe", "run", "status"])
    parser.add_argument("--out", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.command == "freeze":
        target = ROOT / FROZEN_PATH
        text = render_frozen(ROOT)
        if args.check:
            if target.read_text(encoding="utf-8") != text:
                raise SystemExit(f"{FROZEN_PATH} is stale")
            print(f"{FROZEN_PATH} current")
        else:
            target.write_bytes(text.encode("utf-8"))
            print(f"wrote {FROZEN_PATH}")
    elif args.command == "probe":
        if args.out is None:
            raise SystemExit("probe needs --out (a new directory per probe version)")
        probe(args.out)
    elif args.command == "run":
        run(args.out or ROOT / "runs/x1x2_v1")
    else:
        status(args.out)


if __name__ == "__main__":
    main()
