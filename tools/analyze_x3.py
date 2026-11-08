"""Frozen descriptive analysis of the X3 batch (ARGUS-E08-T04, amendment 04).

    python tools/analyze_x3.py --out runs/x3_v1 --json <path> --report <path>
    python tools/analyze_x3.py --out runs/x3_probe_calib_vN --json <path> --report <path> --dry-run

Intervals only: no p-value, no H2 decision. A real run refuses a foreign or unfinished
batch and every run is logged in analysis_log.jsonl. A dry run reads calib data only.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from x3_batch import X3_ANALYSIS_CODE, load_frozen

from argus.evaluation.statistics import (
    ANALYSIS_SEED,
    DRAWS,
    AnalysisError,
    PairedBootstrap,
    _interval,
    building_values,
    complete_prefix,
    family_weighted_mean,
    single_family,
)
from argus.experiments.runset import ImmutableRunSet, source_sha256, text_sha256

ROOT = Path(__file__).resolve().parents[1]
COVERAGE = "coverage_closed_final"


def read_rows(out: Path) -> list[dict[str, Any]]:
    text = (out / "results.jsonl").read_text(encoding="utf-8")
    rows = [json.loads(line) for line in text.splitlines() if line]
    for r in rows:
        r["condition"] = r["key"].split("/")[0]
        r.setdefault(COVERAGE, 0.0)  # amendment 03, A2: a killed episode has no trace
    return rows


def guard(out: Path, frozen_sha256: str, splits: set[str], dry_run: bool) -> str:
    """Refuse interim looks and foreign batches; return the terminal state."""
    if dry_run:
        if splits - {"calib"}:
            raise AnalysisError("a dry run must not read test splits")
        return "dry run"
    if "calib" in splits:
        raise AnalysisError("the X3 analysis needs test lists")
    for marker in ("LOCK", "PAUSE"):
        if (out / marker).exists():
            raise AnalysisError(f"{marker} present: no interim analysis")
    plan = json.loads((out / "run_plan.json").read_text(encoding="utf-8"))
    if plan["frozen_sha256"] != frozen_sha256:
        raise AnalysisError("the batch ran under another frozen run set")
    if plan["src_sha256"] != source_sha256(ROOT):
        raise AnalysisError("the source tree differs from the one the batch ran with")
    text = (out / "sessions.jsonl").read_text(encoding="utf-8")
    sessions = [json.loads(line) for line in text.splitlines() if line]
    if not sessions:
        raise AnalysisError("no session record: the batch state is unknown")
    last = sessions[-1]["stop_reason"]
    if last not in (None, "budget"):
        raise AnalysisError(f"the batch stopped with '{last}': resume it first")
    return "complete" if last is None else "budget"


def complete_buildings(
    rows: list[dict[str, Any]],
    conditions: Mapping[str, Mapping[str, Any]],
    cells: list[str],
    replicates: int,
    methods: set[str],
) -> dict[str, list[str]]:
    """Per list, buildings in slot order that hold every episode of their conditions."""
    expected: dict[str, int] = defaultdict(int)
    for condition in conditions.values():
        expected[condition["list"]] += len(cells) * replicates * len(methods)
    counts: dict[tuple[str, str], int] = defaultdict(int)
    order: dict[str, dict[str, int]] = defaultdict(dict)
    for r in rows:
        counts[(r["split"], r["building_id"])] += 1
        order[r["split"]].setdefault(r["building_id"], r["position"])
    return {
        split: [
            b
            for b, _ in sorted(positions.items(), key=lambda kv: kv[1])
            if counts[(split, b)] == expected[split]
        ]
        for split, positions in order.items()
    }


class Condition:
    """Building means of one condition, on one bootstrap."""

    def __init__(
        self,
        rows: list[dict[str, Any]],
        cells: list[str],
        replicates: int,
        buildings: list[str],
        boot: PairedBootstrap,
    ) -> None:
        self.rows, self.cells, self.replicates = rows, cells, replicates
        self.buildings, self.boot = buildings, boot

    def values(self, method: str, field: str, cells: list[str] | None = None) -> dict[str, float]:
        wanted = cells or self.cells
        rows = [r for r in self.rows if r["cell_id"] in wanted]
        return building_values(rows, method, field, wanted, self.replicates, self.buildings)

    def report(self, methods: list[str], candidate: str, control: str) -> dict[str, Any]:
        boot = self.boot
        cand_loss = self.values(candidate, "mean_loss")
        return {
            "buildings": len(self.buildings),
            "status": sorted({r["status"] for r in self.rows}),
            "mean_loss": {m: boot.mean(self.values(m, "mean_loss")).to_json() for m in methods},
            "coverage": {m: boot.mean(self.values(m, COVERAGE)).to_json() for m in methods},
            "sent_bytes_total": {
                m: boot.mean(self.values(m, "sent_bytes_total")).to_json() for m in methods
            },
            "gain": {
                m: boot.ratio_gain(cand_loss, self.values(m, "mean_loss")).to_json()
                for m in methods
                if m != candidate
            },
            "coverage_loss": boot.ratio_gain(
                self.values(candidate, COVERAGE), self.values(control, COVERAGE)
            ).to_json(),
            "gain_per_cell": {
                c: boot.ratio_gain(
                    self.values(candidate, "mean_loss", [c]),
                    self.values(control, "mean_loss", [c]),
                ).to_json()
                for c in self.cells
            },
        }

    def gain_draws(self, candidate: str, control: str) -> Any:
        boot = self.boot
        return 1 - boot.means(self.values(candidate, "mean_loss")) / boot.means(
            self.values(control, "mean_loss")
        )

    def gain_point(self, candidate: str, control: str) -> float:
        family = self.boot.family
        cand = family_weighted_mean(self.values(candidate, "mean_loss"), family)
        return 1 - cand / family_weighted_mean(self.values(control, "mean_loss"), family)


def shift_effect(
    shifted: Condition, reference: Condition, candidate: str, control: str
) -> dict[str, Any]:
    """Paired change caused by a shift: both conditions share buildings and draws."""
    boot = shifted.boot
    gain_point = shifted.gain_point(candidate, control) - reference.gain_point(candidate, control)
    gain_draws = shifted.gain_draws(candidate, control) - reference.gain_draws(candidate, control)
    return {
        "gain_change": _interval(gain_point, gain_draws).to_json(),
        "candidate_loss_change": boot.difference(
            shifted.values(candidate, "mean_loss"), reference.values(candidate, "mean_loss")
        ).to_json(),
        "control_loss_change": boot.difference(
            shifted.values(control, "mean_loss"), reference.values(control, "mean_loss")
        ).to_json(),
    }


def analyze(out: Path, dry_run: bool = False, draws: int = DRAWS) -> dict[str, Any]:
    runset = ImmutableRunSet.load(ROOT)
    frozen, sha = load_frozen(ROOT, runset)
    x3 = frozen["x3"]
    code = {path: text_sha256(ROOT / path) for path in X3_ANALYSIS_CODE}
    if code != frozen["analysis_code"]:
        raise AnalysisError("the analysis code differs from the frozen hashes")
    rows = read_rows(out)
    terminal = guard(out, sha, {r["split"] for r in rows}, dry_run)
    conditions = x3["conditions"]
    if dry_run:
        conditions = {k: {**v, "list": "calib"} for k, v in conditions.items()}
    replicates = 1 if dry_run else x3["replicates"]
    cells, candidate, control = x3["cells"], x3["candidate"], x3["control"]
    methods = sorted({r["method_id"] for r in rows})
    complete = complete_buildings(rows, conditions, cells, replicates, set(methods))
    prefix = min(len(v) for v in complete.values())
    if not dry_run:
        prefix = complete_prefix([True] * prefix, x3["b_minimum"]) or 0
        if prefix == 0:
            raise AnalysisError("fewer complete positions than b_minimum: inconclusive by design")
    family = {r["building_id"]: r["family"] for r in rows}
    report: dict[str, Any] = {
        "dry_run": dry_run,
        "terminal": terminal,
        "frozen_sha256": sha,
        "analysis_code": code,
        "draws": draws,
        "seed": ANALYSIS_SEED,
        "prefix_b": prefix,
        "conditions": {},
        "shifts": {},
        "per_family": {},
    }
    built: dict[str, Condition] = {}
    boots: dict[str, PairedBootstrap] = {}
    for name, condition in conditions.items():
        split = condition["list"]
        buildings = complete[split][:prefix]
        if split not in boots:
            boots[split] = PairedBootstrap(buildings, family, draws)
        selected = [r for r in rows if r["condition"] == name and r["building_id"] in buildings]
        built[name] = Condition(selected, cells, replicates, buildings, boots[split])
        report["conditions"][name] = built[name].report(methods, candidate, control)
    for name, reference in x3["shifts"].items():
        if reference is not None:
            report["shifts"][name] = shift_effect(built[name], built[reference], candidate, control)
    if not dry_run:
        for name in ("ref_ood", "gauss_ood"):
            for fam in sorted({family[b] for b in built[name].buildings}):
                members = single_family(built[name].boot.family, fam)
                boot = PairedBootstrap(members, family, draws)
                sub = Condition(built[name].rows, cells, replicates, members, boot)
                sub.rows = [r for r in sub.rows if r["building_id"] in members]
                report["per_family"][f"{name}/{fam}"] = sub.report(methods, candidate, control)
    return report


def log(out: Path, report: dict[str, Any]) -> None:
    if report["dry_run"]:
        return
    entry = {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "frozen_sha256": report["frozen_sha256"],
        "analysis_code": report["analysis_code"],
        "prefix_b": report["prefix_b"],
    }
    with (out / "analysis_log.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")


def markdown(r: dict[str, Any]) -> str:
    def iv(x: Mapping[str, float]) -> str:
        return f"{x['estimate']:+.4f} [{x['low']:+.4f}, {x['high']:+.4f}]"

    lines = [
        "# X3 analysis" + (" (dry run, not a test result)" if r["dry_run"] else ""),
        "",
        f"Generated by `tools/analyze_x3.py`. Frozen run set `{r['frozen_sha256']}`; bootstrap "
        f"{r['draws']} draws, seed {r['seed']}; {r['prefix_b']} buildings per list. Descriptive "
        "only (amendment 04, B7).",
        "",
        "| Condition | Gain vs cross_rank | Gain vs dynamics_ids | Coverage loss |",
        "|---|---|---|---|",
    ]
    for name, c in r["conditions"].items():
        lines.append(
            f"| {name} | {iv(c['gain']['cross_rank'])} | {iv(c['gain']['dynamics_ids'])} "
            f"| {iv(c['coverage_loss'])} |"
        )
    lines += ["", "| Shift | Gain change | Candidate loss change | Control loss change |"]
    lines.append("|---|---|---|---|")
    for name, s in r["shifts"].items():
        lines.append(
            f"| {name} | {iv(s['gain_change'])} | {iv(s['candidate_loss_change'])} "
            f"| {iv(s['control_loss_change'])} |"
        )
    lines += ["", "Everything else is in the JSON next to this report.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    report = analyze(args.out, args.dry_run)
    args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.report.write_text(markdown(report), encoding="utf-8")
    log(args.out, report)
    print(f"wrote {args.json} and {args.report}")


if __name__ == "__main__":
    main()
