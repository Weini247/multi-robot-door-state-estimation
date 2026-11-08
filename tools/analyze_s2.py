"""Frozen analysis of study 2 (ARGUS-E09-T02, configs/experiments/preregistration_s2.json).

    python tools/analyze_s2.py --out runs/s2_v1 --json <path> --report <path>
    python tools/analyze_s2.py --out runs/s2_probe_calib_vN --json <path> --report <path> --dry-run

S2-H1 to S2-H3 decide by family-wise (Bonferroni) percentile intervals with three outcomes;
p-values are not used. A set of conditions is the unweighted mean of per-condition gains.
Everything else is descriptive and unadjusted. A real run refuses a foreign or unfinished
batch and every run is logged in analysis_log.jsonl. A dry run reads calib data only.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from s2_batch import S2_ANALYSIS_CODE, conditions, load_frozen, s2_entries

from argus.evaluation.statistics import (
    DRAWS,
    AnalysisError,
    PairedBootstrap,
    building_values,
    complete_prefix,
    family_weighted_mean,
)
from argus.experiments.runset import source_sha256, text_sha256
from argus.replay.interventions import InterventionSpec, open_window_s

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261107
HYPOTHESES = 3
ALPHA = 0.05
SESOI = 0.03  # smallest effect of interest (preregistration_s2.json, decision)
COVERAGE = "coverage_closed_final"
REFERENCE = "no_comm"
OBSERVE = "repair_observe"


def read_rows(out: Path) -> list[dict[str, Any]]:
    text = (out / "results.jsonl").read_text(encoding="utf-8")
    rows = [json.loads(line) for line in text.splitlines() if line]
    for r in rows:
        r["factor_cell"] = r["cell_id"]
        r["cell_id"] = r["key"].split("/")[0]  # the condition is the unit of averaging
        r.setdefault(COVERAGE, 0.0)  # amendment 03, A2: a killed episode has no trace
    return rows


def guard(out: Path, frozen_sha256: str, splits: set[str], dry_run: bool) -> dict[str, Any]:
    """Refuse interim looks and foreign batches; return the run plan (empty for dry runs)."""
    if dry_run:
        if splits - {"calib"}:
            raise AnalysisError("a dry run must not read test splits")
        return {"terminal": "dry run", "replacements": []}
    if splits != {"s2_test"}:
        raise AnalysisError(f"the study 2 analysis needs s2_test only, got {sorted(splits)}")
    for marker in ("LOCK", "PAUSE"):
        if (out / marker).exists():
            raise AnalysisError(f"{marker} present: no interim analysis")
    plan: dict[str, Any] = json.loads((out / "run_plan.json").read_text(encoding="utf-8"))
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
    plan["terminal"] = "complete" if last is None else "budget"
    return plan


def slot_order(b: int, replacements: Sequence[Sequence[str]], entries: Sequence[Any]) -> list[str]:
    """Buildings in run order: the first b manifest entries, a replacement in its slot."""
    swap = {old: new for old, new in replacements}
    return [swap.get(e.building_id, e.building_id) for e in entries[:b]]


def interval(point: float, draws: Any, level: float) -> dict[str, Any]:
    tail = (1 - level) / 2 * 100
    low, high = np.percentile(draws, [tail, 100 - tail])
    return {"estimate": float(point), "low": float(low), "high": float(high), "level": level}


def undefined(level: float) -> dict[str, Any]:
    return {"estimate": None, "low": None, "high": None, "level": level, "undefined": True}


class Gains:
    """Per-condition ratio gains of one arm against another, on shared draws."""

    def __init__(
        self, rows: list[dict[str, Any]], boot: PairedBootstrap, buildings: list[str], reps: int
    ) -> None:
        self.rows, self.boot, self.buildings, self.replicates = rows, boot, buildings, reps

    def values(self, arm: str, field: str, cell: str) -> dict[str, float]:
        rows = [r for r in self.rows if r["cell_id"] == cell]
        return building_values(rows, arm, field, [cell], self.replicates, self.buildings)

    def condition(self, arm: str, ref: str, cell: str) -> tuple[float, Any] | None:
        """Point and draws of 1 - M(arm)/M(ref) in one condition; None if undefined."""
        cand, base = self.values(arm, "mean_loss", cell), self.values(ref, "mean_loss", cell)
        family = self.boot.family
        base_mean = family_weighted_mean(base, family)
        base_draws = self.boot.means(base)
        if base_mean <= 0 or np.any(base_draws <= 0):
            return None  # preregistration: zero reference -> gain undefined
        return 1 - family_weighted_mean(cand, family) / base_mean, 1 - self.boot.means(
            cand
        ) / base_draws

    def mean_gain(self, arm: str, ref: str, cells: Sequence[str]) -> tuple[float, Any] | None:
        """Unweighted mean of per-condition gains (each condition counts equally)."""
        parts = [self.condition(arm, ref, c) for c in cells]
        if any(p is None for p in parts):
            return None
        points = [p[0] for p in parts if p is not None]
        draws = [p[1] for p in parts if p is not None]
        return float(np.mean(points)), np.mean(np.vstack(draws), axis=0)


def condition_ids(
    design: Mapping[str, Any],
    scales: Sequence[float] | None = None,
    ratios: Sequence[float] | None = None,
) -> list[str]:
    return [
        InterventionSpec("repair", float(s), float(c)).condition_id
        for s, c in conditions(design)
        if (scales is None or s in scales) and (ratios is None or c in ratios)
    ]


def outcome(name: str, result: dict[str, Any]) -> str:
    if result.get("undefined"):
        return "inconclusive"
    if name == "S2-H3":  # window-matched: equality predicted
        if -SESOI <= result["low"] and result["high"] <= SESOI:
            return "consistent"
        if result["low"] > 0 or result["high"] < 0:
            return "against"
        return "inconclusive"
    if result["low"] > 0:
        return "supported"
    if result["high"] < SESOI:
        return "against"
    return "inconclusive"


def difference(
    first: tuple[float, Any] | None, second: tuple[float, Any] | None, level: float
) -> dict[str, Any]:
    if first is None or second is None:
        return undefined(level)
    return interval(first[0] - second[0], first[1] - second[1], level)


# window-matched pairs (nearly equal W, cost ratio 2 vs 4), preregistration_s2.json S2-H3
MATCHED = ((0.5, 0.25), (1.0, 0.5), (2.0, 1.0))


def matched(g: Gains, level: float) -> dict[str, Any]:
    """Mean over the matched pairs of gain(s_a, c=2) - gain(s_b, c=4)."""
    points, draws = [], []
    for fast, slow in MATCHED:
        first = g.condition(OBSERVE, REFERENCE, InterventionSpec("repair", fast, 2.0).condition_id)
        second = g.condition(OBSERVE, REFERENCE, InterventionSpec("repair", slow, 4.0).condition_id)
        if first is None or second is None:
            return undefined(level)
        points.append(first[0] - second[0])
        draws.append(first[1] - second[1])
    return interval(float(np.mean(points)), np.mean(np.vstack(draws), axis=0), level)


def hypotheses(g: Gains, design: Mapping[str, Any]) -> dict[str, Any]:
    level = 1 - ALPHA / HYPOTHESES
    dynamic = [s for s in design["rate_scales"] if s > 0]
    h1 = g.mean_gain(OBSERVE, REFERENCE, condition_ids(design, scales=dynamic))
    out = {
        "S2-H1": interval(h1[0], h1[1], level) if h1 is not None else undefined(level),
        "S2-H2": difference(
            g.mean_gain(OBSERVE, REFERENCE, condition_ids(design, scales=[0.25])),
            g.mean_gain(OBSERVE, REFERENCE, condition_ids(design, scales=[2])),
            level,
        ),
        "S2-H3": matched(g, level),
    }
    for name, result in out.items():
        result["outcome"] = outcome(name, result)
    return out


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def describe(g: Gains, design: Mapping[str, Any], arms: list[str]) -> dict[str, Any]:
    boot = g.boot
    out: dict[str, Any] = {}
    for scale, ratio in conditions(design):
        cid = InterventionSpec("repair", float(scale), float(ratio)).condition_id
        entry: dict[str, Any] = {
            "window_s": open_window_s(float(scale), float(ratio)),
            "loss": {},
            "coverage": {},
            "bytes": {},
            "gain": {},
            "actions": {},
        }
        for arm in arms:
            entry["loss"][arm] = boot.mean(g.values(arm, "mean_loss", cid)).to_json()
            entry["coverage"][arm] = boot.mean(g.values(arm, COVERAGE, cid)).to_json()
            entry["bytes"][arm] = boot.mean(g.values(arm, "sent_bytes_total", cid)).to_json()
            counts: dict[str, list[int]] = defaultdict(list)
            for r in g.rows:
                if r["cell_id"] == cid and r["method_id"] == arm and "policy_actions" in r:
                    for name, n in r["policy_actions"].items():
                        counts[name].append(n)
            entry["actions"][arm] = {k: float(np.mean(v)) for k, v in sorted(counts.items())}
        pairs = [(a, "no_comm") for a in ("repair", "repair_observe", "repair_send", "age_revisit")]
        pairs += [(a, "dynamics_ids") for a in ("repair", "repair_observe", "repair_send")]
        for arm, ref in pairs:
            result = g.condition(arm, ref, cid)
            entry["gain"][f"{arm}/{ref}"] = (
                interval(result[0], result[1], 0.95) if result is not None else undefined(0.95)
            )
        out[cid] = entry
    dynamic = [(c, e) for c, e in out.items() if np.isfinite(e["window_s"])]
    gains = [e["gain"]["repair_observe/no_comm"]["estimate"] for _, e in dynamic]
    trend = (
        spearman([e["window_s"] for _, e in dynamic], gains)
        if None not in gains and len(dynamic) > 2
        else None
    )
    return {"per_condition": out, "window_rank_correlation": trend}


def analyze(out: Path, dry_run: bool = False, draws: int = DRAWS) -> dict[str, Any]:
    frozen, sha = load_frozen(ROOT)
    design = frozen["design"]
    code = {path: text_sha256(ROOT / path) for path in S2_ANALYSIS_CODE}
    if code != frozen["analysis_code"]:
        raise AnalysisError("the analysis code differs from the frozen hashes")
    rows = read_rows(out)
    plan = guard(out, sha, {r["split"] for r in rows}, dry_run)
    replicates = 1 if dry_run else design["replicates"]
    arms = sorted(design["arms"])
    expected = len(conditions(design)) * replicates * len(arms)
    counts: dict[str, int] = defaultdict(int)
    for r in rows:
        counts[r["building_id"]] += 1
    if dry_run:
        ordered = sorted(
            counts, key=lambda b: next(r["position"] for r in rows if r["building_id"] == b)
        )
        prefix = len(ordered)
    else:
        ordered = slot_order(design["b"], plan["replacements"], s2_entries(ROOT))
        prefix = complete_prefix([counts[b] == expected for b in ordered], design["b_minimum"])
        if not prefix:
            raise AnalysisError("fewer complete buildings than b_minimum: inconclusive by design")
    buildings = ordered[:prefix]
    family = {r["building_id"]: r["family"] for r in rows}
    boot = PairedBootstrap(buildings, family, draws, SEED)
    selected = [r for r in rows if r["building_id"] in buildings]
    gains = Gains(selected, boot, buildings, replicates)
    return {
        "dry_run": dry_run,
        "terminal": plan["terminal"],
        "frozen_sha256": sha,
        "analysis_code": code,
        "draws": draws,
        "seed": SEED,
        "prefix_b": prefix,
        "families": {
            f: sum(family[b] == f for b in buildings) for f in sorted(set(family.values()))
        },
        "status": sorted({r["status"] for r in selected}),
        "hypotheses": hypotheses(gains, design),
        "descriptive": describe(gains, design, arms),
    }


def log(out: Path, report: dict[str, Any]) -> None:
    if report["dry_run"]:
        return
    entry = {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "frozen_sha256": report["frozen_sha256"],
        "analysis_code": report["analysis_code"],
        "prefix_b": report["prefix_b"],
        "outcomes": {k: v["outcome"] for k, v in report["hypotheses"].items()},
    }
    with (out / "analysis_log.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")


def markdown(r: dict[str, Any]) -> str:
    def iv(x: Mapping[str, Any]) -> str:
        if x.get("undefined"):
            return "undefined"
        return f"{x['estimate']:+.4f} [{x['low']:+.4f}, {x['high']:+.4f}]"

    lines = [
        "# Study 2 analysis" + (" (dry run, not a test result)" if r["dry_run"] else ""),
        "",
        f"Generated by `tools/analyze_s2.py`. Frozen run set `{r['frozen_sha256']}`; bootstrap "
        f"{r['draws']} draws, seed {r['seed']}; {r['prefix_b']} buildings {r['families']}.",
        "",
        "| Hypothesis | Estimate [family-wise interval] | Outcome |",
        "|---|---|---|",
    ]
    for name, h in r["hypotheses"].items():
        lines.append(f"| {name} | {iv(h)} | {h['outcome']} |")
    lines += [
        "",
        "Per condition (95 % intervals, unadjusted, not decisive):",
        "",
        "| Condition | window (s) | repair_observe vs no_comm | repair_send vs no_comm "
        "| age_revisit vs no_comm |",
        "|---|---|---|---|---|",
    ]
    for cid, c in r["descriptive"]["per_condition"].items():
        g = c["gain"]
        lines.append(
            f"| {cid} | {c['window_s']:.1f} | {iv(g['repair_observe/no_comm'])} "
            f"| {iv(g['repair_send/no_comm'])} | {iv(g['age_revisit/no_comm'])} |"
        )
    trend = r["descriptive"]["window_rank_correlation"]
    lines += [
        "",
        f"Rank correlation of the dynamic per-condition gains with the window: {trend}.",
        "Losses, coverage, bytes and counted decisions are in the JSON.",
        "",
    ]
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
