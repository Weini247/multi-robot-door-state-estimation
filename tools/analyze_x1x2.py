"""Preregistered analysis of the X1/X2 test batch (ARGUS-E08-T03).

    python tools/analyze_x1x2.py --out runs/x1x2_v1 --json R.json --report R.md
    python tools/analyze_x1x2.py --dry-run --out docs/results/test_probe_calib_v3 ...

The code is frozen before the test batch runs: the run set records its hash
(X1_X2_frozen.json, analysis.code), and this script refuses to run if it differs.
It runs on the finished batch only: the batch must be complete or stopped by its
budget (sessions.jsonl), no LOCK or PAUSE, and the source tree must be the one the
batch ran with. Every analysis run is appended to analysis_log.jsonl. `--dry-run`
accepts only calibration/validation data and exists to test the pipeline.

The estimands follow the preregistration and amendments 01-03. id_test is the
primary list (H2 decision, Holm family A4); ood_test is reported per family with
the same estimands, descriptively. A dry run uses its single list as primary.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

from argus.evaluation.metrics import (
    EPS,
    js_divergence,
    kaplan_meier,
    restricted_mean_lag,
    runtime_quantiles,
)
from argus.evaluation.statistics import (
    ANALYSIS_SEED,
    DRAWS,
    AnalysisError,
    Interval,
    PairedBootstrap,
    building_cell_values,
    building_values,
    complete_prefix,
    factorial_contrasts,
    h2_verdict,
    holm,
    repair_lags_d2,
    rescored_mean_loss,
)
from argus.experiments.protocol import manifest_params
from argus.experiments.runset import (
    ANALYSIS_CODE,
    CANDIDATE,
    ImmutableRunSet,
    source_sha256,
    text_sha256,
)
from argus.experiments.splits import regenerate

ROOT = Path(__file__).resolve().parents[1]
TEST_PRIMARY, TEST_OOD = "id_test", "ood_test"
LAG_TAU_S = 30.0  # amendment 02, D2
FAILED_PROBE_LOSS = 4.0  # spec M6 sensitivity (amendment 01, O2)
LOSS_RATIOS = {"3:1": 3.0, "2:1": 2.0}  # O5: decision-level re-scoring, policies keep 4:1
COVERAGE_FIELDS = (
    "coverage_closed_final",
    "coverage_closed_auc",
    "coverage_open_final",
    "coverage_open_auc",
)


def rows_of(out: Path) -> list[dict[str, Any]]:
    text = (out / "results.jsonl").read_text(encoding="utf-8")
    rows = [json.loads(line) for line in text.splitlines() if line]
    keys = Counter(r["key"] for r in rows)
    dup = [k for k, n in keys.items() if n > 1]
    if dup:
        raise AnalysisError(f"duplicate keys, e.g. {dup[0]}")
    return rows


def guard(
    out: Path, runset: ImmutableRunSet, dry_run: bool, splits: set[str]
) -> dict[str, Any] | None:
    """Refuse interim looks and foreign batches; return the run plan of a test batch."""
    if dry_run:
        if splits & {TEST_PRIMARY, TEST_OOD}:
            raise AnalysisError("a dry run must not read test splits")
        return None
    if splits != {TEST_PRIMARY, TEST_OOD}:
        raise AnalysisError(f"the test analysis needs id_test and ood_test, got {sorted(splits)}")
    for marker in ("LOCK", "PAUSE"):
        if (out / marker).exists():
            raise AnalysisError(f"{marker} present: no interim analysis")
    plan: dict[str, Any] = json.loads((out / "run_plan.json").read_text(encoding="utf-8"))
    if plan["frozen_sha256"] != runset.sha256:
        raise AnalysisError("the batch ran under another frozen run set")
    if plan["src_sha256"] != source_sha256(ROOT):
        raise AnalysisError("the source tree differs from the one the batch ran with")
    log_path = out / "sessions.jsonl"
    text = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
    sessions = [json.loads(line) for line in text.splitlines() if line]
    if not sessions:
        raise AnalysisError("no session record (sessions.jsonl): the batch state is unknown")
    last = sessions[-1]
    if last["stop_reason"] not in (None, "budget"):
        raise AnalysisError(f"the batch stopped with '{last['stop_reason']}': resume it first")
    plan["terminal"] = "complete" if last["stop_reason"] is None else "budget"
    return plan


def slots(
    rows: list[dict[str, Any]], runset: ImmutableRunSet, plan: dict[str, Any] | None
) -> dict[tuple[str, str], int]:
    """Slot of every building in its list: its manifest position, or for a replacement the
    position of the building it replaced (amendment 03, A1)."""
    position = {(e.split, e.building_id): e.position for e in runset.entries}
    out = {(r["split"], r["building_id"]): position[(r["split"], r["building_id"])] for r in rows}
    for split, pairs in (plan or {}).get("replacements", {}).items():
        for replaced, replacement in pairs:
            if (split, replacement) in out:
                out[(split, replacement)] = position[(split, replaced)]
    shared = Counter((s, slot) for (s, _), slot in out.items())
    if shared and max(shared.values()) > 1:
        raise AnalysisError("two buildings share a slot")
    return out


def complete_slots(
    rows: list[dict[str, Any]], lists: list[str], methods: list[str], cells: list[str], reps: int
) -> list[bool]:
    need = len(methods) * len(cells) * reps
    count = Counter((r["split"], r["slot"]) for r in rows)
    top = max((r["slot"] for r in rows), default=-1)
    return [all(count.get((s, p), 0) == need for s in lists) for p in range(top + 1)]


def normalize(rows: list[dict[str, Any]]) -> None:
    """Amendment 03, A2: an episode killed by the hard timeout has no trace; its coverage
    is 0. Other failed episodes keep the coverage reached before the failure tick."""
    for r in rows:
        r["killed"] = "coverage_closed_final" not in r
        for field in COVERAGE_FIELDS:
            r.setdefault(field, 0.0)


class Episodes:
    """Raw-derived quantities per episode, read once."""

    def __init__(self, out: Path, pair: tuple[str, str]) -> None:
        self.out, self.pair = out, pair
        self.derived: dict[str, dict[str, Any]] = {}
        self.answered: dict[str, dict[tuple[int, str, str], float]] = {}
        self.lags: dict[str, list[Any]] = {}
        self.pooled_lags: dict[str, list[Any]] = {}

    def load(self, rows: list[dict[str, Any]], horizon_of: Callable[[str], int]) -> None:
        for r in rows:
            raw = json.loads(gzip.decompress((self.out / r["raw"]).read_bytes()))
            fo, fc = raw["config"]["losses"]
            primary = rescored_mean_loss(raw, fo, fc) * fc
            if not math.isclose(primary, r["mean_loss"], rel_tol=0, abs_tol=1e-12):
                raise AnalysisError(f"{r['key']}: raw re-scoring differs from the row")
            d: dict[str, Any] = {
                f"loss_{name}": rescored_mean_loss(raw, k, 1.0) for name, k in LOSS_RATIOS.items()
            }
            d["loss_m6"] = rescored_mean_loss(raw, fo, fc, FAILED_PROBE_LOSS * fc) * fc
            answered = raw["answered"]
            clipped = [min(1 - EPS, max(EPS, a[3])) for a in answered]
            d["n_answered"] = len(answered)
            d["brier_sum"] = math.fsum((a[3] - a[4]) ** 2 for a in answered)
            d["nll_sum"] = math.fsum(
                -(a[4] * math.log(q) + (1 - a[4]) * math.log(1 - q))
                for a, q in zip(answered, clipped, strict=True)
            )
            d["nll_clipped"] = sum(q != a[3] for a, q in zip(answered, clipped, strict=True))
            self.derived[r["key"]] = d
            if r["method_id"] in self.pair:
                self.answered[r["key"]] = {(a[0], a[1], a[2]): a[3] for a in answered}
                if r["cell_id"].startswith("dyn") and raw["trace"] is not None:
                    ticks = horizon_of(r["building_id"])
                    self.lags[r["key"]] = repair_lags_d2(raw, ticks)
                    self.pooled_lags[r["key"]] = repair_lags_d2(raw, ticks, pooled=True)


class Context:
    def __init__(
        self,
        eps: Episodes,
        family: dict[str, str],
        methods: list[str],
        control: str,
        cells: list[str],
        reps: int,
        draws: int,
        tau: int,
    ) -> None:
        self.eps, self.family, self.methods, self.control = eps, family, methods, control
        self.cells, self.reps, self.draws, self.tau = cells, reps, draws, tau

    def boot(self, buildings: set[str]) -> PairedBootstrap:
        return PairedBootstrap(sorted(buildings), self.family, self.draws)

    def values(self, rows: list[dict[str, Any]], method: str, field: str) -> dict[str, float]:
        return building_values(rows, method, field, self.cells, self.reps)


def estimands(rows: list[dict[str, Any]], ctx: Context, tests: bool) -> dict[str, Any]:
    """The preregistered estimands on one list (or one OOD family)."""
    methods, control = ctx.methods, ctx.control
    buildings = {r["building_id"] for r in rows}
    boot = ctx.boot(buildings)
    loss = {m: ctx.values(rows, m, "mean_loss") for m in methods}
    cov = {m: ctx.values(rows, m, "coverage_closed_final") for m in methods}
    out: dict[str, Any] = {
        "buildings": len(buildings),
        "families": dict(Counter(ctx.family[b] for b in sorted(buildings))),
        "status": {
            m: dict(Counter(r["status"] for r in rows if r["method_id"] == m)) for m in methods
        },
        "killed": {m: sum(r["killed"] for r in rows if r["method_id"] == m) for m in methods},
        "mean_loss": {m: boot.mean(loss[m]).to_json() for m in methods},
        "coverage_closed_final": {m: boot.mean(cov[m]).to_json() for m in methods},
        "h2_loss": boot.gain_report(loss[CANDIDATE], loss[control]),
        "h2_coverage": boot.gain_report(cov[CANDIDATE], cov[control]),
        "gain_vs_each_control": {
            m: boot.gain_report(loss[CANDIDATE], loss[m]) for m in methods if m != CANDIDATE
        },
    }
    p_values: dict[str, float] = {}
    per_cell = {}
    for c in ctx.cells:
        part = [r for r in rows if r["cell_id"] == c]
        cand = building_values(part, CANDIDATE, "mean_loss", [c], ctx.reps)
        ctrl = building_values(part, control, "mean_loss", [c], ctx.reps)
        per_cell[c] = boot.gain_report(cand, ctrl)
        try:
            p_values[f"cell {c}"] = _p(boot.ratio_gain(cand, ctrl).p_value)
        except AnalysisError:  # zero control mean: test the absolute difference instead
            p_values[f"cell {c}"] = _p(boot.difference(cand, ctrl).p_value)
    out["gain_per_cell"] = per_cell
    out["stress_cell"] = "dyn-cor-relay"
    h1: dict[str, dict[str, Any]] = {}
    for m in (CANDIDATE, control):
        by_cell = building_cell_values(rows, m, "mean_loss", ctx.cells, ctx.reps)
        contrasts = {b: factorial_contrasts(v) for b, v in by_cell.items()}
        h1[m] = {}
        for n in sorted(next(iter(contrasts.values()))):
            iv = boot.mean({b: c[n] for b, c in contrasts.items()})
            h1[m][n] = iv.to_json()
            if m == CANDIDATE:
                p_values[f"h1 {n}"] = _p(iv.p_value)
    out["h1_interaction_contrasts"] = h1
    if tests:
        out["holm_family"] = {
            "members": "8 per-cell gains and 3 candidate H1 contrasts (amendment 03, A4)",
            "p_raw": p_values,
            "p_holm": holm(p_values),
        }
    out["coverage_secondary"] = {
        f: boot.gain_report(ctx.values(rows, CANDIDATE, f), ctx.values(rows, control, f))
        for f in COVERAGE_FIELDS[1:]
    }
    out["sent_bytes_total"] = {
        m: boot.mean(ctx.values(rows, m, "sent_bytes_total")).to_json() for m in methods
    }
    out["calibration_answered"] = calibration(rows, ctx)
    out["js_candidate_vs_control"] = js(rows, ctx)
    out["repair_lag"] = lags(rows, ctx)
    out["sensitivity"] = sensitivity(rows, ctx, boot)
    out["runtime"] = {m: runtime(rows, m) for m in methods}
    return out


def runtime(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    measured = [r for r in rows if r["method_id"] == method and not r["killed"]]
    if not measured:
        return {"episodes": 0}
    return {f: runtime_quantiles([r[f] for r in measured]) for f in ("wall_s", "cpu_s")}


def described(ctx: Context, values: dict[str, float], key: str = "estimate") -> dict[str, Any]:
    """Family-weighted mean with interval; empty sets and single families are flagged."""
    families = dict(Counter(ctx.family[b] for b in sorted(values)))
    if not values:
        return {key: None, "families": families, "warning": "no buildings"}
    out: dict[str, Any] = {key: ctx.boot(set(values)).mean(values).to_json(), "families": families}
    if len(families) < 2:
        out["warning"] = "a single family: not family-weighted"
    return out


def _p(value: float | None) -> float:
    if value is None:
        raise AnalysisError("a Holm family member has no p-value")
    return value


def calibration(rows: list[dict[str, Any]], ctx: Context) -> dict[str, Any]:
    """Amendment 03, A8: Brier and NLL per building pooled over all answered probes."""
    out: dict[str, Any] = {}
    for m in ctx.methods:
        sums: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
        empty = 0
        for r in rows:
            if r["method_id"] != m:
                continue
            d = ctx.eps.derived[r["key"]]
            empty += d["n_answered"] == 0
            s = sums[r["building_id"]]
            s[0] += d["n_answered"]
            s[1] += d["brier_sum"]
            s[2] += d["nll_sum"]
            s[3] += d["nll_clipped"]
        kept = {b: s for b, s in sums.items() if s[0] > 0}
        probes = math.fsum(s[0] for s in kept.values())
        out[m] = {
            "buildings": len(kept),
            "episodes_without_answers": empty,
            "brier": described(ctx, {b: s[1] / s[0] for b, s in kept.items()}),
            "nll": described(ctx, {b: s[2] / s[0] for b, s in kept.items()}),
            "nll_clipping_rate": math.fsum(s[3] for s in kept.values()) / probes
            if probes
            else None,
        }
    return out


def js(rows: list[dict[str, Any]], ctx: Context) -> dict[str, Any]:
    """Amendment 03, A10: JS on identical probes; per pair the mean over common probes,
    then replicates, then cells (equal weights) per building."""
    keys = {
        (r["building_id"], r["cell_id"], r["replicate"], r["method_id"]): r["key"] for r in rows
    }
    per: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for (b, c, rep, m), key in keys.items():
        if m != CANDIDATE:
            continue
        a = ctx.eps.answered[key]
        o = ctx.eps.answered[keys[(b, c, rep, ctx.control)]]
        common = sorted(set(a) & set(o))
        if common:
            per[b][c].append(math.fsum(js_divergence(a[p], o[p]) for p in common) / len(common))
    values = {
        b: math.fsum(math.fsum(v) / len(v) for v in by_cell.values()) / len(by_cell)
        for b, by_cell in per.items()
    }
    return {
        "buildings": len(values),
        "excluded_buildings": len({r["building_id"] for r in rows} - set(values)),
        **described(ctx, values, "js"),
    }


def lags(rows: list[dict[str, Any]], ctx: Context) -> dict[str, Any]:
    out: dict[str, Any] = {"tau_ticks": ctx.tau, "status": "secondary, descriptive (D2)"}
    for name, source in (("per_robot", ctx.eps.lags), ("pooled", ctx.eps.pooled_lags)):
        out[name] = {}
        for m in (CANDIDATE, ctx.control):
            events = [e for r in rows if r["method_id"] == m for e in source.get(r["key"], [])]
            if not events:
                out[name][m] = {"events": 0}
                continue
            out[name][m] = {
                "events": len(events),
                "censoring_rate": sum(e.censored for e in events) / len(events),
                "at_risk_at_tau": sum(1 for e in events if e.lag_ticks >= ctx.tau),
                "restricted_mean_ticks": restricted_mean_lag(events, ctx.tau),
                "kaplan_meier_to_tau": [(t, s) for t, s in kaplan_meier(events) if t <= ctx.tau],
            }
    return out


def sensitivity(rows: list[dict[str, Any]], ctx: Context, boot: PairedBootstrap) -> dict[str, Any]:
    out: dict[str, Any] = {
        "note": "loss ratios re-scored at decision level from recorded probabilities; "
        "policies kept 4:1 (amendment 01, O5)"
    }
    for field in ("loss_m6", *(f"loss_{n}" for n in LOSS_RATIOS)):
        rs = [{**r, field: ctx.eps.derived[r["key"]][field]} for r in rows]
        out[field] = boot.gain_report(
            ctx.values(rs, CANDIDATE, field), ctx.values(rs, ctx.control, field)
        )
    # Amendment 03, A7: pairs with a failure of either method are left out; cells average
    # the remaining pairs; buildings without any pair drop out of this bootstrap.
    index = {(r["building_id"], r["cell_id"], r["replicate"], r["method_id"]): r for r in rows}
    kept: dict[str, dict[str, list[tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
    excluded = 0
    for (b, c, rep, m), r in index.items():
        if m != CANDIDATE:
            continue
        o = index[(b, c, rep, ctx.control)]
        if r["status"] != "ok" or o["status"] != "ok":
            excluded += 1
            continue
        kept[b][c].append((r["mean_loss"], o["mean_loss"]))
    cand, ctrl = {}, {}
    for b, by_cell in kept.items():
        means = [
            (math.fsum(x for x, _ in v) / len(v), math.fsum(y for _, y in v) / len(v))
            for v in by_cell.values()
        ]
        cand[b] = math.fsum(x for x, _ in means) / len(means)
        ctrl[b] = math.fsum(y for _, y in means) / len(means)
    families = dict(Counter(ctx.family[b] for b in sorted(cand)))
    pairs: dict[str, Any] = {
        "excluded_pairs": excluded,
        "buildings": len(cand),
        "families": families,
    }
    if cand:
        pairs.update(ctx.boot(set(cand)).gain_report(cand, ctrl))
    else:
        pairs.update(gain=None, warning="no building kept a pair")
    if len(families) == 1:
        pairs["warning"] = "a single family: not family-weighted"
    out["pairs_without_failure"] = pairs
    return out


def analyze(out: Path, dry_run: bool = False, draws: int = DRAWS) -> dict[str, Any]:
    runset = ImmutableRunSet.load(ROOT)
    frozen = runset.frozen
    code = {path: text_sha256(ROOT / path) for path in ANALYSIS_CODE}
    if code != frozen["analysis"]["code"]:
        raise AnalysisError("the analysis code differs from the code frozen with the run set")
    if runset.prereg["inference"]["analysis_seed"] != ANALYSIS_SEED:
        raise AnalysisError("analysis seed differs from the preregistration")
    rows = rows_of(out)
    splits = {r["split"] for r in rows}
    plan = guard(out, runset, dry_run, splits)
    control = frozen["methods"]["h2_control"]
    methods = sorted(frozen["methods"]["selected"])
    cells = list(frozen["cells"])
    reps = max(r["replicate"] for r in rows) + 1 if dry_run else frozen["test"]["replicates"]
    lists = sorted(splits) if dry_run else [TEST_PRIMARY, TEST_OOD]
    slot = slots(rows, runset, plan)
    for r in rows:
        r["slot"] = slot[(r["split"], r["building_id"])]
    k = complete_prefix(
        complete_slots(rows, lists, methods, cells, reps), minimum=2 if dry_run else 60
    )
    report: dict[str, Any] = {
        "frozen_sha256": runset.sha256,
        "analysis_code": code,
        "dry_run": dry_run,
        "lists": lists,
        "candidate": CANDIDATE,
        "h2_control": control,
        "draws": draws,
        "seed": ANALYSIS_SEED,
        "planned_b": None if plan is None else plan["b"],
        "terminal": None if plan is None else plan["terminal"],
    }
    if plan is not None and plan["terminal"] == "complete" and k != plan["b"]:
        raise AnalysisError("the batch reports completion but the planned prefix is incomplete")
    if k is None:
        report["verdict"] = "inconclusive by design (complete prefix below the minimum)"
        log(out, report, dry_run)
        return report
    rows = [r for r in rows if r["slot"] < k]
    normalize(rows)
    report["prefix_b"] = k
    params = manifest_params(ROOT, runset.prereg)
    entries = {e.building_id: e for e in runset.entries}
    specs: dict[str, Any] = {}

    def spec_of(building: str) -> Any:
        if building not in specs:
            specs[building] = regenerate(entries[building], params).spec
        return specs[building]

    eps = Episodes(out, (CANDIDATE, control))
    eps.load(rows, lambda b: spec_of(b).horizon_ticks)
    tau = round(LAG_TAU_S / spec_of(rows[0]["building_id"]).tick_seconds)
    family = {r["building_id"]: r["family"] for r in rows}
    ctx = Context(eps, family, methods, control, cells, reps, draws, tau)
    main = estimands([r for r in rows if r["split"] == lists[0]], ctx, tests=True)
    gain, cov_loss = main["h2_loss"]["gain"], main["h2_coverage"]["gain"]
    if gain is None or cov_loss is None:
        verdict = "not supported (gain or coverage loss undefined: zero control mean)"
    else:
        verdict = h2_verdict(_iv(gain), _iv(cov_loss))
    report["primary"] = {
        "list": lists[0],
        "verdict": verdict,
        "rule": "gain low >= 0.10 and coverage-loss high <= 0.05; p-values not used",
        **main,
    }
    if not dry_run:
        ood = [r for r in rows if r["split"] == TEST_OOD]
        report["ood"] = {
            fam: estimands([r for r in ood if r["family"] == fam], ctx, tests=False)
            for fam in sorted({r["family"] for r in ood})
        }
    log(out, report, dry_run)
    return report


def _iv(x: dict[str, Any]) -> Interval:
    return Interval(x["estimate"], x["low"], x["high"])


def log(out: Path, report: dict[str, Any], dry_run: bool) -> None:
    """Every analysis of a test batch is logged next to it (no silent repeats)."""
    if dry_run:
        return
    entry = {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "frozen_sha256": report["frozen_sha256"],
        "analysis_code": report["analysis_code"],
        "prefix_b": report.get("prefix_b"),
        "verdict": report.get("primary", {}).get("verdict", report.get("verdict")),
    }
    with (out / "analysis_log.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")


def markdown(r: dict[str, Any]) -> str:
    def iv(x: dict[str, Any] | None) -> str:
        if x is None:
            return "undefined"
        return f"{x['estimate']:+.4f} [{x['low']:+.4f}, {x['high']:+.4f}]"

    lines = [
        "# X1/X2 analysis" + (" (dry run, not a test result)" if r["dry_run"] else ""),
        "",
        f"Generated by `tools/analyze_x1x2.py`. Frozen run set `{r['frozen_sha256']}`; "
        f"bootstrap {r['draws']} draws, seed {r['seed']}.",
        "",
    ]
    if "primary" not in r:
        return "\n".join([*lines, f"**{r['verdict']}**", ""])
    p = r["primary"]
    lines += [
        f"Candidate `{r['candidate']}`, H2 control `{r['h2_control']}`, list `{p['list']}`, "
        f"B = {r['prefix_b']} positions ({p['buildings']} buildings: {p['families']}).",
        "",
        f"- **Gain** 1 − L_cand/L_ctrl: {iv(p['h2_loss']['gain'])}; "
        f"difference {iv(p['h2_loss']['difference'])}",
        f"- **Relative coverage loss** (closed, final): {iv(p['h2_coverage']['gain'])}",
        f"- **H2: {p['verdict']}** ({p['rule']})",
        "",
        "| Method | Mean loss | Coverage closed (final) | Killed |",
        "|---|---|---|---|",
    ]
    for m in sorted(p["mean_loss"]):
        lines.append(
            f"| {m} | {iv(p['mean_loss'][m])} | {iv(p['coverage_closed_final'][m])} "
            f"| {p['killed'][m]} |"
        )
    holm_p = p["holm_family"]["p_holm"]
    lines += ["", "## Secondary", "", "| Cell | Gain | p (Holm) |", "|---|---|---|"]
    for c, x in p["gain_per_cell"].items():
        lines.append(f"| {c} | {iv(x['gain'])} | {holm_p[f'cell {c}']:.4f} |")
    lines += ["", "| H1 contrast (candidate) | Estimate | p (Holm) |", "|---|---|---|"]
    for n, x in p["h1_interaction_contrasts"][r["candidate"]].items():
        lines.append(f"| {n} | {iv(x)} | {holm_p[f'h1 {n}']:.4f} |")
    lines += ["", f"Sensitivity ({p['sensitivity']['note']}):", "", "| Variant | Gain |"]
    lines.append("|---|---|")
    for name, x in p["sensitivity"].items():
        if isinstance(x, dict):
            lines.append(f"| {name} | {iv(x['gain'])} |")
    if "ood" in r:
        lines += ["", "## OOD (per family, descriptive)", "", "| Family | Gain | Coverage loss |"]
        lines.append("|---|---|---|")
        for fam, x in r["ood"].items():
            lines.append(f"| {fam} | {iv(x['h2_loss']['gain'])} | {iv(x['h2_coverage']['gain'])} |")
    lines += [
        "",
        "Calibration, JS, repair lags, bytes and runtime are in the JSON next to this report.",
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
    result = analyze(args.out, args.dry_run)
    args.json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.report.write_text(markdown(result), encoding="utf-8")
    print(f"wrote {args.json} and {args.report}")


if __name__ == "__main__":
    main()
