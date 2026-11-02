"""Calibration pre-pilot for the test size B (ARGUS-E08-T01).

    python tools/calib_pilot.py run --out runs/calib_pilot_v2
    python tools/calib_pilot.py plan --out runs/calib_pilot_v2 \
        --report docs/results/CALIB_PILOT_v2.md

`run` executes the candidate and the validation-selected H2 control on all calib
buildings with the test design (8 cells, 2 replicates) through BatchRunner.
`plan` derives sigma_Delta and B as preregistered and the planned size of the
test batch from measured CPU seconds. No id_test or ood_test building is run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from argus.experiments.batch import (
    BatchCaps,
    BatchJob,
    BatchRunner,
    BatchUsage,
    building_losses,
    plan_cost,
    plan_test_size,
    protocol_episode,
)
from argus.experiments.cells import CELLS
from argus.experiments.protocol import (
    load_manifest_entries,
    load_preregistration,
    manifest_params,
)

ROOT = Path(__file__).resolve().parents[1]
PILOT_CAPS = BatchCaps(wall_hours=1.0, cpu_hours=3.0, workers=5, episode_timeout_s=900.0)
# v1 ran before amendment 01 (method-named transport ids); v2 uses the v2 tuning selection.
SELECTIONS = {
    "v1": ROOT / "docs/results/val_tuning_v1.jsonl",
    "v2": ROOT / "docs/results/val_tuning_v2.jsonl",
}


def version_of(out: Path) -> str:
    return "v1" if out.name.endswith("v1") else "v2"


def pair(version: str) -> tuple[tuple[str, str], tuple[str, str]]:
    """Candidate and H2 control with their configurations from the tuning selection."""
    from argus.experiments.runset import selected_methods

    rows = load_rows(SELECTIONS[version])
    methods = selected_methods(load_preregistration(ROOT), rows)
    sel = methods["selected"]
    control = methods["h2_control"]
    return ("repair", sel["repair"]["config_id"]), (control, sel[control]["config_id"])


def jobs(prereg: dict[str, Any], version: str = "v2") -> list[BatchJob]:
    entries = load_manifest_entries(ROOT, prereg)
    calib = sorted((e for e in entries if e.split == "calib"), key=lambda e: e.position)
    params = manifest_params(ROOT, prereg).to_json()
    grid = prereg["tuning"]["grid"]
    out = []
    for entry in calib:
        for replicate in range(prereg["test"]["replicates"]):
            for method, config_id in pair(version):
                for cell in CELLS:
                    payload = {
                        "entry": list(entry.to_json()),
                        "params": params,
                        "cell_id": cell.cell_id,
                        "method_id": method,
                        "config_id": config_id,
                        "method_params": grid[method][config_id],
                        "replicate": replicate,
                        "failure_loss": prereg["tuning"]["failure_penalty_loss"],
                    }
                    key = f"{entry.building_id}/{cell.cell_id}/{method}/{config_id}/{replicate}"
                    out.append(BatchJob(key, entry.building_id, payload))
    return out


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def selected_cpu(version: str) -> dict[str, float]:
    """Mean CPU seconds per episode of every method's selected configuration (validation)."""
    rows = load_rows(SELECTIONS[version])
    per: dict[tuple[str, str], list[float]] = defaultdict(list)
    loss: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in rows:
        per[(r["method_id"], r["config_id"])].append(r["cpu_s"])
        loss[(r["method_id"], r["config_id"])].append(r["mean_loss"])
    best: dict[str, tuple[float, str]] = {}
    for (m, c), v in loss.items():
        mean = math.fsum(v) / len(v)
        if m not in best or (mean, c) < best[m]:
            best[m] = (mean, c)
    return {m: math.fsum(per[(m, c)]) / len(per[(m, c)]) for m, (_, c) in sorted(best.items())}


def plan(out: Path, report: Path | None) -> dict[str, Any]:
    version = version_of(out)
    CANDIDATE, CONTROL = pair(version)
    prereg = load_preregistration(ROOT)
    rows = load_rows(out / "results.jsonl")
    usage = BatchUsage.load(out / "usage.json")
    status = Counter(r["status"] for r in rows)
    cand = building_losses(rows, CANDIDATE[0], len(CELLS))
    ctrl = building_losses(rows, CONTROL[0], len(CELLS))
    family = {r["building_id"]: r["family"] for r in rows}
    size = plan_test_size(cand, ctrl, family)
    cpu = selected_cpu(version)
    pilot_cpu = {
        m: math.fsum(r["cpu_s"] for r in rows if r["method_id"] == m)
        / sum(r["method_id"] == m for r in rows)
        for m in (CANDIDATE[0], CONTROL[0])
    }
    wall_per_cpu = usage.wall_s / usage.cpu_s
    test = prereg["test"]
    cost = plan_cost(size.b_planned, cpu, len(CELLS), test["replicates"], 2, wall_per_cpu)
    minimum = plan_cost(60, cpu, len(CELLS), test["replicates"], 2, wall_per_cpu)
    rss = [r["peak_rss_bytes"] for r in rows if r.get("peak_rss_bytes")]
    walls = sorted(r["process_wall_s"] for r in rows)
    result = {
        "pilot_episodes": len(rows),
        "status": dict(status),
        "usage": {"wall_h": usage.wall_s / 3600, "cpu_h": usage.cpu_s / 3600},
        "sessions": usage.sessions,
        "wall_per_cpu": wall_per_cpu,
        "peak_rss_mib": {"max": max(rss) / 2**20, "median": sorted(rss)[len(rss) // 2] / 2**20},
        "process_wall_s": {"median": walls[len(walls) // 2], "max": walls[-1]},
        "pilot_cpu_s": pilot_cpu,
        "validation_cpu_s": cpu,
        "size": {
            "buildings": size.buildings,
            "families": size.families,
            "gain_calib": size.gain,
            "sd_pooled": size.sd_pooled,
            "sd_stratified": size.sd_stratified,
            "b_formula": size.b_formula,
            "b_planned": size.b_planned,
            "ci_half_width_at_b": size.ci_half_width_at_b,
        },
        "test_batch": cost,
        "test_batch_at_60": minimum,
    }
    if report is not None:
        write_report(report, out / "results.jsonl", result, version)
    return result


def write_report(path: Path, results: Path, r: dict[str, Any], version: str) -> None:
    CANDIDATE, CONTROL = pair(version)
    s, c, m = r["size"], r["test_batch"], r["test_batch_at_60"]
    digest = hashlib.sha256(results.read_bytes()).hexdigest()
    source = results.resolve().relative_to(ROOT).as_posix()
    families = ", ".join(f"{k} {v}" for k, v in s["families"].items())
    timeout = PILOT_CAPS.episode_timeout_s
    lines = [
        f"# Kalibrierungs-Vorpilot {version} (E08-T01)",
        "",
        f"Generiert von `tools/calib_pilot.py plan` aus `{source}` (SHA-256 `{digest}`).",
        f"Kandidat `{'/'.join(CANDIDATE)}`, H2-Kontrolle `{'/'.join(CONTROL)}` "
        f"(docs/results/VAL_TUNING_{version}.md), alle calib-Gebäude, 8 Zellen, "
        "2 Replikate wie im Test.",
        "",
        "## Durchsatz und Speicher",
        "",
        f"- Episoden: {r['pilot_episodes']}; Status: "
        + ", ".join(f"{k} {v}" for k, v in sorted(r["status"].items())),
        f"- Verbrauch: {r['usage']['wall_h']:.2f} h Wand, {r['usage']['cpu_h']:.2f} CPU-h "
        f"({r['sessions']} Sitzung(en)); Deckel {PILOT_CAPS.wall_hours} h / "
        f"{PILOT_CAPS.cpu_hours} CPU-h, {PILOT_CAPS.workers} Worker",
        f"- Wand pro CPU-Sekunde: {r['wall_per_cpu']:.3f}",
        f"- Prozesswand pro Episode: Median {r['process_wall_s']['median']:.1f} s, "
        f"Maximum {r['process_wall_s']['max']:.1f} s (harter Timeout {timeout:.0f} s)",
        f"- Peak-RSS pro Episodenprozess: Median {r['peak_rss_mib']['median']:.0f} MiB, "
        f"Maximum {r['peak_rss_mib']['max']:.0f} MiB",
        "- CPU-s pro Episode im Pilot: "
        + ", ".join(f"{k} {v:.1f}" for k, v in r["pilot_cpu_s"].items()),
        "- CPU-s pro Episode der gewählten Konfigurationen (Validation): "
        + ", ".join(f"{k} {v:.1f}" for k, v in r["validation_cpu_s"].items()),
        "",
        "## Testgröße B",
        "",
        f"- Gepaarte Gebäude: {s['buildings']} ({families})",
        f"- σ̂_Δ (Einfluss des Ratio-Gains): gepoolt {s['sd_pooled']:.4f}, "
        f"familienstratifiziert {s['sd_stratified']:.4f}; verwendet das größere",
        f"- (2 · 1,96 · σ̂_Δ / 0,05)² = {s['b_formula']:.1f} → **B = {s['b_planned']}**",
        f"- erwartete CI-Halbbreite des Gains bei B: ±{s['ci_half_width_at_b']:.3f}",
        "",
        "## Geplanter Testbatch",
        "",
        "7 Methoden (Kandidat und alle Kontrollen) × 8 Zellen × 2 Replikate × 2 Listen "
        "(id_test, ood_test), Kosten aus gemessenen CPU-s pro Episode:",
        "",
        "| B | Episoden | CPU-h | Wand-h (5 Worker) |",
        "|---|---|---|---|",
        f"| {s['b_planned']} | {c['episodes']:.0f} | {c['cpu_hours']:.1f} "
        f"| {c['wall_hours']:.1f} |",
        f"| 60 (Minimum) | {m['episodes']:.0f} | {m['cpu_hours']:.1f} | {m['wall_hours']:.1f} |",
        "",
        f"Der Gain auf calib ({s['gain_calib']:+.3f}) dient nur der Varianzschätzung. Er ist "
        "kein Ergebnis zu H2 und wird nicht als solches berichtet. σ̂_Δ aus "
        f"{s['buildings']} Gebäuden ist selbst unsicher; die Planung ist keine Powerzusage.",
    ]
    path.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "plan"])
    parser.add_argument("--out", type=Path, default=ROOT / "runs/calib_pilot_v2")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.command == "run":
        prereg = load_preregistration(ROOT)
        summary = BatchRunner(args.out, PILOT_CAPS, protocol_episode).run(
            jobs(prereg, version_of(args.out))
        )
        print(
            f"groups {len(summary.completed_groups)}, ran {summary.ran}, skipped "
            f"{summary.skipped}, stopped before {summary.stopped_before}, wall "
            f"{summary.usage.wall_s / 3600:.2f}h, cpu {summary.usage.cpu_s / 3600:.2f}h"
        )
    else:
        print(json.dumps(plan(args.out, args.report), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
