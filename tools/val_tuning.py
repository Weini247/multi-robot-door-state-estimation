"""Validation tuning under the preregistered budget (ARGUS-E07-T06).

    python tools/val_tuning.py run --out runs/val_tuning_v1
    python tools/val_tuning.py select --out runs/val_tuning_v1
    python tools/val_tuning.py throughput --episodes 10   # development seeds only

`run` works building by building in the preregistered order: all tuning
configurations x 8 cells of one validation building, then the next. A building
is started only if the projected wall and CPU time still fit the caps, so a stop
leaves complete buildings. One process per episode. Results are appended to
results.jsonl; a rerun skips finished episodes. `select` applies the
preregistered selection rule. `throughput` measures on development seeds and
never touches split buildings.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Any

from argus.experiments.cells import CELLS
from argus.experiments.protocol import (
    TuningResult,
    episode_config,
    load_manifest_entries,
    load_preregistration,
    manifest_params,
    select,
    tuning_buildings,
    tuning_configs,
)
from argus.experiments.runner import EpisodeConfig, ExperimentRunner
from argus.experiments.splits import SplitEntry
from argus.policies.baselines import METHOD_SPECS
from argus.world.floorplan import generate_floorplan

ROOT = Path(__file__).resolve().parents[1]


def _run(job: tuple[Any, ...]) -> dict[str, Any]:
    entry_row, params_json, cell_id, method, config_id, values, replicate, penalty = job
    from argus.world.floorplan import FloorplanParams

    params = dict(params_json)
    params["passage_rates"] = tuple(params["passage_rates"])
    cell = next(c for c in CELLS if c.cell_id == cell_id)
    entry = SplitEntry(*entry_row)
    config = episode_config(
        entry, FloorplanParams(**params), cell, method, config_id, values, replicate
    )
    return _episode(config, method, config_id, entry.building_id, cell_id, penalty)


def _episode(
    config: EpisodeConfig, method: str, config_id: str, building: str, cell_id: str, penalty: float
) -> dict[str, Any]:
    started = time.process_time()
    result = ExperimentRunner().run(config)
    cpu = time.process_time() - started
    ok = result.status == "ok" and result.records
    mean_loss = (
        sum(r.decision_loss for r in result.records) / len(result.records) if ok else penalty
    )
    sent = max((r.sent_bytes for r in result.records), default=0)
    return {
        "method_id": method,
        "config_id": config_id,
        "building_id": building,
        "cell_id": cell_id,
        "status": result.status,
        "mean_loss": mean_loss,
        "sent_bytes": sent,
        "requests": len(result.records) + result.unanswered,
        "wall_s": round(result.wall_seconds, 3),
        "cpu_s": round(cpu, 3),
        "error": result.error,
    }


def run(out: Path) -> None:
    prereg = load_preregistration(ROOT)
    tuning = prereg["tuning"]
    entries = load_manifest_entries(ROOT, prereg)
    params = manifest_params(ROOT, prereg).to_json()
    buildings = tuning_buildings(entries, tuning["buildings"])
    configs = tuning_configs(prereg)
    out.mkdir(parents=True, exist_ok=True)
    results_path = out / "results.jsonl"
    done = set()
    if results_path.exists():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done.add((r["building_id"], r["method_id"], r["config_id"], r["cell_id"]))
    budget = tuning["budget"]
    wall_cap, cpu_cap = budget["wall_hours"] * 3600, budget["cpu_hours"] * 3600
    started = time.time()
    cpu_used = 0.0
    per_building: list[tuple[float, float]] = []
    log = out / "progress.log"
    with mp.get_context("spawn").Pool(budget["workers"], maxtasksperchild=1) as pool:
        for entry in buildings:
            jobs = [
                (
                    entry.to_json(),
                    params,
                    cell.cell_id,
                    method,
                    config_id,
                    values,
                    0,
                    tuning["failure_penalty_loss"],
                )
                for method, config_id, values in configs
                for cell in CELLS
                if (entry.building_id, method, config_id, cell.cell_id) not in done
            ]
            if per_building and jobs:
                wall_est = max(w for w, _ in per_building)
                cpu_est = max(c for _, c in per_building)
                if time.time() - started + wall_est > wall_cap or cpu_used + cpu_est > cpu_cap:
                    _log(log, f"STOP before {entry.building_id}: projected budget exceeded")
                    break
            t0, c0 = time.time(), cpu_used
            with results_path.open("a", encoding="utf-8") as handle:
                for row in pool.imap_unordered(_run, jobs):
                    cpu_used += row["cpu_s"]
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                    handle.flush()
            if jobs:
                per_building.append((time.time() - t0, cpu_used - c0))
            _log(
                log,
                f"{entry.building_id}: {len(jobs)} episodes, wall {time.time() - t0:.0f}s, "
                f"cpu total {cpu_used / 3600:.2f}h, elapsed {(time.time() - started) / 3600:.2f}h",
            )


def _log(path: Path, text: str) -> None:
    line = time.strftime("%H:%M:%S ") + text
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def do_select(out: Path) -> None:
    prereg = load_preregistration(ROOT)
    rows = [
        json.loads(line)
        for line in (out / "results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    results = [
        TuningResult(
            r["method_id"],
            r["config_id"],
            r["building_id"],
            r["cell_id"],
            r["mean_loss"],
            r["sent_bytes"],
            r["status"],
        )
        for r in rows
    ]
    roles = {m: s.role for m, s in METHOD_SPECS.items()}
    selection = select(results, roles, prereg["tuning"]["minimum_complete_buildings"])
    report = {
        "complete_buildings": list(selection.complete_buildings),
        "episodes": len(rows),
        "status_counts": {
            s: sum(r["status"] == s for r in rows) for s in sorted({r["status"] for r in rows})
        },
        "per_config": {
            f"{m}/{c}": {"mean_loss": v[0], "mean_sent_bytes": v[1]}
            for (m, c), v in sorted(selection.per_config.items())
        },
        "best_config": selection.best_config,
        "h2_control": selection.h2_control,
        "reason": selection.reason,
        "cpu_hours": sum(r["cpu_s"] for r in rows) / 3600,
    }
    (out / "selection.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


def throughput(episodes: int, workers: int) -> None:
    """Development seeds (< DEV_SEED_LIMIT) only; no split building is run."""
    plans = [generate_floorplan(f, s) for s in range(episodes) for f in ("tree", "loop")][:episodes]
    jobs = [
        EpisodeConfig(f"dev-{i}", p.spec.scenario_id, p.spec, "repair", i, i, i, i, i)
        for i, p in enumerate(plans)
    ]
    started = time.time()
    with mp.get_context("spawn").Pool(workers, maxtasksperchild=1) as pool:
        rows = list(pool.imap_unordered(_dev, jobs))
    wall = time.time() - started
    cpu = sum(rows)
    print(
        f"{episodes} episodes, {workers} workers: wall {wall:.1f}s, cpu {cpu:.1f}s, "
        f"{wall / episodes:.2f}s wall/episode, {cpu / episodes:.2f}s cpu/episode, "
        f"cpus {os.cpu_count()}"
    )


def _dev(config: EpisodeConfig) -> float:
    started = time.process_time()
    ExperimentRunner().run(config)
    return time.process_time() - started


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "select", "throughput"])
    parser.add_argument("--out", type=Path, default=ROOT / "runs/val_tuning_v1")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    if args.command == "run":
        run(args.out)
    elif args.command == "select":
        do_select(args.out)
    else:
        throughput(args.episodes, args.workers)


if __name__ == "__main__":
    main()
