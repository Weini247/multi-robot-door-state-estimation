"""Frozen X1/X2 test run set and the test episode worker (ARGUS-E08-T02).

`configs/experiments/X1_X2_frozen.json` fixes, from committed evidence only,
what the test batch runs: the candidate and every control with its
validation-selected configuration (VAL_TUNING_v2), the 8 cells, 2 replicates,
the lists id_test and ood_test and the planned prefix B from the calibration
pilot. `ImmutableRunSet.load` recomputes all of it and refuses on any mismatch.

A test run additionally needs `configs/experiments/X1_X2_approval.json`, written
only after the user's explicit approval of the batch budget. Without it the run
is refused; the budget caps B (largest even B within budget, at least 60).

Failure rule (preregistration v1, failure_rules.timeout_or_error): a failed
episode is kept as a failure; probes answered before the failure tick keep their
loss and every probe from the failure tick on gets the loss of the default
decision "closed" against the true passage state. The true state comes from the
exogenous passage timeline, a pure function of (world seed, scenario, passage,
tick), so it is exact without the failed run. No flat penalty is used.

Guards (audit 2026-11-02): every episode checks in its own process that the
source tree still has the hash the batch started with, and a test-split episode
runs only if the approval file exists with the hash named in its payload.

Seeds depend on (building, replicate) only (protocol.episode_seeds). Raw data of
every episode (all probes, defaults, passage changes, coverage events, bytes) is
written to raw/ next to the results so that E08-T03 needs no rerun.
"""

from __future__ import annotations

import dataclasses
import gzip
import hashlib
import json
import math
import os
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from argus.core.events import canonical_json
from argus.experiments.batch import (
    BatchJob,
    BatchUsage,
    building_losses,
    plan_test_size,
)
from argus.experiments.cells import CELLS
from argus.experiments.protocol import (
    TuningResult,
    episode_config,
    file_sha256,
    load_manifest_entries,
    load_preregistration,
    manifest_params,
    select,
)
from argus.experiments.runner import (
    TARGET_SETS,
    EpisodeConfig,
    EpisodeResult,
    EvaluationRecord,
    ExperimentRunner,
    InspectionRequest,
    binary_loss,
    generate_requests,
)
from argus.experiments.splits import SplitEntry
from argus.policies.baselines import METHOD_SPECS
from argus.world.floorplan import FloorplanParams
from argus.world.scenario import ExogenousWorld

FROZEN_SCHEMA = "argus-x1x2-frozen-v1"
FROZEN_PATH = "configs/experiments/X1_X2_frozen.json"
APPROVAL_PATH = "configs/experiments/X1_X2_approval.json"
TEST_LISTS = ("id_test", "ood_test")
AMENDMENT_PATH = "configs/experiments/preregistration_v1_amendment_01.json"
AMENDMENT_02_PATH = "configs/experiments/preregistration_v1_amendment_02.json"
# v2 evidence after amendment 01 (O1, fair channel); v1 stays committed and superseded
VAL_RESULTS = "docs/results/val_tuning_v2.jsonl"
PILOT_DIR = "docs/results/calib_pilot_v2"
PROBE_DIR = "docs/results/test_probe_calib_v3"  # cost model of the test worker, if present
CANDIDATE = "repair"
HARD_TIMEOUT_S = 900.0
B_MINIMUM = 60


class RunSetError(ValueError):
    """The frozen run set does not match the committed evidence, or a run is not allowed."""


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def selected_methods(
    prereg: Mapping[str, Any], val_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Candidate and every control with the configuration the preregistered rule selected."""
    roles = {m: s.role for m, s in METHOD_SPECS.items()}
    sel = select(
        [
            TuningResult(
                r["method_id"],
                r["config_id"],
                r["building_id"],
                r["cell_id"],
                r["mean_loss"],
                r["sent_bytes"],
                r["status"],
            )
            for r in val_rows
        ],
        roles,
        prereg["tuning"]["minimum_complete_buildings"],
    )
    if sel.h2_control is None:
        raise RunSetError(f"no H2 control: {sel.reason}")
    keep = [CANDIDATE, *prereg["methods"]["controls"]]
    grid = prereg["tuning"]["grid"]
    return {
        "h2_control": sel.h2_control,
        "selected": {
            m: {
                "role": roles[m],
                "config_id": sel.best_config[m],
                "params": grid[m][sel.best_config[m]],
            }
            for m in sorted(keep)
        },
    }


def cpu_seconds(
    val_rows: Sequence[Mapping[str, Any]], selected: Mapping[str, Any]
) -> dict[str, float]:
    """Measured mean CPU seconds per episode of each selected configuration (validation)."""
    out = {}
    for method, info in sorted(selected.items()):
        values = [
            r["cpu_s"]
            for r in val_rows
            if r["method_id"] == method and r["config_id"] == info["config_id"]
        ]
        out[method] = round(math.fsum(values) / len(values), 3)
    return out


def source_sha256(root: Path) -> str:
    """Content hash of the package sources the workers import."""
    digest = hashlib.sha256()
    base = root / "src" / "argus"
    for path in sorted(base.rglob("*.py")):
        digest.update(path.relative_to(base).as_posix().encode("utf-8") + b"\0")
        digest.update(path.read_bytes().replace(b"\r\n", b"\n") + b"\0")
    return digest.hexdigest()


def probe_cost(root: Path) -> dict[str, Any] | None:
    """Measured CPU seconds per episode and wall/CPU of the test worker (calib probe)."""
    probe = root / PROBE_DIR
    if not (probe / "results.jsonl").exists():
        return None
    rows = _rows(probe / "results.jsonl")
    usage = BatchUsage.load(probe / "usage.json")
    per: dict[str, list[float]] = {}
    for r in rows:
        per.setdefault(r["method_id"], []).append(r["cpu_s"])
    return {
        "source": f"{PROBE_DIR} (test worker, calib only)",
        "cpu_s_per_episode": {m: round(math.fsum(v) / len(v), 3) for m, v in sorted(per.items())},
        "wall_per_cpu": round(usage.wall_s / usage.cpu_s, 4),
    }


def build_frozen(root: Path) -> dict[str, Any]:
    """The frozen run set, derived only from committed files."""
    prereg = load_preregistration(root)
    val_rows = _rows(root / VAL_RESULTS)
    methods = selected_methods(prereg, val_rows)
    pilot = root / PILOT_DIR
    pilot_rows = _rows(pilot / "results.jsonl")
    usage = BatchUsage.load(pilot / "usage.json")
    family = {r["building_id"]: r["family"] for r in pilot_rows}
    size = plan_test_size(
        building_losses(pilot_rows, CANDIDATE, len(CELLS)),
        building_losses(pilot_rows, methods["h2_control"], len(CELLS)),
        family,
    )
    if {r["method_id"] for r in pilot_rows} != {CANDIDATE, methods["h2_control"]}:
        raise RunSetError("the pilot did not run the candidate against the H2 control")
    test = prereg["test"]
    amendment = json.loads((root / AMENDMENT_PATH).read_text(encoding="utf-8"))
    second = json.loads((root / AMENDMENT_02_PATH).read_text(encoding="utf-8"))
    prereg_sha = file_sha256(root / "configs/experiments/preregistration_v1.json")
    if amendment["amends"]["sha256"] != prereg_sha or second["amends"]["sha256"] != prereg_sha:
        raise RunSetError("an amendment does not amend this preregistration")
    if second["follows"]["sha256"] != file_sha256(root / AMENDMENT_PATH):
        raise RunSetError("amendment 02 does not follow this amendment 01")
    decisions = {k: v for k, v in amendment["decisions"].items() if k != "O1_channel_fairness"}
    decisions["O6_definitions"] = {  # amendment 02, D2 replaces the repair-lag definition
        **{k: v for k, v in decisions["O6_definitions"].items() if k != "repair_lag"},
        "repair_lag": "see D2_repair_lag",
    }
    decisions.update(second["decisions"])
    return {
        "schema": FROZEN_SCHEMA,
        "status": "frozen before any test episode; a run also needs " + APPROVAL_PATH,
        "sources": {
            "preregistration": {
                "path": "configs/experiments/preregistration_v1.json",
                "sha256": file_sha256(root / "configs/experiments/preregistration_v1.json"),
            },
            "amendment": {"path": AMENDMENT_PATH, "sha256": file_sha256(root / AMENDMENT_PATH)},
            "amendment_02": {
                "path": AMENDMENT_02_PATH,
                "sha256": file_sha256(root / AMENDMENT_02_PATH),
            },
            "split_manifest": dict(prereg["split_manifest"]),
            "validation_results": {"path": VAL_RESULTS, "sha256": file_sha256(root / VAL_RESULTS)},
            "pilot_results": {
                "path": f"{PILOT_DIR}/results.jsonl",
                "sha256": file_sha256(pilot / "results.jsonl"),
            },
            **(
                {
                    "probe_results": {
                        "path": f"{PROBE_DIR}/results.jsonl",
                        "sha256": file_sha256(root / PROBE_DIR / "results.jsonl"),
                    }
                }
                if (root / PROBE_DIR / "results.jsonl").exists()
                else {}
            ),
        },
        "methods": methods,
        "analysis": {
            "source": [AMENDMENT_PATH, AMENDMENT_02_PATH],
            "decisions": decisions,
        },
        "cells": [c.cell_id for c in CELLS],
        "test": {
            "lists": list(TEST_LISTS),
            "replicates": test["replicates"],
            "b_planned": size.b_planned,
            "b_minimum": B_MINIMUM,
            "sd_delta": round(size.sd_used, 6),
            "order": "positions 0..B-1, per position id_test then ood_test; within a "
            "building replicate, method, cell",
            "replacement": test["replacement"],
            "budget_rule": "B = largest even number <= b_planned whose projected CPU and wall "
            "hours fit the approved budget; below b_minimum the batch is not run "
            "('inconclusive by design')",
        },
        "episode": {
            "worker": "argus.experiments.runset.test_episode",
            "trace": True,
            "soft_timeout_s": EpisodeConfig.__dataclass_fields__["max_wall_seconds"].default,
            "hard_timeout_s": HARD_TIMEOUT_S,
            "failure_rule": prereg["failure_rules"]["timeout_or_error"],
            "seeds": "protocol.episode_seeds(building_id, replicate)",
        },
        "cost_model": {
            **(
                probe_cost(root)
                or {
                    "source": "validation tuning CPU and pilot wall/CPU",
                    "cpu_s_per_episode": cpu_seconds(val_rows, methods["selected"]),
                    "wall_per_cpu": round(usage.wall_s / usage.cpu_s, 4),
                }
            ),
            "workers": prereg["tuning"]["budget"]["workers"],
        },
    }


def render_frozen(root: Path) -> str:
    return json.dumps(build_frozen(root), indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def cost_per_position(frozen: Mapping[str, Any]) -> tuple[float, float]:
    """Projected CPU and wall hours of one list position (both lists, all replicates)."""
    cost = frozen["cost_model"]
    episodes = len(TEST_LISTS) * len(frozen["cells"]) * frozen["test"]["replicates"]
    cpu_h = episodes * math.fsum(cost["cpu_s_per_episode"].values()) / 3600
    return cpu_h, cpu_h * cost["wall_per_cpu"]


def b_for_budget(frozen: Mapping[str, Any], cpu_hours: float, wall_hours: float) -> int | None:
    cpu_h, wall_h = cost_per_position(frozen)
    b = min(
        frozen["test"]["b_planned"],
        math.floor(cpu_hours / cpu_h),
        math.floor(wall_hours / wall_h),
    )
    b -= b % 2
    return b if b >= frozen["test"]["b_minimum"] else None


@dataclass(frozen=True)
class ImmutableRunSet:
    root: Path
    frozen: dict[str, Any]
    prereg: dict[str, Any]
    entries: tuple[SplitEntry, ...]
    sha256: str

    @classmethod
    def load(cls, root: Path) -> ImmutableRunSet:
        path = root / FROZEN_PATH
        text = path.read_text(encoding="utf-8")
        if text != render_frozen(root):
            raise RunSetError(f"{FROZEN_PATH} does not match the committed evidence")
        prereg = load_preregistration(root)
        frozen = json.loads(text)
        return cls(
            root,
            frozen,
            prereg,
            load_manifest_entries(root, prereg),
            hashlib.sha256(text.encode("utf-8")).hexdigest(),
        )

    def approval(self) -> dict[str, Any]:
        path = self.root / APPROVAL_PATH
        if not path.exists():
            raise RunSetError(f"no approval: {APPROVAL_PATH} is missing; the test batch is not run")
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        if data.get("frozen_sha256") != self.sha256:
            raise RunSetError("the approval refers to another frozen run set")
        for key in ("approved_by", "date", "cpu_hours", "wall_hours", "src_sha256"):
            if key not in data:
                raise RunSetError(f"approval lacks {key}")
        return data

    def methods(self) -> list[tuple[str, str, dict[str, float]]]:
        selected = self.frozen["methods"]["selected"]
        return [(m, v["config_id"], dict(v["params"])) for m, v in sorted(selected.items())]

    def buildings(
        self, split: str, b: int, valid: Callable[[SplitEntry], bool] | None = None
    ) -> tuple[list[SplitEntry], list[tuple[str, str]]]:
        """First b entries of a list. With `valid`, an invalid scenario is replaced by the
        next unused entry of the same list and family (preregistered replacement rule);
        the replacements are returned for the log."""
        ordered = sorted((e for e in self.entries if e.split == split), key=lambda e: e.position)
        if b > len(ordered):
            raise RunSetError(f"{split} has only {len(ordered)} entries")
        chosen, spare, replaced = list(ordered[:b]), list(ordered[b:]), []
        if valid is not None:
            for index, entry in enumerate(chosen):
                if valid(entry):
                    continue
                while True:
                    candidates = [e for e in spare if e.family == entry.family]
                    if not candidates:
                        raise RunSetError(f"no replacement left for {entry.building_id}")
                    spare.remove(candidates[0])
                    if valid(candidates[0]):
                        break
                chosen[index] = candidates[0]
                replaced.append((entry.building_id, candidates[0].building_id))
        return chosen, replaced

    def jobs(
        self,
        splits: Sequence[str],
        b: int,
        replicates: int,
        out: Path,
        code_commit: str,
        positions: Sequence[int] | None = None,
        valid: Callable[[SplitEntry], bool] | None = None,
        approval_sha256: str | None = None,
    ) -> list[BatchJob]:
        """Jobs for the first b positions of each list (or the given positions), in the
        frozen order: per position each list, then replicate, method and cell."""
        params = manifest_params(self.root, self.prereg).to_json()
        src = source_sha256(self.root)
        lists = {s: self.buildings(s, b, valid)[0] for s in splits}
        chosen = list(positions) if positions is not None else list(range(b))
        out_jobs = []
        for position in chosen:
            for split in splits:
                entry = lists[split][position]
                for replicate in range(replicates):
                    for method, config_id, values in self.methods():
                        for cell in CELLS:
                            key = (
                                f"{split}/{entry.building_id}/{cell.cell_id}/"
                                f"{method}/{config_id}/{replicate}"
                            )
                            payload = {
                                "entry": list(entry.to_json()),
                                "allowed_splits": list(splits),
                                "params": params,
                                "cell_id": cell.cell_id,
                                "method_id": method,
                                "config_id": config_id,
                                "method_params": values,
                                "replicate": replicate,
                                "key": key,
                                "out": str(out),
                                "frozen_sha256": self.sha256,
                                "code_commit": code_commit,
                                "root": str(self.root),
                                "src_sha256": src,
                                "approval_sha256": approval_sha256,
                            }
                            out_jobs.append(BatchJob(key, f"{split}/{entry.building_id}", payload))
        return out_jobs


# --- test episode ---------------------------------------------------------------------


def _config(payload: Mapping[str, Any]) -> tuple[SplitEntry, EpisodeConfig]:
    entry = SplitEntry(*payload["entry"])
    if entry.split not in payload["allowed_splits"]:
        raise RunSetError(f"{entry.building_id} is in {entry.split}, not allowed here")
    root = Path(payload["root"])
    if entry.split in TEST_LISTS:
        approval = root / APPROVAL_PATH
        if not approval.exists() or payload.get("approval_sha256") != file_sha256(approval):
            raise RunSetError(f"{entry.building_id}: test episode without a matching approval")
    if source_sha256(root) != payload["src_sha256"]:
        raise RunSetError("the source tree changed since the batch started")
    params = dict(payload["params"])
    params["passage_rates"] = tuple(params["passage_rates"])
    cell = next(c for c in CELLS if c.cell_id == payload["cell_id"])
    config = episode_config(
        entry,
        FloorplanParams(**params),
        cell,
        payload["method_id"],
        payload["config_id"],
        payload["method_params"],
        payload["replicate"],
    )
    return entry, dataclasses.replace(config, code_commit=payload["code_commit"])


def failure_rule(
    config: EpisodeConfig,
    requests: Sequence[InspectionRequest],
    answered: Iterable[EvaluationRecord],
    failure_tick: int | None,
) -> tuple[list[EvaluationRecord], list[tuple[InspectionRequest, int, float]]]:
    """Keep probes answered before the failure; default 'closed' for every other probe."""
    kept = [r for r in answered if failure_tick is None or r.tick < failure_tick]
    done = {(r.tick, r.robot_id, r.entity) for r in kept}
    world = ExogenousWorld(config.scenario, config.seed_world)
    defaults = []
    for req in requests:
        if (req.tick, req.recipient, req.entity) in done:
            continue
        truth = world.state(req.entity, req.tick)
        loss = binary_loss(0, truth, config.loss_false_open, config.loss_false_closed)
        defaults.append((req, truth, loss))
    if len(kept) + len(defaults) != len(requests):
        raise RunSetError("answered records do not match the probe plan")
    return kept, defaults


def coverage_from_events(
    events: Sequence[Sequence[int]], targets: int, horizon: int
) -> tuple[float, float]:
    """Final fraction and time-average of covered targets from (tick, count) events."""
    area, last_tick, last = 0.0, 0, 0
    for tick, count in events:
        if tick > horizon:
            break
        area += last * (tick - last_tick)
        last_tick, last = tick, count
    area += last * (horizon - last_tick)
    return last / targets, area / (targets * horizon)


def summarize(
    payload: Mapping[str, Any],
    entry: SplitEntry,
    config: EpisodeConfig,
    result: EpisodeResult | None,
    status: str,
    error: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Summary row and raw record of a test episode under the failure rule."""
    requests = generate_requests(config)
    if result is None:  # killed by the hard timeout: no probe survives
        answered: tuple[EvaluationRecord, ...] = ()
        failure_tick: int | None = 0
    elif status == "ok":
        answered, failure_tick = result.records, None
    else:
        answered, failure_tick = result.partial_records, result.failure_tick
    if status == "ok" and result is not None and result.unanswered:
        status, error = "error", f"ok episode left {result.unanswered} probes unanswered"
        failure_tick = max((r.tick for r in answered), default=-1) + 1
    kept, defaults = failure_rule(config, requests, answered, failure_tick)
    losses = [r.decision_loss for r in kept] + [loss for _, _, loss in defaults]
    trace = result.trace if result is not None else None
    horizon = config.scenario.horizon_ticks
    coverage: dict[str, Any] = {}
    if trace is not None:
        for name in TARGET_SETS:
            final, auc = coverage_from_events(
                trace["coverage_events"][name], trace["targets"][name], horizon
            )
            coverage[f"coverage_{name}_final"] = final
            coverage[f"coverage_{name}_auc"] = auc
    channel = trace["channel_tx_bytes"] if trace is not None else {}
    probs = [r.probability for r in kept]
    truths = [r.truth for r in kept]
    raw = {
        "key": payload["key"],
        "frozen_sha256": payload["frozen_sha256"],
        "code_commit": payload["code_commit"],
        "config": config.to_json(),
        "config_sha256": config.config_sha256,
        "manifest": result.manifest if result is not None else None,
        "status": status,
        "error": error,
        "failure_tick": failure_tick,
        "probes": [[r.tick, r.recipient, r.entity] for r in requests],
        "answered": [
            [
                r.tick,
                r.robot_id,
                r.entity,
                r.probability,
                r.truth,
                r.decision,
                r.decision_loss,
                r.sent_bytes,
            ]
            for r in kept
        ],
        "defaulted": [[q.tick, q.recipient, q.entity, truth, loss] for q, truth, loss in defaults],
        "trace": trace,
    }
    row = {
        "key": payload["key"],
        "split": entry.split,
        "position": entry.position,
        "building_id": entry.building_id,
        "family": entry.family,
        "cell_id": payload["cell_id"],
        "method_id": payload["method_id"],
        "config_id": payload["config_id"],
        "replicate": payload["replicate"],
        "status": status,
        "error": error,
        "failure_tick": failure_tick,
        "requests": len(requests),
        "answered": len(kept),
        "defaulted": len(defaults),
        "mean_loss": math.fsum(losses) / len(losses),
        "brier_answered": (
            math.fsum((p - z) ** 2 for p, z in zip(probs, truths, strict=True)) / len(kept)
            if kept
            else None
        ),
        "sent_bytes_total": sum(channel.values()),
        "sent_bytes_max_robot": max(channel.values(), default=0),
        "passage_changes": (
            sum(len(v) for v in trace["passage_changes"].values()) if trace is not None else None
        ),
        "wall_s": round(result.wall_seconds, 3) if result is not None else None,
        "config_sha256": config.config_sha256,
        **coverage,
    }
    return row, raw


def _write_raw(
    payload: Mapping[str, Any], raw: Mapping[str, Any], infra: bool = False
) -> tuple[str, str]:
    """Raw record next to the results; a killed episode's record goes to raw/infra/ with a
    timestamp so that a rerun never replaces it."""
    out = Path(payload["out"])
    name = payload["key"].replace("/", "__")
    if infra:
        name += f".killed-{time.strftime('%Y%m%dT%H%M%S')}-{os.getpid()}"
        rel = Path("raw") / "infra" / (name + ".json.gz")
    else:
        rel = Path("raw") / payload["key"].split("/")[0] / (name + ".json.gz")
    target = out / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    data = gzip.compress(canonical_json(dict(raw)).encode("utf-8"), mtime=0)
    tmp = target.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_bytes(data)
    tmp.replace(target)
    return rel.as_posix(), hashlib.sha256(data).hexdigest()


def test_episode(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Worker: one test episode with trace, failure rule and raw record."""
    entry, config = _config(payload)
    result = ExperimentRunner(trace=True).run(config)
    if source_sha256(Path(payload["root"])) != payload["src_sha256"]:
        raise RunSetError("the source tree changed during the episode")
    row, raw = summarize(payload, entry, config, result, result.status, result.error)
    row["raw"], row["raw_sha256"] = _write_raw(payload, raw)
    return row


def killed_episode(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Parent-side row for an episode killed by the hard timeout (all probes default)."""
    entry, config = _config(payload)
    row, raw = summarize(payload, entry, config, None, "timeout", "killed by the hard timeout")
    row["raw"], row["raw_sha256"] = _write_raw(payload, raw, infra=True)
    return row
