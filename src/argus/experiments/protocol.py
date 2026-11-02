"""Preregistration v1: episode construction, tuning grid and selection rule (ARGUS-E07-T06).

The frozen protocol lives in configs/experiments/preregistration_v1.json and
docs/EXPERIMENT_PROTOCOL.md; this module only turns it into episodes and applies
its selection rule. Seeds depend on (building, replicate) only, so every method,
configuration and cell of a building replicate shares world, sensor, channel,
request and policy seeds (common random numbers before behaviour diverges).
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from argus.experiments.cells import CELLS, FactorCell, apply_cell
from argus.experiments.runner import EpisodeConfig
from argus.experiments.splits import SplitEntry, SplitManifest, regenerate
from argus.policies.baselines import METHOD_SPECS
from argus.world.floorplan import FloorplanParams

PREREG_SCHEMA = "argus-preregistration-v1"
SEED_STREAMS = ("world", "sensor", "channel", "requests", "policy")


class ProtocolError(ValueError):
    """The preregistration is inconsistent with the code or the split manifest."""


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_preregistration(root: Path) -> dict[str, Any]:
    path = root / "configs/experiments/preregistration_v1.json"
    prereg: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if prereg.get("schema") != PREREG_SCHEMA:
        raise ProtocolError("unknown preregistration schema")
    manifest = prereg["split_manifest"]
    if file_sha256(root / manifest["path"]) != manifest["sha256"]:
        raise ProtocolError("split manifest file does not match the preregistered hash")
    for method, grid in prereg["tuning"]["grid"].items():
        spec = METHOD_SPECS[method]
        if not 1 <= len(grid) <= prereg["tuning"]["max_configs_per_method"]:
            raise ProtocolError(f"{method}: grid size outside the budget")
        for values in grid.values():
            if set(values) - set(spec.tunable):
                raise ProtocolError(f"{method}: grid uses undeclared tunables")
    return prereg


def load_manifest_entries(root: Path, prereg: Mapping[str, Any]) -> tuple[SplitEntry, ...]:
    data = json.loads((root / prereg["split_manifest"]["path"]).read_text(encoding="utf-8"))
    return tuple(SplitEntry(*row) for row in data["entries"])


def manifest_params(root: Path, prereg: Mapping[str, Any]) -> FloorplanParams:
    data = json.loads((root / prereg["split_manifest"]["path"]).read_text(encoding="utf-8"))
    params = dict(data["params"])
    params["passage_rates"] = tuple(params["passage_rates"])
    return FloorplanParams(**params)


def episode_seeds(building_id: str, replicate: int) -> tuple[int, int, int, int, int]:
    def seed(stream: str) -> int:
        digest = hashlib.sha256(f"argus-episode-v1/{building_id}/{replicate}/{stream}".encode())
        return int.from_bytes(digest.digest()[:6], "big")

    a, b, c, d, e = (seed(s) for s in SEED_STREAMS)
    return a, b, c, d, e


TRANSPORT_SCHEMA = "argus-transport-v1"


def transport_id(building_id: str, replicate: int) -> str:
    """Method- and cell-neutral transport key of fixed length (amendment 01, O1)."""
    digest = hashlib.sha256(f"{TRANSPORT_SCHEMA}/{building_id}/{replicate}".encode())
    return "t" + digest.hexdigest()[:15]


def episode_config(
    entry: SplitEntry,
    params: FloorplanParams,
    cell: FactorCell,
    method_id: str,
    config_id: str,
    method_params: Mapping[str, float],
    replicate: int,
) -> EpisodeConfig:
    plan = regenerate(entry, params)
    base = EpisodeConfig(
        f"{entry.building_id}/{cell.cell_id}/{method_id}/{config_id}/{replicate}",
        entry.building_id,
        plan.spec,
        method_id,
        *episode_seeds(entry.building_id, replicate),
        method_params=tuple(sorted(method_params.items())),
        transport_id=transport_id(entry.building_id, replicate),
    )
    return apply_cell(base, cell)


def tuning_configs(prereg: Mapping[str, Any]) -> list[tuple[str, str, dict[str, float]]]:
    return [
        (method, config_id, dict(values))
        for method, grid in sorted(prereg["tuning"]["grid"].items())
        for config_id, values in sorted(grid.items())
    ]


def tuning_buildings(manifest: SplitManifest | tuple[SplitEntry, ...], n: int) -> list[SplitEntry]:
    entries = manifest.entries if isinstance(manifest, SplitManifest) else manifest
    val = sorted((e for e in entries if e.split == "val"), key=lambda e: e.position)
    return val[:n]


@dataclass(frozen=True, slots=True)
class TuningResult:
    method_id: str
    config_id: str
    building_id: str
    cell_id: str
    mean_loss: float  # mean decision loss per request of the episode
    sent_bytes: float
    status: str


@dataclass(frozen=True, slots=True)
class Selection:
    complete_buildings: tuple[str, ...]
    per_config: dict[tuple[str, str], tuple[float, float]]  # (loss, bytes) means
    best_config: dict[str, str]
    h2_control: str | None
    reason: str


def select(
    results: Iterable[TuningResult], roles: Mapping[str, str], minimum_buildings: int
) -> Selection:
    """Preregistered rule: per configuration the mean over complete buildings of the
    mean over the eight cells; failed episodes keep their recorded loss. Per method
    the lowest configuration; the H2 control is the control method with the lowest
    tuned loss. Ties: lower mean sent bytes, then the id."""
    rows = list(results)
    expected = {(r.method_id, r.config_id) for r in rows}
    by_building: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
    for r in rows:
        by_building[r.building_id].add((r.method_id, r.config_id, r.cell_id))
    need = {(m, c, cell.cell_id) for m, c in expected for cell in CELLS}
    complete = tuple(sorted(b for b, seen in by_building.items() if seen >= need))
    if len(complete) < minimum_buildings:
        return Selection(
            complete, {}, {}, None, f"inconclusive: {len(complete)} complete buildings"
        )
    cells: dict[tuple[str, str, str], list[TuningResult]] = defaultdict(list)
    for r in rows:
        if r.building_id in complete:
            cells[(r.method_id, r.config_id, r.building_id)].append(r)
    per_config: dict[tuple[str, str], tuple[float, float]] = {}
    for key in sorted(expected):
        losses, sent = [], []
        for building in complete:
            group = cells[(*key, building)]
            losses.append(math.fsum(r.mean_loss for r in group) / len(group))
            sent.append(math.fsum(r.sent_bytes for r in group) / len(group))
        per_config[key] = (math.fsum(losses) / len(losses), math.fsum(sent) / len(sent))
    best: dict[str, str] = {}
    for method in sorted({m for m, _ in expected}):
        options = [(per_config[(method, c)], c) for m, c in expected if m == method]
        best[method] = min(options, key=lambda o: (o[0][0], o[0][1], o[1]))[1]
    controls = [m for m in best if roles.get(m) == "control"]
    if not controls:
        return Selection(complete, per_config, best, None, "no control method tuned")
    h2 = min(controls, key=lambda m: (*per_config[(m, best[m])], m))
    return Selection(complete, per_config, best, h2, "selected on complete validation buildings")
