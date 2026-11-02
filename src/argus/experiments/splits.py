"""Building splits by origin with global duplicate exclusion (ARGUS-E07-T06).

Every family has its own candidate seed stream (SEED_BASE + k). Candidates are
drawn round-robin over the families and accepted only if they are no near
duplicate (`is_near_duplicate`: label-preserving graph isomorphism or free-cell
IoU >= 0.90 under D4) of any building accepted so far, in any family or split.
Only after a family has all its buildings are they assigned to splits, by a
keyed random permutation. Splits are therefore disjoint by construction and
exchangeable within a family group: test buildings are not the "leftovers"
of a stream that train exhausted first.

Within a split the families alternate, so every prefix of a test list is
balanced; the analysis uses a preregistered prefix (E07-T06 protocol).

Seeds below DEV_SEED_LIMIT are development seeds (tests, exploration, smoke
runs) and never occur in a split.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from argus.core.events import KeyedRandom, canonical_json
from argus.world.floorplan import (
    FAMILIES,
    FloorGraph,
    Floorplan,
    FloorplanParams,
    Mask,
    free_mask,
    generate_floorplan,
    graph_hash,
    isomorphic,
)

SPLIT_SCHEMA = "argus-splits-v1"
SPLIT_ORDER = ("train", "val", "calib", "id_test", "ood_test")
SEED_BASE = {
    "tree": 10_000_000,
    "loop": 20_000_000,
    "double_loop": 30_000_000,
    "dogleg": 40_000_000,
}
DEV_SEED_LIMIT = 1_000_000
QUOTAS_V1: dict[str, dict[str, int]] = {
    "train": {"tree": 100, "loop": 100},
    "val": {"tree": 20, "loop": 20},
    "calib": {"tree": 10, "loop": 10},
    "id_test": {"tree": 75, "loop": 75},
    "ood_test": {"double_loop": 75, "dogleg": 75},
}
ASSIGNMENT_SEED = 20261102
IOU_THRESHOLD = 0.90


class SplitError(ValueError):
    """Inconsistent quotas or a manifest that does not match its regeneration."""


@dataclass(frozen=True, slots=True)
class SplitEntry:
    split: str
    position: int
    family: str
    seed: int
    attempt: int
    scenario_hash: str
    graph_hash: str

    @property
    def building_id(self) -> str:
        return f"{self.family}-{self.seed}"

    def to_json(self) -> list[Any]:
        return [
            self.split,
            self.position,
            self.family,
            self.seed,
            self.attempt,
            self.scenario_hash,
            self.graph_hash,
        ]


@dataclass(frozen=True, slots=True)
class SplitManifest:
    params: FloorplanParams
    quotas: Mapping[str, Mapping[str, int]]
    entries: tuple[SplitEntry, ...]
    rejected: Mapping[str, int]
    assignment_seed: int = ASSIGNMENT_SEED
    iou_threshold: float = IOU_THRESHOLD

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": SPLIT_SCHEMA,
            "params": self.params.to_json(),
            "quotas": {s: dict(q) for s, q in self.quotas.items()},
            "seed_base": SEED_BASE,
            "dev_seed_limit": DEV_SEED_LIMIT,
            "assignment_seed": self.assignment_seed,
            "iou_threshold": self.iou_threshold,
            "rejected_near_duplicates": dict(self.rejected),
            "fields": [
                "split",
                "position",
                "family",
                "seed",
                "attempt",
                "scenario_hash",
                "graph_hash",
            ],
            "entries": [e.to_json() for e in self.entries],
        }

    @property
    def sha256(self) -> str:
        return hashlib.sha256(canonical_json(self.to_json()).encode("utf-8")).hexdigest()

    def split(self, name: str) -> tuple[SplitEntry, ...]:
        return tuple(sorted((e for e in self.entries if e.split == name), key=lambda e: e.position))


class DuplicateIndex:
    """Accepted buildings with cached graph hashes and D4 masks."""

    def __init__(self, iou_threshold: float = IOU_THRESHOLD) -> None:
        self._threshold = iou_threshold
        self._graphs: dict[str, list[FloorGraph]] = {}
        self._masks: dict[tuple[int, ...], list[Mask]] = {}
        self._stacks: dict[tuple[int, ...], Any] = {}

    def __len__(self) -> int:
        return sum(len(v) for v in self._graphs.values())

    def conflicts(self, plan: Floorplan) -> bool:
        h = graph_hash(plan.graph)
        if any(isomorphic(plan.graph, other) for other in self._graphs.get(h, [])):
            return True
        mask = free_mask(plan)
        for k in range(4):
            rotated = np.rot90(mask, k)
            for image in (rotated, np.fliplr(rotated)):
                stack = self._stack(image.shape)
                if stack is None:
                    continue
                inter = np.logical_and(stack, image).sum(axis=(1, 2))
                union = np.logical_or(stack, image).sum(axis=(1, 2))
                if np.any(inter >= self._threshold * union):
                    return True
        return False

    def add(self, plan: Floorplan) -> None:
        self._graphs.setdefault(graph_hash(plan.graph), []).append(plan.graph)
        mask = free_mask(plan)
        self._masks.setdefault(mask.shape, []).append(mask)
        self._stacks.pop(mask.shape, None)

    def _stack(self, shape: tuple[int, ...]) -> Any:
        if shape not in self._masks:
            return None
        if shape not in self._stacks:
            self._stacks[shape] = np.stack(self._masks[shape])
        return self._stacks[shape]


def _family_quotas(quotas: Mapping[str, Mapping[str, int]]) -> dict[str, int]:
    if set(quotas) - set(SPLIT_ORDER):
        raise SplitError(f"unknown splits {sorted(set(quotas) - set(SPLIT_ORDER))}")
    need: dict[str, int] = {}
    for split in quotas.values():
        for family, n in split.items():
            if family not in FAMILIES or n < 0:
                raise SplitError(f"invalid quota {family}: {n}")
            need[family] = need.get(family, 0) + n
    return need


def build_manifest(
    params: FloorplanParams | None = None,
    quotas: Mapping[str, Mapping[str, int]] = QUOTAS_V1,
    assignment_seed: int = ASSIGNMENT_SEED,
    iou_threshold: float = IOU_THRESHOLD,
) -> SplitManifest:
    p = params if params is not None else FloorplanParams()
    need = _family_quotas(quotas)
    index = DuplicateIndex(iou_threshold)
    accepted: dict[str, list[Floorplan]] = {f: [] for f in need}
    rejected = dict.fromkeys(need, 0)
    counter = dict.fromkeys(need, 0)
    while any(len(accepted[f]) < need[f] for f in need):
        for family in (f for f in FAMILIES if f in need):
            if len(accepted[family]) >= need[family]:
                continue
            plan = generate_floorplan(family, SEED_BASE[family] + counter[family], p)
            counter[family] += 1
            if index.conflicts(plan):
                rejected[family] += 1
            else:
                index.add(plan)
                accepted[family].append(plan)
    rng = KeyedRandom(assignment_seed)
    pools = {
        f: sorted(plans, key=lambda pl: (rng.uniform("split.assign", f, pl.seed), pl.seed))
        for f, plans in accepted.items()
    }
    taken = dict.fromkeys(need, 0)
    entries: list[SplitEntry] = []
    for split in (s for s in SPLIT_ORDER if s in quotas):
        per_family = {}
        for family, n in quotas[split].items():
            per_family[family] = pools[family][taken[family] : taken[family] + n]
            taken[family] += n
        order = [f for f in FAMILIES if f in per_family]
        position = 0
        for i in range(max((len(v) for v in per_family.values()), default=0)):
            for family in order:
                if i < len(per_family[family]):
                    pl = per_family[family][i]
                    entries.append(
                        SplitEntry(
                            split,
                            position,
                            family,
                            pl.seed,
                            pl.attempt,
                            pl.spec.scenario_hash,
                            graph_hash(pl.graph),
                        )
                    )
                    position += 1
    return SplitManifest(p, quotas, tuple(entries), rejected, assignment_seed, iou_threshold)


def regenerate(entry: SplitEntry, params: FloorplanParams) -> Floorplan:
    """The building of a manifest entry; fails if it no longer matches the hashes."""
    plan = generate_floorplan(entry.family, entry.seed, params)
    if (plan.attempt, plan.spec.scenario_hash, graph_hash(plan.graph)) != (
        entry.attempt,
        entry.scenario_hash,
        entry.graph_hash,
    ):
        raise SplitError(f"{entry.building_id} does not regenerate to its manifest hashes")
    return plan
