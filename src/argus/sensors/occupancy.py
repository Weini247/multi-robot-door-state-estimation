"""Simulator-side free-space scan for the local exploration map (E05-T02).

Rays are cast across the field of view in steps of half a map cell up to the
sensor range. Every cell a ray reaches is observed and classified exactly: it
is blocked if its square intersects a wall or closed passage, free otherwise.
A ray stops at the first blocked cell. Classifying whole cells (not sample
points) keeps walls thinner than a cell from flickering between free and
blocked, and lets a reopened passage be seen free again. The scan uses
registered poses and is noise-free (no SLAM claim, as in M1). Only the grid
indices leave the simulator, never geometry objects.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from argus.core.types import GridIndex
from argus.world.geometry import Geometry, Pose, Rect


def grid_index(x: float, y: float, resolution: float) -> GridIndex:
    return (math.floor(x / resolution), math.floor(y / resolution))


def scan_map(
    geometry: Geometry,
    pose: Pose,
    passage_open: Mapping[str, int],
    range_m: float,
    fov_rad: float,
    resolution: float,
    rays: int = 61,
) -> tuple[frozenset[GridIndex], frozenset[GridIndex]]:
    obstacles = geometry.obstacles(passage_open)
    bounds = geometry.bounds
    free: set[GridIndex] = set()
    blocked: set[GridIndex] = set()
    known: dict[GridIndex, bool] = {}

    def is_blocked(cell: GridIndex) -> bool:
        if cell not in known:
            square = Rect(
                cell[0] * resolution,
                cell[1] * resolution,
                (cell[0] + 1) * resolution,
                (cell[1] + 1) * resolution,
            )
            known[cell] = any(square.overlaps_interior(rect) for rect in obstacles)
        return known[cell]

    step = resolution / 2
    x0, y0, theta = pose
    for k in range(rays):
        angle = theta - fov_rad / 2 + fov_rad * k / max(1, rays - 1)
        dx, dy = math.cos(angle), math.sin(angle)
        distance = 0.0
        while distance <= range_m:
            x, y = x0 + dx * distance, y0 + dy * distance
            if not (bounds.xmin <= x <= bounds.xmax and bounds.ymin <= y <= bounds.ymax):
                break
            cell = grid_index(x, y, resolution)
            if is_blocked(cell):
                blocked.add(cell)
                break
            free.add(cell)
            distance += step
    return frozenset(free), frozenset(blocked)
