"""Parametric floorplan generator with topology families (ARGUS-E07-T07).

A building is a recursive rectangle split (BSP) of the bounding box on a 0.5 m
module grid. Leaves are rooms; two rooms are adjacent when their shared wall is
long enough for a door with margins. A keyed random spanning tree of the
adjacency graph plus k extra edges gives the passages, so the cycle rank of the
room graph is k. Families are defined by the structure of that graph, not by the
seed, and `classify` recomputes the family from the graph alone:

* tree: cycle rank 0, no corridor rooms.
* loop: cycle rank 1, no corridor rooms.
* double_loop: cycle rank 2 whose 2-core is a theta graph (two cycles sharing at
  least one edge), no corridor rooms.
* dogleg: cycle rank 0 with a bent corridor system: a horizontal corridor and a
  vertical corridor meeting at a right angle; rooms hang off the corridors.

A corridor room is a room narrower than min_room_m; BSP rooms never are, so the
corridor flag cannot appear by accident. Interior walls are 0.2 m thick and
centred on the split lines, outer walls 0.1 m (as in the runner fixtures);
rooms are the leaves inset by 0.1 m, and every passage spans its wall exactly.

Duplicate detection for splits: `graph_hash` (Weisfeiler-Lehman refinement over
size class, corridor flag and degree) with an exact backtracking check in
`isomorphic`, and `raster_iou` over the eight D4 symmetries of the free-cell
mask. `is_near_duplicate` is true if either test fires.

The scale parameters (bounding box, room count, door position, corridor width,
team size, robot radius) are general; ARGUS freezes only the pilot subset. Other
tracks may use larger values without changing this module's semantics.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from argus.core.events import KeyedRandom, canonical_json
from argus.world.geometry import Geometry, Passage, Rect, Room
from argus.world.scenario import CtmcPassage, RobotStart, ScenarioSpec

FLOORPLAN_VERSION = "argus-floorplan-v1"
FAMILIES = ("tree", "loop", "double_loop", "dogleg")
EXTRA_EDGES = {"tree": 0, "dogleg": 0, "loop": 1, "double_loop": 2}
DOOR_POSITIONS = ("random", "center", "corner")
UNIT_M = 0.5  # split lines lie on this module grid
WALL_HALF_M = 0.1
SNAP_M = 0.25  # door starts snap to the map raster

Box = tuple[int, int, int, int]  # x0, y0, x1, y1 in UNIT_M
Mask = npt.NDArray[np.bool_]


class FloorplanError(ValueError):
    """Invalid parameters or no valid building within the attempt budget."""


@dataclass(frozen=True, slots=True)
class FloorplanParams:
    width_m: float = 16.0
    height_m: float = 16.0
    rooms_min: int = 6
    rooms_max: int = 10
    min_room_m: float = 3.0
    door_width_m: float = 1.2
    door_position: str = "random"
    door_margin_m: float = 0.35
    corridor_width_m: float = 2.5
    n_robots: int = 4
    robot_radius_m: float = 0.2
    clearance_m: float = 0.2
    start_spacing_m: float = 1.5  # closer starts let frontier agents deadlock
    tick_seconds: float = 0.1
    horizon_ticks: int = 3000
    v_max: float = 0.5
    omega_max: float = 1.0
    passage_rates: tuple[float, float] = (0.01, 0.02)  # (0, 0): static passages
    static_initial_open: float = 1 / 3
    max_attempts: int = 200

    def __post_init__(self) -> None:
        for name in ("width_m", "height_m", "min_room_m", "corridor_width_m"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0 or not _on_grid(value):
                raise FloorplanError(f"{name} must be a positive multiple of {UNIT_M} m")
        if not 2 <= self.rooms_min <= self.rooms_max:
            raise FloorplanError("need 2 <= rooms_min <= rooms_max")
        if self.door_position not in DOOR_POSITIONS:
            raise FloorplanError(f"door_position must be one of {DOOR_POSITIONS}")
        passable = 2 * self.robot_radius_m + 2 * self.clearance_m
        if self.door_width_m < passable or self.corridor_width_m < passable:
            raise FloorplanError("doors and corridors must clear the largest robot")
        if self.corridor_width_m >= self.min_room_m:
            raise FloorplanError("corridor_width_m must be below min_room_m")
        if self.min_room_m < self.door_width_m + 2 * self.door_margin_m:
            raise FloorplanError("min_room_m must fit a door with margins")
        if self.door_margin_m < WALL_HALF_M:
            raise FloorplanError("door_margin_m must clear the perpendicular walls")
        if self.start_spacing_m < 2 * self.robot_radius_m + self.clearance_m:
            raise FloorplanError("start_spacing_m must keep robots apart")
        if self.n_robots < 1 or self.max_attempts < 1:
            raise FloorplanError("n_robots and max_attempts must be >= 1")
        if any(not math.isfinite(r) or r < 0 for r in self.passage_rates):
            raise FloorplanError("passage rates must be finite and >= 0")

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["passage_rates"] = list(self.passage_rates)
        return data


@dataclass(frozen=True, slots=True)
class FloorGraph:
    """Room graph: nodes (room_id, size_class, corridor) and undirected edges."""

    nodes: tuple[tuple[str, str, bool], ...]
    edges: tuple[tuple[str, str], ...]

    def adjacency(self) -> dict[str, set[str]]:
        adj: dict[str, set[str]] = {n[0]: set() for n in self.nodes}
        for a, b in self.edges:
            adj[a].add(b)
            adj[b].add(a)
        return adj

    def labels(self) -> dict[str, str]:
        return {rid: f"{size}|{int(corridor)}" for rid, size, corridor in self.nodes}


@dataclass(frozen=True, slots=True)
class Floorplan:
    family: str
    seed: int
    attempt: int
    params: FloorplanParams
    spec: ScenarioSpec
    graph: FloorGraph

    def to_json(self) -> dict[str, Any]:
        return {
            "version": FLOORPLAN_VERSION,
            "family": self.family,
            "seed": self.seed,
            "attempt": self.attempt,
            "params": self.params.to_json(),
            "scenario_hash": self.spec.scenario_hash,
            "graph_hash": graph_hash(self.graph),
        }


def _on_grid(value: float) -> bool:
    return math.isclose(value / UNIT_M, round(value / UNIT_M), abs_tol=1e-9)


def _units(value: float) -> int:
    return round(value / UNIT_M)


def size_class(area_m2: float) -> str:
    return "S" if area_m2 < 9.0 else "M" if area_m2 < 20.0 else "L"


# --- graph structure --------------------------------------------------------


def _connected(nodes: set[str], edges: list[tuple[str, str]]) -> bool:
    if not nodes:
        return True
    adj: dict[str, set[str]] = {n: set() for n in nodes}
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    start = min(nodes)
    seen, stack = {start}, [start]
    while stack:
        for nxt in adj[stack.pop()]:
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return seen == nodes


def _two_core(graph: FloorGraph) -> tuple[set[str], list[tuple[str, str]]]:
    adj = graph.adjacency()
    alive = set(adj)
    changed = True
    while changed:
        changed = False
        for node in sorted(alive):
            if len(adj[node] & alive) <= 1:
                alive.remove(node)
                changed = True
    return alive, [(a, b) for a, b in graph.edges if a in alive and b in alive]


def classify(graph: FloorGraph) -> str | None:
    """Family from the graph alone; None if the graph belongs to no family."""
    nodes = {n[0] for n in graph.nodes}
    edges = list(graph.edges)
    if not _connected(nodes, edges):
        return None
    rank = len(edges) - len(nodes) + 1
    corridors = {n[0] for n in graph.nodes if n[2]}
    if corridors:
        joined = any(a in corridors and b in corridors for a, b in edges)
        return "dogleg" if rank == 0 and len(corridors) >= 2 and joined else None
    if rank == 0:
        return "tree"
    if rank == 1:
        return "loop"
    if rank == 2:
        core, core_edges = _two_core(graph)
        degree = {n: sum(n in e for e in core_edges) for n in core}
        if sorted(degree.values()).count(3) != 2 or max(degree.values()) != 3:
            return None  # figure-eight (degree 4) or other shapes
        bridgeless = all(
            _connected(core, [e for e in core_edges if e != edge]) for edge in core_edges
        )
        return "double_loop" if bridgeless else None  # a dumbbell has bridges
    return None


def _wl_colours(graph: FloorGraph) -> dict[str, str]:
    adj = graph.adjacency()
    colour = graph.labels()
    for _ in range(len(adj)):
        colour = {
            n: hashlib.sha256(
                canonical_json([colour[n], sorted(colour[m] for m in adj[n])]).encode("utf-8")
            ).hexdigest()[:16]
            for n in adj
        }
    return colour


def graph_hash(graph: FloorGraph) -> str:
    """Isomorphism-invariant hash; equal hashes still need `isomorphic`."""
    colours = sorted(_wl_colours(graph).values())
    payload = [FLOORPLAN_VERSION, colours, len(graph.edges)]
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def isomorphic(a: FloorGraph, b: FloorGraph) -> bool:
    """Exact label-preserving isomorphism by backtracking within WL colour classes."""
    if len(a.nodes) != len(b.nodes) or len(a.edges) != len(b.edges):
        return False
    if graph_hash(a) != graph_hash(b):
        return False
    ca, cb = _wl_colours(a), _wl_colours(b)
    adj_a, adj_b = a.adjacency(), b.adjacency()
    order = sorted(ca, key=lambda n: (sum(c == ca[n] for c in ca.values()), n))
    mapping: dict[str, str] = {}
    used: set[str] = set()

    def extend(i: int) -> bool:
        if i == len(order):
            return True
        node = order[i]
        for cand in sorted(n for n in cb if cb[n] == ca[node] and n not in used):
            if all((m in adj_a[node]) == (mapping[m] in adj_b[cand]) for m in mapping):
                mapping[node] = cand
                used.add(cand)
                if extend(i + 1):
                    return True
                del mapping[node]
                used.remove(cand)
        return False

    return extend(0)


# --- raster signature -------------------------------------------------------


def free_mask(plan: Floorplan, resolution_m: float = 0.25) -> Mask:
    """Cells whose square does not overlap a wall interior (all passages open)."""
    bounds = plan.spec.geometry.bounds
    nx = round((bounds.xmax - bounds.xmin) / resolution_m)
    ny = round((bounds.ymax - bounds.ymin) / resolution_m)
    free = np.ones((ny, nx), dtype=np.bool_)
    for wall in plan.spec.geometry.walls:
        i0 = max(0, math.floor((wall.xmin - bounds.xmin) / resolution_m + 1e-9))
        i1 = min(nx, math.ceil((wall.xmax - bounds.xmin) / resolution_m - 1e-9))
        j0 = max(0, math.floor((wall.ymin - bounds.ymin) / resolution_m + 1e-9))
        j1 = min(ny, math.ceil((wall.ymax - bounds.ymin) / resolution_m - 1e-9))
        free[j0:j1, i0:i1] = False
    return free


def _d4(mask: Mask) -> list[Mask]:
    out = []
    for k in range(4):
        rotated = np.rot90(mask, k)
        out.extend([rotated, np.fliplr(rotated)])
    return out


def raster_iou(a: Floorplan, b: Floorplan) -> float:
    """Largest free-cell IoU of a against the eight D4 images of b."""
    ma = free_mask(a)
    best = 0.0
    for mb in _d4(free_mask(b)):
        if mb.shape != ma.shape:
            continue
        union = np.logical_or(ma, mb).sum()
        if union:
            best = max(best, float(np.logical_and(ma, mb).sum() / union))
    return best


def is_near_duplicate(a: Floorplan, b: Floorplan, iou_threshold: float = 0.90) -> bool:
    return isomorphic(a.graph, b.graph) or raster_iou(a, b) >= iou_threshold


# --- generation -------------------------------------------------------------


def _split(
    boxes: list[Box], fixed: set[Box], target: int, min_u: int, u: Callable[..., float]
) -> list[Box]:
    boxes = list(boxes)
    while sum(b not in fixed for b in boxes) < target:
        options = [
            b
            for b in boxes
            if b not in fixed and (b[2] - b[0] >= 2 * min_u or b[3] - b[1] >= 2 * min_u)
        ]
        if not options:
            break
        box = max(options, key=lambda b: ((b[2] - b[0]) * (b[3] - b[1]), b))
        x0, y0, x1, y1 = box
        w, h = x1 - x0, y1 - y0
        can_x, can_y = w >= 2 * min_u, h >= 2 * min_u
        if can_x and can_y:
            vertical = w > h or (w == h and u("axis", *box) < 0.5)
        else:
            vertical = can_x
        side = w if vertical else h
        cut = min_u + math.floor(u("cut", *box) * (side - 2 * min_u + 1))
        boxes.remove(box)
        if vertical:
            boxes += [(x0, y0, x0 + cut, y1), (x0 + cut, y0, x1, y1)]
        else:
            boxes += [(x0, y0, x1, y0 + cut), (x0, y0 + cut, x1, y1)]
    return boxes


def _shared(a: Box, b: Box) -> tuple[str, int, int, int] | None:
    """Shared boundary (axis, line, lo, hi) with positive length, in units."""
    for axis, (a0, a1, b0, b1, alo, ahi, blo, bhi) in (
        ("x", (a[0], a[2], b[0], b[2], a[1], a[3], b[1], b[3])),
        ("y", (a[1], a[3], b[1], b[3], a[0], a[2], b[0], b[2])),
    ):
        line = a1 if a1 == b0 else a0 if a0 == b1 else None
        if line is not None:
            lo, hi = max(alo, blo), min(ahi, bhi)
            if hi > lo:
                return axis, line, lo, hi
    return None


def _door_range(lo_m: float, hi_m: float, p: FloorplanParams) -> tuple[float, float] | None:
    """Snapped first and last door start on a wall segment, or None if no door fits."""
    first = math.ceil((lo_m + p.door_margin_m) / SNAP_M - 1e-9) * SNAP_M
    last = math.floor((hi_m - p.door_margin_m - p.door_width_m) / SNAP_M + 1e-9) * SNAP_M
    return (first, last) if first <= last + 1e-9 else None


def _door_start(first: float, last: float, p: FloorplanParams, key: float, side: float) -> float:
    steps = round((last - first) / SNAP_M)
    if p.door_position == "center":
        return first + (steps // 2) * SNAP_M
    if p.door_position == "corner":
        return first if side < 0.5 else last
    return first + math.floor(key * (steps + 1)) * SNAP_M


def _attempt(family: str, seed: int, attempt: int, p: FloorplanParams) -> Floorplan | None:
    rng = KeyedRandom(seed)

    def u(*keys: str | int) -> float:
        return rng.uniform("floorplan", FLOORPLAN_VERSION, family, attempt, *keys)

    W, H = _units(p.width_m), _units(p.height_m)
    min_u, cw = _units(p.min_room_m), _units(p.corridor_width_m)
    boxes: list[Box] = [(0, 0, W, H)]
    fixed: set[Box] = set()
    if family == "dogleg":
        if H - cw < 2 * min_u or W - cw < 2 * min_u:
            return None
        a = min_u + math.floor(u("corridor", "y") * (H - cw - 2 * min_u + 1))
        b = min_u + math.floor(u("corridor", "x") * (W - cw - 2 * min_u + 1))
        hc, vc = (0, a, W, a + cw), (b, a + cw, b + cw, H)
        boxes = [(0, 0, W, a), hc, (0, a + cw, b, H), vc, (b + cw, a + cw, W, H)]
        fixed = {hc, vc}
    target = p.rooms_min + math.floor(u("rooms") * (p.rooms_max - p.rooms_min + 1))
    boxes = _split(boxes, fixed, target, min_u, u)
    if sum(b not in fixed for b in boxes) < p.rooms_min:
        return None
    boxes.sort(key=lambda b: (b[1], b[0]))
    ids = [f"R{i:02d}" for i in range(len(boxes))]
    corridor = {ids[i] for i, b in enumerate(boxes) if b in fixed}

    shared: dict[tuple[int, int], tuple[str, int, int, int]] = {}
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            s = _shared(boxes[i], boxes[j])
            if s is not None:
                shared[(i, j)] = s
    doorable = {k: s for k, s in shared.items() if _door_range(s[2] * UNIT_M, s[3] * UNIT_M, p)}

    def weight(k: tuple[int, int]) -> tuple[int, float, tuple[int, int]]:
        touches_corridor = ids[k[0]] in corridor or ids[k[1]] in corridor
        return (0 if touches_corridor else 1, u("edge", *k), k)

    parent = list(range(len(boxes)))

    def root(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    chosen: list[tuple[int, int]] = []
    rest: list[tuple[int, int]] = []
    for k in sorted(doorable, key=weight):
        ra, rb = root(k[0]), root(k[1])
        if ra != rb:
            parent[ra] = rb
            chosen.append(k)
        else:
            rest.append(k)
    if len(chosen) != len(boxes) - 1:
        return None
    extra = EXTRA_EDGES[family]
    if family == "dogleg":
        rest = []  # corridor buildings stay trees
    if len(rest) < extra:
        return None
    chosen += sorted(rest, key=lambda k: u("extra", *k))[:extra]

    m = UNIT_M
    bounds = Rect(0.0, 0.0, W * m, H * m)
    t = WALL_HALF_M
    walls: list[Rect] = [
        Rect(0.0, 0.0, W * m, t),
        Rect(0.0, H * m - t, W * m, H * m),
        Rect(0.0, 0.0, t, H * m),
        Rect(W * m - t, 0.0, W * m, H * m),
    ]
    passages: list[Passage] = []
    for n, k in enumerate(sorted(chosen)):
        axis, line, lo, hi = shared[k]
        door_range = _door_range(lo * m, hi * m, p)
        assert door_range is not None
        start = _door_start(*door_range, p, u("door", *k), u("door-side", *k))
        end = start + p.door_width_m
        c = line * m
        rect = Rect(c - t, start, c + t, end) if axis == "x" else Rect(start, c - t, end, c + t)
        passages.append(Passage(f"D{n:02d}", rect, (ids[k[0]], ids[k[1]])))
    doors = {k: passages[n].rect for n, k in enumerate(sorted(chosen))}
    for k, (axis, line, lo, hi) in sorted(shared.items()):
        c = line * m
        pieces: list[tuple[float, float]] = []
        a0, a1 = max(0.0, lo * m - t), min((W if axis == "y" else H) * m, hi * m + t)
        if k in doors:
            d = doors[k]
            d0, d1 = (d.ymin, d.ymax) if axis == "x" else (d.xmin, d.xmax)
            pieces = [(a0, d0), (d1, a1)]
        else:
            pieces = [(a0, a1)]
        for s0, s1 in pieces:
            if s1 > s0:
                piece = Rect(c - t, s0, c + t, s1) if axis == "x" else Rect(s0, c - t, s1, c + t)
                walls.append(piece)
    rooms = tuple(
        Room(ids[i], Rect(b[0] * m + t, b[1] * m + t, b[2] * m - t, b[3] * m - t))
        for i, b in enumerate(boxes)
    )
    geometry = Geometry(bounds, tuple(walls), rooms, tuple(passages))

    def area(r: Room) -> float:
        return (r.rect.xmax - r.rect.xmin) * (r.rect.ymax - r.rect.ymin)

    home = max((r for r in rooms if r.room_id not in corridor), key=lambda r: (area(r), r.room_id))
    starts = _starts(home.rect, p)
    if starts is None:
        return None
    rate_01, rate_10 = p.passage_rates
    model = (
        CtmcPassage(0.0, 0.0, initial_open=p.static_initial_open)
        if rate_01 == rate_10 == 0.0
        else CtmcPassage(rate_01, rate_10)
    )
    spec = ScenarioSpec(
        f"{family}-{seed}",
        geometry,
        p.tick_seconds,
        p.horizon_ticks,
        p.robot_radius_m,
        p.v_max,
        p.omega_max,
        starts,
        tuple((ps.passage_id, model) for ps in passages),
    )
    graph = FloorGraph(
        tuple((r.room_id, size_class(area(r)), r.room_id in corridor) for r in rooms),
        tuple((ps.rooms[0], ps.rooms[1]) for ps in passages),
    )
    return Floorplan(family, seed, attempt, p, spec, graph)


def _starts(room: Rect, p: FloorplanParams) -> tuple[RobotStart, ...] | None:
    inset = p.robot_radius_m + p.clearance_m + WALL_HALF_M
    step = p.start_spacing_m
    cols = math.floor((room.xmax - room.xmin - 2 * inset) / step) + 1
    rows = math.floor((room.ymax - room.ymin - 2 * inset) / step) + 1
    if cols < 1 or rows < 1 or cols * rows < p.n_robots:
        return None
    return tuple(
        RobotStart(
            f"r{n}",
            (room.xmin + inset + (n % cols) * step, room.ymin + inset + (n // cols) * step, 0.0),
        )
        for n in range(p.n_robots)
    )


def generate_floorplan(family: str, seed: int, params: FloorplanParams | None = None) -> Floorplan:
    """First valid attempt for (family, seed); deterministic, independent of call order."""
    if family not in FAMILIES:
        raise FloorplanError(f"unknown family {family!r}")
    p = params if params is not None else FloorplanParams()
    for attempt in range(p.max_attempts):
        plan = _attempt(family, seed, attempt, p)
        if plan is not None and classify(plan.graph) == family:
            return plan
    raise FloorplanError(f"no valid {family} building for seed {seed} in {p.max_attempts} tries")
