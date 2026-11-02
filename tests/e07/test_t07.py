"""ARGUS-E07-T07: building generator, topology families and duplicate detection."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from argus.experiments import runner as runner_module
from argus.experiments.runner import EpisodeConfig, ExperimentRunner
from argus.world.floorplan import (
    FAMILIES,
    FloorGraph,
    Floorplan,
    FloorplanError,
    FloorplanParams,
    classify,
    free_mask,
    generate_floorplan,
    graph_hash,
    is_near_duplicate,
    isomorphic,
    raster_iou,
)
from argus.world.geometry import Geometry, Passage, Rect, Room
from argus.world.scenario import CtmcPassage, RobotStart, ScenarioSpec

SEEDS = range(12)


def plans(family: str) -> list[Floorplan]:
    return [generate_floorplan(family, s) for s in SEEDS]


# --- determinism ------------------------------------------------------------


def test_same_seed_same_building_independent_of_call_order() -> None:
    first = generate_floorplan("loop", 7)
    generate_floorplan("dogleg", 3)  # unrelated draws in between
    again = generate_floorplan("loop", 7)
    assert first.spec.scenario_hash == again.spec.scenario_hash
    assert first.to_json() == again.to_json()
    assert generate_floorplan("loop", 8).spec.scenario_hash != first.spec.scenario_hash


# --- families -----------------------------------------------------------------


@pytest.mark.parametrize("family", FAMILIES)
def test_family_is_recomputed_from_the_graph(family: str) -> None:
    for plan in plans(family):
        assert plan.family == family and classify(plan.graph) == family
        rank = len(plan.graph.edges) - len(plan.graph.nodes) + 1
        assert rank == {"tree": 0, "dogleg": 0, "loop": 1, "double_loop": 2}[family]
        corridors = [n for n in plan.graph.nodes if n[2]]
        assert (len(corridors) == 2) == (family == "dogleg")


def g(edges: list[tuple[str, str]], corridor: set[str] | None = None) -> FloorGraph:
    corridor = corridor or set()
    nodes = sorted({n for e in edges for n in e})
    return FloorGraph(tuple((n, "M", n in corridor) for n in nodes), tuple(edges))


def test_classify_separates_theta_dumbbell_and_figure_eight() -> None:
    theta = g([("a", "b"), ("b", "c"), ("c", "d"), ("d", "a"), ("a", "c")])
    dumbbell = g(
        [("a", "b"), ("b", "c"), ("c", "a"), ("c", "d"), ("d", "e"), ("e", "f"), ("f", "d")]
    )
    eight = g([("a", "b"), ("b", "c"), ("c", "a"), ("c", "d"), ("d", "e"), ("e", "c")])
    assert classify(theta) == "double_loop"
    assert classify(dumbbell) is None and classify(eight) is None
    assert classify(g([("a", "b"), ("b", "c")])) == "tree"
    assert classify(g([("a", "b"), ("b", "c"), ("c", "a")])) == "loop"
    assert classify(g([("a", "b"), ("b", "c")], {"a", "b"})) == "dogleg"
    assert classify(g([("a", "b"), ("b", "c")], {"a", "c"})) is None  # corridors not joined
    assert classify(g([("a", "b"), ("c", "d")])) is None  # disconnected


# --- isomorphism and near duplicates -------------------------------------------


def test_relabelled_graph_is_isomorphic_and_labels_matter() -> None:
    plan = generate_floorplan("double_loop", 2)
    rename = {n[0]: f"X{i}" for i, n in enumerate(reversed(plan.graph.nodes))}
    relabelled = FloorGraph(
        tuple((rename[r], s, c) for r, s, c in reversed(plan.graph.nodes)),
        tuple((rename[b], rename[a]) for a, b in reversed(plan.graph.edges)),
    )
    assert graph_hash(relabelled) == graph_hash(plan.graph)
    assert isomorphic(plan.graph, relabelled)
    changed = FloorGraph(
        tuple((r, "S" if s != "S" else "L", c) for r, s, c in plan.graph.nodes),
        plan.graph.edges,
    )
    assert not isomorphic(plan.graph, changed)


def test_isomorphism_is_exact_where_the_hash_is_not() -> None:
    # Two 2-regular graphs with equal WL colours: one hexagon versus two triangles.
    hexagon = g([("a", "b"), ("b", "c"), ("c", "d"), ("d", "e"), ("e", "f"), ("f", "a")])
    triangles = g([("a", "b"), ("b", "c"), ("c", "a"), ("d", "e"), ("e", "f"), ("f", "d")])
    assert graph_hash(hexagon) == graph_hash(triangles)
    assert not isomorphic(hexagon, triangles)


def _transform(plan: Floorplan, kind: str) -> Floorplan:
    """Mirror (x -> W - x) or rotate by 90 degrees; room and passage ids renumbered."""
    b = plan.spec.geometry.bounds
    w, h = b.xmax, b.ymax

    def pt(x: float, y: float) -> tuple[float, float]:
        return (w - x, y) if kind == "mirror" else (h - y, x)

    def rect(r: Rect) -> Rect:
        (x0, y0), (x1, y1) = pt(r.xmin, r.ymin), pt(r.xmax, r.ymax)
        return Rect(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))

    geo = plan.spec.geometry
    rid = {r.room_id: f"Q{i}" for i, r in enumerate(reversed(geo.rooms))}
    pid = {p.passage_id: f"P{i}" for i, p in enumerate(reversed(geo.passages))}
    bounds = Rect(0.0, 0.0, w, h) if kind == "mirror" else Rect(0.0, 0.0, h, w)
    geometry = Geometry(
        bounds,
        tuple(rect(r) for r in geo.walls),
        tuple(Room(rid[r.room_id], rect(r.rect)) for r in geo.rooms),
        tuple(
            Passage(pid[p.passage_id], rect(p.rect), (rid[p.rooms[1]], rid[p.rooms[0]]))
            for p in geo.passages
        ),
    )
    starts = tuple(RobotStart(s.robot_id, (*pt(*s.pose[:2]), 0.0)) for s in plan.spec.starts)
    spec = dataclasses.replace(
        plan.spec,
        scenario_id="copy",
        geometry=geometry,
        starts=starts,
        passages=tuple((pid[p], m) for p, m in plan.spec.passages),
    )
    graph = FloorGraph(
        tuple((rid[r], s, c) for r, s, c in plan.graph.nodes),
        tuple((rid[a], rid[b]) for a, b in plan.graph.edges),
    )
    return dataclasses.replace(plan, spec=spec, graph=graph)


@pytest.mark.parametrize("kind", ["mirror", "rotate"])
def test_mirrored_and_rotated_copies_are_rejected_as_duplicates(kind: str) -> None:
    plan = generate_floorplan("loop", 4)
    copy = _transform(plan, kind)
    assert copy.spec.scenario_hash != plan.spec.scenario_hash  # a naive hash misses it
    assert raster_iou(plan, copy) == pytest.approx(1.0)
    assert isomorphic(plan.graph, copy.graph)
    assert is_near_duplicate(plan, copy)


def test_raster_branch_detects_copies_with_a_changed_graph() -> None:
    plan = generate_floorplan("tree", 1)
    relabel = FloorGraph(
        tuple((r, "S" if s != "S" else "L", c) for r, s, c in plan.graph.nodes),
        plan.graph.edges,
    )
    copy = dataclasses.replace(plan, graph=relabel)
    assert not isomorphic(plan.graph, copy.graph)
    assert is_near_duplicate(plan, copy)  # caught by the raster comparison alone


def test_distinct_buildings_are_not_duplicates() -> None:
    a, b = generate_floorplan("loop", 0), generate_floorplan("double_loop", 0)
    assert not is_near_duplicate(a, b)
    assert raster_iou(a, b) < 0.90


# --- validity -------------------------------------------------------------------


@pytest.mark.parametrize("family", FAMILIES)
def test_generated_buildings_are_valid_and_passable(family: str) -> None:
    for plan in plans(family):
        spec, p = plan.spec, plan.params
        nodes = {n[0] for n in plan.graph.nodes}
        assert nodes == {r.room_id for r in spec.geometry.rooms}
        assert len(spec.starts) == p.n_robots
        home = [r for r in spec.geometry.rooms if r.rect.distance_to(spec.starts[0].pose[:2]) == 0]
        assert home and not dict((n[0], n[2]) for n in plan.graph.nodes)[home[0].room_id]
        mask = free_mask(plan)
        for passage in spec.geometry.passages:
            r = passage.rect
            width = max(r.xmax - r.xmin, r.ymax - r.ymin)
            assert width >= 2 * p.robot_radius_m + 2 * p.clearance_m
            # every raster cell of the door opening is free (all passages open)
            vertical_wall = r.ymax - r.ymin > r.xmax - r.xmin
            lo, hi = (r.ymin, r.ymax) if vertical_wall else (r.xmin, r.xmax)
            t0, t1 = (r.xmin, r.xmax) if vertical_wall else (r.ymin, r.ymax)
            along = slice(int(lo / 0.25), int(hi / 0.25))
            across = slice(int(t0 / 0.25), int(np.ceil(t1 / 0.25)))
            inner = mask[along, across] if vertical_wall else mask[across, along]
            assert inner.size >= 4 and inner.all()


def test_static_passages_and_parameter_validation() -> None:
    static = generate_floorplan("tree", 0, FloorplanParams(passage_rates=(0.0, 0.0)))
    models = {m for _, m in static.spec.passages}
    assert models == {CtmcPassage(0.0, 0.0, initial_open=1 / 3)}
    for bad in (
        {"door_width_m": 0.5},
        {"corridor_width_m": 3.0},
        {"width_m": 16.2},
        {"rooms_min": 1},
        {"door_position": "middle"},
        {"start_spacing_m": 0.4},
    ):
        with pytest.raises(FloorplanError):
            FloorplanParams(**bad)  # type: ignore[arg-type]
    with pytest.raises(FloorplanError):
        generate_floorplan("spiral", 0)
    with pytest.raises(FloorplanError):
        generate_floorplan("tree", 0, FloorplanParams(rooms_min=40, rooms_max=40, max_attempts=3))


@pytest.mark.parametrize("position", ["center", "corner"])
def test_door_position_parameter(position: str) -> None:
    plan = generate_floorplan("tree", 3, FloorplanParams(door_position=position))
    assert classify(plan.graph) == "tree"


def test_scale_parameters_support_larger_buildings_and_teams() -> None:
    params = FloorplanParams(
        width_m=32.0, height_m=24.0, rooms_min=14, rooms_max=20, n_robots=8, robot_radius_m=0.3
    )
    for family in FAMILIES:
        plan = generate_floorplan(family, 0, params)
        assert len(plan.graph.nodes) >= 14 and len(plan.spec.starts) == 8


# --- runner integration --------------------------------------------------------------


def test_a_robot_leaves_the_start_room_through_generated_doors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worlds = []
    original = runner_module.WorldBackend2D

    class Spy(original):  # type: ignore[misc,valid-type]
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)  # type: ignore[arg-type]
            worlds.append(self)

    monkeypatch.setattr(runner_module, "WorldBackend2D", Spy)
    params = FloorplanParams(
        horizon_ticks=1200, passage_rates=(0.0, 0.0), static_initial_open=1.0, n_robots=1
    )
    plan = generate_floorplan("dogleg", 5, params)
    spec: ScenarioSpec = plan.spec
    result = ExperimentRunner().run(EpisodeConfig("e", "b", spec, "frontier", 1, 2, 3, 4, 5))
    assert result.status == "ok", result.error
    home = next(r for r in spec.geometry.rooms if r.rect.distance_to(spec.starts[0].pose[:2]) == 0)
    x, y, _ = worlds[-1].poses()["r0"]
    assert home.rect.distance_to((x, y)) > 0.5
