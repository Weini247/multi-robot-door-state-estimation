"""ARGUS-E02-T02: SE(2) motion, swept-disc collisions, visibility and room graph."""

from __future__ import annotations

import math

import pytest

from argus.world.geometry import (
    CONTACT_TOL,
    Geometry,
    GeometryError,
    Motion,
    Passage,
    Rect,
    Room,
    SweptPath,
    first_wall_contact,
    integrate_pose,
    resolve_motions,
    visible,
    visible_passages,
    wrap_angle,
)

R = 0.2


# --- SE(2) ---------------------------------------------------------------------


def test_quarter_circle_reaches_1_1_halfpi() -> None:
    x, y, theta = integrate_pose((0.0, 0.0, 0.0), 1.0, 1.0, math.pi / 2)
    assert (x, y, theta) == pytest.approx((1.0, 1.0, math.pi / 2), abs=1e-12)


def test_straight_line_distance() -> None:
    x, y, theta = integrate_pose((1.0, 2.0, math.pi / 3), 1.0, 0.0, 2.0)
    assert math.hypot(x - 1.0, y - 2.0) == pytest.approx(2.0, abs=1e-12)
    assert theta == pytest.approx(math.pi / 3, abs=1e-15)


def test_small_turn_series_is_continuous_with_exact_formula() -> None:
    for omega in (1e-9, 1e-7, 2e-6, 1e-5):
        x, y, _ = integrate_pose((0.0, 0.0, 0.3), 0.5, omega, 3.0)
        a = omega * 3.0 / 2
        exact = 0.5 * 3.0 * math.sin(a) / a
        assert x == pytest.approx(exact * math.cos(0.3 + a), abs=1e-13)
        assert y == pytest.approx(exact * math.sin(0.3 + a), abs=1e-13)


def test_turning_in_place_and_wrap() -> None:
    assert integrate_pose((1.0, 1.0, 3.0), 0.0, 1.0, 1.0) == pytest.approx(
        (1.0, 1.0, 4.0 - 2 * math.pi), abs=1e-15
    )
    for angle in (-7.0, -math.pi, math.pi, 3 * math.pi, 0.0):
        wrapped = wrap_angle(angle)
        assert -math.pi <= wrapped < math.pi
        assert math.cos(wrapped) == pytest.approx(math.cos(angle), abs=1e-12)
    with pytest.raises(ValueError):
        integrate_pose((0.0, 0.0, 0.0), 1.0, 0.0, -1.0)


# --- swept collisions ----------------------------------------------------------------

THIN_WALL = Rect(5.0, -2.0, 5.01, 2.0)


def test_thin_wall_cannot_be_tunnelled() -> None:
    poses, events = resolve_motions([Motion("r0", (0.0, 0.0, 0.0), R, 10.0, 0.0)], 1.0, [THIN_WALL])
    x = poses["r0"][0]
    assert x <= 5.0 - R + 1e-9 and x == pytest.approx(5.0 - R, abs=1e-9)
    assert [e.kind for e in events] == ["wall"]
    assert events[0].fraction == pytest.approx((5.0 - R) / 10.0, abs=1e-9)


def test_turning_path_into_wall_is_detected() -> None:
    wall = Rect(0.8, 0.9, 1.2, 1.5)
    path = SweptPath.build((0.0, 0.0, 0.0), 1.0, 1.0, math.pi / 2)
    hit = first_wall_contact(path, R, [wall])
    assert hit is not None and 0.0 < hit < 1.0
    stopped = path.pose_at(hit)
    assert wall.distance_to(stopped[:2]) >= R - 1e-9


def test_touching_wall_is_allowed_but_penetration_is_not() -> None:
    wall = Rect(1.0, -1.0, 2.0, 1.0)
    start = (1.0 - R, 0.0, math.pi / 2)  # touching, facing along the wall
    poses, events = resolve_motions([Motion("r0", start, R, 1.0, 0.0)], 0.5, [wall])
    assert events == () and poses["r0"][1] == pytest.approx(0.5)
    away = (1.0 - R, 0.0, math.pi)
    assert resolve_motions([Motion("r0", away, R, 1.0, 0.0)], 0.5, [wall])[1] == ()
    into = (1.0 - R, 0.0, 0.0)
    poses, events = resolve_motions([Motion("r0", into, R, 1.0, 0.0)], 0.5, [wall])
    # Moving in is blocked after at most the contact tolerance (1e-9 m at 0.5 m per step).
    assert events[0].fraction <= CONTACT_TOL / 0.5 + 1e-12
    assert poses["r0"] == pytest.approx(into, abs=2 * CONTACT_TOL)


def test_head_on_robots_stop_symmetrically_without_overlap() -> None:
    motions = [
        Motion("a", (0.0, 0.0, 0.0), R, 1.0, 0.0),
        Motion("b", (2.0, 0.0, math.pi), R, 1.0, 0.0),
        Motion("c", (0.0, 5.0, 0.0), R, 1.0, 0.0),
    ]
    poses, events = resolve_motions(motions, 2.0, [])
    gap = math.dist(poses["a"][:2], poses["b"][:2])
    assert gap >= 2 * R - 1e-9 and gap == pytest.approx(2 * R, abs=1e-8)
    assert poses["a"][0] == pytest.approx(2.0 - poses["b"][0], abs=1e-9)
    assert poses["c"][:2] == pytest.approx((2.0, 5.0), abs=1e-12)  # unaffected robot
    assert [(e.kind, e.robot_ids) for e in events] == [("robot", ("a", "b"))]


def test_resolution_does_not_depend_on_ids_or_order() -> None:
    def run(names: tuple[str, str], reverse: bool) -> list[float]:
        motions = [
            Motion(names[0], (0.0, 0.0, 0.0), R, 1.0, 0.0),
            Motion(names[1], (1.0, -1.0, math.pi / 2), R, 1.0, 0.0),
        ]
        poses, _ = resolve_motions(motions[::-1] if reverse else motions, 2.0, [])
        return [*poses[names[0]], *poses[names[1]]]

    reference = run(("a", "b"), False)
    for names, reverse in ((("b", "a"), False), (("a", "b"), True), (("z", "a"), True)):
        assert run(names, reverse) == pytest.approx(reference, abs=1e-12)


def test_robot_runs_into_stopped_robot() -> None:
    wall = Rect(3.0, -1.0, 3.5, 1.0)
    motions = [
        Motion("lead", (2.0, 0.0, 0.0), R, 1.0, 0.0),
        # Lead stops at the wall at t=.8; follow (2.5 m/s) reaches it at t=.96.
        Motion("follow", (0.0, 0.0, 0.0), R, 2.5, 0.0),
    ]
    poses, events = resolve_motions(motions, 1.0, [wall])
    assert poses["lead"][0] == pytest.approx(3.0 - R, abs=1e-9)
    assert math.dist(poses["lead"][:2], poses["follow"][:2]) >= 2 * R - 1e-9
    assert {e.kind for e in events} == {"wall", "robot"}
    robot = next(e for e in events if e.kind == "robot")
    assert robot.fraction == pytest.approx(0.96, abs=1e-8)


# --- visibility ---------------------------------------------------------------------


def test_range_and_fov_boundaries_count_as_visible() -> None:
    observer = (0.0, 0.0, 0.0)
    assert visible(observer, (5.0, 0.0), 5.0, math.radians(120), [])
    assert not visible(observer, (5.0 + 1e-9, 0.0), 5.0, math.radians(120), [])
    edge = (math.cos(math.radians(60)), math.sin(math.radians(60)))
    assert visible(observer, edge, 5.0, math.radians(120) + 1e-12, [])
    outside = (math.cos(math.radians(60.001)), math.sin(math.radians(60.001)))
    assert not visible(observer, outside, 5.0, math.radians(120), [])


def test_walls_occlude_including_grazing_rays() -> None:
    observer = (0.0, 0.0, 0.0)
    assert not visible(observer, (4.0, 0.0), 5.0, math.pi, [Rect(2.0, -0.5, 2.5, 0.5)])
    # The sight line runs exactly along the wall's lower edge: occluded.
    assert not visible(observer, (4.0, 0.0), 5.0, math.pi, [Rect(1.0, 0.0, 3.0, 1.0)])
    assert visible(observer, (4.0, 0.0), 5.0, math.pi, [Rect(1.0, 0.001, 3.0, 1.0)])


def two_rooms() -> Geometry:
    return Geometry(
        bounds=Rect(0.0, 0.0, 10.0, 4.0),
        walls=(Rect(4.9, 0.0, 5.1, 1.4), Rect(4.9, 2.6, 5.1, 4.0)),
        rooms=(Room("west", Rect(0.0, 0.0, 4.9, 4.0)), Room("east", Rect(5.1, 0.0, 10.0, 4.0))),
        passages=(Passage("door", Rect(4.9, 1.4, 5.1, 2.6), ("west", "east")),),
    )


def test_only_visible_passages_are_returned() -> None:
    geometry = two_rooms()
    facing = (2.0, 2.0, 0.0)
    assert visible_passages(geometry, facing, 5.0, math.radians(120), {"door": 1}) == ("door",)
    assert visible_passages(geometry, facing, 5.0, math.radians(120), {"door": 0}) == ("door",)
    away = (2.0, 2.0, math.pi)
    assert visible_passages(geometry, away, 5.0, math.radians(120), {"door": 1}) == ()
    behind_wall = (3.0, 0.5, 0.0)
    target = geometry.passage("door").rect.center
    assert not visible(
        behind_wall, target, 5.0, math.pi, [geometry.walls[0], Rect(3.5, 0.0, 3.6, 1.0)]
    )


def test_closed_passage_blocks_sight_and_motion_open_does_not() -> None:
    geometry = two_rooms()
    observer, target = (2.0, 2.0, 0.0), (8.0, 2.0)
    assert visible(observer, target, 10.0, math.pi, geometry.obstacles({"door": 1}))
    assert not visible(observer, target, 10.0, math.pi, geometry.obstacles({"door": 0}))
    motion = [Motion("r0", (4.0, 2.0, 0.0), R, 2.0, 0.0)]
    assert resolve_motions(motion, 1.0, geometry.obstacles({"door": 1}))[1] == ()
    assert resolve_motions(motion, 1.0, geometry.obstacles({"door": 0}))[1] != ()
    assert geometry.neighbours("west", {"door": 1}) == ("east",)
    assert geometry.neighbours("west", {"door": 0}) == ()


# --- room graph consistency ------------------------------------------------------------


@pytest.mark.parametrize(
    "build",
    [
        lambda: Geometry(
            Rect(0, 0, 10, 4),
            (),
            (Room("a", Rect(0, 0, 4, 4)), Room("b", Rect(6, 0, 10, 4))),
            (Passage("p", Rect(4.5, 1, 5.5, 2), ("a", "b")),),
        ),
        lambda: Geometry(
            Rect(0, 0, 10, 4),
            (),
            (Room("a", Rect(0, 0, 5, 4)),),
            (Passage("p", Rect(4.5, 1, 5.5, 2), ("a", "x")),),
        ),
        lambda: Geometry(Rect(0, 0, 10, 4), (Rect(9, 0, 11, 1),), (), ()),
        lambda: Geometry(
            Rect(0, 0, 10, 4),
            (Rect(4, 0, 6, 4),),
            (Room("a", Rect(0, 0, 5, 4)), Room("b", Rect(5, 0, 10, 4))),
            (Passage("p", Rect(4.5, 1, 5.5, 2), ("a", "b")),),
        ),
        lambda: Geometry(
            Rect(0, 0, 10, 4), (), (Room("a", Rect(0, 0, 5, 4)), Room("a", Rect(5, 0, 10, 4))), ()
        ),
        lambda: Rect(1.0, 0.0, 1.0, 2.0),
    ],
)
def test_inconsistent_geometry_rejected(build: object) -> None:
    with pytest.raises(GeometryError):
        build()  # type: ignore[operator]


def test_missing_passage_state_rejected() -> None:
    with pytest.raises(GeometryError):
        two_rooms().obstacles({})
