"""ARGUS-E05-T02: frontier exploration, crossing without overlap, paid reservations."""

from __future__ import annotations

import dataclasses
import math

import pytest

from argus.communication.messages import ReservationMessage, decode_frame
from argus.core.types import AgentView, BeliefState, Reservation
from argus.policies import controller as controller_module
from argus.policies.controller import DEFAULT_CONTROLLER
from argus.policies.frontier import FrontierPolicy
from argus.world.scenario import RobotStart
from tests.e05.sim_helpers import RES, Loop, boxed, scenario


def room_cells(width: float, height: float) -> set[tuple[int, int]]:
    margin = 0.1 + 0.2  # wall thickness plus robot radius: cells a robot can see and reach
    cells = set()
    for i in range(int(width / RES)):
        for j in range(int(height / RES)):
            x, y = (i + 0.5) * RES, (j + 0.5) * RES
            if margin <= x <= width - margin and margin <= y <= height - margin:
                cells.add((i, j))
    return cells


def test_one_room_is_explored() -> None:
    spec = scenario(boxed(6.0, 4.0), (RobotStart("r0", (1.0, 1.0, 0.0)),), 900)
    agents = Loop(spec).run()
    seen = set(agents["r0"].view(900).free_map)
    target = room_cells(6.0, 4.0)
    coverage = len(seen & target) / len(target)
    assert coverage >= 0.9, coverage
    assert FrontierPolicy().plan(agents["r0"].view(900)) is None or coverage < 1.0


def test_two_robots_cross_without_overlap() -> None:
    # 12 m corridor; each robot already knows the part behind it, so its only
    # frontier lies beyond the other robot and both must drive through the meeting.
    corridor = boxed(12.0, 1.6)
    starts = (RobotStart("a", (1.0, 0.8, 0.0)), RobotStart("b", (11.0, 0.8, math.pi)))
    rows = range(1, 6)
    known_a = {(i, j) for i in range(1, 36) for j in rows}  # x < 9 m
    known_b = {(i, j) for i in range(12, 47) for j in rows}  # x > 3 m
    loop = Loop(
        scenario(corridor, starts, 900), preseed={"a": (known_a, set()), "b": (known_b, set())}
    )
    loop.run()
    assert loop.min_gap >= 2 * 0.2 - 1e-9
    assert loop.min_gap < 1.0  # they really met


def view(reservations: tuple[Reservation, ...]) -> AgentView:
    free = tuple((i, j) for i in range(2, 10) for j in range(2, 4))
    return AgentView(
        "r0",
        0,
        (0.75, 0.75, 0.0),
        BeliefState(0, ()),
        RES,
        free,
        (),
        (),
        reservations,
        0.0,
        (),
        False,
    )


def test_partner_targets_are_known_only_from_delivered_reservations() -> None:
    policy = FrontierPolicy()
    alone = policy.plan(view(()))
    assert alone is not None
    target = alone[-1]
    informed = policy.plan(view((Reservation("r1", target, 100),)))
    assert (
        informed is not None and abs(informed[-1][0] - target[0]) > policy.reservation_radius_cells
    )
    # Without delivery (frames dropped or not sent) the partner target stays unknown.
    loop = Loop(
        scenario(boxed(6.0, 4.0), (RobotStart("a", (1, 1, 0)), RobotStart("b", (5, 3, 3.1))), 200),
        deliver_reservations=False,
    )
    agents = loop.run()
    assert all(a.view(200).reservations == () for a in agents.values())


def test_reservations_are_charged_frames() -> None:
    loop = Loop(
        scenario(boxed(6.0, 4.0), (RobotStart("a", (1, 1, 0)), RobotStart("b", (5, 3, 3.1))), 200)
    )
    loop.run()
    frames = [x for x in loop.channel.attempts if isinstance(x.message, ReservationMessage)]
    assert loop.reservations_sent == len(frames) > 0
    assert sum(x.size for x in frames) == sum(loop.channel.stats.tx_attempt_bytes.values())
    sample = frames[0].message
    assert decode_frame(sample.encode()) == sample and sample.size_bytes == len(sample.encode())


def test_all_methods_share_the_controller_settings() -> None:
    assert FrontierPolicy().controller is DEFAULT_CONTROLLER
    assert dataclasses.is_dataclass(DEFAULT_CONTROLLER)
    with pytest.raises(dataclasses.FrozenInstanceError):
        DEFAULT_CONTROLLER.v_max = 1.0  # type: ignore[misc]
    assert controller_module.DEFAULT_CONTROLLER.v_max == 0.5


def test_controller_turns_first_then_drives() -> None:
    ctrl = DEFAULT_CONTROLLER
    v, omega = ctrl.steer((0.0, 0.0, 0.0), (0.0, 2.0))  # target 90 degrees left
    assert v == 0.0 and omega == ctrl.omega_max
    v, omega = ctrl.steer((0.0, 0.0, math.pi / 2), (0.0, 2.0))
    assert v == ctrl.v_max and omega == pytest.approx(0.0, abs=1e-12)
    assert ctrl.steer((0.0, 0.0, 0.0), (0.05, 0.0)) == (0.0, 0.0)


def test_collision_backoff_is_symmetric_and_key_driven() -> None:
    policy = FrontierPolicy()
    base = dataclasses.replace(view(()), last_collision=True)
    for key in ("a/10", "b/10", "x/20"):
        action = policy.act(base, key)
        assert action.v == 0.0 and abs(action.omega) == DEFAULT_CONTROLLER.omega_max
        renamed = dataclasses.replace(base, robot_id="zz")
        assert policy.act(renamed, key) == action  # the robot id plays no role


def test_closed_known_space_has_no_frontier() -> None:
    blocked = tuple((i, j) for i in range(0, 6) for j in (0, 5)) + tuple(
        (i, j) for i in (0, 5) for j in range(0, 6)
    )
    free = tuple((i, j) for i in range(1, 5) for j in range(1, 5))
    closed = dataclasses.replace(view(()), free_map=free, blocked_map=blocked, pose=(0.6, 0.6, 0.0))
    assert FrontierPolicy(clearance_cells=0).plan(closed) is None
    assert FrontierPolicy(clearance_cells=0).act(closed, "k").kind == "wait"


def test_controller_does_not_overshoot_within_one_hold_period() -> None:
    # Regression: gain 2 with a 1 s hold flipped a 0.39 rad error to -0.39 forever.
    ctrl, hold = DEFAULT_CONTROLLER, 1.0
    for error in (0.2, 0.39, -0.5, 1.2, -3.0):
        target = (math.cos(error) * 3, math.sin(error) * 3)
        _, omega = ctrl.steer((0.0, 0.0, 0.0), target)
        remaining = error - omega * hold
        assert abs(remaining) < abs(error)
        assert remaining * error >= -1e-12  # no sign flip beyond rounding: no limit cycle
