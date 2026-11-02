"""Frontier exploration with message-based target reservations (M5, E05-T02).

A frontier is a known free cell with a 4-neighbour that is neither known free
nor known blocked. The policy plans on known free cells that keep one cell of
clearance from known blocked cells, takes the frontier with the shortest
breadth-first path (ties by grid coordinate) that is not near an active
partner reservation, and announces its choice as a reservation. Partner
targets are known only from reservations that were actually delivered, so
coordination is never free. After a collision it backs off by turning in a
direction drawn from its random key, which favours no robot id.

Planner state is local and bounded: the current target per robot. A target is
kept while it is still an unreserved frontier, so newly seen cells after a
turn do not make the robot oscillate between targets. Use one policy object
per agent; its state never contains anything outside that agent's view.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from argus.core.events import KeyedRandom
from argus.core.types import AgentAction, AgentView, GridIndex
from argus.policies.controller import DEFAULT_CONTROLLER, Controller

NEIGHBOURS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _cell_of(view: AgentView) -> GridIndex:
    r = view.map_resolution_m
    return (int(view.pose[0] // r), int(view.pose[1] // r))


def _center(cell: GridIndex, resolution: float) -> tuple[float, float]:
    return ((cell[0] + 0.5) * resolution, (cell[1] + 0.5) * resolution)


@dataclass(frozen=True, slots=True)
class FrontierPolicy:
    controller: Controller = DEFAULT_CONTROLLER
    decision_ticks: int = 10
    reservation_ttl_ticks: int = 100
    reservation_radius_cells: int = 4
    lookahead_cells: int = 2
    clearance_cells: int = 1
    _targets: dict[str, GridIndex] = field(default_factory=dict, repr=False, compare=False)

    def traversable(self, view: AgentView) -> set[GridIndex]:
        blocked = set(view.blocked_map)
        c = self.clearance_cells
        near = {
            (i + di, j + dj)
            for i, j in blocked
            for di in range(-c, c + 1)
            for dj in range(-c, c + 1)
        }
        return (set(view.free_map) - near) | {_cell_of(view)}

    def frontiers(self, view: AgentView, cells: set[GridIndex]) -> set[GridIndex]:
        known = set(view.free_map) | set(view.blocked_map)
        return {
            cell
            for cell in cells
            if any((cell[0] + di, cell[1] + dj) not in known for di, dj in NEIGHBOURS)
        }

    def _reserved(self, view: AgentView, cell: GridIndex) -> bool:
        r = self.reservation_radius_cells
        return any(
            abs(cell[0] - res.target[0]) <= r and abs(cell[1] - res.target[1]) <= r
            for res in view.reservations
        )

    def plan(self, view: AgentView) -> list[GridIndex] | None:
        """Shortest known path to the nearest unreserved frontier, or None."""
        cells = self.traversable(view)
        goals = {f for f in self.frontiers(view, cells) if not self._reserved(view, f)}
        start = _cell_of(view)
        goals.discard(start)
        parents: dict[GridIndex, GridIndex | None] = {start: None}
        queue: deque[GridIndex] = deque([start])
        found: list[GridIndex] = []
        depth = {start: 0}
        while queue:
            cell = queue.popleft()
            if found and depth[cell] > depth[found[0]]:
                break
            if cell in goals:
                found.append(cell)
                continue
            for di, dj in NEIGHBOURS:
                nxt = (cell[0] + di, cell[1] + dj)
                if nxt in cells and nxt not in parents:
                    parents[nxt] = cell
                    depth[nxt] = depth[cell] + 1
                    queue.append(nxt)
        kept = self._targets.get(view.robot_id)
        if kept in goals and kept in parents:
            goal = kept
        elif found:
            goal = min(found)  # tie-break by grid coordinate
        else:
            self._targets.pop(view.robot_id, None)
            return None
        self._targets[view.robot_id] = goal
        path = [goal]
        while (parent := parents[path[-1]]) is not None:
            path.append(parent)
        return path[::-1]

    def act(self, view: AgentView, random_key: str) -> AgentAction:
        if view.last_collision:
            turn = 1.0 if KeyedRandom(0).uniform("policy.backoff", random_key) < 0.5 else -1.0
            return AgentAction("move", self.decision_ticks, 0.0, turn * self.controller.omega_max)
        path = self.plan(view)
        if path is None:
            return AgentAction("wait", self.decision_ticks)
        waypoint = path[min(self.lookahead_cells, len(path) - 1)]
        v, omega = self.controller.steer(view.pose, _center(waypoint, view.map_resolution_m))
        return AgentAction(
            "move",
            self.decision_ticks,
            v,
            omega,
            reserve=path[-1],
            reserve_ttl_ticks=self.reservation_ttl_ticks,
        )
