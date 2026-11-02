"""ARGUS adaptation of the Cross-rank coordination principle, variant CR-ESR_p (E07-T05).

ADAPTATION, not the original implementation: no original code is available (see
docs/baselines/PORT_CONTRACT.md). Ported from the authors' published description:

* Cross-rank of waypoint i: G_i = sum over teammates k of 1 if m_r is greater than the
  distance from waypoint i to k's received path (the nearest segment between
  consecutive received positions, or the single position if only one was received).
* Selection (CR-ESR_p): among the waypoints with the lowest Cross-rank, take the one
  with the lowest Spread-rank sum_k C_r,i / C_k,i, with Euclidean distances from the
  robot and from each teammate's latest received position; ties by known path
  length from the robot, then by grid coordinate.

Changed task semantics in ARGUS: waypoints are the frontier cells of the local map
(not the paper's waypoint generation); only positions are shared, as charged
frames, sent when the robot has moved at least position_filter_m since its last
share; no evidence and no reservations are sent. Movement uses the shared ARGUS
controller and planner, so all methods drive identically.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

from argus.core.types import AgentAction, AgentView, GridIndex
from argus.policies.frontier import NEIGHBOURS, FrontierPolicy

ADAPTATION = "ARGUS adaptation of Cross-rank CR-ESR_p from the published description"

Point = tuple[float, float]


def distance_to_segment(p: Point, a: Point, b: Point) -> float:
    ax, ay = a
    dx, dy = b[0] - ax, b[1] - ay
    length2 = dx * dx + dy * dy
    if length2 == 0.0:
        return math.dist(p, a)
    t = max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / length2))
    return math.dist(p, (ax + t * dx, ay + t * dy))


def path_distance(p: Point, history: list[Point]) -> float:
    """Distance from p to the polyline of received positions (Eq. 7/8 of the paper)."""
    if not history:
        return math.inf
    if len(history) == 1:
        return math.dist(p, history[0])
    return min(distance_to_segment(p, a, b) for a, b in zip(history, history[1:], strict=False))


def cross_rank(p: Point, histories: dict[str, list[Point]], m_r: float) -> int:
    """Number of teammates whose received path passes strictly closer than m_r."""
    return sum(1 for history in histories.values() if m_r > path_distance(p, history))


def spread_rank(p: Point, me: Point, latest: dict[str, Point]) -> float:
    """sum_k C_r,i / C_k,i with Euclidean distances (CR-ESR_p)."""
    mine = math.dist(me, p)
    total = 0.0
    for other in latest.values():
        theirs = math.dist(other, p)
        total += math.inf if theirs == 0.0 else mine / theirs
    return total


@dataclass(slots=True)
class CrossRankPolicy:
    m_r: float = 1.5
    position_filter_m: float = 0.5
    frontier: FrontierPolicy = field(default_factory=FrontierPolicy)
    _histories: dict[str, list[Point]] = field(default_factory=dict, repr=False)
    _last_shared: Point | None = field(default=None, repr=False)
    last_choice: GridIndex | None = field(default=None, repr=False)

    def _path_lengths(self, view: AgentView, cells: set[GridIndex]) -> dict[GridIndex, int]:
        r = view.map_resolution_m
        start = (int(view.pose[0] // r), int(view.pose[1] // r))
        depth = {start: 0}
        queue: deque[GridIndex] = deque([start])
        while queue:
            cell = queue.popleft()
            for di, dj in NEIGHBOURS:
                nxt = (cell[0] + di, cell[1] + dj)
                if nxt in cells and nxt not in depth:
                    depth[nxt] = depth[cell] + 1
                    queue.append(nxt)
        return depth

    def choose(self, view: AgentView) -> tuple[GridIndex, list[GridIndex]] | None:
        r = view.map_resolution_m
        cells = self.frontier.traversable(view)
        depth = self._path_lengths(view, cells)
        start = (int(view.pose[0] // r), int(view.pose[1] // r))
        waypoints = [c for c in self.frontier.frontiers(view, cells) if c in depth and c != start]
        if not waypoints:
            return None

        def center(c: GridIndex) -> Point:
            return ((c[0] + 0.5) * r, (c[1] + 0.5) * r)

        ranks = {c: cross_rank(center(c), self._histories, self.m_r) for c in waypoints}
        lowest = min(ranks.values())
        candidates = [c for c in waypoints if ranks[c] == lowest]
        latest = {k: h[-1] for k, h in self._histories.items() if h}
        me = (view.pose[0], view.pose[1])
        goal = min(candidates, key=lambda c: (spread_rank(center(c), me, latest), depth[c], c))
        return goal, self._path(view, cells, goal)

    def _path(self, view: AgentView, cells: set[GridIndex], goal: GridIndex) -> list[GridIndex]:
        r = view.map_resolution_m
        start = (int(view.pose[0] // r), int(view.pose[1] // r))
        parents: dict[GridIndex, GridIndex | None] = {start: None}
        queue: deque[GridIndex] = deque([start])
        while queue:
            cell = queue.popleft()
            if cell == goal:
                break
            for di, dj in NEIGHBOURS:
                nxt = (cell[0] + di, cell[1] + dj)
                if nxt in cells and nxt not in parents:
                    parents[nxt] = cell
                    queue.append(nxt)
        path = [goal]
        while (parent := parents[path[-1]]) is not None:
            path.append(parent)
        return path[::-1]

    def act(self, view: AgentView, random_key: str) -> AgentAction:
        for report in sorted(view.position_reports, key=lambda p: (p.robot_id, p.tick)):
            self._histories.setdefault(report.robot_id, []).append((report.x, report.y))
        me = (view.pose[0], view.pose[1])
        share = self._last_shared is None or math.dist(me, self._last_shared) >= (
            self.position_filter_m
        )
        if share:
            self._last_shared = me
        ticks = self.frontier.decision_ticks
        if view.last_collision:
            backoff = self.frontier.act(view, random_key)
            return AgentAction(backoff.kind, ticks, backoff.v, backoff.omega, share_position=share)
        chosen = self.choose(view)
        if chosen is None:
            self.last_choice = None
            return AgentAction("wait", ticks, share_position=share)
        goal, path = chosen
        self.last_choice = goal
        r = view.map_resolution_m
        waypoint = path[min(self.frontier.lookahead_cells, len(path) - 1)]
        target = ((waypoint[0] + 0.5) * r, (waypoint[1] + 0.5) * r)
        v, omega = self.frontier.controller.steer(view.pose, target)
        kind = "move" if (v, omega) != (0.0, 0.0) else "wait"
        return AgentAction(kind, ticks, v, omega, share_position=share)
