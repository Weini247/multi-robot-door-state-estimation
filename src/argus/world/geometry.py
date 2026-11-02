"""2D geometry, SE(2) motion, swept-disc collision and visibility (M1/M2, E02-T02).

Units: metres, seconds, radians. Walls and passages are axis-aligned closed
rectangles; a closed passage blocks motion and sight like a wall, an open one
blocks neither. Rooms and passages form the room graph; Geometry checks that
every passage touches the two rooms it claims to connect.

Motion over an interval uses the exact constant-control SE(2) solution with
sinc. Collisions are checked over the whole interval, not at its end, so thin
walls cannot be tunnelled through: the distance from a linearly moving point
to a convex obstacle (or between two linearly moving points) is convex in
time, which gives the first contact by convex minimization and bisection.
Turning paths are split into chords; each chord check is inflated by the
chord's sagitta, which is conservative (a robot may stop marginally early).
Touching (distance equal to the sum of radii) is allowed; penetration is not.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

Point = tuple[float, float]
Pose = tuple[float, float, float]

_TIME_TOL = 1e-13
# Geometric contact tolerance [m]: gaps within it count as touching, which keeps exact
# contact configurations (e.g. 1.0 - 0.8 = 0.19999999999999996) from reading as overlap.
CONTACT_TOL = 1e-9
_MAX_ARC_PER_CHORD = math.pi / 16


class GeometryError(ValueError):
    """Inconsistent geometry or room graph."""


def _finite(*values: float) -> None:
    for value in values:
        if not math.isfinite(value):
            raise ValueError(f"expected a finite number, got {value!r}")


def wrap_angle(theta: float) -> float:
    """Map an angle to [-pi, pi)."""
    _finite(theta)
    wrapped = math.fmod(theta + math.pi, 2 * math.pi)
    if wrapped < 0:
        wrapped += 2 * math.pi
    return wrapped - math.pi


def _sinc(a: float) -> float:
    if abs(a) < 1e-6:
        a2 = a * a
        return 1.0 - a2 / 6.0 + a2 * a2 / 120.0
    return math.sin(a) / a


def integrate_pose(pose: Pose, v: float, omega: float, dt: float) -> Pose:
    """Exact SE(2) pose after constant (v [m/s], omega [rad/s]) for dt [s]."""
    x, y, theta = pose
    _finite(x, y, theta, v, omega, dt)
    if dt < 0:
        raise ValueError(f"dt must be >= 0, got {dt!r}")
    a = omega * dt / 2.0
    step = v * dt * _sinc(a)
    return (
        x + step * math.cos(theta + a),
        y + step * math.sin(theta + a),
        wrap_angle(theta + omega * dt),
    )


@dataclass(frozen=True, slots=True)
class Rect:
    """Closed axis-aligned rectangle [xmin, xmax] x [ymin, ymax]."""

    xmin: float
    ymin: float
    xmax: float
    ymax: float

    def __post_init__(self) -> None:
        _finite(self.xmin, self.ymin, self.xmax, self.ymax)
        if not (self.xmin < self.xmax and self.ymin < self.ymax):
            raise GeometryError(f"rectangle needs positive extent: {self}")

    @property
    def center(self) -> Point:
        return ((self.xmin + self.xmax) / 2, (self.ymin + self.ymax) / 2)

    def distance_to(self, point: Point) -> float:
        dx = max(self.xmin - point[0], 0.0, point[0] - self.xmax)
        dy = max(self.ymin - point[1], 0.0, point[1] - self.ymax)
        return math.hypot(dx, dy)

    def touches(self, other: Rect) -> bool:
        """Closed intersection (shared boundary counts)."""
        return (
            self.xmin <= other.xmax
            and other.xmin <= self.xmax
            and self.ymin <= other.ymax
            and other.ymin <= self.ymax
        )

    def overlaps_interior(self, other: Rect) -> bool:
        return (
            self.xmin < other.xmax
            and other.xmin < self.xmax
            and self.ymin < other.ymax
            and other.ymin < self.ymax
        )

    def within(self, other: Rect) -> bool:
        return (
            other.xmin <= self.xmin
            and self.xmax <= other.xmax
            and other.ymin <= self.ymin
            and self.ymax <= other.ymax
        )

    def intersects_segment(self, p0: Point, p1: Point) -> bool:
        """Closed segment-rectangle intersection (Liang-Barsky)."""
        t0, t1 = 0.0, 1.0
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        for p, q in (
            (-dx, p0[0] - self.xmin),
            (dx, self.xmax - p0[0]),
            (-dy, p0[1] - self.ymin),
            (dy, self.ymax - p0[1]),
        ):
            if p == 0:
                if q < 0:
                    return False
                continue
            t = q / p
            if p < 0:
                t0 = max(t0, t)
            else:
                t1 = min(t1, t)
            if t0 > t1:
                return False
        return True


@dataclass(frozen=True, slots=True)
class Room:
    room_id: str
    rect: Rect


@dataclass(frozen=True, slots=True)
class Passage:
    """Doorway region connecting two rooms; blocks motion and sight when closed."""

    passage_id: str
    rect: Rect
    rooms: tuple[str, str]


@dataclass(frozen=True, slots=True)
class Geometry:
    bounds: Rect
    walls: tuple[Rect, ...]
    rooms: tuple[Room, ...]
    passages: tuple[Passage, ...]
    _passage_by_id: dict[str, Passage] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        room_ids = [room.room_id for room in self.rooms]
        passage_ids = [p.passage_id for p in self.passages]
        if len(set(room_ids)) != len(room_ids) or len(set(passage_ids)) != len(passage_ids):
            raise GeometryError("room and passage ids must be unique")
        if set(room_ids) & set(passage_ids):
            raise GeometryError("room and passage ids must not collide")
        rooms = {room.room_id: room for room in self.rooms}
        for rect in (*self.walls, *(r.rect for r in self.rooms), *(p.rect for p in self.passages)):
            if not rect.within(self.bounds):
                raise GeometryError(f"{rect} lies outside the bounds {self.bounds}")
        for passage in self.passages:
            a, b = passage.rooms
            if a == b or a not in rooms or b not in rooms:
                raise GeometryError(f"{passage.passage_id} must connect two known rooms")
            for room_id in (a, b):
                if not passage.rect.touches(rooms[room_id].rect):
                    raise GeometryError(f"{passage.passage_id} does not touch room {room_id}")
            for wall in self.walls:
                if passage.rect.overlaps_interior(wall):
                    raise GeometryError(f"{passage.passage_id} overlaps a wall interior")
        object.__setattr__(self, "_passage_by_id", {p.passage_id: p for p in self.passages})

    def passage(self, passage_id: str) -> Passage:
        return self._passage_by_id[passage_id]

    def obstacles(self, passage_open: Mapping[str, int]) -> tuple[Rect, ...]:
        """Walls plus every closed passage. Every passage needs a state."""
        missing = set(self._passage_by_id) - set(passage_open)
        if missing:
            raise GeometryError(f"missing passage states: {sorted(missing)}")
        closed = tuple(p.rect for p in self.passages if passage_open[p.passage_id] == 0)
        return self.walls + closed

    def neighbours(self, room_id: str, passage_open: Mapping[str, int]) -> tuple[str, ...]:
        """Rooms reachable through currently open passages (derived room graph)."""
        result = []
        for p in self.passages:
            if passage_open[p.passage_id] and room_id in p.rooms:
                result.append(p.rooms[1] if p.rooms[0] == room_id else p.rooms[0])
        return tuple(sorted(result))


# --- visibility -------------------------------------------------------------


def visible(
    observer: Pose,
    target: Point,
    range_m: float,
    fov_rad: float,
    occluders: Sequence[Rect],
) -> bool:
    """Range, field of view and line of sight. Boundaries of range and FOV count as
    visible; a sight line touching an occluder (even along its edge) is blocked."""
    _finite(range_m, fov_rad, *target)
    dx, dy = target[0] - observer[0], target[1] - observer[1]
    if math.hypot(dx, dy) > range_m:
        return False
    if dx != 0 or dy != 0:
        bearing = wrap_angle(math.atan2(dy, dx) - observer[2])
        if abs(bearing) > fov_rad / 2:
            return False
    origin = (observer[0], observer[1])
    return not any(rect.intersects_segment(origin, target) for rect in occluders)


def visible_passages(
    geometry: Geometry,
    observer: Pose,
    range_m: float,
    fov_rad: float,
    passage_open: Mapping[str, int],
) -> tuple[str, ...]:
    """Passages whose centre passes the sight test; a passage never occludes itself."""
    obstacles = geometry.obstacles(passage_open)
    result = []
    for passage in geometry.passages:
        occluders = [rect for rect in obstacles if rect is not passage.rect]
        if visible(observer, passage.rect.center, range_m, fov_rad, occluders):
            result.append(passage.passage_id)
    return tuple(result)


# --- swept motion -------------------------------------------------------------


def _first_below(
    f: Callable[[float], float], threshold: float, lo: float, hi: float
) -> float | None:
    """First t in [lo, hi] with f(t) < threshold for convex f; returns the last safe time.

    Penetration means f(t) < threshold - CONTACT_TOL.
    """
    threshold -= CONTACT_TOL
    if f(lo) < threshold:
        return lo
    a, b = lo, hi
    ratio = (math.sqrt(5) - 1) / 2
    c, d = b - ratio * (b - a), a + ratio * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(200):
        if b - a < _TIME_TOL:
            break
        if fc <= fd:
            b, d, fd = d, c, fc
            c = b - ratio * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + ratio * (b - a)
            fd = f(d)
    t_min = min((lo, hi, (a + b) / 2), key=f)
    if f(t_min) >= threshold:
        return None
    safe, unsafe = lo, t_min
    while unsafe - safe > _TIME_TOL:
        mid = (safe + unsafe) / 2
        if f(mid) < threshold:
            unsafe = mid
        else:
            safe = mid
    return safe


@dataclass(frozen=True, slots=True)
class SweptPath:
    """Motion over one interval as chords at uniform time fractions."""

    points: tuple[Point, ...]
    sagitta: float
    end_pose: Pose
    start_pose: Pose
    v: float
    omega: float
    dt: float

    @classmethod
    def build(cls, pose: Pose, v: float, omega: float, dt: float) -> SweptPath:
        end = integrate_pose(pose, v, omega, dt)
        turn = abs(omega * dt)
        chords = max(1, math.ceil(turn / _MAX_ARC_PER_CHORD))
        points = tuple(
            integrate_pose(pose, v, omega, dt * k / chords)[:2] for k in range(chords + 1)
        )
        sagitta = 0.0
        if omega != 0 and v != 0:
            radius = abs(v / omega)
            sagitta = radius * (1 - math.cos(turn / chords / 2))
        return cls(points, sagitta, end, pose, v, omega, dt)

    @property
    def breaks(self) -> tuple[float, ...]:
        n = len(self.points) - 1
        return tuple(k / n for k in range(n + 1))

    def at(self, t: float, stop: float = 1.0) -> Point:
        """Chord position at fraction t, frozen after the stop fraction."""
        t = min(t, stop)
        n = len(self.points) - 1
        k = min(int(t * n), n - 1)
        local = t * n - k
        (x0, y0), (x1, y1) = self.points[k], self.points[k + 1]
        return (x0 + (x1 - x0) * local, y0 + (y1 - y0) * local)

    def pose_at(self, fraction: float) -> Pose:
        return integrate_pose(self.start_pose, self.v, self.omega, self.dt * fraction)


def _pieces(*breaks: Sequence[float]) -> list[tuple[float, float]]:
    cuts = sorted({0.0, 1.0, *(b for group in breaks for b in group if 0.0 <= b <= 1.0)})
    return list(zip(cuts[:-1], cuts[1:], strict=True))


def _distance_along(path: SweptPath, rect: Rect) -> Callable[[float], float]:
    def distance(t: float) -> float:
        return rect.distance_to(path.at(t))

    return distance


def _travel(path: SweptPath, lo: float, hi: float, stop: float = 1.0) -> float:
    """Length of the chord motion between fractions lo and hi (an exact Lipschitz bound)."""
    (x0, y0), (x1, y1) = path.at(lo, stop), path.at(hi, stop)
    return math.hypot(x1 - x0, y1 - y0)


def first_wall_contact(path: SweptPath, radius: float, obstacles: Sequence[Rect]) -> float | None:
    """Last safe fraction before the disc would penetrate an obstacle, or None.

    Pieces are linear, so the distance to a rectangle changes by at most the
    travelled length; obstacles that stay farther away are skipped exactly.
    """
    threshold = radius + path.sagitta
    best: float | None = None
    for lo, hi in _pieces(path.breaks):
        travel = _travel(path, lo, hi)
        start = path.at(lo)
        for rect in obstacles:
            if rect.distance_to(start) - travel >= threshold:
                continue
            hit = _first_below(_distance_along(path, rect), threshold, lo, hi)
            if hit is not None and (best is None or hit < best):
                best = hit
        if best is not None:
            return best
    return best


def first_robot_contact(
    a: SweptPath, stop_a: float, radius_a: float, b: SweptPath, stop_b: float, radius_b: float
) -> float | None:
    """Last safe fraction before two (possibly stopped) discs would overlap, or None."""
    threshold = radius_a + radius_b + a.sagitta + b.sagitta
    for lo, hi in _pieces(a.breaks, b.breaks, (stop_a, stop_b)):
        pa, pb = a.at(lo, stop_a), b.at(lo, stop_b)
        reach = _travel(a, lo, hi, stop_a) + _travel(b, lo, hi, stop_b)
        if math.hypot(pa[0] - pb[0], pa[1] - pb[1]) - reach >= threshold:
            continue  # the gap cannot shrink below the threshold on this piece

        def gap(t: float) -> float:
            pa, pb = a.at(t, stop_a), b.at(t, stop_b)
            return math.hypot(pa[0] - pb[0], pa[1] - pb[1])

        hit = _first_below(gap, threshold, lo, hi)
        if hit is not None:
            return hit
    return None


@dataclass(frozen=True, slots=True)
class Motion:
    robot_id: str
    pose: Pose
    radius: float
    v: float
    omega: float


@dataclass(frozen=True, slots=True)
class CollisionEvent:
    kind: str  # "wall" or "robot"
    robot_ids: tuple[str, ...]
    fraction: float


def resolve_motions(
    motions: Sequence[Motion], dt: float, obstacles: Sequence[Rect]
) -> tuple[dict[str, Pose], tuple[CollisionEvent, ...]]:
    """Move all robots simultaneously over dt; stop each at its last safe fraction.

    Pair conflicts stop both robots at the same fraction. Stops are recomputed
    until no robot moves further than allowed; the result does not depend on
    the order or the ids of the robots.
    """
    ids = [m.robot_id for m in motions]
    if len(set(ids)) != len(ids):
        raise ValueError("robot ids must be unique")
    paths = {m.robot_id: SweptPath.build(m.pose, m.v, m.omega, dt) for m in motions}
    radius = {m.robot_id: m.radius for m in motions}
    stop = {rid: 1.0 for rid in ids}
    events: dict[tuple[str, tuple[str, ...]], float] = {}
    for rid in ids:
        hit = first_wall_contact(paths[rid], radius[rid], obstacles)
        if hit is not None:
            stop[rid] = hit
            events[("wall", (rid,))] = hit
    ordered = sorted(ids)
    for _ in range(4 * len(ids) + 4):
        changed = False
        for i, a in enumerate(ordered):
            for b in ordered[i + 1 :]:
                hit = first_robot_contact(
                    paths[a], stop[a], radius[a], paths[b], stop[b], radius[b]
                )
                if hit is not None and hit < max(stop[a], stop[b]) - _TIME_TOL:
                    for rid in (a, b):
                        stop[rid] = min(stop[rid], hit)
                    events[("robot", (a, b))] = hit
                    changed = True
        if not changed:
            break
    else:
        raise RuntimeError("collision resolution did not converge")
    poses = {
        rid: paths[rid].pose_at(stop[rid]) if stop[rid] < 1.0 else paths[rid].end_pose
        for rid in ids
    }
    collisions = tuple(
        CollisionEvent(kind, robots, fraction)
        for (kind, robots), fraction in sorted(events.items())
    )
    return poses, collisions
