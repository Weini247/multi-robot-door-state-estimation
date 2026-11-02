"""Shared navigation controller; every method uses the same settings (M5)."""

from __future__ import annotations

import math
from dataclasses import dataclass

from argus.core.types import Pose2


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


@dataclass(frozen=True, slots=True)
class Controller:
    """Turn towards a waypoint, drive only when roughly aligned."""

    v_max: float = 0.5  # m/s
    omega_max: float = 1.0  # rad/s
    # Policies hold a command for one decision period (10 ticks = 1 s). With
    # turn_gain * hold = 1 a heading error is removed within one period instead of
    # being overshot; a gain of 2 made robots oscillate around +-0.39 rad forever.
    turn_gain: float = 1.0  # rad/s per rad of heading error
    heading_tolerance: float = 0.35  # rad; drive only within this error
    arrival_m: float = 0.1

    def steer(self, pose: Pose2, target: tuple[float, float]) -> tuple[float, float]:
        dx, dy = target[0] - pose[0], target[1] - pose[1]
        distance = math.hypot(dx, dy)
        if distance < self.arrival_m:
            return (0.0, 0.0)
        error = _wrap(math.atan2(dy, dx) - pose[2])
        omega = max(-self.omega_max, min(self.omega_max, self.turn_gain * error))
        v = min(self.v_max, distance) if abs(error) <= self.heading_tolerance else 0.0
        return (v, omega)


DEFAULT_CONTROLLER = Controller()
