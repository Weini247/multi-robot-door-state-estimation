"""Small closed-loop harness for E05 policy tests (the full runner is E05-T03)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from argus.agents.local import LocalAgent
from argus.beliefs.fixed_lag import FixedLagBelief, NominalPassageModel
from argus.communication.messages import ReservationMessage
from argus.communication.queue import Channel, ChannelConfig
from argus.core.types import AgentAction, Reservation
from argus.dynamics.binary_ctmc import BinaryCTMC
from argus.policies.frontier import FrontierPolicy
from argus.sensors.occupancy import scan_map
from argus.world.geometry import Geometry, Rect, Room
from argus.world.scenario import RobotStart, ScenarioSpec, WorldBackend2D

MODEL = NominalPassageModel(BinaryCTMC(0.01, 0.02), BinaryCTMC(0.005, 0.045), 0.05, 1 / 3, 0.1)
RES = 0.25


def boxed(width: float, height: float, inner: tuple[Rect, ...] = ()) -> Geometry:
    t = 0.1
    walls = (
        Rect(0.0, 0.0, width, t),
        Rect(0.0, height - t, width, height),
        Rect(0.0, 0.0, t, height),
        Rect(width - t, 0.0, width, height),
        *inner,
    )
    return Geometry(
        Rect(0.0, 0.0, width, height), walls, (Room("room", Rect(0, 0, width, height)),), ()
    )


def scenario(geometry: Geometry, starts: tuple[RobotStart, ...], horizon: int) -> ScenarioSpec:
    return ScenarioSpec("policy-fixture", geometry, 0.1, horizon, 0.2, 0.5, 1.0, starts, ())


@dataclass
class Loop:
    spec: ScenarioSpec
    policy: FrontierPolicy = field(default_factory=FrontierPolicy)
    deliver_reservations: bool = True
    preseed: dict[str, tuple[set[tuple[int, int]], set[tuple[int, int]]]] = field(
        default_factory=dict
    )
    min_gap: float = math.inf
    reservations_sent: int = 0

    def run(self) -> dict[str, LocalAgent]:
        world = WorldBackend2D(self.spec, 1)
        config = ChannelConfig(0.1, 1024.0, 100, 4096.0, 2, 65536)
        self.channel = Channel(config)
        agents = {
            s.robot_id: LocalAgent(
                s.robot_id,
                FixedLagBelief("ep", {}, 0.1, 300, 4096, default_model=MODEL),
                s.pose,
                RES,
            )
            for s in self.spec.starts
        }
        for rid, (free, blocked) in self.preseed.items():
            agents[rid].update_map(free, blocked)
        actions = {rid: AgentAction("wait", 1) for rid in agents}
        until = dict.fromkeys(agents, 0)
        last_target: dict[str, object] = {}
        seq = 0
        for tick in range(self.spec.horizon_ticks + 1):
            collided: set[str] = set()
            if tick:
                step = world.advance(tick, {r: (a.v, a.omega) for r, a in actions.items()})
                for event in step.collisions:
                    collided |= set(event.robot_ids)
            poses = world.poses()
            ids = sorted(poses)
            for i, a in enumerate(ids):
                for b in ids[i + 1 :]:
                    self.min_gap = min(self.min_gap, math.dist(poses[a][:2], poses[b][:2]))
            for frame in self.channel.deliveries(tick):
                if isinstance(frame, ReservationMessage) and self.deliver_reservations:
                    agents[frame.recipient].receive_reservation(
                        Reservation(frame.sender, frame.target, frame.expires_tick), tick
                    )
            for rid, agent in agents.items():
                agent.set_pose(poses[rid], rid in collided)
                if tick % 5 == 0:
                    free, blocked = scan_map(
                        self.spec.geometry, poses[rid], {}, 5.0, math.radians(120), RES
                    )
                    agent.update_map(free, blocked)
                if tick >= until[rid]:
                    action = agent.decide(tick, self.policy, f"{rid}/{tick}")
                    actions[rid] = action
                    until[rid] = tick + action.duration_ticks
                    if action.reserve is not None and action.reserve != last_target.get(rid):
                        last_target[rid] = action.reserve
                        for other in agents:
                            if other != rid:
                                seq += 1
                                frame = ReservationMessage(
                                    rid,
                                    other,
                                    tick,
                                    action.reserve,
                                    tick + action.reserve_ttl_ticks,
                                    seq,
                                )
                                if self.channel.submit(frame, tick) == "queued":
                                    self.reservations_sent += 1
        return agents
