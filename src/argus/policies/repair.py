"""Decision-relevant repair: re-observe or send, else explore (M5, E06-T02).

At each decision the policy scores two kinds of information actions with the
exact one-step VOI on a common horizon:

* **Re-observe** an entity: drive back to the pose of its latest sighting and
  take one reading there. Distance is the length of the shortest known path
  on the local map; duration is that distance at the controller's speed plus
  one sensor slot. Costs follow the pre-registered unit weights, so extra
  observations cost time and travel.
* **Send** held evidence to a teammate. The sender does not know the
  recipient's belief. Its nominal receiver model assumes the recipient holds
  only evidence the recipient itself originated (as far as the sender has
  seen it); earlier sends by this sender count as delivered only with the
  nominal probability, there are no ACKs. A send bundles the newest unsent
  readings of one entity for one recipient. Its value is the change in loss of
  the recipient's decision, judged with the sender's own belief, times the
  probability that the bundle is new to the recipient and arrives. Sends stay within the
  remaining byte budget using a conservative frame-size bound.

If no candidate has positive value the policy falls back to frontier
exploration. It never sees other agents' true beliefs; its own assumption is
attached to each send so the runner can log the receiver-model error.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

from argus.beliefs.exact_micro import Joint, joint_prior, marginal_open
from argus.beliefs.fixed_lag import NominalPassageModel
from argus.beliefs.joint_filter import JointFilter
from argus.core.events import canonical_json
from argus.core.types import AgentAction, AgentView, EvidenceId, GridIndex, Observation, Send
from argus.policies.frontier import NEIGHBOURS, FrontierPolicy
from argus.policies.voi import (
    CostWeights,
    Losses,
    ObserveAction,
    expected_value,
    send_value,
)

# Upper bound for the transport metadata of a raw-evidence frame (schema v1 adds
# sender, recipient, sent tick, hop count and version, about 90 bytes for short ids).
FRAME_OVERHEAD_BYTES = 160


def estimate_frame_bytes(observation: Observation) -> int:
    """Conservative size of a frame carrying `observation` (never below the real size)."""
    return len(canonical_json(observation.to_json()).encode("utf-8")) + FRAME_OVERHEAD_BYTES


@dataclass(frozen=True, slots=True)
class Candidate:
    kind: str  # "observe" or "send"
    entity: str
    value: float
    target: tuple[float, float, float] | None = None
    send: Send | None = None
    frame_bytes: int = 0


@dataclass(slots=True)
class RepairPolicy:
    team: tuple[str, ...]
    nominal: NominalPassageModel
    losses: Losses
    tick_seconds: float
    window_ticks: int
    delivery_probability: float
    costs: CostWeights = field(default_factory=CostWeights)
    # Nominal decision horizon: the public request cadence of the pilot (one request
    # per robot every 5 s); agents never see the requests themselves.
    horizon_s: float = 5.0
    max_sends_per_decision: int = 2
    frontier: FrontierPolicy = field(default_factory=FrontierPolicy)
    arrival_m: float = 0.3
    _memory: dict[EvidenceId, Observation] = field(default_factory=dict, repr=False)
    _sent: dict[tuple[EvidenceId, str], int] = field(default_factory=dict, repr=False)
    last_candidates: tuple[Candidate, ...] = field(default=(), repr=False)

    # --- nominal models -----------------------------------------------------------

    def _joint_from_history(self, items: list[Observation], now: int) -> Joint:
        m = self.nominal
        filt = JointFilter(
            m.z_model, m.g_model, m.private_error, joint_prior(m.prior_open, m.bias_probability)
        )
        current = 0
        for obs in sorted(items, key=lambda o: (o.observed_tick, o.evidence_id)):
            filt.predict((obs.observed_tick - current) * self.tick_seconds)
            filt.correct(obs.value)
            current = obs.observed_tick
        filt.predict((now - current) * self.tick_seconds)
        return filt.joint

    def receiver_joint(self, recipient: str, entity: str, now: int) -> Joint:
        """Nominal receiver model: evidence the recipient originated, as seen by this sender."""
        floor = now - self.window_ticks
        own = [
            o
            for o in self._memory.values()
            if o.entity == entity and o.evidence_id.origin == recipient and o.observed_tick >= floor
        ]
        return self._joint_from_history(own, now)

    # --- candidates -----------------------------------------------------------------

    def _path_length(self, view: AgentView, target: tuple[float, float]) -> float | None:
        r = view.map_resolution_m
        cells = self.frontier.traversable(view)
        start = (int(view.pose[0] // r), int(view.pose[1] // r))
        goal: GridIndex = (int(target[0] // r), int(target[1] // r))
        cells.add(goal)
        depth = {start: 0}
        queue: deque[GridIndex] = deque([start])
        while queue:
            cell = queue.popleft()
            if cell == goal:
                return depth[cell] * r
            for di, dj in NEIGHBOURS:
                nxt = (cell[0] + di, cell[1] + dj)
                if nxt in cells and nxt not in depth:
                    depth[nxt] = depth[cell] + 1
                    queue.append(nxt)
        return None

    def _observe_candidates(self, view: AgentView) -> list[Candidate]:
        joints = dict(view.joint_beliefs)
        speed = self.frontier.controller.v_max
        found = []
        for sighting in view.sightings:
            if sighting.entity not in joints:
                continue
            target = sighting.observer_pose
            distance = math.dist(view.pose[:2], target[:2])
            if distance > self.arrival_m:
                path = self._path_length(view, target[:2])
                if path is None:
                    continue
                distance = path
            duration = distance / speed + 0.5  # travel plus one sensor slot
            if duration > self.horizon_s:
                continue
            action = ObserveAction(distance, duration, 1)
            estimate = expected_value(
                action,
                joints[sighting.entity],
                self.nominal,
                self.horizon_s,
                self.losses,
                self.costs,
            )
            found.append(Candidate("observe", sighting.entity, estimate.voi, target=target))
        return found

    def _send_candidates(self, view: AgentView) -> list[Candidate]:
        """One bundle per (recipient, entity): the newest unsent readings, up to the cap."""
        found = []
        p = self.delivery_probability
        joints = dict(view.joint_beliefs)
        entities = sorted({o.entity for o in self._memory.values()} & set(joints))
        for entity in entities:
            for recipient in self.team:
                if recipient == view.robot_id:
                    continue
                items = sorted(
                    (
                        o
                        for o in self._memory.values()
                        if o.entity == entity and o.evidence_id.origin != recipient
                    ),
                    key=lambda o: (-o.observed_tick, o.evidence_id),
                )[: self.max_sends_per_decision]
                if not items:
                    continue
                receiver = self.receiver_joint(recipient, entity, view.tick)
                gain = send_value(
                    receiver,
                    joints[entity],
                    [o.value for o in items],
                    self.nominal,
                    self.horizon_s,
                    self.losses,
                )
                unseen = math.prod(
                    (1 - p) ** self._sent.get((o.evidence_id, recipient), 0) for o in items
                )
                value = gain * unseen * p ** len(items)
                for obs in items:
                    send = Send(recipient, obs.evidence_id, marginal_open(receiver))
                    found.append(
                        Candidate(
                            "send", entity, value, send=send, frame_bytes=estimate_frame_bytes(obs)
                        )
                    )
        return found

    # --- decision -----------------------------------------------------------------------

    def _remember(self, view: AgentView) -> None:
        for obs in (*view.new_own_evidence, *view.inbox):
            self._memory[obs.evidence_id] = obs
        floor = view.tick - self.window_ticks
        for eid in [e for e, o in self._memory.items() if o.observed_tick < floor]:
            del self._memory[eid]

    def act(self, view: AgentView, random_key: str) -> AgentAction:
        self._remember(view)
        observe = self._observe_candidates(view)
        sends = self._send_candidates(view)
        self.last_candidates = tuple(observe + sends)
        chosen: list[Send] = []
        budget = view.remaining_bytes
        for candidate in sorted(sends, key=lambda c: (-c.value, c.entity, str(c.send))):
            if candidate.value <= 0 or len(chosen) >= self.max_sends_per_decision:
                break
            if candidate.frame_bytes > budget:
                continue
            assert candidate.send is not None
            chosen.append(candidate.send)
            budget -= candidate.frame_bytes
            key = (candidate.send.evidence_id, candidate.send.recipient)
            self._sent[key] = self._sent.get(key, 0) + 1
        best = max(observe, key=lambda c: (c.value, c.entity), default=None)
        decision_ticks = self.frontier.decision_ticks
        if best is not None and best.value > 0 and best.target is not None:
            v, omega = self._approach(view, best.target)
            kind = "move" if (v, omega) != (0.0, 0.0) else "wait"
            return AgentAction(kind, decision_ticks, v, omega, sends=tuple(chosen))
        fallback = self.frontier.act(view, random_key)
        return AgentAction(
            fallback.kind,
            fallback.duration_ticks,
            fallback.v,
            fallback.omega,
            sends=tuple(chosen),
            reserve=fallback.reserve,
            reserve_ttl_ticks=fallback.reserve_ttl_ticks,
        )

    def _approach(self, view: AgentView, target: tuple[float, float, float]) -> tuple[float, float]:
        controller = self.frontier.controller
        if math.dist(view.pose[:2], target[:2]) > self.arrival_m:
            return controller.steer(view.pose, target[:2])
        error = (target[2] - view.pose[2] + math.pi) % (2 * math.pi) - math.pi
        if abs(error) < 0.05:
            return (0.0, 0.0)  # in position: the next sensor slot takes the reading
        omega = max(-controller.omega_max, min(controller.omega_max, controller.turn_gain * error))
        return (0.0, omega)
