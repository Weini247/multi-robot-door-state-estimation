"""Inspection requests and the closed-loop episode runner (M5/M6, E05-T03).

Per tick k, in the phase order of the architecture: WORLD (motion over
(k-1, k] and passage states z(k)), DELIVERY (frames due at k), SENSOR (passage
readings and map scans in physical slots), BELIEF/POLICY (agents whose
previous action ended decide; their sends and reservations are submitted and
act from k+1), EVALUATE (inspection requests due at k).

Inspection requests (tick, recipient, entity) are generated before the run from
their own seed and the scenario only, so every method faces the same requests.
They are never shown to agents; an agent answers with its current P(open), or
the nominal prior for an entity it never saw. Decision d = 1 iff
p >= c10 / (c10 + c01); loss L(d, z) = c10 [d=1, z=0] + c01 [d=0, z=1].

Status: "ok" (horizon reached), "timeout" (wall-time limit; unanswered requests
are counted), "invalid" (impossible evidence under the nominal model) and
"error" (any other failure). Only "ok" is a successful episode.

Evaluator trace (E08-T02): ExperimentRunner(trace=True) additionally records,
evaluator-side only, the passage change ticks, the coverage of fixed free target
cells and the per-robot channel bytes. It never feeds anything back to agents and
does not touch any random stream. On a failure the result keeps the records
answered before the failure tick in partial_records; records stays empty for
invalid and error episodes as before.

Factor switches (E07-T08): sensor_bias None uses independent flips with the
marginal error; a SensorBias uses the shared-bias field (M2) with the private
error matched to the same marginal. hop_limit bounds forwarding: a robot may
forward evidence it received at hop h only if h + 1 <= hop_limit (None: no
limit); refused forwards are not submitted and are counted in hop_limited.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import time
import tracemalloc
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from argus import __version__
from argus.agents.local import LocalAgent, Policy
from argus.beliefs.exact_micro import ImpossibleEvidence
from argus.beliefs.fixed_lag import NominalPassageModel
from argus.communication.loss import GilbertElliott
from argus.communication.messages import (
    Message,
    PositionMessage,
    ReservationMessage,
    originate,
)
from argus.communication.queue import Channel, ChannelConfig
from argus.core.events import RNG_SCHEMA, KeyedRandom, canonical_json
from argus.core.types import AgentAction, EvidenceId, GridIndex, PositionReport, Reservation
from argus.dynamics.binary_ctmc import BinaryCTMC
from argus.policies.baselines import METHOD_SPECS, MethodContext, MethodSpec
from argus.policies.voi import Losses
from argus.sensors.binary import BinarySensor, IndependentNoise, NoiseModel, SensorConfig
from argus.sensors.correlated import BiasField, SharedBiasNoise, matched_private_error
from argus.sensors.occupancy import scan_map
from argus.world.geometry import Geometry, Rect
from argus.world.scenario import ScenarioSpec, WorldBackend2D

Status = Literal["ok", "timeout", "invalid", "error"]


def method_context(config: EpisodeConfig, team: tuple[str, ...]) -> MethodContext:
    """Public, method-independent facts handed to every method factory."""
    return MethodContext(
        episode_id=config.transport_key,
        team=team,
        nominal=config.nominal,
        marginal_error=config.sensor_error,
        losses=Losses(config.loss_false_open, config.loss_false_closed),
        tick_seconds=config.scenario.tick_seconds,
        window_ticks=config.belief_window_ticks,
        capacity=config.belief_capacity,
        delivery_probability=1.0 - config.nominal_drop_rate,
        params=config.method_params,
    )


def _policy_factory(spec: MethodSpec) -> Callable[[EpisodeConfig, tuple[str, ...]], Policy]:
    def build(config: EpisodeConfig, team: tuple[str, ...]) -> Policy:
        policy: Policy = spec.policy(method_context(config, team))
        return policy

    return build


METHODS: dict[str, Callable[[EpisodeConfig, tuple[str, ...]], Policy]] = {
    method_id: _policy_factory(spec) for method_id, spec in METHOD_SPECS.items()
}
MANIFEST_SCHEMA = 1


@dataclass(frozen=True, slots=True)
class SensorBias:
    """Shared-bias field: P(g=1) = rho, mu_01 + mu_10 = rate_sum, square cells."""

    rho: float
    rate_sum: float = 0.05
    cell_size_m: float = 4.0

    def __post_init__(self) -> None:
        if not 0.0 < self.rho < 0.5:
            raise ValueError("rho must be in (0, 0.5)")
        for name in ("rate_sum", "cell_size_m"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and > 0")

    @property
    def rates(self) -> tuple[float, float]:
        return (self.rho * self.rate_sum, (1.0 - self.rho) * self.rate_sum)


@dataclass(frozen=True, slots=True)
class EpisodeConfig:
    episode_id: str
    building_id: str
    scenario: ScenarioSpec
    method_id: str
    seed_world: int
    seed_sensor: int
    seed_channel: int
    seed_requests: int
    seed_policy: int
    sensor_range_m: float = 5.0
    sensor_fov_rad: float = math.radians(120)
    sensor_slot_ticks: int = 5
    sensor_error: float = 0.14
    map_resolution_m: float = 0.25
    request_period_ticks: int = 50
    loss_false_open: float = 4.0
    loss_false_closed: float = 1.0
    belief_window_ticks: int = 300
    belief_capacity: int = 4096
    nominal: NominalPassageModel = field(
        default_factory=lambda: NominalPassageModel(
            BinaryCTMC(0.01, 0.02), BinaryCTMC(0.005, 0.045), 0.05, 1 / 3, 0.1
        )
    )
    channel: ChannelConfig = field(
        default_factory=lambda: ChannelConfig(0.1, 1024.0, 100, 4096.0, 2, 65536)
    )
    ge_rates: tuple[float, float, float, float] = (0.02, 0.2, 0.01, 0.5)
    max_wall_seconds: float = 600.0
    trace_memory: bool = False  # tracemalloc roughly doubles the run time
    # The sender's nominal belief about loss on a link (no ACKs): stationary GE drop rate
    # of the pilot channel, a nominal assumption, not read from the simulated channel.
    nominal_drop_rate: float = 0.01 * (1 - 0.02 / 0.22) + 0.5 * (0.02 / 0.22)
    sensor_bias: SensorBias | None = None  # None: independent flips
    hop_limit: int | None = None  # None: forwarding unlimited
    method_params: tuple[tuple[str, float], ...] = ()  # only declared tunables
    code_commit: str | None = None
    # Transport identity of evidence (EvidenceId.episode, frame ids, message bytes and the
    # per-frame loss keys). None keeps the historical behaviour (the episode id, which
    # names method and config). Study episodes set a method-neutral key of fixed length
    # (protocol.transport_id), so every method sends equally long frames under the same
    # loss draws (amendment 01, O1).
    transport_id: str | None = None

    def __post_init__(self) -> None:
        if self.method_id not in METHODS:
            raise ValueError(f"unknown method {self.method_id!r}")
        if self.loss_false_open < 0 or self.loss_false_closed < 0:
            raise ValueError("losses must be >= 0")
        if self.loss_false_open + self.loss_false_closed == 0:
            raise ValueError("at least one loss must be positive")
        if self.hop_limit is not None and (type(self.hop_limit) is not int or self.hop_limit < 0):
            raise ValueError("hop_limit must be None or an int >= 0")
        names = [name for name, _ in self.method_params]
        unknown = set(names) - set(METHOD_SPECS[self.method_id].tunable)
        if unknown or len(names) != len(set(names)):
            raise ValueError(f"invalid method_params for {self.method_id}: {sorted(unknown)}")
        if self.sensor_bias is not None and self.sensor_bias.rho > self.sensor_error:
            raise ValueError("rho must not exceed the marginal sensor error")

    def to_json(self) -> dict[str, Any]:
        n = self.nominal
        return {
            "episode_id": self.episode_id,
            "building_id": self.building_id,
            "scenario_hash": self.scenario.scenario_hash,
            "method_id": self.method_id,
            "seeds": {
                "world": self.seed_world,
                "sensor": self.seed_sensor,
                "channel": self.seed_channel,
                "requests": self.seed_requests,
                "policy": self.seed_policy,
            },
            "sensor": [
                self.sensor_range_m,
                self.sensor_fov_rad,
                self.sensor_slot_ticks,
                self.sensor_error,
            ],
            "map_resolution_m": self.map_resolution_m,
            "request_period_ticks": self.request_period_ticks,
            "losses": [self.loss_false_open, self.loss_false_closed],
            "belief": [self.belief_window_ticks, self.belief_capacity],
            "nominal": [
                n.z_model.rate_01,
                n.z_model.rate_10,
                n.g_model.rate_01,
                n.g_model.rate_10,
                n.private_error,
                n.prior_open,
                n.bias_probability,
            ],
            "channel": [
                self.channel.tick_seconds,
                self.channel.sender_bytes_per_second,
                self.channel.window_ticks,
                self.channel.link_bytes_per_second,
                self.channel.propagation_ticks,
                self.channel.queue_capacity_bytes,
            ],
            "ge": list(self.ge_rates),
            "max_wall_seconds": self.max_wall_seconds,
            "sensor_bias": None
            if self.sensor_bias is None
            else [
                self.sensor_bias.rho,
                self.sensor_bias.rate_sum,
                self.sensor_bias.cell_size_m,
            ],
            "hop_limit": self.hop_limit,
            "method_params": [[n, v] for n, v in sorted(self.method_params)],
            **({} if self.transport_id is None else {"transport_id": self.transport_id}),
        }

    @property
    def config_sha256(self) -> str:
        return hashlib.sha256(canonical_json(self.to_json()).encode("utf-8")).hexdigest()

    @property
    def transport_key(self) -> str:
        return self.episode_id if self.transport_id is None else self.transport_id

    @property
    def decision_threshold(self) -> float:
        return self.loss_false_open / (self.loss_false_open + self.loss_false_closed)


@dataclass(frozen=True, slots=True)
class InspectionRequest:
    tick: int
    recipient: str
    entity: str


def generate_requests(config: EpisodeConfig) -> tuple[InspectionRequest, ...]:
    """Each robot gets one request every period; entities drawn by key, not by method."""
    rng = KeyedRandom(config.seed_requests)
    spec = config.scenario
    passages = sorted(p.passage_id for p in spec.geometry.passages)
    requests = []
    for start in sorted(s.robot_id for s in spec.starts):
        for k in range(1, spec.horizon_ticks // config.request_period_ticks + 1):
            index = int(rng.uniform("requests", spec.scenario_id, start, k) * len(passages))
            requests.append(
                InspectionRequest(k * config.request_period_ticks, start, passages[index])
            )
    return tuple(sorted(requests, key=lambda r: (r.tick, r.recipient)))


@dataclass(frozen=True, slots=True)
class EvaluationRecord:
    episode_id: str
    building_id: str
    method_id: str
    tick: int
    robot_id: str
    entity: str
    probability: float
    truth: int
    decision: int
    decision_loss: float
    sent_bytes: int


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    episode_id: str
    status: Status
    records: tuple[EvaluationRecord, ...]
    unanswered: int
    wall_seconds: float
    peak_traced_bytes: int | None  # None unless trace_memory was requested
    manifest: dict[str, Any]
    error: str | None = None
    failure_tick: int | None = None  # last tick reached by a failed episode
    partial_records: tuple[EvaluationRecord, ...] = ()  # answered before a failure
    trace: dict[str, Any] | None = None  # evaluator trace, only with trace=True


TARGET_SETS = ("closed", "open")  # free with every passage closed / open


class EpisodeTrace:
    """Evaluator-side record of passage changes and coverage (spec M6 coverage).

    Targets are fixed before the run: grid cells at map resolution inside the
    bounds whose square does not overlap a wall (and, for "closed", any passage).
    A target counts as covered once a sensor scan returned it as free.
    """

    def __init__(self, geometry: Geometry, resolution: float) -> None:
        self.resolution = resolution
        b = geometry.bounds
        cells = [
            (i, j)
            for i in range(math.floor(b.xmin / resolution), math.ceil(b.xmax / resolution))
            for j in range(math.floor(b.ymin / resolution), math.ceil(b.ymax / resolution))
        ]
        self.targets: dict[str, frozenset[GridIndex]] = {}
        for name in TARGET_SETS:
            state = 0 if name == "closed" else 1
            obstacles = geometry.obstacles({p.passage_id: state for p in geometry.passages})
            self.targets[name] = frozenset(c for c in cells if not self._blocked(c, obstacles))
        self.covered: dict[str, set[GridIndex]] = {name: set() for name in TARGET_SETS}
        self.coverage_events: dict[str, list[tuple[int, int]]] = {n: [] for n in TARGET_SETS}
        self.initial: dict[str, int] | None = None
        self.changes: dict[str, list[int]] = {p.passage_id: [] for p in geometry.passages}
        self._last: dict[str, int] = {}
        self.channel_bytes: dict[str, int] = {}
        self.channel_rx: dict[str, int] = {}
        self.channel_dropped: dict[str, int] = {}
        self.channel_rejects: dict[str, int] = {}
        self.hop_limited = 0

    def _blocked(self, cell: GridIndex, obstacles: tuple[Rect, ...]) -> bool:
        r = self.resolution
        square = Rect(cell[0] * r, cell[1] * r, (cell[0] + 1) * r, (cell[1] + 1) * r)
        return any(square.overlaps_interior(rect) for rect in obstacles)

    def passages(self, tick: int, states: dict[str, int]) -> None:
        if self.initial is None:
            self.initial = dict(sorted(states.items()))
        else:
            for pid, state in states.items():
                if state != self._last[pid]:
                    self.changes[pid].append(tick)
        self._last = dict(states)

    def seen(self, tick: int, cells: frozenset[GridIndex]) -> None:
        for name in TARGET_SETS:
            before = len(self.covered[name])
            self.covered[name] |= cells & self.targets[name]
            if len(self.covered[name]) != before:
                self.coverage_events[name].append((tick, len(self.covered[name])))

    def to_json(self) -> dict[str, Any]:
        return {
            "targets": {n: len(self.targets[n]) for n in TARGET_SETS},
            "coverage_events": {n: [list(e) for e in self.coverage_events[n]] for n in TARGET_SETS},
            "passage_initial": self.initial,
            "passage_changes": {pid: ticks for pid, ticks in sorted(self.changes.items())},
            "channel_tx_bytes": dict(sorted(self.channel_bytes.items())),
            "channel_rx_bytes": dict(sorted(self.channel_rx.items())),
            "channel_dropped_bytes": dict(sorted(self.channel_dropped.items())),
            "channel_rejects": dict(sorted(self.channel_rejects.items())),
            "hop_limited": self.hop_limited,
        }


def binary_loss(decision: int, truth: int, false_open: float, false_closed: float) -> float:
    if decision == 1 and truth == 0:
        return false_open
    if decision == 0 and truth == 1:
        return false_closed
    return 0.0


def build_noise(config: EpisodeConfig) -> tuple[str, NoiseModel]:
    """Generating sensor noise of an episode: independent or shared-bias (M2)."""
    spec = config.scenario
    noise: NoiseModel
    if config.sensor_bias is None:
        model_name = "binary-independent-v1"
        noise = IndependentNoise(
            config.sensor_error, config.sensor_error, KeyedRandom(config.seed_sensor)
        )
    else:
        bias = config.sensor_bias
        model_name = "binary-shared-bias-v1"
        bias_field = BiasField(
            bias.cell_size_m,
            *bias.rates,
            spec.tick_seconds,
            KeyedRandom(config.seed_sensor),
            spec.scenario_id,
        )
        noise = SharedBiasNoise(
            bias_field,
            matched_private_error(config.sensor_error, bias.rho),
            KeyedRandom(config.seed_sensor),
        )
    return model_name, noise


class ExperimentRunner:
    """Runs one episode; privileged wiring of world, sensors, channel and agents lives here."""

    def __init__(self, trace: bool = False) -> None:
        self.trace = trace

    def run(self, config: EpisodeConfig) -> EpisodeResult:
        requests = generate_requests(config)
        self._trace = (
            EpisodeTrace(config.scenario.geometry, config.map_resolution_m) if self.trace else None
        )
        self._progress_tick = 0
        self._progress_records: list[EvaluationRecord] = []
        self.last_channel: Channel | None = None
        started = time.perf_counter()
        status: Status = "ok"
        error: str | None = None
        records: list[EvaluationRecord] = []
        peak: int | None = None
        if config.trace_memory:
            tracemalloc.start()
        try:
            records = self._episode(config, requests, started)
        except _Timeout as stop:
            status, records = "timeout", stop.records
        except ImpossibleEvidence as failure:
            status, error = "invalid", f"ImpossibleEvidence: {failure}"
        except Exception as failure:  # noqa: BLE001 - every other failure is an error result
            status, error = "error", f"{type(failure).__name__}: {failure}"
        finally:
            if config.trace_memory:
                peak = tracemalloc.get_traced_memory()[1]
                tracemalloc.stop()
        wall = time.perf_counter() - started
        failure_tick = None if status == "ok" else self._progress_tick
        partial = () if status == "ok" else tuple(self._progress_records)
        if status in ("invalid", "error"):
            records = []  # partial records of a failed episode are not results
        unanswered = len(requests) - len(records)
        manifest = build_manifest(config, status, len(requests), len(records), error)
        trace = None
        if self._trace is not None:
            if self.last_channel is not None:
                stats = self.last_channel.stats
                self._trace.channel_bytes = dict(stats.tx_attempt_bytes)
                self._trace.channel_rx = dict(stats.rx_bytes)
                self._trace.channel_dropped = dict(stats.dropped_bytes)
                self._trace.channel_rejects = {f"{a}/{b}": n for (a, b), n in stats.rejects.items()}
            self._trace.hop_limited = getattr(self, "hop_limited", 0)
            trace = self._trace.to_json()
        return EpisodeResult(
            config.episode_id,
            status,
            tuple(records),
            unanswered,
            wall,
            peak,
            manifest,
            error,
            failure_tick,
            partial,
            trace,
        )

    def _episode(
        self, config: EpisodeConfig, requests: tuple[InspectionRequest, ...], started: float
    ) -> list[EvaluationRecord]:
        spec = config.scenario
        world = WorldBackend2D(spec, config.seed_world)
        model_name, noise = build_noise(config)
        sensor = BinarySensor(
            SensorConfig(
                config.sensor_range_m,
                config.sensor_fov_rad,
                config.sensor_slot_ticks,
                model_name,
            ),
            noise,
            config.transport_key,
        )
        a, b, eps_g, eps_b = config.ge_rates
        loss = GilbertElliott(
            a,
            b,
            eps_g,
            eps_b,
            spec.tick_seconds,
            KeyedRandom(config.seed_channel),
            spec.scenario_id,
        )
        channel = Channel(config.channel, loss)
        self.last_channel = channel
        team = tuple(sorted(s.robot_id for s in spec.starts))
        policies = {rid: METHODS[config.method_id](config, team) for rid in team}
        self.recipient_model_log: list[RecipientModelRecord] = []
        self.hop_limited = 0
        received_hop: dict[tuple[str, EvidenceId], int] = {}
        agents = {
            s.robot_id: LocalAgent(
                s.robot_id,
                METHOD_SPECS[config.method_id].belief(method_context(config, team)),
                s.pose,
                config.map_resolution_m,
            )
            for s in spec.starts
        }
        actions = {rid: AgentAction("wait", 1) for rid in agents}
        until = dict.fromkeys(agents, 0)
        reserved: dict[str, GridIndex | None] = dict.fromkeys(agents)
        due = list(requests)
        records: list[EvaluationRecord] = []
        self._progress_records = records
        trace = self._trace
        sequence = 0
        for tick in range(spec.horizon_ticks + 1):
            self._progress_tick = tick
            if time.perf_counter() - started > config.max_wall_seconds:
                raise _Timeout(records)
            collided: set[str] = set()
            if tick:  # WORLD
                step = world.advance(tick, {r: (x.v, x.omega) for r, x in actions.items()})
                for event in step.collisions:
                    collided |= set(event.robot_ids)
            for frame in channel.deliveries(tick):  # DELIVERY
                if isinstance(frame, Message):
                    hop_key = (frame.recipient, frame.observation.evidence_id)
                    received_hop[hop_key] = min(
                        received_hop.get(hop_key, frame.hop_count), frame.hop_count
                    )
                    agents[frame.recipient].receive([frame.observation], tick)
                elif isinstance(frame, PositionMessage):
                    agents[frame.recipient].receive_position(
                        PositionReport(frame.sender, frame.sent_tick, frame.x, frame.y)
                    )
                elif isinstance(frame, ReservationMessage):
                    agents[frame.recipient].receive_reservation(
                        Reservation(frame.sender, frame.target, frame.expires_tick), tick
                    )
                else:
                    raise TypeError(f"no receiver for frame {frame.frame_id}")
            poses, states = world.poses(), world.passage_states()
            if trace is not None:
                trace.passages(tick, states)
            for rid, agent in agents.items():  # SENSOR
                agent.set_pose(poses[rid], rid in collided)
                agent.set_budget(channel.remaining_budget(rid, tick + 1))
                if tick % config.sensor_slot_ticks == 0:
                    agent.observe(
                        sensor.observe(world, rid, tick // config.sensor_slot_ticks), tick
                    )
                    free, blocked = scan_map(
                        spec.geometry,
                        poses[rid],
                        states,
                        config.sensor_range_m,
                        config.sensor_fov_rad,
                        config.map_resolution_m,
                    )
                    agent.update_map(free, blocked)
                    if trace is not None:
                        trace.seen(tick, free)
            for rid, agent in agents.items():  # BELIEF + POLICY
                if tick < until[rid]:
                    continue
                key = f"{config.seed_policy}/{rid}/{tick}"
                action = agent.decide(tick, policies[rid], key)
                actions[rid], until[rid] = action, tick + action.duration_ticks
                for send in action.sends:
                    obs = agent.knows(send.evidence_id)
                    assert obs is not None  # checked by LocalAgent.decide
                    if obs.evidence_id.origin == rid:
                        frame = originate(obs, send.recipient, tick)
                    else:
                        hop = received_hop[(rid, obs.evidence_id)] + 1
                        if config.hop_limit is not None and hop > config.hop_limit:
                            self.hop_limited += 1
                            continue
                        frame = Message(rid, send.recipient, tick, hop, obs)
                    status = channel.submit(frame, tick)
                    if send.assumed_receiver_open is not None:
                        # Privileged diagnostics only: compare the sender's nominal receiver
                        # model with the receiver's true belief. Never fed back to agents.
                        true_open = agents[send.recipient].probability_open(obs.entity, tick)
                        self.recipient_model_log.append(
                            RecipientModelRecord(
                                tick,
                                rid,
                                send.recipient,
                                obs.entity,
                                send.assumed_receiver_open,
                                true_open,
                                status,
                            )
                        )
                if action.share_position:
                    x, y, _ = poses[rid]
                    for other in sorted(agents):
                        if other != rid:
                            sequence += 1
                            channel.submit(PositionMessage(rid, other, tick, x, y, sequence), tick)
                if action.reserve is not None and action.reserve != reserved[rid]:
                    reserved[rid] = action.reserve
                    for other in sorted(agents):
                        if other != rid:
                            sequence += 1
                            channel.submit(
                                ReservationMessage(
                                    rid,
                                    other,
                                    tick,
                                    action.reserve,
                                    tick + action.reserve_ttl_ticks,
                                    sequence,
                                ),
                                tick,
                            )
            while due and due[0].tick == tick:  # EVALUATE
                request = due.pop(0)
                p = agents[request.recipient].probability_open(request.entity, tick)
                decision = int(p >= config.decision_threshold)
                truth = states[request.entity]
                records.append(
                    EvaluationRecord(
                        config.episode_id,
                        config.building_id,
                        config.method_id,
                        tick,
                        request.recipient,
                        request.entity,
                        p,
                        truth,
                        decision,
                        binary_loss(
                            decision, truth, config.loss_false_open, config.loss_false_closed
                        ),
                        channel.stats.tx_attempt_bytes[request.recipient],
                    )
                )
        self.last_world_tick = world.tick
        self.last_channel = channel
        return records


@dataclass(frozen=True, slots=True)
class RecipientModelRecord:
    tick: int
    sender: str
    recipient: str
    entity: str
    assumed_open: float
    true_open: float
    submit_status: str

    @property
    def error(self) -> float:
        return self.assumed_open - self.true_open


class _Timeout(Exception):
    def __init__(self, records: list[EvaluationRecord]) -> None:
        super().__init__("wall-time limit reached")
        self.records = records


def build_manifest(
    config: EpisodeConfig, status: Status, requests: int, answered: int, error: str | None
) -> dict[str, Any]:
    import numpy

    return {
        "schema_version": MANIFEST_SCHEMA,
        "run_id": hashlib.sha256(
            f"{config.config_sha256}/{config.code_commit}".encode()
        ).hexdigest()[:16],
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "code_commit": config.code_commit,
        "argus_version": __version__,
        "config_sha256": config.config_sha256,
        "config": config.to_json(),
        "scenario_hash": config.scenario.scenario_hash,
        "scenario_generator": config.scenario.generator_version,
        "rng_schema": RNG_SCHEMA,
        "method_components": [
            *METHOD_SPECS[config.method_id].components,
            "binary-independent-sensor"
            if config.sensor_bias is None
            else "binary-shared-bias-sensor",
            "gilbert-elliott-channel",
        ],
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "platform": platform.platform(),
        "status": status,
        "requests": requests,
        "answered": answered,
        "error": error,
    }


def write_manifest(directory: Path, manifest: dict[str, Any]) -> Path:
    """Write manifest.json into a new directory; temp file then atomic rename."""
    if directory.exists():
        raise FileExistsError(f"output directory already exists: {directory}")
    directory.mkdir(parents=True)
    path = directory / "manifest.json"
    temporary = path.with_name("manifest.json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return path
