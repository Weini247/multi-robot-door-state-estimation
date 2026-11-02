"""Batch execution with resource caps and sample-size planning (ARGUS-E08-T01).

`BatchRunner` runs one process per episode, at most `workers` at a time, and
works group by group (a group is one building, the statistical unit). A group
starts only if the projected wall and CPU time still fit the caps, so a stop
leaves complete buildings. Usage is persisted in `usage.json` next to the
results, so a paused and resumed batch is checked against its total, not only
against the current session. A process that exceeds the hard timeout is killed
and kept as a "timeout" row; nothing is dropped.

Pause: creating the file PAUSE in the output directory stops launching new
episodes; running ones finish and are written, then the batch returns with
paused=True. A rerun continues with the missing keys. A results line that was
cut off by a hard stop is moved to results.truncated and its episode reruns.

Strict mode (E08-T02 test batch): an exception in the worker outside the
episode, a dead worker and a hard-timeout kill are infrastructure events, not
results. They go to infra.jsonl (nothing is lost), their keys stay open and the
batch stops for a human decision. The batch also stops when more than
`breaker_failures` of the last `breaker_window` result rows are not "ok". A LOCK
file prevents two runners on one output directory. Child processes get one BLAS
thread each.

`plan_test_size` turns a pilot into the preregistered prefix length
B = max(60, ceil((2 * 1.96 * sd / 0.05)^2)) and into planned episodes and CPU
hours. It reports what the measurement implies; it makes no power claim.
"""

from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from multiprocessing.connection import Connection, wait
from pathlib import Path
from typing import Any

Worker = Callable[[Any], dict[str, Any]]

USAGE_FILE = "usage.json"
RESULTS_FILE = "results.jsonl"
PAUSE_FILE = "PAUSE"
TRUNCATED_FILE = "results.truncated"
INFRA_FILE = "infra.jsonl"
LOCK_FILE = "LOCK"
INFRA_STATUS = "infra_error"
THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


@dataclass(frozen=True, slots=True)
class BatchCaps:
    wall_hours: float
    cpu_hours: float
    workers: int
    episode_timeout_s: float  # hard kill; the runner's own soft limit should be lower

    def __post_init__(self) -> None:
        if min(self.wall_hours, self.cpu_hours, self.episode_timeout_s) <= 0 or self.workers < 1:
            raise ValueError("caps must be positive")


@dataclass(frozen=True, slots=True)
class BatchJob:
    key: str  # unique; a resumed batch skips keys already in the results
    group: str  # statistical unit (building)
    payload: Any  # picklable argument of the worker


@dataclass(slots=True)
class BatchUsage:
    wall_s: float = 0.0
    cpu_s: float = 0.0
    sessions: int = 0
    groups: dict[str, tuple[float, float]] = field(default_factory=dict)  # wall, cpu

    @classmethod
    def load(cls, path: Path) -> BatchUsage:
        if not path.exists():
            return cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        groups = {k: (float(v[0]), float(v[1])) for k, v in data["groups"].items()}
        return cls(float(data["wall_s"]), float(data["cpu_s"]), int(data["sessions"]), groups)

    def save(self, path: Path) -> None:
        data = {
            "wall_s": round(self.wall_s, 3),
            "cpu_s": round(self.cpu_s, 3),
            "sessions": self.sessions,
            "groups": {k: [round(w, 3), round(c, 3)] for k, (w, c) in sorted(self.groups.items())},
        }
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(path)


@dataclass(frozen=True, slots=True)
class BatchSummary:
    completed_groups: tuple[str, ...]
    stopped_before: str | None
    ran: int
    skipped: int
    usage: BatchUsage
    paused: bool = False
    stop_reason: str | None = None


def peak_rss_bytes() -> int | None:
    """Peak resident set of the calling process (Windows working set, POSIX maxrss)."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        kernel = ctypes.WinDLL("kernel32")
        psapi = ctypes.WinDLL("psapi")
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(Counters),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        handle = kernel.GetCurrentProcess()
        if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return None
        return int(counters.PeakWorkingSetSize)
    try:
        import resource
    except ImportError:  # pragma: no cover - neither Windows nor POSIX
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def _child(conn: Connection, worker: Worker, payload: Any) -> None:
    started = time.process_time()
    try:
        row = dict(worker(payload))
    except Exception as failure:  # noqa: BLE001 - reported as an error row
        row = {
            "status": "error",
            "error": f"{type(failure).__name__}: {failure}",
            "worker_exception": True,
        }
    row["cpu_s"] = round(time.process_time() - started, 3)
    row["peak_rss_bytes"] = peak_rss_bytes()
    conn.send(row)
    conn.close()


def probe_worker(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Diagnostic worker: sleep, allocate, burn CPU or fail on request (tests, overhead)."""
    block = bytearray(int(payload.get("alloc_mb", 0)) * 1024 * 1024)
    block[::4096] = b"\x01" * len(block[::4096])
    deadline = time.process_time() + float(payload.get("cpu_s", 0.0))
    while time.process_time() < deadline:
        pass
    time.sleep(float(payload.get("sleep_s", 0.0)))
    if payload.get("touch"):
        Path(payload["touch"]).touch()
    if payload.get("fail"):
        raise RuntimeError("probe failure")
    return {"status": payload.get("status", "ok"), "allocated_mb": len(block) // (1024 * 1024)}


class BatchRunner:
    def __init__(
        self,
        out: Path,
        caps: BatchCaps,
        worker: Worker,
        on_kill: Worker | None = None,
        strict: bool = False,
        breaker_failures: int = 3,
        breaker_window: int = 20,
    ) -> None:
        """on_kill(payload) runs in the parent for an episode killed by the hard timeout
        and returns extra fields for its row (e.g. the failure-rule loss)."""
        self.out = out
        self.caps = caps
        self.worker = worker
        self.on_kill = on_kill
        self.strict = strict
        self.breaker_failures = breaker_failures
        self.breaker_window = breaker_window
        self.results_path = out / RESULTS_FILE
        self.usage_path = out / USAGE_FILE

    def done_keys(self) -> set[str]:
        if not self.results_path.exists():
            return set()
        text = self.results_path.read_text(encoding="utf-8")
        lines = text.splitlines()
        keys = []
        for index, line in enumerate(lines):
            if not line:
                continue
            try:
                keys.append(json.loads(line)["key"])
            except (json.JSONDecodeError, KeyError):
                if index != len(lines) - 1:
                    raise ValueError(f"corrupt results line {index + 1}") from None
                with (self.out / TRUNCATED_FILE).open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
                kept = "".join(f"{x}\n" for x in lines[:index] if x)
                tmp = self.results_path.with_suffix(".tmp")
                tmp.write_text(kept, encoding="utf-8")
                tmp.replace(self.results_path)
                text = kept
        if text and not text.endswith("\n"):  # complete last line cut before its newline
            with self.results_path.open("a", encoding="utf-8") as handle:
                handle.write("\n")
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate keys in the results")
        return set(keys)

    def killed_keys(self) -> set[str]:
        path = self.out / INFRA_FILE
        if not path.exists():
            return set()
        rows = (json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x)
        return {r["key"] for r in rows if r.get("killed")}

    @property
    def pause_requested(self) -> bool:
        return (self.out / PAUSE_FILE).exists()

    def run(self, jobs: Sequence[BatchJob]) -> BatchSummary:
        keys = [j.key for j in jobs]
        if len(set(keys)) != len(keys):
            raise ValueError("job keys must be unique")
        self.out.mkdir(parents=True, exist_ok=True)
        if self.pause_requested:
            raise RuntimeError(f"remove {self.out / PAUSE_FILE} before resuming")
        lock = self.out / LOCK_FILE
        try:
            with lock.open("x", encoding="utf-8") as handle:
                handle.write(f"{os.getpid()}\n")
        except FileExistsError:
            raise RuntimeError(
                f"{lock} exists: another runner is active, or remove it after a crash"
            ) from None
        for name in THREAD_ENV:
            os.environ.setdefault(name, "1")
        try:
            return self._run(jobs)
        finally:
            lock.unlink(missing_ok=True)

    def _run(self, jobs: Sequence[BatchJob]) -> BatchSummary:
        usage = BatchUsage.load(self.usage_path)
        usage.sessions += 1
        usage.save(self.usage_path)
        done = self.done_keys()
        groups: dict[str, list[BatchJob]] = defaultdict(list)
        for job in jobs:
            groups[job.group].append(job)
        wall_cap, cpu_cap = self.caps.wall_hours * 3600, self.caps.cpu_hours * 3600
        session_start, session_wall_base = time.time(), usage.wall_s
        completed, ran, skipped, stopped = [], 0, 0, None
        recent: list[bool] = []
        killed_before = self.killed_keys()
        self.stop_requested = False
        for group, members in groups.items():
            todo = [j for j in members if j.key not in done]
            skipped += len(members) - len(todo)
            if not todo:
                completed.append(group)
                continue
            if usage.groups:
                wall_est = max(w for w, _ in usage.groups.values())
                cpu_est = max(c for _, c in usage.groups.values())
                if usage.wall_s + wall_est > wall_cap or usage.cpu_s + cpu_est > cpu_cap:
                    stopped = group
                    break
            t0, c0 = time.time(), usage.cpu_s
            reason = None
            for row in self._execute(todo):
                usage.cpu_s += float(row["cpu_s"])
                usage.wall_s = session_wall_base + time.time() - session_start
                # A key killed a second time is a reproducible hang of the method, not an
                # infrastructure event: it becomes a failure row (amendment 01, O4).
                repeat_kill = bool(row.get("killed")) and row["key"] in killed_before
                infra = (
                    self.strict
                    and not repeat_kill
                    and (
                        row.get("worker_exception")
                        or row.get("killed")
                        or row["status"] == INFRA_STATUS
                    )
                )
                target = self.out / INFRA_FILE if infra else self.results_path
                with target.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                usage.save(self.usage_path)
                if infra:
                    reason = reason or f"infrastructure event at {row['key']}: {row.get('error')}"
                    self.stop_requested = True
                    continue
                ran += 1
                recent.append(row["status"] != "ok")
                window = recent[-self.breaker_window :]
                if self.strict and sum(window) > self.breaker_failures:
                    reason = reason or f"breaker: {sum(window)} failures in last {len(window)}"
                    self.stop_requested = True
            usage.wall_s = session_wall_base + time.time() - session_start
            if reason is not None:
                usage.save(self.usage_path)
                return BatchSummary(tuple(completed), group, ran, skipped, usage, False, reason)
            if self.pause_requested:
                usage.save(self.usage_path)
                return BatchSummary(tuple(completed), group, ran, skipped, usage, True, "paused")
            previous = usage.groups.get(group, (0.0, 0.0))
            usage.groups[group] = (
                previous[0] + time.time() - t0,
                previous[1] + usage.cpu_s - c0,
            )
            usage.save(self.usage_path)
            completed.append(group)
        return BatchSummary(
            tuple(completed),
            stopped,
            ran,
            skipped,
            usage,
            False,
            None if stopped is None else "budget",
        )

    def _execute(self, jobs: Iterable[BatchJob]) -> Iterable[dict[str, Any]]:
        ctx = mp.get_context("spawn")
        pending = list(jobs)[::-1]
        active: dict[Any, tuple[Any, BatchJob, float]] = {}  # PipeConnection on Windows
        while pending or active:
            if pending and (self.pause_requested or getattr(self, "stop_requested", False)):
                pending.clear()  # running episodes finish; the rest waits for a rerun
            while pending and len(active) < self.caps.workers:
                job = pending.pop()
                parent, child = ctx.Pipe(duplex=False)
                proc = ctx.Process(target=_child, args=(child, self.worker, job.payload))
                proc.start()
                child.close()
                active[parent] = (proc, job, time.time())
            for ready in wait(list(active), timeout=0.5):
                pipe: Any = next(c for c in active if c is ready)
                proc, job, started = active.pop(pipe)
                try:
                    row = dict(pipe.recv())
                except EOFError:
                    row = {
                        "status": "error",
                        "error": "worker exited without a result",
                        "worker_exception": True,
                    }
                    row["cpu_s"] = round(time.time() - started, 3)
                    row["peak_rss_bytes"] = None
                pipe.close()
                proc.join()
                yield self._finish(job, row, time.time() - started)
            now = time.time()
            for conn, (proc, job, started) in list(active.items()):
                if now - started > self.caps.episode_timeout_s:
                    proc.kill()
                    proc.join()
                    conn.close()
                    del active[conn]
                    elapsed = now - started
                    row = {
                        "status": "timeout",
                        "error": f"killed after {elapsed:.0f}s (batch hard limit)",
                        "cpu_s": round(elapsed, 3),  # upper bound: one core for the whole time
                        "peak_rss_bytes": None,
                        "killed": True,
                    }
                    if self.on_kill is not None:
                        row = {**self.on_kill(job.payload), **row}
                    yield self._finish(job, row, elapsed)

    @staticmethod
    def _finish(job: BatchJob, row: dict[str, Any], wall: float) -> dict[str, Any]:
        row["key"] = job.key
        row["group"] = job.group
        row["process_wall_s"] = round(wall, 3)
        return row


def protocol_episode(payload: Mapping[str, Any]) -> dict[str, Any]:
    """One preregistered episode (split entry, cell, method, config, replicate)."""
    from argus.experiments.cells import CELLS
    from argus.experiments.protocol import episode_config
    from argus.experiments.runner import ExperimentRunner
    from argus.experiments.splits import SplitEntry
    from argus.world.floorplan import FloorplanParams

    params = dict(payload["params"])
    params["passage_rates"] = tuple(params["passage_rates"])
    cell = next(c for c in CELLS if c.cell_id == payload["cell_id"])
    entry = SplitEntry(*payload["entry"])
    if entry.split in ("id_test", "ood_test"):  # test episodes only via runset.test_episode
        raise ValueError(f"{entry.building_id}: protocol_episode never runs test buildings")
    config = episode_config(
        entry,
        FloorplanParams(**params),
        cell,
        payload["method_id"],
        payload["config_id"],
        payload["method_params"],
        payload["replicate"],
    )
    result = ExperimentRunner().run(config)
    ok = result.status == "ok" and bool(result.records)
    loss = (
        math.fsum(r.decision_loss for r in result.records) / len(result.records)
        if ok
        else float(payload["failure_loss"])
    )
    return {
        "episode_id": config.episode_id,
        "building_id": entry.building_id,
        "family": entry.family,
        "cell_id": cell.cell_id,
        "method_id": payload["method_id"],
        "config_id": payload["config_id"],
        "replicate": payload["replicate"],
        "status": result.status,
        "mean_loss": loss,
        "sent_bytes": max((r.sent_bytes for r in result.records), default=0),
        "requests": len(result.records) + result.unanswered,
        "wall_s": round(result.wall_seconds, 3),
        "error": result.error,
    }


# --- sample-size planning -----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SizePlan:
    buildings: int
    families: dict[str, int]
    gain: float  # pilot point estimate; calibration data, not evidence
    sd_pooled: float
    sd_stratified: float
    sd_used: float
    b_formula: float
    b_planned: int
    ci_half_width_at_b: float


def building_losses(rows: Iterable[Mapping[str, Any]], method: str, cells: int) -> dict[str, float]:
    """Mean loss per building over cells and replicates; incomplete buildings are left out."""
    per: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if r["method_id"] == method:
            per[r["building_id"]].append(float(r["mean_loss"]))
    counts = {len(v) for v in per.values()}
    full = max(counts, default=0)
    if full % cells:
        raise ValueError("rows per building are not a multiple of the cell count")
    return {b: math.fsum(v) / len(v) for b, v in per.items() if len(v) == full}


def _sd(values: Sequence[float]) -> float:
    mean = math.fsum(values) / len(values)
    return math.sqrt(math.fsum((v - mean) ** 2 for v in values) / (len(values) - 1))


def plan_test_size(
    candidate: Mapping[str, float],
    control: Mapping[str, float],
    family: Mapping[str, str],
    minimum: int = 60,
    width: float = 0.05,
    z: float = 1.96,
) -> SizePlan:
    """Preregistered B from the delta-method influence of the ratio gain
    g = 1 - mean L_cand / mean L_ctrl, psi_b = -(L_cand,b - (1 - g) L_ctrl,b) / mean L_ctrl.
    Its sd is sigma_Delta on the gain scale, so the CI of g has full width about
    2 * z * sd / sqrt(B). The larger of the pooled and the family-stratified sd is used."""
    buildings = sorted(set(candidate) & set(control))
    if len(buildings) < 4:
        raise ValueError("pilot needs at least four paired buildings")
    cand = [candidate[b] for b in buildings]
    ctrl = [control[b] for b in buildings]
    mean_ctrl = math.fsum(ctrl) / len(ctrl)
    if mean_ctrl <= 0:
        raise ValueError("control loss is zero; the relative gain is undefined")
    ratio = (math.fsum(cand) / len(cand)) / mean_ctrl
    psi = {b: -(candidate[b] - ratio * control[b]) / mean_ctrl for b in buildings}
    pooled = _sd(list(psi.values()))
    by_family: dict[str, list[float]] = defaultdict(list)
    for b in buildings:
        by_family[family[b]].append(psi[b])
    # 50/50 stratification: var = mean of the within-family variances.
    strata = [_sd(v) ** 2 for v in by_family.values() if len(v) > 1]
    stratified = math.sqrt(math.fsum(strata) / len(strata)) if strata else pooled
    used = max(pooled, stratified)
    b_formula = (2 * z * used / width) ** 2
    planned = max(minimum, math.ceil(b_formula))
    planned += planned % 2  # even prefixes are family-balanced
    return SizePlan(
        len(buildings),
        {f: len(v) for f, v in sorted(by_family.items())},
        1 - ratio,
        pooled,
        stratified,
        used,
        b_formula,
        planned,
        z * used / math.sqrt(planned),
    )


def plan_cost(
    buildings: int,
    cpu_s_per_episode: Mapping[str, float],
    cells: int,
    replicates: int,
    splits: int,
    wall_per_cpu: float,
) -> dict[str, float]:
    """Planned episodes and hours of a test batch from measured per-method CPU seconds."""
    per_unit = buildings * cells * replicates * splits
    episodes = per_unit * len(cpu_s_per_episode)
    cpu_h = per_unit * math.fsum(cpu_s_per_episode.values()) / 3600
    return {"episodes": float(episodes), "cpu_hours": cpu_h, "wall_hours": cpu_h * wall_per_cpu}
