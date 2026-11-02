"""Sender budget, link FIFO queues and delivery timing (M4, E04-T02).

A message submitted at tick k starts transmission at s = max(k + 1, link free),
occupies the link for ceil(L / (link_rate * tick_seconds)) >= 1 ticks and is
delivered after an additional propagation delay. Decisions therefore act at
k + 1 at the earliest.

Sender budget: for every window (t - W, t], the bytes of all transmission
starts of a sender must not exceed B * W * tick_seconds. A new start is checked
against every window that contains it. Bytes are charged at transmission
start, including frames that the loss model later drops; a rejection before
transmission (budget, queue memory, oversize) costs no bytes. Received bytes
are booked separately and never against the sender budget.

There are no ACKs or retries in the pilot; a broadcast is N-1 unicasts.
"""

from __future__ import annotations

import bisect
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Literal, Protocol

from argus.communication.messages import (
    Message,
    PoolMessage,
    PositionMessage,
    ReservationMessage,
)

Frame = Message | ReservationMessage | PoolMessage | PositionMessage

Link = tuple[str, str]
SubmitStatus = Literal["queued", "budget_rejected", "queue_rejected", "frame_too_large"]


class LossModel(Protocol):
    def drops(self, link: Link, tick: int, message_key: str) -> bool: ...


class NoLoss:
    def drops(self, link: Link, tick: int, message_key: str) -> bool:
        return False


@dataclass(frozen=True, slots=True)
class ChannelConfig:
    tick_seconds: float
    sender_bytes_per_second: float
    window_ticks: int
    link_bytes_per_second: float
    propagation_ticks: int
    queue_capacity_bytes: int

    def __post_init__(self) -> None:
        for name in ("tick_seconds", "sender_bytes_per_second", "link_bytes_per_second"):
            value = getattr(self, name)
            if not (math.isfinite(value) and value > 0):
                raise ValueError(f"{name} must be finite and > 0")
        for name in ("window_ticks", "queue_capacity_bytes"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be an int >= 1")
        if type(self.propagation_ticks) is not int or self.propagation_ticks < 0:
            raise ValueError("propagation_ticks must be an int >= 0")

    @property
    def budget_bytes(self) -> float:
        """Bytes a sender may start within any window of window_ticks ticks."""
        return self.sender_bytes_per_second * self.window_ticks * self.tick_seconds

    def service_ticks(self, size: int) -> int:
        return max(1, math.ceil(size / (self.link_bytes_per_second * self.tick_seconds)))


@dataclass(frozen=True, slots=True)
class Attempt:
    message: Frame
    link: Link
    size: int
    submitted: int
    start: int
    delivery: int
    dropped: bool
    key: str


@dataclass(slots=True)
class ChannelStats:
    tx_attempt_bytes: Counter[str] = field(default_factory=Counter)
    dropped_bytes: Counter[str] = field(default_factory=Counter)
    rx_bytes: Counter[str] = field(default_factory=Counter)
    rejects: Counter[tuple[str, str]] = field(default_factory=Counter)  # (sender, reason)


def message_key(message: Frame, start: int) -> str:
    return f"{message.frame_id}/{message.sender}>{message.recipient}/{start}"


class Channel:
    """All directed links of an episode; deterministic for a fixed submission order."""

    def __init__(self, config: ChannelConfig, loss: LossModel | None = None) -> None:
        self._config = config
        self._loss: LossModel = loss or NoLoss()
        self._starts: dict[str, list[tuple[int, int]]] = {}  # sender -> sorted (start, size)
        self._link_free: dict[Link, int] = {}
        self._attempts: list[Attempt] = []
        self._delivered: set[int] = set()
        self.stats = ChannelStats()

    @property
    def config(self) -> ChannelConfig:
        return self._config

    def remaining_budget(self, sender: str, tick: int) -> float:
        """Budget left for a transmission starting at `tick` (what a policy may know)."""
        return self._config.budget_bytes - self._window_bytes(sender, tick)

    def _window_bytes(self, sender: str, end: int) -> int:
        w = self._config.window_ticks
        return sum(size for start, size in self._starts.get(sender, []) if end - w < start <= end)

    def _fits_budget(self, sender: str, start: int, size: int) -> bool:
        w = self._config.window_ticks
        later = [s for s, _ in self._starts.get(sender, []) if start < s < start + w]
        return all(
            self._window_bytes(sender, end) + size <= self._config.budget_bytes
            for end in (start, *later)
        )

    def _waiting_bytes(self, link: Link, now: int) -> int:
        return sum(a.size for a in self._attempts if a.link == link and a.start > now)

    def submit(self, message: Frame, now: int) -> SubmitStatus:
        if message.sent_tick != now:
            raise ValueError(f"message sent_tick {message.sent_tick} != submission tick {now}")
        size = message.size_bytes
        link = (message.sender, message.recipient)
        reason: SubmitStatus | None = None
        if size > self._config.budget_bytes or size > self._config.queue_capacity_bytes:
            reason = "frame_too_large"
        elif self._waiting_bytes(link, now) + size > self._config.queue_capacity_bytes:
            reason = "queue_rejected"
        start = max(now + 1, self._link_free.get(link, 0))
        if reason is None and not self._fits_budget(message.sender, start, size):
            reason = "budget_rejected"
        if reason is not None:
            self.stats.rejects[(message.sender, reason)] += 1
            return reason
        service = self._config.service_ticks(size)
        self._link_free[link] = start + service
        key = message_key(message, start)
        dropped = self._loss.drops(link, start, key)
        delivery = start + service + self._config.propagation_ticks
        self._attempts.append(Attempt(message, link, size, now, start, delivery, dropped, key))
        bisect.insort(self._starts.setdefault(message.sender, []), (start, size))
        self.stats.tx_attempt_bytes[message.sender] += size
        if dropped:
            self.stats.dropped_bytes[message.sender] += size
        return "queued"

    def deliveries(self, tick: int) -> tuple[Frame, ...]:
        """Messages arriving at `tick`, in a deterministic order."""
        due = sorted(
            (
                i
                for i, a in enumerate(self._attempts)
                if a.delivery == tick and not a.dropped and i not in self._delivered
            ),
            key=lambda i: (self._attempts[i].link, self._attempts[i].start, self._attempts[i].key),
        )
        for i in due:
            self._delivered.add(i)
            self.stats.rx_bytes[self._attempts[i].message.recipient] += self._attempts[i].size
        return tuple(self._attempts[i].message for i in due)

    @property
    def attempts(self) -> tuple[Attempt, ...]:
        return tuple(self._attempts)
