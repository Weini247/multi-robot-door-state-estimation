"""Trivial reference policies (controls for runner and evaluation tests)."""

from __future__ import annotations

from dataclasses import dataclass

from argus.core.types import AgentAction, AgentView


@dataclass(frozen=True, slots=True)
class WaitPolicy:
    """Never moves and never sends: the no-exploration control."""

    decision_ticks: int = 10

    def act(self, view: AgentView, random_key: str) -> AgentAction:
        return AgentAction("wait", self.decision_ticks)
