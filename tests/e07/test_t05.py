"""ARGUS-E07-T05: Cross-rank CR-ESR_p adaptation against the published description."""

from __future__ import annotations

import math

import pytest

from argus.communication.messages import PositionMessage
from argus.core.types import AgentView, BeliefState, PositionReport
from argus.experiments.runner import ExperimentRunner
from argus.policies import crossrank_adapter
from argus.policies.baselines import METHOD_SPECS
from argus.policies.crossrank_adapter import (
    ADAPTATION,
    CrossRankPolicy,
    cross_rank,
    distance_to_segment,
    path_distance,
    spread_rank,
)
from tests.e05.test_t03 import config

# --- equation-level cases (Eqs. 6-8 as described) ------------------------------------------


def test_distance_to_segment_and_single_position() -> None:
    assert distance_to_segment((1.0, 1.0), (0.0, 0.0), (2.0, 0.0)) == pytest.approx(1.0)
    assert distance_to_segment((3.0, 0.0), (0.0, 0.0), (2.0, 0.0)) == pytest.approx(1.0)
    assert distance_to_segment((1.0, 1.0), (0.0, 0.0), (0.0, 0.0)) == pytest.approx(math.sqrt(2))
    assert path_distance((0.0, 3.0), [(0.0, 0.0)]) == pytest.approx(3.0)  # |H_k| == 1
    assert path_distance((0.0, 0.0), []) == math.inf


def test_cross_rank_counts_teammates_strictly_inside_the_threshold() -> None:
    histories = {"k1": [(0.0, 0.0), (4.0, 0.0)], "k2": [(0.0, 2.0)]}
    assert cross_rank((2.0, 1.0), histories, 1.5) == 1  # 1.0 from k1's segment, 2.24 from k2
    assert cross_rank((2.0, 1.0), histories, 2.5) == 2
    assert cross_rank((2.0, 1.5), histories, 1.5) == 0  # equality is not "closer than m_r"


def test_spread_rank_uses_euclidean_ratios() -> None:
    assert spread_rank((4.0, 0.0), (0.0, 0.0), {"k": (8.0, 0.0)}) == pytest.approx(1.0)
    assert spread_rank((4.0, 0.0), (2.0, 0.0), {"a": (8.0, 0.0), "b": (0.0, 0.0)}) == pytest.approx(
        2 / 4 + 2 / 4
    )


# --- "original task" fixture: the situation of Fig. 4/5 -------------------------------------


def corridor_view(reports: tuple[PositionReport, ...]) -> AgentView:
    """Robot at a crossroads; two branches up (x=1) and right (y=0.5) are frontiers."""
    free = {(i, j) for i in range(0, 4) for j in range(0, 3)}  # junction area
    free |= {(i, 2) for i in range(4, 12)}  # branch A to the right, upper lane
    free |= {(2, j) for j in range(3, 12)}  # branch B upwards
    return AgentView(
        "r1",
        10,
        (0.375, 0.375, 0.0),
        BeliefState(10, ()),
        0.25,
        tuple(sorted(free)),
        (),
        (),
        (),
        1e6,
        (),
        False,
        position_reports=reports,
    )


def test_waypoint_on_the_teammates_path_is_avoided() -> None:
    policy = CrossRankPolicy(m_r=1.0)
    alone = policy.choose(corridor_view(()))
    assert alone is not None
    # Teammate r2 has been reported travelling up branch B.
    reports = tuple(PositionReport("r2", t, 0.6, 0.8 + 0.25 * t) for t in range(10))
    informed = CrossRankPolicy(m_r=1.0)
    informed.act(corridor_view(reports), "k")
    goal = informed.last_choice
    assert goal is not None
    centre = ((goal[0] + 0.5) * 0.25, (goal[1] + 0.5) * 0.25)
    histories = {"r2": [(r.x, r.y) for r in reports]}
    assert cross_rank(centre, histories, 1.0) == 0  # chose an unvisited branch
    assert goal[1] == 2 and goal[0] >= 4  # in branch A, not in branch B (x = 2)
    branch_b = [c for c in corridor_view(()).free_map if c[0] == 2 and c[1] >= 3]
    assert all(
        cross_rank(((c[0] + 0.5) * 0.25, (c[1] + 0.5) * 0.25), histories, 1.0) == 1
        for c in branch_b
    )  # every waypoint of branch B is penalised


# --- ARGUS integration ---------------------------------------------------------------------


def test_marked_as_adaptation_and_registered_as_control() -> None:
    assert "adaptation" in ADAPTATION.lower()
    assert "ADAPTATION, not the original implementation" in (crossrank_adapter.__doc__ or "")
    spec = METHOD_SPECS["cross_rank"]
    assert spec.role == "control" and "position-sharing" in spec.components
    assert set(spec.tunable) == {"m_r", "position_filter_m"}


def test_changed_task_semantics_positions_only_and_same_byte_accounting() -> None:
    runner = ExperimentRunner()
    result = runner.run(config("cross_rank"))
    assert result.status == "ok", result.error
    attempts = runner.last_channel.attempts
    assert attempts and all(isinstance(a.message, PositionMessage) for a in attempts)
    assert all(a.size == len(a.message.encode()) for a in attempts)
    assert sum(runner.last_channel.stats.tx_attempt_bytes.values()) == sum(a.size for a in attempts)
    assert not any(r == "budget_rejected" for _, r in runner.last_channel.stats.rejects)


def test_spatial_filter_limits_position_frames() -> None:
    policy = CrossRankPolicy(position_filter_m=0.5)
    view = corridor_view(())
    assert policy.act(view, "k").share_position is True
    assert policy.act(view, "k").share_position is False  # has not moved
    moved = AgentView(
        **{**{f: getattr(view, f) for f in view.__dataclass_fields__}, "pose": (1.0, 0.375, 0.0)}
    )
    assert policy.act(moved, "k").share_position is True


def test_position_frames_are_validated() -> None:
    from argus.communication.messages import MessageError

    with pytest.raises(MessageError):
        PositionMessage("r0", "r0", 1, 0.0, 0.0, 1)
    with pytest.raises(MessageError):
        PositionMessage("r0", "r1", 1, math.nan, 0.0, 1)
