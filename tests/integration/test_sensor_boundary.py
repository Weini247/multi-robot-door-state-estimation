"""Information-boundary contract: import graph and public payload allowlists (E03-T03).

Typing is not a sandbox, so the boundary is checked structurally: an AST scan
of every module under src/argus against per-package import rules, and exact
field allowlists for every public value type that crosses from the simulator
to agents. Packages without files yet (agents, policies, ...) are covered as
soon as their first module appears.
"""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import pytest

from argus.core import types

SRC = Path(__file__).resolve().parents[2] / "src" / "argus"

PRIVILEGED = {
    "argus.world",
    "argus.sensors",
    "argus.experiments",
    "argus.evaluation",
    "argus.replay",
}

# package -> argus packages it may import (itself always allowed)
ALLOWED: dict[str, set[str]] = {
    "core": set(),
    "dynamics": {"argus.core"},
    "beliefs": {"argus.core", "argus.dynamics"},
    "communication": {"argus.core", "argus.dynamics"},
    "agents": {"argus.core", "argus.dynamics", "argus.beliefs", "argus.policies"},
    "policies": {"argus.core", "argus.dynamics", "argus.beliefs"},
    "world": {"argus.core", "argus.dynamics"},
    "sensors": {"argus.core", "argus.dynamics", "argus.world"},
    "learning": {"argus.core"},
}
# packages whose modules must never see simulator truth
UNPRIVILEGED = {"core", "dynamics", "beliefs", "communication", "agents", "policies", "learning"}

FORBIDDEN_FIELD_WORDS = ("truth", "bias", "group", "latent", "world", "cell", "hidden", "true_")


def argus_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module)
        elif isinstance(node, ast.ImportFrom) and node.level > 0:
            raise AssertionError(f"{path}: relative imports hide the boundary; use absolute ones")
    return {".".join(n.split(".")[:2]) for n in names if n.split(".")[0] == "argus"}


def modules() -> list[tuple[str, Path]]:
    found = []
    for path in sorted(SRC.rglob("*.py")):
        parts = path.relative_to(SRC).parts
        if len(parts) > 1:
            found.append((parts[0], path))
    return found


def test_every_package_has_an_import_rule() -> None:
    packages = {package for package, _ in modules()}
    assert packages <= set(ALLOWED) | {"experiments", "evaluation", "replay", "adapters"}


@pytest.mark.parametrize(("package", "path"), modules(), ids=lambda v: str(v))
def test_import_rules(package: str, path: Path) -> None:
    if package not in ALLOWED:
        return  # composition layers may wire everything
    used = argus_imports(path) - {f"argus.{package}", "argus"}
    assert used <= ALLOWED[package], f"{path.name} imports {sorted(used - ALLOWED[package])}"
    if package in UNPRIVILEGED:
        assert not used & PRIVILEGED, f"{path.name} reaches simulator truth"


def test_import_scanner_detects_violations(tmp_path: Path) -> None:
    bad = tmp_path / "bad.py"
    bad.write_text("from argus.world.scenario import WorldBackend2D\nimport argus.sensors.binary\n")
    assert argus_imports(bad) == {"argus.world", "argus.sensors"}
    relative = tmp_path / "rel.py"
    relative.write_text("from ..world import scenario\n")
    with pytest.raises(AssertionError):
        argus_imports(relative)


OBSERVATION_FIELDS = {
    "evidence_id",
    "observed_tick",
    "entity",
    "value",
    "observer_pose",
    "sensor_model",
}
EVIDENCE_ID_FIELDS = {"episode", "origin", "sequence"}


def test_public_types_match_their_allowlists() -> None:
    assert {f.name for f in dataclasses.fields(types.Observation)} == OBSERVATION_FIELDS
    assert {f.name for f in dataclasses.fields(types.EvidenceId)} == EVIDENCE_ID_FIELDS
    obs = types.Observation(types.EvidenceId("ep", "r0", 0), 0, "door", 1, (0.0, 0.0, 0.0), "m")
    assert set(obs.to_json()) == OBSERVATION_FIELDS


AGENT_VIEW_FIELDS = {
    "robot_id",
    "tick",
    "pose",
    "belief",
    "map_resolution_m",
    "free_map",
    "blocked_map",
    "inbox",
    "reservations",
    "remaining_bytes",
    "known_evidence",
    "last_collision",
    "sightings",
    "joint_beliefs",
    "new_own_evidence",
    "position_reports",
}


def test_agent_view_matches_its_allowlist() -> None:
    assert {f.name for f in dataclasses.fields(types.AgentView)} == AGENT_VIEW_FIELDS
    assert {f.name for f in dataclasses.fields(types.BeliefState)} == {
        "tick",
        "passage_probabilities",
    }
    assert {f.name for f in dataclasses.fields(types.Reservation)} == {
        "robot_id",
        "target",
        "expires_tick",
    }
    assert {f.name for f in dataclasses.fields(types.Sighting)} == {
        "entity",
        "observer_pose",
        "observed_tick",
        "origin",
    }
    assert {f.name for f in dataclasses.fields(types.Send)} == {
        "recipient",
        "evidence_id",
        "assumed_receiver_open",
    }


def public_dataclasses() -> list[type]:
    return [
        value
        for value in vars(types).values()
        if isinstance(value, type)
        and dataclasses.is_dataclass(value)
        and value.__module__ == types.__name__
    ]


@pytest.mark.parametrize("cls", public_dataclasses(), ids=lambda c: c.__name__)
def test_no_truth_or_group_fields_in_public_types(cls: type) -> None:
    for f in dataclasses.fields(cls):
        assert not any(word in f.name.lower() for word in FORBIDDEN_FIELD_WORDS), (cls, f.name)
        assert "Any" not in str(f.type) and "object" not in str(f.type), (cls, f.name)
