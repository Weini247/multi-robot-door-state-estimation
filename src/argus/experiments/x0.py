"""X0 — mathematical correctness: raw outputs of the exact micro references (E01-T04).

`export` evaluates fixed cases with the implementation and writes only the
implementation outputs. Oracles live outside the package (tools/x0_table.py,
tests), so the table compares two independent computations. Every case is
deterministic; there is no sampling and no Monte-Carlo value in X0.

Kinds: "exact" cases are evaluated under the model that generated them;
"misspecified" cases deliberately use a wrong nominal model and are reported
for contrast, never as an oracle; "error" cases record a required exception.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from argus import __version__
from argus.beliefs.exact_micro import ImpossibleEvidence, bayes_update, shared_bias_posterior
from argus.beliefs.joint_filter import JointFilter
from argus.dynamics.binary_ctmc import BinaryCTMC

SCHEMA_VERSION = 1
RAW_FILE = "x0_raw.json"


def _joint_run(
    prior: tuple[float, float, float, float],
    rates_z: tuple[float, float],
    rates_g: tuple[float, float],
    q: float,
    ys: tuple[int, ...],
    gap: float,
) -> list[float]:
    filt = JointFilter(BinaryCTMC(*rates_z), BinaryCTMC(*rates_g), q, prior)
    for k, y in enumerate(ys):
        if k:
            filt.predict(gap)
        filt.correct(y)
    return list(filt.joint)


def _case(
    case_id: str, model: str, kind: str, description: str, inputs: dict[str, Any], output: Any
) -> dict[str, Any]:
    return {
        "id": case_id,
        "model": model,
        "kind": kind,
        "description": description,
        "inputs": inputs,
        "output": output,
    }


def cases() -> list[dict[str, Any]]:
    """Evaluate all X0 cases with the implementation."""
    ctmc = BinaryCTMC(0.2, 0.3)
    result = [
        _case(
            "M1-01",
            "M1",
            "exact",
            "CTMC propagate, spec value .4+.5/e",
            {"rate_01": 0.2, "rate_10": 0.3, "p": 0.9, "dt_seconds": 2.0},
            ctmc.propagate(0.9, 2.0),
        ),
        _case(
            "M1-02",
            "M1",
            "exact",
            "CTMC long horizon reaches stationary .4",
            {"rate_01": 0.2, "rate_10": 0.3, "p": 0.9, "dt_seconds": 1000.0},
            ctmc.propagate(0.9, 1000.0),
        ),
        _case(
            "M1-03",
            "M1",
            "exact",
            "CTMC zero rates keep p",
            {"rate_01": 0.0, "rate_10": 0.0, "p": 0.73, "dt_seconds": 100.0},
            BinaryCTMC(0.0, 0.0).propagate(0.73, 100.0),
        ),
        _case(
            "M1-04",
            "M1",
            "exact",
            "CTMC tiny step keeps its increment",
            {"rate_01": 0.2, "rate_10": 0.3, "p": 0.0, "dt_seconds": 1e-12},
            ctmc.propagate(0.0, 1e-12),
        ),
        _case(
            "M1-05",
            "M1",
            "exact",
            "CTMC transition matrix",
            {"rate_01": 0.2, "rate_10": 0.3, "dt_seconds": 2.0},
            [list(row) for row in ctmc.transition(2.0)],
        ),
        _case(
            "M2-01",
            "M2/M3",
            "exact",
            "shared bias, one measurement y=1",
            {"prior_open": 0.5, "rho": 0.1, "q": 0.05, "observations": [1]},
            shared_bias_posterior(0.5, 0.1, 0.05, [1]),
        ),
        _case(
            "M2-02",
            "M2/M3",
            "exact",
            "shared bias, two distinct measurements y=1,1",
            {"prior_open": 0.5, "rho": 0.1, "q": 0.05, "observations": [1, 1]},
            shared_bias_posterior(0.5, 0.1, 0.05, [1, 1]),
        ),
        _case(
            "M2-03",
            "M2/M3",
            "exact",
            "shared bias, conflicting measurements y=1,0",
            {"prior_open": 0.5, "rho": 0.1, "q": 0.05, "observations": [1, 0]},
            shared_bias_posterior(0.5, 0.1, 0.05, [1, 0]),
        ),
        _case(
            "M2-04",
            "M2/M3",
            "misspecified",
            "naive independence with the same marginal error e=.14 (overconfident)",
            {"prior_open": 0.5, "error": 0.14, "observations": [1, 1]},
            bayes_update((0.5, 0.5), (0.14**2, 0.86**2))[1],
        ),
    ]
    try:
        shared_bias_posterior(0.5, 0.0, 0.0, [1, 0])
        impossible: Any = "no exception"
    except ImpossibleEvidence as error:
        impossible = f"ImpossibleEvidence: {error}"
    result.append(
        _case(
            "M2-05",
            "M2/M3",
            "error",
            "contradicting readings of a perfect unbiased sensor",
            {"prior_open": 0.5, "rho": 0.0, "q": 0.0, "observations": [1, 0]},
            impossible,
        )
    )
    joint = {
        "prior": [0.45, 0.05, 0.45, 0.05],
        "rates_z": [0.2, 0.3],
        "rates_g": [0.01, 0.09],
        "q": 0.05,
        "observations": [1, 0, 1],
        "gap_seconds": 0.7,
    }
    result += [
        _case(
            "M3-01",
            "M3",
            "exact",
            "joint (z,g) filter, three measurements",
            joint,
            _joint_run((0.45, 0.05, 0.45, 0.05), (0.2, 0.3), (0.01, 0.09), 0.05, (1, 0, 1), 0.7),
        ),
        _case(
            "M3-02",
            "M3",
            "exact",
            "joint filter static limit equals 325/362",
            {**joint, "rates_z": [0.0, 0.0], "rates_g": [0.0, 0.0], "observations": [1, 1]},
            _joint_run((0.45, 0.05, 0.45, 0.05), (0.0, 0.0), (0.0, 0.0), 0.05, (1, 1), 0.7),
        ),
    ]
    return result


def export(output: Path) -> Path:
    """Write the raw X0 outputs into a new directory; refuse to overwrite."""
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    output.mkdir(parents=True)
    path = output / RAW_FILE
    document = {"schema_version": SCHEMA_VERSION, "argus_version": __version__, "cases": cases()}
    path.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path
