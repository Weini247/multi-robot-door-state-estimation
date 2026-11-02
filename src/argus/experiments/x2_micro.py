"""X2 micro: exact ablation of independence, cost and receiver history (E06-T03).

Reads the pre-registered configuration, evaluates every factor combination
with the implementation (expected_value, send_value) and returns raw rows. A
separate theory computation (tools/x2_report.py and tests) compares them with
exact closed forms. No sampling: every value is exact under the nominal model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from argus.beliefs.exact_micro import correct_joint, joint_prior
from argus.beliefs.fixed_lag import NominalPassageModel
from argus.dynamics.binary_ctmc import BinaryCTMC
from argus.policies.voi import CostWeights, Losses, ObserveAction, expected_value, send_value


def load(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if data.get("experiment") != "X2_micro" or data.get("schema_version") != 1:
        raise ValueError("not an X2_micro v1 configuration")
    return data


def run(config: dict[str, Any]) -> list[dict[str, Any]]:
    s = config["setting"]
    f = config["factors"]
    rows: list[dict[str, Any]] = []
    for bias_name, g_rates in f["bias"].items():
        model = NominalPassageModel(
            BinaryCTMC(*s["door_rates"]),
            BinaryCTMC(*g_rates),
            s["private_error"],
            s["prior_open"],
            s["bias_probability"],
        )
        after_first = correct_joint(
            joint_prior(s["prior_open"], s["bias_probability"]),
            s["first_reading"],
            s["private_error"],
        )
        for loss_name, (c10, c01) in f["losses"].items():
            losses = Losses(c10, c01)
            for cost_name, weights in f["costs"].items():
                costs = CostWeights(*weights)
                action = ObserveAction(s["distance_m"], s["reobservation_delay_s"], 1)
                est = expected_value(action, after_first, model, s["horizon_s"], losses, costs)
                rows.append(
                    {
                        "kind": "reobserve",
                        "bias": bias_name,
                        "losses": loss_name,
                        "costs": cost_name,
                        "gain": est.gain,
                        "cost": est.cost,
                        "value": est.voi,
                    }
                )
            prior = joint_prior(s["prior_open"], s["bias_probability"])
            decision = send_value(
                prior, after_first, [s["first_reading"]], model, s["horizon_s"], losses
            )
            for receiver in f["receiver"]:
                value = (
                    0.0 if receiver == "holds_reading" else (s["delivery_probability"] * decision)
                )
                rows.append(
                    {
                        "kind": "send",
                        "bias": bias_name,
                        "losses": loss_name,
                        "receiver": receiver,
                        "decision_value": decision,
                        "value": value,
                    }
                )
    return rows
