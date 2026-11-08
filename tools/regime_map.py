"""Regime map of the X1/X2 test batch (ARGUS-E08-T05). Exploratory, written after the
frozen analysis; it reuses the frozen estimators and bootstrap.

    python tools/regime_map.py --json docs/results/REGIME_MAP.json
        --report docs/results/REGIME_MAP.md --figure docs/figures/regime_map.svg

Per cell of the 2^3 design and per control: gain of the candidate in decision loss,
relative coverage loss and bytes, each with a paired cluster-bootstrap interval over the
80 id_test buildings. Intervals are not adjusted for multiplicity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from make_figures import STYLE, pct, svg

from argus.evaluation.statistics import DRAWS, PairedBootstrap, building_values
from argus.experiments.cells import CELLS

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "runs/x1x2_v1/results.jsonl"
CANDIDATE = "repair"
CONTROLS = ("cross_rank", "dynamics_ids", "no_comm", "age_revisit", "ttl_ids", "naive_gossip")
REPLICATES = 2


def load() -> tuple[list[dict[str, Any]], str]:
    data = RESULTS.read_bytes()
    rows = [json.loads(line) for line in data.decode("utf-8").splitlines() if line]
    rows = [r for r in rows if r["split"] == "id_test"]
    for r in rows:
        r.setdefault("coverage_closed_final", 0.0)
    return rows, hashlib.sha256(data).hexdigest()


def regime(rows: list[dict[str, Any]], draws: int = DRAWS) -> dict[str, Any]:
    family = {r["building_id"]: r["family"] for r in rows}
    boot = PairedBootstrap(sorted(family), family, draws)
    cells = [c.cell_id for c in CELLS]

    def values(method: str, field: str, cell: str) -> dict[str, float]:
        sub = [r for r in rows if r["cell_id"] == cell]
        return building_values(sub, method, field, [cell], REPLICATES)

    out: dict[str, Any] = {}
    for cell in cells:
        cand_loss = values(CANDIDATE, "mean_loss", cell)
        cand_cov = values(CANDIDATE, "coverage_closed_final", cell)
        entry: dict[str, Any] = {
            "loss": {
                m: boot.mean(values(m, "mean_loss", cell)).to_json() for m in (CANDIDATE, *CONTROLS)
            },
            "bytes": {
                m: boot.mean(values(m, "sent_bytes_total", cell)).to_json()
                for m in (CANDIDATE, *CONTROLS)
            },
            "gain": {},
            "coverage_loss": {},
        }
        for control in CONTROLS:
            entry["gain"][control] = boot.ratio_gain(
                cand_loss, values(control, "mean_loss", cell)
            ).to_json()
            entry["coverage_loss"][control] = boot.ratio_gain(
                cand_cov, values(control, "coverage_closed_final", cell)
            ).to_json()
        out[cell] = entry
    return out


def verdict(gain: dict[str, float], cov: dict[str, float]) -> str:
    """Reading of one cell against one control (descriptive, no claim)."""
    if gain["low"] >= 0.10 and cov["high"] <= 0.05:
        return "meets H2 rule"
    if gain["low"] > 0:
        return "better"
    if gain["high"] < 0:
        return "worse"
    return "no difference"


LABEL = {
    "sta-ind-direct": "static · indep. · direct",
    "sta-ind-relay": "static · indep. · relay",
    "sta-cor-direct": "static · corr. · direct",
    "sta-cor-relay": "static · corr. · relay",
    "dyn-ind-direct": "dynamic · indep. · direct",
    "dyn-ind-relay": "dynamic · indep. · relay",
    "dyn-cor-direct": "dynamic · corr. · direct",
    "dyn-cor-relay": "dynamic · corr. · relay",
}


def figure(table: dict[str, Any]) -> str:
    controls = ("cross_rank", "dynamics_ids", "no_comm", "ttl_ids")
    left, top, cw, rh = 210, 96, 132, 30
    width, height = left + cw * len(controls) + 20, top + rh * len(table) + 64
    fills = {
        "meets H2 rule": "var(--accent)",
        "better": "var(--zone)",
        "no difference": "var(--grid)",
        "worse": "var(--other)",
    }
    body = [
        '<text class="t" x="20" y="28">Regime map: repair against each control</text>',
        '<text class="s" x="20" y="48">Gain in decision loss per cell, 95 % CI over 80 buildings. '
        "Exploratory, not multiplicity-adjusted.</text>",
    ]
    for j, control in enumerate(controls):
        body.append(
            f'<text class="l" x="{left + cw * j + cw / 2}" y="{top - 14}" text-anchor="middle">'
            f"{control}</text>"
        )
    for i, (cell, entry) in enumerate(table.items()):
        y = top + rh * i
        body.append(
            f'<text class="l" x="{left - 12}" y="{y + 19}" text-anchor="end">{LABEL[cell]}</text>'
        )
        for j, control in enumerate(controls):
            g, c = entry["gain"][control], entry["coverage_loss"][control]
            v = verdict(g, c)
            opacity = "0.35" if v == "better" else "1"
            body.append(
                f'<rect x="{left + cw * j + 2}" y="{y + 2}" width="{cw - 4}" height="{rh - 4}" '
                f'rx="4" style="fill:{fills[v]};opacity:{opacity}"/>'
            )
            body.append(
                f'<text class="v" x="{left + cw * j + cw / 2}" y="{y + 19}" '
                f'text-anchor="middle">{pct(g["estimate"])}</text>'
            )
    legend_y = top + rh * len(table) + 30
    for k, (name, fill) in enumerate(fills.items()):
        x = 20 + 170 * k
        opacity = "0.35" if name == "better" else "1"
        body.append(
            f'<rect x="{x}" y="{legend_y - 11}" width="14" height="14" rx="3" '
            f'style="fill:{fill};opacity:{opacity}"/>'
        )
        body.append(f'<text class="s" x="{x + 20}" y="{legend_y}">{name}</text>')
    assert STYLE  # the shared stylesheet supplies the colour tokens
    return svg(width, height, body, "Regime map: repair against each control")


def markdown(table: dict[str, Any], sha: str) -> str:
    def iv(x: dict[str, float]) -> str:
        return f"{x['estimate']:+.3f} [{x['low']:+.3f}, {x['high']:+.3f}]"

    lines = [
        "# Regime map X1/X2 (exploratory)",
        "",
        f"Generated by `tools/regime_map.py` from `runs/x1x2_v1/results.jsonl` (sha256 `{sha}`), "
        f"id_test, 80 buildings, bootstrap {DRAWS} draws, seed 20261103. Not part of the frozen "
        "analysis; intervals are not adjusted for multiplicity.",
        "",
        "| Cell | vs cross_rank | vs dynamics_ids | vs no_comm | vs ttl_ids "
        "| coverage loss vs cross_rank |",
        "|---|---|---|---|---|---|",
    ]
    for cell, e in table.items():
        g = e["gain"]
        lines.append(
            f"| {cell} | {iv(g['cross_rank'])} | {iv(g['dynamics_ids'])} | {iv(g['no_comm'])} "
            f"| {iv(g['ttl_ids'])} | {iv(e['coverage_loss']['cross_rank'])} |"
        )
    lines += [
        "",
        "Losses, bytes and the remaining controls are in the JSON next to this report.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()
    rows, sha = load()
    table = regime(rows)
    report = {"source_sha256": sha, "draws": DRAWS, "seed": 20261103, "cells": table}
    args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.report.write_text(markdown(table, sha), encoding="utf-8")
    args.figure.write_bytes(figure(table).encode("utf-8"))
    print(f"wrote {args.json}, {args.report}, {args.figure}")


if __name__ == "__main__":
    main()
