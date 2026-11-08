# ruff: noqa: E501  (SVG markup is clearer on one line)
"""Figure of study 2: gain against no_comm per condition, by arm (ARGUS-E09-T02).

python tools/s2_figure.py docs/results/S2_v1.json docs/figures/s2_gain_by_condition.svg
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from make_figures import pct, svg

ARMS = (
    ("repair_observe/no_comm", "re-observe only", "pt", "ci"),
    ("repair_send/no_comm", "send only", "pt2", "ci2"),
)


def figure(report: dict) -> str:
    cells = report["descriptive"]["per_condition"]
    names = list(cells)
    left, top, rh = 150, 80, 34
    width, plot_w = 760, 440
    height = top + rh * len(names) + 60
    lo, hi = -0.04, 0.10

    def x(v: float) -> float:
        return left + (v - lo) / (hi - lo) * plot_w

    body = [
        '<text class="t" x="20" y="28">Study 2: gain against no_comm per condition</text>',
        '<text class="s" x="20" y="48">Door tempo s and cost ratio c; window W in seconds. '
        "95 % CI over 100 buildings, unadjusted.</text>",
    ]
    for t in (-0.02, 0.0, 0.02, 0.04, 0.06, 0.08, 0.10):
        cls = "zero" if t == 0 else "grid"
        body.append(
            f'<line class="{cls}" x1="{x(t):.1f}" y1="{top - 10}" x2="{x(t):.1f}" y2="{top + rh * len(names) - 10}"/>'
        )
        body.append(
            f'<text class="ax" x="{x(t):.1f}" y="{top + rh * len(names) + 8}" text-anchor="middle">{pct(t)}</text>'
        )
    for i, name in enumerate(names):
        cy = top + rh * i + 4
        window = cells[name]["window_s"]
        label = f"{name}  (W {window:.0f} s)" if window != float("inf") else f"{name}  (static)"
        body.append(
            f'<text class="l" x="{left - 12}" y="{cy + 4}" text-anchor="end">{label}</text>'
        )
        for k, (key, _, pt, ci) in enumerate(ARMS):
            g = cells[name]["gain"][key]
            yy = cy - 6 + 12 * k
            body.append(
                f'<line class="{ci}" x1="{x(g["low"]):.1f}" y1="{yy}" x2="{x(g["high"]):.1f}" y2="{yy}"/>'
            )
            body.append(f'<circle class="{pt}" cx="{x(g["estimate"]):.1f}" cy="{yy}" r="4.5"/>')
    legend = top + rh * len(names) + 34
    for k, (_, label, pt, _) in enumerate(ARMS):
        lx = left + 200 * k
        body.append(f'<circle class="{pt}" cx="{lx}" cy="{legend - 4}" r="5"/>')
        body.append(f'<text class="s" x="{lx + 12}" y="{legend}">{label}</text>')
    return svg(width, height, body, "Study 2: gain against no_comm per condition")


if __name__ == "__main__":
    report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    Path(sys.argv[2]).write_bytes(figure(report).encode("utf-8"))
    print("wrote", sys.argv[2])
