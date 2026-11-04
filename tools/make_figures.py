# ruff: noqa: E501  (SVG markup is clearer on one line)
"""Draw the README figures from docs/results/X1_X2_TEST_v1.json (stdlib only, SVG).

    python tools/make_figures.py

Writes docs/figures/*.svg. Every number drawn comes from the result file; the
SVGs carry no metadata. Colors follow the viewer's light or dark scheme.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "docs/results/X1_X2_TEST_v1.json"
OUT = ROOT / "docs/figures"

STYLE = """<style>
svg { --surface: #fcfcfb; --ink: #0b0b0b; --ink2: #52514e; --muted: #8a8984;
  --grid: #e4e3df; --accent: #2a78d6; --zone: #2a78d6; --other: #6f6e69; }
@media (prefers-color-scheme: dark) {
  svg { --surface: #1a1a19; --ink: #f0efec; --ink2: #c3c2b7; --muted: #8f8e86;
    --grid: #33332f; --accent: #3987e5; --zone: #3987e5; --other: #a3a299; }
}
.bg { fill: var(--surface); }
text { font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; fill: var(--ink); }
.t { font-size: 15px; font-weight: 600; }
.s { font-size: 12px; fill: var(--ink2); }
.l { font-size: 12.5px; }
.v { font-size: 11.5px; fill: var(--ink2); font-variant-numeric: tabular-nums; }
.ax { font-size: 11px; fill: var(--muted); font-variant-numeric: tabular-nums; }
.grid { stroke: var(--grid); stroke-width: 1; }
.zero { stroke: var(--muted); stroke-width: 1; }
.thr { stroke: var(--ink2); stroke-width: 1.5; stroke-dasharray: 5 4; }
.zone { fill: var(--zone); opacity: 0.10; }
.ci { stroke: var(--accent); stroke-width: 2.5; stroke-linecap: round; }
.pt { fill: var(--accent); stroke: var(--surface); stroke-width: 2; }
.ci2 { stroke: var(--other); stroke-width: 2; stroke-linecap: round; }
.pt2 { fill: var(--other); stroke: var(--surface); stroke-width: 2; }
</style>"""


def svg(width: int, height: int, body: list[str], title: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="{title}">\n'
        f"<title>{title}</title>\n{STYLE}\n"
        f'<rect class="bg" width="{width}" height="{height}" rx="8"/>\n'
        + "\n".join(body)
        + "\n</svg>\n"
    )


def pct(x: float) -> str:
    return f"{x * 100:+.1f} %".replace("-", "−")


def forest(
    rows: list[tuple[str, dict]],
    lo: float,
    hi: float,
    ticks: list[float],
    title: str,
    subtitle: str,
    threshold: tuple[float, str] | None,
    width: int = 760,
) -> str:
    """Point estimate with 95 % CI per row, a zero line and an optional threshold."""
    left, right, top, row_h = 230, 170, 84, 34
    plot_w = width - left - right
    height = top + row_h * len(rows) + 46

    def x(v: float) -> float:
        return left + (v - lo) / (hi - lo) * plot_w

    body = [
        f'<text class="t" x="20" y="28">{title}</text>',
        f'<text class="s" x="20" y="48">{subtitle}</text>',
    ]
    y0, y1 = top - 8, top + row_h * len(rows) - 8
    for t in ticks:
        body.append(f'<line class="grid" x1="{x(t):.1f}" y1="{y0}" x2="{x(t):.1f}" y2="{y1}"/>')
        body.append(
            f'<text class="ax" x="{x(t):.1f}" y="{y1 + 18}" text-anchor="middle">'
            + f"{t * 100:.0f} %".replace("-", "−")
            + "</text>"
        )
    body.append(f'<line class="zero" x1="{x(0):.1f}" y1="{y0}" x2="{x(0):.1f}" y2="{y1}"/>')
    if threshold:
        value, label = threshold
        body.append(
            f'<rect class="zone" x="{x(value):.1f}" y="{y0}" width="{x(hi) - x(value):.1f}" height="{y1 - y0}"/>'
        )
        body.append(
            f'<line class="thr" x1="{x(value):.1f}" y1="{y0}" x2="{x(value):.1f}" y2="{y1}"/>'
        )
        body.append(f'<text class="v" x="{x(value) + 6:.1f}" y="{y0 - 6}">{label}</text>')
    for i, (label, est) in enumerate(rows):
        cy = top + row_h * i + 8
        body.append(
            f'<text class="l" x="{left - 14}" y="{cy + 4}" text-anchor="end">{label}</text>'
        )
        body.append(
            f'<line class="ci" x1="{x(est["low"]):.1f}" y1="{cy}" x2="{x(est["high"]):.1f}" y2="{cy}"/>'
        )
        body.append(f'<circle class="pt" cx="{x(est["estimate"]):.1f}" cy="{cy}" r="5.5"/>')
        body.append(
            f'<text class="v" x="{width - right + 14}" y="{cy + 4}">{pct(est["estimate"])}  '
            f"[{pct(est['low'])}, {pct(est['high'])}]</text>"
        )
    return svg(width, height, body, title)


def h2(primary: dict) -> str:
    gain = primary["h2_loss"]["gain"]
    cov = primary["h2_coverage"]["gain"]  # relative coverage loss
    left, right, top = 230, 170, 70
    width, row_h = 760, 64
    plot_w = width - left - right
    lo, hi = -0.05, 0.20
    height = top + row_h * 2 + 40

    def x(v: float) -> float:
        return left + (v - lo) / (hi - lo) * plot_w

    body = [
        '<text class="t" x="20" y="28">H2: the two preregistered conditions</text>',
        '<text class="s" x="20" y="48">Point estimate and 95 % CI (paired cluster bootstrap, 80 buildings). '
        "Shaded: where the interval had to lie.</text>",
    ]
    y0, y1 = top - 10, top + row_h * 2 - 18
    for t in (0.0, 0.05, 0.10, 0.15, 0.20):
        body.append(f'<line class="grid" x1="{x(t):.1f}" y1="{y0}" x2="{x(t):.1f}" y2="{y1}"/>')
        body.append(
            f'<text class="ax" x="{x(t):.1f}" y="{y1 + 18}" text-anchor="middle">{t * 100:.0f} %</text>'
        )
    specs = [
        ("Loss gain vs. cross_rank", gain, (0.10, hi), "needs lower bound ≥ 10 %"),
        ("Relative coverage loss", cov, (lo, 0.05), "needs upper bound ≤ 5 %"),
    ]
    for i, (label, est, (z0, z1), rule) in enumerate(specs):
        cy = top + row_h * i + 14
        band_top = cy - 20
        body.append(
            f'<rect class="zone" x="{x(z0):.1f}" y="{band_top}" width="{x(z1) - x(z0):.1f}" height="40"/>'
        )
        edge = z0 if z0 > lo else z1
        body.append(
            f'<line class="thr" x1="{x(edge):.1f}" y1="{band_top}" x2="{x(edge):.1f}" y2="{band_top + 40}"/>'
        )
        body.append(f'<text class="l" x="{left - 14}" y="{cy}" text-anchor="end">{label}</text>')
        body.append(
            f'<text class="v" x="{left - 14}" y="{cy + 16}" text-anchor="end">{rule}</text>'
        )
        body.append(
            f'<line class="ci" x1="{x(est["low"]):.1f}" y1="{cy}" x2="{x(est["high"]):.1f}" y2="{cy}"/>'
        )
        body.append(f'<circle class="pt" cx="{x(est["estimate"]):.1f}" cy="{cy}" r="6"/>')
        body.append(
            f'<text class="v" x="{width - right + 14}" y="{cy + 4}">{pct(est["estimate"])}  '
            f"[{pct(est['low'])}, {pct(est['high'])}]</text>"
        )
        body.append(f'<text class="l" x="{width - right + 14}" y="{cy + 22}">✗ not met</text>')
    return svg(width, height, body, "H2: the two preregistered conditions")


CELL_LABELS = {
    "sta-ind-direct": "static · independent · direct",
    "sta-ind-relay": "static · independent · relay",
    "sta-cor-direct": "static · correlated · direct",
    "sta-cor-relay": "static · correlated · relay",
    "dyn-ind-direct": "dynamic · independent · direct",
    "dyn-ind-relay": "dynamic · independent · relay",
    "dyn-cor-direct": "dynamic · correlated · direct",
    "dyn-cor-relay": "dynamic · correlated · relay",
}


def cells(primary: dict) -> str:
    per = primary["gain_per_cell"]
    rows = [(CELL_LABELS[k], per[k]["gain"]) for k in CELL_LABELS]
    return forest(
        rows,
        -0.05,
        0.20,
        [0.0, 0.05, 0.10, 0.15, 0.20],
        "Where repair helps: loss gain per condition",
        "Doors · sensor error · circulation. 95 % CI; dashed: the 10 % relevance threshold.",
        (0.10, "10 % threshold"),
    )


METHOD_LABELS = {
    "repair": "repair (candidate)",
    "dynamics_ids": "dynamics_ids",
    "age_revisit": "age_revisit",
    "no_comm": "no_comm",
    "cross_rank": "cross_rank (H2 control)",
    "ttl_ids": "ttl_ids",
    "naive_gossip": "naive_gossip",
}


def loss_vs_bytes(primary: dict) -> str:
    loss, sent = primary["mean_loss"], primary["sent_bytes_total"]
    width, height = 760, 430
    left, right, top, bottom = 80, 40, 70, 60
    pw, ph = width - left - right, height - top - bottom
    xlo, xhi = math.log10(4_000), math.log10(1_500_000)
    ylo, yhi = 0.25, 0.40

    def x(v: float) -> float:
        return left + (math.log10(v) - xlo) / (xhi - xlo) * pw

    def y(v: float) -> float:
        return top + (yhi - v) / (yhi - ylo) * ph

    body = [
        '<text class="t" x="20" y="28">Decision loss vs. communication</text>',
        '<text class="s" x="20" y="48">Mean decision loss (lower is better, 95 % CI) against bytes sent '
        "per episode, log scale. In-distribution test, 80 buildings.</text>",
    ]
    for t in (0.26, 0.28, 0.30, 0.32, 0.34, 0.36, 0.38, 0.40):
        body.append(
            f'<line class="grid" x1="{left}" y1="{y(t):.1f}" x2="{left + pw}" y2="{y(t):.1f}"/>'
        )
        body.append(
            f'<text class="ax" x="{left - 8}" y="{y(t) + 4:.1f}" text-anchor="end">{t:.2f}</text>'
        )
    for t, lab in ((10_000, "10 kB"), (100_000, "100 kB"), (1_000_000, "1 MB")):
        body.append(
            f'<line class="grid" x1="{x(t):.1f}" y1="{top}" x2="{x(t):.1f}" y2="{top + ph}"/>'
        )
        body.append(
            f'<text class="ax" x="{x(t):.1f}" y="{top + ph + 18}" text-anchor="middle">{lab}</text>'
        )
    body.append(
        f'<text class="ax" x="{left + pw / 2:.1f}" y="{height - 14}" text-anchor="middle">bytes sent per episode (all robots)</text>'
    )
    offsets = {
        "repair": (10, -10, "start"),
        "dynamics_ids": (10, 16, "start"),
        "age_revisit": (-10, 18, "end"),
        "no_comm": (10, -8, "start"),
        "cross_rank": (10, 4, "start"),
        "ttl_ids": (10, 4, "start"),
        "naive_gossip": (10, 4, "start"),
    }
    for m in (
        "cross_rank",
        "age_revisit",
        "no_comm",
        "ttl_ids",
        "naive_gossip",
        "dynamics_ids",
        "repair",
    ):
        e, b = loss[m], sent[m]["estimate"]
        main = m == "repair"
        ci, pt = ("ci", "pt") if main else ("ci2", "pt2")
        cx = x(b)
        body.append(
            f'<line class="{ci}" x1="{cx:.1f}" y1="{y(e["low"]):.1f}" x2="{cx:.1f}" y2="{y(e["high"]):.1f}"/>'
        )
        body.append(
            f'<circle class="{pt}" cx="{cx:.1f}" cy="{y(e["estimate"]):.1f}" r="{6.5 if main else 5}"/>'
        )
        dx, dy, anchor = offsets[m]
        weight = ' font-weight="600"' if main else ""
        body.append(
            f'<text class="l" x="{cx + dx:.1f}" y="{y(e["estimate"]) + dy:.1f}" text-anchor="{anchor}"{weight}>'
            f"{METHOD_LABELS[m]}</text>"
        )
    return svg(width, height, body, "Decision loss vs. communication")


def main() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    primary = result["primary"]
    OUT.mkdir(parents=True, exist_ok=True)
    figures = {
        "h2_decision.svg": h2(primary),
        "gain_per_cell.svg": cells(primary),
        "loss_vs_bytes.svg": loss_vs_bytes(primary),
    }
    for name, text in figures.items():
        (OUT / name).write_bytes(text.encode("utf-8"))
        print("wrote", (OUT / name).relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
