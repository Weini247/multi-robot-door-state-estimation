"""Generate docs/results/X2_micro.md: implementation vs exact theory, expectations, negatives.

    python -m uv run --locked python tools/x2_report.py (--write | --check)

Theory never calls argus: rational Bayes for the persistent bias (one g for
both readings) and for the decorrelated limit (independent g per reading,
equivalent to an independent sensor with the marginal error e). The
decorrelated factor is not exactly that limit; its deviation is bounded by
exp(-(mu01 + mu10) d) rho (1 - rho), and the tolerance is set from that bound.
"""

from __future__ import annotations

import argparse
import math
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from argus.experiments.x2_micro import load, run  # noqa: E402

CONFIG = ROOT / "configs/experiments/X2_micro.json"
REPORT = ROOT / "docs/results/X2_micro.md"
ZERO = 1e-12  # values within the theory tolerance of 0 count as zero


def dec(value: float) -> Fraction:
    return Fraction(repr(value))


def risk(p: Fraction, c10: Fraction, c01: Fraction) -> Fraction:
    return min(c10 * (1 - p), c01 * p)


def decide(p: Fraction, c10: Fraction, c01: Fraction) -> int:
    return int(p >= c10 / (c10 + c01))


def theory_reobserve(s: dict[str, Any], persistent: bool, c10: Fraction, c01: Fraction) -> Fraction:
    p0, rho, q = dec(s["prior_open"]), dec(s["bias_probability"]), dec(s["private_error"])

    def lik(y: int, z: int, g: int) -> Fraction:
        return (1 - q) if y == (z ^ g) else q

    def weight(ys: tuple[int, ...], z: int) -> Fraction:
        if persistent:  # one g for all readings
            return sum(
                ((1 - rho) if g == 0 else rho) * math.prod(lik(y, z, g) for y in ys) for g in (0, 1)
            )
        return math.prod(
            sum(((1 - rho) if g == 0 else rho) * lik(y, z, g) for g in (0, 1)) for y in ys
        )  # independent g per reading

    def posterior(ys: tuple[int, ...]) -> tuple[Fraction, Fraction]:
        a, b = p0 * weight(ys, 1), (1 - p0) * weight(ys, 0)
        return a / (a + b), a + b

    first = (s["first_reading"],)
    p1, mass1 = posterior(first)
    expected = Fraction(0)
    for y2 in (0, 1):
        p2, mass12 = posterior(first + (y2,))
        expected += (mass12 / mass1) * risk(p2, c10, c01)
    return risk(p1, c10, c01) - expected


def theory_send(s: dict[str, Any], c10: Fraction, c01: Fraction) -> Fraction:
    p0, e = dec(s["prior_open"]), dec(s["marginal_error"])
    y = s["first_reading"]
    like1 = (1 - e) if y == 1 else e
    like0 = e if y == 1 else (1 - e)
    p_after = p0 * like1 / (p0 * like1 + (1 - p0) * like0)
    p_sender = p_after  # the sender holds exactly this reading

    def loss(d: int) -> Fraction:
        return c10 * (1 - p_sender) if d == 1 else c01 * p_sender

    return loss(decide(p0, c10, c01)) - loss(decide(p_after, c10, c01))


def compare(config: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    s, f = config["setting"], config["factors"]
    mixing = sum(f["bias"]["decorrelated"])
    rho = s["bias_probability"]
    bound = math.exp(-mixing * s["reobservation_delay_s"]) * rho * (1 - rho)
    checked = []
    for row in rows:
        c10, c01 = (dec(v) for v in f["losses"][row["losses"]])
        if row["kind"] == "reobserve":
            exact = theory_reobserve(s, row["bias"] == "persistent", c10, c01)
            implementation = row["gain"]
            tol = 1e-12 if row["bias"] == "persistent" else max(1e-12, 20 * bound)
        else:
            exact = theory_send(s, c10, c01)
            implementation = row["decision_value"]
            tol = 1e-12
        deviation = abs(float(exact) - implementation)
        checked.append(
            {
                **row,
                "theory": float(exact),
                "deviation": deviation,
                "tolerance": tol,
                "ok": deviation <= tol,
            }
        )
    return checked


def expectations(config: dict[str, Any], rows: list[dict[str, Any]]) -> list[tuple[str, str, bool]]:
    def get(**kw: str) -> dict[str, Any]:
        return next(r for r in rows if all(r.get(k) == v for k, v in kw.items()))

    results = []
    e1 = all(
        get(kind="reobserve", bias="decorrelated", losses=lo, costs="zero")["value"]
        >= get(kind="reobserve", bias="persistent", losses=lo, costs="zero")["value"] - 1e-12
        for lo in config["factors"]["losses"]
    )
    results.append(("E1", "decorrelated >= persistent at zero cost", e1))
    e2 = all(
        abs(
            (
                get(kind="reobserve", bias=b, losses=lo, costs="zero")["value"]
                - get(kind="reobserve", bias=b, losses=lo, costs="pilot")["value"]
            )
            - 0.16
        )
        <= 1e-12
        for b in config["factors"]["bias"]
        for lo in config["factors"]["losses"]
    )
    results.append(("E2", "pilot costs subtract exactly 0.16", e2))
    e3 = all(
        r["value"] == 0.0 for r in rows if r["kind"] == "send" and r["receiver"] == "holds_reading"
    )
    results.append(("E3", "held reading has send value 0", e3))
    p = config["setting"]["delivery_probability"]
    e4 = all(
        r["value"] >= 0 and abs(r["value"] - p * r["decision_value"]) <= 1e-12
        for r in rows
        if r["kind"] == "send" and r["receiver"] == "uncertain_history"
    )
    results.append(("E4", "uncertain receiver: value = p_d x decision value >= 0", e4))
    # Decided with the same tolerance as the theory check: 1e-16 rounding noise is zero.
    e5 = get(kind="reobserve", bias="persistent", losses="symmetric", costs="zero")["value"] > ZERO
    results.append(("E5", "persistent, symmetric, zero cost: positive value", e5))
    return results


def render(config: dict[str, Any], checked: list[dict[str, Any]]) -> tuple[str, bool]:
    outcome = expectations(config, checked)
    lines = [
        "# X2 micro — exact ablation of independence, cost and receiver history",
        "",
        "> Generiert von `tools/x2_report.py` aus `configs/experiments/X2_micro.json`. "
        "Nicht von Hand ändern.",
        "> Mechanismusprüfung unter dem Nominalmodell, **kein Befund zu H2 oder H3**: "
        "keine Simulation,",
        "> keine Gebäude, keine Statistik. Die Erwartungen wurden vor der Rechnung registriert.",
        "",
        "## Ergebnisse gegen Theorie",
        "",
        "| Art | Bias | Verluste | Kosten/Empfänger | Implementierung | Theorie | Abweichung "
        "| Toleranz | ok |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in checked:
        detail = r.get("costs") or r.get("receiver")
        impl = r["gain"] if r["kind"] == "reobserve" else r["decision_value"]
        lines.append(
            f"| {r['kind']} | {r['bias']} | {r['losses']} | {detail} | {impl:.15g} | "
            f"{r['theory']:.15g} | {r['deviation']:.2e} | {r['tolerance']:.1e} | "
            f"{'PASS' if r['ok'] else 'FAIL'} |"
        )
    lines += [
        "",
        "Für `reobserve` ist die Spalte Implementierung der Informationsgewinn vor Kosten; "
        "für `send` der Entscheidungswert vor Zustellwahrscheinlichkeit.",
        "",
        "## Werte nach Kosten bzw. Zustellung",
        "",
        "| Art | Bias | Verluste | Kosten/Empfänger | Wert |",
        "|---|---|---|---|---|",
    ]
    for r in checked:
        lines.append(
            f"| {r['kind']} | {r['bias']} | {r['losses']} | "
            f"{r.get('costs') or r.get('receiver')} | {r['value']:.6g} |"
        )
    lines += [
        "",
        "## Präregistrierte Erwartungen",
        "",
        "| ID | Prüfung | Ergebnis |",
        "|---|---|---|",
    ]
    lines += [
        f"| {i} | {text} | {'bestätigt' if ok else '**nicht bestätigt**'} |"
        for i, text, ok in outcome
    ]
    negatives = [r for r in checked if r["value"] <= ZERO]
    lines += [
        "",
        "## Negative und Null-Fälle",
        "",
        "Fälle ohne positiven Wert (Information ändert keine Entscheidung "
        "oder Kosten übersteigen den Gewinn):",
        "",
    ]
    lines += [
        f"- {r['kind']}, Bias {r['bias']}, Verluste {r['losses']}, "
        f"{r.get('costs') or r.get('receiver')}: Wert {r['value']:.6g}"
        for r in negatives
    ]
    lines += [
        "",
        "## Einordnung",
        "",
        "Ein Wert von 0 entsteht immer dann, wenn kein mögliches Messergebnis "
        "die Entscheidung ändert; "
        "das ist eine Eigenschaft der Entscheidungsaufgabe, kein Softwarefehler. Nicht bestätigte "
        "Erwartungen bleiben so stehen und werden nicht nachträglich umformuliert. Werte innerhalb "
        f"{ZERO:g} von 0 gelten als 0 (Rundung).",
        "",
    ]
    return "\n".join(lines), all(r["ok"] for r in checked)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    config = load(CONFIG)
    text, all_ok = render(config, compare(config, run(config)))
    if args.write:
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_bytes(text.encode("utf-8"))
        print(f"wrote {REPORT.relative_to(ROOT)}")
    elif not REPORT.exists() or REPORT.read_text(encoding="utf-8") != text:
        print("STALE: X2 micro report differs")
        return 1
    else:
        print("PASS: X2 micro report current")
    if not all_ok:
        print("FAIL: implementation deviates from theory")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
