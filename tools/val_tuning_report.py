"""Render the validation tuning result as docs/results/VAL_TUNING_v1.md (ARGUS-E07-T06).

    python tools/val_tuning_report.py --results docs/results/val_tuning_v1.jsonl

Applies the preregistered selection rule to the raw results and writes the
report; numbers come only from the results file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from argus.experiments.cells import CELLS
from argus.experiments.protocol import TuningResult, load_preregistration, select
from argus.policies.baselines import METHOD_SPECS

ROOT = Path(__file__).resolve().parents[1]
NOTES = {
    "v1": "Abweichung (Commit 0ddbbbb, `project/context.json`): 26 `naive_gossip`-Episoden "
    "brachen im ersten Lauf mit OverflowError in der naiven Log-Odds-Belief ab. Nach dem Fix, "
    "der nur den zuvor abstürzenden Zweig ändert, wurden sie neu gerechnet; die Rohdaten vor "
    "dem Fix sind lokal archiviert und gehen nicht in die Auswahl ein. Der Lauf war über Nacht "
    "pausiert und wurde mit denselben Seeds fortgesetzt.",
    "v2": "Wiederholung nach Änderung 01 (O1, `docs/EXPERIMENT_PROTOCOL_AMENDMENT_01.md`): "
    "Nachrichten tragen eine methodenneutrale Transport-ID fester Länge; Raster, Gebäude, "
    "Seeds, Budget und Auswahlregel sind unverändert. v1 bleibt erhalten und geht nicht ein. "
    "Abweichung (`project/context.json`): Der Lauf stoppte nach 5 Gebäuden an der "
    "Wall-Projektion, weil eine andere Session parallel CPU-Last erzeugte, und wurde mit "
    "denselben Seeds fortgesetzt. CPU 7,11 h liegt im Deckel von 8 h; die Wall-Zeit von "
    "zusammen 2,63 h überschreitet den Deckel von 2 h.",
}


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "docs/results/VAL_TUNING_v1.md")
    args = parser.parse_args()
    args.out = args.out.resolve()
    version = "v1" if args.results.stem.endswith("v1") else "v2"
    prereg = load_preregistration(ROOT)
    args.results = args.results.resolve()
    rows = load(args.results)
    roles = {m: s.role for m, s in METHOD_SPECS.items()}
    sel = select(
        [
            TuningResult(
                r["method_id"],
                r["config_id"],
                r["building_id"],
                r["cell_id"],
                r["mean_loss"],
                r["sent_bytes"],
                r["status"],
            )
            for r in rows
        ],
        roles,
        prereg["tuning"]["minimum_complete_buildings"],
    )
    status = Counter(r["status"] for r in rows)
    cpu = sum(r["cpu_s"] for r in rows) / 3600
    digest = hashlib.sha256(args.results.read_bytes()).hexdigest()
    lines = [
        f"# Validation-Tuning {version} (E07-T06)",
        "",
        "Generiert von `tools/val_tuning_report.py` aus "
        f"`{args.results.relative_to(ROOT).as_posix()}` (SHA-256 `{digest}`). "
        "Auswahlregel und Raster: `configs/experiments/preregistration_v1.json`.",
        "",
        f"- Episoden: {len(rows)}; Status: "
        + ", ".join(f"{k} {v}" for k, v in sorted(status.items())),
        f"- CPU-Zeit der Episoden: {cpu:.2f} h",
        f"- vollständige Gebäude: {len(sel.complete_buildings)} "
        f"({', '.join(sel.complete_buildings)})",
        f"- Ergebnis: {sel.reason}",
        f"- **H2-Kontrolle: `{sel.h2_control}`**"
        + (f" mit Konfiguration `{sel.best_config[sel.h2_control]}`" if sel.h2_control else ""),
        "",
        "Mittlerer Entscheidungsverlust pro Anfrage (Mittel über Gebäude des Mittels über "
        "die 8 Zellen) und mittlere gesendete Bytes pro Roboter. Gewählte Konfigurationen "
        "sind fett.",
        "",
        "| Methode | Rolle | Konfiguration | Verlust | Bytes |",
        "|---|---|---|---|---|",
    ]
    for (method, config), (loss, sent) in sorted(sel.per_config.items(), key=lambda kv: kv[1][0]):
        mark = "**" if sel.best_config.get(method) == config else ""
        lines.append(
            f"| {mark}{method}{mark} | {roles[method]} | {mark}{config}{mark} "
            f"| {loss:.4f} | {sent:.0f} |"
        )
    lines += [
        "",
        "Pro Zelle, gewählte Konfigurationen (Mittel über vollständige Gebäude):",
        "",
        "| Methode | " + " | ".join(c.cell_id for c in CELLS) + " |",
        "|---|" + "---|" * len(CELLS),
    ]
    for method, config in sorted(sel.best_config.items()):
        cells = []
        for cell in CELLS:
            vals = [
                r["mean_loss"]
                for r in rows
                if r["method_id"] == method
                and r["config_id"] == config
                and r["cell_id"] == cell.cell_id
                and r["building_id"] in sel.complete_buildings
            ]
            cells.append(f"{sum(vals) / len(vals):.3f}" if vals else "–")
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    lines += [
        "",
        NOTES[version],
        "",
        "Das sind Validationdaten für die Konfigurationswahl, keine Testergebnisse und keine "
        "Evidenz für H1/H2. Der Kandidat `repair` wird nur getunt, nicht mit Kontrollen "
        "verglichen; dieser Vergleich gehört auf id_test (E08).",
    ]
    args.out.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    print(f"wrote {args.out.relative_to(ROOT).as_posix()}: h2_control={sel.h2_control}")


if __name__ == "__main__":
    main()
