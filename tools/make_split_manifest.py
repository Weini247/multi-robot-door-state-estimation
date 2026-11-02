"""Write (or check) the frozen building split manifest (ARGUS-E07-T06).

    python tools/make_split_manifest.py [--check]

Builds the manifest from default FloorplanParams and QUOTAS_V1 and writes
configs/splits/splits_v1.json as canonical JSON (one entry per line). --check
regenerates and compares byte for byte without writing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from argus.experiments.splits import build_manifest

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "configs/splits/splits_v1.json"


def render() -> str:
    data = build_manifest().to_json()
    entries = data.pop("entries")
    head = json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True)
    rows = ",\n".join("    " + json.dumps(e, ensure_ascii=False) for e in entries)
    return head[:-2] + ',\n  "entries": [\n' + rows + "\n  ]\n}\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    text = render()
    json.loads(text)  # must stay valid JSON
    if args.check:
        same = TARGET.exists() and TARGET.read_bytes() == text.encode("utf-8")
        print("PASS: manifest reproduces" if same else "FAIL: manifest differs")
        sys.exit(0 if same else 1)
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_bytes(text.encode("utf-8"))
    print(f"wrote {TARGET.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
