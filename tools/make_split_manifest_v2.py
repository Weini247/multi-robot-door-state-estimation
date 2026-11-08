"""Write (or check) the follow-up manifest of fresh buildings for study 2 (E09-T02).

    python tools/make_split_manifest_v2.py [--check]

Every building of configs/splits/splits_v1.json enters the duplicate index first, so the
new buildings are disjoint from all earlier ones by hash, isomorphism and raster IoU.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from argus.experiments.splits import SplitEntry, build_followup_manifest
from argus.world.floorplan import FloorplanParams

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "configs/splits/splits_v1.json"
TARGET = ROOT / "configs/splits/splits_v2.json"


def render() -> str:
    prior_data = json.loads(PRIOR.read_text(encoding="utf-8"))
    params = dict(prior_data["params"])
    params["passage_rates"] = tuple(params["passage_rates"])
    prior = tuple(SplitEntry(*row) for row in prior_data["entries"])
    data = build_followup_manifest(prior, FloorplanParams(**params))
    entries = data.pop("entries")
    head = json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True)
    rows = ",\n".join("    " + json.dumps(e, ensure_ascii=False) for e in entries)
    return head[:-2] + ',\n  "entries": [\n' + rows + "\n  ]\n}\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    text = render()
    json.loads(text)
    if args.check:
        same = TARGET.exists() and TARGET.read_bytes() == text.encode("utf-8")
        print("PASS: manifest reproduces" if same else "FAIL: manifest differs")
        sys.exit(0 if same else 1)
    TARGET.write_bytes(text.encode("utf-8"))
    print(f"wrote {TARGET.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
