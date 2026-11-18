"""Compare a reproduced batch with the original, episode by episode (ARGUS-E12-T01).

    python tools/repro_compare.py --original runs/s2_v1 --repro <clone>/runs/s2_repro \
        --report docs/results/REPRO_S2.json

An episode is reproduced when its result row matches on every field except the
volatile ones (timing, memory, raw-file hash) and its raw record matches after removing
run metadata (code commit, creation time, run id, platform). The raw-file hash differs by
construction because the raw file records that metadata.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

VOLATILE_ROW = frozenset({"cpu_s", "peak_rss_bytes", "process_wall_s", "wall_s", "raw_sha256"})
# run metadata, not results: the run id is drawn per run, platform names the OS build
RAW_MANIFEST_METADATA = ("code_commit", "created_utc", "run_id", "platform")


def rows(out: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    with (out / "results.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["key"] in result:
                raise SystemExit(f"duplicate key in {out}: {row['key']}")
            result[row["key"]] = row
    return result


def stable_raw(path: Path) -> str:
    data = json.loads(gzip.decompress(path.read_bytes()))
    data.pop("code_commit", None)
    manifest = data.get("manifest", {})
    for key in RAW_MANIFEST_METADATA:
        manifest.pop(key, None)
    text = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compare(original: Path, repro: Path, sample: bool = False) -> dict[str, Any]:
    """Compare two batches; with sample=True only the episodes present in repro count."""
    a, b = rows(original), rows(repro)
    if sample:
        a = {k: v for k, v in a.items() if k in b}
    missing = sorted(set(a) - set(b))
    extra = sorted(set(b) - set(a))
    row_diffs: dict[str, list[str]] = {}
    raw_diffs: list[str] = []
    for key in sorted(set(a) & set(b)):
        fields = sorted(
            k for k in (set(a[key]) | set(b[key])) - VOLATILE_ROW if a[key].get(k) != b[key].get(k)
        )
        if fields:
            row_diffs[key] = fields
        if stable_raw(original / a[key]["raw"]) != stable_raw(repro / b[key]["raw"]):
            raw_diffs.append(key)
    matched = len(set(a) & set(b)) - len(set(row_diffs) | set(raw_diffs))
    return {
        "original_episodes": len(a),
        "repro_episodes": len(b),
        "missing_in_repro": missing[:50],
        "missing_count": len(missing),
        "extra_in_repro": extra[:50],
        "extra_count": len(extra),
        "row_differences": dict(list(row_diffs.items())[:50]),
        "row_difference_count": len(row_diffs),
        "raw_differences": raw_diffs[:50],
        "raw_difference_count": len(raw_diffs),
        "reproduced_episodes": matched,
        "volatile_row_fields": sorted(VOLATILE_ROW),
        "sample": sample,
        "passed": not (missing or extra or row_diffs or raw_diffs),
    }


def compare_reports(original: Path, repro: Path) -> dict[str, Any]:
    a = json.loads(original.read_text(encoding="utf-8"))
    b = json.loads(repro.read_text(encoding="utf-8"))
    differing = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    return {
        "original": str(original.name),
        "differing_top_level_keys": differing,
        "identical": not differing,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--repro", type=Path, required=True)
    parser.add_argument("--original-report", type=Path)
    parser.add_argument("--repro-report", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--sample", action="store_true", help="repro holds a sample only")
    args = parser.parse_args()
    result = {"episodes": compare(args.original, args.repro, args.sample)}
    if args.original_report and args.repro_report:
        result["analysis"] = compare_reports(args.original_report, args.repro_report)
    args.report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    e = result["episodes"]
    print(
        f"episodes {e['reproduced_episodes']}/{e['original_episodes']} reproduced; "
        f"missing {e['missing_count']}, extra {e['extra_count']}, "
        f"row diffs {e['row_difference_count']}, raw diffs {e['raw_difference_count']}; "
        f"analysis identical: {result.get('analysis', {}).get('identical')}"
    )


if __name__ == "__main__":
    main()
