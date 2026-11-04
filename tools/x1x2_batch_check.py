"""Count and hash the X1/X2 test batch output (evidence for E08-T02)."""

import hashlib
import json
from collections import Counter
from pathlib import Path

out = Path("runs/x1x2_v1")
rows = [
    json.loads(line) for line in (out / "results.jsonl").read_text(encoding="utf-8").splitlines()
]
print("rows", len(rows))
print("status", dict(Counter(r["status"] for r in rows)))
keys = [r["key"] for r in rows]
print("unique_keys", len(set(keys)), "duplicates", len(keys) - len(set(keys)))
plan = json.loads((out / "run_plan.json").read_text(encoding="utf-8"))
print(
    "run_plan",
    {k: plan[k] for k in plan if k in ("b", "frozen_sha256", "src_sha256", "replacements")},
)
for line in (out / "sessions.jsonl").read_text(encoding="utf-8").splitlines():
    s = json.loads(line)
    print(
        "session",
        s["stop_reason"],
        s["completed_groups"],
        round(s["cpu_h"], 3),
        round(s["wall_h"], 3),
        s["commit"],
    )
usage = json.loads((out / "usage.json").read_text(encoding="utf-8"))
print(
    "usage_groups",
    len(usage["groups"]),
    "cpu_h",
    round(usage["cpu_s"] / 3600, 3),
    "wall_h",
    round(usage["wall_s"] / 3600, 3),
)
print("lock_present", (out / "LOCK").exists(), "pause_present", (out / "PAUSE").exists())
raw = sorted((out / "raw").rglob("*"))
files = [p for p in raw if p.is_file()]
h = hashlib.sha256()
for p in files:
    h.update(p.relative_to(out).as_posix().encode())
    h.update(hashlib.sha256(p.read_bytes()).digest())
print("raw_files", len(files), "raw_tree_sha256", h.hexdigest())
for name in (
    "results.jsonl",
    "usage.json",
    "sessions.jsonl",
    "run_plan.json",
    "analysis_log.jsonl",
):
    print(name, hashlib.sha256((out / name).read_bytes()).hexdigest())
