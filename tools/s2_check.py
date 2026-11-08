"""Count and hash the study 2 batch output and bind the result to the frozen code (E09-T02)."""

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from s2_batch import S2_ANALYSIS_CODE, S2_FROZEN_PATH  # noqa: E402

from argus.experiments.runset import text_sha256  # noqa: E402

out = ROOT / "runs/s2_v1"
text = (out / "results.jsonl").read_text(encoding="utf-8")
rows = [json.loads(line) for line in text.splitlines()]
print("rows", len(rows), "unique_keys", len({r["key"] for r in rows}))
print("status", dict(Counter(r["status"] for r in rows)))
print("conditions", dict(Counter(r["key"].split("/")[0] for r in rows)))
print(
    "buildings", len({r["building_id"] for r in rows}), "splits", sorted({r["split"] for r in rows})
)
frozen_text = (ROOT / S2_FROZEN_PATH).read_text(encoding="utf-8")
frozen = json.loads(frozen_text)
sha = hashlib.sha256(frozen_text.encode("utf-8")).hexdigest()
code = {path: text_sha256(ROOT / path) for path in S2_ANALYSIS_CODE}
print("code_matches_frozen", code == frozen["analysis_code"])
log_text = (out / "analysis_log.jsonl").read_text(encoding="utf-8")
log = [json.loads(line) for line in log_text.splitlines()]
print("analysis_runs", len(log), "bound", all(e["frozen_sha256"] == sha for e in log))
result = json.loads((ROOT / "docs/results/S2_v1.json").read_text(encoding="utf-8"))
print("result_dry_run", result["dry_run"], "terminal", result["terminal"], result["prefix_b"])
print("lock", (out / "LOCK").exists(), "pause", (out / "PAUSE").exists())
digest = hashlib.sha256()
files = sorted(p for p in (out / "raw").rglob("*") if p.is_file())
for path in files:
    digest.update(path.relative_to(out).as_posix().encode())
    digest.update(hashlib.sha256(path.read_bytes()).digest())
print("raw_files", len(files), "raw_tree_sha256", digest.hexdigest())
for name in ("results.jsonl", "usage.json", "sessions.jsonl", "run_plan.json"):
    print(name, hashlib.sha256((out / name).read_bytes()).hexdigest())
