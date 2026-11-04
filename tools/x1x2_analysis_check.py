"""Check that the X1/X2 result was produced once by the frozen analysis code (E08-T03)."""

import hashlib
import json
from pathlib import Path

from argus.experiments.runset import FROZEN_PATH, text_sha256

ROOT = Path(__file__).resolve().parents[1]
frozen = json.loads((ROOT / FROZEN_PATH).read_text(encoding="utf-8"))
bound = frozen["analysis"]["code"]
current = {path: text_sha256(ROOT / path) for path in bound}
print("code_matches_frozen", current == bound)
log = [
    json.loads(line)
    for line in (ROOT / "runs/x1x2_v1/analysis_log.jsonl").read_text(encoding="utf-8").splitlines()
]
print("analysis_runs", len(log))
for entry in log:
    print("run", entry["utc"], entry["verdict"], "code_bound", entry["analysis_code"] == bound)
result_path = ROOT / "docs/results/X1_X2_TEST_v1.json"
result = json.loads(result_path.read_text(encoding="utf-8"))
print("result_dry_run", result["dry_run"], "terminal", result["terminal"])
print("result_code_bound", result["analysis_code"] == bound)
print(
    "result_frozen",
    result["frozen_sha256"] == hashlib.sha256((ROOT / FROZEN_PATH).read_bytes()).hexdigest(),
)
print("draws", result["draws"], "seed", result["seed"], "prefix_b", result["prefix_b"])
print("h2_verdict_logged", log[-1]["verdict"])
