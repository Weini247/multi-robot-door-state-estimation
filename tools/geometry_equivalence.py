"""Compare collision resolution of two geometry revisions on random motions.

    python tools/geometry_equivalence.py <git-revision> [cases]

Loads src/argus/world/geometry.py from the given revision (git show) and the
working tree, runs both on the same random rectangles and robot motions, and
exits 1 on the first difference in poses, events or fractions.
"""

from __future__ import annotations

import importlib.util
import random
import subprocess
import sys
import tempfile
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]


def load(name: str, source: str) -> ModuleType:
    path = Path(tempfile.mkdtemp()) / f"{name}.py"
    path.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    revision, cases = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 3000
    old_source = subprocess.run(
        ["git", "show", f"{revision}:src/argus/world/geometry.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
        encoding="utf-8",
    ).stdout
    old = load("geometry_old", old_source)
    new = load("geometry_new", (ROOT / "src/argus/world/geometry.py").read_text(encoding="utf-8"))
    rng = random.Random(7)
    with_contacts = 0
    for case in range(cases):
        corners = [(rng.uniform(0, 8), rng.uniform(0, 8)) for _ in range(4)]
        sizes = [(rng.uniform(0.05, 1.5), rng.uniform(0.05, 1.5)) for _ in range(4)]
        motions = [
            (
                f"r{i}",
                (rng.uniform(0, 10), rng.uniform(0, 10), rng.uniform(-3, 3)),
                0.2,
                rng.uniform(-2, 2),
                rng.uniform(-2, 2),
            )
            for i in range(3)
        ]
        dt = rng.choice([0.1, 1.0, 2.0])
        results = []
        for module in (old, new):
            rects = [module.Rect(x, y, x + w, y + h) for (x, y), (w, h) in zip(corners, sizes)]
            poses, events = module.resolve_motions([module.Motion(*m) for m in motions], dt, rects)
            results.append((poses, [(e.kind, e.robot_ids, e.fraction) for e in events]))
        if results[0] != results[1]:
            print(f"DIFFERENCE in case {case}: {results[0]} != {results[1]}")
            return 1
        with_contacts += bool(results[0][1])
    print(f"identical over {cases} cases against {revision}; {with_contacts} with collisions")
    return 0


if __name__ == "__main__":
    sys.exit(main())
