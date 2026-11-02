"""Per-family statistics of the building generator (ARGUS-E07-T07 evidence).

    python tools/floorplan_stats.py [--seeds N]

Reports attempts, room and passage counts, size classes, corridor rooms and the
share of within-family near duplicates over seeds 0..N-1 with default params.
"""

from __future__ import annotations

import argparse
import itertools
import statistics
from collections import Counter

from argus.world.floorplan import FAMILIES, generate_floorplan, is_near_duplicate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=40)
    args = parser.parse_args()
    print(f"seeds 0..{args.seeds - 1}, default FloorplanParams")
    header = "family       attempts(mean/max) rooms(min/mean/max) passages(mean) sizes S/M/L"
    print(header + "  corridor  near-dup pairs")
    for family in FAMILIES:
        plans = [generate_floorplan(family, s) for s in range(args.seeds)]
        attempts = [p.attempt for p in plans]
        rooms = [len(p.graph.nodes) for p in plans]
        passages = [len(p.graph.edges) for p in plans]
        sizes = Counter(n[1] for p in plans for n in p.graph.nodes)
        corridors = sum(n[2] for p in plans for n in p.graph.nodes)
        pairs = list(itertools.combinations(plans, 2))
        dups = sum(is_near_duplicate(a, b) for a, b in pairs)
        print(
            f"{family:12s} {statistics.mean(attempts):5.2f}/{max(attempts):<3d}"
            f"            {min(rooms)}/{statistics.mean(rooms):.1f}/{max(rooms)}"
            f"           {statistics.mean(passages):5.2f}"
            f"          {sizes['S']}/{sizes['M']}/{sizes['L']}"
            f"   {corridors:4d}      {dups}/{len(pairs)}"
        )


if __name__ == "__main__":
    main()
