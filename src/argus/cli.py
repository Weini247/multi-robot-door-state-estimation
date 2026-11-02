"""Command-line entry point `argus`.

Available: `--help`, `--version` and `x0-reference` (E01-T04). The planned
`run`, `batch`, `analyze` and `replay` commands are added by their backlog
tasks; until then they are rejected instead of being silently accepted.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from argus import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="argus",
        description=(
            "ARGUS research code. No simulation commands exist yet; "
            "see ARGUS_PROJECT_STATE.md for the current stage."
        ),
    )
    parser.add_argument("--version", action="version", version=f"argus {__version__}")
    commands = parser.add_subparsers(dest="command")
    x0 = commands.add_parser(
        "x0-reference", help="export raw outputs of the exact X0 micro references"
    )
    x0.add_argument("--output", type=Path, required=True, help="new directory to create")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "x0-reference":
        from argus.experiments.x0 import export

        try:
            path = export(args.output)
        except FileExistsError as error:
            print(f"argus: {error}")
            return 1
        print(f"wrote {path}")
    return 0
