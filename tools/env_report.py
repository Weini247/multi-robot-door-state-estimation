"""Print the resolved interpreter, platform and package versions of this environment."""

from __future__ import annotations

import importlib.metadata
import platform
import sys

PACKAGES = ("argus", "numpy", "scipy", "pytest", "ruff", "mypy")


def main() -> None:
    print(f"python {sys.version.split()[0]} ({platform.python_implementation()})")
    print(f"platform {platform.platform()} {platform.machine()}")
    for name in PACKAGES:
        print(f"{name} {importlib.metadata.version(name)}")


if __name__ == "__main__":
    main()
