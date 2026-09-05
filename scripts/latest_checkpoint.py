#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve the numerically latest global_step checkpoint")
    parser.add_argument("directory")
    args = parser.parse_args()
    candidates = []
    for path in Path(args.directory).glob("global_step_*"):
        match = re.fullmatch(r"global_step_(\d+)", path.name)
        if match and path.is_dir():
            candidates.append((int(match.group(1)), path))
    if not candidates:
        raise SystemExit(f"no global_step_<integer> directory under {args.directory}")
    print(max(candidates)[1])


if __name__ == "__main__":
    main()
