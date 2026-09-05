#!/usr/bin/env python3
"""Run the repository's function-style tests without third-party test runners."""

from __future__ import annotations

import importlib.util
import inspect
import sys
import traceback
import unittest
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    failures = []
    skipped = []
    count = 0
    for path in sorted((root / "tests").glob("test_*.py")):
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec is None or spec.loader is None:
            failures.append((str(path), "could not load module"))
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for name, function in inspect.getmembers(module, inspect.isfunction):
            if not name.startswith("test_"):
                continue
            count += 1
            try:
                function()
            except unittest.SkipTest as error:
                skipped.append((f"{path.name}::{name}", str(error)))
            except Exception:
                failures.append((f"{path.name}::{name}", traceback.format_exc()))
    if failures:
        for name, details in failures:
            print(f"FAIL {name}\n{details}")
        print(f"{len(failures)} failed, {count - len(failures) - len(skipped)} passed, {len(skipped)} skipped")
        return 1
    for name, reason in skipped:
        print(f"SKIP {name}: {reason}")
    print(f"{count - len(skipped)} passed, {len(skipped)} skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
