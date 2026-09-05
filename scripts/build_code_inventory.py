#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from omniopd.provenance import fingerprint_code_tree, git_revision, sha256_file

INVENTORIED_ROOTS = ("src", "scripts", "integrations", "configs", "tests", "docs", "legacy")
ROOT_FILES = ("README.md", "pyproject.toml", ".gitignore")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a deterministic file/hash inventory for this complete codebase"
    )
    parser.add_argument("--output", default="docs/CODE_INVENTORY.json")
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite code inventory: {output}")
    try:
        output.relative_to(ROOT)
    except ValueError as error:
        raise SystemExit("inventory output must stay inside the repository") from error

    files = []
    for name in INVENTORIED_ROOTS:
        directory = ROOT / name
        files.extend(path for path in directory.rglob("*") if path.is_file())
    files.extend(ROOT / name for name in ROOT_FILES if (ROOT / name).is_file())
    excluded_parts = {"__pycache__", ".pytest_cache", ".ruff_cache"}
    files = sorted(
        {
            path
            for path in files
            if path.resolve() != output
            and not excluded_parts.intersection(path.parts)
            and path.suffix not in {".pyc", ".pyo"}
        },
        key=lambda path: path.relative_to(ROOT).as_posix(),
    )
    counts = Counter(
        path.relative_to(ROOT).parts[0]
        if len(path.relative_to(ROOT).parts) > 1
        else "repository_root"
        for path in files
    )
    entries = [
        {
            "path": path.relative_to(ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "role": (
                "read_only_legacy_snapshot"
                if path.relative_to(ROOT).parts[0] == "legacy"
                else "canonical_or_documentation"
            ),
        }
        for path in files
    ]
    payload = {
        "schema_version": 1,
        "artifact": "complete_codebase_inventory",
        "repository": str(ROOT),
        "code_revision_at_generation": git_revision(ROOT),
        "canonical_code": fingerprint_code_tree(ROOT),
        "files": len(entries),
        "bytes": sum(entry["bytes"] for entry in entries),
        "counts_by_root": dict(sorted(counts.items())),
        "entries": entries,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
