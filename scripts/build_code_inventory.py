#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from omniopd.provenance import fingerprint_code_tree, git_revision, sha256_file

INVENTORIED_ROOTS = ("src", "scripts", "integrations", "configs", "tests", "docs", "legacy")
ROOT_FILES = ("README.md", "pyproject.toml", ".gitignore")


def _git_paths(*args: str) -> set[str]:
    completed = subprocess.run(
        ["git", *args, "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return {item for item in completed.stdout.split("\0") if item}


def _is_inventoried(relative: Path) -> bool:
    parts = relative.parts
    return bool(parts) and (parts[0] in INVENTORIED_ROOTS or relative.as_posix() in ROOT_FILES)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a deterministic file/hash inventory for this complete codebase"
    )
    parser.add_argument("--output", default="docs/CODE_INVENTORY.json")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="atomically replace an existing inventory after all other files are clean",
    )
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if output.exists() and not args.overwrite:
        raise SystemExit(f"refusing to overwrite code inventory: {output}")
    try:
        output_relative = output.relative_to(ROOT)
    except ValueError as error:
        raise SystemExit("inventory output must stay inside the repository") from error

    output_key = output_relative.as_posix()
    dirty = set()
    dirty.update(_git_paths("diff", "--name-only"))
    dirty.update(_git_paths("diff", "--cached", "--name-only"))
    dirty.update(_git_paths("ls-files", "--others", "--exclude-standard"))
    dirty.discard(output_key)
    if dirty:
        preview = ", ".join(sorted(dirty)[:8])
        suffix = " ..." if len(dirty) > 8 else ""
        raise SystemExit(
            "refusing to inventory a dirty repository (excluding the inventory itself): "
            f"{preview}{suffix}"
        )

    tracked = _git_paths("ls-files")
    files = [
        ROOT / name
        for name in tracked
        if name != output_key and _is_inventoried(Path(name))
    ]
    missing = [path.relative_to(ROOT).as_posix() for path in files if not path.is_file()]
    if missing:
        raise SystemExit(f"tracked inventory inputs are missing: {', '.join(sorted(missing))}")
    files.sort(key=lambda path: path.relative_to(ROOT).as_posix())
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
        "repository": ".",
        "code_revision_at_generation": git_revision(ROOT),
        "canonical_code": fingerprint_code_tree(ROOT),
        "files": len(entries),
        "bytes": sum(entry["bytes"] for entry in entries),
        "counts_by_root": dict(sorted(counts.items())),
        "entries": entries,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.",
        suffix=".tmp",
        dir=output.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, output)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    main()
