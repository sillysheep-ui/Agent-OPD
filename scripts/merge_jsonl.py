#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from omniopd.merge import deterministic_jsonl_bytes, merge_records, records_sha256
from omniopd.provenance import fingerprint_code_tree, git_revision, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministic, duplicate-safe JSONL merge")
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    records = merge_records(args.input)
    output = Path(args.output)
    manifest_output = output.with_suffix(output.suffix + ".manifest.json")
    temporary = output.with_suffix(output.suffix + ".tmp")
    if output.exists() or manifest_output.exists() or temporary.exists():
        raise SystemExit("refusing to overwrite a merge artifact, manifest, or stale temp file")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = deterministic_jsonl_bytes(records)
    temporary.write_bytes(payload)
    temporary.replace(output)
    manifest = {
        "protocol_version": "omniopd-v1",
        "artifact": "deterministic_jsonl_merge",
        "code": fingerprint_code_tree(ROOT),
        "code_revision": git_revision(ROOT),
        "inputs": [
            {"path": path, "sha256": sha256_file(path)} for path in sorted(args.input)
        ],
        "records": len(records),
        "sha256": records_sha256(records),
    }
    manifest_output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
