from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from .io import read_jsonl


def stable_record_id(record: dict[str, Any]) -> str:
    if "state" in record and record["state"].get("state_hash"):
        state_id = record["state"]["state_hash"]
    elif "state_hash" in record:
        state_id = record["state_hash"]
    else:
        state_id = f"{record.get('game_id') or record.get('gamefile')}:{record.get('turn_index')}"
    namespace = record.get("group") or record.get("checkpoint") or ""
    if "sample_id" in record:
        sample = record["sample_id"]
    else:
        sample = record.get("teacher_sample_index", "")
    return f"{namespace}:{state_id}:{sample}"


def merge_records(paths: Sequence[str | Path]) -> list[dict[str, Any]]:
    if not paths:
        raise ValueError("at least one input file is required")
    records = []
    seen = set()
    for path in sorted((Path(path) for path in paths), key=lambda value: str(value)):
        for record in read_jsonl(path):
            record_id = stable_record_id(record)
            if record_id in seen:
                raise ValueError(f"duplicate record during merge: {record_id}")
            seen.add(record_id)
            records.append(record)
    return sorted(records, key=stable_record_id)


def deterministic_jsonl_bytes(records: Iterable[dict[str, Any]]) -> bytes:
    return b"".join(
        (
            json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        for record in records
    )


def records_sha256(records: Iterable[dict[str, Any]]) -> str:
    return hashlib.sha256(deterministic_jsonl_bytes(records)).hexdigest()
