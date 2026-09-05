from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

from omniopd.dataset import audit_acceptance
from omniopd.io import correction_from_dict, read_jsonl
from omniopd.validation import audit_correction_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit an OmniOPD correction JSONL")
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    records = [correction_from_dict(item) for item in read_jsonl(args.input)]
    if not records:
        raise SystemExit("correction audit input is empty")
    calls = sum(record.teacher_calls for record in records)
    policies = Counter(record.selection_policy for record in records)
    by_game = defaultdict(lambda: {"selected": 0, "valid": 0})
    for record in records:
        game = by_game[record.state.game_id]
        game["selected"] += 1
        game["valid"] += int(bool(record.valid_teacher_samples))
    issues = audit_correction_records(records)
    report = {
        "records": len(records),
        "teacher_api_calls": calls,
        "selection_policies": policies,
        "protocol_issues": [issue.__dict__ for issue in issues],
        "acceptance": audit_acceptance(records),
        "per_game": dict(sorted(by_game.items())),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if issues:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
