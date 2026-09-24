#!/usr/bin/env python3
"""Step-by-step acceptance harness for the Agent OPD pipeline.

Every stage gets one contract and one cheap, re-runnable check, so a later
mistake is caught in the stage that caused it instead of at the final score:

  S1 states     pool has unique hashes and non-empty admissible actions
  S2 selection  registered policy, inclusion probability recorded, games distinct
  S3 teacher    attempts == attempt ledger == request ledger, per-request hash
                recorded, acceptance reported at sample/state/game level
  S4 rows       every game's weight sum is 1, mask length == input length,
                at least one supervised token, supervision is an action suffix
  S5 training   report exists, loss decreased, ||dW||/||W|| inside 1e-3..1e-2
  S6 eval       closed-loop JSON parses; with two models, paired table + McNemar

Stages with missing artifacts are reported as PENDING, not as failures.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

RESULTS: list[tuple[str, str, str]] = []


def report(stage: str, status: str, detail: str) -> None:
    RESULTS.append((stage, status, detail))


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def check_states(pool: Path) -> list[dict]:
    episodes = load_jsonl(pool)
    if not episodes:
        report("S1 states", "PENDING", f"{pool} not found")
        return []
    turns = [turn for episode in episodes for turn in episode.get("turns", [])]
    hashes = [turn["state"]["state_hash"] for turn in turns]
    empty = sum(1 for turn in turns if not turn["state"].get("admissible_actions"))
    unique = len(set(hashes)) == len(hashes)
    ok = bool(turns) and empty == 0 and unique
    report("S1 states", "PASS" if ok else "FAIL",
           f"{len(episodes)} games / {len(turns)} turns, unique_hash={unique}, empty_admissible={empty}")
    return turns


def check_dataset(directory: Path) -> None:
    records = load_jsonl(directory / "records.jsonl")
    ledger = load_jsonl(directory / "attempt_ledger.jsonl")
    requests = load_jsonl(directory / "requests.jsonl")
    if not records:
        report("S2 selection", "PENDING", f"{directory}/records.jsonl not found")
        report("S3 teacher", "PENDING", "no records")
    else:
        policies = {row.get("selection_policy") for row in records}
        probs = [row.get("inclusion_probability") for row in records]
        games = {row["state"]["game_id"] for row in records}
        ok = policies == {"uniform_per_game_nested_v1"} and all(
            isinstance(p, (int, float)) and 0 < p <= 1 for p in probs
        )
        report("S2 selection", "PASS" if ok else "FAIL",
               f"policy={sorted(policies)}, games={len(games)}, inclusion_prob "
               f"min={min(p for p in probs if p)} max={max(p for p in probs if p)}")
        attempts = sum(row.get("teacher_calls", 0) for row in records)
        valid = sum(1 for row in ledger if row.get("valid"))
        states_ok = sum(1 for row in records if any(s["valid"] for s in row["teacher_samples"]))
        games_ok = len({row["state"]["game_id"] for row in records
                        if any(s["valid"] for s in row["teacher_samples"])})
        hashes = [row.get("messages_sha256") for row in requests]
        hash_ok = bool(hashes) and all(isinstance(h, str) and len(h) == 64 for h in hashes)
        ok = bool(ledger) and attempts == len(ledger) and len(ledger) == len(requests) and hash_ok
        report("S3 teacher", "PASS" if ok else "FAIL",
               f"attempts={attempts} ledger={len(ledger)} requests={len(requests)} "
               f"hash_recorded={hash_ok} | sample {valid}/{len(ledger)}, "
               f"state {states_ok}/{len(records)}, game {games_ok}/{len({r['state']['game_id'] for r in records})}")


def check_rows(directory: Path) -> None:
    rows = load_jsonl(directory / "sft_rows.jsonl")
    if not rows:
        report("S4 rows", "PENDING", f"{directory}/sft_rows.jsonl not found")
        return
    per_game = defaultdict(float)
    for row in rows:
        per_game[row["game_id"]] += float(row["state_weight"])
    worst = max(abs(value - 1.0) for value in per_game.values())
    length_ok = all(len(row["input_ids"]) == len(row["target_token_mask"]) for row in rows)
    tokens_ok = all(sum(row["target_token_mask"]) >= 1 for row in rows)
    suffix_ok = all(
        row["target_token_mask"].index(1) > len(row["target_token_mask"]) * 0.5
        and sum(row["target_token_mask"]) == len(row["target_token_mask"]) - row["target_token_mask"].index(1)
        for row in rows
        if any(row["target_token_mask"])
    )
    ok = worst < 1e-9 and length_ok and tokens_ok and suffix_ok
    report("S4 rows", "PASS" if ok else "FAIL",
           f"{len(rows)} rows / {len(per_game)} games, |weight_sum-1|max={worst:.2e}, "
           f"length={length_ok}, tokens={tokens_ok}, action_suffix={suffix_ok}")


def check_training(directory: Path) -> None:
    path = directory / "training_report.json"
    if not path.is_file():
        report("S5 training", "PENDING", f"{path} not found")
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    decreased = data["loss_last"] < data["loss_first"]
    in_band = 1e-3 <= data["delta_ratio"] <= 1e-2
    nonzero = (data.get("grad_norm_first_step") or 0) > 0
    ok = decreased and in_band and nonzero
    report("S5 training", "PASS" if ok else "FAIL",
           f"loss {data['loss_first']:.4f}->{data['loss_last']:.4f}, "
           f"grad_norm {data.get('grad_norm_first_step'):.3g}, "
           f"|dW|/|W|={data['delta_ratio']:.3e} in_band={in_band}")


def check_eval(directory: Path) -> None:
    files = {name: directory / f"closed_loop_{name}.json" for name in ("opd", "base4b")}
    present = {name: path for name, path in files.items() if path.is_file()}
    if not present:
        report("S6 eval", "PENDING", f"no closed_loop_*.json under {directory}")
        return
    summaries = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in present.items()}
    detail = ", ".join(f"{name} {s['summary']['wins']}/{s['summary']['games']}"
                       for name, s in summaries.items())
    if len(summaries) == 2 and present["opd"].is_file() and present["base4b"].is_file():
        a = {e["game_id"]: e for e in summaries["opd"]["episodes"]}
        b = {e["game_id"]: e for e in summaries["base4b"]["episodes"]}
        shared = sorted(set(a) & set(b))
        both = opd_only = base_only = neither = 0
        for game in shared:
            ow, bw = bool(a[game]["won"]), bool(b[game]["won"])
            both += ow and bw
            opd_only += ow and not bw
            base_only += bw and not ow
            neither += not ow and not bw
        n = opd_only + base_only
        p = (sum(math.comb(n, i) for i in range(min(opd_only, base_only) + 1)) / 2 ** n * 2) if n else 1.0
        detail += (" | paired %d: both=%d opd_only=%d base_only=%d neither=%d net=%+d McNemar_p=%.4f"
                   % (len(shared), both, opd_only, base_only, neither, opd_only - base_only, p))
    report("S6 eval", "PASS", detail)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True, help="MVP output dir (records/rows)")
    parser.add_argument("--training", type=Path, required=True, help="training output dir")
    parser.add_argument("--eval", type=Path, required=True, help="evaluation output dir")
    args = parser.parse_args()

    check_states(args.pool)
    check_dataset(args.dataset)
    check_rows(args.dataset)
    check_training(args.training)
    check_eval(args.eval)

    width = max(len(stage) for stage, _, _ in RESULTS)
    for stage, status, detail in RESULTS:
        print(f"{stage:<{width}} {status:<8} {detail}")
    failures = [row for row in RESULTS if row[1] == "FAIL"]
    print(f"\n{len(RESULTS) - len(failures)}/{len(RESULTS)} stages pass"
          + ("" if not failures else f"; FAILED: {[row[0] for row in failures]}"))
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
