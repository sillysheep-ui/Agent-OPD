#!/usr/bin/env python3
"""Audit Qwen3 Student/Teacher token-ID compatibility without loading weights."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from omniopd.opd_adapter import audit_shared_token_id_space
from omniopd.prompts import STUDENT_SYSTEM_PROMPT, TEACHER_SYSTEM_PROMPT
from omniopd.provenance import fingerprint_code_tree, git_revision, sha256_file
from omniopd.tokenization import apply_chat_template_ids

TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "chat_template.jinja",
)


def _file_hashes(root: Path) -> dict[str, str]:
    return {
        name: sha256_file(root / name)
        for name in TOKENIZER_FILES
        if (root / name).is_file()
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fail-closed token-ID audit before Qwen3 Student/Teacher OPD scoring"
    )
    parser.add_argument("--student-model", type=Path, required=True)
    parser.add_argument("--teacher-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    student_path = args.student_model.expanduser().resolve()
    teacher_path = args.teacher_model.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if student_path == teacher_path or not student_path.is_dir() or not teacher_path.is_dir():
        raise SystemExit("Student and Teacher must be distinct local model directories")
    if (
        output.exists()
        or output in {student_path, teacher_path}
        or student_path in output.parents
        or teacher_path in output.parents
    ):
        raise SystemExit("output must be a new file outside both model directories")
    for role, path in (("Student", student_path), ("Teacher", teacher_path)):
        if not (path / "config.json").is_file() or not (path / "tokenizer.json").is_file():
            raise SystemExit(f"{role} requires local config.json and tokenizer.json: {path}")

    from transformers import AutoConfig, AutoTokenizer

    student_config = AutoConfig.from_pretrained(
        str(student_path), local_files_only=True, trust_remote_code=False
    )
    teacher_config = AutoConfig.from_pretrained(
        str(teacher_path), local_files_only=True, trust_remote_code=False
    )
    if student_config.model_type != "qwen3" or teacher_config.model_type != "qwen3":
        raise SystemExit("this OPD preflight currently supports Qwen3 text models only")
    student = AutoTokenizer.from_pretrained(
        str(student_path), use_fast=True, local_files_only=True, trust_remote_code=False
    )
    teacher = AutoTokenizer.from_pretrained(
        str(teacher_path), use_fast=True, local_files_only=True, trust_remote_code=False
    )
    if not student.is_fast or not teacher.is_fast:
        raise SystemExit("both tokenizers must be fast tokenizers")
    audit = audit_shared_token_id_space(student, teacher)
    user_message = {"role": "user", "content": "Observation: a book.\nAdmissible: take book"}
    student_prompt_ids = apply_chat_template_ids(
        student,
        [{"role": "system", "content": STUDENT_SYSTEM_PROMPT}, user_message],
        add_generation_prompt=True,
        enable_thinking=False,
    )
    teacher_prompt_ids = apply_chat_template_ids(
        teacher,
        [{"role": "system", "content": TEACHER_SYSTEM_PROMPT}, user_message],
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if not student_prompt_ids or not teacher_prompt_ids:
        raise SystemExit("a canonical non-thinking prompt encoded to zero tokens")
    sample_action = "Action: take book"
    student_action_ids = student.encode(sample_action, add_special_tokens=False)
    teacher_action_ids = teacher.encode(sample_action, add_special_tokens=False)
    if not student_action_ids or student_action_ids != teacher_action_ids:
        raise SystemExit("Student action tokens do not have the same Teacher token IDs")

    revision = git_revision(ROOT)
    if not revision:
        raise SystemExit("OPD tokenizer audit requires an immutable OmniOPD Git revision")
    payload = {
        "artifact": "omniopd_opd_tokenizer_compatibility",
        "schema_version": 1,
        "code_revision": revision,
        "code": fingerprint_code_tree(ROOT),
        "student_model": {
            "path": str(student_path),
            "config_sha256": sha256_file(student_path / "config.json"),
            "tokenizer_files_sha256": _file_hashes(student_path),
        },
        "teacher_model": {
            "path": str(teacher_path),
            "config_sha256": sha256_file(teacher_path / "config.json"),
            "tokenizer_files_sha256": _file_hashes(teacher_path),
        },
        "token_id_space": audit,
        "nonthinking_prompt_lengths": {
            "student": len(student_prompt_ids),
            "teacher": len(teacher_prompt_ids),
        },
        "sample_action_token_ids": student_action_ids,
        "teacher_uses_distinct_system_prompt": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"Qwen3 Student/Teacher token-ID audit passed: {output}")


if __name__ == "__main__":
    main()
