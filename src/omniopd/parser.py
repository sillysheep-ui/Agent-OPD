from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Literal

_ACTION_RE = re.compile(r"(?im)^\s*Action\s*:\s*")


@dataclass(frozen=True)
class ParseResult:
    raw_candidate: str
    normalized_candidate: str
    canonical_action: str | None
    valid: bool
    had_action_marker: bool
    failure_reason: str | None


def _normalize(text: str) -> str:
    value = re.sub(r"\s+", " ", str(text or "").strip("`\"' \t\r\n"))
    value = value.rstrip(".;")
    match = re.fullmatch(r"put\s+(.+?)\s+(?:in|on|into|onto)\s+(.+)", value, re.I)
    if match:
        value = f"move {match.group(1)} to {match.group(2)}"
    return value.lower()


def parse_action(
    response: str,
    admissible: Iterable[str],
    *,
    marker_strategy: Literal["first", "last"] = "first",
    allow_unique_prefix: bool = False,
) -> ParseResult:
    text = str(response or "").strip()
    matches = list(_ACTION_RE.finditer(text))
    if not matches:
        reason = "empty_content" if not text else "missing_action_marker"
        return ParseResult("", "", None, False, False, reason)
    if len(matches) > 1:
        return ParseResult("", "", None, False, True, "multiple_action_markers")
    match = matches[0] if marker_strategy == "first" else matches[-1]
    candidate = text[match.end() :].splitlines()[0].strip()
    normalized = _normalize(candidate)
    if not normalized:
        return ParseResult(candidate, normalized, None, False, True, "empty_action")

    canonical: dict[str, list[str]] = {}
    for action in admissible:
        canonical.setdefault(_normalize(action), []).append(str(action).strip())
    if normalized in canonical:
        hits = canonical[normalized]
        if len(hits) != 1:
            return ParseResult(
                candidate, normalized, None, False, True, "ambiguous_normalized_action"
            )
        return ParseResult(candidate, normalized, hits[0], True, True, None)

    if allow_unique_prefix:
        hits = [
            action
            for norm, actions in canonical.items()
            if norm.startswith(normalized + " ")
            for action in actions
        ]
        if len(hits) == 1:
            return ParseResult(candidate, normalized, hits[0], True, True, None)
    return ParseResult(candidate, normalized, None, False, True, "not_admissible")
