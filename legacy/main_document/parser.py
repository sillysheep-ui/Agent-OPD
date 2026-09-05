from __future__ import annotations
 
import re
from dataclasses import asdict, dataclass
from typing import Iterable
 
 
_ACTION_RE = re.compile(r"(?im)^\s*Action\s*:\s*")
 
 
@dataclass(frozen=True)
class CanonicalizationResult:
    raw_candidate: str
    normalized_candidate: str
    canonical_action: str | None
    valid: bool
    used_put_to_move: bool
    used_unique_prefix: bool
    had_action_marker: bool
 
    def to_dict(self) -> dict:
        return asdict(self)
 
 
def extract_action(response: str, marker_strategy: str = "first") -> tuple[str, bool]:
    """Extract one Action line deterministically.
 
    The v5 protocol freezes the parser to an explicit Action: marker.  If no
    marker exists, parsing fails instead of guessing from arbitrary prose.
 
    Returns:
        (candidate_action, had_action_marker)
    """
    text = str(response or "").strip()
    matches = list(_ACTION_RE.finditer(text))
    if not matches:
        return "", False
 
    m = matches[0] if marker_strategy == "first" else matches[-1]
    tail = text[m.end():]
    first_line = tail.splitlines()[0].strip() if tail else ""
    return first_line, True
 
 
def normalize_action(text: str) -> tuple[str, bool]:
    """Normalize an ALFWorld command without changing its decision semantics.
 
    Only deterministic grammar normalization is allowed:
      put X in/on/into/onto Y -> move X to Y
 
    Returns:
        (normalized_action, used_put_to_move)
    """
    raw = str(text or "").strip()
    raw = raw.strip("`\"' ")
    raw = re.sub(r"\s+", " ", raw)
    raw = raw.rstrip(".;")
 
    used_put_to_move = False
    m = re.match(
        r"^put\s+(.+?)\s+(?:in|on|into|onto)\s+(.+)$",
        raw,
        flags=re.IGNORECASE,
    )
    if m:
        raw = f"move {m.group(1)} to {m.group(2)}"
        used_put_to_move = True
 
    return raw.strip().lower(), used_put_to_move
 
 
def canonical_action(
    candidate: str,
    admissible: Iterable[str],
    *,
    had_action_marker: bool = True,
) -> CanonicalizationResult:
    """Map a parsed candidate to one legal ALFWorld command.
 
    Rules:
      1. exact match after deterministic normalization;
      2. unique-prefix completion only when exactly one admissible command
         extends the normalized candidate (e.g. missing object index);
      3. no fuzzy matching, semantic search, or LLM judge.
    """
    admissible_list = [str(a).strip() for a in admissible]
    normalized_candidate, used_put_to_move = normalize_action(candidate)
 
    if not had_action_marker or not normalized_candidate:
        return CanonicalizationResult(
            raw_candidate=str(candidate or ""),
            normalized_candidate=normalized_candidate,
            canonical_action=None,
            valid=False,
            used_put_to_move=used_put_to_move,
            used_unique_prefix=False,
            had_action_marker=had_action_marker,
        )
 
    table: dict[str, str] = {}
    for action in admissible_list:
        norm, _ = normalize_action(action)
        table[norm] = action
 
    if normalized_candidate in table:
        return CanonicalizationResult(
            raw_candidate=candidate,
            normalized_candidate=normalized_candidate,
            canonical_action=table[normalized_candidate],
            valid=True,
            used_put_to_move=used_put_to_move,
            used_unique_prefix=False,
            had_action_marker=had_action_marker,
        )
 
    prefix_hits: list[str] = []
    for cmd in admissible_list:
        norm_cmd, _ = normalize_action(cmd)
        if norm_cmd.startswith(normalized_candidate + " "):
            prefix_hits.append(cmd)
 
    if len(prefix_hits) == 1:
        return CanonicalizationResult(
            raw_candidate=candidate,
            normalized_candidate=normalized_candidate,
            canonical_action=prefix_hits[0],
            valid=True,
            used_put_to_move=used_put_to_move,
            used_unique_prefix=True,
            had_action_marker=had_action_marker,
        )
 
    return CanonicalizationResult(
        raw_candidate=candidate,
        normalized_candidate=normalized_candidate,
        canonical_action=None,
        valid=False,
        used_put_to_move=used_put_to_move,
        used_unique_prefix=False,
        had_action_marker=had_action_marker,
    )
 
 
def parse_and_canonicalize(
    response: str,
    admissible: Iterable[str],
    *,
    marker_strategy: str = "first",
) -> CanonicalizationResult:
    candidate, had_marker = extract_action(response, marker_strategy)
    return canonical_action(candidate, admissible, had_action_marker=had_marker)
 
 
def canonicalize_step_response(raw_response: str, action: str) -> str:
    """Keep only text before the first Action marker, then write canonical action.
 
    This intentionally discards anything after the first Action line, including
    hallucinated future observations/turns.
    """
    text = str(raw_response or "").strip()
    m = _ACTION_RE.search(text)
    if m:
        prefix = text[:m.start()].rstrip()
        if prefix:
            return f"{prefix}\nAction: {action}"
    return f"Action: {action}"
 
 
def build_target_response(raw_response: str, action: str, mode: str) -> str:
    """Build the supervised Teacher target.
 
    v5 defaults to action-only because the formal Route-A objective is written on
    the canonical action/step target.  `full` is retained only as an explicit
    ablation and keeps Teacher text before the first Action marker.
    """
    if mode == "action":
        return f"Action: {action}"
    if mode == "full":
        return canonicalize_step_response(raw_response, action)
    raise ValueError(f"Unknown target mode: {mode}")
