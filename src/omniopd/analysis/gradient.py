from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any, Iterable

from ..sampling import SamplingProtocol


@dataclass(frozen=True)
class GradientAlignment:
    dot_product: float
    cosine: float
    correction_norm: float
    reference_norm: float


def gradient_alignment(correction: Any, reference: Any) -> GradientAlignment:
    import torch

    correction = correction.float().reshape(-1)
    reference = reference.float().reshape(-1)
    if correction.shape != reference.shape:
        raise ValueError("gradient vectors have different shapes")
    dot = torch.dot(correction, reference)
    correction_norm = torch.linalg.vector_norm(correction)
    reference_norm = torch.linalg.vector_norm(reference)
    if not bool(
        torch.isfinite(dot)
        & torch.isfinite(correction_norm)
        & torch.isfinite(reference_norm)
    ):
        raise ValueError("gradient alignment requires finite vectors")
    if correction_norm.item() == 0.0 or reference_norm.item() == 0.0:
        raise ValueError("gradient cosine is undefined for a zero-norm vector")
    cosine = dot / (correction_norm * reference_norm)
    if not bool(torch.isfinite(cosine)):
        raise ValueError("gradient cosine is numerically undefined")
    return GradientAlignment(
        float(dot), float(cosine), float(correction_norm), float(reference_norm)
    )


def online_mean(vectors: Iterable[Any]) -> Any:
    """Avoid retaining every high-dimensional LoRA gradient in memory."""

    mean = None
    count = 0
    for vector in vectors:
        value = vector.detach().float()
        count += 1
        mean = value.clone() if mean is None else mean + (value - mean) / count
    if mean is None:
        raise ValueError("at least one gradient vector is required")
    return mean


def assert_held_out_reference_games(
    correction_games: Iterable[str], reference_games: Iterable[str]
) -> None:
    """Prevent a selection-defined complement from masquerading as G_*."""

    correction = set(correction_games)
    reference = set(reference_games)
    if not reference:
        raise ValueError("the reference-gradient game set cannot be empty")
    overlap = correction & reference
    if overlap:
        raise ValueError(
            f"reference gradients must use preregistered held-out games; overlap={sorted(overlap)[:5]}"
        )


def artifact_identity(value: Any) -> Any:
    """Remove location-only fields while retaining an artifact's content identity."""

    if not isinstance(value, dict):
        return value
    return {
        key: artifact_identity(item)
        for key, item in value.items()
        if key not in {"path", "resolved_path", "value"}
    }


def teacher_contract_from_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Return the complete Teacher contract or reject an unverifiable manifest.

    M1 compares gradients produced from sampled Teacher actions.  A model name on
    its own is not enough provenance: decoding, prompt, tokenizer, provider
    revision, call accounting, and the state/action budget all change the target
    distribution.  This helper intentionally fails closed when any such field is
    absent or internally inconsistent.
    """

    required_strings = [
        "teacher_model",
        "teacher_model_revision",
        "teacher_url",
        "teacher_prompt_sha256",
    ]
    missing_strings = [
        key
        for key in required_strings
        if not isinstance(manifest.get(key), str) or not manifest[key].strip()
    ]
    if missing_strings:
        raise ValueError(f"Teacher manifest lacks non-empty fields: {missing_strings}")
    if manifest.get("artifact") != "teacher_corrections":
        raise ValueError("M1 correction manifest has the wrong artifact type")
    if manifest.get("protocol_version") != "omniopd-v1":
        raise ValueError("M1 correction manifest has the wrong protocol version")
    if manifest.get("invalid_calls_count_toward_budget") is not True:
        raise ValueError("M1 requires invalid Teacher calls to count toward budget")
    try:
        states = int(manifest["distinct_states_M"])
        samples = int(manifest["teacher_samples_per_state_N"])
        declared = int(manifest["declared_teacher_budget_B"])
        actual = int(manifest["actual_teacher_api_calls"])
        free_retries = int(manifest["free_retries"])
        max_tokens = int(manifest["teacher_max_tokens"])
        context_window = int(manifest["teacher_context_window"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"M1 Teacher budget contract is incomplete: {error}") from error
    if (
        states <= 0
        or samples <= 0
        or declared != states * samples
        or actual != declared
        or free_retries != 0
        or max_tokens <= 0
        or context_window <= max_tokens
    ):
        raise ValueError("M1 Teacher budget/context contract is internally inconsistent")
    sampling = manifest.get("teacher_sampling")
    if not isinstance(sampling, dict):
        raise ValueError("M1 Teacher sampling contract is missing")
    if sampling.get("thinking_mode") not in {
        "enabled",
        "disabled",
        "provider_default",
    }:
        raise ValueError("M1 Teacher thinking mode is not canonical")
    try:
        SamplingProtocol(
            str(sampling["thinking_mode"]),
            sampling.get("temperature"),
            sampling.get("reasoning_effort"),
        ).validate(samples_per_state=samples)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"M1 Teacher sampling contract is invalid: {error}") from error
    tokenizer = manifest.get("teacher_tokenizer")
    if not isinstance(tokenizer, dict) or not (
        tokenizer.get("sha256") or tokenizer.get("tree_sha256")
    ):
        raise ValueError("M1 Teacher tokenizer lacks a content fingerprint")
    profile = manifest.get("teacher_profile")
    if not isinstance(profile, dict) or not (
        profile.get("sha256") or profile.get("tree_sha256")
    ):
        raise ValueError("M1 Teacher profile lacks a content fingerprint")
    preflight = manifest.get("teacher_context_preflight")
    if not isinstance(preflight, dict) or preflight.get("passed") is not True:
        raise ValueError("M1 Teacher context preflight was not attested")
    provider_models = manifest.get("provider_response_models")
    provider_fingerprints = manifest.get("provider_system_fingerprints")
    if (
        not isinstance(provider_models, list)
        or len(provider_models) != 1
        or provider_models != [manifest["teacher_model"]]
        or not isinstance(provider_fingerprints, list)
        or len(provider_fingerprints) > 1
    ):
        raise ValueError(
            "M1 Teacher provider identity is missing or response.model differs "
            "from the requested model"
        )
    request_ledger_sha256 = manifest.get("teacher_requests_sha256")
    if not isinstance(request_ledger_sha256, str) or not request_ledger_sha256:
        raise ValueError("M1 Teacher request ledger digest is missing")
    selected_hashes = manifest.get("selected_state_hashes")
    if (
        not isinstance(selected_hashes, list)
        or len(selected_hashes) != states
        or len({str(value) for value in selected_hashes}) != states
    ):
        raise ValueError("M1 correction manifest does not identify exactly M states")
    valid_samples = manifest.get("valid_teacher_samples")
    invalid_samples = manifest.get("invalid_teacher_samples")
    if (
        not isinstance(valid_samples, int)
        or not isinstance(invalid_samples, int)
        or valid_samples < 0
        or invalid_samples < 0
        or valid_samples + invalid_samples != actual
    ):
        raise ValueError("M1 Teacher valid/invalid accounting is inconsistent")
    contract = {
        "teacher_model": manifest["teacher_model"],
        "teacher_model_revision": manifest["teacher_model_revision"],
        "teacher_url": manifest["teacher_url"],
        "teacher_sampling": sampling,
        "teacher_profile": artifact_identity(profile),
        "teacher_max_tokens": max_tokens,
        "teacher_context_window": context_window,
        "teacher_tokenizer": artifact_identity(tokenizer),
        "teacher_prompt_sha256": manifest["teacher_prompt_sha256"],
        "provider_response_models": sorted(str(value) for value in provider_models),
        "provider_system_fingerprints": sorted(
            str(value) for value in provider_fingerprints
        ),
        # Each dataset necessarily has a different request ledger.  Its digest
        # is validated above and preserved by the per-input manifest binding,
        # but it must not be compared as though it were a shared sampling
        # hyperparameter across reference/A1/A3.
        "teacher_request_ledger_bound": bool(request_ledger_sha256),
        "invalid_calls_count_toward_budget": True,
        "free_retries": 0,
    }
    # Ensure the returned object itself is finite/canonically serializable.
    json.dumps(contract, sort_keys=True, allow_nan=False)
    return contract


def assert_shared_teacher_contract(contracts: Iterable[dict[str, Any]]) -> None:
    serialized = {
        json.dumps(contract, sort_keys=True, allow_nan=False)
        for contract in contracts
    }
    if len(serialized) != 1:
        raise ValueError(
            "M1 reference and correction groups must share one exact Teacher contract"
        )


def validate_state_weights(rows: Iterable[dict[str, Any]]) -> None:
    """Reject silent non-finite objective weights before gradient computation."""

    for row in rows:
        try:
            value = float(row["state_weight"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("M1 row has an invalid state weight") from error
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("M1 state weights must be finite and positive")
