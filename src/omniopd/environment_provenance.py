from __future__ import annotations

import hashlib
from importlib import metadata
import platform
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .provenance import fingerprint_path, sha256_json


REQUIRED_RUNTIME_DISTRIBUTIONS = (
    "alfworld",
    "textworld",
    "transformers",
    "torch",
    "openai",
)


def capture_runtime_dependencies(
    distributions: Sequence[str] = REQUIRED_RUNTIME_DISTRIBUTIONS,
    *,
    version_getter: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    """Capture installed versions and reject unverifiable confirmatory runtimes.

    A user-written label such as ``vllm:latest`` is not evidence about the
    Python environment which actually constructs prompts and runs ALFWorld.
    This snapshot is deliberately obtained from installed distribution
    metadata.  Missing metadata is an error instead of being serialized as an
    ambiguous ``unknown`` version.
    """

    getter = version_getter or metadata.version
    names = tuple(str(name).strip().lower() for name in distributions)
    if not names or any(not name for name in names) or len(names) != len(set(names)):
        raise ValueError("runtime distribution names must be unique and non-empty")
    versions: dict[str, str] = {}
    missing: list[str] = []
    for name in names:
        try:
            value = str(getter(name)).strip()
        except metadata.PackageNotFoundError:
            missing.append(name)
            continue
        if not value:
            missing.append(name)
            continue
        versions[name] = value
    if missing:
        raise RuntimeError(
            "confirmatory runtime dependency metadata is unavailable for: "
            + ", ".join(sorted(missing))
        )
    return {
        "schema_version": 1,
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "distributions": dict(sorted(versions.items())),
    }


def verify_runtime_dependencies(
    expected: Mapping[str, Any],
    *,
    actual: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Require the current runtime to equal a previously frozen snapshot."""

    expected_packages = expected.get("distributions")
    expected_python = expected.get("python")
    if (
        expected.get("schema_version") != 1
        or not isinstance(expected_packages, Mapping)
        or not isinstance(expected_python, Mapping)
    ):
        raise ValueError("runtime dependency snapshot is missing or malformed")
    required = set(REQUIRED_RUNTIME_DISTRIBUTIONS)
    if not required.issubset(expected_packages):
        missing = sorted(required - set(expected_packages))
        raise ValueError(f"runtime dependency snapshot omits required packages: {missing}")
    if any(
        not isinstance(expected_packages[name], str)
        or not str(expected_packages[name]).strip()
        for name in required
    ):
        raise ValueError("runtime dependency versions must be non-empty strings")
    observed = dict(actual) if actual is not None else capture_runtime_dependencies()
    if observed != dict(expected):
        raise RuntimeError(
            "current runtime dependencies differ from the frozen state-pool runtime"
        )
    return observed


def derive_environment_seed(master_seed: int, game_id: str) -> int:
    """Derive a stable per-game RNG seed without depending on Python hash()."""

    if master_seed < 0:
        raise ValueError("environment master seed must be non-negative")
    identifier = str(game_id)
    if not identifier:
        raise ValueError("game_id must be non-empty")
    digest = hashlib.sha256(
        f"omniopd-env-seed-v1\0{master_seed}\0{identifier}".encode("utf-8")
    ).digest()
    # Gym/TextWorld commonly route seeds through NumPy RandomState, whose
    # portable accepted range is [0, 2**32 - 1].
    return int.from_bytes(digest[:8], "big") % (2**32)


def validate_environment_seed_contract(
    contract: Mapping[str, Any],
    *,
    game_ids: Sequence[str],
    expected_master_seed: int | None = None,
) -> dict[str, int]:
    """Validate the frozen master→per-game TextWorld seed expansion."""

    games = [str(game) for game in game_ids]
    if not games or len(games) != len(set(games)):
        raise ValueError("environment seed validation needs unique non-empty games")
    if not isinstance(contract, Mapping):
        raise ValueError("environment rollout seed contract is missing")
    try:
        master_seed = int(contract["master_seed"])
        rows = list(contract["per_game"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("environment rollout seed contract is malformed") from error
    if master_seed < 0 or (
        expected_master_seed is not None and master_seed != expected_master_seed
    ):
        raise ValueError("environment master seed is invalid or differs from the run")
    if (
        contract.get("derivation") != "sha256_omniopd_env_seed_v1_uint32"
        or contract.get("adapter_seed_method")
        != "textworld_gym_env.seed_before_first_reset"
        or contract.get("alfred_demangler_shuffle") is not False
    ):
        raise ValueError("environment RNG implementation contract is unsupported")
    try:
        row_games = [str(row["game_id"]) for row in rows]
        seeds = {str(row["game_id"]): int(row["seed"]) for row in rows}
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("per-game environment seed rows are malformed") from error
    if row_games != games or len(seeds) != len(games):
        raise ValueError("per-game environment seeds do not match the ordered game list")
    if any(
        seed < 0
        or seed >= 2**32
        or seed != derive_environment_seed(master_seed, game)
        for game, seed in seeds.items()
    ):
        raise ValueError("one or more per-game environment seeds are invalid")
    return seeds


def fingerprint_game_artifacts(games: Iterable[str | Path]) -> list[dict[str, Any]]:
    """Fingerprint every executable game and, for files, its trial directory.

    The direct artifact binds the exact file (or full directory) passed to
    TextWorld.  A file's containing trial directory is also bound because
    ALFWorld reads sibling metadata such as ``traj_data.json`` to define task
    identity and semantics.
    """

    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_game in games:
        game_id = str(raw_game)
        if not game_id or game_id in seen:
            raise ValueError("game identifiers must be unique and non-empty")
        seen.add(game_id)
        target = Path(game_id)
        if not target.exists():
            raise FileNotFoundError(f"ALFWorld game artifact does not exist: {game_id}")
        direct = fingerprint_path(target)
        if direct.get("kind") not in {"file", "directory"}:
            raise ValueError(f"cannot fingerprint ALFWorld game artifact: {game_id}")
        row: dict[str, Any] = {
            "game_id": game_id,
            "artifact": direct,
        }
        if target.is_file():
            row["trial_directory"] = fingerprint_path(target.parent)
        result.append(row)
    return result


def verify_game_artifacts(expected: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Re-hash frozen game paths and require byte-for-byte manifest identity."""

    if not expected:
        raise ValueError("game artifact manifest must be non-empty")
    try:
        game_ids = [str(row["game_id"]) for row in expected]
    except (KeyError, TypeError) as error:
        raise ValueError("malformed game artifact manifest") from error
    observed = fingerprint_game_artifacts(game_ids)
    if observed != [dict(row) for row in expected]:
        raise RuntimeError("one or more ALFWorld game artifacts changed after freezing")
    return observed


def content_fingerprint_identity(fingerprint: Mapping[str, Any]) -> tuple[Any, ...]:
    """Return path-independent content identity for model/tokenizer bindings."""

    kind = fingerprint.get("kind")
    if kind == "file":
        return ("file", fingerprint.get("bytes"), fingerprint.get("sha256"))
    if kind == "directory":
        return (
            "directory",
            fingerprint.get("files"),
            fingerprint.get("bytes"),
            fingerprint.get("tree_sha256"),
        )
    raise ValueError("only resolved file/directory fingerprints have content identity")


def game_artifacts_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    """Compact checksum used to cross-bind manifests without replacing details."""

    return sha256_json([dict(row) for row in rows])
