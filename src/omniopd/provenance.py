from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    """Hash a JSON value under one canonical, finite-number serialization."""

    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def fingerprint_path(path: str | Path) -> dict[str, Any]:
    """Fingerprint a file or a complete directory tree without path-order drift."""

    target = Path(path)
    if target.is_file():
        return {
            "path": str(target.resolve()),
            "kind": "file",
            "bytes": target.stat().st_size,
            "sha256": sha256_file(target),
        }
    if target.is_dir():
        digest = hashlib.sha256()
        count = 0
        total_bytes = 0
        for child in sorted(item for item in target.rglob("*") if item.is_file()):
            relative = child.relative_to(target).as_posix()
            size = child.stat().st_size
            digest.update(relative.encode("utf-8") + b"\0")
            digest.update(str(size).encode("ascii") + b"\0")
            digest.update(sha256_file(child).encode("ascii") + b"\n")
            count += 1
            total_bytes += size
        return {
            "path": str(target.resolve()),
            "kind": "directory",
            "files": count,
            "bytes": total_bytes,
            "tree_sha256": digest.hexdigest(),
        }
    return {"path_or_identifier": str(path), "kind": "unresolved_identifier"}


def fingerprint_code_tree(repository_root: str | Path) -> dict[str, Any]:
    """Hash only canonical source/config files, excluding legacy and caches."""

    root = Path(repository_root).resolve()
    candidates: list[Path] = []
    for relative in ["src", "scripts", "integrations", "configs", "docker"]:
        directory = root / relative
        if directory.exists():
            candidates.extend(
                child
                for child in directory.rglob("*")
                if child.is_file()
                and "__pycache__" not in child.parts
                and child.suffix not in {".pyc", ".pyo"}
            )
    for relative in ["pyproject.toml", "README.md"]:
        path = root / relative
        if path.is_file():
            candidates.append(path)
    digest = hashlib.sha256()
    total_bytes = 0
    unique = sorted(set(candidates), key=lambda path: path.relative_to(root).as_posix())
    for child in unique:
        relative = child.relative_to(root).as_posix()
        size = child.stat().st_size
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(str(size).encode("ascii") + b"\0")
        digest.update(sha256_file(child).encode("ascii") + b"\n")
        total_bytes += size
    if not unique:
        raise ValueError(f"no canonical code files found under {root}")
    return {
        "tree_sha256": digest.hexdigest(),
        "files": len(unique),
        "bytes": total_bytes,
    }


def git_revision(root: str | Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


@dataclass(frozen=True)
class InputArtifact:
    role: str
    path: str
    sha256: str


def build_manifest(
    *,
    protocol: dict[str, Any],
    inputs: Iterable[tuple[str, str | Path]],
    outputs: Iterable[str | Path],
    repository_root: str | Path,
    notes: Iterable[str] = (),
) -> dict[str, Any]:
    artifacts = [
        InputArtifact(role, str(Path(path)), sha256_file(path)) for role, path in inputs
    ]
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": protocol,
        "inputs": [asdict(artifact) for artifact in artifacts],
        "outputs": [
            {"path": str(Path(path)), "sha256": sha256_file(path)}
            for path in outputs
            if Path(path).exists()
        ],
        "code_revision": git_revision(repository_root),
        "python": platform.python_version(),
        "notes": list(notes),
    }


def write_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite manifest: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
