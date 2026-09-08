from __future__ import annotations

import os
import platform
import shlex
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from .provenance import sha256_json


def one_command_option(command: Sequence[str], option: str) -> str:
    positions = [index for index, value in enumerate(command) if value == option]
    if len(positions) != 1 or positions[0] + 1 >= len(command):
        raise ValueError(f"service command must contain exactly one {option}")
    return command[positions[0] + 1]


def _process_argv_and_parent(pid: int) -> tuple[list[str], int, str]:
    proc = Path("/proc") / str(pid)
    if proc.is_dir():
        raw = (proc / "cmdline").read_bytes()
        argv = [item.decode("utf-8", errors="surrogateescape") for item in raw.split(b"\0") if item]
        fields = (proc / "stat").read_text(encoding="utf-8").split()
        if not argv or len(fields) < 4:
            raise ValueError("could not read the live service process identity from /proc")
        return argv, int(fields[3]), "linux_procfs"

    try:
        parent_text = subprocess.check_output(
            ["ps", "-p", str(pid), "-o", "ppid="], text=True, stderr=subprocess.DEVNULL
        ).strip()
        command_text = subprocess.check_output(
            ["ps", "-ww", "-p", str(pid), "-o", "command="],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("could not inspect the live service process with ps") from error
    try:
        argv = shlex.split(command_text)
        parent = int(parent_text)
    except (ValueError, TypeError) as error:
        raise ValueError("live service process identity is malformed") from error
    if not argv:
        raise ValueError("live service process has an empty command line")
    return argv, parent, "portable_ps"


def _linux_port_owner_pids(port: int) -> set[int]:
    socket_inodes: set[str] = set()
    encoded_port = f"{port:04X}"
    for table in (Path("/proc/net/tcp"), Path("/proc/net/tcp6")):
        if not table.is_file():
            continue
        for line in table.read_text(encoding="utf-8").splitlines()[1:]:
            fields = line.split()
            if len(fields) > 9 and fields[1].rsplit(":", 1)[-1].upper() == encoded_port:
                if fields[3] == "0A":  # TCP_LISTEN
                    socket_inodes.add(fields[9])
    owners: set[int] = set()
    if not socket_inodes:
        return owners
    for process in Path("/proc").iterdir():
        if not process.name.isdigit():
            continue
        descriptor_dir = process / "fd"
        try:
            descriptors = list(descriptor_dir.iterdir())
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        for descriptor in descriptors:
            try:
                target = os.readlink(descriptor)
            except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
                continue
            if target.startswith("socket:[") and target[8:-1] in socket_inodes:
                owners.add(int(process.name))
                break
    return owners


def _portable_port_owner_pids(port: int) -> set[int]:
    try:
        output = subprocess.check_output(
            ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("could not identify the listening port owner with lsof") from error
    try:
        return {int(line.strip()) for line in output.splitlines() if line.strip()}
    except ValueError as error:
        raise ValueError("lsof returned a malformed listening-process identity") from error


def live_service_process_snapshot(
    *, server_pid: int, wrapper_pid: int, port: int, launch_command: Sequence[str]
) -> dict[str, Any]:
    """Bind a local service claim to live argv, direct parent, and TCP listener owner.

    This is a host-local observation, not a cryptographic remote attestation.  It
    detects accidental or ordinary process substitution while the wrapper and
    service remain alive on the same trusted host.
    """

    if server_pid <= 0 or wrapper_pid <= 0 or not 0 < port <= 65535:
        raise ValueError("service PID/parent/port identity is invalid")
    try:
        os.kill(server_pid, 0)
    except OSError as error:
        raise ValueError("attested vLLM service process is not alive") from error
    actual_argv, actual_parent, inspection_method = _process_argv_and_parent(server_pid)
    expected = list(launch_command)
    if not expected or len(actual_argv) != len(expected):
        raise ValueError("live service argv does not match the attested launch command")
    # The OS may canonicalize only argv[0] (e.g. python3 -> /usr/bin/python3).
    if Path(actual_argv[0]).name != Path(expected[0]).name or actual_argv[1:] != expected[1:]:
        raise ValueError("live service argv does not match the attested launch command")
    if actual_parent != wrapper_pid:
        raise ValueError("attested server is not a direct child of the service wrapper")
    owners = (
        _linux_port_owner_pids(port)
        if Path("/proc/net/tcp").is_file()
        else _portable_port_owner_pids(port)
    )
    if server_pid not in owners:
        raise ValueError("attested server PID is not the owner of the listening TCP port")
    return {
        "platform": platform.system(),
        "inspection_method": inspection_method,
        "server_pid": server_pid,
        "parent_pid": actual_parent,
        "argv": actual_argv,
        "argv_sha256": sha256_json(actual_argv),
        "port": port,
        "listening_owner_pids": sorted(owners),
        "trust_boundary": (
            "host_local_process_observation_not_cryptographic_remote_attestation"
        ),
    }


def verify_recorded_live_snapshot(
    recorded: Mapping[str, Any],
    *,
    server_pid: int,
    wrapper_pid: int,
    port: int,
    launch_command: Sequence[str],
) -> dict[str, Any]:
    current = live_service_process_snapshot(
        server_pid=server_pid,
        wrapper_pid=wrapper_pid,
        port=port,
        launch_command=launch_command,
    )
    required_equal = (
        "server_pid",
        "parent_pid",
        "argv_sha256",
        "port",
        "trust_boundary",
    )
    if any(recorded.get(key) != current[key] for key in required_equal):
        raise ValueError("recorded service process observation no longer matches the live service")
    if recorded.get("listening_owner_pids") != current["listening_owner_pids"]:
        raise ValueError("recorded listening-port owners no longer match the live service")
    return current
