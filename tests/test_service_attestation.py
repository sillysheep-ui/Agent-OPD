from contextlib import contextmanager
from unittest.mock import patch

from omniopd import service_attestation


@contextmanager
def _process_fakes(*, argv, parent=88, owners=None):
    owner_set = {99} if owners is None else set(owners)
    with (
        patch.object(service_attestation.os, "kill", lambda pid, signal: None),
        patch.object(
            service_attestation,
            "_process_argv_and_parent",
            lambda pid: (list(argv), parent, "test"),
        ),
        patch.object(
            service_attestation,
            "_linux_port_owner_pids",
            lambda port: owner_set,
        ),
        patch.object(
            service_attestation,
            "_portable_port_owner_pids",
            lambda port: owner_set,
        ),
    ):
        yield


def test_live_service_snapshot_binds_exact_argv_parent_and_port_owner():
    command = ["python3", "-m", "vllm.entrypoints.openai.api_server", "--port", "8000"]
    with _process_fakes(argv=command):
        snapshot = service_attestation.live_service_process_snapshot(
            server_pid=99, wrapper_pid=88, port=8000, launch_command=command
        )
    assert snapshot["listening_owner_pids"] == [99]
    assert snapshot["parent_pid"] == 88


def test_live_service_snapshot_rejects_wrong_process_identity():
    command = ["python3", "-m", "vllm.entrypoints.openai.api_server", "--port", "8000"]
    cases = [
        (["--unexpected"], 88, {99}, "argv"),
        ([], 87, {99}, "direct child"),
        ([], 88, {100}, "owner"),
    ]
    for argv_suffix, parent, owners, message in cases:
        with _process_fakes(
            argv=[*command, *argv_suffix], parent=parent, owners=owners
        ):
            try:
                service_attestation.live_service_process_snapshot(
                    server_pid=99,
                    wrapper_pid=88,
                    port=8000,
                    launch_command=command,
                )
            except ValueError as error:
                assert message in str(error)
            else:
                raise AssertionError(f"wrong service {message} identity must be rejected")
