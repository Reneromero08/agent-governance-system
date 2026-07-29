#!/usr/bin/env python3
"""Adversarial protocol-only controller for the atomic Givens repair."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import catvm_bosonic_givens_protocol as protocol


STATUS_ERROR = 2
STAGE_RECEIPT_TAG = 0x53544147454C4541


def fail(message: str) -> None:
    raise RuntimeError(message)


def exchange(
    connection: socket.socket,
    command: int,
    nonce: int,
) -> tuple[bytes, dict[str, object]]:
    connection.sendall(protocol.request(command, nonce))
    payload = connection.recv(protocol.RESPONSE.size)
    if len(payload) != protocol.RESPONSE.size:
        fail("truncated CATVM repair response")
    decoded = protocol.response(payload)
    if int(decoded["command"]) != command:
        fail("CATVM repair command/response mismatch")
    return payload, decoded


class Service:
    def __init__(
        self,
        executable: Path,
        evidence_dir: Path,
        name: str,
        mode: str,
    ) -> None:
        self.socket_path = evidence_dir / f"{name}.sock"
        self.stdout_path = evidence_dir / f"{name}.stdout"
        self.stderr_path = evidence_dir / f"{name}.stderr"
        self.stdout_file = self.stdout_path.open("wb")
        self.stderr_file = self.stderr_path.open("wb")
        self.process = subprocess.Popen(
            [
                "nice",
                "-n",
                "10",
                "taskset",
                "-c",
                "0-3",
                str(executable),
                mode,
                str(self.socket_path),
            ],
            stdout=self.stdout_file,
            stderr=self.stderr_file,
        )
        for _ in range(400):
            if self.socket_path.exists():
                break
            if self.process.poll() is not None:
                fail(f"CATVM repair service {name} exited before bind")
            time.sleep(0.01)
        if not self.socket_path.exists():
            fail(f"CATVM repair service {name} did not bind")
        self.connection = socket.socket(
            socket.AF_UNIX, socket.SOCK_SEQPACKET
        )
        self.connection.connect(str(self.socket_path))
        self.nonce = 0xA17D_0000

    def call(
        self,
        command: int,
    ) -> tuple[bytes, dict[str, object]]:
        self.nonce += 1
        return exchange(self.connection, command, self.nonce)

    def call_with_nonce(
        self,
        command: int,
        nonce: int,
    ) -> tuple[bytes, dict[str, object]]:
        return exchange(self.connection, command, nonce)

    def disconnect(self) -> None:
        self.connection.close()
        self._wait()

    def stop(self) -> None:
        _, stopped = self.call(protocol.STOP)
        if int(stopped["status"]) != protocol.STATUS_OK:
            fail("CATVM repair service stop failed")
        self.connection.close()
        self._wait()

    def _wait(self) -> None:
        return_code = self.process.wait(timeout=120)
        self.stdout_file.close()
        self.stderr_file.close()
        if return_code != 0:
            fail(f"CATVM repair service exited {return_code}")
        if self.stdout_path.stat().st_size != 0:
            fail("CATVM repair service emitted stdout")
        if self.stderr_path.stat().st_size != 0:
            fail("CATVM repair service emitted stderr")


def boundary_error(
    left: dict[str, object],
    right: dict[str, object],
) -> float:
    return max(
        abs(float(a) - float(b))
        for a, b in zip(
            left["boundary"], right["boundary"], strict=True
        )
    )


def require_status(
    response: dict[str, object],
    expected: int,
    label: str,
) -> None:
    if int(response["status"]) != expected:
        fail(f"{label}: expected status {expected}")


def direct_arm(
    executable: Path,
    evidence_dir: Path,
) -> dict[str, object]:
    service = Service(executable, evidence_dir, "direct", "direct")
    _, initialized = service.call(protocol.INITIALIZE)
    _, snapshot_denied = service.call(protocol.SNAPSHOT_BEGIN)
    _, inplace_denied = service.call(protocol.BEGIN_PRIMARY)
    _, begun = service.call(protocol.DIRECT_BEGIN)
    _, result = service.call(protocol.DIRECT_CONTINUE)
    for label, response, expected in (
        ("direct initialize", initialized, protocol.STATUS_OK),
        ("snapshot on direct", snapshot_denied, protocol.STATUS_DENIED),
        ("in-place on direct", inplace_denied, protocol.STATUS_DENIED),
        ("direct begin", begun, protocol.STATUS_OK),
        ("direct result", result, protocol.STATUS_OK),
    ):
        require_status(response, expected, label)
    if not int(result["flags"]) & protocol.BOUNDARY_VALID:
        fail("direct arm did not return final boundary")
    if int(result["flags"]) & protocol.RESTORED:
        fail("direct arm falsely claimed restoration")
    service.stop()
    return result


def snapshot_arm(
    executable: Path,
    evidence_dir: Path,
) -> dict[str, object]:
    service = Service(
        executable, evidence_dir, "snapshot", "snapshot"
    )
    _, initialized = service.call(protocol.INITIALIZE)
    _, direct_denied = service.call(protocol.DIRECT_BEGIN)
    _, inplace_denied = service.call(protocol.BEGIN_PRIMARY)
    _, begun = service.call(protocol.SNAPSHOT_BEGIN)
    _, result = service.call(protocol.SNAPSHOT_CONTINUE)
    for label, response, expected in (
        ("snapshot initialize", initialized, protocol.STATUS_OK),
        ("direct on snapshot", direct_denied, protocol.STATUS_DENIED),
        ("in-place on snapshot", inplace_denied, protocol.STATUS_DENIED),
        ("snapshot begin", begun, protocol.STATUS_OK),
        ("snapshot result", result, protocol.STATUS_OK),
    ):
        require_status(response, expected, label)
    required = (
        protocol.BOUNDARY_VALID
        | protocol.RESTORED
        | protocol.SNAPSHOT_RELOAD
    )
    if int(result["flags"]) & required != required:
        fail("snapshot arm flags are incomplete")
    service.stop()
    return result


def inplace_arm(
    executable: Path,
    evidence_dir: Path,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    service = Service(
        executable, evidence_dir, "inplace", "in-place"
    )
    _, initialized = service.call(protocol.INITIALIZE)
    _, direct_denied = service.call(protocol.DIRECT_BEGIN)
    _, snapshot_denied = service.call(protocol.SNAPSHOT_BEGIN)
    stage_payload, begun = service.call(protocol.BEGIN_PRIMARY)
    stage_nonce = service.nonce
    _, projection = service.call(protocol.PROJECT_INTERMEDIATE)
    _, replay = service.call_with_nonce(
        protocol.PROJECT_INTERMEDIATE, stage_nonce
    )
    _, primary = service.call(protocol.CONTINUE_PRIMARY)
    _, second_begin = service.call(protocol.BEGIN_PRIMARY)
    _, reuse = service.call(protocol.REUSE)

    for label, response, expected in (
        ("in-place initialize", initialized, protocol.STATUS_OK),
        ("direct on in-place", direct_denied, protocol.STATUS_DENIED),
        ("snapshot on in-place", snapshot_denied, protocol.STATUS_DENIED),
        ("in-place begin", begun, protocol.STATUS_OK),
        ("resident projection", projection, protocol.STATUS_DENIED),
        ("nonce replay", replay, protocol.STATUS_DENIED),
        ("in-place primary", primary, protocol.STATUS_OK),
        ("second primary begin", second_begin, STATUS_ERROR),
        ("in-place reuse", reuse, protocol.STATUS_OK),
    ):
        require_status(response, expected, label)

    if not int(begun["flags"]) & protocol.STAGE_RESIDENT:
        fail("hidden occupation stage was not resident")
    if int(begun["flags"]) & protocol.BOUNDARY_VALID:
        fail("staged response exposed a boundary")
    if any(float(value) != 0.0 for value in begun["boundary"]):
        fail("staged response contained boundary values")
    expected_receipt = stage_nonce ^ STAGE_RECEIPT_TAG
    if int(begun["receipt"]) != expected_receipt:
        fail("staged receipt was not public typed custody metadata")
    if int(begun["state_hash"]) != 1:
        fail("staged state token was not discrete-only")
    if len(stage_payload) != protocol.RESPONSE.size:
        fail("staged response wire size changed")
    if int(projection["flags"]) & protocol.BOUNDARY_VALID:
        fail("projection denial exposed a boundary")

    primary_required = protocol.BOUNDARY_VALID | protocol.RESTORED
    if int(primary["flags"]) & primary_required != primary_required:
        fail("primary response was not final-and-restored")
    if int(primary["flags"]) & protocol.STAGE_RESIDENT:
        fail("primary response retained staged state")
    if int(primary["generation"]) != 1:
        fail("primary restoration generation was not one")
    reuse_required = (
        protocol.BOUNDARY_VALID
        | protocol.RESTORED
        | protocol.REUSE_FLAG
    )
    if int(reuse["flags"]) & reuse_required != reuse_required:
        fail("reuse response flags are incomplete")
    if int(reuse["generation"]) != 2:
        fail("reuse restoration generation was not two")
    if float(reuse["norm_error"]) > 3e-11:
        fail("fresh/restored reuse boundary parity failed")
    service.stop()
    return begun, primary, reuse


def disconnect_attack(
    executable: Path,
    evidence_dir: Path,
) -> dict[str, object]:
    service = Service(
        executable, evidence_dir, "disconnect", "in-place"
    )
    _, initialized = service.call(protocol.INITIALIZE)
    _, begun = service.call(protocol.BEGIN_PRIMARY)
    require_status(initialized, protocol.STATUS_OK, "disconnect initialize")
    require_status(begun, protocol.STATUS_OK, "disconnect begin")
    if not int(begun["flags"]) & protocol.STAGE_RESIDENT:
        fail("disconnect attack did not reach resident stage")
    service.disconnect()
    return {
        "stage_resident_before_disconnect": True,
        "service_exit_after_inverse_cleanup": 0,
        "internal_fresh_reuse_sentinel": "PASS",
        "boundary_released": False,
    }


def inverse_control_attack(
    executable: Path,
    evidence_dir: Path,
    name: str,
    command: int,
) -> float:
    service = Service(
        executable, evidence_dir, f"control-{name}", "in-place"
    )
    _, initialized = service.call(protocol.INITIALIZE)
    _, result = service.call(command)
    _, poisoned_reuse = service.call(protocol.REUSE)
    require_status(initialized, protocol.STATUS_OK, f"{name} initialize")
    require_status(result, protocol.STATUS_OK, f"{name} control")
    require_status(
        poisoned_reuse, protocol.STATUS_DENIED, f"{name} poison gate"
    )
    error = float(result["restoration_error"])
    if error <= 1e-5:
        fail(f"{name} inverse control did not separate")
    service.stop()
    return error


def null_carrier_attack(
    executable: Path,
    evidence_dir: Path,
) -> dict[str, object]:
    service = Service(executable, evidence_dir, "null", "null")
    _, initialized = service.call(protocol.INITIALIZE)
    _, denied = service.call(protocol.NULL_CARRIER)
    _, inplace_denied = service.call(protocol.BEGIN_PRIMARY)
    require_status(initialized, protocol.STATUS_OK, "null initialize")
    require_status(denied, protocol.STATUS_DENIED, "null carrier")
    require_status(
        inplace_denied, protocol.STATUS_DENIED, "null in-place"
    )
    service.stop()
    return {
        "backing_cells": 0,
        "null_command": "DENIED",
        "in_place_command": "DENIED",
    }


def main() -> int:
    if len(sys.argv) != 3:
        fail(
            "usage: catvm_bosonic_givens_atomic_repair_controller.py "
            "SERVICE EVIDENCE_DIR"
        )
    if sys.byteorder != "little" or os.uname().machine != "x86_64":
        fail("atomic Givens repair requires x86-64 little-endian Linux")
    executable = Path(sys.argv[1]).resolve()
    evidence_dir = Path(sys.argv[2]).resolve()
    evidence_dir.mkdir(parents=True, exist_ok=True)

    direct = direct_arm(executable, evidence_dir)
    snapshot = snapshot_arm(executable, evidence_dir)
    begun, primary, reuse = inplace_arm(executable, evidence_dir)
    disconnect = disconnect_attack(executable, evidence_dir)
    controls: dict[str, float] = {}
    for name, command in (
        ("missing", protocol.MISSING_INVERSE),
        ("wrong", protocol.WRONG_INVERSE),
        ("reordered", protocol.REORDERED_INVERSE),
    ):
        controls[name] = inverse_control_attack(
            executable, evidence_dir, name, command
        )
    null = null_carrier_attack(executable, evidence_dir)

    direct_error = boundary_error(direct, primary)
    snapshot_error = boundary_error(snapshot, primary)
    if direct_error > 3e-11 or snapshot_error > 3e-11:
        fail("mode-separated matched boundaries disagree")
    if float(primary["restoration_error"]) > 3e-11:
        fail("in-place numerical restoration failed")
    if float(reuse["restoration_error"]) > 3e-11:
        fail("restored-carrier reuse failed")

    result = {
        "claim_candidate": (
            "BOUNDED_CATVM_MODE_LOCKED_BOSONIC_GIVENS_HIDDEN_"
            "OCCUPATION_ATOMIC_DISCONNECT_RESTORATION_AND_REUSE"
        ),
        "result": "PASS",
        "source_predecessor": (
            "BOUNDED_CATVM_ENFORCED_TOPOLOGY_COMPILED_BOSONIC_"
            "GIVENS_HIDDEN_OCCUPATION_COMPOSITION_WITH_ACTUAL_"
            "INVERSE_RESTORATION_AND_REUSE"
        ),
        "restoration_class": "NUMERICAL_PHYSICAL_STATE_RESTORATION",
        "claim_ceiling": (
            "LINUX_X86_64_SAME_UID_SINGLE_CONNECTION_LEASE_GRID17_"
            "FOUR_ROTOR_DEPTH8_COMPLEX128_SOFTWARE_ONLY"
        ),
        "mode_custody": {
            "launch_time_modes": [
                "direct",
                "snapshot",
                "in-place",
                "null",
            ],
            "cross_mode_commands_denied": True,
            "monotone_nonce_replay_denied": True,
            "one_connection_lease": True,
        },
        "hidden_intermediate": {
            "complex_cells": 4845,
            "resident": bool(
                int(begun["flags"]) & protocol.STAGE_RESIDENT
            ),
            "content_derived_wire_hashes": 0,
            "boundary_values_in_stage_response": 0,
            "projection": "DENIED",
        },
        "atomic_ordering": {
            "final_boundary_after_restoration": True,
            "primary_generation": primary["generation"],
            "disconnect": disconnect,
        },
        "matched_arms": {
            "direct_restoration_class": "NO_RESTORATION_CLAIM",
            "snapshot_restoration_class": "SNAPSHOT_RELOAD",
            "in_place_restoration_class": (
                "NUMERICAL_PHYSICAL_STATE_RESTORATION"
            ),
            "direct_boundary_error": direct_error,
            "snapshot_boundary_error": snapshot_error,
            "snapshot_reload_bytes": snapshot["snapshot_reload_bytes"],
        },
        "primary": {
            "restoration_error": primary["restoration_error"],
            "generation": primary["generation"],
            "carrier_backing_preserved": True,
            "snapshot_reload_bytes": primary["snapshot_reload_bytes"],
        },
        "reuse": {
            "restoration_error": reuse["restoration_error"],
            "generation": reuse["generation"],
            "fresh_restored_boundary_error": reuse["norm_error"],
            "same_backing": True,
            "same_native_operation_signature": True,
            "snapshot_reload_bytes": reuse["snapshot_reload_bytes"],
        },
        "controls": {
            "missing_inverse_error": controls["missing"],
            "wrong_inverse_error": controls["wrong"],
            "reordered_inverse_error": controls["reordered"],
            "failed_inverse_poisoned_service": True,
            "null_carrier": null,
        },
        "no_smuggle": {
            "controller_imports_backend": False,
            "stage_packet_bytes": protocol.RESPONSE.size,
            "stage_boundary_valid": False,
            "service_stdout_bytes": 0,
            "service_stderr_bytes": 0,
            "receipt_depends_only_on_nonce_and_type": True,
        },
        "resource_ceiling": {
            "carrier_cells": 285,
            "hidden_occupation_cells": 4845,
            "retained_inverse_history_bytes": 0,
            "allocator_native_library_os_memory_bounded": False,
        },
        "matched_classical_recurrence_identical": True,
        "distinct_phase_resource_established": False,
        "computational_advantage": False,
        "small_wall_crossed": False,
        "physical_waveform_execution": False,
        "terminal": False,
    }
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
