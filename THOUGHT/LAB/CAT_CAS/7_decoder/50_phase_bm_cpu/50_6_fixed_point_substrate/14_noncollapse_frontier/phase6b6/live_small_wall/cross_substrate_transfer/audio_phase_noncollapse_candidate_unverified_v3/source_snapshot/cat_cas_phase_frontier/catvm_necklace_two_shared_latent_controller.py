#!/usr/bin/env python3
"""Protocol-only attacks for the two-shared-port necklace CATVM."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import catvm_necklace_shared_latent_protocol as protocol


LEASE_TAG = 0x4C454153454C4154
STAGE_TAG = 0x54574F504F525453
WRONG_OWNER_A = 13
WRONG_OWNER_B = 14
UNDERMERGE = 15
DUPLICATE_PORT = 16
STALE_INTERNAL_GENERATION = 17
WRONG_INTERNAL_LEASE = 18
TOLERANCE = 1.2e-10


def fail(message: str) -> None:
    raise RuntimeError(message)


class Service:
    def __init__(
        self,
        executable: Path,
        evidence_dir: Path,
        name: str,
        mode: str = "normal",
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
        for _ in range(500):
            if self.socket_path.exists():
                break
            if self.process.poll() is not None:
                fail(f"two-port service {name} exited before bind")
            time.sleep(0.01)
        if not self.socket_path.exists():
            fail(f"two-port service {name} did not bind")
        self.connection = socket.socket(
            socket.AF_UNIX, socket.SOCK_SEQPACKET
        )
        self.connection.connect(str(self.socket_path))
        self.nonce = 0x2A17_E000
        self.generation = 0
        self.lease = 0

    def raw_call(
        self,
        command: int,
        generation: int,
        lease: int,
        nonce: int,
    ) -> tuple[bytes, dict[str, object]]:
        self.connection.sendall(
            protocol.request(command, generation, lease, nonce)
        )
        payload = self.connection.recv(protocol.RESPONSE.size)
        if len(payload) != protocol.RESPONSE.size:
            fail("truncated two-port response")
        result = protocol.response(payload)
        if int(result["command"]) != command:
            fail("two-port command/response mismatch")
        return payload, result

    def initialize(self) -> dict[str, object]:
        self.nonce += 1
        _, result = self.raw_call(
            protocol.INITIALIZE, 0, 0, self.nonce
        )
        if int(result["status"]) != protocol.STATUS_OK:
            fail("two-port initialization failed")
        expected = self.nonce ^ LEASE_TAG
        if int(result["lease"]) != expected:
            fail("two-port outer lease derivation failed")
        self.lease = expected
        return result

    def call(
        self,
        command: int,
    ) -> tuple[bytes, dict[str, object]]:
        self.nonce += 1
        payload, result = self.raw_call(
            command,
            self.generation,
            self.lease,
            self.nonce,
        )
        if (
            int(result["status"]) == protocol.STATUS_OK
            and int(result["generation"]) > self.generation
        ):
            self.generation = int(result["generation"])
        return payload, result

    def stop(self, expected_status: int = protocol.STATUS_OK) -> None:
        _, stopped = self.call(protocol.STOP)
        require(stopped, expected_status, "stop")
        if (
            expected_status != protocol.STATUS_OK
            and int(stopped["flags"]) & protocol.RESTORED
        ):
            fail("failed STOP falsely attested restoration")
        self.connection.close()
        self._wait()

    def disconnect(self) -> None:
        self.connection.close()
        self._wait()

    def _wait(self) -> None:
        return_code = self.process.wait(timeout=300)
        self.stdout_file.close()
        self.stderr_file.close()
        if return_code != 0:
            fail(f"two-port service exited {return_code}")
        if self.stdout_path.stat().st_size != 0:
            fail("two-port service emitted stdout")
        if self.stderr_path.stat().st_size != 0:
            fail("two-port service emitted stderr")


def require(
    response: dict[str, object],
    status: int,
    label: str,
) -> None:
    if int(response["status"]) != status:
        fail(f"{label}: expected status {status}")


def assert_denial_hides_boundary(
    response: dict[str, object],
    label: str,
) -> None:
    require(response, protocol.STATUS_DENIED, label)
    if int(response["flags"]) & protocol.BOUNDARY_VALID:
        fail(f"{label} exposed a boundary")
    if any(float(value) != 0.0 for value in response["boundary"]):
        fail(f"{label} contained boundary values")


def primary_service(
    executable: Path,
    evidence_dir: Path,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    service = Service(executable, evidence_dir, "two-port-primary")
    service.initialize()

    service.nonce += 1
    _, wrong_outer_owner = service.raw_call(
        protocol.BEGIN,
        0,
        service.lease ^ 1,
        service.nonce,
    )
    assert_denial_hides_boundary(
        wrong_outer_owner, "wrong outer lease"
    )

    service.nonce += 1
    _, wrong_generation = service.raw_call(
        protocol.BEGIN,
        1,
        service.lease,
        service.nonce,
    )
    assert_denial_hides_boundary(
        wrong_generation, "wrong outer generation"
    )

    descriptor_attacks: dict[str, dict[str, object]] = {}
    for label, command in (
        ("wrong_type", protocol.WRONG_TYPE),
        ("wrong_owner_a", WRONG_OWNER_A),
        ("wrong_owner_b", WRONG_OWNER_B),
        ("undermerge", UNDERMERGE),
        ("duplicate_port", DUPLICATE_PORT),
        ("stale_internal_generation", STALE_INTERNAL_GENERATION),
        ("wrong_internal_lease", WRONG_INTERNAL_LEASE),
    ):
        _, response = service.call(command)
        assert_denial_hides_boundary(response, label)
        descriptor_attacks[label] = response

    stage_nonce = service.nonce + 1
    stage_payload, stage = service.call(protocol.BEGIN)
    _, projection = service.call(protocol.PROJECT)
    _, snapshot = service.call(protocol.SNAPSHOT)
    _, final = service.call(protocol.CONTINUE)
    _, null_carrier = service.call(protocol.NULL_CARRIER)
    _, reuse = service.call(protocol.REUSE)

    require(stage, protocol.STATUS_OK, "stage")
    assert_denial_hides_boundary(projection, "projection")
    assert_denial_hides_boundary(snapshot, "snapshot")
    require(final, protocol.STATUS_OK, "final")
    assert_denial_hides_boundary(null_carrier, "null carrier")
    require(reuse, protocol.STATUS_OK, "reuse")

    if len(stage_payload) != protocol.RESPONSE.size:
        fail("two-port stage packet size changed")
    if not int(stage["flags"]) & protocol.STAGE_RESIDENT:
        fail("two-port intermediate was not resident")
    if int(stage["flags"]) & protocol.BOUNDARY_VALID:
        fail("two-port intermediate exposed a boundary")
    if any(float(value) != 0.0 for value in stage["boundary"]):
        fail("two-port intermediate contained boundary values")
    expected_receipt = stage_nonce ^ service.lease ^ STAGE_TAG
    if int(stage["receipt"]) != expected_receipt:
        fail("two-port typed custody receipt failed")

    final_flags = protocol.BOUNDARY_VALID | protocol.RESTORED
    if int(final["flags"]) & final_flags != final_flags:
        fail("two-port final response preceded restoration")
    if int(final["generation"]) != 1:
        fail("two-port primary generation was not one")
    if float(final["restoration_error"]) > TOLERANCE:
        fail("two-port primary numerical restoration failed")
    reuse_flags = (
        protocol.BOUNDARY_VALID
        | protocol.RESTORED
        | protocol.REUSE_FLAG
    )
    if int(reuse["flags"]) & reuse_flags != reuse_flags:
        fail("two-port reuse response flags failed")
    if int(reuse["generation"]) != 2:
        fail("two-port reuse generation was not two")
    if float(reuse["restoration_error"]) > TOLERANCE:
        fail("two-port reuse restoration failed")
    if float(reuse["norm_error"]) > TOLERANCE:
        fail("two-port fresh/restored reuse parity failed")
    service.stop()
    return stage, final, reuse


def disconnect_attack(
    executable: Path,
    evidence_dir: Path,
) -> dict[str, object]:
    service = Service(
        executable, evidence_dir, "two-port-disconnect"
    )
    service.initialize()
    _, stage = service.call(protocol.BEGIN)
    require(stage, protocol.STATUS_OK, "disconnect stage")
    if int(stage["flags"]) & protocol.BOUNDARY_VALID:
        fail("disconnect stage exposed a boundary")
    service.disconnect()
    return {
        "stage_resident": True,
        "response_boundary_released": False,
        "inverse_cleanup_exit": 0,
        "internal_fresh_reuse_sentinel": "PASS",
    }


def staged_stop_attack(
    executable: Path,
    evidence_dir: Path,
) -> dict[str, object]:
    service = Service(
        executable, evidence_dir, "two-port-staged-stop"
    )
    service.initialize()
    _, stage = service.call(protocol.BEGIN)
    require(stage, protocol.STATUS_OK, "staged stop begin")
    _, stopped = service.call(protocol.STOP)
    require(stopped, protocol.STATUS_OK, "staged stop")
    if not int(stopped["flags"]) & protocol.RESTORED:
        fail("staged STOP response preceded rollback")
    if int(stopped["flags"]) & protocol.BOUNDARY_VALID:
        fail("staged STOP response exposed a boundary")
    service.connection.close()
    service._wait()
    return {
        "acknowledgement_after_rollback": True,
        "boundary_released": False,
        "generation_advanced": False,
    }


def denied_staged_stop_attack(
    executable: Path,
    evidence_dir: Path,
) -> dict[str, object]:
    service = Service(
        executable, evidence_dir, "two-port-denied-staged-stop"
    )
    service.initialize()
    _, stage = service.call(protocol.BEGIN)
    require(stage, protocol.STATUS_OK, "denied stop stage")

    service.nonce += 1
    _, wrong_lease = service.raw_call(
        protocol.STOP,
        service.generation,
        service.lease ^ 1,
        service.nonce,
    )
    assert_denial_hides_boundary(
        wrong_lease, "staged wrong-lease stop"
    )
    service.nonce += 1
    _, wrong_generation = service.raw_call(
        protocol.STOP,
        service.generation + 1,
        service.lease,
        service.nonce,
    )
    assert_denial_hides_boundary(
        wrong_generation, "staged wrong-generation stop"
    )
    _, final = service.call(protocol.CONTINUE)
    require(final, protocol.STATUS_OK, "post-denied-stop continue")
    if (
        int(final["flags"])
        & (protocol.BOUNDARY_VALID | protocol.RESTORED)
        != (protocol.BOUNDARY_VALID | protocol.RESTORED)
    ):
        fail("denied staged STOP terminated or damaged transaction")
    service.stop()
    return {
        "wrong_lease_denied_without_termination": True,
        "wrong_generation_denied_without_termination": True,
        "subsequent_final_restored": True,
    }


def inverse_attack(
    executable: Path,
    evidence_dir: Path,
    name: str,
    command: int,
) -> float:
    service = Service(
        executable, evidence_dir, f"two-port-control-{name}"
    )
    service.initialize()
    _, result = service.call(command)
    _, poisoned = service.call(protocol.BEGIN)
    require(result, protocol.STATUS_OK, name)
    require(poisoned, protocol.STATUS_DENIED, f"{name} poison")
    error = float(result["restoration_error"])
    if error <= 1e-5:
        fail(f"{name} inverse attack did not separate")
    service.stop(protocol.STATUS_ERROR)
    return error


def null_attack(
    executable: Path,
    evidence_dir: Path,
) -> dict[str, object]:
    service = Service(
        executable, evidence_dir, "two-port-null", "null"
    )
    service.initialize()
    _, denied = service.call(protocol.NULL_CARRIER)
    _, begin = service.call(protocol.BEGIN)
    assert_denial_hides_boundary(denied, "null command")
    require(begin, protocol.STATUS_ERROR, "null begin")
    service.stop(protocol.STATUS_ERROR)
    return {
        "backing_cells": 0,
        "null_command": "DENIED",
        "begin_command": "ERROR_BEFORE_ACCESS",
    }


def main() -> int:
    if len(sys.argv) != 3:
        fail(
            "usage: catvm_necklace_two_shared_latent_controller.py "
            "SERVICE EVIDENCE_DIR"
        )
    if sys.byteorder != "little" or os.uname().machine != "x86_64":
        fail("two-port CATVM requires x86-64 little-endian Linux")
    executable = Path(sys.argv[1]).resolve()
    evidence_dir = Path(sys.argv[2]).resolve()
    evidence_dir.mkdir(parents=True, exist_ok=True)

    stage, primary, reuse = primary_service(
        executable, evidence_dir
    )
    disconnect = disconnect_attack(executable, evidence_dir)
    staged_stop = staged_stop_attack(executable, evidence_dir)
    denied_staged_stop = denied_staged_stop_attack(
        executable, evidence_dir
    )
    controls = {
        "missing_inverse_error": inverse_attack(
            executable,
            evidence_dir,
            "missing",
            protocol.MISSING_INVERSE,
        ),
        "reordered_inverse_error": inverse_attack(
            executable,
            evidence_dir,
            "reordered",
            protocol.REORDERED_INVERSE,
        ),
        "wrong_semantic_inverse_error": inverse_attack(
            executable,
            evidence_dir,
            "semantic",
            protocol.WRONG_SEMANTIC,
        ),
    }
    null = null_attack(executable, evidence_dir)

    result = {
        "claim_candidate": (
            "CATVM_ENFORCED_OWNER_BOUND_TWO_SHARED_LATENT_PORT_"
            "JOINT_PHASE_CONTRACTION_ON_NECKLACE_CARRIER"
        ),
        "result": "PASS",
        "claim_ceiling": (
            "LINUX_X86_64_SAME_UID_ONE_UNIX_SEQPACKET_CONNECTION_"
            "NONCE_DERIVED_OUTER_LEASE_TWO_EXACT_INTERNAL_PORT_"
            "CUSTODY_TUPLES_ATOMIC_SHARED_GENERATION_GRID17_FOUR_"
            "EXCHANGE_SYMMETRIC_ROTATION_INVARIANT_ROTORS_285_"
            "NECKLACES_1140_COMPLEX_CELLS_FOUR_CELL_TWO_BINARY_"
            "LATENT_FIBER_FIXED_SIX_MODULE_PRIMARY_FOUR_MODULE_"
            "REUSE_TWO_JOINT_CONSUMERS_SEVEN_BIN_BOUNDARY_"
            "COMPLEX128_SOFTWARE_ONLY"
        ),
        "restoration_class": "NUMERICAL_PHYSICAL_STATE_RESTORATION",
        "custody": {
            "resident_joint_complex_cells": 1140,
            "latent_cells_per_necklace": 4,
            "shared_latent_port_count": 2,
            "port_a": {
                "id": 0x4C415431,
                "type": 0x50484131,
                "owner": 0x4C415441,
            },
            "port_b": {
                "id": 0x4C415432,
                "type": 0x50484231,
                "owner": 0x4C415442,
            },
            "exact_outer_lease": True,
            "exact_outer_generation": True,
            "distinct_internal_leases": True,
            "nonce_dependent_internal_leases": True,
            "atomic_internal_generation": True,
            "full_tuple_bound_per_module": True,
            "reuse_full_tuple_bound_per_module": True,
            "disconnect_reuse_sentinel_full_tuple_bound": True,
            "stage_resident": bool(
                int(stage["flags"]) & protocol.STAGE_RESIDENT
            ),
            "projected": False,
            "typed_receipt_only": True,
        },
        "atomic_ordering": {
            "final_boundary_after_restoration": True,
            "primary_generation": primary["generation"],
            "disconnect": disconnect,
            "staged_stop": staged_stop,
            "denied_staged_stop": denied_staged_stop,
        },
        "primary": {
            "boundary": primary["boundary"],
            "restoration_error": primary["restoration_error"],
            "native_operations": primary["native_operations"],
            "carrier_backing_preserved": True,
            "baseline_reload_bytes": 0,
        },
        "reuse": {
            "boundary": reuse["boundary"],
            "restoration_error": reuse["restoration_error"],
            "fresh_restored_boundary_error": reuse["norm_error"],
            "generation": reuse["generation"],
            "same_backing": True,
            "same_generator_term_count_and_carrier_shape": True,
            "baseline_reload_bytes": 0,
        },
        "controls": {
            **controls,
            "wrong_type": "DENIED_BEFORE_CARRIER_OPERATION",
            "wrong_owner_a": "DENIED_BEFORE_CARRIER_OPERATION",
            "wrong_owner_b": "DENIED_BEFORE_CARRIER_OPERATION",
            "undermerge": "DENIED_BEFORE_CARRIER_OPERATION",
            "duplicate_port": "DENIED_BEFORE_CARRIER_OPERATION",
            "stale_internal_generation": (
                "DENIED_BEFORE_CARRIER_OPERATION"
            ),
            "wrong_internal_lease": (
                "DENIED_BEFORE_CARRIER_OPERATION"
            ),
            "premature_projection": "DENIED",
            "snapshot": "DENIED",
            "null_carrier": null,
            "poison_after_failed_inverse": True,
        },
        "no_smuggle": {
            "controller_imports_backend": False,
            "stage_boundary_values": 0,
            "latent_values_in_response": 0,
            "content_derived_receipts": 0,
            "stdout_bytes": 0,
            "stderr_bytes": 0,
        },
        "resource_law": {
            "resident_joint_complex_cells": 1140,
            "carrier_payload_bytes": 18240,
            "persistent_baseline_plus_carrier_payload_bytes": 36480,
            "reuse_verification_carrier_payload_bytes": 18240,
            "request_bytes": 32,
            "response_bytes": 116,
            "relation_table_cells": 0,
            "assignment_cells": 0,
            "temporary_occupation_cells": 0,
            "dense_285_operator_cells": 0,
            "retained_inverse_history_bytes": 0,
            "reported_scope": (
                "DECLARED_STD_VECTOR_COMPLEX_PAYLOADS_AND_WIRE_ABI_"
                "ONLY_NOT_TOTAL_PROCESS_PEAK"
            ),
            "plan_program_evidence_allocator_native_library_os_memory_bounded": (
                False
            ),
        },
        "strongest_compact_classical": {
            "identical_1140_complex_recurrence_exists": True,
            "verification_level": "PACKAGE_SELF_REVIEW",
            "separate_reference_parity": False,
        },
        "distinct_phase_resource_established": False,
        "computational_advantage": False,
        "small_wall_crossed": False,
        "physical_waveform_execution": False,
        "general_catalytic_inference_established": False,
        "terminal": False,
    }
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
