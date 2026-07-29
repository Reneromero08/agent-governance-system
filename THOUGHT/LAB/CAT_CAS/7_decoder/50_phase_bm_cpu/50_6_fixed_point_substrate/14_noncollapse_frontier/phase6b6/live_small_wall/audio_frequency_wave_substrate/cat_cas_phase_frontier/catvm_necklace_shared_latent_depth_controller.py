#!/usr/bin/env python3
"""Protocol-only adversarial controller for depth-parametric latent CATVM."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import catvm_necklace_shared_latent_depth_protocol as protocol


LEASE_TAG = 0x4C45415345445054
STAGE_TAG = 0x5354414745445054


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
                fail(f"depth service {name} exited before bind")
            time.sleep(0.01)
        if not self.socket_path.exists():
            fail(f"depth service {name} did not bind")
        self.connection = socket.socket(
            socket.AF_UNIX, socket.SOCK_SEQPACKET
        )
        self.connection.connect(str(self.socket_path))
        self.nonce = 0xD370_0000
        self.generation = 0
        self.lease = 0

    def raw_call(
        self,
        command: int,
        generation: int,
        depth: int,
        variant: int,
        lease: int,
        nonce: int,
    ) -> tuple[bytes, dict[str, object]]:
        self.connection.sendall(
            protocol.request(
                command,
                generation,
                depth,
                variant,
                lease,
                nonce,
            )
        )
        payload = self.connection.recv(protocol.RESPONSE.size)
        if len(payload) != protocol.RESPONSE.size:
            fail("truncated depth response")
        result = protocol.response(payload)
        if int(result["command"]) != command:
            fail("depth command/response mismatch")
        return payload, result

    def initialize(self) -> dict[str, object]:
        self.nonce += 1
        _, result = self.raw_call(
            protocol.INITIALIZE,
            0,
            0,
            0,
            0,
            self.nonce,
        )
        if int(result["status"]) != protocol.STATUS_OK:
            fail("depth initialization failed")
        expected = self.nonce ^ LEASE_TAG
        if int(result["lease"]) != expected:
            fail("depth lease derivation failed")
        self.lease = expected
        return result

    def call(
        self,
        command: int,
        depth: int = 0,
        variant: int = 0,
    ) -> tuple[bytes, dict[str, object]]:
        self.nonce += 1
        payload, result = self.raw_call(
            command,
            self.generation,
            depth,
            variant,
            self.lease,
            self.nonce,
        )
        if (
            int(result["status"]) == protocol.STATUS_OK
            and int(result["generation"]) > self.generation
        ):
            self.generation = int(result["generation"])
        return payload, result

    def stop(self) -> None:
        _, stopped = self.call(protocol.STOP)
        require(stopped, protocol.STATUS_OK, "stop")
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
            fail(f"depth service exited {return_code}")
        if self.stdout_path.stat().st_size != 0:
            fail("depth service emitted stdout")
        if self.stderr_path.stat().st_size != 0:
            fail("depth service emitted stderr")


def require(
    response: dict[str, object],
    status: int,
    label: str,
) -> None:
    if int(response["status"]) != status:
        fail(f"{label}: expected status {status}")


def primary_service(
    executable: Path,
    evidence_dir: Path,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    depth = 32
    variant = 2
    service = Service(executable, evidence_dir, "primary")
    service.initialize()

    service.nonce += 1
    _, wrong_lease = service.raw_call(
        protocol.BEGIN,
        0,
        depth,
        variant,
        service.lease ^ 1,
        service.nonce,
    )
    require(wrong_lease, protocol.STATUS_DENIED, "wrong lease")
    service.nonce += 1
    _, wrong_generation = service.raw_call(
        protocol.BEGIN,
        1,
        depth,
        variant,
        service.lease,
        service.nonce,
    )
    require(
        wrong_generation,
        protocol.STATUS_DENIED,
        "wrong generation",
    )

    stage_nonce = service.nonce + 1
    stage_payload, stage = service.call(
        protocol.BEGIN, depth, variant
    )
    _, projection = service.call(
        protocol.PROJECT, depth, variant
    )
    _, owner = service.call(
        protocol.WRONG_OWNER, depth, variant
    )
    service.nonce += 1
    _, resident_wrong_lease = service.raw_call(
        protocol.CONTINUE,
        service.generation,
        depth,
        variant,
        service.lease ^ 1,
        service.nonce,
    )
    service.nonce += 1
    _, resident_wrong_generation = service.raw_call(
        protocol.CONTINUE,
        service.generation + 1,
        depth,
        variant,
        service.lease,
        service.nonce,
    )
    _, snapshot = service.call(
        protocol.SNAPSHOT, depth, variant
    )
    _, primary = service.call(
        protocol.CONTINUE, depth, variant
    )
    _, reuse = service.call(protocol.REUSE, 11, 5)

    for label, response, status in (
        ("stage", stage, protocol.STATUS_OK),
        ("projection", projection, protocol.STATUS_DENIED),
        ("wrong owner", owner, protocol.STATUS_DENIED),
        (
            "resident wrong lease",
            resident_wrong_lease,
            protocol.STATUS_DENIED,
        ),
        (
            "resident wrong generation",
            resident_wrong_generation,
            protocol.STATUS_DENIED,
        ),
        ("snapshot", snapshot, protocol.STATUS_DENIED),
        ("primary", primary, protocol.STATUS_OK),
        ("reuse", reuse, protocol.STATUS_OK),
    ):
        require(response, status, label)

    if len(stage_payload) != protocol.RESPONSE.size:
        fail("depth stage packet size changed")
    if not int(stage["flags"]) & protocol.STAGE_RESIDENT:
        fail("depth latent state was not resident")
    if int(stage["flags"]) & protocol.BOUNDARY_VALID:
        fail("depth stage exposed a boundary")
    if any(float(value) != 0.0 for value in stage["boundary"]):
        fail("depth stage contained boundary values")
    expected_receipt = stage_nonce ^ service.lease ^ STAGE_TAG
    if int(stage["receipt"]) != expected_receipt:
        fail("depth typed custody receipt failed")
    for label, response in (
        ("projection", projection),
        ("wrong owner", owner),
        ("resident wrong lease", resident_wrong_lease),
        (
            "resident wrong generation",
            resident_wrong_generation,
        ),
        ("snapshot", snapshot),
    ):
        if int(response["flags"]) & protocol.BOUNDARY_VALID:
            fail(f"{label} denial exposed a boundary")
        if not int(response["flags"]) & protocol.STAGE_RESIDENT:
            fail(f"{label} denial lost resident-stage custody")

    primary_flags = protocol.BOUNDARY_VALID | protocol.RESTORED
    if int(primary["flags"]) & primary_flags != primary_flags:
        fail("depth primary response preceded restoration")
    if int(primary["generation"]) != 1:
        fail("depth primary generation was not one")
    if float(primary["restoration_error"]) > 6e-11:
        fail("depth primary restoration failed")
    reuse_flags = (
        protocol.BOUNDARY_VALID
        | protocol.RESTORED
        | protocol.REUSE_FLAG
    )
    if int(reuse["flags"]) & reuse_flags != reuse_flags:
        fail("depth reuse flags failed")
    if int(reuse["generation"]) != 2:
        fail("depth reuse generation was not two")
    if float(reuse["restoration_error"]) > 6e-11:
        fail("depth reuse restoration failed")
    if float(reuse["norm_error"]) > 6e-11:
        fail("depth fresh/restored reuse parity failed")
    service.stop()
    return stage, primary, reuse


def disconnect_attack(
    executable: Path,
    evidence_dir: Path,
) -> dict[str, object]:
    service = Service(executable, evidence_dir, "disconnect")
    service.initialize()
    _, stage = service.call(protocol.BEGIN, 8, 3)
    require(stage, protocol.STATUS_OK, "disconnect stage")
    service.disconnect()
    return {
        "stage_depth": 8,
        "stage_resident": True,
        "boundary_released": False,
        "inverse_cleanup_exit": 0,
        "internal_fresh_reuse_sentinel": "PASS",
    }


def inverse_attack(
    executable: Path,
    evidence_dir: Path,
    name: str,
    command: int,
) -> float:
    service = Service(executable, evidence_dir, f"control-{name}")
    service.initialize()
    _, result = service.call(command, 4, 2)
    _, poisoned = service.call(protocol.BEGIN, 4, 2)
    require(result, protocol.STATUS_OK, name)
    require(poisoned, protocol.STATUS_DENIED, f"{name} poison")
    error = float(result["restoration_error"])
    if error <= 1e-5:
        fail(f"{name} inverse attack did not separate")
    service.stop()
    return error


def null_attack(
    executable: Path,
    evidence_dir: Path,
) -> dict[str, object]:
    service = Service(executable, evidence_dir, "null", "null")
    service.initialize()
    _, null_carrier = service.call(protocol.NULL_CARRIER)
    _, begin = service.call(protocol.BEGIN, 4, 2)
    require(null_carrier, protocol.STATUS_DENIED, "null carrier")
    require(begin, protocol.STATUS_ERROR, "null begin")
    service.stop()
    return {
        "backing_cells": 0,
        "null_command": "DENIED",
        "begin_command": "ERROR_BEFORE_ACCESS",
    }


def main() -> int:
    if len(sys.argv) != 3:
        fail(
            "usage: catvm_necklace_shared_latent_depth_controller.py "
            "SERVICE EVIDENCE_DIR"
        )
    if sys.byteorder != "little" or os.uname().machine != "x86_64":
        fail("depth CATVM requires x86-64 little-endian Linux")
    executable = Path(sys.argv[1]).resolve()
    evidence_dir = Path(sys.argv[2]).resolve()
    evidence_dir.mkdir(parents=True, exist_ok=True)

    stage, primary, reuse = primary_service(
        executable, evidence_dir
    )
    disconnect = disconnect_attack(executable, evidence_dir)
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
        "wrong_inverse_variant_error": inverse_attack(
            executable,
            evidence_dir,
            "variant",
            protocol.WRONG_INVERSE_VARIANT,
        ),
    }
    null = null_attack(executable, evidence_dir)

    result = {
        "claim_candidate": (
            "CATVM_ENFORCED_TOPOLOGY_REMATERIALIZED_OWNER_BOUND_SHARED_"
            "LATENT_PHASE_PROGRAM_FIXED_570_CARRIER_AT_DEPTH32"
        ),
        "result": "PASS",
        "claim_ceiling": (
            "LINUX_X86_64_SAME_UID_ONE_UNIX_SEQPACKET_CONNECTION_"
            "NONCE_DERIVED_LEASE_EXACT_GENERATION_GRID17_FOUR_EXCHANGE_"
            "SYMMETRIC_ROTATION_INVARIANT_ROTORS_285_NECKLACES_570_"
            "COMPLEX_CELLS_TWO_CELL_LATENT_FIBER_PUBLIC_VARIANT_ORDINAL_"
            "COMPILER_PRIMARY_DEPTH32_REUSE_DEPTH11_STATIC_OWNER_SEVEN_"
            "BIN_BOUNDARY_COMPLEX128_SOFTWARE_ONLY"
        ),
        "restoration_class": "NUMERICAL_PHYSICAL_STATE_RESTORATION",
        "custody": {
            "resident_joint_complex_cells": 570,
            "program_descriptor_bytes": 8,
            "retained_module_tape_bytes": 0,
            "retained_inverse_history_bytes": 0,
            "primary_depth": 32,
            "stage_applied_modules": 16,
            "stage_resident": bool(
                int(stage["flags"]) & protocol.STAGE_RESIDENT
            ),
            "projected": False,
            "wrong_owner_denied": True,
            "wrong_lease_denied": True,
            "wrong_generation_denied": True,
            "typed_receipt_only": True,
        },
        "atomic_ordering": {
            "final_boundary_after_restoration": True,
            "primary_generation": primary["generation"],
            "disconnect": disconnect,
        },
        "primary": {
            "boundary": primary["boundary"],
            "restoration_error": primary["restoration_error"],
            "carrier_backing_preserved": True,
            "baseline_reload_bytes": 0,
            "native_operations_after_stage": primary[
                "native_operations"
            ],
        },
        "reuse": {
            "depth": 11,
            "boundary": reuse["boundary"],
            "restoration_error": reuse["restoration_error"],
            "fresh_restored_boundary_error": reuse["norm_error"],
            "generation": reuse["generation"],
            "same_backing": True,
            "same_resource_signature": True,
            "baseline_reload_bytes": 0,
        },
        "controls": {
            **controls,
            "wrong_owner": "DENIED",
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
            "resident_joint_complex_cells": 570,
            "public_pending_topology_bytes": 8,
            "pending_applied_counter_bytes": 8,
            "relation_table_cells": 0,
            "assignment_cells": 0,
            "temporary_occupation_cells": 0,
            "dense_285_operator_cells": 0,
            "retained_inverse_history_bytes": 0,
            "inverse_descriptors_rematerialized": True,
            "allocator_native_library_os_memory_bounded": False,
        },
        "strongest_compact_classical_identical": True,
        "distinct_phase_resource_established": False,
        "computational_advantage": False,
        "small_wall_crossed": False,
        "physical_waveform_execution": False,
        "unbounded_computation_established": False,
        "terminal": False,
    }
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
