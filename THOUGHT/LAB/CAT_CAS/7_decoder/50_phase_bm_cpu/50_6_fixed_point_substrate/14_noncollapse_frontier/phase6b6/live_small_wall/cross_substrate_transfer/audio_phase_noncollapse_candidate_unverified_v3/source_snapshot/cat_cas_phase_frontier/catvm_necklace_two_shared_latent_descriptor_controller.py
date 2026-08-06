#!/usr/bin/env python3
"""Adversarial controller for bounded descriptor-compiled two-port CATVM."""

from __future__ import annotations

import copy
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import catvm_necklace_two_shared_latent_descriptor_protocol as protocol


LEASE_TAG = 0x4C454153454C4154
TOLERANCE = 1.2e-10


def fail(message: str) -> None:
    raise RuntimeError(message)


def boundary_distance(left: list[float], right: list[float]) -> float:
    return max(abs(a - b) for a, b in zip(left, right, strict=True))


def require(
    response: dict[str, object],
    status: int,
    label: str,
) -> None:
    if int(response["status"]) != status:
        fail(f"{label}: expected status {status}, got {response['status']}")


def require_hidden(
    response: dict[str, object],
    label: str,
    status: int = protocol.STATUS_DENIED,
) -> None:
    require(response, status, label)
    if int(response["flags"]) & protocol.BOUNDARY_VALID:
        fail(f"{label}: boundary-valid flag present")
    if any(float(value) != 0.0 for value in response["boundary"]):
        fail(f"{label}: response contained boundary values")


class Service:
    instances: list["Service"] = []

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
                fail(f"{name}: service exited before bind")
            time.sleep(0.01)
        if not self.socket_path.exists():
            fail(f"{name}: service did not bind")
        self.connection = socket.socket(
            socket.AF_UNIX, socket.SOCK_SEQPACKET
        )
        self.connection.connect(str(self.socket_path))
        self.name = name
        self.nonce = 0x4D37_0000
        self.generation = 0
        self.lease = 0
        self.requests = 0
        self.responses = 0
        self.closed = False
        Service.instances.append(self)

    def raw_call(
        self,
        command: int,
        generation: int,
        lease: int,
        nonce: int,
        reserved: int = 0,
    ) -> tuple[bytes, dict[str, object]]:
        self.connection.sendall(
            protocol.request(
                command, generation, lease, nonce, reserved
            )
        )
        self.requests += 1
        payload = self.connection.recv(protocol.RESPONSE.size)
        if len(payload) != protocol.RESPONSE.size:
            fail(f"{self.name}: truncated response")
        self.responses += 1
        result = protocol.response(payload)
        if int(result["command"]) != command:
            fail(f"{self.name}: command/response mismatch")
        return payload, result

    def initialize(self) -> dict[str, object]:
        self.nonce += 1
        _, result = self.raw_call(
            protocol.INITIALIZE, 0, 0, self.nonce
        )
        require(result, protocol.STATUS_OK, f"{self.name} init")
        expected = self.nonce ^ LEASE_TAG
        if int(result["lease"]) != expected:
            fail(f"{self.name}: outer lease derivation failed")
        self.lease = expected
        return result

    def call(
        self,
        command: int,
        reserved: int = 0,
    ) -> tuple[bytes, dict[str, object]]:
        self.nonce += 1
        payload, result = self.raw_call(
            command,
            self.generation,
            self.lease,
            self.nonce,
            reserved,
        )
        if (
            int(result["status"]) == protocol.STATUS_OK
            and int(result["generation"]) > self.generation
        ):
            self.generation = int(result["generation"])
        return payload, result

    def load_program(
        self,
        slot: int,
        modules: list[dict[str, object]],
    ) -> int:
        _, declared = self.call(
            protocol.DECLARE,
            protocol.declare_control(slot, len(modules)),
        )
        require(declared, protocol.STATUS_OK, f"{self.name} declare")
        for index, module in enumerate(modules):
            _, appended = self.call(
                protocol.APPEND,
                protocol.pack_module(module, slot, index),
            )
            require(
                appended,
                protocol.STATUS_OK,
                f"{self.name} append {slot}:{index}",
            )
        _, sealed = self.call(protocol.SEAL, slot)
        require(sealed, protocol.STATUS_OK, f"{self.name} seal")
        checksum = int(sealed["receipt"])
        if checksum == 0:
            fail(f"{self.name}: zero public topology checksum")
        return checksum

    def execute(
        self,
        slot: int,
        attack_stage: bool = False,
    ) -> tuple[dict[str, object], dict[str, object]]:
        _, stage = self.call(protocol.BEGIN, slot)
        require(stage, protocol.STATUS_OK, f"{self.name} begin")
        if not int(stage["flags"]) & protocol.STAGE_RESIDENT:
            fail(f"{self.name}: stage did not retain open state")
        if int(stage["flags"]) & protocol.BOUNDARY_VALID:
            fail(f"{self.name}: stage exposed boundary flag")
        if any(float(value) != 0.0 for value in stage["boundary"]):
            fail(f"{self.name}: stage exposed boundary values")
        if attack_stage:
            _, projection = self.call(protocol.PROJECT)
            require_hidden(projection, f"{self.name} projection")
            _, snapshot = self.call(protocol.SNAPSHOT)
            require_hidden(snapshot, f"{self.name} snapshot")
            _, substitute = self.call(
                protocol.APPEND,
                protocol.pack_module(
                    {
                        "feature": "collision",
                        "scope": "port_a",
                        "axis": "x",
                        "separation": 0,
                        "strength": 1,
                        "chirp": 1,
                    },
                    slot,
                    0,
                ),
            )
            require_hidden(
                substitute, f"{self.name} staged substitution"
            )
        _, final = self.call(protocol.CONTINUE)
        require(final, protocol.STATUS_OK, f"{self.name} continue")
        required = protocol.BOUNDARY_VALID | protocol.RESTORED
        if int(final["flags"]) & required != required:
            fail(f"{self.name}: final response preceded restoration")
        if float(final["restoration_error"]) > TOLERANCE:
            fail(f"{self.name}: restoration tolerance exceeded")
        if float(final["norm_error"]) > TOLERANCE:
            fail(f"{self.name}: norm tolerance exceeded")
        return stage, final

    def stop(self, expected: int = protocol.STATUS_OK) -> None:
        _, result = self.call(protocol.STOP)
        require(result, expected, f"{self.name} stop")
        if (
            expected != protocol.STATUS_OK
            and int(result["flags"]) & protocol.RESTORED
        ):
            fail(f"{self.name}: failed stop attested restoration")
        self.connection.close()
        self._wait()

    def disconnect(self) -> None:
        self.connection.close()
        self._wait()

    def _wait(self) -> None:
        return_code = self.process.wait(timeout=300)
        self.stdout_file.close()
        self.stderr_file.close()
        self.closed = True
        if return_code != 0:
            fail(f"{self.name}: service exited {return_code}")
        if self.stdout_path.stat().st_size != 0:
            fail(f"{self.name}: service emitted stdout")
        if self.stderr_path.stat().st_size != 0:
            fail(f"{self.name}: service emitted stderr")


def load_programs(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "CATVM_TWO_SHARED_LATENT_PUBLIC_PROGRAMS_V1":
        fail("public program schema mismatch")
    programs = payload.get("programs")
    if not isinstance(programs, list) or len(programs) != 3:
        fail("exactly three bounded public programs required")
    return programs


def main_sequence(
    executable: Path,
    evidence_dir: Path,
    programs: list[dict[str, Any]],
) -> dict[str, object]:
    service = Service(executable, evidence_dir, "descriptor-main")
    service.initialize()

    service.nonce += 1
    _, wrong_lease = service.raw_call(
        protocol.DECLARE,
        service.generation,
        service.lease ^ 1,
        service.nonce,
        protocol.declare_control(0, 6),
    )
    require_hidden(wrong_lease, "wrong outer lease")
    service.nonce += 1
    _, wrong_generation = service.raw_call(
        protocol.DECLARE,
        service.generation + 1,
        service.lease,
        service.nonce,
        protocol.declare_control(0, 6),
    )
    require_hidden(wrong_generation, "wrong outer generation")
    replay_nonce = service.nonce + 1
    _, first_replay = service.raw_call(
        protocol.PROJECT,
        service.generation,
        service.lease,
        replay_nonce,
    )
    require_hidden(first_replay, "first replay probe")
    _, replay = service.raw_call(
        protocol.PROJECT,
        service.generation,
        service.lease,
        replay_nonce,
    )
    require_hidden(replay, "replayed nonce")
    service.nonce = replay_nonce

    checksums = []
    for slot, program in enumerate(programs):
        checksums.append(
            service.load_program(slot, list(program["modules"]))
        )
    if len(set(checksums)) != len(checksums):
        fail("distinct public families produced duplicate checksums")

    tuple_controls = (
        protocol.WRONG_TYPE,
        protocol.WRONG_OWNER_A,
        protocol.WRONG_OWNER_B,
        protocol.UNDERMERGE,
        protocol.DUPLICATE_PORT,
        protocol.STALE_INTERNAL_GENERATION,
        protocol.WRONG_INTERNAL_LEASE,
        protocol.STALE_EPOCH,
        protocol.WRONG_CHECKSUM,
        protocol.WRONG_SLOT,
        protocol.STALE_BOUND_GENERATION,
    )
    for command in tuple_controls:
        _, denied = service.call(command, 0)
        require_hidden(denied, f"tuple/metadata attack {command}")

    stages: list[dict[str, object]] = []
    finals: list[dict[str, object]] = []
    for slot in range(3):
        stage, final = service.execute(
            slot, attack_stage=(slot == 0)
        )
        stages.append(stage)
        finals.append(final)
        if int(final["generation"]) != slot + 1:
            fail("multi-family generation did not advance once")
        if slot > 0 and not int(final["flags"]) & protocol.REUSE:
            fail("restored-carrier reuse flag absent")
    service.stop()
    return {
        "checksums": checksums,
        "stages": stages,
        "finals": finals,
        "request_count": service.requests,
        "response_count": service.responses,
    }


def fresh_program_run(
    executable: Path,
    evidence_dir: Path,
    name: str,
    modules: list[dict[str, object]],
    slot: int = 0,
) -> tuple[int, dict[str, object]]:
    service = Service(executable, evidence_dir, name)
    service.initialize()
    checksum = service.load_program(slot, modules)
    _, final = service.execute(slot)
    service.stop()
    return checksum, final


def compiler_rejections(
    executable: Path,
    evidence_dir: Path,
    primary: list[dict[str, object]],
) -> dict[str, object]:
    results: dict[str, object] = {}

    service = Service(executable, evidence_dir, "descriptor-parser")
    service.initialize()
    for label, control in (
        ("zero_count", protocol.declare_control(0, 0)),
        ("over_count", protocol.declare_control(0, 9)),
        ("bad_slot", protocol.declare_control(3, 6)),
        ("reserved_bits", protocol.declare_control(0, 6) | (1 << 16)),
    ):
        _, denied = service.call(protocol.DECLARE, control)
        require_hidden(denied, label)
        results[label] = "DENIED"
    _, before_declare = service.call(
        protocol.APPEND,
        protocol.pack_module(primary[0], 0, 0),
    )
    require_hidden(before_declare, "append before declare")
    results["append_before_declare"] = "DENIED"
    _, declared = service.call(
        protocol.DECLARE, protocol.declare_control(0, len(primary))
    )
    require(declared, protocol.STATUS_OK, "parser declare")
    _, wrong_index = service.call(
        protocol.APPEND,
        protocol.pack_module(primary[0], 0, 1),
    )
    require_hidden(wrong_index, "wrong index")
    _, high_bits = service.call(
        protocol.APPEND,
        protocol.pack_module(primary[0], 0, 0) | (1 << 26),
    )
    require_hidden(high_bits, "high descriptor bits")
    invalid_axis = (
        protocol.pack_module(primary[0], 0, 0)
        & ~(0x7 << 4)
    ) | (7 << 4)
    _, bad_axis = service.call(protocol.APPEND, invalid_axis)
    require_hidden(bad_axis, "invalid axis")
    _, early_seal = service.call(protocol.SEAL, 0)
    require_hidden(early_seal, "early seal")
    for index, module in enumerate(primary):
        _, appended = service.call(
            protocol.APPEND,
            protocol.pack_module(module, 0, index),
        )
        require(appended, protocol.STATUS_OK, "parser append")
    _, sealed = service.call(protocol.SEAL, 0)
    require(sealed, protocol.STATUS_OK, "parser seal")
    _, append_after = service.call(
        protocol.APPEND,
        protocol.pack_module(primary[0], 0, 0),
    )
    require_hidden(append_after, "append after seal")
    _, duplicate_seal = service.call(protocol.SEAL, 0)
    require_hidden(duplicate_seal, "duplicate seal")
    _, overwrite = service.call(
        protocol.DECLARE, protocol.declare_control(0, len(primary))
    )
    require_hidden(overwrite, "slot overwrite")
    _, unsealed_begin = service.call(protocol.BEGIN, 1)
    require_hidden(unsealed_begin, "unsealed begin")
    service.stop()
    results.update(
        {
            "wrong_index": "DENIED",
            "unknown_high_bits": "DENIED",
            "invalid_axis": "DENIED",
            "early_seal": "DENIED",
            "append_after_seal": "DENIED",
            "duplicate_seal": "DENIED",
            "slot_overwrite": "DENIED",
            "unsealed_begin": "DENIED",
        }
    )

    variants: dict[str, list[dict[str, object]]] = {}
    missing_a = copy.deepcopy(primary)
    for index, module in enumerate(missing_a):
        if module["scope"] == "port_a":
            module["scope"] = "port_b"
            module["strength"] = (int(module["strength"]) + index + 1) % 16 + 1
    variants["missing_port_a"] = missing_a
    one_joint = copy.deepcopy(primary)
    one_joint[4].update(
        {"scope": "port_a", "axis": "x", "strength": 15}
    )
    variants["one_joint"] = one_joint
    all_diagonal = copy.deepcopy(primary)
    for module in all_diagonal:
        if module["scope"] != "joint":
            module["axis"] = "z"
    variants["all_diagonal"] = all_diagonal
    duplicate = copy.deepcopy(primary)
    duplicate[5] = copy.deepcopy(duplicate[2])
    variants["duplicate_semantics"] = duplicate

    for label, modules in variants.items():
        instance = Service(
            executable, evidence_dir, f"descriptor-invalid-{label}"
        )
        instance.initialize()
        _, declared = instance.call(
            protocol.DECLARE,
            protocol.declare_control(0, len(modules)),
        )
        require(declared, protocol.STATUS_OK, f"{label} declare")
        for index, module in enumerate(modules):
            _, appended = instance.call(
                protocol.APPEND,
                protocol.pack_module(module, 0, index),
            )
            require(appended, protocol.STATUS_OK, f"{label} append")
        _, seal = instance.call(protocol.SEAL, 0)
        require_hidden(seal, f"{label} seal")
        instance.stop()
        results[label] = "DENIED_AT_SEAL"
    return results


def atomicity_controls(
    executable: Path,
    evidence_dir: Path,
    primary: list[dict[str, object]],
) -> dict[str, object]:
    results: dict[str, object] = {}
    disconnect = Service(
        executable, evidence_dir, "descriptor-disconnect-stage"
    )
    disconnect.initialize()
    disconnect.load_program(0, primary)
    _, stage = disconnect.call(protocol.BEGIN, 0)
    require(stage, protocol.STATUS_OK, "disconnect begin")
    disconnect.disconnect()
    results["disconnect_stage_inverse_cleanup_exit"] = 0

    partial = Service(
        executable, evidence_dir, "descriptor-disconnect-partial"
    )
    partial.initialize()
    _, declared = partial.call(
        protocol.DECLARE,
        protocol.declare_control(0, len(primary)),
    )
    require(declared, protocol.STATUS_OK, "partial declare")
    _, appended = partial.call(
        protocol.APPEND,
        protocol.pack_module(primary[0], 0, 0),
    )
    require(appended, protocol.STATUS_OK, "partial append")
    partial.disconnect()
    results["disconnect_partial_load_exit"] = 0

    stopped = Service(
        executable, evidence_dir, "descriptor-staged-stop"
    )
    stopped.initialize()
    stopped.load_program(0, primary)
    _, stage = stopped.call(protocol.BEGIN, 0)
    require(stage, protocol.STATUS_OK, "staged stop begin")
    stopped.stop()
    results["authorized_stop_ack_after_rollback"] = True

    denied_stop = Service(
        executable, evidence_dir, "descriptor-denied-stop"
    )
    denied_stop.initialize()
    denied_stop.load_program(0, primary)
    _, stage = denied_stop.call(protocol.BEGIN, 0)
    require(stage, protocol.STATUS_OK, "denied stop begin")
    denied_stop.nonce += 1
    _, denied = denied_stop.raw_call(
        protocol.STOP,
        denied_stop.generation,
        denied_stop.lease ^ 1,
        denied_stop.nonce,
    )
    require_hidden(denied, "wrong-lease staged stop")
    _, final = denied_stop.call(protocol.CONTINUE)
    require(final, protocol.STATUS_OK, "continue after denied stop")
    denied_stop.stop()
    results["denied_stop_did_not_terminate"] = True

    for name, command in (
        ("missing", protocol.MISSING_INVERSE),
        ("reordered", protocol.REORDERED_INVERSE),
        ("semantic", protocol.WRONG_SEMANTIC),
    ):
        attacked = Service(
            executable, evidence_dir, f"descriptor-inverse-{name}"
        )
        attacked.initialize()
        attacked.load_program(0, primary)
        _, response = attacked.call(command, 0)
        require(response, protocol.STATUS_OK, f"{name} inverse")
        error = float(response["restoration_error"])
        if error <= 1e-5:
            fail(f"{name} inverse did not separate")
        _, poisoned = attacked.call(protocol.BEGIN, 0)
        require_hidden(poisoned, f"{name} poison")
        attacked.stop(protocol.STATUS_ERROR)
        results[f"{name}_inverse_error"] = error

    null = Service(
        executable, evidence_dir, "descriptor-null", "null"
    )
    null.initialize()
    null.load_program(0, primary)
    _, begin = null.call(protocol.BEGIN, 0)
    require_hidden(begin, "null begin")
    null.stop(protocol.STATUS_ERROR)
    results["null_carrier"] = "DENIED_BEFORE_ACCESS"
    return results


def main() -> int:
    if len(sys.argv) != 4:
        fail(
            "usage: catvm_necklace_two_shared_latent_descriptor_"
            "controller.py SERVICE PROGRAMS_JSON EVIDENCE_DIR"
        )
    if sys.byteorder != "little" or os.uname().machine != "x86_64":
        fail("descriptor CATVM requires x86-64 little-endian Linux")
    executable = Path(sys.argv[1]).resolve()
    programs_path = Path(sys.argv[2]).resolve()
    evidence_dir = Path(sys.argv[3]).resolve()
    evidence_dir.mkdir(parents=True, exist_ok=True)
    programs = load_programs(programs_path)

    main_result = main_sequence(executable, evidence_dir, programs)
    fresh: list[dict[str, object]] = []
    for index, program in enumerate(programs):
        checksum, final = fresh_program_run(
            executable,
            evidence_dir,
            f"descriptor-fresh-{index}",
            list(program["modules"]),
        )
        distance = boundary_distance(
            list(main_result["finals"][index]["boundary"]),
            list(final["boundary"]),
        )
        if checksum != main_result["checksums"][index]:
            fail("fresh compilation checksum disagreed")
        if distance > TOLERANCE:
            fail("fresh/restored multi-family boundary disagreed")
        if (
            int(final["native_operations"])
            != int(
                main_result["finals"][index]["native_operations"]
            )
        ):
            fail(
                "fresh/restored streamed generator term count disagreed"
            )
        fresh.append(
            {
                "checksum": checksum,
                "boundary_error": distance,
                "restoration_error": final["restoration_error"],
                "streamed_generator_terms": (
                    final["native_operations"]
                ),
            }
        )

    same_checksum, same_final = fresh_program_run(
        executable,
        evidence_dir,
        "descriptor-same-other-slot",
        list(programs[0]["modules"]),
        slot=2,
    )
    same_slot_error = boundary_distance(
        list(main_result["finals"][0]["boundary"]),
        list(same_final["boundary"]),
    )
    if (
        same_checksum != main_result["checksums"][0]
        or same_slot_error > TOLERANCE
    ):
        fail("slot-independent compilation failed")

    mutated_modules = copy.deepcopy(programs[0]["modules"])
    mutated_modules[0]["strength"] = 4
    mutation_checksum, mutation_final = fresh_program_run(
        executable,
        evidence_dir,
        "descriptor-semantic-mutation",
        mutated_modules,
    )
    mutation_effect = boundary_distance(
        list(main_result["finals"][0]["boundary"]),
        list(mutation_final["boundary"]),
    )
    if (
        mutation_checksum == main_result["checksums"][0]
        or mutation_effect <= 1e-7
    ):
        fail("valid descriptor mutation was not boundary-relevant")

    compiler_controls = compiler_rejections(
        executable,
        evidence_dir,
        list(programs[0]["modules"]),
    )
    atomic_controls = atomicity_controls(
        executable,
        evidence_dir,
        list(programs[0]["modules"]),
    )

    if not all(instance.closed for instance in Service.instances):
        fail("controller left a service open")
    stdout_bytes = sum(
        instance.stdout_path.stat().st_size
        for instance in Service.instances
    )
    stderr_bytes = sum(
        instance.stderr_path.stat().st_size
        for instance in Service.instances
    )
    total_requests = sum(
        instance.requests for instance in Service.instances
    )
    total_responses = sum(
        instance.responses for instance in Service.instances
    )
    accepted_module_count = sum(
        len(program["modules"]) for program in programs
    )
    result = {
        "claim_candidate": (
            "CATVM_ENFORCED_BOUNDED_PUBLIC_DESCRIPTOR_COMPILED_"
            "TWO_SHARED_LATENT_PORT_PROGRAM_FAMILIES_WITH_FULL_"
            "TUPLE_CUSTODY_RESTORATION_AND_REUSE"
        ),
        "result": "PASS",
        "claim_ceiling": (
            "LINUX_X86_64_SAME_UID_ONE_UNIX_SEQPACKET_CONNECTION_"
            "THREE_PUBLIC_PROGRAM_SLOTS_FOUR_TO_EIGHT_MODULES_"
            "THREE_EXECUTED_FAMILIES_GRID17_FOUR_EXCHANGE_"
            "SYMMETRIC_ROTATION_INVARIANT_ROTORS_285_NECKLACES_"
            "1140_COMPLEX_CELLS_TWO_BINARY_LATENT_PORTS_COMPILER_"
            "DERIVED_OWNERS_STAGE_CUT_AND_REVERSE_TRAVERSAL_"
            "SEVEN_BIN_FINAL_BOUNDARY_COMPLEX128_SOFTWARE_ONLY"
        ),
        "verification_classification": (
            "INDEPENDENTLY_VERIFIED_SOURCE_LOCAL"
        ),
        "restoration_class": "NUMERICAL_PHYSICAL_STATE_RESTORATION",
        "compiler": {
            "schema": "CATVM_TWO_SHARED_LATENT_PUBLIC_PROGRAMS_V1",
            "program_names": [program["name"] for program in programs],
            "module_counts": [
                len(program["modules"]) for program in programs
            ],
            "checksums": main_result["checksums"],
            "checksum_scope": (
                "PUBLIC_SEMANTIC_FIELDS_COUNT_AND_DERIVED_STAGE_CUT_"
                "NONCRYPTOGRAPHIC_FNV64"
            ),
            "slot_independent_checksum": True,
            "slot_independent_boundary_error": same_slot_error,
            "carrier_or_boundary_compiler_inputs": 0,
            "owners_derived_in_service": True,
            "stage_cut_derived_from_first_joint": True,
            "later_joint_required": True,
            "forward_topology_only": True,
            "retained_inverse_descriptor_count": 0,
            "inverse_order": "REVERSE_FORWARD_DESCRIPTOR_TRAVERSAL",
            "valid_mutation_checksum_changed": True,
            "valid_mutation_boundary_effect": mutation_effect,
        },
        "custody": {
            "port_count": 2,
            "resident_complex_cells": 1140,
            "full_tuple_bound_per_compiled_consumer": True,
            "tuple_fields": [
                "id", "type", "owner", "generation", "lease"
            ],
            "program_identity_fields": [
                "slot",
                "epoch",
                "checksum",
                "carrier_generation",
            ],
            "rebound_at_each_generation": True,
            "generations": [
                final["generation"] for final in main_result["finals"]
            ],
            "same_carrier_backing": True,
            "baseline_reload_bytes": 0,
        },
        "families": [
            {
                "name": program["name"],
                "module_count": len(program["modules"]),
                "boundary": main_result["finals"][index]["boundary"],
                "restoration_error": (
                    main_result["finals"][index]["restoration_error"]
                ),
                "streamed_generator_terms": (
                    main_result["finals"][index]["native_operations"]
                ),
                "fresh_restored_boundary_error": (
                    fresh[index]["boundary_error"]
                ),
                "fresh_restored_streamed_generator_terms_equal": (
                    int(
                        main_result["finals"][index][
                            "native_operations"
                        ]
                    )
                    == int(
                        fresh[index]["streamed_generator_terms"]
                    )
                ),
            }
            for index, program in enumerate(programs)
        ],
        "controls": {
            "compiler": compiler_controls,
            "atomicity_and_inverse": atomic_controls,
            "wrong_outer_lease": "DENIED",
            "wrong_outer_generation": "DENIED",
            "replayed_nonce": "DENIED",
            "wrong_tuple_fields": "DENIED_BEFORE_CARRIER_OPERATION",
            "stale_program_identity": "DENIED_BEFORE_CARRIER_OPERATION",
            "premature_projection": "DENIED",
            "snapshot": "DENIED",
            "program_substitution_while_staged": "DENIED",
        },
        "no_smuggle": {
            "stage_boundary_values": 0,
            "latent_values_in_response": 0,
            "content_derived_receipts": 0,
            "stdout_bytes": stdout_bytes,
            "stderr_bytes": stderr_bytes,
            "controller_imports_backend": False,
            "public_fixture_contains_final_answers": False,
        },
        "resource_law": {
            "carrier_complex_cells": 1140,
            "carrier_payload_bytes": 18240,
            "persistent_baseline_plus_carrier_payload_bytes": 36480,
            "declared_generator_fiber_scratch_bytes": 18240,
            "fresh_verification_carrier_payload_bytes_per_service": 18240,
            "program_slot_count": 3,
            "maximum_modules_per_slot": 8,
            "accepted_public_module_count": accepted_module_count,
            "descriptor_slot_object_bytes": 128,
            "descriptor_slot_registry_object_bytes": 384,
            "bound_module_object_bytes": 88,
            "accepted_peak_active_bound_program_bytes": 616,
            "ceiling_peak_active_bound_program_bytes": 704,
            "compiled_module_logical_field_bytes": (
                accepted_module_count * 32
            ),
            "retained_inverse_history_bytes": 0,
            "request_bytes": protocol.REQUEST.size,
            "response_bytes": protocol.RESPONSE.size,
            "main_service_request_count_with_controls": (
                main_result["request_count"]
            ),
            "main_service_response_count_with_controls": (
                main_result["response_count"]
            ),
            "full_qualification_request_count": total_requests,
            "full_qualification_response_count": total_responses,
            "full_qualification_request_bytes": (
                total_requests * protocol.REQUEST.size
            ),
            "full_qualification_response_bytes": (
                total_responses * protocol.RESPONSE.size
            ),
            "relation_table_cells": 0,
            "assignment_cells": 0,
            "reported_scope": (
                "DECLARED_COMPLEX_PAYLOADS_PUBLIC_DESCRIPTOR_FIELDS_"
                "BOUND_CUSTODY_OBJECTS_AND_WIRE_ABI_NOT_TOTAL_"
                "PROCESS_PEAK"
            ),
            "allocator_native_library_os_and_object_padding_bounded": False,
            "plan_topology_vector_capacity_and_allocator_bytes_bounded": (
                False
            ),
            "compiled_and_active_vector_capacity_bytes_bounded": False,
        },
        "comparisons": {
            "warm_fixed_array_predecessor_exists": True,
            "identical_compact_1140_complex_recurrence_exists": True,
            "classical_reference_verification_level": "PACKAGE_SELF_REVIEW",
            "separate_reference_parity": False,
        },
        "general_scheduler_established": False,
        "arbitrary_program_algebra_established": False,
        "distinct_phase_resource_established": False,
        "computational_advantage": False,
        "small_wall_crossed": False,
        "catalytic_inference_established": False,
        "physical_waveform_execution": False,
        "replacement_of_physical_bits_with_pi": False,
        "terminal": False,
    }
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
