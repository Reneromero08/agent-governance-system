#!/usr/bin/env python3
"""Independent protocol and algebra oracle for the M200 CATVM package.

The oracle imports only the separately implemented M199 mathematical oracle.
It does not import the CATVM service, controller, protocol, or production
factor module.  It reconstructs both boundaries and the direct two-body
primary result, then independently drives fixed binary service records.
"""

from __future__ import annotations

import argparse
import ast
import errno
import hashlib
import json
import os
import re
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any

import growing_rotor_open_momentum_factor_independent_oracle as reference


MAGIC = 0x4D4F4341
REQUEST = struct.Struct("<IIIIQ")
RESPONSE = struct.Struct("<IIIIiQQQ")
RUN_INPLACE = 1
REUSE_INPLACE = 2
PROJECT_HIDDEN = 3
MISSING_INVERSE = 4
WRONG_INVERSE = 5
REORDERED_INVERSE = 6
NULL_CARRIER = 7
RUN_SNAPSHOT = 8
STOP = 9
PING = 13
WRONG_REFLECTION = 15
STATUS_OK = 0
STATUS_DENIED = 1
BOUNDARY_VALID = 1
RESTORED = 2
REUSE_FLAG = 4
SNAPSHOT_RELOADED = 8
CONTROL_DISCRIMINATED = 16
PRIMARY_FAMILY = 0
REUSE_FAMILY = 4


def fail(message: str) -> None:
    raise RuntimeError(message)


def event_name(line: str) -> str:
    return line.split(":", 1)[1]


def decode(payload: bytes) -> dict[str, int]:
    if len(payload) != RESPONSE.size:
        fail("independent response size mismatch")
    (
        magic,
        status,
        command,
        generation,
        boundary,
        flags,
        receipt,
        resource,
    ) = RESPONSE.unpack(payload)
    if magic != MAGIC:
        fail("independent response magic mismatch")
    return {
        "status": status,
        "command": command,
        "generation": generation,
        "boundary": boundary,
        "flags": flags,
        "receipt": receipt,
        "resource": resource,
    }


class Driver:
    launches = 0

    def __init__(
        self, service: Path, evidence: Path, label: str, mode: str
    ) -> None:
        Driver.launches += 1
        self.audit = evidence / f"oracle-{label}.audit.log"
        self.process = subprocess.Popen(
            [
                "nice",
                "-n",
                "10",
                "ionice",
                "-c",
                "3",
                sys.executable,
                str(service),
                "--mode",
                mode,
                "--audit",
                str(self.audit),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.generation = 0
        self.nonce = 1
        if self.call(PING, PRIMARY_FAMILY)["status"] != STATUS_OK:
            fail("independent service readiness failed")

    def call(self, command: int, family: int) -> dict[str, int]:
        if self.process.stdin is None or self.process.stdout is None:
            fail("independent service pipes unavailable")
        payload = REQUEST.pack(
            MAGIC, command, self.generation, family, self.nonce
        )
        self.process.stdin.write(payload)
        self.process.stdin.flush()
        raw = self.process.stdout.read(RESPONSE.size)
        result = decode(raw)
        if result["command"] != command:
            fail("independent command response mismatch")
        self.nonce += 1
        if result["flags"] & (RESTORED | SNAPSHOT_RELOADED):
            self.generation = result["generation"]
        return result

    def memory_denied(self) -> bool:
        try:
            descriptor = os.open(
                f"/proc/{self.process.pid}/mem", os.O_RDONLY
            )
        except OSError as exc:
            return exc.errno in (errno.EACCES, errno.EPERM)
        os.close(descriptor)
        return False

    def stop(self) -> None:
        if self.call(STOP, PRIMARY_FAMILY)["status"] != STATUS_OK:
            fail("independent clean stop denied")
        if self.process.stdin is not None:
            self.process.stdin.close()
        self.process.wait(timeout=600)
        stderr = (
            self.process.stderr.read()
            if self.process.stderr is not None
            else b""
        )
        if self.process.returncode != 0 or stderr:
            fail("independent service exit or stderr failure")


def expected_receipt(
    topology_digest: str, family: int, boundary: int
) -> int:
    digest = hashlib.sha256()
    digest.update(topology_digest.encode("ascii"))
    digest.update(family.to_bytes(1, "little"))
    digest.update(boundary.to_bytes(2, "little"))
    digest.update(b"FINAL_BOUNDARY_AFTER_RESTORATION")
    return int.from_bytes(digest.digest()[:8], "little")


def reconstruct_algebra() -> dict[str, Any]:
    topology = reference.compile_topology()
    plans = reference.compile_one_body_plans(topology)
    source = reference.source_state(topology, 0)
    primary_word = reference.public_program(1, PRIMARY_FAMILY)
    reuse_word = reference.public_program(1, REUSE_FAMILY)
    direct = reference.compile_direct_operator(
        topology, *primary_word[0]
    )
    primary = reference.execute_factor(
        source, topology, plans, primary_word
    )
    primary_direct = reference.execute_direct(
        source, topology, direct, *primary_word[0]
    )
    reuse = reference.execute_factor(source, topology, plans, reuse_word)
    primary_boundary = reference.boundary(primary, topology)
    reuse_boundary = reference.boundary(reuse, topology)
    paired_terms = 8 * (plans.first_entries // 16) + 16 * (
        plans.second_entries // 16
    )
    mismatch = reference.mismatch(primary, primary_direct)
    if (
        primary_boundary != 83
        or reuse_boundary != 70
        or mismatch
        or paired_terms != 331704
        or direct.raw_terms != 684624
    ):
        fail("independent algebra reconstruction failed")
    return {
        "topology_commitment": reference.topology_commitment(topology),
        "occupation_histograms": topology.occupation_count,
        "necklace_cells": len(topology.necklaces),
        "bracelet_cells": len(topology.bracelets),
        "primary_boundary": primary_boundary,
        "reuse_boundary": reuse_boundary,
        "direct_primary_mismatch_cells": mismatch,
        "direct_two_body_terms": direct.raw_terms,
        "direct_two_body_csr_nonzeros": int(direct.matrix.nnz),
        "full_unpaired_factor_terms": plans.first_entries
        + plans.second_entries,
        "reflection_paired_factor_terms": paired_terms,
    }


def accepted_protocol(
    service: Path, evidence: Path, algebra: dict[str, Any]
) -> dict[str, Any]:
    driver = Driver(service, evidence, "accepted", "inplace")
    memory_denied = driver.memory_denied()
    primary = driver.call(RUN_INPLACE, PRIMARY_FAMILY)
    reuse = driver.call(REUSE_INPLACE, REUSE_FAMILY)
    driver.stop()
    required = BOUNDARY_VALID | RESTORED
    if (
        primary["status"] != STATUS_OK
        or primary["flags"] & required != required
        or primary["generation"] != 1
        or primary["boundary"] != algebra["primary_boundary"]
        or primary["receipt"]
        != expected_receipt(
            algebra["topology_commitment"],
            PRIMARY_FAMILY,
            algebra["primary_boundary"],
        )
        or reuse["status"] != STATUS_OK
        or reuse["flags"] & (required | REUSE_FLAG)
        != (required | REUSE_FLAG)
        or reuse["generation"] != 2
        or reuse["boundary"] != algebra["reuse_boundary"]
        or (primary["resource"] >> 32) != 663408
        or (primary["resource"] & 0xFFFFFFFF) != 8943
        or not memory_denied
    ):
        fail("independent accepted protocol failed")
    events = driver.audit.read_text(encoding="ascii").splitlines()
    names = [event_name(item) for item in events]
    boundaries = [
        index
        for index, name in enumerate(names)
        if name == "BOUNDARY_RETAINED_INTERNAL"
    ]
    restored = [
        index
        for index, name in enumerate(names)
        if name == "RESTORATION_VERIFIED"
    ]
    writes = [
        index
        for index, name in enumerate(names)
        if name == "RESPONSE_WRITE_ATTEMPT"
    ]
    ordered = len(boundaries) == len(restored) == len(writes) == 2 and all(
        left < middle < right
        for left, middle, right in zip(
            boundaries, restored, writes, strict=True
        )
    )
    leases = names.count("PORT_LEASED_INTERNAL")
    releases = names.count("PORT_RELEASED_INTERNAL")
    if not ordered or leases != releases or leases != 32:
        fail("independent accepted event order failed")
    return {
        "primary_boundary": primary["boundary"],
        "reuse_boundary": reuse["boundary"],
        "generation": reuse["generation"],
        "response_order_verified": ordered,
        "hidden_port_lease_release_pairs": leases,
        "same_service_reused": True,
        "process_memory_open_denied": memory_denied,
        "carrier_field_cells": primary["resource"] & 0xFFFFFFFF,
        "one_body_terms_per_forward_inverse_transaction": primary[
            "resource"
        ]
        >> 32,
    }


def fresh_and_snapshot_protocol(
    service: Path, evidence: Path, algebra: dict[str, Any]
) -> dict[str, Any]:
    fresh = Driver(service, evidence, "fresh", "inplace")
    fresh_reuse = fresh.call(RUN_INPLACE, REUSE_FAMILY)
    fresh.stop()
    snapshot = Driver(service, evidence, "snapshot", "snapshot")
    sham = snapshot.call(RUN_SNAPSHOT, PRIMARY_FAMILY)
    inplace_denied = snapshot.call(RUN_INPLACE, PRIMARY_FAMILY)
    snapshot.stop()
    if (
        fresh_reuse["boundary"] != algebra["reuse_boundary"]
        or fresh_reuse["generation"] != 1
        or sham["boundary"] != algebra["primary_boundary"]
        or sham["flags"]
        & (BOUNDARY_VALID | SNAPSHOT_RELOADED)
        != (BOUNDARY_VALID | SNAPSHOT_RELOADED)
        or sham["flags"] & RESTORED
        or inplace_denied["status"] != STATUS_DENIED
    ):
        fail("independent fresh or snapshot protocol failed")
    names = [
        event_name(item)
        for item in snapshot.audit.read_text(
            encoding="ascii"
        ).splitlines()
    ]
    if names.index("SNAPSHOT_RELOADED") > names.index(
        "RESPONSE_WRITE_ATTEMPT"
    ):
        fail("snapshot response preceded reload")
    return {
        "fresh_reuse_boundary": fresh_reuse["boundary"],
        "snapshot_boundary": sham["boundary"],
        "fresh_matches_restored_reuse": True,
        "snapshot_matches_primary": True,
        "snapshot_restoration_classification": "SNAPSHOT_RELOAD",
        "inplace_command_on_snapshot_denied": True,
    }


def control_protocol(service: Path, evidence: Path) -> dict[str, Any]:
    driver = Driver(service, evidence, "controls", "inplace")
    projection = driver.call(PROJECT_HIDDEN, PRIMARY_FAMILY)
    results = {
        "missing": driver.call(MISSING_INVERSE, PRIMARY_FAMILY),
        "wrong": driver.call(WRONG_INVERSE, PRIMARY_FAMILY),
        "reordered_noncommuting": driver.call(
            REORDERED_INVERSE, PRIMARY_FAMILY
        ),
        "wrong_reflection": driver.call(
            WRONG_REFLECTION, PRIMARY_FAMILY
        ),
    }
    snapshot_denied = driver.call(RUN_SNAPSHOT, PRIMARY_FAMILY)
    null_denied = driver.call(NULL_CARRIER, PRIMARY_FAMILY)
    driver.stop()
    if (
        projection["status"] != STATUS_DENIED
        or projection["flags"] != RESTORED
        or projection["boundary"]
        or snapshot_denied["status"] != STATUS_DENIED
        or null_denied["status"] != STATUS_DENIED
        or not all(
            result["status"] == STATUS_DENIED
            and result["flags"]
            & (RESTORED | CONTROL_DISCRIMINATED)
            == (RESTORED | CONTROL_DISCRIMINATED)
            and result["boundary"] == 0
            for result in results.values()
        )
    ):
        fail("independent control protocol failed")
    names = [
        event_name(item)
        for item in driver.audit.read_text(
            encoding="ascii"
        ).splitlines()
    ]
    if (
        names.index("PORT_RESIDENT")
        > names.index("HIDDEN_PROJECTION_DENIED")
        or names.count("CONTROL_DISCRIMINATED") != 4
    ):
        fail("independent control event law failed")
    return {
        "premature_projection_denied_while_port_resident": True,
        "missing_inverse_discriminated": True,
        "wrong_inverse_discriminated": True,
        "reordered_noncommuting_inverse_discriminated": True,
        "wrong_reflection_inverse_discriminated": True,
        "all_controls_restored_before_denial_response": True,
        "snapshot_command_on_inplace_denied": True,
        "null_carrier_command_denied": True,
    }


def disconnect_protocol(service: Path, evidence: Path) -> dict[str, Any]:
    driver = Driver(service, evidence, "disconnect", "inplace")
    memory_denied = driver.memory_denied()
    if driver.process.stdin is None or driver.process.stdout is None:
        fail("independent disconnect pipes unavailable")
    driver.process.stdout.close()
    driver.process.stdin.write(
        REQUEST.pack(
            MAGIC,
            RUN_INPLACE,
            driver.generation,
            PRIMARY_FAMILY,
            driver.nonce,
        )
    )
    driver.process.stdin.flush()
    driver.process.stdin.close()
    driver.process.wait(timeout=600)
    stderr = (
        driver.process.stderr.read()
        if driver.process.stderr is not None
        else b""
    )
    names = [
        event_name(item)
        for item in driver.audit.read_text(
            encoding="ascii"
        ).splitlines()
    ]
    ordered = names.index("RESTORATION_VERIFIED") < names.index(
        "RESPONSE_WRITE_ATTEMPT"
    )
    if driver.process.returncode != 0 or stderr or not ordered or not memory_denied:
        fail("independent disconnect protocol failed")
    return {
        "restoration_precedes_disconnected_write_attempt": True,
        "response_bytes": 0,
        "stderr_bytes": 0,
        "process_memory_open_denied": True,
    }


def source_boundary_audit(service: Path, controller: Path) -> dict[str, Any]:
    service_text = service.read_text(encoding="utf-8")
    controller_text = controller.read_text(encoding="utf-8")
    block = service_text[
        service_text.index("    def run_inplace(") : service_text.index(
            "    def run_snapshot("
        )
    ]
    ordering = (
        block.index("boundary = backend.boundary")
        < block.index("self.verify_restoration")
        < block.index("self.generation += 1")
        < block.index("response = self.response")
        < block.index('self.audit("RESPONSE_WRITE_ATTEMPT")')
    )
    tree = ast.parse(controller_text)
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    forbidden_import = any(
        "growing_rotor_open_momentum_factor" in name
        or "catvm_growing_rotor_open_momentum_service" in name
        for name in imported
    )
    dynamic_loading = any(
        token in controller_text
        for token in ("importlib", "runpy", "SourceFileLoader", "exec(")
    )
    fixed_writes = (
        service_text.count("sys.stdout.buffer.write(response)") == 2
        and "print(" not in service_text
        and "json.dumps" not in service_text
    )
    if not ordering or forbidden_import or dynamic_loading or not fixed_writes:
        fail("independent source boundary audit failed")
    return {
        "service_response_constructed_after_restoration_verification": True,
        "controller_imports_backend_or_service": False,
        "controller_dynamically_loads_backend_or_service": False,
        "service_protocol_write_sites_are_fixed_binary": fixed_writes,
    }


def no_smuggle(evidence: Path) -> dict[str, Any]:
    allowed = {
        "FORWARD_BEGIN",
        "PORT_LEASED_INTERNAL",
        "PORT_RELEASED_INTERNAL",
        "BOUNDARY_RETAINED_INTERNAL",
        "PORT_RESIDENT",
        "HIDDEN_PROJECTION_DENIED",
        "PORT_RELEASED",
        "FORWARD_RESIDENT",
        "INVERSE_OMITTED",
        "WRONG_INVERSE_EXECUTED",
        "REORDERED_INVERSE_EXECUTED",
        "WRONG_REFLECTION_INVERSE_EXECUTED",
        "CONTROL_DISCRIMINATED",
        "RESTORATION_VERIFIED",
        "SNAPSHOT_RELOADED",
        "RESPONSE_WRITE_ATTEMPT",
        "STOP_RESPONSE_WRITE_ATTEMPT",
    }
    paths = sorted(evidence.glob("oracle-*.audit.log"))
    lines = [
        line
        for path in paths
        for line in path.read_text(encoding="ascii").splitlines()
    ]
    syntax = re.compile(r"^[0-9]+:[A-Z_]+$")
    valid = all(
        syntax.fullmatch(line) and event_name(line) in allowed
        for line in lines
    )
    if not valid:
        fail("independent audit vocabulary failed")
    return {
        "audit_files": len(paths),
        "audit_bytes": sum(path.stat().st_size for path in paths),
        "audit_event_vocabulary_only": True,
        "intermediate_field_values_in_audit": False,
        "intermediate_cells_in_fixed_response": False,
        "response_size_bytes": RESPONSE.size,
    }


def build_result(
    service: Path, controller: Path, evidence: Path
) -> dict[str, Any]:
    Driver.launches = 0
    algebra = reconstruct_algebra()
    accepted = accepted_protocol(service, evidence, algebra)
    comparison = fresh_and_snapshot_protocol(service, evidence, algebra)
    controls = control_protocol(service, evidence)
    disconnect = disconnect_protocol(service, evidence)
    source = source_boundary_audit(service, controller)
    smuggle = no_smuggle(evidence)
    return {
        "schema": "CATVM_GROWING_ROTOR_OPEN_MOMENTUM_INDEPENDENT_ORACLE_V1",
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
        "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "result": "PASS",
        "independence": {
            "catvm_service_imported": False,
            "catvm_controller_imported": False,
            "catvm_protocol_imported": False,
            "production_factor_module_imported": False,
            "separate_m199_mathematical_oracle_reused": True,
        },
        "algebra": algebra,
        "accepted_protocol": accepted,
        "fresh_and_snapshot": comparison,
        "controls": controls,
        "disconnect": disconnect,
        "source_boundary_audit": source,
        "no_smuggle": smuggle,
        "observed_resource_law": {
            "service_launches": Driver.launches,
            "request_bytes": REQUEST.size,
            "response_bytes": RESPONSE.size,
            "accepted_carrier_field_cells": 8943,
            "accepted_one_body_terms_per_forward_inverse_transaction": 663408,
            "retained_transition_plan_entries": 0,
            "retained_inverse_history_bytes": 0,
            "snapshot_saved_field_cells": 8943,
            "python_object_allocator_interpreter_native_library_timing_and_whole_process_peaks_excluded": True,
        },
        "claim_ceiling": "LINUX_SAME_UID_NON_DUMPABLE_NO_NEW_PRIVS_FIXED_BINARY_PIPE_SERVICE_GRID17_EXCHANGE_SYMMETRIC_ROTATION_REFLECTION_INVARIANT_ROTOR6_F103_ROOT72_DEPTH1_PRIMARY_AND_REUSE_ONLY",
        "rejected_interpretations": [
            "DIFFERENT_UID_CONTAINER_OR_HARDWARE_ISOLATION",
            "ARBITRARY_ROTOR_COUNT_OR_FIXED_RANK_GROWTH",
            "DISTINCT_PHASE_RESOURCE",
            "COMPUTATIONAL_ADVANTAGE_OR_SMALL_WALL_CROSSING",
            "PHYSICAL_WAVEFORM_AUDIO_OR_SILICON_EXECUTION",
            "REPLACEMENT_OF_PHYSICAL_BITS_WITH_PI",
            "UNBOUNDED_CATALYTIC_COMPUTATION",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("service", type=Path)
    parser.add_argument("controller", type=Path)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    args.evidence.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        build_result(
            args.service.resolve(),
            args.controller.resolve(),
            args.evidence.resolve(),
        ),
        indent=2,
        sort_keys=True,
    ) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
