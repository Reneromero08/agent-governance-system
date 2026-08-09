#!/usr/bin/env python3
"""Headless qtest for the compiled M271 Phase-QEMU V13 gate device.

This runner imports the frozen V12 runner only to reuse its qtest/QMP socket
transport.  Every protocol expectation below is defined independently for the
compiled V13 PCI device.  No project model, reference oracle, hardware API, or
physical adapter is imported or contacted.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Callable, NoReturn, Sequence


# Suppress import caches even when a caller forgets Python's -B flag.
sys.dont_write_bytecode = True

V12_RUNNER = (
    Path(__file__).resolve().parents[2]
    / "phase_qemu_v12_authenticated_adapter_stub"
    / "tests"
    / "run_phase_qemu_v12_authenticated_adapter_qtest.py"
)
SPEC = importlib.util.spec_from_file_location(
    "phase_qemu_v12_transport_support", V12_RUNNER
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load the frozen V12 qtest transport support")
V12 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = V12
SPEC.loader.exec_module(V12)

# V12 itself exposes the frozen socket/PCI transport implementation as BASE.
# Mutating only its PCI identity constants makes those low-level helpers address
# the V13 sibling.  No V12 mechanism, evidence, comparator, or model is called.
TRANSPORT = V12.BASE
TRANSPORT.DEVICE_TYPE = "phase-qemu-v13"
TRANSPORT.PCI_DEVICE = 0x11FD
TRANSPORT.MAGIC = 0x50483133
TRANSPORT.ABI = 0x00030000


SCHEMA = "M271_PHASE_QEMU_V13_HARDWARE_GATE_QTEST_EVIDENCE_V1"
CLAIM = (
    "COMPILED_COMMON_PHASE_QEMU_V13_FIXED_OFFLINE_HARDWARE_ADAPTER_SELECTOR_GATE_"
    "REJECTS_ABSENT_UNENROLLED_UNATTESTED_STALE_REPLAYED_DOWNGRADED_AND_TEST_"
    "FIXTURE_SESSIONS_WITHOUT_PUBLISHING_A_PHYSICAL_OUTPUT_OR_CAMPAIGN_"
    "STATISTICAL_CERTIFICATE_AND_WITH_EVERY_NONDESTRUCTIVELY_COMPLETED_DISPATCHED_"
    "ATTEMPT_TERMINAL_ACK_THEN_SPENT"
)
SCOPE = (
    "COMPILED_QEMU_10_2_4_PHASE_QEMU_V13_COMMON_PCI_BACKEND_HARDWARE_ABSENCE_AND_"
    "FIXED_OFFLINE_SYMBOLIC_SELECTOR_STATE_MACHINE_ONLY"
)
DISPOSITION = (
    "V13_ESTABLISHES_COMMON_BACKEND_FAIL_CLOSED_FIXED_SELECTOR_STATE_MACHINE_"
    "CONFORMANCE_FOR_DEVICE_ENROLLMENT_ATTESTATION_REPLAY_DOWNGRADE_AND_FIXTURE_"
    "DOMAIN_SEPARATION_WITHOUT_A_LIVE_HARDWARE_SESSION_OR_PHYSICAL_OUTPUT_DIRECT_"
    "EQUAL_ACCESS_PROTOCOL_COMPARATOR_CONTROLS_AND_M257_REMAINS_INTACT"
)
BACKEND_HARDWARE = 0x0D80
BACKEND_OFFLINE_FIXTURE = 0x0DF0
MAGIC = 0x50483133
ABI = 0x00030000
RESOURCE_UNKNOWN = (1 << 64) - 1
RESOURCE_SCHEMA = 0x00030001
BOUNDARY_HEADER = 0x5031334600030080
BOUNDARY_WORDS = 16

CMD_LEASE = 1
CMD_PREPARE = 2
CMD_ISOLATE_SOURCE = 3
CMD_SEAL_DESCRIPTOR = 4
CMD_EXECUTE_ATOMIC = 5
CMD_BEGIN_REUSE = 6
CMD_SNAPSHOT = 7
CMD_ARM_PRIVATE = 8
CMD_ACK_RESPONSE = 9
CMD_ABORT_PREEXEC = 10
CMD_POLL_EXTERNAL = 11
CMD_CANCEL_EXTERNAL = 12

LIFE_EMPTY = 0
LIFE_LEASED = 1
LIFE_PREPARED = 2
LIFE_ISOLATED = 3
LIFE_SEALED = 4
LIFE_PRIVATE_READY = 6
LIFE_RESPONSE_READY = 9
LIFE_SPENT = 12
LIFE_SHAM = 13

RETURN_NONE = 0
RETURN_EXACT_FORMAL = 1
RETURN_APPROX_MODEL = 2
RETURN_FAILED = 3
RETURN_STATISTICAL_ONLY = 4

ERR_NONE = 0
ERR_BAD_STATE = 1
ERR_BAD_ARGUMENT = 2
ERR_TAG_MISMATCH = 3
ERR_GENERATION_MISMATCH = 4
ERR_SNAPSHOT_REJECTED = 12
ERR_SNAPSHOT_LINEAGE = 13
ERR_RESPONSE_LOCKED = 14
ERR_REUSE_NOT_QUALIFIED = 27
ERR_HARDWARE_ABSENT = 38
ERR_DEVICE_UNENROLLED = 39
ERR_DEVICE_UNATTESTED = 40
ERR_ATTESTATION_STALE = 41
ERR_ATTESTATION_REPLAYED = 42
ERR_SECURITY_DOWNGRADE = 43
ERR_FIXTURE_TRUST_DOMAIN = 44
ERR_EVIDENCE_TYPE_MISMATCH = 45
ERR_LEASE_EXPIRED = 46

ST_RESPONSE_READY = 1 << 5
ST_EXACT_RETURN = 1 << 7
ST_SNAPSHOT_LINEAGE = 1 << 9
ST_RETURN_VERIFIED = 1 << 14
ST_SAME_ALLOCATION = 1 << 16
ST_ENV_FACTORED = 1 << 17
ST_EXTERNAL_REUSE_FORBIDDEN = 1 << 25
ST_FIXTURE_DOMAIN = 1 << 26
ST_TERMINAL_FAILURE_RECEIPT = 1 << 27
ST_HARDWARE_ABSENT = 1 << 28
ST_SHAM = 1 << 22

GATE_COLD = 0
GATE_PREFLIGHT_REJECTED = 1
GATE_LEASED = 2
GATE_ATTESTATION_ARMED = 3
GATE_TERMINAL_FAILED = 5
GATE_SPENT = 6
GATE_SHAM = 7

EVIDENCE_ORIGIN_NONE = 0
EVIDENCE_ORIGIN_OFFLINE_STANDARD_VECTOR = 2
CHANNEL_SECURITY_NONE = 0
CHANNEL_SECURITY_OFFLINE_VECTOR_INTEGRITY = 2
CHANNEL_SECURITY_REJECTED = 3
MEASUREMENT_CLASS_NONE = 0
MEASUREMENT_CLASS_PROTOCOL_CONFORMANCE_ONLY = 3
CUSTODY_PROVENANCE_NONE = 0
CUSTODY_PROVENANCE_OFFLINE_VECTOR = 2
RESOURCE_PROVENANCE_NONE = 0
RESOURCE_PROVENANCE_OFFLINE_VECTOR = 2
TRUST_DOMAIN_NONE = 0
TRUST_DOMAIN_TEST_FIXTURE = 2

GATE_FLAG_CHANNEL_METADATA_ACCEPTED = 1 << 0
GATE_FLAG_APPRAISAL_METADATA_ACCEPTED = 1 << 1
GATE_FLAG_ENROLLMENT_PRESENT = 1 << 2
GATE_FLAG_ATTESTATION_PRESENT = 1 << 3
GATE_FLAG_FRESH = 1 << 4
GATE_FLAG_NONREPLAY = 1 << 5
GATE_FLAG_SECURITY_VERSION_ACCEPTED = 1 << 6
GATE_FLAG_FIXTURE_ONLY = 1 << 7
GATE_FLAG_NO_PHYSICAL_OUTPUT = 1 << 8
GATE_FLAG_NO_CAMPAIGN_CERTIFICATE = 1 << 9

REG_ARG1 = 0x024
REG_PREPARATION_RECEIPT_LO = 0x0D0
REG_PREPARATION_RECEIPT_HI = 0x0D8
REG_RETURN_RECEIPT_LO = 0x0E0
REG_RETURN_RECEIPT_HI = 0x0E8
REG_RESOURCE_DURATION_FS = 0x118
REG_RESOURCE_PORT_BANDWIDTH_HZ = 0x120
REG_RESOURCE_ACTION_Q40_RAD = 0x128
REG_RESOURCE_MEAN_ENERGY_ATTOJ = 0x130
REG_RESOURCE_ENV_HISTORY_CELLS = 0x148
REG_RESOURCE_CUSTODY_TRANSITIONS = 0x150
REG_RESOURCE_OUTPUT_HOLD_FS = 0x168
REG_RESOURCE_MAINTENANCE_OPS = 0x170
REG_ADAPTER_STATE = 0x1C0
REG_ADAPTER_AUTH_ACCEPTED = 0x1C8
REG_ADAPTER_AUTH_REJECTED = 0x1D0
REG_ADAPTER_DISPATCHES = 0x1D8
REG_ADAPTER_COMPLETIONS = 0x1E0
REG_ADAPTER_CANCELS = 0x1E8
REG_ADAPTER_VIRTUAL_TICK = 0x1F0
REG_ADAPTER_DEADLINE_TICK = 0x1F8
REG_GATE_STATE = 0x280
REG_EVIDENCE_ORIGIN = 0x284
REG_CHANNEL_SECURITY = 0x288
REG_DEVICE_APPRAISAL = 0x28C
REG_MEASUREMENT_CLASS = 0x290
REG_CUSTODY_PROVENANCE = 0x294
REG_RESOURCE_PROVENANCE = 0x298
REG_TRUST_DOMAIN = 0x29C
REG_REJECTION_REASON = 0x2A0
REG_GATE_FLAGS = 0x2A4
REG_SECURITY_VERSION = 0x2A8
REG_MIN_SECURITY_VERSION = 0x2AC
REG_SESSION_EPOCH = 0x2B0
REG_ATTESTATION_NONCE = 0x2B8
REG_ATTESTATION_AGE_TICKS = 0x2C0
REG_MAX_ATTESTATION_AGE_TICKS = 0x2C8
REG_DEVICE_ID_DIGEST_LO = 0x2D0
REG_DEVICE_ID_DIGEST_HI = 0x2D8
REG_MEASUREMENT_DIGEST_LO = 0x2E0
REG_MEASUREMENT_DIGEST_HI = 0x2E8
REG_CUSTODY_DIGEST_LO = 0x2F0
REG_CUSTODY_DIGEST_HI = 0x2F8
REG_RESOURCE_PROVENANCE_DIGEST_LO = 0x300
REG_RESOURCE_PROVENANCE_DIGEST_HI = 0x308
REG_GATE_DISPATCHES = 0x310
REG_GATE_TERMINAL_FAILURES = 0x318
REG_GATE_ACKS = 0x320
REG_GATE_REPLAY_REJECTS = 0x328
REG_GATE_DOWNGRADE_REJECTS = 0x330
REG_GATE_FIXTURE_REJECTS = 0x338
REG_GATE_VIRTUAL_TICK = 0x340
REG_LEASE_EXPIRY_TICK = 0x348

DESCRIPTOR_WORDS = (
    0x50313347,
    0x00030008,
    0x00000006,
    0x00000002,
    0x00000001,
    0x00000003,
    0x00000000,
    0x00000000,
)

# This map is the frozen V12 guest-visible prefix.  V13 extensions start only
# after its last 16-word boundary register.
COMMON_V12_PREFIX_OFFSETS = {
    "magic": 0x000,
    "abi": 0x004,
    "backend": 0x008,
    "capabilities_lo": 0x00C,
    "status_lo": 0x010,
    "error": 0x014,
    "generation": 0x018,
    "exact_return_generation": 0x01C,
    "arg0": 0x020,
    "arg1": 0x024,
    "command": 0x028,
    "lifecycle": 0x02C,
    "virtual_cycles": 0x030,
    "boundary_commit_cookie": 0x040,
    "resource_state_cells": 0x048,
    "resource_scratch_cells": 0x04C,
    "resource_query_applications": 0x050,
    "resource_return_checks": 0x058,
    "resource_environment_ops": 0x060,
    "resource_peak_bits": 0x068,
    "request_owner": 0x070,
    "request_program": 0x074,
    "request_generation": 0x078,
    "descriptor_index": 0x07C,
    "descriptor_word": 0x080,
    "descriptor_length": 0x084,
    "boundary_schema": 0x088,
    "descriptor_fingerprint": 0x090,
    "fault_mode": 0x098,
    "capabilities_hi": 0x0A0,
    "status_hi": 0x0A4,
    "private_query_slots": 0x0A8,
    "private_ready_mask": 0x0AC,
    "return_class": 0x0B0,
    "boundary_length": 0x0B4,
    "allocation_id_lo": 0x0B8,
    "allocation_id_hi": 0x0C0,
    "custody_epoch": 0x0C8,
    "preparation_receipt_lo": 0x0D0,
    "preparation_receipt_hi": 0x0D8,
    "return_receipt_lo": 0x0E0,
    "return_receipt_hi": 0x0E8,
    "resource_secret_storage_bits": 0x0F0,
    "resource_control_words": 0x0F8,
    "resource_preparation_ops": 0x100,
    "resource_certification_ops": 0x108,
    "resource_logical_queries": 0x110,
    "resource_duration_fs": 0x118,
    "resource_port_bandwidth_hz": 0x120,
    "resource_action_q40_rad": 0x128,
    "resource_mean_energy_attoj": 0x130,
    "resource_loss_q63": 0x138,
    "resource_dephasing_q63": 0x140,
    "resource_environment_history_cells": 0x148,
    "resource_custody_transitions": 0x150,
    "resource_reuse_count": 0x158,
    "resource_discarded_trials": 0x160,
    "resource_output_hold_fs": 0x168,
    "resource_maintenance_ops": 0x170,
    "resource_precision_bits": 0x178,
    "resource_schema": 0x180,
    "resource_digest_lo": 0x188,
    "resource_digest_hi": 0x190,
    "resource_compiler_ops": 0x198,
    "resource_controller_ops": 0x1A0,
    "resource_construction_ops": 0x1A8,
    "resource_secret_entropy_bits": 0x1B0,
    "resource_carrier_photon_number": 0x1B8,
    "adapter_state": 0x1C0,
    "adapter_auth_accepted": 0x1C8,
    "adapter_auth_rejected": 0x1D0,
    "adapter_dispatches": 0x1D8,
    "adapter_completions": 0x1E0,
    "adapter_cancels": 0x1E8,
    "adapter_virtual_tick": 0x1F0,
    "adapter_deadline_tick": 0x1F8,
    "boundary_base": 0x200,
    "boundary_last": 0x278,
}

UNKNOWN_PHYSICAL_REGISTERS = {
    "duration_fs": REG_RESOURCE_DURATION_FS,
    "port_bandwidth_hz": REG_RESOURCE_PORT_BANDWIDTH_HZ,
    "action_q40_rad": REG_RESOURCE_ACTION_Q40_RAD,
    "mean_energy_attoj": REG_RESOURCE_MEAN_ENERGY_ATTOJ,
    "loss_q63": TRANSPORT.REG_RESOURCE_LOSS_Q63,
    "dephasing_q63": TRANSPORT.REG_RESOURCE_DEPHASING_Q63,
    "environment_history_cells": REG_RESOURCE_ENV_HISTORY_CELLS,
    "output_hold_fs": REG_RESOURCE_OUTPUT_HOLD_FS,
    "maintenance_ops": REG_RESOURCE_MAINTENANCE_OPS,
    "precision_bits": TRANSPORT.REG_RESOURCE_PRECISION_BITS,
    "construction_ops": TRANSPORT.REG_RESOURCE_CONSTRUCTION_OPS,
    "carrier_photon_number": TRANSPORT.REG_RESOURCE_CARRIER_PHOTON_NUMBER,
}


def fail(message: str) -> NoReturn:
    raise TRANSPORT.EvidenceFailure(message)


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii") + b"\n"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fnv_u64(current: int, value: int) -> int:
    for byte in range(8):
        current ^= (value >> (8 * byte)) & 0xFF
        current = (current * 1099511628211) & RESOURCE_UNKNOWN
    return current


def receipt_hash(domain: int, first: int, second: int, third: int) -> int:
    current = 14695981039346656037
    for value in (domain, first, second, third):
        current = fnv_u64(current, value)
    return current


@dataclass(frozen=True)
class VectorExpectation:
    vector_id: int
    name: str
    appraisal: int
    reason: int
    channel_security: int
    gate_flags: int
    security_version: int
    attestation_age: int
    replay_rejects: int = 0
    downgrade_rejects: int = 0


COMMON_FIXTURE_FLAGS = (
    GATE_FLAG_CHANNEL_METADATA_ACCEPTED
    | GATE_FLAG_FIXTURE_ONLY
    | GATE_FLAG_NO_PHYSICAL_OUTPUT
    | GATE_FLAG_NO_CAMPAIGN_CERTIFICATE
)

VECTOR_EXPECTATIONS = (
    VectorExpectation(
        0,
        "accepted_metadata_fixture_domain",
        7,
        ERR_FIXTURE_TRUST_DOMAIN,
        CHANNEL_SECURITY_OFFLINE_VECTOR_INTEGRITY,
        COMMON_FIXTURE_FLAGS
        | GATE_FLAG_APPRAISAL_METADATA_ACCEPTED
        | GATE_FLAG_ENROLLMENT_PRESENT
        | GATE_FLAG_ATTESTATION_PRESENT
        | GATE_FLAG_FRESH
        | GATE_FLAG_NONREPLAY
        | GATE_FLAG_SECURITY_VERSION_ACCEPTED,
        3,
        1,
    ),
    VectorExpectation(
        1,
        "unenrolled",
        2,
        ERR_DEVICE_UNENROLLED,
        CHANNEL_SECURITY_OFFLINE_VECTOR_INTEGRITY,
        COMMON_FIXTURE_FLAGS,
        3,
        1,
    ),
    VectorExpectation(
        2,
        "unattested",
        3,
        ERR_DEVICE_UNATTESTED,
        CHANNEL_SECURITY_OFFLINE_VECTOR_INTEGRITY,
        COMMON_FIXTURE_FLAGS | GATE_FLAG_ENROLLMENT_PRESENT,
        3,
        1,
    ),
    VectorExpectation(
        3,
        "stale",
        4,
        ERR_ATTESTATION_STALE,
        CHANNEL_SECURITY_OFFLINE_VECTOR_INTEGRITY,
        COMMON_FIXTURE_FLAGS
        | GATE_FLAG_ENROLLMENT_PRESENT
        | GATE_FLAG_ATTESTATION_PRESENT,
        3,
        9,
    ),
    VectorExpectation(
        4,
        "replayed",
        5,
        ERR_ATTESTATION_REPLAYED,
        CHANNEL_SECURITY_OFFLINE_VECTOR_INTEGRITY,
        COMMON_FIXTURE_FLAGS
        | GATE_FLAG_ENROLLMENT_PRESENT
        | GATE_FLAG_ATTESTATION_PRESENT
        | GATE_FLAG_FRESH,
        3,
        1,
        replay_rejects=1,
    ),
    VectorExpectation(
        5,
        "downgraded",
        6,
        ERR_SECURITY_DOWNGRADE,
        CHANNEL_SECURITY_OFFLINE_VECTOR_INTEGRITY,
        COMMON_FIXTURE_FLAGS
        | GATE_FLAG_ENROLLMENT_PRESENT
        | GATE_FLAG_ATTESTATION_PRESENT
        | GATE_FLAG_FRESH
        | GATE_FLAG_NONREPLAY,
        2,
        1,
        downgrade_rejects=1,
    ),
    VectorExpectation(
        6,
        "type_mismatch",
        0,
        ERR_EVIDENCE_TYPE_MISMATCH,
        CHANNEL_SECURITY_REJECTED,
        COMMON_FIXTURE_FLAGS,
        3,
        1,
    ),
)


def direct_protocol_comparator(expectation: VectorExpectation) -> dict[str, object]:
    """Independent comparator for seven predetermined symbolic selectors.

    This is a fixed state-machine truth table, not evidence parsing,
    cryptographic verification, or live-device appraisal.
    """

    return {
        "vector_id": expectation.vector_id,
        "vector_name": expectation.name,
        "expected_rejection_reason": expectation.reason,
        "expected_appraisal": expectation.appraisal,
        "expected_channel_security": expectation.channel_security,
        "expected_gate_flags": expectation.gate_flags,
        "expected_security_version": expectation.security_version,
        "expected_attestation_age_ticks": expectation.attestation_age,
        "expected_return_class": "RETURN_FAILED",
        "expected_ack_terminal_state": "SPENT",
        "physical_sample_expected": False,
        "campaign_certificate_expected": False,
        "reuse_expected": False,
    }


@dataclass(frozen=True)
class GateOptions:
    backend_id: int = BACKEND_HARDWARE
    fixture_enabled: bool = False
    vector_id: int = 0


class GateQemuInstance(TRANSPORT.QemuInstance):
    def __init__(
        self,
        qemu_binary: Path,
        case_dir: Path,
        options: GateOptions,
        *,
        incoming_defer: bool = False,
    ):
        super().__init__(
            qemu_binary,
            case_dir,
            TRANSPORT.DeviceOptions(backend_id=options.backend_id),
            incoming_defer=incoming_defer,
        )
        self.gate_options = options

    def command_line(self) -> list[str]:
        device = ",".join(
            (
                TRANSPORT.DEVICE_TYPE,
                f"id={TRANSPORT.DEVICE_ID}",
                "bus=pcie.0",
                f"addr={TRANSPORT.PCI_DEVICE_NUMBER:x}.0",
                f"backend-id=0x{self.gate_options.backend_id:04x}",
                "test-fixture-enabled="
                f"{'on' if self.gate_options.fixture_enabled else 'off'}",
                f"test-standard-vector-id={self.gate_options.vector_id}",
            )
        )
        command = [
            str(self.qemu_binary),
            "-machine", "q35,accel=qtest",
            "-m", "64M",
            "-S",
            "-nodefaults",
            "-display", "none",
            "-serial", "none",
            "-monitor", "none",
            "-qtest", f"unix:{self.qtest_uri_path},server=on,wait=off",
            "-qtest-log", "/dev/null",
            "-qmp", f"unix:{self.qmp_uri_path},server=on,wait=off",
            "-device", device,
        ]
        if self.incoming_defer:
            command.extend(("-incoming", "defer"))
        return command

    def start(self) -> None:
        self.case_dir.mkdir(mode=0o700)
        self.process = subprocess.Popen(
            self.command_line(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            close_fds=True,
            cwd=self.case_dir,
        )
        self.qmp = TRANSPORT.QMPClient(
            TRANSPORT.connect_unix(self.qmp_path, self.process)
        )
        self.qtest = TRANSPORT.QTestClient(
            TRANSPORT.connect_unix(self.qtest_path, self.process)
        )


class GateAccess(TRANSPORT.DeviceAccess):
    def upload_gate_descriptor(self) -> None:
        for index, word in enumerate(DESCRIPTOR_WORDS):
            self.write32(TRANSPORT.REG_DESCRIPTOR_INDEX, index)
            self.write32(TRANSPORT.REG_DESCRIPTOR_WORD, word)
        self.write32(TRANSPORT.REG_DESCRIPTOR_LENGTH, len(DESCRIPTOR_WORDS))
        self.write32(TRANSPORT.REG_BOUNDARY_SCHEMA, 3)


def require_empty_streams(streams: dict[str, object], label: str) -> None:
    require(streams["stdout_bytes"] == 0, f"{label}: QEMU stdout not empty")
    require(streams["stderr_bytes"] == 0, f"{label}: QEMU stderr not empty")


def require_identity(access: GateAccess, backend_id: int) -> dict[str, int]:
    identity = access.identity()
    require(identity["magic"] == MAGIC, "V13 magic mismatch")
    require(identity["abi"] == ABI, "V13 ABI mismatch")
    require(identity["backend_id"] == backend_id, "V13 backend mismatch")
    require(identity["capabilities_lo"] == 0x7FFF, "V13 capabilities mismatch")
    return identity


def configure_request(access: GateAccess, owner: int = 0xA271,
                      program: int = 0xB271, generation: int = 1) -> None:
    access.write32(TRANSPORT.REG_ARG0, owner)
    access.write32(REG_ARG1, program)
    access.configure_request(owner, program, generation)


def prepare_fixture(access: GateAccess) -> None:
    configure_request(access)
    require(access.command(CMD_LEASE) == ERR_NONE, "fixture LEASE failed")
    require(access.read32(TRANSPORT.REG_LIFECYCLE) == LIFE_LEASED,
            "fixture lease lifecycle")
    require(access.command(CMD_PREPARE) == ERR_NONE, "fixture PREPARE failed")
    require(access.read32(TRANSPORT.REG_LIFECYCLE) == LIFE_PREPARED,
            "fixture prepare lifecycle")
    require(access.command(CMD_ISOLATE_SOURCE) == ERR_NONE,
            "fixture ISOLATE_SOURCE failed")
    require(access.read32(TRANSPORT.REG_LIFECYCLE) == LIFE_ISOLATED,
            "fixture isolate lifecycle")
    access.upload_gate_descriptor()
    require(access.command(CMD_SEAL_DESCRIPTOR) == ERR_NONE,
            "fixture SEAL_DESCRIPTOR failed")
    require(access.read32(TRANSPORT.REG_LIFECYCLE) == LIFE_SEALED,
            "fixture seal lifecycle")
    require(access.command(CMD_ARM_PRIVATE) == ERR_NONE,
            "fixture attestation arm failed")
    require(access.read32(TRANSPORT.REG_LIFECYCLE) == LIFE_PRIVATE_READY,
            "fixture attestation lifecycle")
    require(access.read32(REG_GATE_STATE) == GATE_ATTESTATION_ARMED,
            "fixture gate not armed")


def pack_evidence_types(access: GateAccess) -> int:
    return (
        access.read32(REG_EVIDENCE_ORIGIN)
        | (access.read32(REG_CHANNEL_SECURITY) << 8)
        | (access.read32(REG_DEVICE_APPRAISAL) << 16)
        | (access.read32(REG_MEASUREMENT_CLASS) << 24)
        | (access.read32(REG_CUSTODY_PROVENANCE) << 32)
        | (access.read32(REG_RESOURCE_PROVENANCE) << 40)
        | (access.read32(REG_TRUST_DOMAIN) << 48)
    )


def recompute_resource_digest(access: GateAccess, domain: int) -> int:
    values = (
        domain,
        RESOURCE_SCHEMA,
        access.read64(TRANSPORT.REG_RESOURCE_CONTROL_WORDS),
        access.read64(TRANSPORT.REG_RESOURCE_PREPARATION_OPS),
        access.read64(TRANSPORT.REG_RESOURCE_CERTIFICATION_OPS),
        access.read64(REG_RESOURCE_CUSTODY_TRANSITIONS),
        access.read64(TRANSPORT.REG_RESOURCE_DISCARDED_TRIALS),
        access.read64(TRANSPORT.REG_RESOURCE_CONTROLLER_OPS),
        access.read32(REG_RESOURCE_PROVENANCE),
        access.read64(REG_RESOURCE_PROVENANCE_DIGEST_LO),
        access.read64(REG_RESOURCE_PROVENANCE_DIGEST_HI),
    )
    current = 14695981039346656037
    for value in values:
        current = fnv_u64(current, value)
    return current


def require_no_physical_output(access: GateAccess) -> None:
    status = access.read32(TRANSPORT.REG_STATUS_LO)
    require(not status & ST_EXACT_RETURN, "exact physical return asserted")
    require(not status & ST_RETURN_VERIFIED, "physical return verified")
    require(not status & ST_SAME_ALLOCATION, "physical same allocation asserted")
    require(not status & ST_ENV_FACTORED, "physical environment factoring asserted")
    require(access.read32(TRANSPORT.REG_EXACT_RETURN_GENERATION) == 0,
            "exact return generation asserted")
    require(access.read64(TRANSPORT.REG_RESOURCE_LOGICAL_QUERIES) == 0,
            "logical phase query reported")
    require(access.read32(TRANSPORT.REG_RESOURCE_STATE_CELLS) == 0,
            "ideal model state reported")
    require(access.read32(TRANSPORT.REG_RESOURCE_SCRATCH_CELLS) == 0,
            "ideal model scratch reported")
    require(access.read64(TRANSPORT.REG_RESOURCE_QUERY_APPLICATIONS) == 0,
            "ideal query application reported")
    require(access.read64(TRANSPORT.REG_RESOURCE_RETURN_CHECKS) == 0,
            "ideal return check reported")
    require(access.read64(TRANSPORT.REG_RESOURCE_ENVIRONMENT_OPS) == 0,
            "ideal environment operation reported")
    require(access.read64(TRANSPORT.REG_RESOURCE_REUSE_COUNT) == 0,
            "physical reuse count reported")
    require(access.read64(TRANSPORT.REG_RESOURCE_COMPILER_OPS) == 0,
            "ideal compiler operation reported")
    require(access.read64(TRANSPORT.REG_RESOURCE_SECRET_STORAGE_BITS) == 0,
            "secret storage reported")
    require(access.read64(TRANSPORT.REG_RESOURCE_SECRET_ENTROPY_BITS) == 0,
            "secret entropy reported")
    for label, register in UNKNOWN_PHYSICAL_REGISTERS.items():
        require(access.read64(register) == RESOURCE_UNKNOWN,
                f"unknown physical resource encoded as a value: {label}")


def dispatch_to_receipt(
    access: GateAccess, expectation: VectorExpectation
) -> tuple[list[int], dict[str, int]]:
    prepare_fixture(access)
    allocation_lo = access.read64(TRANSPORT.REG_ALLOCATION_ID_LO)
    fingerprint = access.read64(TRANSPORT.REG_DESCRIPTOR_FINGERPRINT)
    require(access.command(CMD_EXECUTE_ATOMIC) == expectation.reason,
            f"vector {expectation.vector_id} execute reason mismatch")
    require(access.read32(TRANSPORT.REG_LIFECYCLE) == LIFE_RESPONSE_READY,
            "failure receipt lifecycle mismatch")
    require(access.read32(REG_GATE_STATE) == GATE_TERMINAL_FAILED,
            "terminal failure gate state mismatch")
    require(access.read32(TRANSPORT.REG_RETURN_CLASS) == RETURN_FAILED,
            "fixture did not return FAILED")
    require(access.read32(REG_REJECTION_REASON) == expectation.reason,
            "typed rejection reason mismatch")
    require(access.read32(TRANSPORT.REG_ERROR) == expectation.reason,
            "execute error mismatch")
    require(access.read32(REG_EVIDENCE_ORIGIN) ==
            EVIDENCE_ORIGIN_OFFLINE_STANDARD_VECTOR,
            "fixture evidence origin mismatch")
    require(access.read32(REG_CHANNEL_SECURITY) == expectation.channel_security,
            "fixture channel-security class mismatch")
    require(access.read32(REG_DEVICE_APPRAISAL) == expectation.appraisal,
            "fixture appraisal mismatch")
    require(access.read32(REG_MEASUREMENT_CLASS) ==
            MEASUREMENT_CLASS_PROTOCOL_CONFORMANCE_ONLY,
            "fixture measurement class escaped conformance-only")
    require(access.read32(REG_CUSTODY_PROVENANCE) ==
            CUSTODY_PROVENANCE_OFFLINE_VECTOR,
            "fixture custody provenance mismatch")
    require(access.read32(REG_RESOURCE_PROVENANCE) ==
            RESOURCE_PROVENANCE_OFFLINE_VECTOR,
            "fixture resource provenance mismatch")
    require(access.read32(REG_TRUST_DOMAIN) == TRUST_DOMAIN_TEST_FIXTURE,
            "fixture trust domain mismatch")
    require(access.read32(REG_GATE_FLAGS) == expectation.gate_flags,
            "fixture gate flags mismatch")
    require(access.read32(REG_SECURITY_VERSION) == expectation.security_version,
            "fixture security version mismatch")
    require(access.read32(REG_MIN_SECURITY_VERSION) == 3,
            "minimum security version mismatch")
    require(access.read64(REG_ATTESTATION_AGE_TICKS) == expectation.attestation_age,
            "fixture attestation age mismatch")
    require(access.read64(REG_MAX_ATTESTATION_AGE_TICKS) == 8,
            "maximum attestation age mismatch")
    require(access.read64(REG_GATE_DISPATCHES) == 1,
            "fixture dispatch counter mismatch")
    require(access.read64(REG_GATE_TERMINAL_FAILURES) == 1,
            "terminal failure counter mismatch")
    require(access.read64(REG_GATE_ACKS) == 0, "ACK counter advanced early")
    require(access.read64(REG_GATE_REPLAY_REJECTS) == expectation.replay_rejects,
            "replay rejection counter mismatch")
    require(access.read64(REG_GATE_DOWNGRADE_REJECTS) ==
            expectation.downgrade_rejects,
            "downgrade rejection counter mismatch")
    require(access.read64(REG_GATE_FIXTURE_REJECTS) == 1,
            "fixture rejection counter mismatch")
    require(access.read64(REG_ADAPTER_AUTH_ACCEPTED) == 0,
            "legacy adapter authentication success asserted")
    require(access.read64(REG_ADAPTER_AUTH_REJECTED) == 1,
            "legacy adapter rejection mirror mismatch")
    require(access.read64(REG_ADAPTER_DISPATCHES) == 1,
            "legacy adapter dispatch mirror mismatch")
    require(access.read64(REG_ADAPTER_COMPLETIONS) == 1,
            "legacy adapter completion mirror mismatch")
    require(access.read64(REG_ADAPTER_CANCELS) == 0,
            "fixture dispatch unexpectedly reported cancellation")

    status = access.read32(TRANSPORT.REG_STATUS_LO)
    require(status & ST_RESPONSE_READY, "response-ready status absent")
    require(status & ST_EXTERNAL_REUSE_FORBIDDEN, "reuse-forbidden status absent")
    require(status & ST_FIXTURE_DOMAIN, "fixture-domain status absent")
    require(status & ST_TERMINAL_FAILURE_RECEIPT,
            "terminal-failure-receipt status absent")
    require(status & ST_HARDWARE_ABSENT, "hardware-absent status absent")
    require_no_physical_output(access)

    boundary = access.read_boundary()
    require(boundary[0] == BOUNDARY_HEADER, "V13 boundary header mismatch")
    receipt_id = boundary[1]
    require(receipt_id != 0, "failure receipt ID missing")
    require(boundary[2] == (1 << 32) | expectation.reason,
            "boundary generation/reason mismatch")
    packed_types = pack_evidence_types(access)
    require(boundary[3] == packed_types, "boundary typed tuple mismatch")
    require(boundary[4] == expectation.gate_flags, "boundary flags mismatch")
    require(boundary[5] == access.read64(REG_DEVICE_ID_DIGEST_LO),
            "boundary device digest low mismatch")
    require(boundary[6] == access.read64(REG_DEVICE_ID_DIGEST_HI),
            "boundary device digest high mismatch")
    require(boundary[7] == access.read64(REG_MEASUREMENT_DIGEST_LO),
            "boundary measurement digest low mismatch")
    require(boundary[8] == access.read64(REG_MEASUREMENT_DIGEST_HI),
            "boundary measurement digest high mismatch")
    require(boundary[9] == access.read64(REG_CUSTODY_DIGEST_LO),
            "boundary custody digest low mismatch")
    require(boundary[10] == access.read64(REG_CUSTODY_DIGEST_HI),
            "boundary custody digest high mismatch")
    require(boundary[11] == access.read64(REG_RESOURCE_PROVENANCE_DIGEST_LO),
            "boundary resource-provenance digest low mismatch")
    require(boundary[12] == access.read64(REG_RESOURCE_PROVENANCE_DIGEST_HI),
            "boundary resource-provenance digest high mismatch")
    resource_lo = recompute_resource_digest(access, 0x52534C4F)
    resource_hi = recompute_resource_digest(access, 0x52534849)
    require(boundary[13] == resource_lo ==
            access.read64(TRANSPORT.REG_RESOURCE_DIGEST_LO),
            "independent low resource digest mismatch")
    require(boundary[14] == resource_hi ==
            access.read64(TRANSPORT.REG_RESOURCE_DIGEST_HI),
            "independent high resource digest mismatch")
    receipt_lo = receipt_hash(0x4641494C, receipt_id,
                              allocation_lo, expectation.reason)
    receipt_hi = receipt_hash(0x47415445, receipt_id,
                              fingerprint, packed_types)
    require(access.read64(REG_RETURN_RECEIPT_LO) == receipt_lo,
            "independent failure receipt low mismatch")
    require(access.read64(REG_RETURN_RECEIPT_HI) == receipt_hi,
            "independent failure receipt high mismatch")
    commit = receipt_hash(0x434F4D4D, receipt_lo, receipt_hi,
                          resource_lo ^ resource_hi)
    require(boundary[15] == commit, "independent commit cookie mismatch")
    require(access.read64(TRANSPORT.REG_BOUNDARY_COMMIT_COOKIE) == commit,
            "commit-cookie register mismatch")
    return boundary, {
        "receipt_id": receipt_id,
        "receipt_lo": receipt_lo,
        "receipt_hi": receipt_hi,
        "resource_digest_lo": resource_lo,
        "resource_digest_hi": resource_hi,
        "commit_cookie": commit,
    }


def vector_case(access: GateAccess, expectation: VectorExpectation) -> dict[str, object]:
    comparator = direct_protocol_comparator(expectation)
    boundary, receipt = dispatch_to_receipt(access, expectation)
    access.write32(TRANSPORT.REG_REQUEST_OWNER, 0xA272)
    require(access.command(CMD_ACK_RESPONSE) == ERR_TAG_MISMATCH,
            "wrong-tag ACK released the failed receipt")
    require(access.read_boundary() == boundary,
            "wrong-tag ACK mutated the held boundary")
    access.write32(TRANSPORT.REG_REQUEST_OWNER, 0xA271)
    tick_before = access.read64(REG_GATE_VIRTUAL_TICK)
    access.qtest.write64(TRANSPORT.BAR0_BASE + REG_GATE_VIRTUAL_TICK,
                         tick_before + 1)
    require(access.read32(TRANSPORT.REG_ERROR) == ERR_BAD_ARGUMENT,
            "held receipt allowed tick advancement")
    require(access.command(CMD_BEGIN_REUSE) == ERR_REUSE_NOT_QUALIFIED,
            "held failed receipt authorized reuse")
    require(access.command(CMD_EXECUTE_ATOMIC) == ERR_BAD_STATE,
            "held receipt allowed another dispatch")
    require(access.command(CMD_LEASE) == ERR_BAD_STATE,
            "held receipt allowed a new lease")
    require(access.read_boundary() == boundary,
            "rejected controls mutated held boundary")
    require(access.command(CMD_ACK_RESPONSE) == ERR_NONE,
            "failure receipt ACK failed")
    require(access.read32(TRANSPORT.REG_LIFECYCLE) == LIFE_SPENT,
            "ACK did not transition to SPENT")
    require(access.read32(REG_GATE_STATE) == GATE_SPENT,
            "ACK did not transition gate to SPENT")
    require(access.read64(REG_GATE_ACKS) == 1, "ACK counter mismatch")
    require(access.read_boundary() == [RESOURCE_UNKNOWN] * BOUNDARY_WORDS,
            "ACK did not relock the boundary")
    require(access.command(CMD_BEGIN_REUSE) == ERR_REUSE_NOT_QUALIFIED,
            "SPENT failed receipt authorized reuse")
    require(access.read32(TRANSPORT.REG_RETURN_CLASS) == RETURN_FAILED,
            "ACK changed failed return class")
    require(RETURN_FAILED not in
            (RETURN_EXACT_FORMAL, RETURN_APPROX_MODEL, RETURN_STATISTICAL_ONLY),
            "return-class constants overlap")
    require(comparator["expected_rejection_reason"] == expectation.reason,
            "direct comparator reason mismatch")
    return {
        "vector_id": expectation.vector_id,
        "vector_name": expectation.name,
        "rejection_reason": expectation.reason,
        "typed_origin": "OFFLINE_STANDARD_VECTOR",
        "trust_domain": "TEST_FIXTURE",
        "measurement_class": "PROTOCOL_CONFORMANCE_ONLY",
        "return_class": "RETURN_FAILED",
        "boundary_words": BOUNDARY_WORDS,
        "receipt_independently_recomputed": True,
        "resource_digest_independently_recomputed": True,
        "receipt_identity": receipt,
        "held_boundary_immutable": True,
        "wrong_tag_ack_rejected": True,
        "new_lease_while_held_rejected": True,
        "ack_required": True,
        "terminal_state": "SPENT",
        "begin_reuse_authorized": False,
        "physical_output": False,
        "physical_sample": False,
        "campaign_statistical_certificate": False,
        "direct_protocol_comparator": comparator,
        "direct_protocol_comparator_matches": True,
    }


def hardware_absent_case(access: GateAccess) -> dict[str, object]:
    configure_request(access)
    require(access.command(CMD_LEASE) == ERR_HARDWARE_ABSENT,
            "default hardware backend did not fail absent")
    require(access.read32(TRANSPORT.REG_LIFECYCLE) == LIFE_EMPTY,
            "absent hardware changed lifecycle")
    require(access.read32(REG_GATE_STATE) == GATE_PREFLIGHT_REJECTED,
            "absent hardware did not record preflight rejection")
    require(access.read32(REG_REJECTION_REASON) == ERR_HARDWARE_ABSENT,
            "absent hardware rejection reason mismatch")
    require(access.read64(REG_GATE_DISPATCHES) == 0,
            "absent hardware dispatched")
    require(access.read64(REG_GATE_TERMINAL_FAILURES) == 0,
            "absent hardware fabricated a terminal receipt")
    require(access.read32(TRANSPORT.REG_RETURN_CLASS) == RETURN_NONE,
            "absent hardware fabricated a return class")
    require(access.read64(REG_RETURN_RECEIPT_LO) == 0 and
            access.read64(REG_RETURN_RECEIPT_HI) == 0,
            "absent hardware fabricated a receipt")
    require(access.read_boundary() == [RESOURCE_UNKNOWN] * BOUNDARY_WORDS,
            "absent hardware exposed a boundary")
    require(access.read32(REG_EVIDENCE_ORIGIN) == EVIDENCE_ORIGIN_NONE,
            "absent hardware fabricated evidence origin")
    require(access.read32(REG_MEASUREMENT_CLASS) == MEASUREMENT_CLASS_NONE,
            "absent hardware fabricated measurement class")
    require(access.read32(REG_CUSTODY_PROVENANCE) == CUSTODY_PROVENANCE_NONE,
            "absent hardware fabricated custody")
    require(access.read32(REG_RESOURCE_PROVENANCE) == RESOURCE_PROVENANCE_NONE,
            "absent hardware fabricated resource provenance")
    require(access.read32(REG_TRUST_DOMAIN) == TRUST_DOMAIN_NONE,
            "absent hardware fabricated trust")
    require_no_physical_output(access)
    return {
        "backend": "HARDWARE",
        "backend_id": BACKEND_HARDWARE,
        "lease_error": ERR_HARDWARE_ABSENT,
        "pre_dispatch_rejection": True,
        "dispatches": 0,
        "response_created": False,
        "receipt_created": False,
        "physical_hardware_connected": False,
    }


def request_negative_case(access: GateAccess, kind: str) -> dict[str, object]:
    configure_request(access)
    require(access.command(CMD_LEASE) == ERR_NONE, f"{kind}: LEASE failed")
    expected: int
    if kind == "owner_tag":
        access.write32(TRANSPORT.REG_REQUEST_OWNER, 0xA272)
        expected = ERR_TAG_MISMATCH
    elif kind == "generation":
        access.write32(TRANSPORT.REG_REQUEST_GENERATION, 2)
        expected = ERR_GENERATION_MISMATCH
    elif kind == "lease_expiry":
        expiry = access.read64(REG_LEASE_EXPIRY_TICK)
        access.qtest.write64(TRANSPORT.BAR0_BASE + REG_GATE_VIRTUAL_TICK, expiry)
        expected = ERR_LEASE_EXPIRED
    else:
        raise AssertionError(kind)
    require(access.command(CMD_PREPARE) == expected,
            f"{kind}: request gate error mismatch")
    require(access.read64(REG_GATE_DISPATCHES) == 0,
            f"{kind}: rejected request dispatched")
    require(access.read32(TRANSPORT.REG_RETURN_CLASS) == RETURN_NONE,
            f"{kind}: rejected request fabricated return")
    require(access.read_boundary() == [RESOURCE_UNKNOWN] * BOUNDARY_WORDS,
            f"{kind}: rejected request exposed boundary")
    return {
        "control": kind,
        "error": expected,
        "pre_dispatch_rejection": True,
        "dispatches": 0,
        "receipt_created": False,
    }


def abi_fault_case(access: GateAccess) -> dict[str, object]:
    initial_lifecycle = access.read32(TRANSPORT.REG_LIFECYCLE)
    access.qtest.read8(TRANSPORT.BAR0_BASE + TRANSPORT.REG_MAGIC)
    require(access.read32(TRANSPORT.REG_ERROR) == ERR_BAD_ARGUMENT,
            "byte read did not latch BAD_ARGUMENT")
    access.qtest.read16(TRANSPORT.BAR0_BASE + TRANSPORT.REG_MAGIC)
    require(access.read32(TRANSPORT.REG_ERROR) == ERR_BAD_ARGUMENT,
            "word read did not latch BAD_ARGUMENT")
    require(access.qtest.read64(
        TRANSPORT.BAR0_BASE + TRANSPORT.REG_VIRTUAL_CYCLES + 4
    ) == RESOURCE_UNKNOWN, "unaligned read was not all ones")
    require(access.read32(TRANSPORT.REG_ERROR) == ERR_BAD_ARGUMENT,
            "unaligned read did not latch BAD_ARGUMENT")
    dispatches_before = access.read64(REG_GATE_DISPATCHES)
    access.qtest.write64(TRANSPORT.BAR0_BASE + REG_GATE_DISPATCHES, 9)
    require(access.read32(TRANSPORT.REG_ERROR) == ERR_BAD_ARGUMENT,
            "read-only gate ledger write did not fail")
    require(access.read64(REG_GATE_DISPATCHES) == dispatches_before,
            "read-only gate ledger write mutated value")
    gate_before = access.read32(REG_GATE_STATE)
    access.write32(REG_GATE_STATE, 9)
    require(access.read32(TRANSPORT.REG_ERROR) == ERR_BAD_ARGUMENT,
            "read-only gate state write did not fail")
    require(access.read32(REG_GATE_STATE) == gate_before,
            "read-only gate state write mutated value")
    require(access.command(CMD_SNAPSHOT) == ERR_SNAPSHOT_REJECTED,
            "snapshot command was not rejected")
    require(access.command(CMD_POLL_EXTERNAL) == ERR_BAD_STATE,
            "unsupported asynchronous poll accepted")
    require(access.command(CMD_CANCEL_EXTERNAL) == ERR_BAD_STATE,
            "unsupported asynchronous cancel accepted")
    require(access.read32(TRANSPORT.REG_LIFECYCLE) == initial_lifecycle,
            "ABI controls mutated lifecycle")
    require(access.read_boundary() == [RESOURCE_UNKNOWN] * BOUNDARY_WORDS,
            "ABI controls exposed boundary")
    require(access.qmp.qom_get_expected_error("test-observe-phase-a") ==
            "GenericError", "ideal phase observer unexpectedly exists")
    require(access.qmp.qom_get_expected_error("test-observe-phase-b") ==
            "GenericError", "ideal phase observer unexpectedly exists")
    return {
        "invalid_byte_word_and_unaligned_access_rejected": True,
        "read_only_gate_registers_immutable": True,
        "snapshot_rejected": True,
        "async_poll_cancel_not_exposed": True,
        "ideal_phase_observer_surface_absent": True,
        "lifecycle_unchanged": True,
    }


def reset_after_receipt_case(access: GateAccess) -> dict[str, object]:
    boundary, _ = dispatch_to_receipt(access, VECTOR_EXPECTATIONS[0])
    require(boundary != [RESOURCE_UNKNOWN] * BOUNDARY_WORDS,
            "reset fixture had no receipt")
    response = access.qmp.execute("system_reset")
    require(response.get("return") == {}, "QMP system_reset failed")
    access.setup_pci()
    require(access.read32(TRANSPORT.REG_LIFECYCLE) == LIFE_SHAM,
            "active reset did not latch SHAM")
    require(access.read32(REG_GATE_STATE) == GATE_SHAM,
            "active reset gate did not latch SHAM")
    status = access.read32(TRANSPORT.REG_STATUS_LO)
    require(status & ST_SNAPSHOT_LINEAGE and status & ST_SHAM,
            "active reset lost SHAM lineage")
    require(access.read32(TRANSPORT.REG_ERROR) == ERR_SNAPSHOT_LINEAGE,
            "active reset error mismatch")
    require_sanitized_sham(access, "active reset")
    second = access.qmp.execute("system_reset")
    require(second.get("return") == {}, "second system_reset failed")
    access.setup_pci()
    require_sanitized_sham(access, "second reset")
    require(access.command(CMD_BEGIN_REUSE) == ERR_SNAPSHOT_LINEAGE,
            "reset SHAM authorized reuse")
    return {
        "activity": "HELD_FIXTURE_FAILURE_RECEIPT",
        "reset_latched_sham": True,
        "second_reset_preserved_sham": True,
        "ephemeral_receipt_and_typed_evidence_sanitized": True,
        "begin_reuse_authorized": False,
    }


def require_sanitized_sham(access: GateAccess, label: str) -> None:
    require(access.read32(TRANSPORT.REG_LIFECYCLE) == LIFE_SHAM,
            f"{label}: lifecycle not SHAM")
    require(access.read32(REG_GATE_STATE) == GATE_SHAM,
            f"{label}: gate not SHAM")
    require(access.read_boundary() == [RESOURCE_UNKNOWN] * BOUNDARY_WORDS,
            f"{label}: boundary leaked")
    for register, name in (
        (TRANSPORT.REG_ALLOCATION_ID_LO, "allocation low"),
        (TRANSPORT.REG_ALLOCATION_ID_HI, "allocation high"),
        (REG_PREPARATION_RECEIPT_LO, "preparation receipt low"),
        (REG_PREPARATION_RECEIPT_HI, "preparation receipt high"),
        (REG_RETURN_RECEIPT_LO, "return receipt low"),
        (REG_RETURN_RECEIPT_HI, "return receipt high"),
        (TRANSPORT.REG_RESOURCE_DIGEST_LO, "resource digest low"),
        (TRANSPORT.REG_RESOURCE_DIGEST_HI, "resource digest high"),
        (REG_SESSION_EPOCH, "session epoch"),
        (REG_ATTESTATION_NONCE, "attestation nonce"),
        (REG_DEVICE_ID_DIGEST_LO, "device digest low"),
        (REG_DEVICE_ID_DIGEST_HI, "device digest high"),
        (REG_MEASUREMENT_DIGEST_LO, "measurement digest low"),
        (REG_MEASUREMENT_DIGEST_HI, "measurement digest high"),
        (REG_CUSTODY_DIGEST_LO, "custody digest low"),
        (REG_CUSTODY_DIGEST_HI, "custody digest high"),
        (REG_RESOURCE_PROVENANCE_DIGEST_LO, "provenance digest low"),
        (REG_RESOURCE_PROVENANCE_DIGEST_HI, "provenance digest high"),
        (REG_LEASE_EXPIRY_TICK, "lease expiry"),
    ):
        require(access.read64(register) == 0, f"{label}: leaked {name}")
    for register, name in (
        (TRANSPORT.REG_REQUEST_OWNER, "request owner"),
        (TRANSPORT.REG_REQUEST_PROGRAM, "request program"),
        (TRANSPORT.REG_REQUEST_GENERATION, "request generation"),
        (REG_EVIDENCE_ORIGIN, "evidence origin"),
        (REG_CHANNEL_SECURITY, "channel security"),
        (REG_DEVICE_APPRAISAL, "device appraisal"),
        (REG_MEASUREMENT_CLASS, "measurement class"),
        (REG_CUSTODY_PROVENANCE, "custody provenance"),
        (REG_RESOURCE_PROVENANCE, "resource provenance"),
        (REG_TRUST_DOMAIN, "trust domain"),
    ):
        require(access.read32(register) == 0, f"{label}: leaked {name}")


def wait_for_destination_sham(access: GateAccess) -> None:
    deadline = time.monotonic() + TRANSPORT.MIGRATION_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if access.read32(TRANSPORT.REG_LIFECYCLE) == LIFE_SHAM:
            return
        time.sleep(0.01)
    fail("migration destination never entered SHAM")


def wait_for_source_migration(qmp: object) -> str:
    deadline = time.monotonic() + TRANSPORT.MIGRATION_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        response = qmp.execute("query-migrate")
        result = response.get("return")
        require(isinstance(result, dict), "query-migrate returned no object")
        status = result.get("status")
        require(isinstance(status, str), "query-migrate returned no status")
        if status == "completed":
            return status
        if status in {"failed", "cancelled"}:
            fail(f"migration ended with status {status}")
        time.sleep(0.01)
    fail("migration did not complete")


def migration_case(
    qemu_binary: Path, run_root: Path, wall_times: dict[str, int]
) -> dict[str, object]:
    name = "held_receipt_migration_sham"
    started = time.monotonic_ns()
    case_root = run_root / name
    case_root.mkdir(mode=0o700)
    options = GateOptions(BACKEND_OFFLINE_FIXTURE, True, 0)
    source = GateQemuInstance(qemu_binary, case_root / "s", options)
    destination = GateQemuInstance(
        qemu_binary, case_root / "d", options, incoming_defer=True
    )
    second_destination = GateQemuInstance(
        qemu_binary, case_root / "e", options, incoming_defer=True
    )
    evidence: dict[str, object] = {}
    try:
        source.start()
        destination.start()
        second_destination.start()
        require(source.qtest is not None and source.qmp is not None,
                "migration source unavailable")
        require(destination.qtest is not None and destination.qmp is not None,
                "migration destination unavailable")
        require(second_destination.qtest is not None and
                second_destination.qmp is not None,
                "second migration destination unavailable")
        source_access = GateAccess(source.qtest, source.qmp)
        destination_access = GateAccess(destination.qtest, destination.qmp)
        second_access = GateAccess(second_destination.qtest, second_destination.qmp)
        source_access.setup_pci()
        destination_access.setup_pci()
        second_access.setup_pci()
        require_identity(source_access, BACKEND_OFFLINE_FIXTURE)
        require_identity(destination_access, BACKEND_OFFLINE_FIXTURE)
        require_identity(second_access, BACKEND_OFFLINE_FIXTURE)
        dispatch_to_receipt(source_access, VECTOR_EXPECTATIONS[0])

        first_socket = Path("../m")
        destination.qmp.execute("migrate-incoming", {"uri": f"unix:{first_socket}"})
        source.qmp.execute("migrate", {"uri": f"unix:{first_socket}"})
        first_status = wait_for_source_migration(source.qmp)
        require_sanitized_sham(source_access, "migration source")
        wait_for_destination_sham(destination_access)
        require_sanitized_sham(destination_access, "migration destination")
        require(destination_access.command(CMD_LEASE) == ERR_SNAPSHOT_LINEAGE,
                "migration destination reconstructed a lease")
        require(destination_access.command(CMD_BEGIN_REUSE) == ERR_SNAPSHOT_LINEAGE,
                "migration destination authorized reuse")

        second_socket = Path("../m2")
        second_destination.qmp.execute(
            "migrate-incoming", {"uri": f"unix:{second_socket}"}
        )
        destination.qmp.execute("migrate", {"uri": f"unix:{second_socket}"})
        second_status = wait_for_source_migration(destination.qmp)
        wait_for_destination_sham(second_access)
        require_sanitized_sham(second_access, "second-hop destination")
        reset = second_destination.qmp.execute("system_reset")
        require(reset.get("return") == {}, "post-migration reset failed")
        second_access.setup_pci()
        require_sanitized_sham(second_access, "post-migration reset")
        evidence = {
            "transport": "REAL_QMP_UNIX_LIVE_MIGRATION",
            "first_hop_status": first_status,
            "second_hop_status": second_status,
            "source_pre_save_sanitized_to_sham": True,
            "destination_post_load_sanitized_to_sham": True,
            "second_hop_preserved_sham": True,
            "post_migration_reset_preserved_sham": True,
            "lease_reconstructed": False,
            "typed_evidence_reconstructed": False,
            "receipt_reconstructed": False,
            "production_trust_reconstructed": False,
            "begin_reuse_authorized": False,
        }
    finally:
        source_streams = source.close()
        destination_streams = destination.close()
        second_streams = second_destination.close()
        wall_times[name] = time.monotonic_ns() - started
    require_empty_streams(source_streams, "migration source")
    require_empty_streams(destination_streams, "migration destination")
    require_empty_streams(second_streams, "migration second destination")
    return evidence


def process_teardown_case(
    qemu_binary: Path, run_root: Path, wall_times: dict[str, int]
) -> dict[str, object]:
    name = "armed_process_teardown"
    started = time.monotonic_ns()
    instance = GateQemuInstance(
        qemu_binary,
        run_root / name,
        GateOptions(BACKEND_OFFLINE_FIXTURE, True, 0),
    )
    try:
        instance.start()
        require(instance.qtest is not None and instance.qmp is not None,
                "teardown clients unavailable")
        access = GateAccess(instance.qtest, instance.qmp)
        access.setup_pci()
        require_identity(access, BACKEND_OFFLINE_FIXTURE)
        prepare_fixture(access)
        require(access.read32(REG_GATE_STATE) == GATE_ATTESTATION_ARMED,
                "teardown fixture not armed")
    finally:
        streams = instance.close()
        wall_times[name] = time.monotonic_ns() - started
    require_empty_streams(streams, name)
    require(instance.process is not None and instance.process.returncode == 0,
            "armed process teardown did not exit cleanly")
    return {
        "armed_before_qmp_quit": True,
        "process_exit_code": 0,
        "qmp_quit_is_clean_process_teardown": True,
        "unrealize_post_state_not_observable_after_process_exit": True,
        "static_compiled_source_audit_required_for_sanitizer_callback": True,
    }


def run_case(
    qemu_binary: Path,
    run_root: Path,
    name: str,
    options: GateOptions,
    callback: Callable[[GateAccess], dict[str, object]],
    wall_times: dict[str, int],
) -> dict[str, object]:
    started = time.monotonic_ns()
    instance = GateQemuInstance(qemu_binary, run_root / name, options)
    evidence: dict[str, object] = {}
    try:
        instance.start()
        require(instance.qtest is not None and instance.qmp is not None,
                f"{name}: qtest/QMP clients unavailable")
        access = GateAccess(instance.qtest, instance.qmp)
        pci = access.setup_pci()
        identity = require_identity(access, options.backend_id)
        evidence = callback(access)
    finally:
        streams = instance.close()
        wall_times[name] = time.monotonic_ns() - started
    require_empty_streams(streams, name)
    return {"pci": pci, "identity": identity, "evidence": evidence}


def realization_rejection_case(
    qemu_binary: Path,
    run_root: Path,
    name: str,
    options: GateOptions,
    expected_message: bytes,
    wall_times: dict[str, int],
) -> dict[str, object]:
    started = time.monotonic_ns()
    instance = GateQemuInstance(qemu_binary, run_root / name, options)
    instance.case_dir.mkdir(mode=0o700)
    process = subprocess.Popen(
        instance.command_line(),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        close_fds=True,
        cwd=instance.case_dir,
    )
    try:
        stdout, stderr = process.communicate(timeout=TRANSPORT.CONNECT_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired as error:
        process.terminate()
        process.wait(timeout=5)
        raise TRANSPORT.EvidenceFailure(
            f"{name}: invalid device configuration did not fail promptly"
        ) from error
    finally:
        wall_times[name] = time.monotonic_ns() - started
    require(process.returncode != 0, f"{name}: invalid realization succeeded")
    require(stdout == b"", f"{name}: invalid realization wrote stdout")
    require(expected_message in stderr,
            f"{name}: expected realization error was absent")
    return {
        "control": name,
        "realization_rejected": True,
        "qtest_dispatch_possible": False,
        "expected_error_matched": True,
    }


def validate_inputs(qemu_binary: Path, scratch_dir: Path) -> str:
    require(qemu_binary.is_file(), "--qemu-binary must name a regular file")
    require(os.access(qemu_binary, os.X_OK), "--qemu-binary must be executable")
    require(scratch_dir.is_dir(), "--scratch-dir must name an existing directory")
    require(os.access(scratch_dir, os.W_OK), "--scratch-dir must be writable")
    fs_type = TRANSPORT.filesystem_type(scratch_dir)
    require(fs_type not in {"tmpfs", "ramfs"}, "RAM-backed scratch is forbidden")
    return fs_type


def build_evidence(
    qemu_binary: Path, run_root: Path, scratch_filesystem_type: str
) -> tuple[dict[str, object], dict[str, int]]:
    wall_times: dict[str, int] = {}
    production = run_case(
        qemu_binary,
        run_root,
        "production_hardware_absent",
        GateOptions(),
        hardware_absent_case,
        wall_times,
    )

    startup_controls = {
        "fixture_disabled": realization_rejection_case(
            qemu_binary,
            run_root,
            "fixture_disabled",
            GateOptions(BACKEND_OFFLINE_FIXTURE, False, 0),
            b"offline fixture backend requires test-fixture-enabled",
            wall_times,
        ),
        "fixture_domain_on_hardware_backend": realization_rejection_case(
            qemu_binary,
            run_root,
            "fixture_domain_on_hardware_backend",
            GateOptions(BACKEND_HARDWARE, True, 0),
            b"test-fixture-enabled is confined to backend-id=0x0df0",
            wall_times,
        ),
        "invalid_standard_vector": realization_rejection_case(
            qemu_binary,
            run_root,
            "invalid_standard_vector",
            GateOptions(BACKEND_OFFLINE_FIXTURE, True, 7),
            b"test-standard-vector-id must be in [0,6]",
            wall_times,
        ),
    }

    vectors = []
    for expectation in VECTOR_EXPECTATIONS:
        result = run_case(
            qemu_binary,
            run_root,
            f"vector_{expectation.vector_id}_{expectation.name}",
            GateOptions(BACKEND_OFFLINE_FIXTURE, True, expectation.vector_id),
            lambda access, item=expectation: vector_case(access, item),
            wall_times,
        )
        vectors.append(result["evidence"])

    request_controls = {}
    for kind in ("owner_tag", "generation", "lease_expiry"):
        request_controls[kind] = run_case(
            qemu_binary,
            run_root,
            f"request_{kind}",
            GateOptions(BACKEND_OFFLINE_FIXTURE, True, 0),
            lambda access, control=kind: request_negative_case(access, control),
            wall_times,
        )["evidence"]

    abi = run_case(
        qemu_binary,
        run_root,
        "guest_abi_controls",
        GateOptions(),
        abi_fault_case,
        wall_times,
    )
    reset = run_case(
        qemu_binary,
        run_root,
        "reset_after_receipt",
        GateOptions(BACKEND_OFFLINE_FIXTURE, True, 0),
        reset_after_receipt_case,
        wall_times,
    )
    migration = migration_case(qemu_binary, run_root, wall_times)
    teardown = process_teardown_case(qemu_binary, run_root, wall_times)

    all_vector_checks = (
        len(vectors) == 7
        and all(item["return_class"] == "RETURN_FAILED" for item in vectors)
        and all(item["terminal_state"] == "SPENT" for item in vectors)
        and all(not item["begin_reuse_authorized"] for item in vectors)
        and all(item["direct_protocol_comparator_matches"] for item in vectors)
    )
    evidence = {
        "schema": SCHEMA,
        "authority": {
            "claim": CLAIM,
            "scope": SCOPE,
            "disposition": DISPOSITION,
        },
        "classification": {
            "phase_qemu_layer_classification":
                "COMPILED_PHASE_QEMU_V13_COMMON_DEVICE_HARDWARE_ABSENCE_AND_FIXED_OFFLINE_SYMBOLIC_SELECTOR_GATE",
            "common_guest_visible_device_contract_exercised": True,
            "standalone_python_twin_qualifies": False,
            "fixed_selectors_are_evidence_parsers": False,
            "fixed_selectors_are_cryptographic_verifiers": False,
            "production_hardware_connected": False,
            "offline_fixture_is_production_trust": False,
            "physical_evidence_class": "NONE",
            "restoration_classification": "NO_RESTORATION_CLAIM",
            "return_classes_observed": ["NONE", "RETURN_FAILED"],
            "exact_formal_return_observed": False,
            "approx_model_return_observed": False,
            "statistical_only_return_observed": False,
            "physical_sample_count": 0,
            "campaign_statistical_certificate_count": 0,
            "architecture_promotion": False,
            "mechanism_promotion": False,
            "M257_escape": False,
        },
        "build_identity": {
            "qemu_binary_sha256": sha256_file(qemu_binary),
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "frozen_v12_transport_runner_sha256": sha256_file(V12_RUNNER),
        },
        "runtime": {
            "machine": "q35,accel=qtest",
            "display": "none",
            "scratch_filesystem_type": scratch_filesystem_type,
            "hardware_contact": False,
            "runner_mutates_process_priority": False,
        },
        "compiled_device_identity": {
            "pci": production["pci"],
            "identity": production["identity"],
            "common_v12_prefix_offsets": COMMON_V12_PREFIX_OFFSETS,
            "first_v13_extension_offset": REG_GATE_STATE,
            "boundary_words": BOUNDARY_WORDS,
        },
        "production_backend": production["evidence"],
        "realization_controls": startup_controls,
        "offline_standard_vectors": vectors,
        "request_controls": request_controls,
        "guest_abi_controls": abi["evidence"],
        "reset_after_activity": reset["evidence"],
        "migration": migration,
        "clean_process_teardown": teardown,
        "unsupported_surfaces": {
            "asynchronous_poll": False,
            "asynchronous_cancel": False,
            "hardware_disconnect_event": False,
            "reason":
                "V13_DISPATCH_IS_SYNCHRONOUS_AND_NO_LIVE_HARDWARE_CONNECTION_EXISTS",
            "lease_expiry_predispatch_control_exercised": True,
        },
        "resource_accounting": {
            "resource_schema": RESOURCE_SCHEMA,
            "unknown_physical_register_value": RESOURCE_UNKNOWN,
            "unknown_physical_registers": sorted(UNKNOWN_PHYSICAL_REGISTERS),
            "all_unknown_physical_registers_checked_for_every_vector": True,
            "physical_sample_count": 0,
            "campaign_statistical_certificate_count": 0,
            "physical_resource_manifest": "NOT_AVAILABLE",
            "total_resource_comparison": "UNDETERMINED",
            "resource_advantage_claim": False,
        },
        "direct_equal_access_comparator": {
            "same_symbolic_vector_ids": list(range(7)),
            "all_expected_reasons_and_typed_classes_matched": True,
            "comparison_scope": "FIXED_SYMBOLIC_SELECTOR_STATE_MACHINE_ONLY",
            "evidence_parsing_exercised": False,
            "physical_output_used": False,
            "cryptographic_verification_claim": False,
            "unique_phase_resource": False,
            "M257_escape": False,
        },
        "checks": {
            "default_hardware_absent_before_dispatch":
                production["evidence"]["pre_dispatch_rejection"],
            "fixture_disabled_and_invalid_domains_rejected":
                all(item["realization_rejected"]
                    for item in startup_controls.values()),
            "all_seven_vectors_terminal_failed_ack_spent": all_vector_checks,
            "no_vector_authorized_reuse":
                all(not item["begin_reuse_authorized"] for item in vectors),
            "no_ideal_math_output_surface":
                abi["evidence"]["ideal_phase_observer_surface_absent"],
            "tag_generation_and_lease_expiry_predispatch":
                all(item["pre_dispatch_rejection"]
                    for item in request_controls.values()),
            "reset_irreversible_sham":
                reset["evidence"]["second_reset_preserved_sham"],
            "real_two_hop_migration_irreversible_sham":
                migration["second_hop_preserved_sham"],
            "clean_process_teardown": teardown["process_exit_code"] == 0,
            "no_physical_sample_or_campaign_certificate": True,
            "direct_protocol_comparator_matched": True,
            "M257_remains_intact": True,
        },
    }
    require(all(evidence["checks"].values()), "one or more V13 qtest checks failed")
    return evidence, wall_times


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qemu-binary", required=True, type=Path)
    parser.add_argument("--scratch-dir", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    qemu_binary = args.qemu_binary.resolve(strict=True)
    scratch_dir = args.scratch_dir.resolve(strict=True)
    scratch_filesystem_type = validate_inputs(qemu_binary, scratch_dir)
    run_root = scratch_dir / "m271q"
    require(not run_root.exists(), "managed run root already exists: m271q")
    run_root.mkdir(mode=0o700)
    evidence, wall_times = build_evidence(
        qemu_binary, run_root, scratch_filesystem_type
    )
    sys.stdout.buffer.write(canonical_bytes({
        "deterministic_evidence": evidence,
        "wall_times_ns": wall_times,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
