#!/usr/bin/env python3
"""Headless qtest for the compiled M270 Phase-QEMU V12 adapter device.

This runner contacts no hardware.  It drives only QEMU's qtest/QMP sockets and
the test-only deterministic adapter authenticator compiled into the distinct
V12 sibling device alongside frozen V11.  The internal model may close exactly, but every
external result is classified APPROX_MODEL and is never eligible for reuse.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Callable, Sequence


BASE_RUNNER = (
    Path(__file__).resolve().parents[2]
    / "phase_qemu_v11"
    / "tests"
    / "run_phase_qemu_v11_qtest.py"
)
SPEC = importlib.util.spec_from_file_location("phase_qemu_v11_base_runner", BASE_RUNNER)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load the frozen V11 qtest support")
BASE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BASE
SPEC.loader.exec_module(BASE)

# Reuse only the frozen qtest/QMP transport helpers.  V12 has an independent
# PCI/QOM identity while preserving the V11 guest register prefix.
BASE.DEVICE_TYPE = "phase-qemu-v12"
BASE.PCI_DEVICE = 0x11FC
BASE.MAGIC = 0x50483132
BASE.ABI = 0x00020000


BACKEND_EXTERNAL = 0x0B80
RETURN_APPROX_MODEL = 2
RETURN_FAILED = 3
LIFE_PRIVATE_ARMED = 5
LIFE_PRIVATE_READY = 6
LIFE_EXECUTING = 7
LIFE_RESPONSE_READY = 9
LIFE_SPENT = 12
ADAPTER_READY = 1
ADAPTER_WAITING_A = 2
ADAPTER_WAITING_B = 3
ADAPTER_COMPLETE = 4
ADAPTER_CANCELED = 5
ADAPTER_TIMED_OUT = 6
CMD_POLL_EXTERNAL = 11
CMD_CANCEL_EXTERNAL = 12
ERR_NONE = 0
ERR_BAD_STATE = 1
ERR_TAG_MISMATCH = 3
ERR_GENERATION_MISMATCH = 4
ERR_BACKEND_UNAVAILABLE = 15
ERR_REUSE_NOT_QUALIFIED = 27
ERR_ADAPTER_TIMEOUT = 28
ERR_ADAPTER_AUTH_BINDING = 32
ERR_ADAPTER_REPLAY = 33
ERR_ADAPTER_ORDER = 34
ERR_ADAPTER_EXPIRED = 35
ERR_ADAPTER_SLOT = 36
ERR_ADAPTER_CANCELED = 37
ST_RESPONSE_READY = 1 << 5
ST_EXACT_RETURN = 1 << 7
ST_OUTPUTS_HELD = 1 << 13
ST_RETURN_VERIFIED = 1 << 14
ST_SAME_ALLOCATION = 1 << 16
ST_ENV_FACTORED = 1 << 17
ST_ADAPTER_AUTHENTICATED = 1 << 23
ST_ADAPTER_PENDING = 1 << 24
ST_EXTERNAL_REUSE_FORBIDDEN = 1 << 25
REG_ADAPTER_STATE = 0x1C0
REG_ADAPTER_AUTH_ACCEPTED = 0x1C8
REG_ADAPTER_AUTH_REJECTED = 0x1D0
REG_ADAPTER_DISPATCHES = 0x1D8
REG_ADAPTER_COMPLETIONS = 0x1E0
REG_ADAPTER_CANCELS = 0x1E8
REG_ADAPTER_VIRTUAL_TICK = 0x1F0
REG_ADAPTER_DEADLINE_TICK = 0x1F8
RESOURCE_UNKNOWN = (1 << 64) - 1
RESOURCE_SCHEMA = 0x00020001
REG_RESOURCE_DURATION_FS = 0x118
REG_RESOURCE_PORT_BANDWIDTH_HZ = 0x120
REG_RESOURCE_ACTION_Q40_RAD = 0x128
REG_RESOURCE_MEAN_ENERGY_ATTOJ = 0x130
REG_RESOURCE_ENV_HISTORY_CELLS = 0x148
REG_RESOURCE_CUSTODY_TRANSITIONS = 0x150
REG_RESOURCE_OUTPUT_HOLD_FS = 0x168
REG_RESOURCE_MAINTENANCE_OPS = 0x170
QOM_ENVELOPE_A = "test-adapter-envelope-a"
QOM_ENVELOPE_B = "test-adapter-envelope-b"
QOM_ARM_NONCE = "test-observe-arm-nonce"
TAG_MASK = (1 << 21) - 1


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BASE.EvidenceFailure(message)


def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fnv_u64(current: int, value: int) -> int:
    for byte in range(8):
        current ^= (value >> (8 * byte)) & 0xFF
        current = (current * 1099511628211) & ((1 << 64) - 1)
    return current


def receipt_hash(domain: int, first: int, second: int, third: int) -> int:
    current = 14695981039346656037
    for value in (domain, first, second, third):
        current = fnv_u64(current, value)
    return current


def test_tag(
    allocation_lo: int,
    allocation_hi: int,
    fingerprint: int,
    arm_nonce: int,
    payload: int,
) -> int:
    current = receipt_hash(0x4D323730, allocation_lo, allocation_hi, fingerprint)
    current = fnv_u64(current, arm_nonce)
    current = fnv_u64(current, BACKEND_EXTERNAL)
    return fnv_u64(current, payload) & TAG_MASK


def omega_power(exponent: int) -> list[int]:
    return ([1, 0], [0, 1], [-1, -1])[exponent % 3]


def direct_phase_compiler(residue_a: int, residue_b: int) -> dict[str, object]:
    return {
        "client_a_diagonal": [[1, 0], omega_power(residue_a)],
        "client_b_diagonal": [[1, 0], omega_power(residue_b)],
        "residue_accesses": 2,
    }


def recompute_resource_digest(access: BASE.DeviceAccess, domain: int) -> int:
    status = access.read32(BASE.REG_STATUS_LO)
    values = (
        domain,
        access.read64(BASE.REG_RESOURCE_QUERY_APPLICATIONS),
        access.read64(BASE.REG_RESOURCE_RETURN_CHECKS),
        access.read64(BASE.REG_RESOURCE_ENVIRONMENT_OPS),
        access.read64(BASE.REG_RESOURCE_CONTROL_WORDS),
        access.read64(BASE.REG_RESOURCE_PREPARATION_OPS),
        access.read64(BASE.REG_RESOURCE_CERTIFICATION_OPS),
        access.read64(BASE.REG_RESOURCE_LOGICAL_QUERIES),
        access.read64(REG_RESOURCE_DURATION_FS),
        access.read64(REG_RESOURCE_PORT_BANDWIDTH_HZ),
        access.read64(REG_RESOURCE_ACTION_Q40_RAD),
        access.read64(REG_RESOURCE_MEAN_ENERGY_ATTOJ),
        access.read64(BASE.REG_RESOURCE_LOSS_Q63),
        access.read64(BASE.REG_RESOURCE_DEPHASING_Q63),
        access.read64(REG_RESOURCE_ENV_HISTORY_CELLS),
        access.read64(REG_RESOURCE_CUSTODY_TRANSITIONS),
        access.read64(BASE.REG_RESOURCE_REUSE_COUNT),
        access.read64(BASE.REG_RESOURCE_DISCARDED_TRIALS),
        access.read64(REG_RESOURCE_OUTPUT_HOLD_FS),
        access.read64(REG_RESOURCE_MAINTENANCE_OPS),
        access.read64(REG_ADAPTER_AUTH_ACCEPTED),
        access.read64(REG_ADAPTER_AUTH_REJECTED),
        access.read64(REG_ADAPTER_DISPATCHES),
        access.read64(REG_ADAPTER_COMPLETIONS),
        access.read64(REG_ADAPTER_CANCELS),
        access.read64(REG_ADAPTER_VIRTUAL_TICK),
        access.read64(REG_ADAPTER_DEADLINE_TICK),
        access.read32(REG_ADAPTER_STATE),
        int(bool(status & ST_ADAPTER_AUTHENTICATED)),
        access.read32(BASE.REG_RESOURCE_SCHEMA),
        access.read32(BASE.REG_RESOURCE_STATE_CELLS),
        access.read32(BASE.REG_RESOURCE_SCRATCH_CELLS),
        access.read64(BASE.REG_RESOURCE_PEAK_BITS),
        access.read64(BASE.REG_RESOURCE_SECRET_STORAGE_BITS),
        access.read64(BASE.REG_RESOURCE_PRECISION_BITS),
        access.read64(BASE.REG_RESOURCE_SECRET_ENTROPY_BITS),
        access.read64(BASE.REG_RESOURCE_CARRIER_PHOTON_NUMBER),
        access.read64(BASE.REG_RESOURCE_COMPILER_OPS),
        access.read64(BASE.REG_RESOURCE_CONTROLLER_OPS),
        access.read64(BASE.REG_RESOURCE_CONSTRUCTION_OPS),
    )
    current = 14695981039346656037
    for value in values:
        current = fnv_u64(current, value)
    return current


def make_envelope(
    access: BASE.DeviceAccess,
    slot: int,
    residue: int,
    *,
    generation: int = 1,
    sequence: int | None = None,
    issued: int = 0,
    expires: int = 8,
    tag_delta: int = 0,
) -> int:
    if sequence is None:
        sequence = slot + 1
    payload = (
        residue
        | (slot << 2)
        | ((generation & 0xFFFF) << 3)
        | ((sequence & 0xFF) << 19)
        | ((issued & 0xFF) << 27)
        | ((expires & 0xFF) << 35)
    )
    tag = test_tag(
        access.read64(BASE.REG_ALLOCATION_ID_LO),
        access.read64(BASE.REG_ALLOCATION_ID_HI),
        access.read64(BASE.REG_DESCRIPTOR_FINGERPRINT),
        access.qmp.qom_get(QOM_ARM_NONCE),
        payload,
    )
    return payload | (((tag + tag_delta) & TAG_MASK) << 43)


@dataclass(frozen=True)
class AdapterOptions:
    provider_enabled: bool = True
    adapter_enabled: bool = True
    adapter_mode: int = 0


class AdapterQemuInstance(BASE.QemuInstance):
    def __init__(
        self,
        qemu_binary: Path,
        case_dir: Path,
        options: AdapterOptions,
        *,
        incoming_defer: bool = False,
    ):
        super().__init__(
            qemu_binary,
            case_dir,
            BASE.DeviceOptions(backend_id=BACKEND_EXTERNAL),
            incoming_defer=incoming_defer,
        )
        self.adapter_options = options

    def start(self) -> None:
        self.case_dir.mkdir(mode=0o700)
        device = ",".join(
            (
                BASE.DEVICE_TYPE,
                f"id={BASE.DEVICE_ID}",
                "bus=pcie.0",
                f"addr={BASE.PCI_DEVICE_NUMBER:x}.0",
                f"backend-id=0x{BACKEND_EXTERNAL:04x}",
                "carrier-present=on",
                "test-fault-mode=0",
                "open-model-q32=0",
                f"test-provider-enabled={'on' if self.adapter_options.provider_enabled else 'off'}",
                f"test-adapter-enabled={'on' if self.adapter_options.adapter_enabled else 'off'}",
                f"test-adapter-mode={self.adapter_options.adapter_mode}",
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
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            close_fds=True,
            cwd=self.case_dir,
        )
        self.qmp = BASE.QMPClient(BASE.connect_unix(self.qmp_path, self.process))
        self.qtest = BASE.QTestClient(BASE.connect_unix(self.qtest_path, self.process))


def prepare(access: BASE.DeviceAccess) -> None:
    access.configure_request(0xA270, 0xB270, 1)
    require(access.command(BASE.CMD_LEASE) == ERR_NONE, "LEASE failed")
    require(access.command(BASE.CMD_PREPARE) == ERR_NONE, "PREPARE failed")
    require(access.command(BASE.CMD_ISOLATE_SOURCE) == ERR_NONE, "ISOLATE failed")
    access.upload_descriptor()
    require(access.command(BASE.CMD_SEAL_DESCRIPTOR) == ERR_NONE, "SEAL failed")
    require(access.command(BASE.CMD_ARM_PRIVATE) == ERR_NONE, "ARM failed")
    require(access.read32(BASE.REG_LIFECYCLE) == LIFE_PRIVATE_ARMED, "ARM lifecycle")
    require(access.read32(REG_ADAPTER_STATE) == ADAPTER_READY, "adapter not ready")


def bind(access: BASE.DeviceAccess, residue_a: int, residue_b: int) -> None:
    access.qmp.qom_set(QOM_ENVELOPE_A, make_envelope(access, 0, residue_a))
    access.qmp.qom_set(QOM_ENVELOPE_B, make_envelope(access, 1, residue_b))
    require(access.read32(BASE.REG_LIFECYCLE) == LIFE_PRIVATE_READY, "private not ready")
    require(access.read64(REG_ADAPTER_AUTH_ACCEPTED) == 2, "auth accept count")
    require(access.read32(BASE.REG_STATUS_LO) & ST_ADAPTER_AUTHENTICATED,
            "authenticated status absent")


def success_case(access: BASE.DeviceAccess, residue_a: int, residue_b: int) -> dict[str, object]:
    prepare(access)
    bind(access, residue_a, residue_b)
    require(access.command(BASE.CMD_EXECUTE_ATOMIC) == ERR_NONE, "execute start failed")
    require(access.read32(BASE.REG_LIFECYCLE) == LIFE_EXECUTING, "not executing")
    require(access.read32(REG_ADAPTER_STATE) == ADAPTER_WAITING_A, "not waiting A")
    require(access.read32(BASE.REG_STATUS_LO) & ST_ADAPTER_PENDING, "pending bit absent")
    require(access.command(CMD_POLL_EXTERNAL) == ERR_NONE, "poll A failed")
    require(access.read32(REG_ADAPTER_STATE) == ADAPTER_WAITING_B, "not waiting B")
    require(access.command(CMD_POLL_EXTERNAL) == ERR_NONE, "poll B failed")
    require(access.read32(BASE.REG_LIFECYCLE) == LIFE_RESPONSE_READY, "no response")
    require(access.read32(REG_ADAPTER_STATE) == ADAPTER_COMPLETE, "not complete")
    require(access.read32(BASE.REG_RETURN_CLASS) == RETURN_APPROX_MODEL,
            "external model was not APPROX_MODEL")
    status = access.read32(BASE.REG_STATUS_LO)
    require(status & ST_RESPONSE_READY, "response-ready bit absent")
    require(status & ST_OUTPUTS_HELD, "outputs not held")
    require(status & ST_ADAPTER_AUTHENTICATED,
            "authenticated lineage absent from committed status")
    require(status & ST_EXTERNAL_REUSE_FORBIDDEN, "reuse-forbidden bit absent")
    require(not status & ST_EXACT_RETURN, "physical exact-return bit asserted")
    require(not status & ST_RETURN_VERIFIED, "physical return verified")
    require(not status & ST_SAME_ALLOCATION, "physical same-allocation asserted")
    require(not status & ST_ENV_FACTORED, "physical environment factorized")
    observers = access.read_observers()
    require(observers["test-observe-phase-a"] == residue_a, "phase A mismatch")
    require(observers["test-observe-phase-b"] == residue_b, "phase B mismatch")
    require(observers["test-observe-kr-return"] == 0, "physical KR return asserted")
    require(observers["test-observe-factorized"] == 0, "physical factorization asserted")
    require(access.read64(REG_ADAPTER_DISPATCHES) == 2, "dispatch count mismatch")
    require(access.read64(REG_ADAPTER_COMPLETIONS) == 2, "completion count mismatch")
    require(access.read64(REG_ADAPTER_VIRTUAL_TICK) == 2, "virtual tick mismatch")
    require(access.read64(REG_ADAPTER_DEADLINE_TICK) == 2, "deadline mismatch")
    require(access.read64(BASE.REG_RESOURCE_LOSS_Q63) == RESOURCE_UNKNOWN,
            "unknown physical loss encoded as value")
    require(access.read64(BASE.REG_RESOURCE_DEPHASING_Q63) == RESOURCE_UNKNOWN,
            "unknown physical dephasing encoded as value")
    require(access.read64(BASE.REG_RESOURCE_CARRIER_PHOTON_NUMBER) == RESOURCE_UNKNOWN,
            "ideal carrier photon count was misreported as physical evidence")
    for register, label in (
        (REG_RESOURCE_DURATION_FS, "duration"),
        (REG_RESOURCE_PORT_BANDWIDTH_HZ, "bandwidth"),
        (REG_RESOURCE_MEAN_ENERGY_ATTOJ, "mean energy"),
        (REG_RESOURCE_OUTPUT_HOLD_FS, "output hold"),
        (REG_RESOURCE_MAINTENANCE_OPS, "maintenance"),
        (BASE.REG_RESOURCE_COMPILER_OPS, "compiler"),
        (BASE.REG_RESOURCE_CONSTRUCTION_OPS, "construction"),
    ):
        require(access.read64(register) == RESOURCE_UNKNOWN,
                f"unknown physical {label} encoded as a value")
    require(access.read32(BASE.REG_RESOURCE_SCHEMA) == RESOURCE_SCHEMA,
            "resource schema mismatch")
    boundary = access.read_boundary()
    require((boundary[1] >> 48) & 0xFFFF == RETURN_APPROX_MODEL,
            "boundary return class mismatch")
    require(boundary[2] & ST_ADAPTER_AUTHENTICATED,
            "immutable boundary omitted authenticated lineage")
    require(recompute_resource_digest(access, 0x52534C4F) ==
            access.read64(BASE.REG_RESOURCE_DIGEST_LO),
            "independent low resource digest mismatch")
    require(recompute_resource_digest(access, 0x52534849) ==
            access.read64(BASE.REG_RESOURCE_DIGEST_HI),
            "independent high resource digest mismatch")
    require(boundary[12] == access.read64(BASE.REG_RESOURCE_DIGEST_LO),
            "boundary omitted the sealed low resource digest")
    require(boundary[13] == access.read64(BASE.REG_RESOURCE_DIGEST_HI),
            "boundary omitted the sealed high resource digest")
    digest_before = (
        access.read64(BASE.REG_RESOURCE_DIGEST_LO),
        access.read64(BASE.REG_RESOURCE_DIGEST_HI),
    )
    controller_before = access.read64(BASE.REG_RESOURCE_CONTROLLER_OPS)
    rejected_before = access.read64(REG_ADAPTER_AUTH_REJECTED)
    require(access.command(CMD_POLL_EXTERNAL) == BASE.ERR_RESPONSE_LOCKED,
            "post-commit poll was not rejected")
    error_before = access.read32(BASE.REG_ERROR)
    late_property_error = access.qmp.qom_set_expected_error(QOM_ENVELOPE_A, 0)
    require(late_property_error == "GenericError",
            "post-commit adapter envelope did not fail closed")
    require(access.read_boundary() == boundary,
            "post-commit rejected control mutated boundary")
    require((
        access.read64(BASE.REG_RESOURCE_DIGEST_LO),
        access.read64(BASE.REG_RESOURCE_DIGEST_HI),
    ) == digest_before, "post-commit rejected control mutated sealed digest")
    require(access.read64(BASE.REG_RESOURCE_CONTROLLER_OPS) == controller_before,
            "post-commit rejected control mutated sealed resource snapshot")
    require(access.read64(REG_ADAPTER_AUTH_REJECTED) == rejected_before,
            "post-commit adapter envelope mutated rejection accounting")
    require(access.read32(BASE.REG_ERROR) == error_before,
            "post-commit adapter envelope mutated the sealed response error")
    require(access.command(BASE.CMD_ACK_RESPONSE) == ERR_NONE, "ACK failed")
    require(access.read32(BASE.REG_LIFECYCLE) == LIFE_SPENT, "external result not spent")
    require(access.command(BASE.CMD_BEGIN_REUSE) == ERR_REUSE_NOT_QUALIFIED,
            "BEGIN_REUSE was accepted")
    comparator = direct_phase_compiler(residue_a, residue_b)
    require(comparator["client_a_diagonal"][1] == omega_power(observers["test-observe-phase-a"]),
            "direct compiler client A parity failed")
    require(comparator["client_b_diagonal"][1] == omega_power(observers["test-observe-phase-b"]),
            "direct compiler client B parity failed")
    return {
        "fixture": f"ideal_pair_{residue_a}_{residue_b}",
        "residues": [residue_a, residue_b],
        "authenticated_envelopes": 2,
        "dispatches": 2,
        "completions": 2,
        "virtual_ticks": 2,
        "return_class": "APPROX_MODEL",
        "ideal_algebra_phases": [residue_a, residue_b],
        "physical_return_claim": False,
        "begin_reuse_authorized": False,
        "outputs_held_until_ack": True,
        "sealed_resource_snapshot_immutable_before_ack": True,
        "post_commit_qom_envelope_rejected_without_live_counter_mutation": True,
        "resource_digest_independently_recomputed": True,
        "direct_equal_access_compiler": comparator,
        "direct_compiler_matches_observed_model_phases": True,
    }


def auth_negative_case(access: BASE.DeviceAccess, kind: str) -> dict[str, object]:
    prepare(access)
    expected: int
    if kind == "bad_tag":
        envelope = make_envelope(access, 0, 1, tag_delta=1)
        expected = ERR_ADAPTER_AUTH_BINDING
        property_name = QOM_ENVELOPE_A
    elif kind == "wrong_slot":
        envelope = make_envelope(access, 1, 1)
        expected = ERR_ADAPTER_SLOT
        property_name = QOM_ENVELOPE_A
    elif kind == "stale_generation":
        envelope = make_envelope(access, 0, 1, generation=0)
        expected = ERR_GENERATION_MISMATCH
        property_name = QOM_ENVELOPE_A
    elif kind == "slot_b_before_a":
        envelope = make_envelope(access, 1, 2)
        expected = ERR_ADAPTER_ORDER
        property_name = QOM_ENVELOPE_B
    elif kind == "expired":
        envelope = make_envelope(access, 0, 1, expires=0)
        expected = ERR_ADAPTER_EXPIRED
        property_name = QOM_ENVELOPE_A
    elif kind == "direct_bypass":
        error_class = access.qmp.qom_set_expected_error(BASE.QOM_PRIVATE_A, 1)
        require(access.read32(BASE.REG_ERROR) == ERR_ADAPTER_AUTH_BINDING,
                "direct bypass error mismatch")
        require(access.read64(REG_ADAPTER_DISPATCHES) == 0,
                "direct private bypass dispatched external work")
        return {
            "fixture": "auth_reject_direct_private_bypass",
            "qmp_error_class": error_class,
            "error": ERR_ADAPTER_AUTH_BINDING,
            "dispatch_delta": 0,
        }
    elif kind == "replay":
        envelope = make_envelope(access, 0, 1)
        access.qmp.qom_set(QOM_ENVELOPE_A, envelope)
        error_class = access.qmp.qom_set_expected_error(QOM_ENVELOPE_A, envelope)
        require(access.read32(BASE.REG_ERROR) == ERR_ADAPTER_REPLAY, "replay error mismatch")
        require(access.read64(REG_ADAPTER_AUTH_REJECTED) == 1, "replay reject count")
        require(access.read64(REG_ADAPTER_DISPATCHES) == 0,
                "replayed envelope dispatched external work")
        return {
            "fixture": "auth_reject_exact_replay",
            "qmp_error_class": error_class,
            "error": ERR_ADAPTER_REPLAY,
            "dispatch_delta": 0,
        }
    else:
        raise AssertionError(kind)
    error_class = access.qmp.qom_set_expected_error(property_name, envelope)
    require(access.read32(BASE.REG_ERROR) == expected, f"{kind} error mismatch")
    require(access.read64(REG_ADAPTER_AUTH_REJECTED) == 1, f"{kind} reject count")
    require(access.read64(REG_ADAPTER_DISPATCHES) == 0, f"{kind} dispatched")
    return {
        "fixture": f"auth_reject_{kind}",
        "qmp_error_class": error_class,
        "error": expected,
        "dispatch_delta": 0,
    }


def timeout_case(access: BASE.DeviceAccess) -> dict[str, object]:
    prepare(access)
    bind(access, 1, 2)
    require(access.command(BASE.CMD_EXECUTE_ATOMIC) == ERR_NONE, "timeout execute")
    require(access.command(CMD_POLL_EXTERNAL) == ERR_NONE, "timeout poll A")
    require(access.command(CMD_POLL_EXTERNAL) == ERR_ADAPTER_TIMEOUT, "timeout deadline")
    require(access.read32(BASE.REG_LIFECYCLE) == LIFE_RESPONSE_READY,
            "timeout failure receipt not ready")
    require(access.read32(REG_ADAPTER_STATE) == ADAPTER_TIMED_OUT, "timeout state")
    require(access.read32(BASE.REG_RETURN_CLASS) == RETURN_FAILED,
            "timeout return was not FAILED")
    require(access.read32(BASE.REG_STATUS_LO) & BASE.ST_RESOURCE_SEALED,
            "timeout resource attempt was not sealed")
    boundary = access.read_boundary()
    require((boundary[1] >> 48) & 0xFFFF == RETURN_FAILED,
            "timeout boundary class mismatch")
    require(recompute_resource_digest(access, 0x52534C4F) == boundary[12],
            "timeout resource digest mismatch")
    require(recompute_resource_digest(access, 0x52534849) == boundary[13],
            "timeout high resource digest mismatch")
    require(access.read64(REG_ADAPTER_CANCELS) == 1,
            "timeout did not cancel external work exactly once")
    require(access.command(BASE.CMD_ACK_RESPONSE) == ERR_NONE,
            "timeout failure receipt ACK failed")
    require(access.read32(BASE.REG_LIFECYCLE) == LIFE_SPENT, "timeout not spent")
    require(access.command(BASE.CMD_BEGIN_REUSE) == ERR_REUSE_NOT_QUALIFIED,
            "timeout failure authorized BEGIN_REUSE")
    require(access.command(CMD_POLL_EXTERNAL) == BASE.ERR_RESPONSE_LOCKED,
            "late poll accepted")
    return {
        "fixture": "async_timeout_exact_deadline",
        "deadline_tick": 2,
        "terminal_tick": access.read64(REG_ADAPTER_VIRTUAL_TICK),
        "cancel_count": 1,
        "failed_attempt_resource_sealed": True,
        "failure_receipt_acked_to_spent": True,
        "late_completion_rejected": True,
        "begin_reuse_authorized": False,
    }


def cancel_case(access: BASE.DeviceAccess) -> dict[str, object]:
    prepare(access)
    bind(access, 2, 1)
    require(access.command(BASE.CMD_EXECUTE_ATOMIC) == ERR_NONE, "cancel execute")
    require(access.command(CMD_CANCEL_EXTERNAL) == ERR_ADAPTER_CANCELED, "cancel failed")
    require(access.read32(BASE.REG_LIFECYCLE) == LIFE_RESPONSE_READY,
            "cancel failure receipt not ready")
    require(access.read32(REG_ADAPTER_STATE) == ADAPTER_CANCELED, "cancel state")
    require(access.read32(BASE.REG_RETURN_CLASS) == RETURN_FAILED,
            "cancel return was not FAILED")
    require(access.read32(BASE.REG_STATUS_LO) & BASE.ST_RESOURCE_SEALED,
            "cancel resource attempt was not sealed")
    boundary = access.read_boundary()
    require((boundary[1] >> 48) & 0xFFFF == RETURN_FAILED,
            "cancel boundary class mismatch")
    require(recompute_resource_digest(access, 0x52534C4F) == boundary[12],
            "cancel resource digest mismatch")
    require(recompute_resource_digest(access, 0x52534849) == boundary[13],
            "cancel high resource digest mismatch")
    require(access.read64(REG_ADAPTER_CANCELS) == 1, "cancel count")
    require(access.command(BASE.CMD_ACK_RESPONSE) == ERR_NONE,
            "cancel failure receipt ACK failed")
    require(access.read32(BASE.REG_LIFECYCLE) == LIFE_SPENT, "cancel not spent")
    require(access.command(BASE.CMD_BEGIN_REUSE) == ERR_REUSE_NOT_QUALIFIED,
            "canceled failure authorized BEGIN_REUSE")
    require(access.command(CMD_POLL_EXTERNAL) == BASE.ERR_RESPONSE_LOCKED,
            "late completion accepted")
    return {
        "fixture": "async_cancel_before_completion",
        "cancel_count": 1,
        "failed_attempt_resource_sealed": True,
        "failure_receipt_acked_to_spent": True,
        "late_completion_rejected": True,
        "begin_reuse_authorized": False,
    }


def pending_reset_case(access: BASE.DeviceAccess) -> dict[str, object]:
    prepare(access)
    bind(access, 1, 2)
    require(access.command(BASE.CMD_EXECUTE_ATOMIC) == ERR_NONE, "reset execute")
    require(access.read32(REG_ADAPTER_STATE) == ADAPTER_WAITING_A,
            "reset fixture not pending")
    response = access.qmp.execute("system_reset")
    require(response.get("return") == {}, "QMP system_reset failed")
    pci = access.setup_pci()
    status = access.read32(BASE.REG_STATUS_LO)
    require(access.read32(BASE.REG_LIFECYCLE) == BASE.LIFE_SHAM,
            "pending reset revived external state")
    require(status & BASE.ST_SNAPSHOT_LINEAGE, "pending reset lost snapshot lineage")
    require(status & BASE.ST_SHAM, "pending reset did not latch SHAM")
    require(access.read32(BASE.REG_ERROR) == BASE.ERR_SNAPSHOT_LINEAGE,
            "pending reset did not latch snapshot error")
    require(access.read32(REG_ADAPTER_STATE) == 7, "pending reset adapter not SHAM")
    require(access.read64(REG_ADAPTER_CANCELS) == 1,
            "pending reset did not cancel exactly once")
    require(access.read_boundary() == [BASE.LOCKED64] * BASE.BOUNDARY_WORDS,
            "pending reset leaked boundary")
    require(access.command(BASE.CMD_LEASE) == BASE.ERR_SNAPSHOT_LINEAGE,
            "pending reset made lineage leasable")
    second = access.qmp.execute("system_reset")
    require(second.get("return") == {}, "second QMP system_reset failed")
    access.setup_pci()
    require(access.read32(BASE.REG_LIFECYCLE) == BASE.LIFE_SHAM,
            "second reset escaped SHAM")
    require(access.read64(REG_ADAPTER_CANCELS) == 1,
            "second reset repeated cancellation")
    require(access.command(BASE.CMD_BEGIN_REUSE) == BASE.ERR_SNAPSHOT_LINEAGE,
            "pending-reset SHAM authorized BEGIN_REUSE")
    return {
        "fixture": "pending_reset_latches_irreversible_sham",
        "post_reset_pci": pci,
        "cancel_count": 1,
        "second_reset_preserved_sham": True,
        "boundary_locked": True,
        "begin_reuse_authorized": False,
    }


def unavailable_case(access: BASE.DeviceAccess) -> dict[str, object]:
    access.configure_request(0xA270, 0xB270, 1)
    require(access.command(BASE.CMD_LEASE) == ERR_BACKEND_UNAVAILABLE,
            "service-mode external lease did not fail")
    require(access.read32(BASE.REG_LIFECYCLE) == BASE.LIFE_EMPTY,
            "unavailable lease changed lifecycle")
    return {
        "fixture": "external_service_mode_unavailable",
        "lease_error": ERR_BACKEND_UNAVAILABLE,
        "lifecycle_unchanged": True,
        "physical_hardware_connected": False,
    }


def run_case(
    qemu_binary: Path,
    run_root: Path,
    name: str,
    options: AdapterOptions,
    callback: Callable[[BASE.DeviceAccess], dict[str, object]],
    wall_times: dict[str, int],
) -> dict[str, object]:
    started = time.monotonic_ns()
    instance = AdapterQemuInstance(qemu_binary, run_root / name, options)
    try:
        instance.start()
        require(instance.qtest is not None and instance.qmp is not None, "clients unavailable")
        access = BASE.DeviceAccess(instance.qtest, instance.qmp)
        pci = access.setup_pci()
        identity = BASE.require_identity(access, BACKEND_EXTERNAL)
        evidence = callback(access)
    finally:
        streams = instance.close()
        wall_times[name] = time.monotonic_ns() - started
    BASE.require_empty_streams(streams, name)
    return {"pci": pci, "identity": identity, "evidence": evidence, "streams": streams}


def migration_case(
    qemu_binary: Path,
    run_root: Path,
    wall_times: dict[str, int],
) -> dict[str, object]:
    name = "migration_inflight_sham"
    started = time.monotonic_ns()
    case_root = run_root / name
    case_root.mkdir(mode=0o700)
    options = AdapterOptions()
    source = AdapterQemuInstance(qemu_binary, case_root / "s", options)
    destination = AdapterQemuInstance(
        qemu_binary, case_root / "d", options, incoming_defer=True
    )
    second_destination = AdapterQemuInstance(
        qemu_binary, case_root / "e", options, incoming_defer=True
    )
    source_streams: dict[str, object]
    destination_streams: dict[str, object]
    second_destination_streams: dict[str, object]
    try:
        source.start()
        destination.start()
        second_destination.start()
        require(source.qtest is not None and source.qmp is not None,
                "migration source unavailable")
        require(destination.qtest is not None and destination.qmp is not None,
                "migration destination unavailable")
        require(second_destination.qtest is not None and second_destination.qmp is not None,
                "second migration destination unavailable")
        source_access = BASE.DeviceAccess(source.qtest, source.qmp)
        destination_access = BASE.DeviceAccess(destination.qtest, destination.qmp)
        second_access = BASE.DeviceAccess(second_destination.qtest, second_destination.qmp)
        source_pci = source_access.setup_pci()
        destination_pci = destination_access.setup_pci()
        second_pci = second_access.setup_pci()
        BASE.require_identity(source_access, BACKEND_EXTERNAL)
        BASE.require_identity(destination_access, BACKEND_EXTERNAL)
        BASE.require_identity(second_access, BACKEND_EXTERNAL)
        prepare(source_access)
        bind(source_access, 2, 1)
        require(source_access.command(BASE.CMD_EXECUTE_ATOMIC) == ERR_NONE,
                "migration source execute failed")
        require(source_access.read32(REG_ADAPTER_STATE) == ADAPTER_WAITING_A,
                "migration source not pending")

        first_socket = Path("../m")
        require(not (case_root / "m").exists(), "first migration socket exists")
        destination.qmp.execute("migrate-incoming", {"uri": f"unix:{first_socket}"})
        source.qmp.execute("migrate", {"uri": f"unix:{first_socket}"})
        first_status = BASE.wait_source_migration(source.qmp)
        require(source_access.read32(BASE.REG_LIFECYCLE) == BASE.LIFE_SHAM,
                "migration source did not burn lineage in pre-save")
        require(source_access.read32(REG_ADAPTER_STATE) == 7,
                "migration source adapter not SHAM")
        require(source_access.read64(REG_ADAPTER_CANCELS) == 1,
                "migration source did not cancel pending work once")
        require(source_access.read_boundary() ==
                [BASE.LOCKED64] * BASE.BOUNDARY_WORDS,
                "migration source retained a boundary")
        BASE.wait_destination_sham(destination_access)
        require(destination_access.read32(BASE.REG_LIFECYCLE) == BASE.LIFE_SHAM,
                "migration destination not SHAM")
        require(destination_access.read32(REG_ADAPTER_STATE) == 7,
                "migration destination adapter not SHAM")
        require(destination_access.read_boundary() ==
                [BASE.LOCKED64] * BASE.BOUNDARY_WORDS,
                "migration destination leaked boundary")
        late_error = destination_access.command(CMD_POLL_EXTERNAL)
        require(late_error != ERR_NONE, "migration destination accepted completion")

        second_socket = Path("../m2")
        require(not (case_root / "m2").exists(), "second migration socket exists")
        second_destination.qmp.execute(
            "migrate-incoming", {"uri": f"unix:{second_socket}"}
        )
        destination.qmp.execute("migrate", {"uri": f"unix:{second_socket}"})
        second_status = BASE.wait_source_migration(destination.qmp)
        BASE.wait_destination_sham(second_access)
        require(second_access.read32(BASE.REG_LIFECYCLE) == BASE.LIFE_SHAM,
                "second-hop destination not SHAM")
        require(second_access.read_boundary() ==
                [BASE.LOCKED64] * BASE.BOUNDARY_WORDS,
                "second-hop destination leaked boundary")
        reset = second_destination.qmp.execute("system_reset")
        require(reset.get("return") == {}, "post-migration reset failed")
        second_access.setup_pci()
        require(second_access.read32(BASE.REG_LIFECYCLE) == BASE.LIFE_SHAM,
                "post-migration reset escaped SHAM")
        require(second_access.read32(BASE.REG_STATUS_LO) & BASE.ST_SHAM,
                "post-migration reset cleared SHAM status")
        require(second_access.command(BASE.CMD_BEGIN_REUSE) ==
                BASE.ERR_SNAPSHOT_LINEAGE,
                "migrated SHAM authorized BEGIN_REUSE")
        evidence = {
            "fixture": "migration_inflight_latches_sham",
            "transport": "REAL_QMP_UNIX_LIVE_MIGRATION",
            "source_query_migrate_status": first_status,
            "source_pre_save_canceled_once_and_latched_sham": True,
            "second_hop_query_migrate_status": second_status,
            "source_pci": source_pci,
            "destination_pci": destination_pci,
            "second_destination_pci": second_pci,
            "destination_rejected_late_completion": True,
            "second_hop_preserved_sham": True,
            "system_reset_preserved_sham": True,
            "boundary_locked": True,
            "begin_reuse_authorized": False,
        }
    finally:
        source_streams = source.close()
        destination_streams = destination.close()
        second_destination_streams = second_destination.close()
        wall_times[name] = time.monotonic_ns() - started
    BASE.require_empty_streams(source_streams, "migration_source")
    BASE.require_empty_streams(destination_streams, "migration_destination")
    BASE.require_empty_streams(second_destination_streams,
                               "migration_second_destination")
    evidence["captured_process_streams"] = {
        "source": source_streams,
        "destination": destination_streams,
        "second_destination": second_destination_streams,
    }
    return evidence


def pending_process_teardown_case(
    qemu_binary: Path,
    run_root: Path,
    wall_times: dict[str, int],
) -> dict[str, object]:
    name = "pending_process_teardown"
    started = time.monotonic_ns()
    instance = AdapterQemuInstance(qemu_binary, run_root / name, AdapterOptions())
    try:
        instance.start()
        require(instance.qtest is not None and instance.qmp is not None,
                "process-teardown clients unavailable")
        access = BASE.DeviceAccess(instance.qtest, instance.qmp)
        access.setup_pci()
        BASE.require_identity(access, BACKEND_EXTERNAL)
        prepare(access)
        bind(access, 1, 1)
        require(access.command(BASE.CMD_EXECUTE_ATOMIC) == ERR_NONE,
                "process-teardown execute failed")
        require(access.read32(REG_ADAPTER_STATE) == ADAPTER_WAITING_A,
                "process-teardown fixture not pending")
        evidence = {
            "fixture": "pending_process_teardown_with_unrealize_sanitizer",
            "adapter_was_pending_before_qmp_quit": True,
            "qmp_quit_is_process_teardown_not_hot_unplug": True,
            "unrealize_callback_semantics_not_observable_from_qmp_quit": True,
            "static_source_audit_required_for_sanitizer_claim": True,
        }
    finally:
        streams = instance.close()
        wall_times[name] = time.monotonic_ns() - started
    BASE.require_empty_streams(streams, name)
    require(instance.process is not None and instance.process.returncode == 0,
            "pending process teardown did not exit cleanly")
    evidence["process_exit_code"] = instance.process.returncode
    evidence["process_teardown_streams_empty"] = True
    evidence["captured_process_streams"] = streams
    return evidence


def abi_fault_case(access: BASE.DeviceAccess) -> dict[str, object]:
    invalid = BASE.invalid_mmio_width_case(access)
    before = access.read32(BASE.REG_LIFECYCLE)
    ledger_before = access.read64(REG_ADAPTER_AUTH_ACCEPTED)
    access.qtest.write64(BASE.BAR0_BASE + REG_ADAPTER_AUTH_ACCEPTED, 1)
    require(access.read32(BASE.REG_ERROR) == BASE.ERR_BAD_ARGUMENT,
            "write to read-only adapter ledger did not latch BAD_ARGUMENT")
    require(access.read32(BASE.REG_LIFECYCLE) == before,
            "write to read-only adapter ledger mutated lifecycle")
    require(access.read64(REG_ADAPTER_AUTH_ACCEPTED) == ledger_before,
            "write to read-only adapter ledger mutated its value")
    return {
        "fixture": "common_prefix_width_alignment_and_adapter_read_only",
        "invalid_width_alignment": invalid,
        "adapter_read_only_write_error": BASE.ERR_BAD_ARGUMENT,
        "adapter_ledger_value_unchanged": True,
        "lifecycle_unchanged": True,
    }


def build_evidence(qemu_binary: Path, run_root: Path) -> tuple[dict[str, object], dict[str, int]]:
    wall_times: dict[str, int] = {}
    pairs = []
    for residue_a, residue_b in itertools.product(range(3), repeat=2):
        name = f"ideal_pair_{residue_a}_{residue_b}"
        case = run_case(
            qemu_binary,
            run_root,
            name,
            AdapterOptions(),
            lambda access, a=residue_a, b=residue_b: success_case(access, a, b),
            wall_times,
        )
        pairs.append(case["evidence"])

    auth = {}
    for kind in (
        "bad_tag", "wrong_slot", "stale_generation", "slot_b_before_a",
        "expired", "direct_bypass", "replay",
    ):
        name = f"auth_{kind}"
        auth[kind] = run_case(
            qemu_binary,
            run_root,
            name,
            AdapterOptions(),
            lambda access, fixture=kind: auth_negative_case(access, fixture),
            wall_times,
        )["evidence"]

    timeout = run_case(
        qemu_binary, run_root, "async_timeout", AdapterOptions(adapter_mode=1),
        timeout_case, wall_times,
    )["evidence"]
    cancel = run_case(
        qemu_binary, run_root, "async_cancel", AdapterOptions(), cancel_case, wall_times,
    )["evidence"]
    pending_reset = run_case(
        qemu_binary, run_root, "pending_reset", AdapterOptions(),
        pending_reset_case, wall_times,
    )["evidence"]
    migration = migration_case(qemu_binary, run_root, wall_times)
    process_teardown = pending_process_teardown_case(
        qemu_binary, run_root, wall_times
    )
    abi_faults = run_case(
        qemu_binary, run_root, "abi_faults", AdapterOptions(),
        abi_fault_case, wall_times,
    )["evidence"]
    unavailable = run_case(
        qemu_binary, run_root, "service_unavailable",
        AdapterOptions(provider_enabled=False, adapter_enabled=False),
        unavailable_case, wall_times,
    )["evidence"]

    evidence = {
        "schema": "M270_PHASE_QEMU_V12_AUTHENTICATED_ADAPTER_QTEST_EVIDENCE_V1",
        "classification": {
            "phase_qemu_layer_classification":
                "COMPILED_PHASE_QEMU_V12_COMMON_DEVICE_HARDWARE_DISCONNECTED_EXTERNAL_ADAPTER_TEST_STUB",
            "common_guest_visible_device_contract_exercised": True,
            "external_hardware_connected": False,
            "authentication_is_test_only_deterministic_integrity_tag": True,
            "cryptographic_security_claim": False,
            "physical_evidence_class": "NONE",
            "model_return_class": "APPROX_MODEL",
            "restoration_classification": "NO_RESTORATION_CLAIM",
            "begin_reuse_supported": False,
            "architecture_promotion": False,
            "mechanism_promotion": False,
        },
        "build_identity": {
            "qemu_binary_sha256": sha256_file(qemu_binary),
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "base_runner_sha256": sha256_file(BASE_RUNNER),
        },
        "ideal_algebra_pairs": pairs,
        "authentication_controls": auth,
        "asynchronous_controls": {
            "timeout": timeout,
            "cancel": cancel,
            "pending_reset": pending_reset,
        },
        "migration_reset": migration,
        "unrealize": process_teardown,
        "guest_abi_fault_controls": abi_faults,
        "service_mode": unavailable,
        "resource_accounting": {
            "resource_schema": RESOURCE_SCHEMA,
            "unknown_physical_values_use_uint64_max_or_null": True,
            "external_carrier_photon_number_is_unknown": True,
            "device_object_plus_largest_named_local_peak_is_only_a_floor": True,
            "whole_qemu_process_allocator_and_rss_peak": "NOT_ESTABLISHED",
            "model_action_is_not_physical_action": True,
            "unknown_physical_resources": {
                name: {"status": "UNKNOWN", "value": None, "reason": reason}
                for name, reason in (
                    ("adapter_energy", "NO_HARDWARE_ADAPTER_CONNECTED"),
                    ("authentication_energy", "TEST_TAG_HAS_NO_PHYSICAL_METER"),
                    ("calibration", "NO_PHYSICAL_CALIBRATION_EXECUTED"),
                    ("carrier_custody", "NO_PHYSICAL_CARRIER_EXISTS"),
                    ("cryogenic_wall_energy", "NO_HARDWARE_ADAPTER_CONNECTED"),
                    ("physical_action", "ONLY_INTERNAL_MODEL_ACTION_EXECUTED"),
                    ("physical_duration", "ONLY_HOST_WALL_METADATA_EXISTS"),
                    ("rf_dac_fpga_work", "NO_RF_DAC_OR_FPGA_CONNECTED"),
                )
            },
            "total_resource_comparison": "UNDETERMINED",
            "resource_advantage_claim": False,
        },
        "equal_access_comparator": {
            "direct_phase_compiler_has_same_two_residue_accesses": True,
            "all_nine_direct_compilers_executed": True,
            "ideal_algebra_output_equal": all(
                item["direct_compiler_matches_observed_model_phases"] for item in pairs
            ),
            "unique_query_advantage": False,
            "M257_escape": False,
        },
        "checks": {
            "all_nine_pairs": len(pairs) == 9,
            "all_external_results_approx_model": all(
                item["return_class"] == "APPROX_MODEL" for item in pairs
            ),
            "no_reuse": all(
                not item["begin_reuse_authorized"]
                for item in pairs + [timeout, cancel, pending_reset, migration]
            ),
            "all_auth_controls_rejected_before_dispatch": all(
                item["dispatch_delta"] == 0 for item in auth.values()
            ),
            "timeout_and_cancel_terminal": (
                timeout["late_completion_rejected"] and cancel["late_completion_rejected"]
            ),
            "pending_reset_is_irreversible_sham":
                pending_reset["second_reset_preserved_sham"],
            "real_migration_is_second_hop_reset_irreversible_sham": (
                migration["source_pre_save_canceled_once_and_latched_sham"] and
                migration["second_hop_preserved_sham"] and
                migration["system_reset_preserved_sham"]
            ),
            "pending_process_teardown_exits_cleanly": (
                process_teardown["process_exit_code"] == 0 and
                process_teardown["process_teardown_streams_empty"]
            ),
            "guest_abi_fault_controls_pass": (
                abi_faults["lifecycle_unchanged"] and
                abi_faults["adapter_ledger_value_unchanged"]
            ),
            "service_mode_unavailable": unavailable["lease_error"] == ERR_BACKEND_UNAVAILABLE,
            "no_physical_or_advantage_claim": True,
        },
    }
    require(all(evidence["checks"].values()), "one or more M270 checks failed")
    return evidence, wall_times


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qemu-binary", type=Path, required=True)
    parser.add_argument("--scratch-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    qemu_binary = args.qemu_binary.resolve()
    scratch_dir = args.scratch_dir.resolve()
    BASE.validate_inputs(qemu_binary, scratch_dir)
    run_root = scratch_dir / "m270q"
    require(not run_root.exists(), "managed run root already exists")
    run_root.mkdir(mode=0o700)
    evidence, wall_times = build_evidence(qemu_binary, run_root)
    payload = {"deterministic_evidence": evidence, "wall_times_ns": wall_times}
    sys.stdout.buffer.write(canonical_bytes(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
