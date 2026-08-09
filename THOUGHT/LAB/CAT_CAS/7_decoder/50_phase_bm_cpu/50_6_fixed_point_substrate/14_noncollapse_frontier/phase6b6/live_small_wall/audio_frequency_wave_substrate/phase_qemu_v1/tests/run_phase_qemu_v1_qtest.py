#!/usr/bin/env python3
"""Drive the Phase-QEMU V1 ideal bosonic backend through qtest MMIO.

The runner receives only the public gate descriptor, final parity boundary,
and non-secret lifecycle/resource receipts.  It never reads carrier or pointer
amplitudes.  Fault controls use fresh QEMU processes because a failed native
inverse lawfully spends that emulated carrier.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


PCI_CONFIG_ADDRESS = 0xCF8
PCI_CONFIG_DATA = 0xCFC
PCI_BDF = 0x80002000
BAR0 = 0xE1000000
LOCKED_QWORD = (1 << 64) - 1

REG_MAGIC = 0x000
REG_ABI = 0x004
REG_BACKEND = 0x008
REG_CAPABILITIES = 0x00C
REG_STATUS = 0x010
REG_ERROR = 0x014
REG_GENERATION = 0x018
REG_RESTORATION_GENERATION = 0x01C
REG_ARG0 = 0x020
REG_ARG1 = 0x024
REG_COMMAND = 0x028
REG_LIFECYCLE = 0x02C
REG_VIRTUAL_CYCLES = 0x030
REG_BOUNDARY_VALUE = 0x040
REG_RESOURCE_COEFFICIENT_CELLS = 0x048
REG_RESOURCE_SCRATCH_CELLS = 0x04C
REG_FORWARD_GATES = 0x050
REG_INVERSE_GATES = 0x058
REG_POINTER_GATES = 0x060
REG_PEAK_COEFFICIENT_BITS = 0x068
REG_REQUEST_OWNER = 0x070
REG_REQUEST_PROGRAM = 0x074
REG_REQUEST_GENERATION = 0x078
REG_DESCRIPTOR_INDEX = 0x07C
REG_DESCRIPTOR_WORD = 0x080
REG_DESCRIPTOR_LENGTH = 0x084
REG_BOUNDARY_MODE = 0x088
REG_DESCRIPTOR_FINGERPRINT = 0x090
REG_FAULT_MODE = 0x098

CMD_LEASE = 1
CMD_PREPARE = 2
CMD_ISOLATE_SOURCE = 3
CMD_SEAL_DESCRIPTOR = 4
CMD_EXECUTE_ATOMIC = 5
CMD_BEGIN_REUSE = 6
CMD_SNAPSHOT = 7

GATE_A = 1
GATE_A_DAG = 2
GATE_B = 3
GATE_B_DAG = 4
GATE_K01 = 5
GATE_K03 = 6
GATE_KERR_IDENTITY_SHAM = 7

ERR_NONE = 0
ERR_BAD_STATE = 1
ERR_BAD_ARGUMENT = 2
ERR_TAG_MISMATCH = 3
ERR_GENERATION_MISMATCH = 4
ERR_DESCRIPTOR_INVALID = 5
ERR_SOURCE_NOT_ISOLATED = 6
ERR_CARRIER_ABSENT = 7
ERR_POINTER_ENTANGLED = 8
ERR_RESTORATION_FAILED = 9
ERR_SNAPSHOT_REJECTED = 12
ERR_RESPONSE_LOCKED = 14

ST_CANONICAL = 1 << 0
ST_LEASED = 1 << 1
ST_PREPARED = 1 << 2
ST_SOURCE_ISOLATED = 1 << 3
ST_DESCRIPTOR_SEALED = 1 << 4
ST_RESPONSE_READY = 1 << 5
ST_SPENT = 1 << 6
ST_RESTORED = 1 << 7
ST_CARRIER_PRESENT = 1 << 8
ST_SNAPSHOT_LINEAGE = 1 << 9
ST_POINTER_CLEAR = 1 << 10

PRIMARY = (GATE_A, GATE_B, GATE_K01, GATE_A, GATE_B)
REUSE = (GATE_B, GATE_A, GATE_K03, GATE_B, GATE_A)
PRIMARY_SHAM = (GATE_A, GATE_B, GATE_KERR_IDENTITY_SHAM, GATE_A, GATE_B)
HELD_OUT = (GATE_A, GATE_B, GATE_K01, GATE_A, GATE_B_DAG)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def signed64(value: int) -> int:
    return value - (1 << 64) if value & (1 << 63) else value


class QTest:
    closed_process_streams: list[tuple[str, str]] = []
    reset_commands_issued = 0

    def __init__(
        self,
        qemu: Path,
        socket_path: Path,
        *,
        carrier_present: bool = True,
        fault_mode: int = 0,
    ) -> None:
        self.socket_path = socket_path
        self.phase_commands: list[int] = []
        self.process = subprocess.Popen(
            [
                "nice",
                "-n",
                "10",
                "ionice",
                "-c",
                "2",
                "-n",
                "7",
                str(qemu),
                "-machine",
                "q35,accel=qtest",
                "-nodefaults",
                "-display",
                "none",
                "-monitor",
                "none",
                "-serial",
                "none",
                "-S",
                "-device",
                (
                    "phase-qemu-v1,addr=04.0,"
                    f"carrier-present={'on' if carrier_present else 'off'},"
                f"test-fault-mode={fault_mode}"
                ),
                "-qtest",
                "stdio",
                "-qtest-log",
                "none",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self.configure_pci()

    def command(self, text: str, *, expect_value: bool = False) -> int | None:
        assert self.process.stdin is not None
        assert self.process.stdout is not None
        if text.strip().split(maxsplit=1)[0] in {"system_reset", "reset"}:
            type(self).reset_commands_issued += 1
        self.process.stdin.write(text + "\n")
        self.process.stdin.flush()
        response = self.process.stdout.readline().strip()
        if response.startswith("FAIL"):
            raise RuntimeError(f"qtest command failed: {text!r}: {response!r}")
        if not response.startswith("OK"):
            raise RuntimeError(f"unexpected qtest response: {response!r}")
        if not expect_value:
            return None
        fields = response.split()
        if len(fields) != 2:
            raise RuntimeError(f"qtest read lacked a value: {response!r}")
        return int(fields[1], 0)

    def configure_pci(self) -> None:
        self.command(f"outl 0x{PCI_CONFIG_ADDRESS:x} 0x{PCI_BDF:x}")
        identity = self.command(f"inl 0x{PCI_CONFIG_DATA:x}", expect_value=True)
        if identity != 0x11F11234:
            raise RuntimeError(f"unexpected Phase-QEMU V1 PCI identity: {identity:#x}")
        self.command(f"outl 0x{PCI_CONFIG_ADDRESS:x} 0x{PCI_BDF + 0x10:x}")
        self.command(f"outl 0x{PCI_CONFIG_DATA:x} 0x{BAR0:x}")
        self.command(f"outl 0x{PCI_CONFIG_ADDRESS:x} 0x{PCI_BDF + 0x04:x}")
        self.command(f"outw 0x{PCI_CONFIG_DATA:x} 0x2")

    def readl(self, offset: int) -> int:
        value = self.command(f"readl 0x{BAR0 + offset:x}", expect_value=True)
        assert value is not None
        return value

    def readq(self, offset: int) -> int:
        value = self.command(f"readq 0x{BAR0 + offset:x}", expect_value=True)
        assert value is not None
        return value

    def writel(self, offset: int, value: int) -> None:
        self.command(f"writel 0x{BAR0 + offset:x} 0x{value & 0xFFFFFFFF:x}")

    def close(self) -> None:
        process = getattr(self, "process", None)
        if process is not None:
            if process.poll() is None:
                process.terminate()
            try:
                remaining_stdout, stderr = process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                remaining_stdout, stderr = process.communicate(timeout=5)
            self.closed_process_streams.append((remaining_stdout, stderr))

    def __enter__(self) -> QTest:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


@dataclass(frozen=True)
class Transaction:
    owner: int
    program: int
    generation: int
    descriptor: tuple[int, ...]
    boundary_mode: int


def set_request(device: QTest, transaction: Transaction) -> None:
    device.writel(REG_REQUEST_OWNER, transaction.owner)
    device.writel(REG_REQUEST_PROGRAM, transaction.program)
    device.writel(REG_REQUEST_GENERATION, transaction.generation)


def issue(device: QTest, command: int, expected: int = ERR_NONE) -> None:
    device.phase_commands.append(command)
    device.writel(REG_COMMAND, command)
    observed = device.readl(REG_ERROR)
    if observed != expected:
        raise AssertionError(
            f"command {command} returned error {observed}, expected {expected}"
        )


def stage_descriptor(device: QTest, transaction: Transaction) -> None:
    for index, gate in enumerate(transaction.descriptor):
        device.writel(REG_DESCRIPTOR_INDEX, index)
        device.writel(REG_DESCRIPTOR_WORD, gate)
    device.writel(REG_DESCRIPTOR_LENGTH, len(transaction.descriptor))
    device.writel(REG_BOUNDARY_MODE, transaction.boundary_mode)
    issue(device, CMD_SEAL_DESCRIPTOR)


def locked_boundary(device: QTest) -> bool:
    return device.readq(REG_BOUNDARY_VALUE) == LOCKED_QWORD


def resource_receipt(device: QTest) -> dict[str, int]:
    return {
        "coefficient_cells": device.readl(REG_RESOURCE_COEFFICIENT_CELLS),
        "scratch_cells": device.readl(REG_RESOURCE_SCRATCH_CELLS),
        "forward_gates": device.readq(REG_FORWARD_GATES),
        "inverse_gates": device.readq(REG_INVERSE_GATES),
        "pointer_gates": device.readq(REG_POINTER_GATES),
        "postcanonical_resident_peak_coefficient_signed_bits": device.readl(
            REG_PEAK_COEFFICIENT_BITS
        ),
        "virtual_cycles": device.readq(REG_VIRTUAL_CYCLES),
    }


def machine_receipt(device: QTest) -> dict[str, int | bool]:
    status = device.readl(REG_STATUS)
    return {
        "status": status,
        "lifecycle": device.readl(REG_LIFECYCLE),
        "generation": device.readl(REG_GENERATION),
        "restoration_generation": device.readl(REG_RESTORATION_GENERATION),
        "descriptor_fingerprint": device.readq(REG_DESCRIPTOR_FINGERPRINT),
        "boundary_locked": locked_boundary(device),
        "forward_gates": device.readq(REG_FORWARD_GATES),
        "inverse_gates": device.readq(REG_INVERSE_GATES),
        "pointer_gates": device.readq(REG_POINTER_GATES),
    }


def initial_prepare(device: QTest, transaction: Transaction) -> None:
    set_request(device, transaction)
    issue(device, CMD_LEASE)
    issue(device, CMD_PREPARE)
    issue(device, CMD_ISOLATE_SOURCE)


def execute_transaction(
    device: QTest,
    transaction: Transaction,
    *,
    reuse: bool = False,
) -> dict[str, object]:
    if reuse:
        issue(device, CMD_BEGIN_REUSE)
        set_request(device, transaction)
        issue(device, CMD_LEASE)
    else:
        initial_prepare(device, transaction)
    stage_descriptor(device, transaction)
    if not locked_boundary(device):
        raise AssertionError("boundary became readable before atomic execution")
    descriptor_fingerprint = device.readq(REG_DESCRIPTOR_FINGERPRINT)
    issue(device, CMD_EXECUTE_ATOMIC)
    error = device.readl(REG_ERROR)
    status = device.readl(REG_STATUS)
    boundary = device.readq(REG_BOUNDARY_VALUE)
    return {
        "error": error,
        "status": status,
        "lifecycle": device.readl(REG_LIFECYCLE),
        "generation": device.readl(REG_GENERATION),
        "restoration_generation": device.readl(REG_RESTORATION_GENERATION),
        "boundary_parity_bit": boundary if boundary != LOCKED_QWORD else None,
        "boundary_pointer_z": (
            1 - 2 * boundary if boundary != LOCKED_QWORD else None
        ),
        "boundary_locked": boundary == LOCKED_QWORD,
        "descriptor_fingerprint": descriptor_fingerprint,
        "resources": resource_receipt(device),
        "canonical": bool(status & ST_CANONICAL),
        "response_ready": bool(status & ST_RESPONSE_READY),
        "restored": bool(status & ST_RESTORED),
        "spent": bool(status & ST_SPENT),
        "pointer_clear": bool(status & ST_POINTER_CLEAR),
        "snapshot_lineage": bool(status & ST_SNAPSHOT_LINEAGE),
    }


def run_success_suite(qemu: Path, scratch: Path) -> dict[str, object]:
    primary = Transaction(0x1101, 0x2101, 1, PRIMARY, 3)
    reuse = Transaction(0x1102, 0x2102, 2, REUSE, 1)
    plus = Transaction(0x1103, 0x2103, 1, PRIMARY, 2)
    held_out = Transaction(0x1104, 0x2104, 1, HELD_OUT, 0)
    with QTest(qemu, scratch / "success.qtest") as device:
        success_process_id = device.process.pid
        abi = {
            "magic": device.readl(REG_MAGIC),
            "abi": device.readl(REG_ABI),
            "backend": device.readl(REG_BACKEND),
            "capabilities": device.readl(REG_CAPABILITIES),
            "fault_mode": device.readl(REG_FAULT_MODE),
        }
        locked_at_reset = locked_boundary(device)
        primary_result = execute_transaction(device, primary)
        reuse_result = execute_transaction(device, reuse, reuse=True)
        same_process_primary_reuse = device.process.pid == success_process_id
        before_snapshot_command = machine_receipt(device)
        set_request(device, reuse)
        issue(device, CMD_SNAPSHOT, ERR_SNAPSHOT_REJECTED)
        snapshot_rejected_without_machine_state_change = (
            machine_receipt(device) == before_snapshot_command
        )
        primary_reuse_command_counts = {
            "lease": device.phase_commands.count(CMD_LEASE),
            "prepare": device.phase_commands.count(CMD_PREPARE),
            "isolate_source": device.phase_commands.count(CMD_ISOLATE_SOURCE),
            "seal_descriptor": device.phase_commands.count(CMD_SEAL_DESCRIPTOR),
            "execute_atomic": device.phase_commands.count(CMD_EXECUTE_ATOMIC),
            "begin_reuse": device.phase_commands.count(CMD_BEGIN_REUSE),
            "snapshot_reject_control": device.phase_commands.count(CMD_SNAPSHOT),
        }

    with QTest(qemu, scratch / "plus.qtest") as plus_device:
        plus_result = execute_transaction(plus_device, plus)
    with QTest(qemu, scratch / "held-out.qtest") as held_out_device:
        held_out_result = execute_transaction(held_out_device, held_out)

    if not (
        primary_result["boundary_parity_bit"] == 1
        and primary_result["boundary_pointer_z"] == -1
        and primary_result["generation"] == 1
        and primary_result["restoration_generation"] == 1
        and primary_result["canonical"]
        and primary_result["restored"]
        and primary_result["response_ready"]
        and primary_result["pointer_clear"]
        and not primary_result["spent"]
        and not primary_result["snapshot_lineage"]
        and primary_result["resources"]
        == {
            "coefficient_cells": 20,
            "scratch_cells": 20,
            "forward_gates": 5,
            "inverse_gates": 5,
            "pointer_gates": 2,
            "postcanonical_resident_peak_coefficient_signed_bits": 2,
            "virtual_cycles": 12,
        }
    ):
        raise AssertionError(f"primary transaction law failed: {primary_result}")
    if not (
        reuse_result["boundary_parity_bit"] == 1
        and reuse_result["boundary_pointer_z"] == -1
        and reuse_result["generation"] == 2
        and reuse_result["restoration_generation"] == 2
        and reuse_result["canonical"]
        and reuse_result["restored"]
        and reuse_result["response_ready"]
        and reuse_result["pointer_clear"]
        and not reuse_result["spent"]
        and not reuse_result["snapshot_lineage"]
        and reuse_result["descriptor_fingerprint"]
        != primary_result["descriptor_fingerprint"]
        and reuse_result["resources"]
        == {
            "coefficient_cells": 20,
            "scratch_cells": 20,
            "forward_gates": 10,
            "inverse_gates": 10,
            "pointer_gates": 4,
            "postcanonical_resident_peak_coefficient_signed_bits": 2,
            "virtual_cycles": 24,
        }
    ):
        raise AssertionError(f"reuse transaction law failed: {reuse_result}")
    if not (
        plus_result["boundary_parity_bit"] == 0
        and plus_result["boundary_pointer_z"] == 1
        and plus_result["canonical"]
        and plus_result["restored"]
    ):
        raise AssertionError(f"valid +1 selector failed: {plus_result}")
    if not (
        held_out_result["boundary_parity_bit"] in (0, 1)
        and held_out_result["canonical"]
        and held_out_result["restored"]
        and held_out_result["response_ready"]
        and held_out_result["resources"]["forward_gates"] == len(HELD_OUT)
        and held_out_result["resources"]["inverse_gates"] == len(HELD_OUT)
    ):
        raise AssertionError(f"held-out descriptor failed: {held_out_result}")
    if not (
        locked_at_reset
        and same_process_primary_reuse
        and snapshot_rejected_without_machine_state_change
        and QTest.reset_commands_issued == 0
        and primary_reuse_command_counts
        == {
            "lease": 2,
            "prepare": 1,
            "isolate_source": 1,
            "seal_descriptor": 2,
            "execute_atomic": 2,
            "begin_reuse": 1,
            "snapshot_reject_control": 1,
        }
    ):
        raise AssertionError("reset/same-process/snapshot controls failed")

    return {
        "abi": abi,
        "locked_at_reset": locked_at_reset,
        "primary": primary_result,
        "descriptor_distinct_reuse": reuse_result,
        "valid_plus_one_selector": plus_result,
        "held_out_public_descriptor": held_out_result,
        "same_qemu_process_primary_and_reuse": same_process_primary_reuse,
        "snapshot_command_rejected_without_machine_state_change": (
            snapshot_rejected_without_machine_state_change
        ),
        "qtest_reset_commands_issued": QTest.reset_commands_issued,
        "same_device_primary_reuse_phase_command_counts": (
            primary_reuse_command_counts
        ),
    }


def run_tag_controls(qemu: Path, scratch: Path) -> dict[str, bool]:
    transaction = Transaction(0x1201, 0x2201, 1, PRIMARY, 3)
    with QTest(qemu, scratch / "tags.qtest") as device:
        set_request(device, transaction)
        before_wrong_owner = machine_receipt(device)
        issue(device, CMD_LEASE)
        device.writel(REG_REQUEST_OWNER, transaction.owner + 1)
        issue(device, CMD_PREPARE, ERR_TAG_MISMATCH)
        after_wrong_owner = machine_receipt(device)
        wrong_owner_rejected = (
            after_wrong_owner["generation"] == before_wrong_owner["generation"]
            and after_wrong_owner["restoration_generation"]
            == before_wrong_owner["restoration_generation"]
            and after_wrong_owner["forward_gates"] == 0
            and after_wrong_owner["inverse_gates"] == 0
            and after_wrong_owner["boundary_locked"]
        )
        set_request(device, transaction)
        issue(device, CMD_PREPARE)
        issue(device, CMD_ISOLATE_SOURCE)
        stage_descriptor(device, transaction)
        before_wrong_program = machine_receipt(device)
        device.writel(REG_REQUEST_PROGRAM, transaction.program + 1)
        issue(device, CMD_EXECUTE_ATOMIC, ERR_TAG_MISMATCH)
        after_wrong_program = machine_receipt(device)
        wrong_program_rejected = after_wrong_program == before_wrong_program
        set_request(device, transaction)
        before_wrong_generation = machine_receipt(device)
        device.writel(REG_REQUEST_GENERATION, 2)
        issue(device, CMD_EXECUTE_ATOMIC, ERR_GENERATION_MISMATCH)
        after_wrong_generation = machine_receipt(device)
        wrong_generation_rejected = (
            after_wrong_generation == before_wrong_generation
        )
        set_request(device, transaction)
        issue(device, CMD_EXECUTE_ATOMIC)
        recovery_succeeds = device.readq(REG_BOUNDARY_VALUE) != LOCKED_QWORD
    return {
        "wrong_owner_rejected_before_prepare": wrong_owner_rejected,
        "wrong_program_rejected_before_execution": wrong_program_rejected,
        "wrong_generation_rejected_before_execution": wrong_generation_rejected,
        "valid_request_after_rejections_succeeds": recovery_succeeds,
    }


def run_descriptor_controls(qemu: Path, scratch: Path) -> dict[str, bool]:
    transaction = Transaction(0x1301, 0x2301, 1, PRIMARY, 3)
    with QTest(qemu, scratch / "descriptor.qtest") as device:
        initial_prepare(device, transaction)
        stage_descriptor(device, transaction)
        fingerprint = device.readq(REG_DESCRIPTOR_FINGERPRINT)
        device.writel(REG_DESCRIPTOR_INDEX, 0)
        device.writel(REG_DESCRIPTOR_WORD, GATE_B)
        mutation_rejected = (
            device.readl(REG_ERROR) == ERR_BAD_STATE
            and device.readq(REG_DESCRIPTOR_FINGERPRINT) == fingerprint
        )
        issue(device, CMD_EXECUTE_ATOMIC)
        sealed_descriptor_still_executes = device.readq(REG_BOUNDARY_VALUE) != LOCKED_QWORD

    invalid_mode = Transaction(0x1302, 0x2302, 1, PRIMARY, 4)
    with QTest(qemu, scratch / "bad-mode.qtest") as device:
        initial_prepare(device, invalid_mode)
        for index, gate in enumerate(invalid_mode.descriptor):
            device.writel(REG_DESCRIPTOR_INDEX, index)
            device.writel(REG_DESCRIPTOR_WORD, gate)
        device.writel(REG_DESCRIPTOR_LENGTH, len(invalid_mode.descriptor))
        device.writel(REG_BOUNDARY_MODE, invalid_mode.boundary_mode)
        invalid_boundary_mode_rejected = (
            device.readl(REG_ERROR) == ERR_DESCRIPTOR_INVALID
            and locked_boundary(device)
            and not (device.readl(REG_STATUS) & ST_DESCRIPTOR_SEALED)
        )

    unknown_gate = Transaction(0x1303, 0x2303, 1, (99,), 0)
    with QTest(qemu, scratch / "unknown-gate.qtest") as device:
        initial_prepare(device, unknown_gate)
        device.writel(REG_DESCRIPTOR_INDEX, 0)
        device.writel(REG_DESCRIPTOR_WORD, 99)
        device.writel(REG_DESCRIPTOR_LENGTH, 1)
        device.writel(REG_BOUNDARY_MODE, 0)
        issue(device, CMD_SEAL_DESCRIPTOR, ERR_DESCRIPTOR_INVALID)
        unknown_gate_rejected = locked_boundary(device)

    hole = Transaction(0x1304, 0x2304, 1, (GATE_A, GATE_B), 0)
    with QTest(qemu, scratch / "descriptor-hole.qtest") as device:
        initial_prepare(device, hole)
        device.writel(REG_DESCRIPTOR_INDEX, 1)
        device.writel(REG_DESCRIPTOR_WORD, GATE_B)
        device.writel(REG_DESCRIPTOR_LENGTH, 2)
        device.writel(REG_BOUNDARY_MODE, 0)
        issue(device, CMD_SEAL_DESCRIPTOR, ERR_DESCRIPTOR_INVALID)
        descriptor_hole_rejected = locked_boundary(device)

    overlength = Transaction(0x1305, 0x2305, 1, PRIMARY, 0)
    with QTest(qemu, scratch / "overlength.qtest") as device:
        initial_prepare(device, overlength)
        device.writel(REG_DESCRIPTOR_LENGTH, 17)
        device.writel(REG_BOUNDARY_MODE, 0)
        issue(device, CMD_SEAL_DESCRIPTOR, ERR_DESCRIPTOR_INVALID)
        descriptor_overlength_rejected = locked_boundary(device)
    return {
        "post_seal_descriptor_mutation_rejected": mutation_rejected,
        "sealed_descriptor_remained_effective": sealed_descriptor_still_executes,
        "invalid_boundary_mode_rejected": invalid_boundary_mode_rejected,
        "unknown_gate_rejected": unknown_gate_rejected,
        "descriptor_hole_rejected": descriptor_hole_rejected,
        "descriptor_overlength_rejected": descriptor_overlength_rejected,
    }


def run_sham_control(qemu: Path, scratch: Path) -> dict[str, object]:
    transaction = Transaction(0x1401, 0x2401, 1, PRIMARY_SHAM, 3)
    reuse = Transaction(0x1402, 0x2402, 2, REUSE, 1)
    with QTest(qemu, scratch / "sham.qtest") as device:
        initial_prepare(device, transaction)
        stage_descriptor(device, transaction)
        issue(device, CMD_EXECUTE_ATOMIC, ERR_POINTER_ENTANGLED)
        status = device.readl(REG_STATUS)
        sham_result = {
            "pointer_entanglement_rejected": True,
            "boundary_locked": locked_boundary(device),
            "carrier_restored_after_result_free_unwind": bool(status & ST_RESTORED),
            "canonical_after_result_free_unwind": bool(status & ST_CANONICAL),
            "spent": bool(status & ST_SPENT),
            "generation": device.readl(REG_GENERATION),
            "resources": resource_receipt(device),
        }
        reuse_result = execute_transaction(device, reuse, reuse=True)
        sham_result["same_carrier_generation2_reuse_after_result_free_unwind"] = {
            "boundary_parity_bit": reuse_result["boundary_parity_bit"],
            "generation": reuse_result["generation"],
            "restoration_generation": reuse_result["restoration_generation"],
            "canonical": reuse_result["canonical"],
            "restored": reuse_result["restored"],
            "pointer_clear": reuse_result["pointer_clear"],
            "response_ready": reuse_result["response_ready"],
            "snapshot_lineage": reuse_result["snapshot_lineage"],
            "resources": reuse_result["resources"],
        }
        return sham_result


def run_fault_control(
    qemu: Path,
    scratch: Path,
    fault_mode: int,
    label: str,
) -> dict[str, object]:
    transaction = Transaction(0x1500 + fault_mode, 0x2500 + fault_mode, 1, PRIMARY, 3)
    with QTest(
        qemu,
        scratch / f"fault-{fault_mode}.qtest",
        fault_mode=fault_mode,
    ) as device:
        initial_prepare(device, transaction)
        stage_descriptor(device, transaction)
        issue(device, CMD_EXECUTE_ATOMIC, ERR_RESTORATION_FAILED)
        status = device.readl(REG_STATUS)
        return {
            "label": label,
            "restoration_failure_detected": True,
            "boundary_locked": locked_boundary(device),
            "spent": bool(status & ST_SPENT),
            "restored": bool(status & ST_RESTORED),
            "canonical": bool(status & ST_CANONICAL),
            "pointer_clear": bool(status & ST_POINTER_CLEAR),
            "response_ready": bool(status & ST_RESPONSE_READY),
            "generation": device.readl(REG_GENERATION),
            "restoration_generation": device.readl(REG_RESTORATION_GENERATION),
            "resources": resource_receipt(device),
        }


def run_null_carrier_control(qemu: Path, scratch: Path) -> bool:
    transaction = Transaction(0x1601, 0x2601, 1, PRIMARY, 3)
    with QTest(
        qemu,
        scratch / "null.qtest",
        carrier_present=False,
    ) as device:
        set_request(device, transaction)
        issue(device, CMD_LEASE, ERR_CARRIER_ABSENT)
        return locked_boundary(device) and device.readl(REG_GENERATION) == 1


def experiment(qemu: Path, scratch: Path) -> dict[str, object]:
    QTest.closed_process_streams = []
    QTest.reset_commands_issued = 0
    scratch.mkdir(parents=True, exist_ok=False)
    success = run_success_suite(qemu, scratch)
    controls = {
        **run_tag_controls(qemu, scratch),
        **run_descriptor_controls(qemu, scratch),
        "null_carrier_rejected": run_null_carrier_control(qemu, scratch),
    }
    sham = run_sham_control(qemu, scratch)
    faults = {
        "missing_inverse": run_fault_control(qemu, scratch, 1, "MISSING_INVERSE"),
        "wrong_inverse": run_fault_control(qemu, scratch, 2, "WRONG_KERR_EDGE"),
        "reordered_inverse": run_fault_control(
            qemu, scratch, 3, "FORWARD_ORDER_ADJOINTS"
        ),
    }
    if not (
        sham["pointer_entanglement_rejected"]
        and sham["boundary_locked"]
        and sham["carrier_restored_after_result_free_unwind"]
        and sham["canonical_after_result_free_unwind"]
        and not sham["spent"]
        and sham["generation"] == 1
        and sham["resources"]
        == {
            "coefficient_cells": 20,
            "scratch_cells": 20,
            "forward_gates": 5,
            "inverse_gates": 5,
            "pointer_gates": 2,
            "postcanonical_resident_peak_coefficient_signed_bits": 2,
            "virtual_cycles": 12,
        }
        and sham["same_carrier_generation2_reuse_after_result_free_unwind"]
        == {
            "boundary_parity_bit": 1,
            "generation": 2,
            "restoration_generation": 2,
            "canonical": True,
            "restored": True,
            "pointer_clear": True,
            "response_ready": True,
            "snapshot_lineage": False,
            "resources": {
                "coefficient_cells": 20,
                "scratch_cells": 20,
                "forward_gates": 10,
                "inverse_gates": 10,
                "pointer_gates": 4,
                "postcanonical_resident_peak_coefficient_signed_bits": 2,
                "virtual_cycles": 24,
            },
        }
    ):
        raise AssertionError(f"Kerr-disabled sham law failed: {sham}")
    if not all(controls.values()):
        raise AssertionError(f"one or more MMIO controls failed: {controls}")
    if not all(
        item["restoration_failure_detected"]
        and item["boundary_locked"]
        and item["spent"]
        and not item["restored"]
        and not item["canonical"]
        and item["pointer_clear"]
        and not item["response_ready"]
        and item["restoration_generation"] == 0
        and item["resources"]["coefficient_cells"] == 20
        and item["resources"]["scratch_cells"] == 20
        and item["resources"]["forward_gates"] == 5
        and item["resources"]["pointer_gates"] == 2
        and item["resources"][
            "postcanonical_resident_peak_coefficient_signed_bits"
        ]
        == 2
        for item in faults.values()
    ):
        raise AssertionError(f"one or more inverse fault controls failed: {faults}")
    if not (
        faults["missing_inverse"]["resources"]["inverse_gates"] == 0
        and faults["missing_inverse"]["resources"]["virtual_cycles"] == 7
        and faults["wrong_inverse"]["resources"]["inverse_gates"] == 5
        and faults["wrong_inverse"]["resources"]["virtual_cycles"] == 12
        and faults["reordered_inverse"]["resources"]["inverse_gates"] == 5
        and faults["reordered_inverse"]["resources"]["virtual_cycles"] == 12
    ):
        raise AssertionError(f"inverse fault resource law failed: {faults}")
    nonempty_streams = [
        {"stdout": stdout, "stderr": stderr}
        for stdout, stderr in QTest.closed_process_streams
        if stdout or stderr
    ]
    if nonempty_streams:
        raise AssertionError(f"unexpected QEMU process output: {nonempty_streams}")
    return {
        "schema": "phase-qemu-v1-qtest-result-v1",
        "qemu_binary_sha256": sha256(qemu),
        "success": success,
        "controls": controls,
        "kerr_disabled_sham": sham,
        "inverse_fault_controls": faults,
        "qemu_stdout_stderr_no_smuggle": True,
        "resource_law": {
            "carrier_pointer_coefficient_cells": 20,
            "gate_scratch_coefficient_cells": 20,
            "coefficient_cell_counts_are_allocated_backing_counts": True,
            "denominator_power_scalar_cells": 1,
            "retained_private_boundary_bit_cells_during_inverse": 1,
            "public_descriptor_allocated_uint32_cells": 16,
            "postcanonical_resident_peak_coefficient_width_instrumented": True,
            "transient_and_whole_process_live_payload_instrumented": False,
            "canonicalization_factorization_norm_and_verification_work_instrumented": False,
            "public_descriptor_gate_cells_primary": len(PRIMARY),
            "retained_dynamic_inverse_history_cells": 0,
            "accepted_inverse_compiled_from_public_forward_descriptor": True,
            "migration_state_is_trusted_backend_sham_not_controller_boundary": True,
            "whole_process_memory_and_qemu_runtime_state_accounted": False,
            "mmio_traffic_count_instrumented": False,
            "physical_energy_latency_noise_and_control_precision_modeled": False,
            "resource_measurement_verification_level": "PACKAGE_SELF_REVIEW",
        },
        "claim": "PHASE_QEMU_V1_BOUNDED_EXACT_FOUR_MODE_TWO_BOSON_CROSS_KERR_QND_PARITY_POINTER_PUBLIC_ADJOINT_RESTORATION_AND_GENERATION2_REUSE",
        "claim_ceiling": "DETERMINISTIC_EXACT_IDEAL_QEMU_4_MODE_2_BOSON_BACKEND_ONLY",
        "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "claim_limits": {
            "physical_bosons_or_phonons": False,
            "physical_qnd_detector": False,
            "authenticated_custody": False,
            "m257_escape": False,
            "distinct_phase_resource": False,
            "computational_advantage": False,
            "small_wall_crossing": False,
            "unbounded_compute": False,
            "replacement_of_physical_bits_with_pi": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qemu", required=True, type=Path)
    parser.add_argument("--scratch", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = experiment(args.qemu.resolve(), args.scratch.resolve())
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.write_text(encoded, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
