#!/usr/bin/env python3
"""Bare-metal MMIO experiment against the actual Phase-QEMU V0 PCI device."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


PCI_SLOT = 4
PCI_CONFIG_ADDRESS = 0xCF8
PCI_CONFIG_DATA = 0xCFC
BAR0 = 0xE0000000

REG_MAGIC = 0x000
REG_ABI = 0x004
REG_BACKEND = 0x008
REG_CAPABILITIES = 0x00C
REG_STATUS = 0x010
REG_ERROR = 0x014
REG_GENERATION = 0x018
REG_REINITIALIZATIONS = 0x01C
REG_ARG0 = 0x020
REG_ARG1 = 0x024
REG_COMMAND = 0x028
REG_BARRIER_RECEIPT = 0x02C
REG_VIRTUAL_CYCLES = 0x030
REG_BOUNDARY_I = 0x040
REG_BOUNDARY_Q = 0x048
REG_BOUNDARY_ENERGY = 0x050
REG_RESOURCE_CELLS = 0x058
REG_RETAINED_HISTORY_CELLS = 0x05C
REG_REQUEST_OWNER = 0x060
REG_REQUEST_PROGRAM = 0x064

CMD_LEASE = 1
CMD_PREPARE = 2
CMD_ISOLATE_SOURCE = 3
CMD_EVOLVE = 4
CMD_PROJECT_BOUNDARY = 5
CMD_INVERT = 6
CMD_RESTORE = 7
CMD_REUSE = 8
CMD_RELEASE_DIAGNOSTIC = 9
CMD_CANONICAL_MODEL_REINITIALIZE = 10

ERR_NONE = 0
ERR_PREMATURE_PROJECTION = 3
ERR_P0_INVERSE_UNAVAILABLE = 4
ERR_P0_RESTORATION_UNAVAILABLE = 5
ERR_REUSE_REQUIRES_RESTORATION = 7
ERR_CUSTODY_MISMATCH = 8

ST_CANONICAL = 1 << 0
ST_RESPONSE_READY = 1 << 5
ST_SPENT = 1 << 6
ST_REINITIALIZATION_USED = 1 << 7
ST_CARRIER_PRESENT = 1 << 8

LOCKED_QWORD = (1 << 64) - 1


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def signed64(value: int) -> int:
    return value - (1 << 64) if value & (1 << 63) else value


class QTest:
    def __init__(self, binary: Path, carrier_present: bool = True) -> None:
        present = "on" if carrier_present else "off"
        self.process = subprocess.Popen(
            [
                str(binary),
                "-machine",
                "q35,accel=qtest",
                "-nodefaults",
                "-display",
                "none",
                "-monitor",
                "none",
                "-serial",
                "none",
                "-device",
                f"phase-qemu-v0,addr=04.0,carrier-present={present}",
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
        self._configure_pci_bar()

    def command(self, text: str, expect_value: bool = False) -> int | None:
        assert self.process.stdin is not None
        assert self.process.stdout is not None
        self.process.stdin.write(text + "\n")
        self.process.stdin.flush()
        response = self.process.stdout.readline().strip()
        if not response.startswith("OK"):
            raise RuntimeError(f"qtest command failed: {text!r}: {response!r}")
        if expect_value:
            fields = response.split()
            if len(fields) != 2:
                raise RuntimeError(f"qtest read lacked a value: {response!r}")
            return int(fields[1], 0)
        return None

    def outl(self, port: int, value: int) -> None:
        self.command(f"outl 0x{port:x} 0x{value:x}")

    def outw(self, port: int, value: int) -> None:
        self.command(f"outw 0x{port:x} 0x{value:x}")

    def inl(self, port: int) -> int:
        value = self.command(f"inl 0x{port:x}", expect_value=True)
        assert value is not None
        return value

    def readl(self, offset: int) -> int:
        value = self.command(f"readl 0x{BAR0 + offset:x}", expect_value=True)
        assert value is not None
        return value

    def readq(self, offset: int) -> int:
        value = self.command(f"readq 0x{BAR0 + offset:x}", expect_value=True)
        assert value is not None
        return value

    def writel(self, offset: int, value: int) -> None:
        self.command(f"writel 0x{BAR0 + offset:x} 0x{value:x}")

    def _configure_pci_bar(self) -> None:
        config = 0x80000000 | (PCI_SLOT << 11)
        self.outl(PCI_CONFIG_ADDRESS, config | 0x00)
        identity = self.inl(PCI_CONFIG_DATA)
        if identity != 0x11F01234:
            raise RuntimeError(f"unexpected PCI identity 0x{identity:08x}")
        self.outl(PCI_CONFIG_ADDRESS, config | 0x10)
        self.outl(PCI_CONFIG_DATA, BAR0)
        self.outl(PCI_CONFIG_ADDRESS, config | 0x04)
        self.outw(PCI_CONFIG_DATA, 0x0002)

    def close(self) -> str:
        self.process.terminate()
        try:
            _, stderr = self.process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            _, stderr = self.process.communicate(timeout=5)
        if self.process.returncode not in (-15, 0, None):
            raise RuntimeError(f"QEMU exited {self.process.returncode}: {stderr}")
        return stderr


@dataclass(frozen=True)
class Boundary:
    i: int
    q: int
    energy_q30: int
    barrier_receipt: int
    virtual_cycles: int


def set_arg(device: QTest, arg0: int, arg1: int | None = None) -> None:
    device.writel(REG_ARG0, arg0)
    if arg1 is not None:
        device.writel(REG_ARG1, arg1)


def set_custody(device: QTest, owner: int, program: int) -> None:
    device.writel(REG_REQUEST_OWNER, owner)
    device.writel(REG_REQUEST_PROGRAM, program)


def issue(device: QTest, command: int, expected_error: int = ERR_NONE) -> None:
    device.writel(REG_COMMAND, command)
    observed = device.readl(REG_ERROR)
    if observed != expected_error:
        raise AssertionError(
            f"command {command}: expected error {expected_error}, got {observed}"
        )


def run_arm(
    device: QTest,
    *,
    owner: int,
    program: int,
    phase_arm: int,
    steps: int,
) -> tuple[Boundary, dict[str, bool | int]]:
    set_custody(device, owner, program)
    issue(device, CMD_LEASE)
    set_arg(device, phase_arm)
    issue(device, CMD_PREPARE)
    set_custody(device, owner ^ 0x01000000, program)
    issue(device, CMD_ISOLATE_SOURCE, ERR_CUSTODY_MISMATCH)
    wrong_owner_left_barrier_unchanged = device.readl(REG_BARRIER_RECEIPT) == 0
    set_custody(device, owner, program ^ 0x02000000)
    issue(device, CMD_ISOLATE_SOURCE, ERR_CUSTODY_MISMATCH)
    wrong_program_left_barrier_unchanged = (
        device.readl(REG_BARRIER_RECEIPT) == 0
    )
    set_custody(device, owner, program)
    issue(device, CMD_ISOLATE_SOURCE)
    set_arg(device, steps)
    issue(device, CMD_EVOLVE)
    issue(device, CMD_PROJECT_BOUNDARY)

    locked_before_release = (
        device.readq(REG_BOUNDARY_I) == LOCKED_QWORD
        and device.readq(REG_BOUNDARY_Q) == LOCKED_QWORD
        and device.readq(REG_BOUNDARY_ENERGY) == LOCKED_QWORD
    )
    issue(device, CMD_INVERT, ERR_P0_INVERSE_UNAVAILABLE)
    issue(device, CMD_RESTORE, ERR_P0_RESTORATION_UNAVAILABLE)
    issue(device, CMD_RELEASE_DIAGNOSTIC)
    status = device.readl(REG_STATUS)
    boundary = Boundary(
        i=signed64(device.readq(REG_BOUNDARY_I)),
        q=signed64(device.readq(REG_BOUNDARY_Q)),
        energy_q30=device.readq(REG_BOUNDARY_ENERGY),
        barrier_receipt=device.readl(REG_BARRIER_RECEIPT),
        virtual_cycles=device.readq(REG_VIRTUAL_CYCLES),
    )
    issue(device, CMD_REUSE, ERR_REUSE_REQUIRES_RESTORATION)
    controls = {
        "boundary_locked_until_explicit_diagnostic_release": locked_before_release,
        "response_ready_after_diagnostic_release": bool(status & ST_RESPONSE_READY),
        "carrier_spent_after_nonrestoring_diagnostic": bool(status & ST_SPENT),
        "native_inverse_unavailable": True,
        "native_restoration_unavailable": True,
        "restored_reuse_rejected": True,
        "wrong_owner_rejected_before_source_isolation": wrong_owner_left_barrier_unchanged,
        "wrong_program_rejected_before_source_isolation": (
            wrong_program_left_barrier_unchanged
        ),
    }
    return boundary, controls


def run_projection_before_isolation_control(device: QTest) -> bool:
    set_custody(device, 0x1701, 0x2701)
    issue(device, CMD_LEASE)
    set_arg(device, 0)
    issue(device, CMD_PREPARE)
    issue(device, CMD_PROJECT_BOUNDARY, ERR_PREMATURE_PROJECTION)
    issue(device, CMD_ISOLATE_SOURCE)
    set_arg(device, 8)
    issue(device, CMD_EVOLVE)
    issue(device, CMD_PROJECT_BOUNDARY)
    issue(device, CMD_RELEASE_DIAGNOSTIC)
    return True


def canonical_model_reinitialize(device: QTest) -> dict[str, int | bool]:
    before = device.readl(REG_GENERATION)
    issue(device, CMD_CANONICAL_MODEL_REINITIALIZE)
    after = device.readl(REG_GENERATION)
    status = device.readl(REG_STATUS)
    return {
        "generation_before": before,
        "generation_after": after,
        "reinitialization_count": device.readl(REG_REINITIALIZATIONS),
        "canonical_after_reinitialization": bool(status & ST_CANONICAL),
        "reinitialization_marker_persistent": bool(
            status & ST_REINITIALIZATION_USED
        ),
    }


def experiment(binary: Path) -> dict[str, object]:
    device = QTest(binary, carrier_present=True)
    try:
        identity = {
            "magic": device.readl(REG_MAGIC),
            "abi": device.readl(REG_ABI),
            "backend": device.readl(REG_BACKEND),
            "capabilities": device.readl(REG_CAPABILITIES),
            "model_numeric_state_cells": device.readl(REG_RESOURCE_CELLS),
            "retained_dynamic_inverse_history_cells": device.readl(
                REG_RETAINED_HISTORY_CELLS
            ),
        }
        boundary_0, controls_0 = run_arm(
            device, owner=0x1001, program=0x2001, phase_arm=0, steps=64
        )
        reinitialize_0 = canonical_model_reinitialize(device)
        boundary_pi, controls_pi = run_arm(
            device, owner=0x1002, program=0x2002, phase_arm=1, steps=64
        )
        reinitialize_pi = canonical_model_reinitialize(device)
        projection_before_isolation_rejected = (
            run_projection_before_isolation_control(device)
        )
        controls = {
            **controls_0,
            "pi_arm_has_same_response_ordering": controls_pi == controls_0,
            "projection_before_source_isolation_rejected": (
                projection_before_isolation_rejected
            ),
            "antipodal_i_within_fixed_point_quantization": abs(
                boundary_pi.i + boundary_0.i
            ) <= 16,
            "antipodal_q_within_fixed_point_quantization": abs(
                boundary_pi.q + boundary_0.q
            ) <= 128,
            "matched_energy_within_fixed_point_quantization": abs(
                boundary_pi.energy_q30 - boundary_0.energy_q30
            ) <= 32,
            "barrier_code_8_both_arms": (
                boundary_0.barrier_receipt == 8
                and boundary_pi.barrier_receipt == 8
            ),
            "canonical_model_reinitialization_required_between_arms": (
                reinitialize_0["reinitialization_count"] == 1
                and reinitialize_pi["reinitialization_count"] == 2
            ),
        }
        stderr = device.close()
    except BaseException:
        device.close()
        raise

    removed = QTest(binary, carrier_present=False)
    try:
        removed_boundary, removed_controls = run_arm(
            removed, owner=0x3001, program=0x4001, phase_arm=0, steps=64
        )
        removed_stderr = removed.close()
    except BaseException:
        removed.close()
        raise

    controls["removed_carrier_boundary_strictly_smaller"] = (
        removed_boundary.energy_q30 < boundary_0.energy_q30
    )
    controls["removed_carrier_response_ordering_preserved"] = (
        removed_controls == controls_0
    )
    controls["qemu_stderr_empty"] = stderr == "" and removed_stderr == ""

    if not all(controls.values()):
        failed = [key for key, value in controls.items() if not value]
        raise AssertionError(f"failed controls: {failed}")

    return {
        "schema": "phase-qemu-v0-qtest-result-v1",
        "qemu_binary_sha256": sha256(binary),
        "device": identity,
        "primary_0": boundary_0.__dict__,
        "primary_pi": boundary_pi.__dict__,
        "removed_carrier": removed_boundary.__dict__,
        "canonical_model_reinitialization_after_0": reinitialize_0,
        "canonical_model_reinitialization_after_pi": reinitialize_pi,
        "controls": controls,
        "resource_law": {
            "carrier_model_state": "TWO_Q30_MECHANICAL_ROTATING_FRAME_QUADRATURES",
            "source_state_cells": 2,
            "carrier_state_cells": 2,
            "detector_state_cells": 2,
            "environment_dissipation_cells": 1,
            "boundary_state_cells": 3,
            "declared_numeric_model_cells": 10,
            "controller_custody_boolean_fields": 6,
            "controller_custody_u32_fields": 10,
            "topology_property_and_migration_guard_boolean_fields": 2,
            "barrier_state_fields": 1,
            "clock_state_fields": 2,
            "retained_dynamic_inverse_history_cells": 0,
            "preparation_steps": 256,
            "ringdown_steps_per_primary": 64,
            "virtual_carrier_cycles_per_step": 64,
            "exact_fixed_point_fraction_bits": 30,
            "mmio_read_write_traffic_instrumented": False,
            "fixed_point_primitive_operation_counts_instrumented": False,
            "qemu_build_cost_instrumented": False,
            "migration_snapshot_bytes_and_work_instrumented": False,
            "whole_qemu_process_memory_accounted": False,
        },
        "classification": {
            "verification_level": "PACKAGE_SELF_REVIEW",
            "restoration_classification": "NO_RESTORATION_CLAIM",
            "model_reinitialization_classification": "NO_RESTORATION_CLAIM",
            "claim": "PHASE_QEMU_V0_P0_REFERENCE_PROCESS_GEOMETRY_AND_SOURCE_ISOLATED_UPSTREAM_ENERGIZED_RINGDOWN_CALIBRATION",
            "claim_ceiling": "DETERMINISTIC_FIXED_POINT_QEMU_PCI_MODEL_OF_SELECTED_P0_PROCESS_GEOMETRY_ONLY",
        },
        "boundary_security_scope": {
            "hidden_process_coordinates_absent_from_guest_mmio_reads": True,
            "migration_and_snapshot_streams_are_trusted_backend_state": True,
            "host_or_migration_stream_no_smuggle_enforcement_established": False,
        },
        "claim_limits": {
            "physical_waveform_execution": False,
            "physical_restoration": False,
            "catalytic_restoration": False,
            "restored_carrier_reuse": False,
            "distinct_phase_resource": False,
            "computational_advantage": False,
            "small_wall_crossed": False,
            "physical_bit_replacement": False,
            "unbounded_compute": False,
            "p0_is_final_phase_architecture": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qemu", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = experiment(args.qemu.resolve())
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
