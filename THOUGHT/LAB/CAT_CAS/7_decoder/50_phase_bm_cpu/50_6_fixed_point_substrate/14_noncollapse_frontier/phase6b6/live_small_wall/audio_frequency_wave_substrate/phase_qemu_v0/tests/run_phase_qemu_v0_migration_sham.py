#!/usr/bin/env python3
"""Exercise a real QEMU migration stream as a non-catalytic reload sham."""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import subprocess
import time
from pathlib import Path


PCI_SLOT = 4
PCI_CONFIG_ADDRESS = 0xCF8
PCI_CONFIG_DATA = 0xCFC
BAR0 = 0xE0000000

REG_MAGIC = 0x000
REG_STATUS = 0x010
REG_ERROR = 0x014
REG_GENERATION = 0x018
REG_ARG0 = 0x020
REG_COMMAND = 0x028
REG_BARRIER_RECEIPT = 0x02C
REG_VIRTUAL_CYCLES = 0x030
REG_BOUNDARY_I = 0x040
REG_BOUNDARY_Q = 0x048
REG_BOUNDARY_ENERGY = 0x050
REG_REQUEST_OWNER = 0x060
REG_REQUEST_PROGRAM = 0x064

CMD_LEASE = 1
CMD_PREPARE = 2
CMD_ISOLATE_SOURCE = 3
CMD_EVOLVE = 4
CMD_PROJECT_BOUNDARY = 5
CMD_RELEASE_DIAGNOSTIC = 9

ERR_NONE = 0
ST_LEASED = 1 << 1
ST_PREPARED = 1 << 2
ST_SOURCE_ISOLATED = 1 << 3
ST_RESPONSE_READY = 1 << 5
ST_SPENT = 1 << 6
LOCKED_QWORD = (1 << 64) - 1


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_managed_scratch(path: Path) -> Path:
    resolved = path.resolve()
    text = str(resolved)
    if "/Codex/Scratch/turns/" not in text:
        raise RuntimeError("migration sham requires codex-scratch managed storage")
    if text.startswith("/dev/shm/") or text.startswith("/run/shm/"):
        raise RuntimeError("RAM-backed scratch is forbidden")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def signed64(value: int) -> int:
    return value - (1 << 64) if value & (1 << 63) else value


class QmpQtest:
    def __init__(
        self,
        binary: Path,
        abstract_socket: str,
        *,
        incoming: Path | None = None,
    ) -> None:
        command = [
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
            "phase-qemu-v0,addr=04.0,carrier-present=on",
            "-qtest",
            f"unix:{abstract_socket},abstract=on,server=on,wait=off",
            "-qtest-log",
            "none",
            "-qmp",
            "stdio",
        ]
        if incoming is not None:
            command.extend(["-incoming", f"file:{incoming}"])
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        assert self.process.stdin is not None
        assert self.process.stdout is not None
        greeting = json.loads(self.process.stdout.readline())
        if "QMP" not in greeting:
            raise RuntimeError(f"missing QMP greeting: {greeting!r}")
        self.qmp("qmp_capabilities")
        self.qtest_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        address = "\0" + abstract_socket
        for _ in range(200):
            try:
                self.qtest_socket.connect(address)
                break
            except ConnectionRefusedError:
                time.sleep(0.01)
        else:
            raise RuntimeError("timed out connecting to abstract qtest socket")
        self.qtest_stream = self.qtest_socket.makefile("rw", buffering=1)

    def qmp(self, execute: str, arguments: dict[str, object] | None = None) -> dict[str, object]:
        assert self.process.stdin is not None
        assert self.process.stdout is not None
        request: dict[str, object] = {"execute": execute}
        if arguments is not None:
            request["arguments"] = arguments
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        while True:
            response = json.loads(self.process.stdout.readline())
            if "event" in response:
                continue
            if "error" in response:
                raise RuntimeError(f"QMP {execute} failed: {response['error']!r}")
            return response.get("return", {})

    def qtest(self, command: str, expect_value: bool = False) -> int | None:
        self.qtest_stream.write(command + "\n")
        response = self.qtest_stream.readline().strip()
        if not response.startswith("OK"):
            raise RuntimeError(f"qtest failed: {command!r}: {response!r}")
        if expect_value:
            fields = response.split()
            if len(fields) != 2:
                raise RuntimeError(f"qtest read lacked a value: {response!r}")
            return int(fields[1], 0)
        return None

    def outl(self, port: int, value: int) -> None:
        self.qtest(f"outl 0x{port:x} 0x{value:x}")

    def outw(self, port: int, value: int) -> None:
        self.qtest(f"outw 0x{port:x} 0x{value:x}")

    def inl(self, port: int) -> int:
        value = self.qtest(f"inl 0x{port:x}", expect_value=True)
        assert value is not None
        return value

    def readl(self, offset: int) -> int:
        value = self.qtest(f"readl 0x{BAR0 + offset:x}", expect_value=True)
        assert value is not None
        return value

    def readq(self, offset: int) -> int:
        value = self.qtest(f"readq 0x{BAR0 + offset:x}", expect_value=True)
        assert value is not None
        return value

    def writel(self, offset: int, value: int) -> None:
        self.qtest(f"writel 0x{BAR0 + offset:x} 0x{value:x}")

    def configure_pci_bar(self) -> None:
        config = 0x80000000 | (PCI_SLOT << 11)
        self.outl(PCI_CONFIG_ADDRESS, config | 0x00)
        if self.inl(PCI_CONFIG_DATA) != 0x11F01234:
            raise RuntimeError("unexpected Phase-QEMU PCI identity")
        self.outl(PCI_CONFIG_ADDRESS, config | 0x10)
        self.outl(PCI_CONFIG_DATA, BAR0)
        self.outl(PCI_CONFIG_ADDRESS, config | 0x04)
        self.outw(PCI_CONFIG_DATA, 0x0002)

    def migrate_to_file(self, path: Path) -> None:
        self.qmp("migrate", {"uri": f"file:{path}"})
        for _ in range(1000):
            state = self.qmp("query-migrate")
            status = state.get("status")
            if status == "completed":
                return
            if status in {"failed", "cancelled"}:
                raise RuntimeError(f"migration failed: {state!r}")
            time.sleep(0.01)
        raise RuntimeError("migration did not complete")

    def close(self) -> str:
        try:
            self.qmp("quit")
        except (BrokenPipeError, RuntimeError, json.JSONDecodeError):
            self.process.terminate()
        try:
            _, stderr = self.process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            _, stderr = self.process.communicate(timeout=5)
        return stderr


def issue(device: QmpQtest, command: int, expected_error: int = ERR_NONE) -> None:
    device.writel(REG_COMMAND, command)
    error = device.readl(REG_ERROR)
    if error != expected_error:
        raise AssertionError(
            f"command {command}: expected error {expected_error}, got {error}"
        )


def set_custody(device: QmpQtest, owner: int, program: int) -> None:
    device.writel(REG_REQUEST_OWNER, owner)
    device.writel(REG_REQUEST_PROGRAM, program)


def set_arg(device: QmpQtest, value: int) -> None:
    device.writel(REG_ARG0, value)


def experiment(binary: Path, scratch: Path, qtest_result: Path) -> dict[str, object]:
    expected = json.loads(qtest_result.read_text(encoding="utf-8"))["primary_0"]
    migration = scratch / "phase-qemu-v0-p0-midstate.migration"
    owner = 0x5101
    program = 0x6101

    source = QmpQtest(binary, "phase-qemu-v0-migration-source")
    try:
        source.configure_pci_bar()
        set_custody(source, owner, program)
        issue(source, CMD_LEASE)
        set_arg(source, 0)
        issue(source, CMD_PREPARE)
        issue(source, CMD_ISOLATE_SOURCE)
        set_arg(source, 16)
        issue(source, CMD_EVOLVE)
        source_status = source.readl(REG_STATUS)
        source_prefix = {
            "status": source_status,
            "generation": source.readl(REG_GENERATION),
            "barrier_receipt": source.readl(REG_BARRIER_RECEIPT),
            "virtual_cycles": source.readq(REG_VIRTUAL_CYCLES),
            "boundary_locked": source.readq(REG_BOUNDARY_I) == LOCKED_QWORD,
        }
        source.migrate_to_file(migration)
        source_stderr = source.close()
    except BaseException:
        source.close()
        raise

    target = QmpQtest(
        binary,
        "phase-qemu-v0-migration-target",
        incoming=migration,
    )
    try:
        target_status = target.readl(REG_STATUS)
        target_prefix = {
            "status": target_status,
            "generation": target.readl(REG_GENERATION),
            "barrier_receipt": target.readl(REG_BARRIER_RECEIPT),
            "virtual_cycles": target.readq(REG_VIRTUAL_CYCLES),
            "boundary_locked": target.readq(REG_BOUNDARY_I) == LOCKED_QWORD,
        }
        restored_bar_without_reconfiguration = target.readl(REG_MAGIC) == 0x50485630
        set_custody(target, owner, program)
        set_arg(target, 48)
        issue(target, CMD_EVOLVE)
        issue(target, CMD_PROJECT_BOUNDARY)
        locked_before_release = target.readq(REG_BOUNDARY_I) == LOCKED_QWORD
        issue(target, CMD_RELEASE_DIAGNOSTIC)
        target_status_after = target.readl(REG_STATUS)
        boundary = {
            "i": signed64(target.readq(REG_BOUNDARY_I)),
            "q": signed64(target.readq(REG_BOUNDARY_Q)),
            "energy_q30": target.readq(REG_BOUNDARY_ENERGY),
            "barrier_receipt": target.readl(REG_BARRIER_RECEIPT),
            "virtual_cycles": target.readq(REG_VIRTUAL_CYCLES),
        }
        target_stderr = target.close()
    except BaseException:
        target.close()
        raise

    controls = {
        "source_prefix_is_leased_prepared_isolated": (
            source_status & (ST_LEASED | ST_PREPARED | ST_SOURCE_ISOLATED)
            == (ST_LEASED | ST_PREPARED | ST_SOURCE_ISOLATED)
        ),
        "migration_stream_nonempty": migration.stat().st_size > 0,
        "pci_bar_and_command_state_restored_without_guest_replay": (
            restored_bar_without_reconfiguration
        ),
        "nonsecret_prefix_state_exact_after_reload": source_prefix == target_prefix,
        "hidden_boundary_remains_locked_after_reload": (
            target_prefix["boundary_locked"] and locked_before_release
        ),
        "continued_boundary_matches_uninterrupted_qtest": boundary == expected,
        "response_released_only_after_continuation": bool(
            target_status_after & ST_RESPONSE_READY
        ),
        "reloaded_carrier_is_spent_not_restored": bool(target_status_after & ST_SPENT),
        "qemu_stderr_empty": source_stderr == "" and target_stderr == "",
    }
    if not all(controls.values()):
        failed = [name for name, value in controls.items() if not value]
        raise AssertionError(f"migration sham controls failed: {failed}")

    return {
        "schema": "phase-qemu-v0-migration-sham-v1",
        "qemu_binary_sha256": sha256(binary),
        "production_qtest_result_sha256": sha256(qtest_result),
        "migration_stream_sha256": sha256(migration),
        "migration_stream_bytes": migration.stat().st_size,
        "source_prefix": source_prefix,
        "reloaded_prefix": target_prefix,
        "continued_boundary": boundary,
        "controls": controls,
        "classification": {
            "recovery_classification": "SNAPSHOT_RELOAD",
            "restoration_classification": "NO_RESTORATION_CLAIM",
            "carrier_backing_identity_preserved_across_processes": False,
            "migration_stream_is_trusted_backend_state": True,
            "migration_stream_guest_mmio_visible": False,
            "host_migration_no_smuggle_enforcement_established": False,
            "accepted_catalytic_path": False,
        },
        "resource_law": {
            "migration_stream_bytes": migration.stat().st_size,
            "migration_stream_writes": 1,
            "migration_stream_reads": 1,
            "migration_stream_hashing_reads": 1,
            "source_and_target_qemu_processes": 2,
            "process_recreation_used": True,
            "guest_replayed_prefix_commands": 0,
            "warm_uninterrupted_qtest_boundary_used_as_baseline": True,
            "phase_device_v1_nominal_vmstate_payload_bytes_before_pci_and_framing": 143,
            "phase_device_vmstate_boolean_fields": 7,
            "phase_device_vmstate_u32_fields": 12,
            "phase_device_vmstate_u64_fields": 3,
            "phase_device_vmstate_i64_fields": 8,
            "qmp_and_qtest_message_bytes_instrumented": False,
            "source_and_target_peak_rss_pss_instrumented": False,
            "migration_storage_lifetime_instrumented": False,
            "snapshot_creation_and_reload_wall_time_instrumented": False,
            "whole_qemu_process_memory_accounted": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qemu", required=True, type=Path)
    parser.add_argument("--scratch", required=True, type=Path)
    parser.add_argument("--qtest-result", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = experiment(
        args.qemu.resolve(),
        require_managed_scratch(args.scratch),
        args.qtest_result.resolve(),
    )
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
