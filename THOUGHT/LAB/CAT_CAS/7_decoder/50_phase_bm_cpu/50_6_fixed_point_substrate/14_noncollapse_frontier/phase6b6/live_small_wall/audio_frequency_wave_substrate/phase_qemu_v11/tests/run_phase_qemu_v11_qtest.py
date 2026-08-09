#!/usr/bin/env python3
"""Fail-closed headless qtest/QMP evidence runner for Phase-QEMU V11.

The caller must supply both a compiled QEMU binary and a disk-backed scratch
directory.  This runner creates only a deterministic run directory below that
explicit scratch root.  It never allocates an implicit temporary directory.
Scientific/evidence fields are hashed separately from segregated wall times.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence


SCHEMA = "PHASE_QEMU_V11_HEADLESS_QTEST_QMP_EVIDENCE_V1"
DEVICE_TYPE = "phase-qemu-v11"
DEVICE_ID = "phase0"
QOM_PATH = "/machine/peripheral/phase0"
PCI_DEVICE_NUMBER = 4
PCI_FUNCTION = 0
PCI_VENDOR = 0x1234
PCI_DEVICE = 0x11FB
PCI_REVISION = 0x01
BAR0_BASE = 0x10000000
BAR0_SIZE = 0x1000
SOCKET_PATH_LIMIT = 100
CONNECT_TIMEOUT_SECONDS = 20
MIGRATION_TIMEOUT_SECONDS = 30

BACKEND_IDEAL = 0x0B01
BACKEND_OPEN = 0x0B02
BACKEND_EXTERNAL = 0x0B80

DESCRIPTOR_WORDS = (
    0x50313144,
    0x00010008,
    0x00000002,
    0x00020102,
    0x00020011,
    0x00030021,
    0x00000003,
    0x00010001,
)

LOCKED64 = 0xFFFFFFFFFFFFFFFF
MAGIC = 0x50483131
ABI = 0x00010000
BOUNDARY_HEADER = 0x5031314200010080

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

LIFE_EMPTY = 0
LIFE_LEASED = 1
LIFE_PREPARED = 2
LIFE_ISOLATED = 3
LIFE_SEALED = 4
LIFE_PRIVATE_ARMED = 5
LIFE_PRIVATE_READY = 6
LIFE_RESPONSE_READY = 9
LIFE_RESPONSE_ACKED = 10
LIFE_REUSABLE = 11
LIFE_SPENT = 12
LIFE_SHAM = 13

RETURN_NONE = 0
RETURN_EXACT_FORMAL = 1
RETURN_APPROX_MODEL = 2
RETURN_FAILED = 3

ERR_NONE = 0
ERR_BAD_STATE = 1
ERR_BAD_ARGUMENT = 2
ERR_TAG_MISMATCH = 3
ERR_GENERATION_MISMATCH = 4
ERR_DESCRIPTOR_INVALID = 5
ERR_PORT_NOT_CLEAR = 8
ERR_INVARIANT = 11
ERR_SNAPSHOT_REJECTED = 12
ERR_SNAPSHOT_LINEAGE = 13
ERR_RESPONSE_LOCKED = 14
ERR_BACKEND_UNAVAILABLE = 15
ERR_SECRET_SMUGGLE = 21
ERR_ENVIRONMENT_NOT_FACTORED = 24
ERR_REUSE_NOT_QUALIFIED = 27
ERR_CARRIER_REFERENCE_FAILED = 23
ERR_RESOURCE_UNSEALED = 30

FAULT_RESOURCE_UNSEALED = 7

CAP_PRIVATE_QOM_PROVIDER = 1 << 9
CAP_TEST_QOM_OBSERVER = 1 << 19

ST_RESPONSE_READY = 1 << 5
ST_EXACT_RETURN = 1 << 7
ST_SNAPSHOT_LINEAGE = 1 << 9
ST_OUTPUTS_HELD = 1 << 13
ST_RETURN_VERIFIED = 1 << 14
ST_SAME_ALLOCATION = 1 << 16
ST_ENV_FACTORED = 1 << 17
ST_NOISY_BACKEND = 1 << 18
ST_EXTERNAL_BACKEND = 1 << 19
ST_RESOURCE_SEALED = 1 << 21
ST_SHAM = 1 << 22

REG_MAGIC = 0x000
REG_ABI = 0x004
REG_BACKEND = 0x008
REG_CAPABILITIES_LO = 0x00C
REG_STATUS_LO = 0x010
REG_ERROR = 0x014
REG_GENERATION = 0x018
REG_EXACT_RETURN_GENERATION = 0x01C
REG_ARG0 = 0x020
REG_COMMAND = 0x028
REG_LIFECYCLE = 0x02C
REG_VIRTUAL_CYCLES = 0x030
REG_BOUNDARY_COMMIT_COOKIE = 0x040
REG_RESOURCE_STATE_CELLS = 0x048
REG_RESOURCE_SCRATCH_CELLS = 0x04C
REG_RESOURCE_QUERY_APPLICATIONS = 0x050
REG_RESOURCE_RETURN_CHECKS = 0x058
REG_RESOURCE_ENVIRONMENT_OPS = 0x060
REG_RESOURCE_PEAK_BITS = 0x068
REG_REQUEST_OWNER = 0x070
REG_REQUEST_PROGRAM = 0x074
REG_REQUEST_GENERATION = 0x078
REG_DESCRIPTOR_INDEX = 0x07C
REG_DESCRIPTOR_WORD = 0x080
REG_DESCRIPTOR_LENGTH = 0x084
REG_BOUNDARY_SCHEMA = 0x088
REG_DESCRIPTOR_FINGERPRINT = 0x090
REG_FAULT_MODE = 0x098
REG_PRIVATE_READY_MASK = 0x0AC
REG_RETURN_CLASS = 0x0B0
REG_BOUNDARY_LENGTH = 0x0B4
REG_ALLOCATION_ID_LO = 0x0B8
REG_ALLOCATION_ID_HI = 0x0C0
REG_CUSTODY_EPOCH = 0x0C8
REG_RESOURCE_SECRET_STORAGE_BITS = 0x0F0
REG_RESOURCE_CONTROL_WORDS = 0x0F8
REG_RESOURCE_PREPARATION_OPS = 0x100
REG_RESOURCE_CERTIFICATION_OPS = 0x108
REG_RESOURCE_LOGICAL_QUERIES = 0x110
REG_RESOURCE_LOSS_Q63 = 0x138
REG_RESOURCE_DEPHASING_Q63 = 0x140
REG_RESOURCE_REUSE_COUNT = 0x158
REG_RESOURCE_DISCARDED_TRIALS = 0x160
REG_RESOURCE_PRECISION_BITS = 0x178
REG_RESOURCE_SCHEMA = 0x180
REG_RESOURCE_DIGEST_LO = 0x188
REG_RESOURCE_DIGEST_HI = 0x190
REG_RESOURCE_COMPILER_OPS = 0x198
REG_RESOURCE_CONTROLLER_OPS = 0x1A0
REG_RESOURCE_CONSTRUCTION_OPS = 0x1A8
REG_RESOURCE_SECRET_ENTROPY_BITS = 0x1B0
REG_RESOURCE_CARRIER_PHOTON_NUMBER = 0x1B8
REG_BOUNDARY_BASE = 0x200
BOUNDARY_WORDS = 16

RESOURCE_SCHEMA = 0x00010001
RESOURCE_UNKNOWN = LOCKED64
ALLOCATED_PRIVATE_SECRET_STORAGE_BITS = 168
LOGICAL_SECRET_ENTROPY_BITS = 4

QOM_PRIVATE_A = "test-private-a"
QOM_PRIVATE_B = "test-private-b"
QOM_DEVICE_PROPERTIES = (
    "backend-id",
    "carrier-present",
    "test-provider-enabled",
    "test-fault-mode",
    "open-model-q32",
)
QOM_OBSERVERS = (
    "test-observe-phase-a",
    "test-observe-phase-b",
    "test-observe-prepare-count",
    "test-observe-client-supply-count",
    "test-observe-allocation-lo",
    "test-observe-allocation-hi",
    "test-observe-same-backing",
    "test-observe-kr-return",
    "test-observe-factorized",
    "test-observe-port-clear",
    "test-observe-env-factored",
    "test-observe-return-class",
)


class EvidenceFailure(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceFailure(message)


def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def filesystem_type(path: Path) -> str:
    resolved = path.resolve()
    best_mount = Path("/")
    best_type = "UNKNOWN"
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        fields = line.split()
        if "-" not in fields:
            continue
        separator = fields.index("-")
        if separator + 1 >= len(fields):
            continue
        mount_point = Path(fields[4].replace("\\040", " ")).resolve()
        try:
            resolved.relative_to(mount_point)
        except ValueError:
            continue
        if len(mount_point.parts) >= len(best_mount.parts):
            best_mount = mount_point
            best_type = fields[separator + 1]
    return best_type


def validate_inputs(qemu_binary: Path, scratch_dir: Path) -> dict[str, object]:
    qemu_binary = qemu_binary.resolve(strict=True)
    scratch_dir = scratch_dir.resolve(strict=True)
    require(qemu_binary.is_file(), "--qemu-binary must name a regular file")
    require(os.access(qemu_binary, os.X_OK), "--qemu-binary must be executable")
    require(scratch_dir.is_dir(), "--scratch-dir must name an existing directory")
    require(os.access(scratch_dir, os.W_OK), "--scratch-dir must be writable")
    fs_type = filesystem_type(scratch_dir)
    require(fs_type not in {"tmpfs", "ramfs"}, "RAM-backed scratch is forbidden")
    run_root = scratch_dir / "v11q"
    require(not run_root.exists(), "scratch run directory already exists: v11q")
    run_root.mkdir(mode=0o700)
    return {
        "qemu_binary": qemu_binary,
        "scratch_dir": scratch_dir,
        "run_root": run_root,
        "scratch_filesystem_type": fs_type,
        "qemu_binary_sha256": sha256_file(qemu_binary),
    }


class QTestClient:
    def __init__(self, connection: socket.socket):
        self.connection = connection
        self.reader = connection.makefile("rb", buffering=0)

    def close(self) -> None:
        try:
            self.reader.close()
        finally:
            self.connection.close()

    def command(self, command: str) -> int | None:
        self.connection.sendall(command.encode("ascii") + b"\n")
        try:
            response = self.reader.readline()
        except (TimeoutError, socket.timeout) as error:
            raise EvidenceFailure(
                f"qtest timed out during command: {command}"
            ) from error
        if not response:
            raise EvidenceFailure(f"qtest closed during command: {command}")
        parts = response.decode("ascii", "strict").strip().split()
        if not parts or parts[0] != "OK":
            raise EvidenceFailure(f"qtest failure for command {command!r}: {parts!r}")
        return int(parts[1], 0) if len(parts) > 1 else None

    def inw(self, port: int) -> int:
        value = self.command(f"inw 0x{port:x}")
        require(value is not None, "qtest inw returned no value")
        return int(value)

    def inl(self, port: int) -> int:
        value = self.command(f"inl 0x{port:x}")
        require(value is not None, "qtest inl returned no value")
        return int(value)

    def outw(self, port: int, value: int) -> None:
        require(self.command(f"outw 0x{port:x} 0x{value & 0xffff:x}") is None,
                "qtest outw returned a value")

    def outl(self, port: int, value: int) -> None:
        require(self.command(f"outl 0x{port:x} 0x{value & 0xffffffff:x}") is None,
                "qtest outl returned a value")

    def read8(self, address: int) -> int:
        value = self.command(f"readb 0x{address:x}")
        require(value is not None, "qtest readb returned no value")
        return int(value)

    def read16(self, address: int) -> int:
        value = self.command(f"readw 0x{address:x}")
        require(value is not None, "qtest readw returned no value")
        return int(value)

    def read32(self, address: int) -> int:
        value = self.command(f"readl 0x{address:x}")
        require(value is not None, "qtest readl returned no value")
        return int(value)

    def read64(self, address: int) -> int:
        value = self.command(f"readq 0x{address:x}")
        require(value is not None, "qtest readq returned no value")
        return int(value)

    def write32(self, address: int, value: int) -> None:
        require(self.command(f"writel 0x{address:x} 0x{value & 0xffffffff:x}") is None,
                "qtest writel returned a value")

    def write8(self, address: int, value: int) -> None:
        require(self.command(f"writeb 0x{address:x} 0x{value & 0xff:x}") is None,
                "qtest writeb returned a value")

    def write16(self, address: int, value: int) -> None:
        require(self.command(f"writew 0x{address:x} 0x{value & 0xffff:x}") is None,
                "qtest writew returned a value")

    def write64(self, address: int, value: int) -> None:
        require(self.command(f"writeq 0x{address:x} 0x{value & LOCKED64:x}") is None,
                "qtest writeq returned a value")

class QMPClient:
    def __init__(self, connection: socket.socket):
        self.connection = connection
        self.reader = connection.makefile("rb", buffering=0)
        self.request_id = 0
        greeting = self._read_message()
        require("QMP" in greeting, "missing QMP greeting")
        response = self.execute("qmp_capabilities")
        require("return" in response, "QMP capability negotiation failed")

    def close(self) -> None:
        try:
            self.reader.close()
        finally:
            self.connection.close()

    def _read_message(self) -> dict[str, Any]:
        try:
            line = self.reader.readline()
        except (TimeoutError, socket.timeout) as error:
            raise EvidenceFailure("QMP timed out waiting for a response") from error
        if not line:
            raise EvidenceFailure("QMP socket closed unexpectedly")
        value = json.loads(line)
        require(isinstance(value, dict), "QMP response is not an object")
        return value

    def execute(
        self,
        command: str,
        arguments: dict[str, object] | None = None,
        *,
        allow_error: bool = False,
    ) -> dict[str, Any]:
        self.request_id += 1
        request: dict[str, object] = {
            "execute": command,
            "id": self.request_id,
        }
        if arguments is not None:
            request["arguments"] = arguments
        self.connection.sendall(canonical_bytes(request) + b"\n")
        while True:
            response = self._read_message()
            if "event" in response:
                continue
            if response.get("id") != self.request_id:
                continue
            if "error" in response and not allow_error:
                error = response["error"]
                error_class = error.get("class", "UNKNOWN") if isinstance(error, dict) else "UNKNOWN"
                raise EvidenceFailure(f"QMP {command} failed with {error_class}")
            return response

    def qom_get(self, property_name: str) -> int:
        response = self.execute(
            "qom-get", {"path": QOM_PATH, "property": property_name}
        )
        value = response.get("return")
        require(isinstance(value, int), f"QOM property {property_name} is not integer")
        return value

    def qom_set(self, property_name: str, value: int) -> None:
        response = self.execute(
            "qom-set",
            {"path": QOM_PATH, "property": property_name, "value": value},
        )
        require(response.get("return") == {}, f"QOM set {property_name} failed")

    def qom_set_expected_error(self, property_name: str, value: int) -> str:
        response = self.execute(
            "qom-set",
            {"path": QOM_PATH, "property": property_name, "value": value},
            allow_error=True,
        )
        error = response.get("error")
        require(isinstance(error, dict), f"QOM set {property_name} unexpectedly succeeded")
        error_class = error.get("class")
        require(isinstance(error_class, str), "QMP error has no class")
        return error_class

    def qom_get_expected_error(self, property_name: str) -> str:
        response = self.execute(
            "qom-get",
            {"path": QOM_PATH, "property": property_name},
            allow_error=True,
        )
        error = response.get("error")
        require(isinstance(error, dict), f"QOM get {property_name} unexpectedly succeeded")
        error_class = error.get("class")
        require(isinstance(error_class, str), "QMP error has no class")
        return error_class


def connect_unix(path: Path, process: subprocess.Popen[bytes]) -> socket.socket:
    connect_path = path
    if len(os.fsencode(connect_path)) >= SOCKET_PATH_LIMIT:
        connect_path = Path("/proc") / str(process.pid) / "cwd" / path.name
    require(len(os.fsencode(connect_path)) < SOCKET_PATH_LIMIT,
            f"Unix client socket path too long: {path.name}")
    deadline = time.monotonic() + CONNECT_TIMEOUT_SECONDS
    last_error: OSError | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise EvidenceFailure(f"QEMU exited before socket became ready: {path.name}")
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            connection.connect(str(connect_path))
            connection.settimeout(CONNECT_TIMEOUT_SECONDS)
            return connection
        except OSError as error:
            last_error = error
            connection.close()
            time.sleep(0.01)
    raise EvidenceFailure(
        f"timed out connecting to {path.name}: {type(last_error).__name__}"
    )


@dataclass(frozen=True)
class DeviceOptions:
    backend_id: int = BACKEND_IDEAL
    carrier_present: bool = True
    provider_enabled: bool = True
    fault_mode: int = 0
    open_model_q32: int = 0


class QemuInstance:
    def __init__(
        self,
        qemu_binary: Path,
        case_dir: Path,
        options: DeviceOptions,
        *,
        incoming_defer: bool = False,
    ):
        self.qemu_binary = qemu_binary
        self.case_dir = case_dir
        self.options = options
        self.incoming_defer = incoming_defer
        self.process: subprocess.Popen[bytes] | None = None
        self.qtest: QTestClient | None = None
        self.qmp: QMPClient | None = None
        self.qtest_path = case_dir / "qt"
        self.qmp_path = case_dir / "qm"
        self.qtest_uri_path = Path("qt")
        self.qmp_uri_path = Path("qm")
        self.closed = False

    def start(self) -> None:
        self.case_dir.mkdir(mode=0o700)
        for path in (self.qtest_uri_path, self.qmp_uri_path):
            require(len(os.fsencode(path)) < SOCKET_PATH_LIMIT,
                    f"Unix socket path too long: {path.name}")
        device = ",".join(
            (
                DEVICE_TYPE,
                f"id={DEVICE_ID}",
                "bus=pcie.0",
                f"addr={PCI_DEVICE_NUMBER:x}.0",
                f"backend-id=0x{self.options.backend_id:04x}",
                f"carrier-present={'on' if self.options.carrier_present else 'off'}",
                f"test-provider-enabled={'on' if self.options.provider_enabled else 'off'}",
                f"test-fault-mode={self.options.fault_mode}",
                f"open-model-q32={self.options.open_model_q32}",
            )
        )
        command = [
            str(self.qemu_binary),
            "-machine",
            "q35,accel=qtest",
            "-m",
            "64M",
            "-S",
            "-nodefaults",
            "-display",
            "none",
            "-serial",
            "none",
            "-monitor",
            "none",
            "-qtest",
            f"unix:{self.qtest_uri_path},server=on,wait=off",
            "-qtest-log",
            "/dev/null",
            "-qmp",
            f"unix:{self.qmp_uri_path},server=on,wait=off",
            "-device",
            device,
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
        qmp_connection = connect_unix(self.qmp_path, self.process)
        self.qmp = QMPClient(qmp_connection)
        qtest_connection = connect_unix(self.qtest_path, self.process)
        self.qtest = QTestClient(qtest_connection)

    def close(self) -> dict[str, object]:
        if self.closed:
            raise EvidenceFailure("QEMU instance closed twice")
        self.closed = True
        if self.process is None:
            return {
                "stdout_bytes": 0,
                "stderr_bytes": 0,
                "stdout_sha256": sha256_bytes(b""),
                "stderr_sha256": sha256_bytes(b""),
            }
        if self.qmp is not None and self.process.poll() is None:
            try:
                self.qmp.execute("quit", allow_error=True)
            except (EvidenceFailure, OSError, socket.timeout):
                pass
        if self.qtest is not None:
            try:
                self.qtest.close()
            except OSError:
                pass
        if self.qmp is not None:
            try:
                self.qmp.close()
            except OSError:
                pass
        try:
            stdout, stderr = self.process.communicate(timeout=CONNECT_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                stdout, stderr = self.process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                stdout, stderr = self.process.communicate(timeout=5)
        return {
            "stdout_bytes": len(stdout),
            "stderr_bytes": len(stderr),
            "stdout_sha256": sha256_bytes(stdout),
            "stderr_sha256": sha256_bytes(stderr),
        }


class DeviceAccess:
    def __init__(self, qtest: QTestClient, qmp: QMPClient):
        self.qtest = qtest
        self.qmp = qmp

    @staticmethod
    def _pci_config_address(offset: int) -> int:
        return (
            0x80000000
            | (PCI_DEVICE_NUMBER << 11)
            | (PCI_FUNCTION << 8)
            | (offset & 0xFC)
        )

    def pci_read32(self, offset: int) -> int:
        self.qtest.outl(0xCF8, self._pci_config_address(offset))
        return self.qtest.inl(0xCFC + (offset & 3))

    def pci_read16(self, offset: int) -> int:
        self.qtest.outl(0xCF8, self._pci_config_address(offset))
        return self.qtest.inw(0xCFC + (offset & 2))

    def pci_write32(self, offset: int, value: int) -> None:
        self.qtest.outl(0xCF8, self._pci_config_address(offset))
        self.qtest.outl(0xCFC + (offset & 3), value)

    def pci_write16(self, offset: int, value: int) -> None:
        self.qtest.outl(0xCF8, self._pci_config_address(offset))
        self.qtest.outw(0xCFC + (offset & 2), value)

    def setup_pci(self) -> dict[str, int]:
        identity = self.pci_read32(0x00)
        vendor = identity & 0xFFFF
        device = (identity >> 16) & 0xFFFF
        revision = self.pci_read32(0x08) & 0xFF
        require(vendor == PCI_VENDOR, f"unexpected PCI vendor: 0x{vendor:04x}")
        require(device == PCI_DEVICE, f"unexpected PCI device: 0x{device:04x}")
        require(revision == PCI_REVISION, f"unexpected PCI revision: {revision}")
        self.pci_write32(0x10, BAR0_BASE)
        command = self.pci_read16(0x04)
        self.pci_write16(0x04, command | 0x2)
        bar = self.pci_read32(0x10) & 0xFFFFFFF0
        require(bar == BAR0_BASE, f"BAR0 mapping mismatch: 0x{bar:x}")
        return {
            "vendor_id": vendor,
            "device_id": device,
            "revision": revision,
            "bar0_base": bar,
            "bar0_size": BAR0_SIZE,
        }

    def read32(self, offset: int) -> int:
        return self.qtest.read32(BAR0_BASE + offset)

    def read64(self, offset: int) -> int:
        return self.qtest.read64(BAR0_BASE + offset)

    def write32(self, offset: int, value: int) -> None:
        self.qtest.write32(BAR0_BASE + offset, value)

    def command(self, command: int) -> int:
        self.write32(REG_COMMAND, command)
        return self.read32(REG_ERROR)

    def identity(self) -> dict[str, int]:
        return {
            "magic": self.read32(REG_MAGIC),
            "abi": self.read32(REG_ABI),
            "backend_id": self.read32(REG_BACKEND),
            "capabilities_lo": self.read32(REG_CAPABILITIES_LO),
        }

    def configure_request(self, owner: int, program: int, generation: int) -> None:
        self.write32(REG_REQUEST_OWNER, owner)
        self.write32(REG_REQUEST_PROGRAM, program)
        self.write32(REG_REQUEST_GENERATION, generation)

    def upload_descriptor(self, words: Sequence[int] = DESCRIPTOR_WORDS) -> None:
        for index, word in enumerate(words):
            self.write32(REG_DESCRIPTOR_INDEX, index)
            self.write32(REG_DESCRIPTOR_WORD, word)
        self.write32(REG_DESCRIPTOR_LENGTH, len(words))
        self.write32(REG_BOUNDARY_SCHEMA, 1)

    def read_boundary(self) -> list[int]:
        words: list[int] = []
        for index in range(BOUNDARY_WORDS):
            try:
                words.append(self.read64(REG_BOUNDARY_BASE + 8 * index))
            except EvidenceFailure as error:
                raise EvidenceFailure(
                    f"boundary read failed at word {index}: {error}"
                ) from error
        return words

    def read_observers(self) -> dict[str, int]:
        return {name: self.qmp.qom_get(name) for name in QOM_OBSERVERS}

    def bind_private(self, residue_a: int, residue_b: int) -> None:
        self.qmp.qom_set(QOM_PRIVATE_A, residue_a)
        self.qmp.qom_set(QOM_PRIVATE_B, residue_b)


def require_identity(access: DeviceAccess, expected_backend: int) -> dict[str, int]:
    identity = access.identity()
    require(identity["magic"] == MAGIC, "device magic mismatch")
    require(identity["abi"] == ABI, "device ABI mismatch")
    require(identity["backend_id"] == expected_backend, "backend ID mismatch")
    return identity


def require_empty_streams(streams: dict[str, object], case_name: str) -> None:
    require(streams["stdout_bytes"] == 0, f"{case_name}: QEMU stdout not empty")
    require(streams["stderr_bytes"] == 0, f"{case_name}: QEMU stderr not empty")


def prepare_descriptor_arm(
    access: DeviceAccess,
    generation: int,
    owner: int,
    program: int,
    *,
    first_generation: bool,
    descriptor_words: Sequence[int] = DESCRIPTOR_WORDS,
) -> dict[str, object]:
    access.configure_request(owner, program, generation)
    require(access.command(CMD_LEASE) == ERR_NONE, "LEASE failed")
    if first_generation:
        require(access.read32(REG_LIFECYCLE) == LIFE_LEASED, "LEASE lifecycle mismatch")
        require(access.command(CMD_PREPARE) == ERR_NONE, "PREPARE failed")
    else:
        require(access.read32(REG_LIFECYCLE) == LIFE_PREPARED,
                "reused LEASE did not return to PREPARED")
    require(access.read32(REG_LIFECYCLE) == LIFE_PREPARED,
            "PREPARED lifecycle mismatch before source isolation")
    require(access.command(CMD_ISOLATE_SOURCE) == ERR_NONE,
            "per-generation ISOLATE_SOURCE failed")
    require(access.read32(REG_LIFECYCLE) == LIFE_ISOLATED, "ISOLATED lifecycle mismatch")
    access.upload_descriptor(descriptor_words)
    require(access.command(CMD_SEAL_DESCRIPTOR) == ERR_NONE, "SEAL_DESCRIPTOR failed")
    require(access.read32(REG_LIFECYCLE) == LIFE_SEALED, "SEALED lifecycle mismatch")
    fingerprint = access.read64(REG_DESCRIPTOR_FINGERPRINT)
    require(access.read64(REG_BOUNDARY_COMMIT_COOKIE) == LOCKED64,
            "boundary commit cookie leaked before ARM_PRIVATE")
    locked = access.read_boundary()
    require(locked == [LOCKED64] * BOUNDARY_WORDS, "boundary leaked before ARM_PRIVATE")
    require(access.read32(REG_ERROR) == ERR_RESPONSE_LOCKED, "locked boundary error mismatch")
    require(access.command(CMD_ARM_PRIVATE) == ERR_NONE, "ARM_PRIVATE failed")
    require(access.read32(REG_LIFECYCLE) == LIFE_PRIVATE_ARMED, "PRIVATE_ARMED mismatch")
    return {
        "descriptor_fingerprint": fingerprint,
        "boundary_locked_before_private": True,
        "lease_result_lifecycle": LIFE_LEASED if first_generation else LIFE_PREPARED,
        "source_isolation_issued_this_generation": True,
    }


def successful_transaction(
    access: DeviceAccess,
    residue_a: int,
    residue_b: int,
    *,
    first_generation: bool,
    expected_backend: int,
) -> dict[str, object]:
    generation = access.read32(REG_GENERATION)
    owner = 0xA1100000 | generation
    program = 0xB1100000 | generation
    prefix = prepare_descriptor_arm(
        access,
        generation,
        owner,
        program,
        first_generation=first_generation,
    )
    access.qmp.qom_set(QOM_PRIVATE_A, residue_a)
    require(access.read32(REG_PRIVATE_READY_MASK) == 1, "private A mask mismatch")
    require(access.read_boundary() == [LOCKED64] * BOUNDARY_WORDS,
            "boundary leaked after only private A")
    access.qmp.qom_set(QOM_PRIVATE_B, residue_b)
    require(access.read32(REG_PRIVATE_READY_MASK) == 3, "private ready mask mismatch")
    require(access.read32(REG_LIFECYCLE) == LIFE_PRIVATE_READY,
            "PRIVATE_READY lifecycle mismatch")
    require(access.read_boundary() == [LOCKED64] * BOUNDARY_WORDS,
            "boundary leaked before EXECUTE_ATOMIC")
    require(access.command(CMD_EXECUTE_ATOMIC) == ERR_NONE, "EXECUTE_ATOMIC failed")
    require(access.read32(REG_LIFECYCLE) == LIFE_RESPONSE_READY,
            "RESPONSE_READY lifecycle mismatch")
    status = access.read32(REG_STATUS_LO)
    return_class = access.read32(REG_RETURN_CLASS)
    require(return_class == RETURN_EXACT_FORMAL, "exact transaction return class mismatch")
    require(status & ST_RESPONSE_READY, "response-ready status absent")
    require(status & ST_EXACT_RETURN, "exact-return status absent")
    require(status & ST_RETURN_VERIFIED, "return-verified status absent")
    require(status & ST_SAME_ALLOCATION, "same-allocation status absent")
    require(status & ST_ENV_FACTORED, "environment-factored status absent")
    require(status & ST_OUTPUTS_HELD, "outputs-held status absent before ACK")
    require(status & ST_RESOURCE_SEALED, "resource-sealed status absent")
    boundary = access.read_boundary()
    require(boundary[0] == BOUNDARY_HEADER, "boundary header mismatch")
    require(access.read64(REG_BOUNDARY_COMMIT_COOKIE) == BOUNDARY_HEADER,
            "boundary commit cookie did not release atomically")
    require(boundary != [LOCKED64] * BOUNDARY_WORDS, "atomic boundary did not release")
    require((boundary[1] & 0xFFFFFFFF) == generation, "boundary generation mismatch")
    require(((boundary[1] >> 32) & 0xFFFF) == expected_backend,
            "boundary backend mismatch")
    require(((boundary[1] >> 48) & 0xFFFF) == RETURN_EXACT_FORMAL,
            "boundary return class mismatch")
    require(boundary[2] & ST_RESPONSE_READY,
            "committed boundary omitted response-ready status")
    require(boundary[2] & ST_OUTPUTS_HELD,
            "committed boundary omitted outputs-held status")
    observers = access.read_observers()
    require(observers["test-observe-phase-a"] == residue_a, "phase A observer mismatch")
    require(observers["test-observe-phase-b"] == residue_b, "phase B observer mismatch")
    require(observers["test-observe-same-backing"] == 1, "same-backing observer false")
    require(observers["test-observe-kr-return"] == 1, "KR return observer false")
    require(observers["test-observe-factorized"] == 1, "factorization observer false")
    require(observers["test-observe-port-clear"] == 1, "port-clear observer false")
    require(observers["test-observe-env-factored"] == 1,
            "environment observer false")
    require(observers["test-observe-return-class"] == RETURN_EXACT_FORMAL,
            "observer return class mismatch")
    allocation = [
        access.read64(REG_ALLOCATION_ID_LO),
        access.read64(REG_ALLOCATION_ID_HI),
    ]
    resources = {
        "state_cells": access.read32(REG_RESOURCE_STATE_CELLS),
        "scratch_cells": access.read32(REG_RESOURCE_SCRATCH_CELLS),
        "query_applications": access.read64(REG_RESOURCE_QUERY_APPLICATIONS),
        "return_checks": access.read64(REG_RESOURCE_RETURN_CHECKS),
        "environment_ops": access.read64(REG_RESOURCE_ENVIRONMENT_OPS),
        "peak_bits": access.read64(REG_RESOURCE_PEAK_BITS),
        "allocated_secret_storage_bits": access.read64(
            REG_RESOURCE_SECRET_STORAGE_BITS
        ),
        "control_words": access.read64(REG_RESOURCE_CONTROL_WORDS),
        "preparation_ops": access.read64(REG_RESOURCE_PREPARATION_OPS),
        "certification_ops": access.read64(REG_RESOURCE_CERTIFICATION_OPS),
        "logical_queries": access.read64(REG_RESOURCE_LOGICAL_QUERIES),
        "reuse_count": access.read64(REG_RESOURCE_REUSE_COUNT),
        "discarded_trials": access.read64(REG_RESOURCE_DISCARDED_TRIALS),
        "precision_bits": access.read64(REG_RESOURCE_PRECISION_BITS),
        "resource_schema": access.read32(REG_RESOURCE_SCHEMA),
        "resource_digest": [
            access.read64(REG_RESOURCE_DIGEST_LO),
            access.read64(REG_RESOURCE_DIGEST_HI),
        ],
        "compiler_ops": access.read64(REG_RESOURCE_COMPILER_OPS),
        "controller_ops": access.read64(REG_RESOURCE_CONTROLLER_OPS),
        "construction_ops": access.read64(REG_RESOURCE_CONSTRUCTION_OPS),
        "logical_secret_entropy_bits": access.read64(
            REG_RESOURCE_SECRET_ENTROPY_BITS
        ),
        "carrier_photon_number": access.read64(
            REG_RESOURCE_CARRIER_PHOTON_NUMBER
        ),
    }
    require(resources["state_cells"] == 96 * 96, "state-cell resource mismatch")
    require(resources["scratch_cells"] == 96 * 96, "scratch-cell resource mismatch")
    require(resources["query_applications"] == 2 * generation,
            "query-application resource mismatch")
    require(resources["logical_queries"] == 2 * generation,
            "logical-query resource mismatch")
    expected_control_words = 29 + 30 * (generation - 1)
    require(resources["control_words"] == expected_control_words,
            "BAR/QOM control-word resource mismatch")
    require(resources["controller_ops"] == resources["control_words"],
            "controller-op resource mismatch")
    require(
        resources["allocated_secret_storage_bits"]
        == ALLOCATED_PRIVATE_SECRET_STORAGE_BITS,
        "allocated private-secret storage resource mismatch",
    )
    require(
        resources["logical_secret_entropy_bits"] == LOGICAL_SECRET_ENTROPY_BITS,
        "logical secret-entropy resource mismatch",
    )
    require(
        resources["allocated_secret_storage_bits"]
        >= resources["logical_secret_entropy_bits"],
        "allocated secret storage is below logical secret entropy",
    )
    require(resources["compiler_ops"] == RESOURCE_UNKNOWN,
            "compiler-op resource must be explicitly unknown")
    require(resources["construction_ops"] == RESOURCE_UNKNOWN,
            "construction-op resource must be explicitly unknown")
    require(resources["carrier_photon_number"] == 1,
            "carrier photon-number resource mismatch")
    require(resources["precision_bits"] == 64, "precision resource mismatch")
    require(resources["resource_schema"] == RESOURCE_SCHEMA,
            "resource schema mismatch")
    require(resources["resource_digest"] != [0, 0], "resource digest unsealed")
    return {
        "generation": generation,
        "residue_pair": [residue_a, residue_b],
        "descriptor_fingerprint": prefix["descriptor_fingerprint"],
        "lease_result_lifecycle": prefix["lease_result_lifecycle"],
        "source_isolation_issued_this_generation": prefix[
            "source_isolation_issued_this_generation"
        ],
        "boundary_locked_until_atomic_commit": True,
        "commit_cookie_locked_then_released": True,
        "outputs_held_before_ack": True,
        "boundary_header": boundary[0],
        "boundary_status_word": boundary[2],
        "allocation_id": allocation,
        "custody_epoch": access.read64(REG_CUSTODY_EPOCH),
        "observers": observers,
        "resources": resources,
    }


def resource_seal_immutability_case(access: DeviceAccess) -> dict[str, object]:
    transaction = successful_transaction(
        access,
        1,
        2,
        first_generation=True,
        expected_backend=BACKEND_IDEAL,
    )
    sealed_resources = transaction["resources"]
    require(isinstance(sealed_resources, dict), "sealed resources missing")
    sealed_control_words = sealed_resources["control_words"]
    sealed_digest = list(sealed_resources["resource_digest"])
    sealed_boundary = access.read_boundary()
    require(access.read32(REG_ARG0) == 0, "ARG0 was nonzero before seal probe")

    # A width-correct, otherwise writable BAR access is counted internally but
    # may not mutate state or the response-local sealed resource snapshot.
    access.write32(REG_ARG0, 0xA55AA55A)
    require(access.read32(REG_ERROR) == ERR_RESPONSE_LOCKED,
            "post-seal BAR mutation was not response-locked")
    require(access.read32(REG_ARG0) == 0, "post-seal BAR write changed ARG0")
    require(access.read64(REG_RESOURCE_CONTROL_WORDS) == sealed_control_words,
            "post-seal BAR write changed the sealed control receipt")
    require(
        [
            access.read64(REG_RESOURCE_DIGEST_LO),
            access.read64(REG_RESOURCE_DIGEST_HI),
        ]
        == sealed_digest,
        "post-seal BAR write changed the resource digest",
    )
    require(access.read_boundary() == sealed_boundary,
            "post-seal BAR write changed the committed boundary")
    return {
        "sealed_control_words": sealed_control_words,
        "sealed_resource_digest": sealed_digest,
        "width_correct_post_seal_write_error": ERR_RESPONSE_LOCKED,
        "argument_state_unchanged": True,
        "resource_snapshot_unchanged": True,
        "boundary_unchanged": True,
    }


def run_instance_case(
    qemu_binary: Path,
    run_root: Path,
    name: str,
    options: DeviceOptions,
    callback: Callable[[DeviceAccess], dict[str, object]],
    wall_times: dict[str, int],
) -> dict[str, object]:
    started = time.monotonic_ns()
    instance = QemuInstance(qemu_binary, run_root / name, options)
    streams: dict[str, object]
    try:
        instance.start()
        require(instance.qtest is not None and instance.qmp is not None,
                "QEMU clients unavailable")
        access = DeviceAccess(instance.qtest, instance.qmp)
        pci = access.setup_pci()
        identity = require_identity(access, options.backend_id)
        evidence = callback(access)
    finally:
        streams = instance.close()
        wall_times[name] = time.monotonic_ns() - started
    require_empty_streams(streams, name)
    return {
        "pci": pci,
        "identity": identity,
        "evidence": evidence,
        "captured_process_streams": streams,
    }


def ideal_all_pairs_case(access: DeviceAccess) -> dict[str, object]:
    transactions: list[dict[str, object]] = []
    residue_pairs = list(itertools.product(range(3), repeat=2))
    for index, (residue_a, residue_b) in enumerate(residue_pairs):
        try:
            transaction = successful_transaction(
                access,
                residue_a,
                residue_b,
                first_generation=index == 0,
                expected_backend=BACKEND_IDEAL,
            )
        except EvidenceFailure as error:
            raise EvidenceFailure(
                f"ideal residue pair {index} ({residue_a},{residue_b}) failed: {error}"
            ) from error
        transactions.append(transaction)
        require(access.command(CMD_ACK_RESPONSE) == ERR_NONE, "ACK_RESPONSE failed")
        require(access.read32(REG_LIFECYCLE) == LIFE_RESPONSE_ACKED,
                "response ACK lifecycle mismatch")
        post_ack_status = access.read32(REG_STATUS_LO)
        require(not post_ack_status & ST_OUTPUTS_HELD,
                "outputs remained held after ACK")
        require(not post_ack_status & ST_RESPONSE_READY,
                "response remained ready after ACK")
        require(access.read_boundary() == [LOCKED64] * BOUNDARY_WORDS,
                "boundary remained visible after ACK")
        transaction["outputs_held_after_ack"] = False
        transaction["response_ready_after_ack"] = False
        if index + 1 < len(residue_pairs):
            require(access.command(CMD_BEGIN_REUSE) == ERR_NONE, "BEGIN_REUSE failed")
            require(access.read32(REG_LIFECYCLE) == LIFE_REUSABLE,
                    "reusable lifecycle mismatch")
            require(access.read32(REG_GENERATION) == index + 2,
                    "generation did not advance")
    fingerprints = {item["descriptor_fingerprint"] for item in transactions}
    allocations = {tuple(item["allocation_id"]) for item in transactions}
    final_observers = access.read_observers()
    require(len(fingerprints) == 1, "public descriptor fingerprint varied by secret pair")
    require(len(allocations) == 1, "allocation changed across reuse")
    require(final_observers["test-observe-prepare-count"] == 1,
            "prepare count changed across reuse")
    require(final_observers["test-observe-client-supply-count"] == 18,
            "fresh-client supply count mismatch")
    require(access.read64(REG_RESOURCE_REUSE_COUNT) == 8, "reuse counter mismatch")
    two_generation_reuse = {
        "generation_1": transactions[0]["generation"],
        "generation_2": transactions[1]["generation"],
        "allocation_id_equal": (
            transactions[0]["allocation_id"] == transactions[1]["allocation_id"]
        ),
        "prepare_count_after_generation_2": transactions[1]["observers"]
        ["test-observe-prepare-count"],
        "client_supply_count_after_generation_2": transactions[1]["observers"]
        ["test-observe-client-supply-count"],
    }
    require(two_generation_reuse["allocation_id_equal"],
            "two-generation allocation equality failed")
    require(two_generation_reuse["prepare_count_after_generation_2"] == 1,
            "second generation prepared again")
    require(two_generation_reuse["client_supply_count_after_generation_2"] == 4,
            "second generation did not receive two fresh clients")
    return {
        "ordered_residue_pairs": [list(pair) for pair in residue_pairs],
        "transactions": transactions,
        "all_nine_pairs_exercised": len(transactions) == 9,
        "descriptor_byte_identical_for_all_pairs": len(fingerprints) == 1,
        "same_allocation_for_all_pairs": len(allocations) == 1,
        "two_generation_reuse": two_generation_reuse,
        "final_prepare_count": final_observers["test-observe-prepare-count"],
        "final_client_supply_count": final_observers[
            "test-observe-client-supply-count"
        ],
        "final_reuse_count": access.read64(REG_RESOURCE_REUSE_COUNT),
    }


def open_zero_case(access: DeviceAccess) -> dict[str, object]:
    transaction = successful_transaction(
        access, 1, 2, first_generation=True, expected_backend=BACKEND_OPEN
    )
    status = access.read32(REG_STATUS_LO)
    require(status & ST_NOISY_BACKEND, "open backend status bit absent")
    return {
        "zero_open_model_exact_parity": True,
        "transaction": transaction,
        "loss_q63": access.read64(REG_RESOURCE_LOSS_Q63),
        "dephasing_q63": access.read64(REG_RESOURCE_DEPHASING_Q63),
    }


def open_nonzero_case(access: DeviceAccess) -> dict[str, object]:
    generation = access.read32(REG_GENERATION)
    prepare_descriptor_arm(
        access, generation, 0xA1200001, 0xB1200001, first_generation=True
    )
    access.bind_private(1, 2)
    require(access.command(CMD_EXECUTE_ATOMIC) == ERR_NONE,
            "nonzero open execute command failed")
    require(access.read32(REG_RETURN_CLASS) == RETURN_APPROX_MODEL,
            "nonzero open backend was not approximate")
    status = access.read32(REG_STATUS_LO)
    require(status & ST_RESPONSE_READY, "approximate response not committed")
    require(status & ST_OUTPUTS_HELD, "approximate outputs not held before ACK")
    require(status & ST_NOISY_BACKEND, "open status bit absent")
    require(not status & ST_EXACT_RETURN, "nonzero open backend reported exact")
    require(access.read64(REG_RESOURCE_LOSS_Q63) != 0, "loss resource was zero")
    require(access.read64(REG_RESOURCE_DEPHASING_Q63) != 0,
            "dephasing resource was zero")
    boundary = access.read_boundary()
    require(boundary[0] == BOUNDARY_HEADER, "approximate boundary header mismatch")
    require(boundary[2] & ST_OUTPUTS_HELD,
            "approximate boundary omitted outputs-held status")
    require(access.command(CMD_ACK_RESPONSE) == ERR_NONE, "approximate ACK failed")
    post_ack_status = access.read32(REG_STATUS_LO)
    require(not post_ack_status & ST_OUTPUTS_HELD,
            "approximate outputs remained held after ACK")
    require(not post_ack_status & ST_RESPONSE_READY,
            "approximate response remained ready after ACK")
    require(access.read32(REG_LIFECYCLE) == LIFE_SPENT,
            "approximate backend did not become spent")
    require(access.command(CMD_BEGIN_REUSE) == ERR_REUSE_NOT_QUALIFIED,
            "approximate backend unexpectedly reused")
    return {
        "return_class": RETURN_APPROX_MODEL,
        "exact_return_status": False,
        "outputs_held_before_ack": True,
        "outputs_held_after_ack": False,
        "reuse_rejected": True,
        "loss_q63": boundary[14],
        "dephasing_q63": boundary[15],
        "environment_factored_observer": access.qmp.qom_get(
            "test-observe-env-factored"
        ),
    }


def external_case(access: DeviceAccess) -> dict[str, object]:
    access.configure_request(0xA1800001, 0xB1800001, 1)
    error = access.command(CMD_LEASE)
    require(error == ERR_BACKEND_UNAVAILABLE, "external backend did not fail unavailable")
    require(access.read32(REG_LIFECYCLE) == LIFE_EMPTY,
            "external failure mutated lifecycle")
    status = access.read32(REG_STATUS_LO)
    require(status & ST_EXTERNAL_BACKEND, "external backend status bit absent")
    return {
        "lease_error": error,
        "lifecycle": access.read32(REG_LIFECYCLE),
        "generation": access.read32(REG_GENERATION),
        "response_ready": bool(status & ST_RESPONSE_READY),
        "external_status": True,
    }


FAULT_EXPECTED_ERROR = {
    1: ERR_CARRIER_REFERENCE_FAILED,
    2: ERR_CARRIER_REFERENCE_FAILED,
    3: ERR_CARRIER_REFERENCE_FAILED,
    4: ERR_PORT_NOT_CLEAR,
    5: ERR_ENVIRONMENT_NOT_FACTORED,
    6: ERR_INVARIANT,
    FAULT_RESOURCE_UNSEALED: ERR_RESOURCE_UNSEALED,
}


def fault_case(access: DeviceAccess, fault_mode: int) -> dict[str, object]:
    generation = access.read32(REG_GENERATION)
    prepare_descriptor_arm(
        access,
        generation,
        0xAF000000 | fault_mode,
        0xBF000000 | fault_mode,
        first_generation=True,
    )
    access.bind_private(1, 2)
    error = access.command(CMD_EXECUTE_ATOMIC)
    require(error == FAULT_EXPECTED_ERROR[fault_mode],
            f"fault {fault_mode} error mismatch")
    require(access.read32(REG_RETURN_CLASS) == RETURN_FAILED,
            f"fault {fault_mode} return class mismatch")
    require(access.read32(REG_LIFECYCLE) == LIFE_SPENT,
            f"fault {fault_mode} did not spend generation")
    require(access.read_boundary() == [LOCKED64] * BOUNDARY_WORDS,
            f"fault {fault_mode} leaked boundary")
    return {
        "fault_mode": fault_mode,
        "error": error,
        "return_class": RETURN_FAILED,
        "lifecycle": LIFE_SPENT,
        "boundary_locked": True,
        "discarded_trials": access.read64(REG_RESOURCE_DISCARDED_TRIALS),
        "observers": access.read_observers(),
    }


def order_case(access: DeviceAccess) -> dict[str, object]:
    access.configure_request(0xAA000001, 0xBB000001, 1)
    execute_error = access.command(CMD_EXECUTE_ATOMIC)
    require(execute_error == ERR_TAG_MISMATCH, "pre-lease execute error mismatch")
    prepare_error = access.command(CMD_PREPARE)
    require(prepare_error == ERR_TAG_MISMATCH, "pre-lease prepare error mismatch")
    access.configure_request(0xAA000001, 0xBB000001, 2)
    generation_error = access.command(CMD_LEASE)
    require(generation_error == ERR_GENERATION_MISMATCH,
            "wrong-generation lease error mismatch")
    return {
        "execute_before_lease_error": execute_error,
        "prepare_before_lease_error": prepare_error,
        "wrong_generation_lease_error": generation_error,
        "lifecycle_unchanged": access.read32(REG_LIFECYCLE) == LIFE_EMPTY,
    }


def invalid_mmio_width_case(access: DeviceAccess) -> dict[str, object]:
    invalid_read_byte = access.qtest.read8(BAR0_BASE + REG_MAGIC)
    read_byte_error = access.read32(REG_ERROR)
    require(invalid_read_byte == 0xFF, "invalid byte read was not all-ones")
    require(read_byte_error == ERR_BAD_ARGUMENT,
            "invalid byte read did not latch BAD_ARGUMENT")

    invalid_read_word = access.qtest.read16(BAR0_BASE + REG_MAGIC)
    read_word_error = access.read32(REG_ERROR)
    require(invalid_read_word == 0xFFFF, "invalid word read was not all-ones")
    require(read_word_error == ERR_BAD_ARGUMENT,
            "invalid word read did not latch BAD_ARGUMENT")

    invalid_read_unaligned = access.qtest.read64(
        BAR0_BASE + REG_VIRTUAL_CYCLES + 4
    )
    read_unaligned_error = access.read32(REG_ERROR)
    require(invalid_read_unaligned == LOCKED64,
            "unaligned 64-bit read was not all-ones")
    require(read_unaligned_error == ERR_BAD_ARGUMENT,
            "unaligned 64-bit read did not latch BAD_ARGUMENT")

    access.qtest.write8(BAR0_BASE + REG_COMMAND, CMD_LEASE)
    write_byte_error = access.read32(REG_ERROR)
    require(write_byte_error == ERR_BAD_ARGUMENT,
            "invalid byte write did not latch BAD_ARGUMENT")
    access.qtest.write16(BAR0_BASE + REG_COMMAND, CMD_LEASE)
    write_word_error = access.read32(REG_ERROR)
    require(write_word_error == ERR_BAD_ARGUMENT,
            "invalid word write did not latch BAD_ARGUMENT")
    access.qtest.write64(BAR0_BASE + REG_VIRTUAL_CYCLES + 4, 0)
    write_unaligned_error = access.read32(REG_ERROR)
    require(write_unaligned_error == ERR_BAD_ARGUMENT,
            "unaligned 64-bit write did not latch BAD_ARGUMENT")
    require(access.read32(REG_LIFECYCLE) == LIFE_EMPTY,
            "invalid MMIO controls mutated lifecycle")
    require(access.read32(REG_GENERATION) == 1,
            "invalid MMIO controls mutated generation")
    return {
        "invalid_byte_read": {
            "returned": invalid_read_byte,
            "latched_error": read_byte_error,
        },
        "invalid_word_read": {
            "returned": invalid_read_word,
            "latched_error": read_word_error,
        },
        "invalid_unaligned_read": {
            "returned": invalid_read_unaligned,
            "latched_error": read_unaligned_error,
        },
        "invalid_byte_write_latched_error": write_byte_error,
        "invalid_word_write_latched_error": write_word_error,
        "invalid_unaligned_write_latched_error": write_unaligned_error,
        "all_invalid_widths_fail_closed": True,
        "lifecycle_unchanged": True,
    }


def service_mode_rejection_case(access: DeviceAccess) -> dict[str, object]:
    capabilities = access.read32(REG_CAPABILITIES_LO)
    require(not capabilities & CAP_PRIVATE_QOM_PROVIDER,
            "service mode exposed the private-provider capability")
    require(not capabilities & CAP_TEST_QOM_OBSERVER,
            "service mode exposed the observer capability")
    private_error_class = access.qmp.qom_set_expected_error(QOM_PRIVATE_A, 1)
    observer_error_class = access.qmp.qom_get_expected_error(QOM_OBSERVERS[0])
    require(private_error_class == "GenericError",
            "service-mode private QOM rejection class mismatch")
    require(observer_error_class == "GenericError",
            "service-mode observer QOM rejection class mismatch")

    access.configure_request(0xAA500001, 0xBB500001, 1)
    require(access.command(CMD_LEASE) == ERR_NONE, "service-mode LEASE failed")
    require(access.command(CMD_PREPARE) == ERR_NONE, "service-mode PREPARE failed")
    require(access.command(CMD_ISOLATE_SOURCE) == ERR_NONE,
            "service-mode ISOLATE_SOURCE failed")
    access.upload_descriptor()
    require(access.command(CMD_SEAL_DESCRIPTOR) == ERR_NONE,
            "service-mode SEAL_DESCRIPTOR failed")
    arm_error = access.command(CMD_ARM_PRIVATE)
    require(arm_error == ERR_BACKEND_UNAVAILABLE,
            "service mode did not reject ARM_PRIVATE")
    require(access.read32(REG_LIFECYCLE) == LIFE_SEALED,
            "service-mode ARM rejection mutated lifecycle")
    require(access.read_boundary() == [LOCKED64] * BOUNDARY_WORDS,
            "service-mode rejection leaked a boundary")
    return {
        "test_provider_enabled": False,
        "private_provider_capability_exposed": False,
        "observer_capability_exposed": False,
        "private_property": QOM_PRIVATE_A,
        "private_qom_error_class": private_error_class,
        "observer_property": QOM_OBSERVERS[0],
        "observer_qom_error_class": observer_error_class,
        "arm_private_error": arm_error,
        "boundary_locked": True,
    }


def missing_private_case(access: DeviceAccess) -> dict[str, object]:
    prepare_descriptor_arm(
        access, 1, 0xAA100001, 0xBB100001, first_generation=True
    )
    access.qmp.qom_set(QOM_PRIVATE_A, 1)
    require(access.read32(REG_PRIVATE_READY_MASK) == 1, "missing-private mask mismatch")
    error = access.command(CMD_EXECUTE_ATOMIC)
    require(error == ERR_BAD_STATE, "missing-private execute did not fail closed")
    require(access.read32(REG_LIFECYCLE) == LIFE_PRIVATE_ARMED,
            "missing-private failure mutated lifecycle")
    require(access.read_boundary() == [LOCKED64] * BOUNDARY_WORDS,
            "missing-private failure leaked boundary")
    return {
        "private_ready_mask": 1,
        "execute_error": error,
        "lifecycle": LIFE_PRIVATE_ARMED,
        "boundary_locked": True,
    }


def duplicate_private_case(access: DeviceAccess) -> dict[str, object]:
    prepare_descriptor_arm(
        access, 1, 0xAA200001, 0xBB200001, first_generation=True
    )
    access.qmp.qom_set(QOM_PRIVATE_A, 1)
    error_class = access.qmp.qom_set_expected_error(QOM_PRIVATE_A, 2)
    require(access.read32(REG_PRIVATE_READY_MASK) == 1,
            "duplicate private write changed ready mask")
    access.qmp.qom_set(QOM_PRIVATE_B, 2)
    require(access.command(CMD_EXECUTE_ATOMIC) == ERR_NONE,
            "transaction failed after rejected duplicate")
    require(access.read32(REG_RETURN_CLASS) == RETURN_EXACT_FORMAL,
            "duplicate rejection corrupted transaction")
    return {
        "duplicate_qmp_error_class": error_class,
        "ready_mask_after_rejected_duplicate": 1,
        "original_binding_preserved": access.qmp.qom_get(
            "test-observe-phase-a"
        )
        == 1,
        "transaction_remained_exact": True,
    }


def private_smuggle_case(access: DeviceAccess) -> dict[str, object]:
    access.configure_request(0xAA300001, 0xBB300001, 1)
    require(access.command(CMD_LEASE) == ERR_NONE, "smuggle LEASE failed")
    require(access.command(CMD_PREPARE) == ERR_NONE, "smuggle PREPARE failed")
    require(access.command(CMD_ISOLATE_SOURCE) == ERR_NONE,
            "smuggle ISOLATE_SOURCE failed")
    altered = list(DESCRIPTOR_WORDS)
    altered[4] ^= 1
    access.upload_descriptor(altered)
    error = access.command(CMD_SEAL_DESCRIPTOR)
    require(error == ERR_SECRET_SMUGGLE, "secret-derived descriptor was accepted")
    require(access.read32(REG_LIFECYCLE) == LIFE_ISOLATED,
            "smuggle rejection mutated lifecycle")
    require(access.read_boundary() == [LOCKED64] * BOUNDARY_WORDS,
            "smuggle rejection leaked boundary")
    return {
        "classification": "PRIVATE_SECRET_SMUGGLE_CONTROL",
        "altered_descriptor_word_index": 4,
        "seal_error": error,
        "descriptor_rejected": True,
        "private_provider_not_armed": access.read32(REG_PRIVATE_READY_MASK) == 0,
        "boundary_locked": True,
    }


def snapshot_case(access: DeviceAccess) -> dict[str, object]:
    initial_lifecycle = access.read32(REG_LIFECYCLE)
    error = access.command(CMD_SNAPSHOT)
    require(error == ERR_SNAPSHOT_REJECTED, "snapshot command was not rejected")
    require(access.read32(REG_LIFECYCLE) == initial_lifecycle,
            "snapshot rejection mutated lifecycle")
    require(access.read_boundary() == [LOCKED64] * BOUNDARY_WORDS,
            "snapshot rejection leaked boundary")
    return {
        "snapshot_command_error": error,
        "lifecycle_before": initial_lifecycle,
        "lifecycle_after": access.read32(REG_LIFECYCLE),
        "boundary_locked": True,
    }


def wait_source_migration(qmp: QMPClient) -> str:
    deadline = time.monotonic() + MIGRATION_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        response = qmp.execute("query-migrate")
        result = response.get("return")
        require(isinstance(result, dict), "query-migrate returned no object")
        status = result.get("status")
        require(isinstance(status, str), "query-migrate returned no status")
        if status == "completed":
            return status
        if status in {"failed", "cancelled"}:
            raise EvidenceFailure(f"migration ended with status {status}")
        time.sleep(0.01)
    raise EvidenceFailure("migration did not complete")


def wait_destination_sham(access: DeviceAccess) -> None:
    deadline = time.monotonic() + MIGRATION_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if access.read32(REG_LIFECYCLE) == LIFE_SHAM:
            return
        time.sleep(0.01)
    raise EvidenceFailure("destination never entered SHAM lifecycle")


def migration_case(
    qemu_binary: Path,
    run_root: Path,
    wall_times: dict[str, int],
) -> dict[str, object]:
    name = "migration_sham"
    started = time.monotonic_ns()
    case_root = run_root / name
    case_root.mkdir(mode=0o700)
    source = QemuInstance(qemu_binary, case_root / "s", DeviceOptions())
    destination = QemuInstance(
        qemu_binary,
        case_root / "d",
        DeviceOptions(),
        incoming_defer=True,
    )
    second_destination = QemuInstance(
        qemu_binary,
        case_root / "e",
        DeviceOptions(),
        incoming_defer=True,
    )
    source_streams: dict[str, object]
    destination_streams: dict[str, object]
    second_destination_streams: dict[str, object]
    try:
        source.start()
        destination.start()
        second_destination.start()
        require(source.qtest is not None and source.qmp is not None,
                "source migration clients unavailable")
        require(destination.qtest is not None and destination.qmp is not None,
                "destination migration clients unavailable")
        require(
            second_destination.qtest is not None and
            second_destination.qmp is not None,
            "second destination migration clients unavailable",
        )
        source_access = DeviceAccess(source.qtest, source.qmp)
        destination_access = DeviceAccess(destination.qtest, destination.qmp)
        second_destination_access = DeviceAccess(
            second_destination.qtest, second_destination.qmp
        )
        source_pci = source_access.setup_pci()
        destination_pci = destination_access.setup_pci()
        second_destination_pci = second_destination_access.setup_pci()
        require_identity(source_access, BACKEND_IDEAL)
        require_identity(destination_access, BACKEND_IDEAL)
        require_identity(second_destination_access, BACKEND_IDEAL)

        prepare_descriptor_arm(
            source_access, 1, 0xAA400001, 0xBB400001, first_generation=True
        )
        source_access.bind_private(1, 2)
        require(source_access.read32(REG_LIFECYCLE) == LIFE_PRIVATE_READY,
                "source not private-ready before migration")

        migration_socket = Path("../m")
        require(len(os.fsencode(migration_socket)) < SOCKET_PATH_LIMIT,
                "migration Unix socket URI path too long")
        require(not (case_root / "m").exists(),
                "migration Unix socket already exists")
        destination.qmp.execute(
            "migrate-incoming", {"uri": f"unix:{migration_socket}"}
        )
        source.qmp.execute("migrate", {"uri": f"unix:{migration_socket}"})
        source_status = wait_source_migration(source.qmp)
        wait_destination_sham(destination_access)

        destination_status = destination_access.read32(REG_STATUS_LO)
        require(destination_status & ST_SNAPSHOT_LINEAGE,
                "destination snapshot-lineage status absent")
        require(destination_status & ST_SHAM, "destination sham status absent")
        require(destination_access.read32(REG_ERROR) == ERR_SNAPSHOT_LINEAGE,
                "destination snapshot-lineage error absent")
        require(destination_access.read32(REG_PRIVATE_READY_MASK) == 0,
                "destination retained private-ready state")
        require(destination_access.read32(REG_RETURN_CLASS) == RETURN_FAILED,
                "destination return class was not failed")
        require(destination_access.read_boundary() == [LOCKED64] * BOUNDARY_WORDS,
                "migrated sham leaked boundary")
        lease_error = destination_access.command(CMD_LEASE)
        execute_error = destination_access.command(CMD_EXECUTE_ATOMIC)
        reuse_error = destination_access.command(CMD_BEGIN_REUSE)
        require(lease_error == ERR_SNAPSHOT_LINEAGE,
                "migrated sham accepted LEASE")
        require(execute_error == ERR_TAG_MISMATCH,
                "migrated sham unexpectedly passed execute request validation")
        require(reuse_error == ERR_SNAPSHOT_LINEAGE,
                "migrated sham accepted BEGIN_REUSE")

        # A sanitized SHAM lineage remains admissible only as another SHAM.
        # This closes the second-hop edge without ever reviving carrier state.
        second_migration_socket = Path("../m2")
        require(len(os.fsencode(second_migration_socket)) < SOCKET_PATH_LIMIT,
                "second migration Unix socket URI path too long")
        require(not (case_root / "m2").exists(),
                "second migration Unix socket already exists")
        second_destination.qmp.execute(
            "migrate-incoming", {"uri": f"unix:{second_migration_socket}"}
        )
        destination.qmp.execute(
            "migrate", {"uri": f"unix:{second_migration_socket}"}
        )
        second_source_status = wait_source_migration(destination.qmp)
        wait_destination_sham(second_destination_access)
        require(second_destination_access.read32(REG_LIFECYCLE) == LIFE_SHAM,
                "second-hop destination did not remain SHAM")
        require(
            second_destination_access.read32(REG_STATUS_LO) & ST_SNAPSHOT_LINEAGE,
            "second-hop destination lost snapshot lineage",
        )
        require(
            second_destination_access.read_boundary() ==
            [LOCKED64] * BOUNDARY_WORDS,
            "second-hop SHAM leaked boundary",
        )

        reset_response = second_destination.qmp.execute("system_reset")
        require(reset_response.get("return") == {},
                "QMP system_reset failed")
        post_reset_pci = second_destination_access.setup_pci()
        reset_lifecycle = second_destination_access.read32(REG_LIFECYCLE)
        reset_status = second_destination_access.read32(REG_STATUS_LO)
        reset_error = second_destination_access.read32(REG_ERROR)
        reset_private_mask = second_destination_access.read32(REG_PRIVATE_READY_MASK)
        reset_return_class = second_destination_access.read32(REG_RETURN_CLASS)
        require(reset_lifecycle == LIFE_SHAM,
                "system_reset escaped the migration SHAM lifecycle: "
                f"{reset_lifecycle}")
        require(reset_status & ST_SNAPSHOT_LINEAGE,
                "system_reset cleared snapshot lineage")
        require(reset_status & ST_SHAM, "system_reset cleared SHAM status")
        require(reset_error == ERR_SNAPSHOT_LINEAGE,
                "system_reset cleared snapshot-lineage error")
        require(reset_private_mask == 0,
                "system_reset restored migrated private state")
        require(reset_return_class == RETURN_FAILED,
                "system_reset restored a non-failed return class")
        require(second_destination_access.read_boundary() ==
                [LOCKED64] * BOUNDARY_WORDS,
                "system_reset released a migrated boundary")
        reset_lease_error = second_destination_access.command(CMD_LEASE)
        reset_reuse_error = second_destination_access.command(CMD_BEGIN_REUSE)
        require(reset_lease_error == ERR_SNAPSHOT_LINEAGE,
                "system_reset made migrated SHAM leasable")
        require(reset_reuse_error == ERR_SNAPSHOT_LINEAGE,
                "system_reset made migrated SHAM reusable")
        evidence = {
            "transport": "REAL_QMP_UNIX_LIVE_MIGRATION",
            "source_query_migrate_status": source_status,
            "source_pci": source_pci,
            "destination_pci": destination_pci,
            "destination_lifecycle": LIFE_SHAM,
            "destination_snapshot_lineage": True,
            "destination_sham": True,
            "destination_private_ready_mask": 0,
            "destination_return_class": RETURN_FAILED,
            "destination_boundary_locked": True,
            "destination_lease_error": lease_error,
            "destination_execute_error": execute_error,
            "destination_reuse_error": reuse_error,
            "second_hop_query_migrate_status": second_source_status,
            "second_destination_pci": second_destination_pci,
            "second_hop_preserved_sham": True,
            "second_hop_boundary_locked": True,
            "qmp_system_reset_issued_after_migration": True,
            "post_reset_pci": post_reset_pci,
            "pci_bar_reenumerated_after_system_reset": True,
            "sham_latched_across_system_reset": True,
            "post_reset_lifecycle": reset_lifecycle,
            "post_reset_snapshot_lineage": True,
            "post_reset_sham": True,
            "post_reset_error": reset_error,
            "post_reset_private_ready_mask": reset_private_mask,
            "post_reset_return_class": reset_return_class,
            "post_reset_boundary_locked": True,
            "post_reset_lease_error": reset_lease_error,
            "post_reset_reuse_error": reset_reuse_error,
        }
    finally:
        source_streams = source.close()
        destination_streams = destination.close()
        second_destination_streams = second_destination.close()
        wall_times[name] = time.monotonic_ns() - started
    require_empty_streams(source_streams, "migration_source")
    require_empty_streams(destination_streams, "migration_destination")
    require_empty_streams(second_destination_streams,
                          "migration_second_destination")
    evidence["captured_process_streams"] = {
        "source": source_streams,
        "destination": destination_streams,
        "second_destination": second_destination_streams,
    }
    return evidence


def build_evidence(qemu_binary: Path, run_root: Path) -> tuple[dict[str, object], dict[str, int]]:
    wall_times: dict[str, int] = {}
    cases: dict[str, object] = {}
    cases["ideal_all_nine_and_reuse"] = run_instance_case(
        qemu_binary,
        run_root,
        "ideal",
        DeviceOptions(),
        ideal_all_pairs_case,
        wall_times,
    )
    cases["sealed_resource_snapshot"] = run_instance_case(
        qemu_binary,
        run_root,
        "rseal",
        DeviceOptions(),
        resource_seal_immutability_case,
        wall_times,
    )
    cases["open_zero_parity"] = run_instance_case(
        qemu_binary,
        run_root,
        "open0",
        DeviceOptions(backend_id=BACKEND_OPEN),
        open_zero_case,
        wall_times,
    )
    cases["open_nonzero_fail_closed"] = run_instance_case(
        qemu_binary,
        run_root,
        "open1",
        DeviceOptions(backend_id=BACKEND_OPEN, open_model_q32=1),
        open_nonzero_case,
        wall_times,
    )
    cases["external_unavailable"] = run_instance_case(
        qemu_binary,
        run_root,
        "ext",
        DeviceOptions(backend_id=BACKEND_EXTERNAL),
        external_case,
        wall_times,
    )
    fault_results: dict[str, object] = {}
    for fault_mode in range(1, FAULT_RESOURCE_UNSEALED + 1):
        name = f"f{fault_mode}"
        fault_results[str(fault_mode)] = run_instance_case(
            qemu_binary,
            run_root,
            name,
            DeviceOptions(fault_mode=fault_mode),
            lambda access, mode=fault_mode: fault_case(access, mode),
            wall_times,
        )
    cases["fault_modes"] = fault_results
    cases["wrong_order_and_generation"] = run_instance_case(
        qemu_binary, run_root, "order", DeviceOptions(), order_case, wall_times
    )
    cases["invalid_mmio_widths"] = run_instance_case(
        qemu_binary,
        run_root,
        "mmio",
        DeviceOptions(),
        invalid_mmio_width_case,
        wall_times,
    )
    cases["service_mode_private_and_observer_rejection"] = run_instance_case(
        qemu_binary,
        run_root,
        "svc",
        DeviceOptions(provider_enabled=False),
        service_mode_rejection_case,
        wall_times,
    )
    cases["missing_private_binding"] = run_instance_case(
        qemu_binary,
        run_root,
        "miss",
        DeviceOptions(),
        missing_private_case,
        wall_times,
    )
    cases["duplicate_private_binding"] = run_instance_case(
        qemu_binary,
        run_root,
        "dup",
        DeviceOptions(),
        duplicate_private_case,
        wall_times,
    )
    cases["private_smuggle_descriptor"] = run_instance_case(
        qemu_binary,
        run_root,
        "smug",
        DeviceOptions(),
        private_smuggle_case,
        wall_times,
    )
    cases["snapshot_command_rejection"] = run_instance_case(
        qemu_binary,
        run_root,
        "snap",
        DeviceOptions(),
        snapshot_case,
        wall_times,
    )
    cases["real_migration_sham"] = migration_case(
        qemu_binary, run_root, wall_times
    )
    process_stream_records: list[dict[str, object]] = []
    for case in cases.values():
        if isinstance(case, dict) and "captured_process_streams" in case:
            streams = case["captured_process_streams"]
            if isinstance(streams, dict) and "stdout_bytes" in streams:
                process_stream_records.append(streams)
    all_streams_empty = all(
        stream["stdout_bytes"] == 0 and stream["stderr_bytes"] == 0
        for stream in process_stream_records
    )
    evidence = {
        "contract": {
            "device_type": DEVICE_TYPE,
            "qom_path": QOM_PATH,
            "pci_identity": [PCI_VENDOR, PCI_DEVICE, PCI_REVISION],
            "magic": MAGIC,
            "abi": ABI,
            "bar0_size": BAR0_SIZE,
            "descriptor_words": list(DESCRIPTOR_WORDS),
            "device_property_names": list(QOM_DEVICE_PROPERTIES),
            "private_property_names": [QOM_PRIVATE_A, QOM_PRIVATE_B],
            "observer_property_names": list(QOM_OBSERVERS),
            "backend_ids": [BACKEND_IDEAL, BACKEND_OPEN, BACKEND_EXTERNAL],
            "fault_modes": list(range(1, FAULT_RESOURCE_UNSEALED + 1)),
            "fault_resource_unsealed": FAULT_RESOURCE_UNSEALED,
            "error_resource_unsealed": ERR_RESOURCE_UNSEALED,
            "error_secret_smuggle": ERR_SECRET_SMUGGLE,
            "error_bad_argument": ERR_BAD_ARGUMENT,
            "resource_schema": RESOURCE_SCHEMA,
            "resource_registers": {
                "allocated_secret_storage_bits": REG_RESOURCE_SECRET_STORAGE_BITS,
                "compiler_ops": REG_RESOURCE_COMPILER_OPS,
                "controller_ops": REG_RESOURCE_CONTROLLER_OPS,
                "construction_ops": REG_RESOURCE_CONSTRUCTION_OPS,
                "logical_secret_entropy_bits": REG_RESOURCE_SECRET_ENTROPY_BITS,
                "carrier_photon_number": REG_RESOURCE_CARRIER_PHOTON_NUMBER,
            },
            "resource_unknown_u64": RESOURCE_UNKNOWN,
            "allocated_private_secret_storage_bits": (
                ALLOCATED_PRIVATE_SECRET_STORAGE_BITS
            ),
            "logical_secret_entropy_bits": LOGICAL_SECRET_ENTROPY_BITS,
        },
        "runner_assumptions": {
            "machine": "q35,accel=qtest",
            "pci_bus": "pcie.0",
            "pci_device_function": [PCI_DEVICE_NUMBER, PCI_FUNCTION],
            "bar0_assigned_base": BAR0_BASE,
            "bar0_size": BAR0_SIZE,
            "qom_path": QOM_PATH,
            "qtest_pci_config_mechanism": "PORT_CF8_CFC",
            "qtest_mmio_endianness": "LITTLE_ENDIAN",
            "migration_transport": "QMP_UNIX_LIVE_MIGRATION",
            "migration_sham_is_reset_irreversible": True,
            "qmp_system_reset_command": "system_reset",
            "reused_lease_lifecycle": LIFE_PREPARED,
            "source_isolation_required_every_generation": True,
            "response_outputs_held_until_ack": True,
            "service_mode_test_provider_enabled": False,
            "service_mode_private_and_observer_qom_rejected": True,
            "qemu_stdout_stderr_required_empty": True,
            "scratch_must_be_explicit_disk_backed_and_run_root_absent": True,
            "unix_socket_paths_must_fit_declared_limit": SOCKET_PATH_LIMIT,
        },
        "cases": cases,
        "all_qemu_stdout_stderr_empty": all_streams_empty,
        "headless_qtest_qmp_only": True,
        "physical_oracle_or_carrier_custody_claim": False,
        "query_separation_or_advantage_claim": False,
        "status": "PASS_PHASE_QEMU_V11_QTEST_QMP_EVIDENCE",
    }
    require(all_streams_empty, "one or more QEMU process streams were nonempty")
    return evidence, wall_times


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run headless Phase-QEMU V11 qtest/QMP evidence"
    )
    parser.add_argument("--qemu-binary", required=True, type=Path)
    parser.add_argument("--scratch-dir", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    started = time.monotonic_ns()
    args = parse_args(sys.argv[1:] if argv is None else argv)
    wall_times: dict[str, int] = {}
    try:
        validated = validate_inputs(args.qemu_binary, args.scratch_dir)
        evidence, wall_times = build_evidence(
            validated["qemu_binary"], validated["run_root"]
        )
        deterministic_evidence = {
            "schema": SCHEMA,
            "qemu_binary_sha256": validated["qemu_binary_sha256"],
            "scratch_filesystem_type": validated["scratch_filesystem_type"],
            "evidence": evidence,
        }
        result = {
            "schema": SCHEMA,
            "status": "PASS_PHASE_QEMU_V11_QTEST_QMP_EVIDENCE",
            "deterministic_evidence": deterministic_evidence,
            "deterministic_payload_sha256": sha256_bytes(
                canonical_bytes(deterministic_evidence)
            ),
            "source_sha256": sha256_file(Path(__file__)),
            "wall_times_ns": {
                "cases": wall_times,
                "total": time.monotonic_ns() - started,
                "excluded_from_deterministic_payload": True,
            },
        }
        print(canonical_bytes(result).decode("utf-8"))
        return 0
    except Exception as error:
        failure = {
            "schema": SCHEMA,
            "status": "FAIL_CLOSED_PHASE_QEMU_V11_QTEST_QMP_EVIDENCE",
            "error_type": type(error).__name__,
            "error": str(error),
            "source_sha256": sha256_file(Path(__file__)),
            "wall_times_ns": {
                "cases": wall_times,
                "total": time.monotonic_ns() - started,
                "excluded_from_deterministic_payload": True,
            },
        }
        print(canonical_bytes(failure).decode("utf-8"))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
