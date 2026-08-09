#!/usr/bin/env python3
"""Fail-closed installer for the compiled Phase-QEMU V13 gate device.

Only an explicitly supplied unpacked QEMU 10.2.4 tree is accepted.  Frozen
V0, V1, V11, and V12 device sources plus their canonical Kconfig/Meson entries
must already be present and remain byte-identical.  ``--check`` validates the
prospective integration without writing.  Normal mode installs exactly one
V13 C source and one canonical build-system entry, with recoverable rollback
on any write or post-install validation failure.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path


EXPECTED_QEMU_VERSION = "10.2.4"
DEVICE_FILENAME = "phase-qemu-v13.c"
DEVICE_RELATIVE = Path("hw") / "misc" / DEVICE_FILENAME
DEVICE_ID_PATTERN = re.compile(r"\b0x11fd\b", re.IGNORECASE)
TYPE_PATTERN = re.compile(r'"phase-qemu-v13"')

FROZEN_SOURCE_SHA256 = {
    "phase-qemu-v0.c":
        "b79ec06f870b611142f5df5c97db2f8e34027458da5acc933f90b694b2055764",
    "phase-qemu-v1.c":
        "8b991d8961a6e108d1a4aa7498172564b017c6e622cb8192c6fa15c33638e362",
    "phase-qemu-v11.c":
        "84c2ec576ae54b298046fadcda719dcb4c2e97bbe31aa0ac3c77ae5455027771",
    "phase-qemu-v12.c":
        "5fe4f99e9bf2ff78774e75149c532293adb03c01c035fcac3d05e2fe74b6153f",
}

FROZEN_CONFIGS = ("PHASE_QEMU_V0", "PHASE_QEMU_V1",
                  "PHASE_QEMU_V11", "PHASE_QEMU_V12")
V12_KCONFIG_BLOCK = (
    "config PHASE_QEMU_V12\n"
    "    bool\n"
    "    default y if PCI\n"
    "    depends on PCI\n"
)
V12_MESON_LINE = (
    "system_ss.add(when: 'CONFIG_PHASE_QEMU_V12', "
    "if_true: files('phase-qemu-v12.c'))\n"
)
KCONFIG_BLOCK = (
    "\nconfig PHASE_QEMU_V13\n"
    "    bool\n"
    "    default y if PCI\n"
    "    depends on PCI\n"
)
MESON_LINE = (
    "system_ss.add(when: 'CONFIG_PHASE_QEMU_V13', "
    "if_true: files('phase-qemu-v13.c'))\n"
)

REQUIRED_DESCRIPTOR_WORDS = (
    "0x50313347", "0x00030008", "0x00000006", "0x00000002",
    "0x00000001", "0x00000003", "0x00000000", "0x00000000",
)


@dataclass(frozen=True)
class IntegrationState:
    target_present: bool
    target_current: bool
    kconfig_count: int
    meson_count: int

    @property
    def installed(self) -> bool:
        return (
            self.target_present and self.target_current and
            self.kconfig_count == 1 and self.meson_count == 1
        )


def require_directory_no_symlink(path: Path, label: str) -> None:
    try:
        metadata = os.lstat(path)
    except OSError as exc:
        raise RuntimeError(f"{label} is unavailable: {path}: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode):
        raise RuntimeError(f"{label} must not be a symlink: {path}")
    if not stat.S_ISDIR(metadata.st_mode):
        raise RuntimeError(f"{label} must be a directory: {path}")


def require_regular_file_no_symlink(path: Path, label: str) -> None:
    try:
        metadata = os.lstat(path)
    except OSError as exc:
        raise RuntimeError(f"{label} is unavailable: {path}: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode):
        raise RuntimeError(f"{label} must not be a symlink: {path}")
    if not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError(f"{label} must be a regular file: {path}")


def optional_regular_file_no_symlink(path: Path, label: str) -> bool:
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise RuntimeError(f"cannot inspect {label}: {path}: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode):
        raise RuntimeError(f"{label} must not be a symlink: {path}")
    if not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError(f"{label} must be a regular file: {path}")
    return True


def read_bytes_no_follow(path: Path, label: str) -> bytes:
    require_regular_file_no_symlink(path, label)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise RuntimeError(f"cannot open {label} without following links: {path}: {exc}") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError(f"opened {label} is not regular: {path}")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            return stream.read()
    finally:
        os.close(descriptor)


def write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("short write while updating QEMU integration")
        view = view[written:]


def write_existing_regular_no_follow(path: Path, payload: bytes,
                                     label: str) -> None:
    require_regular_file_no_symlink(path, label)
    flags = os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError(f"opened {label} is not regular: {path}")
        os.ftruncate(descriptor, 0)
        write_all(descriptor, payload)
    finally:
        os.close(descriptor)


def create_regular_exclusive_no_follow(path: Path, payload: bytes,
                                       label: str) -> None:
    require_directory_no_symlink(path.parent, f"{label} parent")
    if optional_regular_file_no_symlink(path, label):
        raise RuntimeError(f"refusing to replace existing {label}: {path}")
    flags = (os.O_WRONLY | os.O_CREAT | os.O_EXCL |
             getattr(os, "O_NOFOLLOW", 0))
    descriptor = os.open(path, flags, 0o644)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError(f"created {label} is not regular: {path}")
        write_all(descriptor, payload)
    finally:
        os.close(descriptor)


def normalize_source_root(path: Path) -> Path:
    absolute = Path(os.path.abspath(os.fspath(path)))
    require_directory_no_symlink(absolute, "QEMU source root")
    try:
        resolved = absolute.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(f"cannot resolve QEMU source root: {exc}") from exc
    if resolved != absolute:
        raise RuntimeError(
            "QEMU source root path contains a symlink component: "
            f"{absolute} -> {resolved}"
        )
    return absolute


def validate_critical_qemu_paths(source: Path) -> None:
    require_directory_no_symlink(source, "QEMU source root")
    require_directory_no_symlink(source / "hw", "QEMU hw directory")
    require_directory_no_symlink(source / "hw" / "misc",
                                 "QEMU hw/misc directory")
    require_regular_file_no_symlink(source / "VERSION", "QEMU VERSION")
    require_regular_file_no_symlink(source / "hw" / "misc" / "Kconfig",
                                    "QEMU hw/misc/Kconfig")
    require_regular_file_no_symlink(source / "hw" / "misc" / "meson.build",
                                    "QEMU hw/misc/meson.build")
    for filename in FROZEN_SOURCE_SHA256:
        require_regular_file_no_symlink(
            source / "hw" / "misc" / filename,
            f"frozen predecessor {filename}",
        )


def sha256(path: Path) -> str:
    return hashlib.sha256(read_bytes_no_follow(path, str(path))).hexdigest()


def read_utf8(path: Path) -> str:
    try:
        return read_bytes_no_follow(path, str(path)).decode("utf-8")
    except (OSError, UnicodeError) as exc:
        raise RuntimeError(f"cannot read UTF-8 source {path}: {exc}") from exc


def require_once(text: str, fragment: str, label: str) -> None:
    count = text.count(fragment)
    if count != 1:
        raise RuntimeError(f"expected exactly one {label}, found {count}")


def validate_installer_ast(path: Path) -> None:
    tree = ast.parse(read_utf8(path), filename=str(path))
    functions = {
        node.name for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    required = {
        "validate_installer_ast", "validate_c_source", "validate_qemu_tree",
        "validate_integration_texts", "validate_v12_prefix",
        "validate_frozen_sources", "scan_pci_collisions", "integrate",
        "rollback_integration", "validate_critical_qemu_paths",
        "normalize_source_root", "read_bytes_no_follow",
        "write_existing_regular_no_follow",
        "create_regular_exclusive_no_follow", "main",
    }
    missing = sorted(required - functions)
    if missing:
        raise RuntimeError(f"installer AST missing required hooks: {missing}")


def strip_c_comments_and_literals(text: str) -> str:
    output: list[str] = []
    index = 0
    state = "code"
    quote = ""
    while index < len(text):
        char = text[index]
        following = text[index + 1] if index + 1 < len(text) else ""
        if state == "code":
            if char == "/" and following == "/":
                output.extend("  ")
                index += 2
                state = "line"
                continue
            if char == "/" and following == "*":
                output.extend("  ")
                index += 2
                state = "block"
                continue
            if char in {'"', "'"}:
                quote = char
                output.append(" ")
                index += 1
                state = "literal"
                continue
            output.append(char)
            index += 1
            continue
        if state == "line":
            output.append("\n" if char == "\n" else " ")
            index += 1
            if char == "\n":
                state = "code"
            continue
        if state == "block":
            if char == "*" and following == "/":
                output.extend("  ")
                index += 2
                state = "code"
            else:
                output.append("\n" if char == "\n" else " ")
                index += 1
            continue
        if char == "\\":
            output.append(" ")
            if following:
                output.append("\n" if following == "\n" else " ")
                index += 2
            else:
                index += 1
            continue
        output.append("\n" if char == "\n" else " ")
        index += 1
        if char == quote:
            state = "code"
    if state in {"block", "literal"}:
        raise RuntimeError(f"unterminated C {state}")
    return "".join(output)


def validate_balanced_c_structure(text: str) -> str:
    stripped = strip_c_comments_and_literals(text)
    opening = {"(": ")", "[": "]", "{": "}"}
    closing = {value: key for key, value in opening.items()}
    stack: list[tuple[str, int]] = []
    for offset, char in enumerate(stripped):
        if char in opening:
            stack.append((char, offset))
        elif char in closing:
            if not stack or stack[-1][0] != closing[char]:
                raise RuntimeError(f"unbalanced C delimiter at byte {offset}")
            stack.pop()
    if stack:
        raise RuntimeError(f"unclosed C delimiter at byte {stack[-1][1]}")
    return stripped


def extract_register_map(text: str, enum_name: str) -> dict[str, int]:
    stripped = strip_c_comments_and_literals(text)
    match = re.search(
        rf"enum\s+{re.escape(enum_name)}\s*\{{(?P<body>.*?)\}};",
        stripped,
        re.S,
    )
    if not match:
        raise RuntimeError(f"register enum {enum_name} is missing")
    pairs = re.findall(
        r"\b(REG_[A-Z0-9_]+)\s*=\s*(0x[0-9a-fA-F]+)",
        match.group("body"),
    )
    result = {name: int(encoded, 16) for name, encoded in pairs}
    if len(result) != len(pairs):
        raise RuntimeError(f"duplicate register names in {enum_name}")
    by_offset: dict[int, str] = {}
    for name, offset in result.items():
        if offset in by_offset:
            raise RuntimeError(
                f"register overlap: {by_offset[offset]} and {name}"
            )
        if offset >= 0x1000:
            raise RuntimeError(f"register {name} lies outside the 4K BAR")
        by_offset[offset] = name
    return result


def validate_c_source(path: Path) -> None:
    text = read_utf8(path)
    if "\r" in text:
        raise RuntimeError("V13 device source must use LF line endings")
    stripped = validate_balanced_c_structure(text)
    required = {
        '#define TYPE_PHASE_QEMU_V13 "phase-qemu-v13"': "QOM type",
        "#define PHASE_V13_DEVICE_ID 0x11fd": "PCI device ID",
        "#define PHASE_V13_MAGIC 0x50483133u": "PH13 magic",
        "#define PHASE_V13_ABI 0x00030000u": "ABI",
        "#define PHASE_V13_BAR_SIZE (4 * KiB)": "4K BAR",
        "#define PHASE_V13_STANDALONE_TWIN_QUALIFIES 0": "promotion gate",
        "#define PHASE_V13_INTERNAL_IDEAL_SUBSTITUTION_ALLOWED 0":
            "anti-substitution gate",
        "#define PHASE_V13_LIVE_HARDWARE_AVAILABLE 0": "hardware absence",
        "#define PHASE_V13_PHYSICAL_OUTPUT_ALLOWED 0": "output ceiling",
        "#define PHASE_V13_CAMPAIGN_STATISTICAL_CLASS_ALLOWED 0":
            "statistical ceiling",
        "COMPILED_COMMON_PHASE_QEMU_V13_FIXED_OFFLINE_HARDWARE_ADAPTER_SELECTOR_GATE_REJECTS_ABSENT_UNENROLLED_UNATTESTED_STALE_REPLAYED_DOWNGRADED_AND_TEST_FIXTURE_SESSIONS_WITHOUT_PUBLISHING_A_PHYSICAL_OUTPUT_OR_CAMPAIGN_STATISTICAL_CERTIFICATE_AND_WITH_EVERY_NONDESTRUCTIVELY_COMPLETED_DISPATCHED_ATTEMPT_TERMINAL_ACK_THEN_SPENT":
            "bounded claim authority",
        "COMPILED_QEMU_10_2_4_PHASE_QEMU_V13_COMMON_PCI_BACKEND_HARDWARE_ABSENCE_AND_FIXED_OFFLINE_SYMBOLIC_SELECTOR_STATE_MACHINE_ONLY":
            "bounded scope authority",
        "V13_ESTABLISHES_COMMON_BACKEND_FAIL_CLOSED_FIXED_SELECTOR_STATE_MACHINE_CONFORMANCE_FOR_DEVICE_ENROLLMENT_ATTESTATION_REPLAY_DOWNGRADE_AND_FIXTURE_DOMAIN_SEPARATION_WITHOUT_A_LIVE_HARDWARE_SESSION_OR_PHYSICAL_OUTPUT_DIRECT_EQUAL_ACCESS_PROTOCOL_COMPARATOR_CONTROLS_AND_M257_REMAINS_INTACT":
            "bounded disposition authority",
        "typedef struct PhaseV13BackendOps": "backend abstraction",
        "static bool hardware_absent_lease_preflight":
            "hardware absence preflight",
        "static uint32_t fixture_dispatch": "offline fixture dispatch",
        "static bool seal_terminal_failure": "terminal failure sealer",
        "static void enter_migration_sham": "migration sanitizer",
        "static int phase_qemu_v13_pre_save": "source migration sanitizer",
        "static int phase_qemu_v13_post_load": "destination migration sham",
        "static void phase_qemu_v13_unrealize": "unrealize sanitizer",
        "static const VMStateDescription vmstate_phase_qemu_v13": "VMState",
        "type_init(phase_qemu_v13_register_types)": "type registration",
    }
    for fragment, label in required.items():
        require_once(text, fragment, label)
    if text.count(".min_access_size = 1") != 2 or \
            text.count(".unaligned = true") != 2:
        raise RuntimeError(
            "V13 MemoryRegionOps must deliver 1..8-byte unaligned accesses "
            "to the MMIO callbacks"
        )
    for word in REQUIRED_DESCRIPTOR_WORDS:
        count = text.lower().count(f"uint32_c({word})")
        expected = 2 if word == "0x00000000" else 1
        if count != expected:
            raise RuntimeError(
                f"descriptor word {word} count is {count}, expected {expected}"
            )

    registers = extract_register_map(text, "PhaseV13Register")
    if len(registers) < 90:
        raise RuntimeError(
            f"V13 register map unexpectedly short: {len(registers)}"
        )
    if registers.get("REG_BOUNDARY_BASE") != 0x200 or \
            registers.get("REG_BOUNDARY_LAST") != 0x278:
        raise RuntimeError("locked V12 boundary offsets changed")
    if registers.get("REG_GATE_STATE") != 0x280:
        raise RuntimeError("V13 extension does not begin at 0x280")

    vmstate = re.search(
        r"static\s+const\s+VMStateDescription\s+vmstate_phase_qemu_v13"
        r"\s*=\s*\{(?P<body>.*?)\n\};",
        text,
        re.S,
    )
    if not vmstate:
        raise RuntimeError("cannot isolate V13 VMState")
    forbidden_vmstate = (
        "leased", "prepared", "descriptor", "boundary", "response",
        "session_epoch", "attestation_nonce", "digest", "receipt",
        "evidence_origin", "channel_security", "device_appraisal",
        "measurement_class", "custody_provenance", "resource_provenance",
        "trust_domain", "lease_expiry_tick",
    )
    leaked = [name for name in forbidden_vmstate if name in vmstate.group("body")]
    if leaked:
        raise RuntimeError(f"ephemeral/live V13 state leaked into VMState: {leaked}")

    if re.search(r"\bideal_[A-Za-z0-9_]+\b", stripped):
        raise RuntimeError("V13 contains an internal ideal-backend symbol")
    production_assignments = re.findall(
        r"trust_domain\s*=\s*TRUST_DOMAIN_PRODUCTION", stripped
    )
    if production_assignments:
        raise RuntimeError("V13 assigns production trust despite absent hardware")
    physical_assignments = re.findall(
        r"measurement_class\s*=\s*"
        r"(?:MEASUREMENT_CLASS_PHYSICAL_SAMPLE|"
        r"MEASUREMENT_CLASS_CAMPAIGN_STATISTICAL_CERTIFICATE)",
        stripped,
    )
    if physical_assignments:
        raise RuntimeError("V13 fixture path assigns a physical evidence class")
    return_assignments = set(re.findall(
        r"return_class\s*=\s*(RETURN_[A-Z0-9_]+)", stripped
    ))
    if not return_assignments <= {"RETURN_NONE", "RETURN_FAILED"}:
        raise RuntimeError(
            f"V13 assigns a forbidden output class: {sorted(return_assignments)}"
        )


def validate_canonical_update_target(path: Path) -> None:
    text = read_utf8(path)
    validate_balanced_c_structure(text)
    for anchor in (
        '#define TYPE_PHASE_QEMU_V13 "phase-qemu-v13"',
        "#define PHASE_V13_DEVICE_ID 0x11fd",
        "#define PHASE_V13_MAGIC 0x50483133u",
        "#define PHASE_V13_ABI 0x00030000u",
        "type_init(phase_qemu_v13_register_types)",
    ):
        require_once(text, anchor, f"canonical V13 update anchor {anchor}")


def insert_once_text(text: str, anchor: str, addition: str, label: str) -> str:
    if text.count(addition.strip()) > 1:
        raise RuntimeError(f"duplicate {label}")
    if text.count(addition.strip()) == 1:
        return text
    if text.count(anchor.strip()) != 1:
        raise RuntimeError(f"expected one canonical anchor for {label}")
    return text.replace(anchor, anchor + addition, 1)


def validate_integration_texts(kconfig: str, meson: str) -> tuple[int, int]:
    for symbol in FROZEN_CONFIGS:
        ktoken = len(re.findall(rf"(?m)^config {symbol}\s*$", kconfig))
        msymbol = len(re.findall(rf"CONFIG_{symbol}'", meson))
        mfilename = meson.count(symbol.lower().replace("_", "-") + ".c")
        if ktoken != 1 or msymbol != 1 or mfilename != 1:
            raise RuntimeError(
                f"frozen {symbol} integration is not canonical: "
                f"kconfig={ktoken}, meson_symbol={msymbol}, "
                f"meson_filename={mfilename}"
            )

    kcanonical = kconfig.count(KCONFIG_BLOCK.strip())
    ktoken = len(re.findall(r"(?m)^config PHASE_QEMU_V13\s*$", kconfig))
    ksymbol = kconfig.count("PHASE_QEMU_V13")
    mcanonical = meson.count(MESON_LINE.strip())
    msymbol = meson.count("CONFIG_PHASE_QEMU_V13")
    mfilename = meson.count(DEVICE_FILENAME)
    if not (kcanonical == ktoken == ksymbol and kcanonical <= 1):
        raise RuntimeError(
            "partial/malformed V13 Kconfig integration: "
            f"canonical={kcanonical}, token={ktoken}, symbol={ksymbol}"
        )
    if not (mcanonical == msymbol == mfilename and mcanonical <= 1):
        raise RuntimeError(
            "partial/malformed V13 Meson integration: "
            f"canonical={mcanonical}, symbol={msymbol}, filename={mfilename}"
        )
    return kcanonical, mcanonical


def validate_frozen_sources(source: Path) -> dict[str, str]:
    observed: dict[str, str] = {}
    for filename, expected in FROZEN_SOURCE_SHA256.items():
        path = source / "hw" / "misc" / filename
        require_regular_file_no_symlink(path, f"frozen predecessor {filename}")
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(
                f"frozen predecessor source changed: {filename}: {actual}"
            )
        observed[filename] = actual
    return observed


def validate_v12_prefix(v12: Path, package_device: Path) -> None:
    v12_map = extract_register_map(read_utf8(v12), "PhaseV12Register")
    v13_map = extract_register_map(read_utf8(package_device), "PhaseV13Register")
    changed = sorted(
        name for name, offset in v12_map.items()
        if v13_map.get(name) != offset
    )
    if changed:
        raise RuntimeError(f"V13 changed the frozen V12 register prefix: {changed}")


def integration_state(source: Path, package_device: Path) -> IntegrationState:
    target = source / DEVICE_RELATIVE
    recovery = target.with_name(target.name + ".failed-install-recovery")
    present = optional_regular_file_no_symlink(target, "V13 target")
    if optional_regular_file_no_symlink(recovery, "V13 recovery target"):
        raise RuntimeError(
            f"recoverable V13 quarantine must be resolved first: {recovery}"
        )
    current = present and (
        read_bytes_no_follow(target, "V13 target") ==
        read_bytes_no_follow(package_device, "package V13 source")
    )
    if present and not current:
        validate_canonical_update_target(target)
        raise RuntimeError(
            "nonidentical V13 target exists; no overwrite revision is allowlisted"
        )
    counts = validate_integration_texts(
        read_utf8(source / "hw" / "misc" / "Kconfig"),
        read_utf8(source / "hw" / "misc" / "meson.build"),
    )
    return IntegrationState(present, current, *counts)


def scan_pci_collisions(source: Path, target: Path) -> None:
    collisions: list[str] = []
    for path in sorted((source / "hw").rglob("*")):
        try:
            metadata = os.lstat(path)
        except OSError as exc:
            raise RuntimeError(
                f"cannot inspect QEMU source candidate {path}: {exc}"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise RuntimeError(
                f"QEMU source candidate must not be a symlink: {path}"
            )
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if path.suffix not in {".c", ".h"}:
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError(
                f"QEMU source candidate must be regular: {path}"
            )
        text = read_utf8(path)
        if not DEVICE_ID_PATTERN.search(text) and not TYPE_PATTERN.search(text):
            continue
        if path == target:
            validate_canonical_update_target(path)
        else:
            collisions.append(str(path.relative_to(source)))
    if collisions:
        raise RuntimeError("V13 PCI/QOM collision: " + ", ".join(collisions))


def validate_qemu_tree(
    source: Path, package_device: Path
) -> tuple[str, IntegrationState]:
    validate_critical_qemu_paths(source)
    version = read_utf8(source / "VERSION").strip()
    if version != EXPECTED_QEMU_VERSION:
        raise RuntimeError(f"expected QEMU {EXPECTED_QEMU_VERSION}, got {version}")
    validate_frozen_sources(source)
    validate_v12_prefix(source / "hw" / "misc" / "phase-qemu-v12.c",
                        package_device)
    scan_pci_collisions(source, source / DEVICE_RELATIVE)
    return version, integration_state(source, package_device)


def frozen_snapshot(source: Path) -> tuple[dict[str, str], tuple[int, int]]:
    hashes = validate_frozen_sources(source)
    kconfig = read_utf8(source / "hw" / "misc" / "Kconfig")
    meson = read_utf8(source / "hw" / "misc" / "meson.build")
    counts = (
        sum(len(re.findall(rf"(?m)^config {symbol}\s*$", kconfig))
            for symbol in FROZEN_CONFIGS),
        sum(len(re.findall(rf"CONFIG_{symbol}'", meson))
            for symbol in FROZEN_CONFIGS),
    )
    return hashes, counts


def prospective_integration(source: Path) -> tuple[str, str]:
    kconfig = read_utf8(source / "hw" / "misc" / "Kconfig")
    meson = read_utf8(source / "hw" / "misc" / "meson.build")
    new_kconfig = insert_once_text(
        kconfig, V12_KCONFIG_BLOCK, KCONFIG_BLOCK, "V13 Kconfig entry"
    )
    new_meson = insert_once_text(
        meson, V12_MESON_LINE, MESON_LINE, "V13 Meson entry"
    )
    if validate_integration_texts(new_kconfig, new_meson) != (1, 1):
        raise RuntimeError("prospective V13 integration is incomplete")
    return new_kconfig, new_meson


def rollback_integration(
    target: Path,
    recovery: Path,
    kpath: Path,
    mpath: Path,
    old_target: bytes | None,
    old_kconfig: str,
    old_meson: str,
) -> str:
    write_existing_regular_no_follow(
        kpath, old_kconfig.encode("utf-8"), "QEMU hw/misc/Kconfig"
    )
    write_existing_regular_no_follow(
        mpath, old_meson.encode("utf-8"), "QEMU hw/misc/meson.build"
    )
    if old_target is not None:
        write_existing_regular_no_follow(target, old_target, "V13 target")
        return "the prior target bytes were restored"
    if optional_regular_file_no_symlink(target, "V13 target"):
        if optional_regular_file_no_symlink(recovery, "V13 recovery target"):
            raise RuntimeError(
                "cannot quarantine the first-install V13 target because "
                f"the recoverable path already exists: {recovery}"
            )
        target.replace(recovery)
        return f"the first-install target was moved recoverably to {recovery}"
    return "the V13 target remained absent"


def integrate(source: Path, package_device: Path) -> IntegrationState:
    target = source / DEVICE_RELATIVE
    recovery = target.with_name(target.name + ".failed-install-recovery")
    target_present = optional_regular_file_no_symlink(target, "V13 target")
    if optional_regular_file_no_symlink(recovery, "V13 recovery target"):
        raise RuntimeError(
            f"recoverable V13 quarantine already exists: {recovery}"
        )
    package_bytes = read_bytes_no_follow(package_device, "package V13 source")
    if target_present and read_bytes_no_follow(target, "V13 target") != package_bytes:
        validate_canonical_update_target(target)
        raise RuntimeError(
            "nonidentical V13 target exists; no overwrite revision is allowlisted"
        )

    kpath = source / "hw" / "misc" / "Kconfig"
    mpath = source / "hw" / "misc" / "meson.build"
    require_regular_file_no_symlink(kpath, "QEMU hw/misc/Kconfig")
    require_regular_file_no_symlink(mpath, "QEMU hw/misc/meson.build")
    old_kconfig = read_utf8(kpath)
    old_meson = read_utf8(mpath)
    old_target = read_bytes_no_follow(target, "V13 target") if target_present else None
    new_kconfig, new_meson = prospective_integration(source)
    frozen_before = frozen_snapshot(source)
    try:
        if new_kconfig != old_kconfig:
            write_existing_regular_no_follow(
                kpath, new_kconfig.encode("utf-8"), "QEMU hw/misc/Kconfig"
            )
        if new_meson != old_meson:
            write_existing_regular_no_follow(
                mpath, new_meson.encode("utf-8"), "QEMU hw/misc/meson.build"
            )
        if not target_present:
            create_regular_exclusive_no_follow(target, package_bytes, "V13 target")
        _, state = validate_qemu_tree(source, package_device)
        if not state.installed:
            raise RuntimeError(f"V13 integration incomplete after install: {state}")
        if frozen_snapshot(source) != frozen_before:
            raise RuntimeError("Phase-QEMU V0/V1/V11/V12 preservation failed")
        validate_frozen_sources(source)
        return state
    except (OSError, RuntimeError) as exc:
        note = rollback_integration(
            target, recovery, kpath, mpath, old_target, old_kconfig, old_meson
        )
        raise RuntimeError(
            f"V13 integration failed and was rolled back; {note}: {exc}"
        ) from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="validate without writes")
    parser.add_argument("qemu_source", type=Path)
    args = parser.parse_args()

    installer = Path(__file__).resolve()
    package_device = installer.parent / "qemu" / DEVICE_FILENAME
    require_regular_file_no_symlink(package_device, "package V13 source")
    source = normalize_source_root(args.qemu_source)

    validate_installer_ast(installer)
    validate_c_source(package_device)
    version, before = validate_qemu_tree(source, package_device)
    if args.check:
        prospective_integration(source)
        state = before
        action = "installed" if state.installed else "ready"
    else:
        state = integrate(source, package_device)
        _, state = validate_qemu_tree(source, package_device)
        if not state.installed:
            raise RuntimeError("post-install V13 validation failed")
        action = "installed"

    print(f"mode={'check' if args.check else 'install'}")
    print(f"qemu_version={version}")
    print(f"integration_state={action}")
    print(f"package_device_sha256={sha256(package_device)}")
    print(f"installer_sha256={sha256(installer)}")
    for filename, digest in sorted(validate_frozen_sources(source).items()):
        print(f"frozen_{filename.replace('-', '_').replace('.', '_')}_sha256={digest}")
    print(f"target_present={int(state.target_present)}")
    print(f"target_current={int(state.target_current)}")
    print(f"kconfig_entries={state.kconfig_count}")
    print(f"meson_entries={state.meson_count}")
    print(f"qemu_device_source_integrated={int(state.installed)}")
    print("qemu_device_implemented=0")
    print(f"common_guest_visible_contract_source_integrated={int(state.installed)}")
    print("common_guest_visible_contract_compiled=0")
    print("reintegration_gate_passed=0")
    print("compiled_qtest_qualification_required=1")
    print("standalone_twin_qualifies=0")
    print("live_hardware_available=0")
    print("physical_output_published=0")
    print("campaign_statistical_certificate_published=0")
    if state.target_present:
        print(f"installed_device_sha256={sha256(source / DEVICE_RELATIVE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
