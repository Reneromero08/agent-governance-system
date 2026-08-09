#!/usr/bin/env python3
"""Fail-closed installer for the Phase-QEMU V12 compiled adapter device.

Only an explicitly supplied unpacked QEMU 10.2.4 tree is accepted.  The
frozen V11 source and its one canonical Kconfig/Meson integration must already
be present and are verified byte-for-byte before and after installation.
``--check`` proves that the exact V12 integration can be produced without
writing.  Normal mode installs one C source plus one Kconfig and Meson entry.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import re
import shutil
from dataclasses import dataclass
from pathlib import Path


EXPECTED_QEMU_VERSION = "10.2.4"
FROZEN_V11_SHA256 = "84c2ec576ae54b298046fadcda719dcb4c2e97bbe31aa0ac3c77ae5455027771"
V11_FILENAME = "phase-qemu-v11.c"
DEVICE_FILENAME = "phase-qemu-v12.c"
DEVICE_RELATIVE = Path("hw") / "misc" / DEVICE_FILENAME
V11_RELATIVE = Path("hw") / "misc" / V11_FILENAME
DEVICE_ID_PATTERN = re.compile(r"\b0x11fc\b", re.IGNORECASE)
TYPE_PATTERN = re.compile(r'"phase-qemu-v12"')
ALLOWED_PRIOR_V12_SHA256 = {
    "1346ddc003ed23ac6384e725611e42794bbbad17e74ffddb50b6c49e3be76e3e",
    "f1aadd53191e20d23dc1764df55fdb8dbd73f147ae1d0d722974036b24fd6ed7",
    "0537189523a5dca87df54c10a31ed114ac915d5f466653745059b77faecff72d",
    "c9321d725fc1d687c872473852cc455314b21fb311183e69ef415506aaa9d143",
}

V11_KCONFIG_BLOCK = (
    "config PHASE_QEMU_V11\n"
    "    bool\n"
    "    default y if PCI\n"
    "    depends on PCI\n"
)
V11_MESON_LINE = (
    "system_ss.add(when: 'CONFIG_PHASE_QEMU_V11', "
    "if_true: files('phase-qemu-v11.c'))\n"
)
KCONFIG_BLOCK = (
    "\nconfig PHASE_QEMU_V12\n"
    "    bool\n"
    "    default y if PCI\n"
    "    depends on PCI\n"
)
MESON_LINE = (
    "system_ss.add(when: 'CONFIG_PHASE_QEMU_V12', "
    "if_true: files('phase-qemu-v12.c'))\n"
)

REQUIRED_DESCRIPTOR_WORDS = (
    "0x50313144", "0x00010008", "0x00000002", "0x00020102",
    "0x00020011", "0x00030021", "0x00000003", "0x00010001",
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


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_utf8(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
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
        "validate_integration_texts", "validate_v11_prefix",
        "scan_pci_collisions", "integrate", "main",
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
                output.extend("  "); index += 2; state = "line"; continue
            if char == "/" and following == "*":
                output.extend("  "); index += 2; state = "block"; continue
            if char in {'"', "'"}:
                quote = char; output.append(" "); index += 1; state = "literal"; continue
            output.append(char); index += 1; continue
        if state == "line":
            output.append("\n" if char == "\n" else " ")
            index += 1
            if char == "\n": state = "code"
            continue
        if state == "block":
            if char == "*" and following == "/":
                output.extend("  "); index += 2; state = "code"
            else:
                output.append("\n" if char == "\n" else " "); index += 1
            continue
        if state == "literal":
            if char == "\\":
                output.append(" ")
                if following:
                    output.append("\n" if following == "\n" else " "); index += 2
                else:
                    index += 1
                continue
            output.append("\n" if char == "\n" else " "); index += 1
            if char == quote: state = "code"
            continue
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


def validate_c_source(path: Path) -> None:
    text = read_utf8(path)
    if "\r" in text:
        raise RuntimeError("V12 device source must use LF line endings")
    stripped = validate_balanced_c_structure(text)
    required = {
        '#define TYPE_PHASE_QEMU_V12 "phase-qemu-v12"': "QOM type",
        "#define PHASE_V12_DEVICE_ID 0x11fc": "PCI device ID",
        "#define PHASE_V12_MAGIC 0x50483132u": "PH12 magic",
        "#define PHASE_V12_ABI 0x00020000u": "ABI",
        "#define PHASE_V12_BAR_SIZE (4 * KiB)": "4K BAR",
        "#define PHASE_V12_STANDALONE_TWIN_QUALIFIES 0": "promotion gate",
        "typedef struct PhaseV12BackendOps": "backend abstraction",
        "static const VMStateDescription vmstate_phase_qemu_v12": "VMState",
        "type_init(phase_qemu_v12_register_types)": "type registration",
        "static uint32_t external_poll": "asynchronous poll",
        "static uint32_t external_verify_return": "external return classifier",
        "static void enter_migration_sham": "migration sanitizer",
        "static int phase_qemu_v12_pre_save": "source migration sanitizer",
        "static void phase_qemu_v12_unrealize": "unrealize sanitizer",
        "static uint64_t resource_carrier_photon_number": "physical unknown gate",
        "bool adapter_authenticated_lineage;": "committed authentication latch",
    }
    for fragment, label in required.items():
        require_once(text, fragment, label)
    for word in REQUIRED_DESCRIPTOR_WORDS:
        require_once(text.lower(), f"uint32_c({word})", f"descriptor word {word}")
    enum = re.search(r"enum\s+PhaseV12Register\s*\{(?P<body>.*?)\};", stripped, re.S)
    if not enum:
        raise RuntimeError("PhaseV12Register enum missing")
    registers = re.findall(
        r"\b(REG_[A-Z0-9_]+)\s*=\s*(0x[0-9a-fA-F]+)", enum.group("body")
    )
    if len(registers) < 60:
        raise RuntimeError(f"V12 register map unexpectedly short: {len(registers)}")
    by_offset: dict[int, str] = {}
    for name, encoded in registers:
        offset = int(encoded, 16)
        if offset in by_offset:
            raise RuntimeError(f"register overlap: {by_offset[offset]} and {name}")
        if offset >= 0x1000:
            raise RuntimeError(f"register {name} outside 4K BAR")
        by_offset[offset] = name
    if by_offset.get(0x200) != "REG_BOUNDARY_BASE" or by_offset.get(0x278) != "REG_BOUNDARY_LAST":
        raise RuntimeError("locked boundary offsets changed")
    vmstate = re.search(
        r"static\s+const\s+VMStateDescription\s+vmstate_phase_qemu_v12"
        r"\s*=\s*\{(?P<body>.*?)\n\};", text, re.S
    )
    if not vmstate:
        raise RuntimeError("cannot isolate V12 VMState")
    forbidden = ("adapter_envelope", "private_residue", "density", "scratch",
                 "descriptor", "boundary", "arm_nonce")
    leaked = [name for name in forbidden if name in vmstate.group("body")]
    if leaked:
        raise RuntimeError(f"hidden adapter state leaked into VMState: {leaked}")


def validate_canonical_update_target(path: Path) -> None:
    text = read_utf8(path)
    validate_balanced_c_structure(text)
    require_once(text, '#define TYPE_PHASE_QEMU_V12 "phase-qemu-v12"',
                 "canonical V12 update type")
    for anchor in (
        "#define PHASE_V12_DEVICE_ID 0x11fc",
        "#define PHASE_V12_MAGIC 0x50483132u",
        "#define PHASE_V12_ABI 0x00020000u",
        "type_init(phase_qemu_v12_register_types)",
    ):
        require_once(text, anchor, f"canonical V12 update anchor {anchor}")


def insert_once_text(text: str, anchor: str, addition: str, label: str) -> str:
    if text.count(addition.strip()) > 1:
        raise RuntimeError(f"duplicate {label}")
    if text.count(addition.strip()) == 1:
        return text
    if text.count(anchor.strip()) != 1:
        raise RuntimeError(f"expected one canonical anchor for {label}")
    return text.replace(anchor, anchor + addition, 1)


def validate_integration_texts(kconfig: str, meson: str) -> tuple[int, int]:
    kcanonical = kconfig.count(KCONFIG_BLOCK.strip())
    ktoken = len(re.findall(r"(?m)^config PHASE_QEMU_V12\s*$", kconfig))
    ksymbol = kconfig.count("PHASE_QEMU_V12")
    mcanonical = meson.count(MESON_LINE.strip())
    msymbol = meson.count("CONFIG_PHASE_QEMU_V12")
    mfilename = meson.count(DEVICE_FILENAME)
    if not (kcanonical == ktoken == ksymbol and kcanonical <= 1):
        raise RuntimeError(
            "partial/malformed V12 Kconfig integration: "
            f"canonical={kcanonical}, token={ktoken}, symbol={ksymbol}"
        )
    if not (mcanonical == msymbol == mfilename and mcanonical <= 1):
        raise RuntimeError(
            "partial/malformed V12 Meson integration: "
            f"canonical={mcanonical}, symbol={msymbol}, filename={mfilename}"
        )
    if kconfig.count(V11_KCONFIG_BLOCK.strip()) != 1:
        raise RuntimeError("frozen V11 Kconfig integration is not canonical")
    if meson.count(V11_MESON_LINE.strip()) != 1:
        raise RuntimeError("frozen V11 Meson integration is not canonical")
    return kcanonical, mcanonical


def integration_state(source: Path, package_device: Path) -> IntegrationState:
    target = source / DEVICE_RELATIVE
    present = target.is_file()
    current = present and target.read_bytes() == package_device.read_bytes()
    if present and not current:
        if sha256(target) not in ALLOWED_PRIOR_V12_SHA256:
            raise RuntimeError(
                "nonidentical V12 target is not an explicitly allowlisted prior revision"
            )
        validate_canonical_update_target(target)
    counts = validate_integration_texts(
        read_utf8(source / "hw" / "misc" / "Kconfig"),
        read_utf8(source / "hw" / "misc" / "meson.build"),
    )
    return IntegrationState(present, current, *counts)


def scan_pci_collisions(source: Path, target: Path) -> None:
    collisions: list[str] = []
    for path in sorted((source / "hw").rglob("*")):
        if not path.is_file() or path.suffix not in {".c", ".h"}:
            continue
        text = read_utf8(path)
        if not DEVICE_ID_PATTERN.search(text) and not TYPE_PATTERN.search(text):
            continue
        if path == target:
            validate_canonical_update_target(path)
        else:
            collisions.append(str(path.relative_to(source)))
    if collisions:
        raise RuntimeError("V12 PCI/QOM collision: " + ", ".join(collisions))


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
        r"\b(REG_[A-Z0-9_]+)\s*=\s*(0x[0-9a-fA-F]+)", match.group("body")
    )
    result = {name: int(encoded, 16) for name, encoded in pairs}
    if len(result) != len(pairs):
        raise RuntimeError(f"duplicate names in {enum_name}")
    return result


def validate_v11_prefix(v11: Path, package_device: Path) -> None:
    v11_map = extract_register_map(read_utf8(v11), "PhaseV11Register")
    v12_map = extract_register_map(read_utf8(package_device), "PhaseV12Register")
    missing = sorted(name for name in v11_map if v12_map.get(name) != v11_map[name])
    if missing:
        raise RuntimeError(f"V12 changed the frozen V11 register prefix: {missing}")


def validate_qemu_tree(source: Path, package_device: Path) -> tuple[str, IntegrationState]:
    for path in (source / "VERSION", source / "hw" / "misc" / "Kconfig",
                 source / "hw" / "misc" / "meson.build"):
        if not path.is_file():
            raise RuntimeError(f"QEMU source prerequisite missing: {path}")
    version = read_utf8(source / "VERSION").strip()
    if version != EXPECTED_QEMU_VERSION:
        raise RuntimeError(f"expected QEMU {EXPECTED_QEMU_VERSION}, got {version}")
    v11 = source / V11_RELATIVE
    if not v11.is_file() or sha256(v11) != FROZEN_V11_SHA256:
        raise RuntimeError("frozen V11 source is absent or changed")
    validate_v11_prefix(v11, package_device)
    scan_pci_collisions(source, source / DEVICE_RELATIVE)
    return version, integration_state(source, package_device)


def frozen_snapshot(source: Path) -> tuple[dict[str, str], tuple[int, int]]:
    hashes = {
        name: sha256(source / "hw" / "misc" / name)
        for name in ("phase-qemu-v0.c", "phase-qemu-v1.c", V11_FILENAME)
        if (source / "hw" / "misc" / name).is_file()
    }
    kconfig = read_utf8(source / "hw" / "misc" / "Kconfig")
    meson = read_utf8(source / "hw" / "misc" / "meson.build")
    counts = (
        len(re.findall(r"(?m)^config PHASE_QEMU_V(?:0|1|11)\s*$", kconfig)),
        len(re.findall(r"CONFIG_PHASE_QEMU_V(?:0|1|11)'", meson)),
    )
    return hashes, counts


def prospective_integration(source: Path) -> tuple[str, str]:
    kconfig = read_utf8(source / "hw" / "misc" / "Kconfig")
    meson = read_utf8(source / "hw" / "misc" / "meson.build")
    new_kconfig = insert_once_text(
        kconfig, V11_KCONFIG_BLOCK, KCONFIG_BLOCK, "V12 Kconfig entry"
    )
    new_meson = insert_once_text(
        meson, V11_MESON_LINE, MESON_LINE, "V12 Meson entry"
    )
    if validate_integration_texts(new_kconfig, new_meson) != (1, 1):
        raise RuntimeError("prospective V12 integration is incomplete")
    return new_kconfig, new_meson


def integrate(source: Path, package_device: Path) -> IntegrationState:
    target = source / DEVICE_RELATIVE
    failed_target_recovery = target.with_name(
        target.name + ".failed-install-recovery"
    )
    if target.is_symlink() or (target.exists() and not target.is_file()):
        raise RuntimeError(
            "refusing to replace a symlink or non-regular V12 target: "
            f"{target}"
        )
    kpath = source / "hw" / "misc" / "Kconfig"
    mpath = source / "hw" / "misc" / "meson.build"
    old_kconfig = read_utf8(kpath)
    old_meson = read_utf8(mpath)
    old_target = target.read_bytes() if target.is_file() else None
    new_kconfig, new_meson = prospective_integration(source)
    frozen_before = frozen_snapshot(source)
    try:
        # Integration texts are prevalidated and written before the device;
        # an existing canonical device has an exact byte rollback image.
        if new_kconfig != old_kconfig:
            kpath.write_text(new_kconfig, encoding="utf-8")
        if new_meson != old_meson:
            mpath.write_text(new_meson, encoding="utf-8")
        if not target.exists() or target.read_bytes() != package_device.read_bytes():
            shutil.copyfile(package_device, target)
    except OSError as exc:
        kpath.write_text(old_kconfig, encoding="utf-8")
        mpath.write_text(old_meson, encoding="utf-8")
        if old_target is not None:
            target.write_bytes(old_target)
            recovery_note = "the prior target bytes were restored"
        elif target.exists():
            if (failed_target_recovery.exists() or
                    failed_target_recovery.is_symlink()):
                raise RuntimeError(
                    "V12 integration write failed; the partial target could "
                    "not be moved because its recoverable quarantine path "
                    f"already exists: {failed_target_recovery}"
                ) from exc
            target.replace(failed_target_recovery)
            recovery_note = (
                "the partial first-install target was moved recoverably to "
                f"{failed_target_recovery}"
            )
        else:
            recovery_note = "the target remained absent"
        raise RuntimeError(
            f"V12 integration write failed and was rolled back; {recovery_note}: {exc}"
        ) from exc
    state = integration_state(source, package_device)
    if not state.installed:
        raise RuntimeError(f"V12 integration incomplete after install: {state}")
    if frozen_before != frozen_snapshot(source):
        raise RuntimeError("Phase-QEMU V0/V1/V11 content or integration changed")
    return state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="validate without writes")
    parser.add_argument("qemu_source", type=Path)
    args = parser.parse_args()
    installer = Path(__file__).resolve()
    package_device = installer.parent / "qemu" / DEVICE_FILENAME
    if not package_device.is_file():
        raise RuntimeError(f"package V12 device missing: {package_device}")
    source = args.qemu_source.resolve(strict=True)
    if not source.is_dir():
        raise RuntimeError("QEMU source argument is not a directory")
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
            raise RuntimeError("post-install validation failed")
        action = "installed"
    print(f"mode={'check' if args.check else 'install'}")
    print(f"qemu_version={version}")
    print(f"integration_state={action}")
    print(f"package_device_sha256={sha256(package_device)}")
    print(f"installer_sha256={sha256(installer)}")
    print(f"frozen_v11_sha256={sha256(source / V11_RELATIVE)}")
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
    if state.target_present:
        print(f"installed_device_sha256={sha256(source / DEVICE_RELATIVE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
