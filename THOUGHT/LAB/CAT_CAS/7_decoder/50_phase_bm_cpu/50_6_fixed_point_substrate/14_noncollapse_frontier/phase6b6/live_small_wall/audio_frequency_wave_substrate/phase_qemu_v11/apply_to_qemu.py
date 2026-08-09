#!/usr/bin/env python3
"""Fail-closed installer for the Phase-QEMU V11 PCI device.

The installer accepts only an explicitly supplied unpacked QEMU 10.2.4 source
tree.  ``--check`` performs every structural, collision, and integration check
without writing.  Normal mode copies one device source and adds exactly one
Kconfig and one Meson entry.  Existing Phase-QEMU V0/V1 sources and integration
entries are measured before the operation and must remain byte/count identical.
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
DEVICE_FILENAME = "phase-qemu-v11.c"
DEVICE_RELATIVE = Path("hw") / "misc" / DEVICE_FILENAME
DEVICE_ID_PATTERN = re.compile(r"\b0x11fb\b", re.IGNORECASE)
TYPE_PATTERN = re.compile(r'"phase-qemu-v11"')

KCONFIG_ANCHOR = (
    "config EDU\n"
    "    bool\n"
    "    default y if TEST_DEVICES\n"
    "    depends on PCI && MSI_NONBROKEN\n"
)
KCONFIG_BLOCK = (
    "\n\nconfig PHASE_QEMU_V11\n"
    "    bool\n"
    "    default y if PCI\n"
    "    depends on PCI\n"
)
KCONFIG_TOKEN = "config PHASE_QEMU_V11"

MESON_ANCHOR = "system_ss.add(when: 'CONFIG_EDU', if_true: files('edu.c'))\n"
MESON_LINE = (
    "system_ss.add(when: 'CONFIG_PHASE_QEMU_V11', "
    "if_true: files('phase-qemu-v11.c'))\n"
)

REQUIRED_DESCRIPTOR_WORDS = (
    "0x50313144",
    "0x00010008",
    "0x00000002",
    "0x00020102",
    "0x00020011",
    "0x00030021",
    "0x00000003",
    "0x00010001",
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
            self.target_present
            and self.target_current
            and self.kconfig_count == 1
            and self.meson_count == 1
        )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_utf8(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise RuntimeError(f"cannot read UTF-8 source {path}: {exc}") from exc


def validate_installer_ast(path: Path) -> None:
    """Parse this installer and require its fail-closed implementation hooks."""

    try:
        tree = ast.parse(read_utf8(path), filename=str(path))
    except SyntaxError as exc:
        raise RuntimeError(f"installer AST parse failed: {exc}") from exc
    functions = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    required = {
        "validate_installer_ast",
        "validate_c_source",
        "validate_qemu_tree",
        "scan_pci_collisions",
        "integrate",
        "main",
    }
    missing = sorted(required - functions)
    if missing:
        raise RuntimeError(f"installer AST is missing required hooks: {missing}")


def strip_c_comments_and_literals(text: str) -> str:
    """Return C structure while replacing comments and literals with spaces."""

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
                state = "line-comment"
                continue
            if char == "/" and following == "*":
                output.extend("  ")
                index += 2
                state = "block-comment"
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
        if state == "line-comment":
            output.append("\n" if char == "\n" else " ")
            index += 1
            if char == "\n":
                state = "code"
            continue
        if state == "block-comment":
            if char == "*" and following == "/":
                output.extend("  ")
                index += 2
                state = "code"
            else:
                output.append("\n" if char == "\n" else " ")
                index += 1
            continue
        if state == "literal":
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
            continue
        raise AssertionError(f"unexpected scanner state {state}")
    if state in {"block-comment", "literal"}:
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
                raise RuntimeError(
                    f"unbalanced C delimiter {char!r} at byte {offset}"
                )
            stack.pop()
    if stack:
        char, offset = stack[-1]
        raise RuntimeError(f"unclosed C delimiter {char!r} at byte {offset}")
    return stripped


def require_once(text: str, fragment: str, label: str) -> None:
    count = text.count(fragment)
    if count != 1:
        raise RuntimeError(f"expected exactly one {label}, found {count}")


def validate_c_source(path: Path) -> None:
    """Perform a fail-closed structural validation without executing a build."""

    text = read_utf8(path)
    if "\r" in text:
        raise RuntimeError("device source must use LF line endings")
    stripped = validate_balanced_c_structure(text)

    required_once = {
        '#define TYPE_PHASE_QEMU_V11 "phase-qemu-v11"': "QOM type",
        "#define PHASE_V11_DEVICE_ID 0x11fb": "PCI device ID",
        "#define PHASE_V11_REVISION 0x01": "PCI revision",
        "#define PHASE_V11_MAGIC 0x50483131u": "PH11 magic",
        "#define PHASE_V11_ABI 0x00010000u": "ABI",
        "#define PHASE_V11_BAR_SIZE (4 * KiB)": "4K BAR",
        "#define PHASE_V11_STANDALONE_TWIN_QUALIFIES 0":
            "standalone non-qualification gate",
        "typedef struct PhaseV11BackendOps": "backend ops abstraction",
        "static const VMStateDescription vmstate_phase_qemu_v11":
            "VMState description",
        "type_init(phase_qemu_v11_register_types)": "type registration",
    }
    for fragment, label in required_once.items():
        require_once(text, fragment, label)

    for word in REQUIRED_DESCRIPTOR_WORDS:
        require_once(text.lower(), f"uint32_c({word})", f"descriptor word {word}")

    enum_match = re.search(
        r"enum\s+PhaseV11Register\s*\{(?P<body>.*?)\};", stripped, re.S
    )
    if not enum_match:
        raise RuntimeError("PhaseV11Register enum was not found")
    registers = re.findall(
        r"\b(REG_[A-Z0-9_]+)\s*=\s*(0x[0-9a-fA-F]+)",
        enum_match.group("body"),
    )
    if len(registers) < 40:
        raise RuntimeError(f"register map unexpectedly short: {len(registers)}")
    by_offset: dict[int, str] = {}
    for name, encoded in registers:
        offset = int(encoded, 16)
        previous = by_offset.get(offset)
        if previous is not None:
            raise RuntimeError(
                f"register overlap at 0x{offset:03x}: {previous} and {name}"
            )
        if offset >= 0x1000:
            raise RuntimeError(f"register {name} lies outside the 4K BAR")
        by_offset[offset] = name
    if by_offset.get(0x200) != "REG_BOUNDARY_BASE":
        raise RuntimeError("boundary base must be 0x200")
    if by_offset.get(0x278) != "REG_BOUNDARY_LAST":
        raise RuntimeError("boundary last word must be 0x278")

    vmstate_match = re.search(
        r"static\s+const\s+VMStateDescription\s+vmstate_phase_qemu_v11"
        r"\s*=\s*\{(?P<body>.*?)\n\};",
        text,
        re.S,
    )
    if not vmstate_match:
        raise RuntimeError("cannot isolate V11 VMState description")
    vmstate_body = vmstate_match.group("body")
    forbidden_migration_fields = (
        "private_residue",
        "private_bound",
        "density",
        "scratch",
        "descriptor",
        "boundary",
        "arm_nonce",
    )
    leaked = [name for name in forbidden_migration_fields if name in vmstate_body]
    if leaked:
        raise RuntimeError(f"hidden state leaked into VMState: {leaked}")

    if "private_residue_set" not in stripped:
        raise RuntimeError("runtime private QOM setter is missing")
    if "verify_complete_expected_state" not in stripped:
        raise RuntimeError("complete exact-state verifier is missing")
    if "PHASE_V11_STATE_CELLS" not in stripped:
        raise RuntimeError("full density cell accounting is missing")


def validate_canonical_update_target(path: Path) -> None:
    """Recognize a prior V11 revision that normal install may update in place."""

    text = read_utf8(path)
    validate_balanced_c_structure(text)
    anchors = (
        '#define TYPE_PHASE_QEMU_V11 "phase-qemu-v11"',
        "#define PHASE_V11_DEVICE_ID 0x11fb",
        "#define PHASE_V11_MAGIC 0x50483131u",
        "#define PHASE_V11_ABI 0x00010000u",
        "type_init(phase_qemu_v11_register_types)",
    )
    for anchor in anchors:
        require_once(text, anchor, f"canonical update anchor {anchor}")


def insert_once_text(text: str, anchor: str, addition: str, label: str) -> str:
    count = text.count(addition.strip())
    if count > 1:
        raise RuntimeError(f"duplicate {label}: found {count}")
    if count == 1:
        return text
    if text.count(anchor) != 1:
        raise RuntimeError(
            f"expected one QEMU anchor for {label}, found {text.count(anchor)}"
        )
    return text.replace(anchor, anchor + addition, 1)


def validate_integration_texts(kconfig: str, meson: str) -> tuple[int, int]:
    """Reject partial, malformed, or duplicate V11 integration entries."""

    kconfig_count = kconfig.count(KCONFIG_BLOCK.strip())
    kconfig_token_count = kconfig.count(KCONFIG_TOKEN)
    kconfig_symbol_count = kconfig.count("PHASE_QEMU_V11")
    meson_count = meson.count(MESON_LINE.strip())
    meson_symbol_count = meson.count("CONFIG_PHASE_QEMU_V11")
    meson_filename_count = meson.count(DEVICE_FILENAME)
    if not (
        kconfig_count == kconfig_token_count == kconfig_symbol_count
        and kconfig_count <= 1
    ):
        raise RuntimeError(
            "partial, malformed, or duplicate V11 Kconfig integration: "
            f"canonical={kconfig_count}, token={kconfig_token_count}, "
            f"symbol={kconfig_symbol_count}"
        )
    if not (
        meson_count == meson_symbol_count == meson_filename_count
        and meson_count <= 1
    ):
        raise RuntimeError(
            "partial, malformed, or duplicate V11 Meson integration: "
            f"canonical={meson_count}, symbol={meson_symbol_count}, "
            f"filename={meson_filename_count}"
        )
    return kconfig_count, meson_count


def integration_state(source: Path, package_device: Path) -> IntegrationState:
    target = source / DEVICE_RELATIVE
    target_present = target.is_file()
    target_current = target_present and target.read_bytes() == package_device.read_bytes()
    if target_present and not target_current:
        # A prior V11 revision at the canonical path is an update target, not a
        # second PCI claimant.  It must still pass the full V11 source validator.
        validate_canonical_update_target(target)
    kconfig = read_utf8(source / "hw" / "misc" / "Kconfig")
    meson = read_utf8(source / "hw" / "misc" / "meson.build")
    kconfig_count, meson_count = validate_integration_texts(kconfig, meson)
    return IntegrationState(
        target_present, target_current, kconfig_count, meson_count
    )


def scan_pci_collisions(source: Path, target: Path, package_device: Path) -> None:
    """Reject the V11 PCI ID or QOM type anywhere except an exact target."""

    collisions: list[str] = []
    for path in sorted((source / "hw").rglob("*")):
        if not path.is_file() or path.suffix not in {".c", ".h"}:
            continue
        text = read_utf8(path)
        if not DEVICE_ID_PATTERN.search(text) and not TYPE_PATTERN.search(text):
            continue
        if path == target:
            validate_canonical_update_target(path)
            continue
        collisions.append(str(path.relative_to(source)))
    if collisions:
        raise RuntimeError(
            "PCI ID 0x11fb or QOM type phase-qemu-v11 collision: "
            + ", ".join(collisions)
        )


def validate_qemu_tree(source: Path, package_device: Path) -> tuple[str, IntegrationState]:
    version_file = source / "VERSION"
    kconfig = source / "hw" / "misc" / "Kconfig"
    meson = source / "hw" / "misc" / "meson.build"
    if not version_file.is_file() or not kconfig.is_file() or not meson.is_file():
        raise RuntimeError(
            "not an unpacked QEMU source tree: VERSION, hw/misc/Kconfig, "
            "and hw/misc/meson.build are required"
        )
    version = read_utf8(version_file).strip()
    if version != EXPECTED_QEMU_VERSION:
        raise RuntimeError(f"expected QEMU {EXPECTED_QEMU_VERSION}, got {version}")
    target = source / DEVICE_RELATIVE
    scan_pci_collisions(source, target, package_device)
    return version, integration_state(source, package_device)


def phase_legacy_snapshot(source: Path) -> tuple[dict[str, str], tuple[int, int]]:
    hashes: dict[str, str] = {}
    for name in ("phase-qemu-v0.c", "phase-qemu-v1.c"):
        path = source / "hw" / "misc" / name
        if path.is_file():
            hashes[name] = sha256(path)
    kconfig = read_utf8(source / "hw" / "misc" / "Kconfig")
    meson = read_utf8(source / "hw" / "misc" / "meson.build")
    count = (
        len(re.findall(r"(?m)^config PHASE_QEMU_V[01]\s*$", kconfig)),
        len(re.findall(r"CONFIG_PHASE_QEMU_V[01]'", meson)),
    )
    return hashes, count


def integrate(source: Path, package_device: Path) -> IntegrationState:
    """Integrate only after all target contents have been precomputed."""

    target = source / DEVICE_RELATIVE
    kconfig_path = source / "hw" / "misc" / "Kconfig"
    meson_path = source / "hw" / "misc" / "meson.build"
    old_kconfig = read_utf8(kconfig_path)
    old_meson = read_utf8(meson_path)
    new_kconfig = insert_once_text(
        old_kconfig, KCONFIG_ANCHOR, KCONFIG_BLOCK, "V11 Kconfig entry"
    )
    new_meson = insert_once_text(
        old_meson, MESON_ANCHOR, MESON_LINE, "V11 Meson entry"
    )
    prospective_counts = validate_integration_texts(new_kconfig, new_meson)
    if prospective_counts != (1, 1):
        raise RuntimeError(
            "prospective V11 integration is incomplete: "
            f"Kconfig/Meson counts={prospective_counts}"
        )
    legacy_before = phase_legacy_snapshot(source)

    if not target.exists() or target.read_bytes() != package_device.read_bytes():
        shutil.copyfile(package_device, target)
    if new_kconfig != old_kconfig:
        kconfig_path.write_text(new_kconfig, encoding="utf-8")
    if new_meson != old_meson:
        meson_path.write_text(new_meson, encoding="utf-8")

    state = integration_state(source, package_device)
    if not state.installed:
        raise RuntimeError(f"V11 integration incomplete after install: {state}")
    if target.read_bytes() != package_device.read_bytes():
        raise RuntimeError("installed V11 device hash mismatch")
    legacy_after = phase_legacy_snapshot(source)
    if legacy_before != legacy_after:
        raise RuntimeError("Phase-QEMU V0/V1 content or integration counts changed")
    return state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="validate without writes")
    parser.add_argument("qemu_source", type=Path, help="explicit QEMU 10.2.4 source")
    args = parser.parse_args()

    installer = Path(__file__).resolve()
    package = installer.parent
    package_device = package / "qemu" / "hw" / "misc" / DEVICE_FILENAME
    if not package_device.is_file():
        raise RuntimeError(f"package device source is missing: {package_device}")
    source = args.qemu_source.resolve(strict=True)
    if not source.is_dir():
        raise RuntimeError(f"QEMU source is not a directory: {source}")

    validate_installer_ast(installer)
    validate_c_source(package_device)
    version, before = validate_qemu_tree(source, package_device)
    if args.check:
        # Prove a missing entry could be inserted uniquely without writing.
        # An already-installed entry is returned unchanged by insert_once_text.
        prospective_kconfig = insert_once_text(
            read_utf8(source / "hw" / "misc" / "Kconfig"),
            KCONFIG_ANCHOR,
            KCONFIG_BLOCK,
            "V11 Kconfig entry",
        )
        prospective_meson = insert_once_text(
            read_utf8(source / "hw" / "misc" / "meson.build"),
            MESON_ANCHOR,
            MESON_LINE,
            "V11 Meson entry",
        )
        prospective_counts = validate_integration_texts(
            prospective_kconfig, prospective_meson
        )
        if prospective_counts != (1, 1):
            raise RuntimeError(
                "prospective V11 integration is incomplete: "
                f"Kconfig/Meson counts={prospective_counts}"
            )
        state = before
        action = "installed" if state.installed else "ready"
    else:
        state = integrate(source, package_device)
        action = "installed"
        _, state = validate_qemu_tree(source, package_device)
        if not state.installed:
            raise RuntimeError("post-install validation did not find one complete V11")

    print(f"mode={'check' if args.check else 'install'}")
    print(f"qemu_version={version}")
    print(f"integration_state={action}")
    print(f"package_device_sha256={sha256(package_device)}")
    print(f"installer_sha256={sha256(installer)}")
    print(f"target_present={int(state.target_present)}")
    print(f"target_current={int(state.target_current)}")
    print(f"kconfig_entries={state.kconfig_count}")
    print(f"meson_entries={state.meson_count}")
    print(f"qemu_device_source_integrated={int(state.installed)}")
    print("qemu_device_implemented=0")
    print(f"common_guest_visible_contract_source_integrated={int(state.installed)}")
    print("common_guest_visible_contract_compiled=0")
    print("reintegration_gate_passed=0")
    print("compiled_device_qtest_qualification_required=1")
    print("standalone_twin_qualifies=0")
    if state.target_present:
        print(f"installed_device_sha256={sha256(source / DEVICE_RELATIVE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
