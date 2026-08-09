#!/usr/bin/env python3
"""Install the Phase-QEMU V0 device into an unpacked QEMU 10.2 source tree."""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path


EXPECTED_QEMU_VERSION = "10.2"
KCONFIG_ANCHOR = "config EDU\n    bool\n    default y if TEST_DEVICES\n    depends on PCI && MSI_NONBROKEN\n"
KCONFIG_BLOCK = """

config PHASE_QEMU_V0
    bool
    default y if PCI
    depends on PCI
"""
MESON_ANCHOR = "system_ss.add(when: 'CONFIG_EDU', if_true: files('edu.c'))\n"
MESON_LINE = (
    "system_ss.add(when: 'CONFIG_PHASE_QEMU_V0', "
    "if_true: files('phase-qemu-v0.c'))\n"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def insert_once(path: Path, anchor: str, addition: str) -> None:
    text = path.read_text(encoding="utf-8")
    if addition.strip() in text:
        return
    if anchor not in text:
        raise RuntimeError(f"QEMU anchor missing in {path}")
    path.write_text(text.replace(anchor, anchor + addition, 1), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("qemu_source", type=Path)
    args = parser.parse_args()

    source = args.qemu_source.resolve()
    version_file = source / "VERSION"
    if not version_file.is_file():
        raise RuntimeError("not a QEMU source tree: VERSION is missing")
    version = version_file.read_text(encoding="utf-8").strip()
    if not version.startswith(EXPECTED_QEMU_VERSION + "."):
        raise RuntimeError(f"expected QEMU {EXPECTED_QEMU_VERSION}.x, got {version}")

    package = Path(__file__).resolve().parent
    device_source = package / "qemu" / "hw" / "misc" / "phase-qemu-v0.c"
    target_source = source / "hw" / "misc" / "phase-qemu-v0.c"
    shutil.copyfile(device_source, target_source)
    insert_once(source / "hw" / "misc" / "Kconfig", KCONFIG_ANCHOR, KCONFIG_BLOCK)
    insert_once(source / "hw" / "misc" / "meson.build", MESON_ANCHOR, MESON_LINE)

    print(f"qemu_version={version}")
    print(f"device_sha256={sha256(device_source)}")
    print(f"installed_sha256={sha256(target_source)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
