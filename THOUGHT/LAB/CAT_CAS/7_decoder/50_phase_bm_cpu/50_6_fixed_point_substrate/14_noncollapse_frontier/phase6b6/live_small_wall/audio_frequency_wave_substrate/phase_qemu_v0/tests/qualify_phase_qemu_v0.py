#!/usr/bin/env python3
"""Reexecute and byte-qualify the Phase-QEMU V0 evidence package."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_managed_scratch(path: Path) -> Path:
    resolved = path.resolve()
    text = str(resolved)
    if "/Codex/Scratch/turns/" not in text:
        raise RuntimeError("qualifier scratch must be a codex-scratch managed payload")
    if text.startswith("/dev/shm/") or text.startswith("/run/shm/"):
        raise RuntimeError("RAM-backed scratch is forbidden")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def run(command: list[str], cwd: Path) -> None:
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    subprocess.run(command, cwd=cwd, env=env, check=True)


def stable_migration_result(result: dict[str, object]) -> dict[str, object]:
    stable = json.loads(json.dumps(result))
    stable.pop("migration_stream_sha256", None)
    stable.pop("migration_stream_bytes", None)
    resource = stable["resource_law"]
    assert isinstance(resource, dict)
    resource.pop("migration_stream_bytes", None)
    return stable


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qemu", required=True, type=Path)
    parser.add_argument("--qemu-source", required=True, type=Path)
    parser.add_argument("--scratch", required=True, type=Path)
    args = parser.parse_args()

    package = Path(__file__).resolve().parent.parent
    scratch = require_managed_scratch(args.scratch)
    qemu = args.qemu.resolve()
    qemu_source = args.qemu_source.resolve()
    actual = scratch / "phase-qemu-v0-qtest-regenerated.json"
    reference = scratch / "phase-qemu-v0-reference-regenerated.json"
    migration = scratch / "phase-qemu-v0-migration-regenerated.json"

    run(
        [
            sys.executable,
            "-B",
            "tests/run_phase_qemu_v0_qtest.py",
            "--qemu",
            str(qemu),
            "--output",
            str(actual),
        ],
        package,
    )
    run(
        [
            sys.executable,
            "-B",
            "tests/run_phase_qemu_v0_migration_sham.py",
            "--qemu",
            str(qemu),
            "--scratch",
            str(scratch / "migration"),
            "--qtest-result",
            str(actual),
            "--output",
            str(migration),
        ],
        package,
    )
    run(
        [
            sys.executable,
            "-B",
            "tests/phase_qemu_v0_separate_reference.py",
            "--result",
            str(actual),
            "--device-source",
            "qemu/hw/misc/phase-qemu-v0.c",
            "--output",
            str(reference),
        ],
        package,
    )

    sealed_actual = package / "evidence" / "PHASE_QEMU_V0_P0_QTEST_RESULTS.json"
    sealed_reference = (
        package / "evidence" / "PHASE_QEMU_V0_P0_SEPARATE_REFERENCE.json"
    )
    sealed_migration = (
        package / "evidence" / "PHASE_QEMU_V0_P0_MIGRATION_SHAM.json"
    )
    if actual.read_bytes() != sealed_actual.read_bytes():
        raise AssertionError("regenerated qtest evidence differs from the seal")
    if reference.read_bytes() != sealed_reference.read_bytes():
        raise AssertionError("regenerated separate reference differs from the seal")
    regenerated_migration = json.loads(migration.read_text(encoding="utf-8"))
    recorded_migration = json.loads(
        sealed_migration.read_text(encoding="utf-8")
    )
    if stable_migration_result(regenerated_migration) != stable_migration_result(
        recorded_migration
    ):
        raise AssertionError("regenerated migration semantics differ from the seal")

    result = json.loads(sealed_actual.read_text(encoding="utf-8"))
    oracle = json.loads(sealed_reference.read_text(encoding="utf-8"))
    receipt = json.loads(
        (package / "evidence" / "PHASE_QEMU_V0_BUILD_RECEIPT.json").read_text(
            encoding="utf-8"
        )
    )
    if not all(result["controls"].values()):
        raise AssertionError("a production control is false")
    if not all(oracle["controls"].values()):
        raise AssertionError("an independent control is false")
    if not all(recorded_migration["controls"].values()):
        raise AssertionError("a migration sham control is false")
    if sha256(qemu) != result["qemu_binary_sha256"]:
        raise AssertionError("QEMU binary differs from the executed seal")
    if sha256(qemu) != receipt["qemu_binary_sha256"]:
        raise AssertionError("QEMU binary differs from the build receipt")
    device = package / "qemu" / "hw" / "misc" / "phase-qemu-v0.c"
    if sha256(device) != oracle["device_source_audit"]["source_sha256"]:
        raise AssertionError("device source differs from the independent audit")
    installed_device = qemu_source / "hw" / "misc" / "phase-qemu-v0.c"
    if sha256(installed_device) != sha256(device):
        raise AssertionError("built QEMU tree does not contain the sealed device source")
    if (qemu_source / "VERSION").read_text(encoding="utf-8").strip() != "10.2.4":
        raise AssertionError("unexpected QEMU source version")
    for relative, expected in receipt["source_dependencies"].items():
        if sha256(package / relative) != expected:
            raise AssertionError(f"source dependency differs: {relative}")
    if sha256(sealed_actual) != receipt["sealed_qtest_result_sha256"]:
        raise AssertionError("qtest seal differs from build receipt")
    if sha256(sealed_reference) != receipt["sealed_separate_reference_sha256"]:
        raise AssertionError("reference seal differs from build receipt")
    if sha256(sealed_migration) != receipt["sealed_migration_sham_sha256"]:
        raise AssertionError("migration seal differs from build receipt")
    if result["classification"]["restoration_classification"] != "NO_RESTORATION_CLAIM":
        raise AssertionError("P0 V0 must not acquire a restoration claim")
    if any(result["claim_limits"].values()):
        raise AssertionError("one strict negative claim limit was promoted")
    if recorded_migration["classification"]["recovery_classification"] != "SNAPSHOT_RELOAD":
        raise AssertionError("actual QEMU migration must remain a reload sham")
    if recorded_migration["classification"]["accepted_catalytic_path"]:
        raise AssertionError("migration was promoted into the catalytic path")

    print("PASS_STRICT_SCOPE PHASE_QEMU_V0_P0_REFERENCE_MODEL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
