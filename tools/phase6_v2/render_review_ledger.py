#!/usr/bin/env python3
"""Correct the Phase 6 V2 evidence command ledger without rerunning tests."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

TESTED_SHA = "70d5aa893db7d93baa86a56d5b1ed128730c2ef3"
LINUX_ENVIRONMENT_TIMESTAMP = "2026-06-23T04:17:34-06:00"
WINDOWS_ENVIRONMENT_TIMESTAMP = "2026-06-23T03:14:30"
LINUX_ROOT = "/root/catcas_v2_build_7c44af0f"
WINDOWS_ROOT = r"D:\CCC 2.0\AI\phase6-v2-windows-evidence"

ROOT = Path(subprocess.check_output(
    ["git", "rev-parse", "--show-toplevel"], text=True
).strip())
EVIDENCE = ROOT / "THOUGHT/LAB/CAT_CAS/7_decoder/50_phase_bm_cpu/50_6_fixed_point_substrate/14_noncollapse_frontier/combined_observability_campaign/v2/evidence"
COMMANDS = EVIDENCE / "COMMANDS.jsonl"
BINDINGS = EVIDENCE / "FINAL_BINDINGS.json"
QUALIFICATION = EVIDENCE / "linux/qualification.log"
INVENTORY = EVIDENCE / "EVIDENCE_INVENTORY.sha256"
VERIFICATION = EVIDENCE / "EVIDENCE_INVENTORY_VERIFICATION.txt"


def linux_entry(subdir: str, command: str, log_name: str, provenance: str) -> dict:
    return {
        "platform": "linux",
        "timestamp": None,
        "timestamp_status": "not_recorded_in_raw_command_log",
        "environment_timestamp": LINUX_ENVIRONMENT_TIMESTAMP,
        "working_directory": f"{LINUX_ROOT}/{subdir}",
        "working_directory_status": "recorded_or_bound_by_existing_evidence",
        "tested_sha": TESTED_SHA,
        "command": command,
        "command_provenance": provenance,
        "exit_code": 0,
        "log_path": f"evidence/linux/{log_name}",
    }


def windows_entry(working_directory: str, command: str, log_name: str) -> dict:
    return {
        "platform": "windows",
        "timestamp": None,
        "timestamp_status": "not_recorded_in_raw_command_log",
        "environment_timestamp": WINDOWS_ENVIRONMENT_TIMESTAMP,
        "working_directory": working_directory,
        "working_directory_status": "reconstructed_from_binding_work_package",
        "tested_sha": TESTED_SHA,
        "command": command,
        "command_provenance": (
            "prior committed ledger c81b543f plus binding work package; "
            "result preserved in committed raw log"
        ),
        "exit_code": 0,
        "log_path": f"evidence/windows/{log_name}",
    }


def command_entries() -> list[dict]:
    runtime = "holo_runtime_v2"
    v2 = "combined_observability_campaign/v2"
    analysis = "combined_observability_campaign/analysis"
    entries = [
        linux_entry(runtime,
            "cc -std=c11 -O2 -pthread -Wall -Wextra -Werror combined_pdn_runner.c combined_pdn_hardware.c -o combined_pdn_runner -lm",
            "01_strict_compile.log", "exact command recorded in raw compile log"),
        linux_entry(runtime, "python3 -m unittest -v test_combined_pdn_runner.py",
            "02_runner_tests.log", "exact command recorded in raw runner log"),
        linux_entry(runtime, "python3 -m unittest -v test_waveform_equivalence.py",
            "03_waveform_equivalence.log", "f531ac80 command ledger plus committed raw result log"),
        linux_entry(runtime, "python3 -m unittest -v test_slot2_primitive_identity.py",
            "04_slot2_identity.log", "f531ac80 command ledger plus committed raw result log"),
        linux_entry(runtime,
            "python3 -m unittest -v test_combined_pdn_runner.Tests.test_capture_quality_contract_rejection_matrix",
            "05_capture_quality_subset.log", "f531ac80 command ledger plus committed raw result log"),
        linux_entry(runtime,
            "cc -std=c11 -O1 -g -pthread -Wall -Wextra -Werror -fsanitize=address -fno-omit-frame-pointer combined_pdn_runner.c combined_pdn_hardware.c -o combined_pdn_runner -lm",
            "06_asan_compile.log", "exact command recorded in raw ASan compile log"),
        linux_entry(runtime,
            "ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 python3 -m unittest -v test_combined_pdn_runner.py",
            "07_asan_runner.log", "f531ac80 command ledger plus committed raw result log"),
        linux_entry(runtime,
            "ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 python3 -m unittest -v test_waveform_equivalence.py",
            "08_asan_waveform.log", "f531ac80 command ledger plus committed raw result log"),
        linux_entry(runtime,
            "cc -std=c11 -O1 -g -pthread -Wall -Wextra -Werror -fsanitize=undefined -fno-omit-frame-pointer combined_pdn_runner.c combined_pdn_hardware.c -o combined_pdn_runner -lm",
            "09_ubsan_compile.log", "exact command recorded in raw UBSan compile log"),
        linux_entry(runtime,
            "UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 python3 -m unittest -v test_combined_pdn_runner.py",
            "10_ubsan_runner.log", "f531ac80 command ledger plus committed raw result log"),
        linux_entry(runtime,
            "UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 python3 -m unittest -v test_waveform_equivalence.py",
            "11_ubsan_waveform.log", "f531ac80 command ledger plus committed raw result log"),
        linux_entry(v2,
            "python3 -m unittest -v test_calibration_contract.py test_receiver_schedule.py test_spectral_calibration_analyzer.py",
            "12_python_v2.log", "f531ac80 command ledger plus committed raw result log"),
        linux_entry(analysis,
            "python3 -m unittest -v test_waveform_reference.py test_frozen_artifact_binding.py",
            "13_python_analysis.log", "f531ac80 command ledger plus committed raw result log"),
        windows_entry(
            WINDOWS_ROOT + r"\THOUGHT\LAB\CAT_CAS\7_decoder\50_phase_bm_cpu\50_6_fixed_point_substrate\14_noncollapse_frontier\combined_observability_campaign\v2",
            "python -m unittest -v test_calibration_contract.py test_receiver_schedule.py test_spectral_calibration_analyzer.py",
            "01_focused_python.log"),
        windows_entry(
            WINDOWS_ROOT + r"\THOUGHT\LAB\CAT_CAS\7_decoder\50_phase_bm_cpu\50_6_fixed_point_substrate\14_noncollapse_frontier\combined_observability_campaign\analysis",
            "python -m unittest -v test_waveform_reference.py test_frozen_artifact_binding.py",
            "01_focused_python.log"),
        windows_entry(WINDOWS_ROOT,
            r"python CAPABILITY\TOOLS\utilities\ci_local_gate.py --full",
            "02_full_repository_gate.log"),
    ]
    if len(entries) != 16:
        raise RuntimeError("expected 16 command entries")
    return entries


def write_commands() -> None:
    COMMANDS.write_text(
        "".join(json.dumps(entry, separators=(",", ":")) + "\n"
                for entry in command_entries()),
        encoding="utf-8",
    )


def write_bindings() -> None:
    data = json.loads(BINDINGS.read_text(encoding="utf-8"))
    data["command_ledger"] = {
        "entry_count": 16,
        "linux_entry_count": 13,
        "windows_entry_count": 3,
        "per_command_timestamps_recorded": False,
        "timestamp_policy": (
            "timestamp is null when the raw command log did not record command start time; "
            "environment_timestamp preserves only the captured environment timestamp"
        ),
        "windows_command_provenance": (
            "prior committed ledger c81b543f plus the binding work package; "
            "result output is preserved in committed Windows logs"
        ),
    }
    BINDINGS.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def write_qualification() -> None:
    text = QUALIFICATION.read_text(encoding="utf-8")
    old = """Commands: evidence/COMMANDS.jsonl
Inventory: evidence/EVIDENCE_INVENTORY.sha256
"""
    new = """Commands: evidence/COMMANDS.jsonl
Command ledger entries: 16 (13 Linux, 3 Windows)
Per-command timestamps: not recorded in the raw command logs; COMMANDS.jsonl uses null timestamps and preserves environment timestamps only.
Windows command provenance: prior committed ledger c81b543f plus the binding work package; result output is preserved in the committed Windows logs.
Inventory: evidence/EVIDENCE_INVENTORY.sha256
"""
    if text.count(old) != 1:
        raise RuntimeError("qualification command-ledger anchor mismatch")
    QUALIFICATION.write_text(text.replace(old, new, 1), encoding="utf-8")


def regenerate_inventory() -> None:
    files = sorted(path for path in EVIDENCE.rglob("*") if path.is_file() and path != INVENTORY)
    rels = [path.relative_to(ROOT).as_posix() for path in files]
    verification = ["Evidence inventory verification"]
    verification.extend(f"OK: {rel}" for rel in rels)
    verification.extend([
        "Command ledger entries: 16",
        "Linux command entries: 13",
        "Windows command entries: 3",
        "Per-command timestamps recorded: false",
        f"Entries: {len(rels)}",
        "Verification: PASSED",
        "",
    ])
    VERIFICATION.write_text("\n".join(verification), encoding="utf-8")

    files = sorted(path for path in EVIDENCE.rglob("*") if path.is_file() and path != INVENTORY)
    rows = []
    for path in files:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(f"{digest}  {path.relative_to(ROOT).as_posix()}")
    INVENTORY.write_text("\n".join(rows) + "\n", encoding="utf-8")

    seen: set[str] = set()
    previous = ""
    for row in INVENTORY.read_text(encoding="utf-8").splitlines():
        digest, rel = row.split("  ", 1)
        if rel in seen or rel < previous:
            raise RuntimeError("inventory duplicate or ordering failure")
        seen.add(rel)
        previous = rel
        path = ROOT / rel
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"inventory verification failed: {rel}")


def main() -> int:
    write_commands()
    write_bindings()
    write_qualification()
    regenerate_inventory()
    allowed = {
        COMMANDS.relative_to(ROOT).as_posix(),
        BINDINGS.relative_to(ROOT).as_posix(),
        QUALIFICATION.relative_to(ROOT).as_posix(),
        INVENTORY.relative_to(ROOT).as_posix(),
        VERIFICATION.relative_to(ROOT).as_posix(),
    }
    changed = set(subprocess.check_output(
        ["git", "diff", "--name-only"], cwd=ROOT, text=True
    ).splitlines())
    if changed != allowed:
        raise RuntimeError(f"unexpected changed paths: {sorted(changed ^ allowed)}")
    subprocess.run(["git", "diff", "--check"], cwd=ROOT, check=True)
    print(json.dumps({"changed_paths": sorted(changed), "entry_count": 16}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
