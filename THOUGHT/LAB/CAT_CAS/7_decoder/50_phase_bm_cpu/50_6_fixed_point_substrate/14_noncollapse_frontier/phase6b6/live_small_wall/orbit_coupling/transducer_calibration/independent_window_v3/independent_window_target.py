#!/usr/bin/env python3
"""Target-side helpers for Independent-Window Transducer V3.

This file is intentionally safe to run offline. It validates the frozen source
package and schedule shape without opening network connections or touching PMU
state. Live transport is outside the offline freeze authority.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Any

import independent_window_public as public


HERE = Path(__file__).resolve().parent
STRICT_C_FLAGS = (
    "-std=c11",
    "-Wall",
    "-Wextra",
    "-Werror",
    "-pedantic",
    "-O2",
    "-fno-lto",
)
SOURCE_FILES = (
    "INDEPENDENT_WINDOW_CONTRACT_V3.md",
    "RETRY1_MEASUREMENT_TOPOLOGY_AUDIT.md",
    "INDEPENDENT_WINDOW_PUBLIC_SCHEDULE.json",
    "INDEPENDENT_WINDOW_PUBLIC_SCHEDULE.sha256",
    "INDEPENDENT_WINDOW_PUBLIC_SCHEDULE.tsv",
    "independent_window_public.py",
    "independent_window_runtime.c",
    "independent_window_runtime.h",
    "independent_window_target.py",
    "run_independent_window_v3.py",
)


class TargetError(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise TargetError(message)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def sha256_file(path: Path) -> str:
    return public.sha256_file(path)


def windows_to_wsl_path(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    tail = resolved.as_posix().split(":", 1)[1]
    return f"/mnt/{drive}{tail}"


def source_hashes(source_root: Path) -> dict[str, str]:
    hashes = {}
    for name in SOURCE_FILES:
        path = source_root / name
        if path.is_file():
            hashes[name] = sha256_file(path)
    return hashes


def deterministic_source_bundle(source_root: Path, output_path: Path) -> tuple[str, dict[str, str]]:
    hashes = source_hashes(source_root)
    require(set(SOURCE_FILES).issubset(hashes), "source bundle file set incomplete")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w") as archive:
                for name in sorted(SOURCE_FILES):
                    data = (source_root / name).read_bytes()
                    info = tarfile.TarInfo(name)
                    info.size = len(data)
                    info.mtime = 0
                    info.mode = 0o644
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    archive.addfile(info, io.BytesIO(data))
    return sha256_file(output_path), hashes


def validate_schedule_artifacts(source_root: Path) -> dict[str, Any]:
    schedule_path = source_root / "INDEPENDENT_WINDOW_PUBLIC_SCHEDULE.json"
    tsv_path = source_root / "INDEPENDENT_WINDOW_PUBLIC_SCHEDULE.tsv"
    sha_path = source_root / "INDEPENDENT_WINDOW_PUBLIC_SCHEDULE.sha256"
    require(schedule_path.is_file(), "schedule JSON missing")
    require(tsv_path.is_file(), "schedule TSV missing")
    require(sha_path.is_file(), "schedule SHA missing")
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    public.validate_schedule(schedule)
    require(sha256_file(schedule_path) == sha_path.read_text(encoding="utf-8").strip(), "schedule SHA mismatch")
    require(tsv_path.read_text(encoding="utf-8") == public.schedule_tsv(schedule), "schedule TSV mismatch")
    return {
        "schedule_json_sha256": sha256_file(schedule_path),
        "schedule_tsv_sha256": sha256_file(tsv_path),
        "schedule_semantic_sha256": schedule["schedule_semantic_sha256"],
        "total_mapping_leg_records": len(schedule["trials"]),
        "total_component_measurement_windows": schedule["total_component_measurement_windows"],
    }


def compile_runtime(source_root: Path, binary_path: Path) -> dict[str, Any]:
    compiler = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if compiler is None:
        wsl_check = subprocess.run(
            ["wsl", "--", "gcc", "--version"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
            check=False,
        )
        if wsl_check.returncode != 0:
            return {"available": False, "passed": False, "compiler": None, "binary_sha256": None}
        command = [
            "wsl",
            "--",
            "gcc",
            *STRICT_C_FLAGS,
            "-I",
            windows_to_wsl_path(source_root),
            windows_to_wsl_path(source_root / "independent_window_runtime.c"),
            "-o",
            windows_to_wsl_path(binary_path),
        ]
        runtime_command = ["wsl", "--", windows_to_wsl_path(binary_path)]
        compiler_name = "wsl:gcc"
    else:
        command = [
            compiler,
            *STRICT_C_FLAGS,
            "-I",
            str(source_root),
            str(source_root / "independent_window_runtime.c"),
            "-o",
            str(binary_path),
        ]
        runtime_command = [str(binary_path)]
        compiler_name = compiler
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )
    if completed.returncode != 0:
        return {
            "available": True,
            "passed": False,
            "compiler": compiler_name,
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
            "binary_sha256": None,
        }
    return {
        "available": True,
        "passed": True,
        "compiler": compiler_name,
        "command": command,
        "runtime_command": runtime_command,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "binary_sha256": sha256_file(binary_path),
    }


def run_runtime_self_test(binary_path: Path, runtime_command: list[str] | None = None) -> dict[str, Any]:
    command = list(runtime_command or [str(binary_path)]) + ["--self-test"]
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=10,
        check=False,
    )
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "passed": completed.returncode == 0 and "INDEPENDENT_WINDOW_V3_RUNTIME_SELF_TEST_OK" in completed.stdout,
    }


def run_runtime_schedule_validation(
    binary_path: Path,
    schedule_tsv: Path,
    runtime_command: list[str] | None = None,
) -> dict[str, Any]:
    schedule_arg = str(schedule_tsv)
    if runtime_command and runtime_command[:2] == ["wsl", "--"]:
        schedule_arg = windows_to_wsl_path(schedule_tsv)
    command = list(runtime_command or [str(binary_path)]) + ["--validate-schedule-tsv", schedule_arg]
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=10,
        check=False,
    )
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "passed": completed.returncode == 0 and "INDEPENDENT_WINDOW_V3_SCHEDULE_TSV_OK" in completed.stdout,
    }


def offline_validate(source_root: Path, output_root: Path) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    schedule_receipt = validate_schedule_artifacts(source_root)
    with tempfile.TemporaryDirectory(prefix="independent_window_v3_target_") as temp:
        temp_root = Path(temp)
        bundle_sha, hashes = deterministic_source_bundle(source_root, temp_root / "INDEPENDENT_WINDOW_SOURCE_BUNDLE.tar.gz")
        compile_receipt = compile_runtime(source_root, temp_root / "independent_window_runtime")
        runtime_self = {"passed": False, "skipped": True}
        schedule_tsv_check = {"passed": False, "skipped": True}
        if compile_receipt["passed"]:
            runtime_self = run_runtime_self_test(
                temp_root / "independent_window_runtime",
                compile_receipt.get("runtime_command"),
            )
            schedule_tsv_check = run_runtime_schedule_validation(
                temp_root / "independent_window_runtime",
                source_root / "INDEPENDENT_WINDOW_PUBLIC_SCHEDULE.tsv",
                compile_receipt.get("runtime_command"),
            )
    public_self = public.self_test()
    result = {
        "schema_id": "CAT_CAS_INDEPENDENT_WINDOW_TARGET_OFFLINE_VALIDATION_V3",
        "run_id": public.RUN_ID,
        "schedule": schedule_receipt,
        "source_hashes": hashes,
        "source_bundle_sha256": bundle_sha,
        "compile": compile_receipt,
        "runtime_self_test": runtime_self,
        "runtime_schedule_validation": schedule_tsv_check,
        "public_self_test_sha256": public_self["self_test_sha256"],
        "public_self_test_passed": public_self["self_test_passed"],
        "network_connections": 0,
        "ssh_executions": 0,
        "scp_executions": 0,
    }
    result["passed"] = (
        result["public_self_test_passed"]
        and result["compile"]["passed"]
        and result["runtime_self_test"]["passed"]
        and result["runtime_schedule_validation"]["passed"]
    )
    result["validation_sha256"] = public.digest({k: v for k, v in result.items() if k != "validation_sha256"})
    write_json(output_root / "INDEPENDENT_WINDOW_TARGET_OFFLINE_VALIDATION.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=HERE)
    parser.add_argument("--output-root", type=Path, default=HERE)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--self-test", action="store_true")
    modes.add_argument("--offline-validate", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.self_test:
            with tempfile.TemporaryDirectory(prefix="independent_window_v3_target_self_") as temp:
                result = offline_validate(args.source_root, Path(temp))
        else:
            result = offline_validate(args.source_root, args.output_root)
        print(json.dumps(result, sort_keys=True))
        return 0 if result["passed"] else 1
    except Exception as exc:
        print(f"independent_window_target: {exc}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
