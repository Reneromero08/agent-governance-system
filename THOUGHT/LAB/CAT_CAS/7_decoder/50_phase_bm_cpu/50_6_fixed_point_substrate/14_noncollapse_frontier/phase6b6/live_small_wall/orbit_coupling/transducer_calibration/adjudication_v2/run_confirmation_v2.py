#!/usr/bin/env python3
"""Controller for the frozen prospective Confirmation V2 run."""

from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any

import confirmation_v2_public as public
import confirmation_v2_target as target_model


HERE = Path(__file__).resolve().parent
CALIBRATION_ROOT = HERE.parent
TARGET = "root@192.168.137.100"
REMOTE_BASE = "/root/catcas_live_small_wall"
IMPLEMENTATION_MANIFEST = HERE / "CONFIRMATION_V2_IMPLEMENTATION_MANIFEST.json"
SELF_TEST_PATH = HERE / "CONFIRMATION_V2_SELF_TEST.json"
COMMIT_BINDING_ENV = "CONFIRMATION_V2_COMMIT_BINDING"
LIVE_AUTHORITY_ENV = "CONFIRMATION_V2_LIVE_AUTHORITY"
LIVE_AUTHORITY_VALUE = "balanced_transducer_confirmation_v2_0"
SOURCE_FILE_MAP = {
    HERE / "CONFIRMATION_CONTRACT_V2.md": "CONFIRMATION_CONTRACT_V2.md",
    HERE / "ADJUDICATION_LAW_AUDIT.md": "ADJUDICATION_LAW_AUDIT.md",
    HERE / "CONFIRMATION_PUBLIC_TRIAL_SCHEDULE.json": "CONFIRMATION_PUBLIC_TRIAL_SCHEDULE.json",
    HERE / "CONFIRMATION_PUBLIC_TRIAL_SCHEDULE.sha256": "CONFIRMATION_PUBLIC_TRIAL_SCHEDULE.sha256",
    HERE / "CONFIRMATION_PUBLIC_TRIAL_SCHEDULE.tsv": "CONFIRMATION_PUBLIC_TRIAL_SCHEDULE.tsv",
    HERE / "CONFIRMATION_V2_IMPLEMENTATION_MANIFEST.json": "CONFIRMATION_V2_IMPLEMENTATION_MANIFEST.json",
    HERE / "confirmation_v2_public.py": "confirmation_v2_public.py",
    HERE / "confirmation_v2_runtime.c": "confirmation_v2_runtime.c",
    HERE / "confirmation_v2_runtime.h": "confirmation_v2_runtime.h",
    HERE / "confirmation_v2_target.py": "confirmation_v2_target.py",
    HERE / "run_confirmation_v2.py": "run_confirmation_v2.py",
    HERE / "balanced_transducer_adjudication_v2.py": "balanced_transducer_adjudication_v2.py",
    CALIBRATION_ROOT / "balanced_transducer_public.py": "balanced_transducer_public.py",
}


class ControllerError(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise ControllerError(message)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    return public.sha256_file(path)


def run(command: list[str], *, timeout: float, check: bool = True) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    if check and completed.returncode != 0:
        raise ControllerError(f"command failed ({completed.returncode}): {command!r}\n{completed.stderr.strip()}")
    return completed


def source_hashes() -> dict[str, str]:
    hashes = {}
    for source, remote_name in SOURCE_FILE_MAP.items():
        if remote_name == "CONFIRMATION_V2_IMPLEMENTATION_MANIFEST.json":
            continue
        require(source.is_file(), f"source file missing: {source}")
        hashes[remote_name] = sha256_file(source)
    return hashes


def deterministic_source_bundle(path: Path) -> str:
    bundle_files = [
        (source, remote_name)
        for source, remote_name in SOURCE_FILE_MAP.items()
        if remote_name != "CONFIRMATION_V2_IMPLEMENTATION_MANIFEST.json"
    ]
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w") as archive:
                for source, remote_name in sorted(bundle_files, key=lambda item: item[1]):
                    require(source.is_file(), f"bundle source missing: {source}")
                    data = source.read_bytes()
                    info = tarfile.TarInfo(remote_name)
                    info.size = len(data)
                    info.mtime = 0
                    info.mode = 0o644
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    archive.addfile(info, io.BytesIO(data))
    return sha256_file(path)


def manifest_digest(manifest: dict[str, Any]) -> str:
    return public.digest({k: v for k, v in manifest.items() if k != "implementation_manifest_sha256"})


def execution_manifest_digest(manifest: dict[str, Any]) -> str:
    return public.digest({k: v for k, v in manifest.items() if k != "manifest_sha256"})


def git_head_and_status() -> tuple[str, str]:
    head = run(["git", "rev-parse", "HEAD"], timeout=10).stdout.strip()
    status = run(["git", "status", "--porcelain=v1"], timeout=10).stdout
    return head, status


def git_origin_main() -> str:
    return run(["git", "rev-parse", "origin/main"], timeout=10).stdout.strip()


def is_full_sha(value: str) -> bool:
    return re.fullmatch(r"[0-9a-f]{40}", value) is not None


def compile_runtime_if_available() -> dict[str, Any]:
    compiler = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if not compiler:
        wsl_check = run(["wsl", "--", "gcc", "--version"], timeout=10, check=False)
        if wsl_check.returncode != 0:
            return {"available": False, "compiler": None, "runtime_binary_sha256": None, "note": "no local C compiler on PATH"}
        binary = HERE / "_confirmation_v2_runtime_check"
        wsl_binary = windows_to_wsl_path(binary)
        command = [
            "wsl",
            "--",
            "gcc",
            "-std=c11",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-pedantic",
            "-O2",
            "-march=amdfam10",
            "-mtune=amdfam10",
            "-fno-lto",
            "-pthread",
            "-I",
            windows_to_wsl_path(HERE),
            windows_to_wsl_path(HERE / "confirmation_v2_runtime.c"),
            "-o",
            wsl_binary,
        ]
        try:
            completed = run(command, timeout=30, check=False)
            if completed.returncode != 0:
                return {
                    "available": True,
                    "compiler": "wsl:gcc",
                    "command": command,
                    "returncode": completed.returncode,
                    "runtime_binary_sha256": None,
                    "stderr": completed.stderr[-4000:],
                }
            self_test = run(["wsl", "--", wsl_binary, "--self-test"], timeout=10, check=False)
            return {
                "available": True,
                "compiler": "wsl:gcc",
                "command": command,
                "returncode": completed.returncode,
                "runtime_self_test_returncode": self_test.returncode,
                "runtime_self_test_stdout": self_test.stdout.strip(),
                "runtime_binary_sha256": sha256_file(binary),
            }
        finally:
            if binary.exists():
                binary.unlink()
    with tempfile.TemporaryDirectory(prefix="confirmation_v2_compile_") as temp:
        binary = Path(temp) / ("confirmation_v2_runtime.exe" if sys.platform.startswith("win") else "confirmation_v2_runtime")
        command = [
            compiler,
            "-std=c11",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-pedantic",
            "-O2",
            "-march=amdfam10",
            "-mtune=amdfam10",
            "-fno-lto",
            "-pthread",
            "-I",
            str(HERE),
            str(HERE / "confirmation_v2_runtime.c"),
            "-o",
            str(binary),
        ]
        completed = run(command, timeout=30, check=False)
        if completed.returncode != 0:
            return {
                "available": True,
                "compiler": compiler,
                "command": command,
                "returncode": completed.returncode,
                "runtime_binary_sha256": None,
                "stderr": completed.stderr[-4000:],
            }
        self_test = run([str(binary), "--self-test"], timeout=10, check=False)
        return {
            "available": True,
            "compiler": compiler,
            "command": command,
            "returncode": completed.returncode,
            "runtime_self_test_returncode": self_test.returncode,
            "runtime_self_test_stdout": self_test.stdout.strip(),
            "runtime_binary_sha256": sha256_file(binary),
        }


def windows_to_wsl_path(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    rest = resolved.as_posix().split(":", 1)[1].lstrip("/")
    return f"/mnt/{drive}/{rest}"


def build_self_test() -> dict[str, Any]:
    public_test = public.self_test()
    with tempfile.TemporaryDirectory(prefix="confirmation_v2_target_self_") as temp:
        target_test = target_model.self_test(HERE, Path(temp))
    compile_test = compile_runtime_if_available()
    copyback_failure_status_aware = False
    success_copyback_requires_execution_manifest = False
    with tempfile.TemporaryDirectory(prefix="confirmation_v2_copyback_self_") as temp:
        copyback_root = Path(temp)
        target_failure = {
            "schema_id": "CAT_CAS_CONFIRMATION_V2_TARGET_FAILURE",
            "status": "CONFIRMATION_V2_TARGET_FAILED",
            "run_id": public.RUN_ID,
            "failure_phase": "self_test",
            "exception_type": "ControllerSelfTest",
            "scientific_classification_emitted": False,
        }
        write_json(copyback_root / "TARGET_FAILURE_CONFIRMATION_V2.json", target_failure)
        failure_manifest = {
            "schema_id": "CAT_CAS_CONFIRMATION_V2_FAILURE_MANIFEST",
            "run_id": public.RUN_ID,
            "target_failure_sha256": sha256_file(copyback_root / "TARGET_FAILURE_CONFIRMATION_V2.json"),
            "scientific_classification_emitted": False,
        }
        failure_manifest["manifest_sha256"] = failure_manifest_digest(failure_manifest)
        write_json(copyback_root / "CONFIRMATION_V2_FAILURE_MANIFEST.json", failure_manifest)
        write_json(
            copyback_root / "FINAL_RESULT_CONFIRMATION_V2.json",
            {
                "schema_id": "CAT_CAS_BALANCED_TRANSDUCER_CONFIRMATION_TARGET_RESULT_V2",
                "status": "CONFIRMATION_V2_TARGET_FAILED",
                "run_id": public.RUN_ID,
                "failure_sha256": failure_manifest["target_failure_sha256"],
                "scientific_classification_emitted": False,
            },
        )
        write_json(copyback_root / "LIVE_CUSTODY_LOG.json", {"schema_id": "CAT_CAS_CONFIRMATION_V2_LIVE_CUSTODY_LOG"})
        copyback_manifest = {
            "schema_id": "CAT_CAS_CONFIRMATION_V2_COPYBACK_MANIFEST",
            "run_id": public.RUN_ID,
            "files": [],
        }
        for name in [
            "TARGET_FAILURE_CONFIRMATION_V2.json",
            "CONFIRMATION_V2_FAILURE_MANIFEST.json",
            "FINAL_RESULT_CONFIRMATION_V2.json",
            "LIVE_CUSTODY_LOG.json",
        ]:
            path = copyback_root / name
            copyback_manifest["files"].append({"path": name, "size": path.stat().st_size, "sha256": sha256_file(path)})
        write_json(copyback_root / "COPYBACK_MANIFEST.json", copyback_manifest)
        verified_copyback = verify_copy(copyback_root)
        verified_paths = copyback_paths(verified_copyback)
        require_copyback_entries(
            verified_paths,
            [
                "TARGET_FAILURE_CONFIRMATION_V2.json",
                "CONFIRMATION_V2_FAILURE_MANIFEST.json",
                "LIVE_CUSTODY_LOG.json",
            ],
            "target failure self-test",
        )
        copyback_failure_status_aware = "CONFIRMATION_V2_MANIFEST.json" not in verified_paths
    try:
        require_copyback_entries(
            {"FINAL_RESULT_CONFIRMATION_V2.json"},
            ["CONFIRMATION_V2_MANIFEST.json"],
            "target complete self-test",
        )
    except ControllerError:
        success_copyback_requires_execution_manifest = True
    result = {
        "schema_id": "CAT_CAS_CONFIRMATION_V2_CONTROLLER_SELF_TEST",
        "public_self_test_sha256": public_test["self_test_sha256"],
        "public_self_test_passed": public_test["self_test_passed"],
        "target_self_test_sha256": target_test["self_test_sha256"],
        "target_self_test_passed": target_test["self_test_passed"],
        "compile_test": compile_test,
        "binary_custody_mode": "target_compile_bound_by_source_and_compiler_contract",
        "offline_validation_binary_sha256": compile_test.get("runtime_binary_sha256"),
        "target_binary_self_test_required": True,
        "copyback_failure_status_aware": copyback_failure_status_aware,
        "success_copyback_requires_execution_manifest": success_copyback_requires_execution_manifest,
        "network_connections": 0,
        "ssh_executions": 0,
        "scp_executions": 0,
        "hardware_executions": 0,
        "frequency_writes": 0,
        "voltage_writes": 0,
        "msr_reads": 0,
        "msr_writes": 0,
    }
    result["self_test_passed"] = (
        public_test["self_test_passed"]
        and target_test["self_test_passed"]
        and copyback_failure_status_aware
        and success_copyback_requires_execution_manifest
    )
    result["self_test_sha256"] = public.digest({k: v for k, v in result.items() if k != "self_test_sha256"})
    write_json(SELF_TEST_PATH, result)
    return result


def build_manifest(*, sol_audit: dict[str, Any] | None = None, final_commit: str = "AWAITING_LIVE_AUTHORIZATION") -> dict[str, Any]:
    schedule_hashes = public.write_schedule_artifacts(HERE)
    self_test = build_self_test()
    with tempfile.TemporaryDirectory(prefix="confirmation_v2_bundle_") as temp:
        source_bundle_sha = deterministic_source_bundle(Path(temp) / "CONFIRMATION_SOURCE_BUNDLE.tar.gz")
    contract_sha = sha256_file(HERE / "CONFIRMATION_CONTRACT_V2.md")
    manifest = {
        "schema_id": "CAT_CAS_CONFIRMATION_V2_IMPLEMENTATION_MANIFEST",
        "starting_commit": "8281578a785a27338e5caf8154758ac71adf425c",
        "final_commit": final_commit,
        "contract_sha256": contract_sha,
        "source_hashes": source_hashes(),
        "schedule_json_sha256": schedule_hashes["schedule_json_sha256"],
        "schedule_tsv_sha256": schedule_hashes["schedule_tsv_sha256"],
        "schedule_semantic_sha256": schedule_hashes["schedule_semantic_sha256"],
        "self_test_sha256": self_test["self_test_sha256"],
        "public_self_test_sha256": self_test["public_self_test_sha256"],
        "target_self_test_sha256": self_test["target_self_test_sha256"],
        "expected_source_bundle_sha256": source_bundle_sha,
        "binary_custody": {
            "mode": "target_compile_bound_by_source_and_compiler_contract",
            "offline_validation_binary_sha256": self_test["offline_validation_binary_sha256"],
            "offline_validation_compiler": self_test["compile_test"].get("compiler"),
            "offline_validation_command": self_test["compile_test"].get("command"),
            "target_compiler": "cc",
            "target_strict_c_flags": list(target_model.STRICT_C_FLAGS),
            "target_compile_command_template": [
                "cc",
                *target_model.STRICT_C_FLAGS,
                "-I",
                "<source_root>",
                "<source_root>/confirmation_v2_runtime.c",
                "-o",
                "<source_root>/confirmation_v2_runtime",
            ],
            "target_binary_self_test_required": True,
            "live_runtime_binary_sha256": "RECORDED_ON_TARGET_AFTER_TARGET_COMPILE",
            "transferred_live_binary_sha256": None,
        },
        "expected_run_id": public.RUN_ID,
        "expected_evidence_root": (
            "THOUGHT/LAB/CAT_CAS/7_decoder/50_phase_bm_cpu/50_6_fixed_point_substrate/"
            "14_noncollapse_frontier/phase6b6/live_small_wall/orbit_coupling/"
            "transducer_calibration/runs/balanced_transducer_confirmation_v2_0"
        ),
        "expected_command": (
            ".\\.venv\\Scripts\\python.exe THOUGHT\\LAB\\CAT_CAS\\7_decoder\\50_phase_bm_cpu\\50_6_fixed_point_substrate\\"
            "14_noncollapse_frontier\\phase6b6\\live_small_wall\\orbit_coupling\\transducer_calibration\\adjudication_v2\\"
            "run_confirmation_v2.py --run-id balanced_transducer_confirmation_v2_0 --contract THOUGHT\\LAB\\CAT_CAS\\"
            "7_decoder\\50_phase_bm_cpu\\50_6_fixed_point_substrate\\14_noncollapse_frontier\\phase6b6\\live_small_wall\\"
            "orbit_coupling\\transducer_calibration\\adjudication_v2\\CONFIRMATION_CONTRACT_V2.md --execute-authorized"
        ),
        "allowed_classifications": list(public.ALLOWED_CLASSES),
        "forbidden_classifications": list(public.FORBIDDEN_CLASSES),
        "primary_coordinate": public.PRIMARY_COORDINATE,
        "trial_counts": {
            "replicates": len(public.REPLICATES),
            "trials_per_replicate": public.TRIALS_PER_REPLICATE,
            "total_trial_legs": public.TOTAL_TRIALS,
            "restoration_sentinel_records": public.TOTAL_TRIALS,
        },
        "no_live_contact_attestation": {
            "network_connections": 0,
            "ssh_executions": 0,
            "scp_executions": 0,
            "hardware_executions": 0,
            "frequency_writes": 0,
            "voltage_writes": 0,
            "msr_reads": 0,
            "msr_writes": 0,
        },
        "sol_audit_disposition": sol_audit or {"status": "PENDING_READ_ONLY_SOL_AUDIT"},
        "final_commit_binding": {
            "mode": "exact_authorized_head_environment_gate",
            "placeholder_value": final_commit,
            "commit_binding_env": COMMIT_BINDING_ENV,
            "live_authority_env": LIVE_AUTHORITY_ENV,
            "live_authority_required_value": LIVE_AUTHORITY_VALUE,
            "live_controller_requirement": (
                "before SSH/SCP, current clean git HEAD and origin/main must equal the 40-character SHA supplied in "
                f"{COMMIT_BINDING_ENV}; {LIVE_AUTHORITY_ENV} must separately equal the closed run ID"
            ),
        },
    }
    manifest["implementation_manifest_sha256"] = manifest_digest(manifest)
    write_json(IMPLEMENTATION_MANIFEST, manifest)
    return manifest


def validate_only() -> dict[str, Any]:
    schedule = json.loads((HERE / "CONFIRMATION_PUBLIC_TRIAL_SCHEDULE.json").read_text(encoding="utf-8"))
    public.validate_schedule(schedule)
    require((HERE / "CONFIRMATION_PUBLIC_TRIAL_SCHEDULE.tsv").read_text(encoding="utf-8") == public.schedule_tsv(schedule), "TSV round-trip mismatch")
    require(sha256_file(HERE / "CONFIRMATION_PUBLIC_TRIAL_SCHEDULE.json") == (HERE / "CONFIRMATION_PUBLIC_TRIAL_SCHEDULE.sha256").read_text(encoding="utf-8").strip(), "schedule sha mismatch")
    manifest = json.loads(IMPLEMENTATION_MANIFEST.read_text(encoding="utf-8"))
    self_test = json.loads(SELF_TEST_PATH.read_text(encoding="utf-8"))
    require(manifest["implementation_manifest_sha256"] == manifest_digest(manifest), "manifest self digest mismatch")
    require("expected_runtime_binary_sha256" not in manifest, "obsolete expected runtime binary field present")
    require(manifest["binary_custody"]["mode"] == "target_compile_bound_by_source_and_compiler_contract", "binary custody mode mismatch")
    require(manifest["binary_custody"]["target_binary_self_test_required"], "target binary self-test not required")
    require(self_test["self_test_passed"], "self-test file is not passing")
    require(self_test["self_test_sha256"] == manifest["self_test_sha256"], "self-test manifest mismatch")
    require(source_hashes() == manifest["source_hashes"], "source hashes drifted")
    require(sha256_file(HERE / "CONFIRMATION_PUBLIC_TRIAL_SCHEDULE.json") == manifest["schedule_json_sha256"], "manifest schedule JSON mismatch")
    require(sha256_file(HERE / "CONFIRMATION_PUBLIC_TRIAL_SCHEDULE.tsv") == manifest["schedule_tsv_sha256"], "manifest schedule TSV mismatch")
    with tempfile.TemporaryDirectory(prefix="confirmation_v2_validate_bundle_") as temp:
        require(
            deterministic_source_bundle(Path(temp) / "CONFIRMATION_SOURCE_BUNDLE.tar.gz") == manifest["expected_source_bundle_sha256"],
            "manifest source bundle mismatch",
        )
    return {
        "schema_id": "CAT_CAS_CONFIRMATION_V2_VALIDATE_ONLY",
        "schedule_trial_count": len(schedule["trials"]),
        "manifest_sha256": manifest["implementation_manifest_sha256"],
        "self_test_sha256": self_test["self_test_sha256"],
        "network_connections": 0,
        "ssh_executions": 0,
        "scp_executions": 0,
        "hardware_executions": 0,
    }


def copyback_paths(manifest: dict[str, Any]) -> set[str]:
    return {entry["path"] for entry in manifest["files"]}


def require_copyback_entries(paths: set[str], required: list[str], status: str) -> None:
    missing = sorted(set(required) - paths)
    require(not missing, f"{status} copyback missing required files: {missing}")


def failure_manifest_digest(manifest: dict[str, Any]) -> str:
    return public.digest({k: v for k, v in manifest.items() if k != "manifest_sha256"})


def verify_copy(local_root: Path) -> dict[str, Any]:
    manifest = json.loads((local_root / "COPYBACK_MANIFEST.json").read_text(encoding="utf-8"))
    require(manifest["schema_id"] == "CAT_CAS_CONFIRMATION_V2_COPYBACK_MANIFEST", "copyback schema mismatch")
    require(isinstance(manifest.get("files"), list) and manifest["files"], "copyback files missing")
    paths = copyback_paths(manifest)
    require("FINAL_RESULT_CONFIRMATION_V2.json" in paths, "final result missing from copyback manifest")
    for entry in manifest["files"]:
        relative_path = entry["path"]
        relative_parts = Path(relative_path).parts
        require(
            isinstance(relative_path, str)
            and relative_path
            and ":" not in relative_path
            and not relative_path.startswith(("/", "\\"))
            and ".." not in relative_parts,
            f"unsafe copyback path: {relative_path}",
        )
        path = local_root / relative_path
        require(path.is_file(), f"copied file missing: {entry['path']}")
        require(path.stat().st_size == entry["size"], f"copied size mismatch: {entry['path']}")
        require(sha256_file(path) == entry["sha256"], f"copied sha mismatch: {entry['path']}")
    return manifest


def live_execute(run_id: str, contract: Path, *, keep_remote: bool) -> dict[str, Any]:
    require(run_id == public.RUN_ID, "run ID mismatch")
    require(contract.resolve() == (HERE / "CONFIRMATION_CONTRACT_V2.md").resolve(), "contract path mismatch")
    require(re.fullmatch(r"[a-z0-9_]{8,80}", run_id) is not None, "run ID is not closed")
    manifest = json.loads(IMPLEMENTATION_MANIFEST.read_text(encoding="utf-8"))
    require(manifest["implementation_manifest_sha256"] == manifest_digest(manifest), "implementation manifest self digest mismatch")
    require(sha256_file(contract) == manifest["contract_sha256"], "contract SHA mismatch")
    self_test = json.loads(SELF_TEST_PATH.read_text(encoding="utf-8"))
    require(self_test["self_test_passed"], "self-test file is not passing")
    require(self_test["self_test_sha256"] == manifest["self_test_sha256"], "self-test manifest mismatch")
    require(source_hashes() == manifest["source_hashes"], "source hashes drifted")
    require(sha256_file(HERE / "CONFIRMATION_PUBLIC_TRIAL_SCHEDULE.json") == manifest["schedule_json_sha256"], "manifest schedule JSON mismatch")
    require(sha256_file(HERE / "CONFIRMATION_PUBLIC_TRIAL_SCHEDULE.tsv") == manifest["schedule_tsv_sha256"], "manifest schedule TSV mismatch")
    with tempfile.TemporaryDirectory(prefix="confirmation_v2_live_bundle_") as temp:
        require(
            deterministic_source_bundle(Path(temp) / "CONFIRMATION_SOURCE_BUNDLE.tar.gz") == manifest["expected_source_bundle_sha256"],
            "manifest source bundle mismatch",
        )
    git_head, git_status = git_head_and_status()
    require(git_status.strip() == "", "live execution requires clean git working tree")
    bound_commit = os.environ.get(COMMIT_BINDING_ENV, "").strip()
    require(is_full_sha(bound_commit), f"{COMMIT_BINDING_ENV} must be set to the authorized 40-character commit before live execution")
    require(git_head == bound_commit, "commit binding does not match current HEAD")
    require(git_origin_main() == bound_commit, "origin/main does not match commit binding")
    live_authority = os.environ.get(LIVE_AUTHORITY_ENV, "").strip()
    require(live_authority == LIVE_AUTHORITY_VALUE, f"{LIVE_AUTHORITY_ENV} must equal the closed run ID before live execution")
    remote_run = f"{REMOTE_BASE}/{run_id}"
    remote_source = f"{remote_run}/source"
    remote_output = f"{remote_run}/output"
    local_run = CALIBRATION_ROOT / "runs" / run_id
    require(not local_run.exists(), f"local run already exists: {local_run}")
    validation = validate_only()
    require(validation["network_connections"] == 0 and validation["ssh_executions"] == 0 and validation["scp_executions"] == 0, "validate-only preflight used live transport")
    local_run.mkdir(mode=0o700, parents=True, exist_ok=False)
    preflight = f"set -eu; test ! -e {shlex.quote(remote_run)}; install -d -m 700 -- {shlex.quote(remote_source)}"
    run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", TARGET, preflight], timeout=15)
    for source, remote_name in SOURCE_FILE_MAP.items():
        run(["scp", "-q", str(source), f"{TARGET}:{remote_source}/{remote_name}"], timeout=30)
    remote_command = (
        f"timeout --signal=TERM --kill-after=5s 240s python3 "
        f"{shlex.quote(remote_source + '/confirmation_v2_target.py')} "
        f"--source-root {shlex.quote(remote_source)} "
        f"--output-root {shlex.quote(remote_output)} "
        f"--run-id {shlex.quote(run_id)} "
        f"--expected-manifest-sha {shlex.quote(manifest['implementation_manifest_sha256'])}"
    )
    completed = run(["ssh", "-o", "BatchMode=yes", TARGET, remote_command], timeout=270, check=False)
    (local_run / "CONTROLLER_STDOUT.txt").write_text(completed.stdout, encoding="utf-8")
    (local_run / "CONTROLLER_STDERR.txt").write_text(completed.stderr, encoding="utf-8")
    copied = run(["scp", "-q", "-r", f"{TARGET}:{remote_output}/.", str(local_run)], timeout=120, check=False)
    require(copied.returncode == 0, f"copy-back failed; remote retained at {remote_run}: {copied.stderr.strip()}")
    copy_manifest = verify_copy(local_run)
    copy_paths = copyback_paths(copy_manifest)
    copy_manifest_sha = sha256_file(local_run / "COPYBACK_MANIFEST.json")
    final = json.loads((local_run / "FINAL_RESULT_CONFIRMATION_V2.json").read_text(encoding="utf-8"))
    require(final.get("run_id") == run_id, "final result run ID mismatch")
    final_status = final["status"]
    cleaned = False
    if final_status == "CONFIRMATION_V2_TARGET_COMPLETE":
        require(completed.returncode == 0, "target reported complete with nonzero return code")
        require_copyback_entries(copy_paths, ["CONFIRMATION_V2_MANIFEST.json"], "target complete")
        execution_manifest = json.loads((local_run / "CONFIRMATION_V2_MANIFEST.json").read_text(encoding="utf-8"))
        require(execution_manifest["manifest_sha256"] == execution_manifest_digest(execution_manifest), "execution manifest digest mismatch")
        require(execution_manifest["implementation_manifest_sha256"] == manifest["implementation_manifest_sha256"], "execution manifest implementation SHA mismatch")
        require(execution_manifest["final_result_sha256"] == sha256_file(local_run / "FINAL_RESULT_CONFIRMATION_V2.json"), "execution manifest final-result SHA mismatch")
        if not keep_remote:
            require(remote_run.startswith(REMOTE_BASE + "/") and remote_run != REMOTE_BASE, "unsafe cleanup root")
            cleanup = run(["ssh", "-o", "BatchMode=yes", TARGET, f"rm -rf -- {shlex.quote(remote_run)}"], timeout=20, check=False)
            require(cleanup.returncode == 0, f"verified copy retained but remote cleanup failed: {cleanup.stderr.strip()}")
            absent = run(["ssh", "-o", "BatchMode=yes", TARGET, f"test ! -e {shlex.quote(remote_run)}"], timeout=15, check=False)
            require(absent.returncode == 0, "remote run root remained after cleanup")
            cleaned = True
        controller = {
            "schema_id": "CAT_CAS_CONFIRMATION_V2_CONTROLLER_RESULT",
            "status": "CONFIRMATION_V2_CONTROLLER_TARGET_COMPLETE",
            "run_id": run_id,
            "target": TARGET,
            "remote_run": remote_run,
            "local_run": str(local_run),
            "remote_returncode": completed.returncode,
            "verified_file_count": len(copy_manifest["files"]),
            "copy_verified": True,
            "remote_cleaned": cleaned,
            "remote_retained": not cleaned,
            "final_status": final_status,
            "adjudication_status": final["adjudication_status"],
            "primary_coordinate": final["primary_coordinate"],
            "contract_sha256": manifest["contract_sha256"],
            "implementation_manifest_sha256": manifest["implementation_manifest_sha256"],
            "execution_manifest_sha256": execution_manifest["manifest_sha256"],
            "git_head_at_execution": git_head,
            "commit_binding": bound_commit,
            "origin_main_at_execution": bound_commit,
            "live_authority": live_authority,
            "schedule_json_sha256": final["schedule_json_sha256"],
            "schedule_tsv_sha256": final["schedule_tsv_sha256"],
            "source_bundle_sha256": final["source_bundle_sha256"],
            "binary_custody_mode": final["binary_custody_mode"],
            "offline_validation_binary_sha256": final["offline_validation_binary_sha256"],
            "live_runtime_binary_sha256": final["live_runtime_binary_sha256"],
            "runtime_binary_sha256": final["live_runtime_binary_sha256"],
            "raw_capture_sha256": final["raw_capture_sha256"],
            "restoration_sentinels_sha256": final["restoration_sentinels_sha256"],
            "features_sha256": final["features_sha256"],
            "adjudication_sha256": final["adjudication_sha256"],
            "copyback_manifest_sha256": copy_manifest_sha,
        }
    elif final_status == "CONFIRMATION_V2_TARGET_FAILED":
        require(
            "CONFIRMATION_V2_MANIFEST.json" not in copy_paths,
            "target failure copyback must not include success execution manifest",
        )
        require_copyback_entries(
            copy_paths,
            [
                "TARGET_FAILURE_CONFIRMATION_V2.json",
                "CONFIRMATION_V2_FAILURE_MANIFEST.json",
                "LIVE_CUSTODY_LOG.json",
            ],
            "target failure",
        )
        target_failure_path = local_run / "TARGET_FAILURE_CONFIRMATION_V2.json"
        failure_manifest_path = local_run / "CONFIRMATION_V2_FAILURE_MANIFEST.json"
        target_failure = json.loads(target_failure_path.read_text(encoding="utf-8"))
        failure_manifest = json.loads(failure_manifest_path.read_text(encoding="utf-8"))
        require(failure_manifest["manifest_sha256"] == failure_manifest_digest(failure_manifest), "failure manifest digest mismatch")
        require(failure_manifest["target_failure_sha256"] == sha256_file(target_failure_path), "failure manifest target-failure SHA mismatch")
        require(final["failure_sha256"] == sha256_file(target_failure_path), "final failure SHA mismatch")
        require(final.get("scientific_classification_emitted") is False, "failed target emitted scientific classification")
        require(target_failure.get("scientific_classification_emitted") is False, "failure packet emitted scientific classification")
        require(failure_manifest.get("scientific_classification_emitted") is False, "failure manifest emitted scientific classification")
        require("adjudication_status" not in final, "failed target emitted adjudication status")
        controller = {
            "schema_id": "CAT_CAS_CONFIRMATION_V2_CONTROLLER_RESULT",
            "status": "CONFIRMATION_V2_CONTROLLER_TARGET_FAILED",
            "run_id": run_id,
            "target": TARGET,
            "remote_run": remote_run,
            "local_run": str(local_run),
            "remote_returncode": completed.returncode,
            "verified_file_count": len(copy_manifest["files"]),
            "copy_verified": True,
            "remote_cleaned": False,
            "remote_retained": True,
            "final_status": final_status,
            "scientific_classification_emitted": False,
            "failure_phase": target_failure.get("failure_phase"),
            "exception_type": target_failure.get("exception_type"),
            "contract_sha256": manifest["contract_sha256"],
            "implementation_manifest_sha256": manifest["implementation_manifest_sha256"],
            "git_head_at_execution": git_head,
            "commit_binding": bound_commit,
            "origin_main_at_execution": bound_commit,
            "live_authority": live_authority,
            "failure_manifest_sha256": failure_manifest["manifest_sha256"],
            "target_failure_sha256": failure_manifest["target_failure_sha256"],
            "copyback_manifest_sha256": copy_manifest_sha,
        }
    else:
        raise ControllerError(f"unknown target final status: {final_status}")
    write_json(local_run / "CONTROLLER_RESULT.json", controller)
    print(json.dumps(controller, sort_keys=True))
    return controller


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=public.RUN_ID)
    parser.add_argument("--contract", type=Path, default=HERE / "CONFIRMATION_CONTRACT_V2.md")
    parser.add_argument("--keep-remote", action="store_true")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--self-test", action="store_true")
    modes.add_argument("--validate-only", action="store_true")
    modes.add_argument("--prepare-only", action="store_true")
    modes.add_argument("--execute-authorized", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.self_test:
            result = build_self_test()
            ok = result["self_test_passed"]
        elif args.prepare_only:
            result = build_manifest()
            ok = True
        elif args.validate_only:
            result = validate_only()
            ok = True
        elif args.execute_authorized:
            result = live_execute(args.run_id, args.contract, keep_remote=args.keep_remote)
            ok = result["final_status"] == "CONFIRMATION_V2_TARGET_COMPLETE"
        else:
            raise ControllerError("no execution mode selected")
        print(json.dumps(result, sort_keys=True))
        return 0 if ok else 1
    except Exception as exc:
        print(f"run_confirmation_v2: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
