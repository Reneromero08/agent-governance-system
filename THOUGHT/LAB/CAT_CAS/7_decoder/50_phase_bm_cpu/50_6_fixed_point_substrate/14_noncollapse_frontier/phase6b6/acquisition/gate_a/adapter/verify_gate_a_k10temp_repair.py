#!/usr/bin/env python3
"""Verify the no-contact Gate A k10temp source-repair boundary.

The complete pre-repair adapter and target qualification verifiers are executed
separately against the immutable sealed-evidence head.  This verifier closes the
new source layer: consumed-authority archival, failed-packet immutability,
exact k10temp custody, deterministic bundle bindings, and active-authority
absence.  It opens no network connection and cannot execute hardware.
"""

from __future__ import annotations

import hashlib
import json
import py_compile
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import build_gate_a_execution_bundle as bundle
import gate_a_target_bundle as target_bundle
import gate_a_temperature_custody as temperature

HERE = Path(__file__).resolve().parent
REPO_ROOT = bundle.repo_root().resolve()
SEALED_EVIDENCE_HEAD = "0cb60899f531c185424f605c92be4622dcc1efef"
SEALED_EVIDENCE_TREE = "6c8175bab7d17eda1f1d26865ab793be1e5b97f4"
MERGED_PR37_BASELINE = "e03502b1859d6a3b79699fa56252710ba85f3595"
HISTORICAL_EVIDENCE_TREE = "a02edbfb85bb2b1816a3be92089112f29c639da9"
CONSUMED_AUTHORITY_SHA256 = "7e1e8835bd67590e4e554ae112a2c8a6ca99dd8b9b3a9aafdb23fee31907d682"
CONSUMED_AUTHORITY_BLOB = "709c799f60e30984d3c80715af480fbe5deac952"
ACTIVE_AUTHORITY = HERE / "GATE_A_EXECUTION_AUTHORITY.json"
CONSUMED_AUTHORITY = HERE / "GATE_A_EXECUTION_AUTHORITY_CONSUMED_7e1e8835.json"
MANIFEST = HERE / "GATE_A_EXECUTION_BUNDLE_MANIFEST.json"
FAILED_EVIDENCE = (
    HERE.parents[2]
    / "evidence"
    / "gate_a_engineering_smoke_7e1e8835"
)
HISTORICAL_EVIDENCE = (
    HERE.parents[2]
    / "evidence"
    / "gate_a_target_nonexecuting_qualification_6f243b1a_bundle_abc9e50a"
)
RUNTIME_ROOT = HERE.parents[3] / "holo_runtime_v2"
EXPECTED_TEMPERATURE_PACKAGE_ROLES = {
    "target_execution_gate",
    "target_execution_gate_base",
    "temperature_custody",
}


class VerifyError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerifyError(message)


def run(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(args),
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=check,
    )


def rel(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def committed_sha256(path: Path, treeish: str = "HEAD") -> str:
    completed = subprocess.run(
        ["git", "show", f"{treeish}:{rel(path)}"],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return hashlib.sha256(completed.stdout).hexdigest()


def git_blob(path: Path, treeish: str = "HEAD") -> str:
    return run("git", "rev-parse", f"{treeish}:{rel(path)}").stdout.strip()


def verify_consumed_authority() -> dict[str, Any]:
    require(not ACTIVE_AUTHORITY.exists(), "canonical active authority path must be absent")
    require(CONSUMED_AUTHORITY.is_file() and not CONSUMED_AUTHORITY.is_symlink(), "consumed authority archive missing")
    require(sha256(CONSUMED_AUTHORITY) == CONSUMED_AUTHORITY_SHA256, "consumed authority SHA-256 mismatch")
    require(git_blob(CONSUMED_AUTHORITY) == CONSUMED_AUTHORITY_BLOB, "consumed authority committed blob mismatch")
    require(run("git", "hash-object", str(CONSUMED_AUTHORITY)).stdout.strip() == CONSUMED_AUTHORITY_BLOB, "consumed authority worktree blob mismatch")
    status = run(
        "git", "status", "--porcelain=v1", "--untracked-files=all", "--",
        ":(icase,glob)**/gate_a_execution_authority*.json",
        f":(exclude){rel(HERE / 'schemas' / 'gate_a_execution_authority.schema.json')}",
        check=False,
    )
    require(status.returncode == 0 and status.stdout == "", "execution-authority namespace differs from HEAD")
    tracked = sorted(
        line for line in run("git", "ls-files").stdout.splitlines()
        if Path(line).name.casefold().startswith("gate_a_execution_authority")
        and Path(line).name.casefold().endswith(".json")
        and not line.endswith("schemas/gate_a_execution_authority.schema.json")
    )
    require(tracked == [rel(CONSUMED_AUTHORITY)], f"execution-authority tracked set mismatch: {tracked}")
    value = json.loads(CONSUMED_AUTHORITY.read_text(encoding="utf-8"))
    require(value["maximum_execution_count"] == 1, "consumed authority maximum execution count changed")
    require(value["authority_state"]["automatic_retry"] is False, "consumed authority retry state changed")
    return {
        "status": "CONSUMED_AUTHORITY_ARCHIVED_EXACT",
        "sha256": CONSUMED_AUTHORITY_SHA256,
        "git_blob_sha1": CONSUMED_AUTHORITY_BLOB,
        "active_authority_absent": True,
    }


def verify_failed_packet_immutable() -> dict[str, Any]:
    require(FAILED_EVIDENCE.is_dir() and not FAILED_EVIDENCE.is_symlink(), "failed-attempt evidence root missing")
    sealed_tree = run("git", "rev-parse", f"{SEALED_EVIDENCE_HEAD}^{{tree}}").stdout.strip()
    require(sealed_tree == SEALED_EVIDENCE_TREE, "sealed evidence commit tree mismatch")
    diff = run("git", "diff", "--quiet", SEALED_EVIDENCE_HEAD, "HEAD", "--", rel(FAILED_EVIDENCE), check=False)
    require(diff.returncode == 0, "failed-attempt evidence changed after sealing")
    status = run("git", "status", "--porcelain=v1", "--untracked-files=all", "--", rel(FAILED_EVIDENCE), check=False)
    require(status.returncode == 0 and status.stdout == "", "failed-attempt evidence differs from HEAD")
    return {
        "status": "FAILED_ATTEMPT_EVIDENCE_IMMUTABLE",
        "sealed_head": SEALED_EVIDENCE_HEAD,
        "sealed_tree": SEALED_EVIDENCE_TREE,
    }


def verify_historical_evidence_immutable() -> dict[str, Any]:
    historical = rel(HISTORICAL_EVIDENCE)
    baseline = run("git", "rev-parse", f"{MERGED_PR37_BASELINE}:{historical}")
    current = run("git", "rev-parse", f"HEAD:{historical}")
    require(baseline.stdout.strip() == HISTORICAL_EVIDENCE_TREE, "historical evidence baseline tree changed")
    require(current.stdout.strip() == HISTORICAL_EVIDENCE_TREE, "historical evidence current tree changed")
    worktree = run("git", "diff", "--quiet", "--", historical, check=False)
    index = run("git", "diff", "--cached", "--quiet", "--", historical, check=False)
    require(worktree.returncode == 0 and index.returncode == 0, "historical evidence differs from HEAD")
    return {
        "status": "HISTORICAL_EVIDENCE_IMMUTABLE",
        "baseline_head": MERGED_PR37_BASELINE,
        "tree_sha1": HISTORICAL_EVIDENCE_TREE,
        "worktree_exit_code": worktree.returncode,
        "index_exit_code": index.returncode,
    }


def run_python_suite(pattern: str, minimum: int) -> int:
    tests = subprocess.run(
        [sys.executable, "-B", "-m", "unittest", "discover", "-s", str(HERE), "-p", pattern, "-v"],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    require(tests.returncode == 0, f"{pattern} failed:\n{tests.stdout}\n{tests.stderr}")
    test_count = sum(
        1 for line in tests.stderr.splitlines()
        if line.rstrip().endswith("... ok") or " ... skipped " in line
    )
    require(test_count >= minimum, f"{pattern} test count below minimum")
    return test_count


def run_native_temperature_test() -> dict[str, Any]:
    compiler = shutil.which("cc")
    if compiler is None:
        return {
            "status": "NATIVE_TEST_DEFERRED_NO_LOCAL_COMPILER",
            "hosted_compiler_required": True,
        }
    with tempfile.TemporaryDirectory() as temporary:
        executable = Path(temporary) / "test_gate_a_native_temperature"
        completed = subprocess.run(
            [
                compiler,
                "-DGATE_A_NATIVE_TEMPERATURE_TESTING",
                "-std=c11",
                "-O2",
                "-pthread",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-pedantic",
                str(HERE / "test_gate_a_native_temperature.c"),
                str(RUNTIME_ROOT / "gate_a_engineering_smoke_runtime.c"),
                str(RUNTIME_ROOT / "captured_file.c"),
                f"-I{RUNTIME_ROOT}",
                "-lm",
                "-o",
                str(executable),
            ],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        require(completed.returncode == 0, f"native temperature strict compile failed:\n{completed.stderr}")
        executed = subprocess.run(
            [str(executable)],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        require(executed.returncode == 0, f"native temperature test failed:\n{executed.stdout}\n{executed.stderr}")
        result = json.loads(executed.stdout)
        require(result["status"] == "GATE_A_NATIVE_TEMPERATURE_TEST_OK", "native temperature test status mismatch")
        require(result["hardware_executions"] == 0 and result["network_connections"] == 0, "native temperature test crossed no-drive boundary")
        return {
            "status": result["status"],
            "compiler": Path(compiler).name,
            "hardware_executions": 0,
            "network_connections": 0,
        }


def verify_temperature_sources() -> dict[str, Any]:
    source_paths = [
        HERE / "gate_a_temperature_custody.py",
        HERE / "gate_a_engineering_smoke_executor_k10temp.py",
        HERE / "test_gate_a_temperature_custody.py",
        HERE / "test_gate_a_k10temp_executor.py",
    ]
    for path in source_paths:
        require(path.is_file() and not path.is_symlink(), f"temperature repair source missing: {path.name}")
        py_compile.compile(str(path), doraise=True)
    require(temperature.HWMON_ROOT == Path("/sys/class/hwmon"), "temperature hwmon root changed")
    require(temperature.DRIVER_NAME == "k10temp", "temperature driver changed")
    require(temperature.TEMPERATURE_INPUT == "temp1_input", "temperature input changed")
    require(temperature.MILLIDEGREES_PER_C == 1000, "temperature scale changed")
    require(temperature.VETO_C == 68.0, "temperature veto changed")
    custody_count = run_python_suite("test_gate_a_temperature_custody.py", 9)
    executor_count = run_python_suite("test_gate_a_k10temp_executor.py", 6)
    native = run_native_temperature_test()
    return {
        "status": "K10TEMP_SOURCE_AND_TESTS_PASS",
        "tests_run": custody_count + executor_count,
        "native": native,
        "mutation_baseline": "CLOSED_NATIVE_TEMPERATURE_RECEIPT_MUTATIONS_REJECTED",
        "target_contacts": 0,
        "hardware_executions": 0,
    }


def verify_manifest() -> dict[str, Any]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    expected = bundle.validate_committed_manifest_exact(manifest, "HEAD")
    target_bundle.validate_manifest_shape(expected)
    roles = {entry["role"] for entry in expected["files"]}
    require(EXPECTED_TEMPERATURE_PACKAGE_ROLES <= roles, "temperature sources missing from execution bundle")
    entries = {entry["role"]: entry for entry in expected["files"]}
    require(entries["target_execution_gate"]["source_repository_path"].endswith("gate_a_engineering_smoke_executor_k10temp.py"), "canonical target executor is not k10temp-aware")
    require(entries["target_execution_gate_base"]["source_repository_path"].endswith("gate_a_engineering_smoke_executor.py"), "reviewed base executor missing")
    require(entries["temperature_custody"]["source_repository_path"].endswith("gate_a_temperature_custody.py"), "temperature custody source binding missing")
    manifest_b = bundle.render_manifest("HEAD")
    require(expected == manifest_b, "execution bundle is not deterministic")
    return {
        "status": "K10TEMP_EXECUTION_BUNDLE_EXACT",
        "execution_bundle_sha256": expected["execution_bundle_sha256"],
        "deterministic_archive_sha256": expected["deterministic_archive_sha256"],
        "manifest_sha256": committed_sha256(MANIFEST),
        "file_count": len(expected["files"]),
    }


def verify_no_contact_surface() -> dict[str, Any]:
    texts = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            HERE / "gate_a_temperature_custody.py",
            HERE / "gate_a_engineering_smoke_executor_k10temp.py",
            HERE / "test_gate_a_temperature_custody.py",
            HERE / "test_gate_a_k10temp_executor.py",
            HERE / "test_gate_a_native_temperature.c",
            RUNTIME_ROOT / "gate_a_engineering_smoke_runtime.c",
            RUNTIME_ROOT / "gate_a_engineering_smoke_runtime.h",
        )
    )
    for forbidden in ("ssh ", "scp ", "socket.", "paramiko", "192.168.137.100"):
        require(forbidden not in texts, f"temperature repair exposes target-contact surface: {forbidden}")
    return {"status": "ZERO_CONTACT_SOURCE_SURFACE", "target_contacts": 0}


def main() -> int:
    head = run("git", "rev-parse", "HEAD").stdout.strip()
    require(run("git", "merge-base", "--is-ancestor", SEALED_EVIDENCE_HEAD, head, check=False).returncode == 0, "repair head is not descended from sealed evidence")
    report = {
        "status": "GATE_A_K10TEMP_REPAIR_QUALIFICATION_PASS",
        "head": head,
        "consumed_authority": verify_consumed_authority(),
        "failed_packet": verify_failed_packet_immutable(),
        "historical_evidence": verify_historical_evidence_immutable(),
        "temperature": verify_temperature_sources(),
        "bundle": verify_manifest(),
        "contact_surface": verify_no_contact_surface(),
        "active_authority_created": False,
        "target_contact_count": 0,
        "hardware_execution_count": 0,
        "next_boundary": "INDEPENDENT_EXACT_HEAD_REVIEW_FOR_GATE_A_K10TEMP_REPAIR",
    }
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (VerifyError, temperature.TemperatureCustodyError, bundle.BundleError, target_bundle.TargetBundleError, subprocess.CalledProcessError, OSError, ValueError, json.JSONDecodeError, py_compile.PyCompileError) as exc:
        print(f"verify_gate_a_k10temp_repair: {exc}", file=sys.stderr)
        raise SystemExit(1)
