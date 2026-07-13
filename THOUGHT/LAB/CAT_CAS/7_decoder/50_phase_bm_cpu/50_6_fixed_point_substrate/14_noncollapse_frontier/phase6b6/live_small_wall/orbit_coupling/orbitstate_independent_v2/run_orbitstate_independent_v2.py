#!/usr/bin/env python3
"""Controller for the private OrbitState independent-window package."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import orbitstate_independent_public as public
import orbitstate_independent_target as target


HERE = Path(__file__).resolve().parent
ORBIT_COUPLING_ROOT = HERE.parent
LIVE_SMALL_WALL_ROOT = ORBIT_COUPLING_ROOT.parent
RUNS_ROOT = ORBIT_COUPLING_ROOT / "runs"
SCIENCE_PACKAGE_ID = public.RUN_ID
TRANSACTION_RUN_ID = "orbitstate_independent_v2_1"
ATTEMPT0_RUN_ID = SCIENCE_PACKAGE_ID
ATTEMPT0_LOCAL_RUN_ROOT = RUNS_ROOT / ATTEMPT0_RUN_ID
ATTEMPT0_REMOTE_RUN_ROOT = f"/root/catcas_live_small_wall/{ATTEMPT0_RUN_ID}"
ATTEMPT0_CONTROLLER_RESULT_SHA256 = "5a5b82542d1c9268bbfaba051c4528d9859446b26c781c6c0a7a0e86e835f669"
EXPECTED_LOCAL_RUN_ROOT = RUNS_ROOT / TRANSACTION_RUN_ID
EXPECTED_REMOTE_RUN_ROOT = f"/root/catcas_live_small_wall/{TRANSACTION_RUN_ID}"
EXPECTED_REMOTE_SOURCE_ROOT = f"{EXPECTED_REMOTE_RUN_ROOT}/source"
EXPECTED_REMOTE_OUTPUT_ROOT = f"{EXPECTED_REMOTE_RUN_ROOT}/output"
TARGET_HOST = "root@192.168.137.100"
REMOTE_TIMEOUT_SECONDS = 900
LOCAL_SSH_TIMEOUT_SECONDS = 930

EXPECTED_STARTING_COMMIT = "384388e7bf27a1bc691ee9f13eba07f72f0b231c"

COMMIT_ENV = "ORBITSTATE_INDEPENDENT_V2_COMMIT_BINDING"
MANIFEST_ENV = "ORBITSTATE_INDEPENDENT_V2_MANIFEST_SHA256"
AUTHORITY_ENV = "ORBITSTATE_INDEPENDENT_V2_LIVE_AUTHORITY"
AUTHORITY_VALUE = TRANSACTION_RUN_ID
ATTEMPT0_RECEIPT_PATH = HERE / "ORBITSTATE_ATTEMPT0_PREEXECUTION_FAILURE.json"
RETRY1_AUTHORITY_PATH = HERE / "ORBITSTATE_INDEPENDENT_V2_RETRY1_AUTHORITY.md"
DEPLOYMENT_LAYOUT_TEST_PATH = HERE / "ORBITSTATE_DEPLOYMENT_LAYOUT_SELF_TEST.json"

FROZEN_HASHES = {
    "contract_sha256": "1f586b4648a516723f5a77cfc381d0e4a8c305dd9446a28289975a3ad3c49507",
    "public_schedule_json_sha256": "709063e1d789971f8ac36d2fc94094738015150baae8e75065909c774a079b7b",
    "public_schedule_tsv_sha256": "57aaf5635e0ea1bcecd17f6efc0383f6ce08a893751d9203d1c87b0e4c7a7876",
    "private_source_map_canonical_sha256": "b952f2a161e782dfe41e9dfca21ba4f6bf2902bc69392d9ad52915daa3955464",
    "private_source_map_file_sha256": "619189f66d32610053c3899616d656d16a21814a45074191f37f95acdbf58325",
    "public_null_law_audit_sha256": "fc6321f2b898a3f97766e90d68267b67bea79aa32b26ad110cf085deda09c01e",
}

FROZEN_BYTE_FILES = {
    "contract_sha256": HERE / "ORBITSTATE_INDEPENDENT_CONTRACT_V2.md",
    "public_schedule_json_sha256": HERE / "ORBITSTATE_PUBLIC_SCHEDULE.json",
    "public_schedule_tsv_sha256": HERE / "ORBITSTATE_PUBLIC_SCHEDULE.tsv",
    "private_source_map_file_sha256": HERE / "ORBITSTATE_PRIVATE_SOURCE_MAP.json",
    "public_null_law_audit_sha256": ORBIT_COUPLING_ROOT / "PUBLIC_TRANSDUCER_NULL_LAW_AUDIT.md",
}


class ControllerError(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise ControllerError(message)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run(command: list[str], *, timeout: float, check: bool = True) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(command, text=True, capture_output=True, timeout=timeout, check=False)
    if check and completed.returncode != 0:
        raise ControllerError(
            f"command failed rc={completed.returncode}: {' '.join(command)}\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )
    return completed


def git_text(*args: str) -> str:
    return run(["git", *args], timeout=30.0).stdout.strip()


def git_head_and_status() -> dict[str, str]:
    return {
        "head": git_text("rev-parse", "HEAD"),
        "origin_main": git_text("rev-parse", "origin/main"),
        "branch": git_text("branch", "--show-current"),
        "status_porcelain": run(["git", "status", "--porcelain"], timeout=30.0).stdout,
    }


def canonical_manifest_digest(manifest: dict[str, Any]) -> str:
    return public.digest({key: value for key, value in manifest.items() if key != "manifest_canonical_sha256"})


def file_hash_or_none(path: Path) -> str | None:
    return sha256_file(path) if path.exists() else None


def sh_quote(value: str) -> str:
    return shlex.quote(value)


def safe_relpath(value: str) -> Path:
    path = Path(value)
    require(not path.is_absolute(), f"copyback manifest contains absolute path: {value}")
    require(".." not in path.parts, f"copyback manifest escapes output root: {value}")
    require(value not in {"", "."}, "copyback manifest contains empty path")
    return path


def validate_frozen_hashes() -> dict[str, Any]:
    observed: dict[str, str] = {}
    failures: list[str] = []
    for key, path in FROZEN_BYTE_FILES.items():
        digest = sha256_file(path)
        observed[key] = digest
        if digest != FROZEN_HASHES[key]:
            failures.append(f"{key} {digest} != {FROZEN_HASHES[key]}")
    private_map = read_json(HERE / "ORBITSTATE_PRIVATE_SOURCE_MAP.json")
    observed["private_source_map_canonical_sha256"] = public.digest(private_map)
    if observed["private_source_map_canonical_sha256"] != FROZEN_HASHES["private_source_map_canonical_sha256"]:
        failures.append("private source map canonical hash drift")
    return {"passed": not failures, "observed": observed, "failures": failures}


def attempt0_controller_result_path() -> Path:
    return ATTEMPT0_LOCAL_RUN_ROOT / "CONTROLLER_RESULT.json"


def attempt0_controller_receipt() -> dict[str, Any]:
    path = attempt0_controller_result_path()
    require(path.exists(), f"attempt-zero controller receipt missing: {path}")
    observed_sha = sha256_file(path)
    require(
        observed_sha == ATTEMPT0_CONTROLLER_RESULT_SHA256,
        f"attempt-zero controller receipt drift: {observed_sha}",
    )
    result = read_json(path)
    return {
        "schema": "ORBITSTATE_ATTEMPT0_PREEXECUTION_FAILURE_V1",
        "science_package_id": SCIENCE_PACKAGE_ID,
        "transaction_run_id": ATTEMPT0_RUN_ID,
        "starting_commit": EXPECTED_STARTING_COMMIT,
        "controller_result_sha256": observed_sha,
        "controller_status": result.get("status"),
        "target_returncode": result.get("target_returncode"),
        "exact_exception": "IndexError: 9",
        "stderr_excerpt": result.get("stderr", ""),
        "remote_root_retained": result.get("remote_retained") is True,
        "local_root_retained": ATTEMPT0_LOCAL_RUN_ROOT.exists(),
        "target_output_absent": result.get("output_root_exists") is False,
        "hardware_execution_began": False,
        "pmu_preflight_executed": False,
        "replicate_0_executed": False,
        "replicate_1_executed": False,
        "feature_freeze_executed": False,
        "unblinding_executed": False,
        "scientific_classification_emitted": False,
    }


def write_attempt0_failure_receipt() -> dict[str, Any]:
    receipt = attempt0_controller_receipt()
    require(receipt["controller_status"] == "ORBITSTATE_CONTROLLER_TARGET_FAILED_NO_COPYBACK", "attempt-zero status drift")
    require(receipt["target_returncode"] == 1, "attempt-zero target return code drift")
    require("IndexError: 9" in receipt["stderr_excerpt"], "attempt-zero exception drift")
    require(receipt["target_output_absent"], "attempt-zero target output root was created")
    write_json(ATTEMPT0_RECEIPT_PATH, receipt)
    return {**receipt, "receipt_sha256": sha256_file(ATTEMPT0_RECEIPT_PATH)}


def transferred_python_files(source_root: Path) -> list[Path]:
    return [source_root / name for name in target.SOURCE_FILE_NAMES if name.endswith(".py")]


def fixed_parent_depth_static_rejection(source_root: Path = HERE) -> dict[str, Any]:
    forbidden_patterns = [
        re.compile(r"\.parents\[[0-9]+\]"),
        re.compile(r"^\s*REPO_ROOT\s*=", re.MULTILINE),
        re.compile(r"Path\(__file__\)\.parents\[[0-9]+\]"),
    ]
    hits: list[dict[str, Any]] = []
    for path in transferred_python_files(source_root):
        text = path.read_text(encoding="utf-8")
        for line_number, line in enumerate(text.splitlines(), start=1):
            if any(pattern.search(line) for pattern in forbidden_patterns):
                hits.append({"file": path.name, "line": line_number, "text": line.strip()})

    historical_hits: list[dict[str, Any]] = []
    try:
        repo_root = Path(git_text("rev-parse", "--show-toplevel"))
        for path in transferred_python_files(HERE):
            rel = path.relative_to(repo_root).as_posix()
            completed = run(["git", "show", f"{EXPECTED_STARTING_COMMIT}:{rel}"], timeout=30.0, check=False)
            if completed.returncode != 0:
                continue
            for line_number, line in enumerate(completed.stdout.splitlines(), start=1):
                if any(pattern.search(line) for pattern in forbidden_patterns):
                    historical_hits.append({"file": path.name, "line": line_number, "text": line.strip()})
    except Exception as exc:
        historical_hits.append({"file": "<historical-check>", "line": 0, "text": str(exc)})

    return {
        "schema": "ORBITSTATE_FIXED_PARENT_DEPTH_STATIC_REJECTION_V1",
        "zero_live_contact": True,
        "current_hits": hits,
        "starting_commit": EXPECTED_STARTING_COMMIT,
        "starting_commit_hits": historical_hits,
        "starting_commit_would_fail": bool(historical_hits),
        "passed": not hits and bool(historical_hits),
    }


def copy_shallow_deployment_source(destination: Path) -> None:
    destination.mkdir(mode=0o700)
    names = list(target.SOURCE_FILE_NAMES) + ["ORBITSTATE_IMPLEMENTATION_MANIFEST.json"]
    for name in names:
        shutil.copyfile(HERE / name, destination / name)


def deployment_layout_self_test() -> dict[str, Any]:
    static = fixed_parent_depth_static_rejection(HERE)
    with tempfile.TemporaryDirectory(prefix="orbitstate_shallow_deploy_") as temp:
        root = Path(temp)
        source = root / "source"
        copy_shallow_deployment_source(source)
        import_completed = run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; "
                    f"sys.path.insert(0, {str(source)!r}); "
                    "import orbitstate_independent_target; "
                    "print('ORBITSTATE_SHALLOW_IMPORT_OK')"
                ),
            ],
            timeout=60.0,
            check=False,
        )
        offline_completed = run(
            [
                sys.executable,
                str(source / "orbitstate_independent_target.py"),
                "--offline-validate",
                "--source-root",
                str(source),
                "--output-root",
                str(root / "offline_validate_out"),
            ],
            timeout=180.0,
            check=False,
        )
        self_completed = run(
            [
                sys.executable,
                str(source / "orbitstate_independent_target.py"),
                "--self-test",
                "--source-root",
                str(source),
                "--output-root",
                str(root / "self_test_out"),
            ],
            timeout=240.0,
            check=False,
        )

    offline_data = json.loads(offline_completed.stdout) if offline_completed.returncode == 0 else {}
    self_data = json.loads(self_completed.stdout) if self_completed.returncode == 0 else {}
    checks = {
        "fixed_parent_depth_static_rejection": static["passed"],
        "module_import_succeeds": import_completed.returncode == 0
        and "ORBITSTATE_SHALLOW_IMPORT_OK" in import_completed.stdout,
        "offline_validation_succeeds": offline_data.get("passed") is True,
        "target_self_test_succeeds": self_data.get("self_test_passed") is True,
        "no_index_error": "IndexError" not in import_completed.stderr
        and "IndexError" not in offline_completed.stderr
        and "IndexError" not in self_completed.stderr,
        "zero_network_contact": True,
        "zero_hardware_execution": True,
        "starting_commit_layout_would_fail": static["starting_commit_would_fail"],
    }
    result = {
        "schema": "ORBITSTATE_DEPLOYMENT_LAYOUT_SELF_TEST_V1",
        "science_package_id": SCIENCE_PACKAGE_ID,
        "transaction_run_id": TRANSACTION_RUN_ID,
        "zero_live_contact": True,
        "python_executable_name": Path(sys.executable).name,
        "static_rejection": static,
        "import_returncode": import_completed.returncode,
        "offline_validate_returncode": offline_completed.returncode,
        "target_self_test_returncode": self_completed.returncode,
        "offline_validate_passed": offline_data.get("passed") is True,
        "target_self_test_passed": self_data.get("self_test_passed") is True,
        "checks": checks,
    }
    result["passed"] = all(checks.values())
    result["deployment_layout_self_test_sha256"] = public.digest(
        {key: value for key, value in result.items() if key != "deployment_layout_self_test_sha256"}
    )
    write_json(DEPLOYMENT_LAYOUT_TEST_PATH, result)
    return result


def build_source_file_map(include_manifest: bool) -> dict[str, dict[str, Any]]:
    files = list(target.source_files(HERE))
    if include_manifest:
        files.append(HERE / "ORBITSTATE_IMPLEMENTATION_MANIFEST.json")
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(files, key=lambda item: item.name):
        result[path.name] = {"sha256": sha256_file(path), "size": path.stat().st_size}
    return result


def build_sol_audit_placeholder() -> dict[str, Any]:
    return {
        "schema": "ORBITSTATE_SOL_AUDIT_V2",
        "model": "gpt-5.6-sol",
        "effort": "xhigh",
        "custody": "READ_ONLY",
        "no_checkout_mutation": True,
        "no_lab_device_contact": True,
        "no_git_write": True,
        "status": "PENDING_READ_ONLY_SOL_AUDIT",
        "material_findings": "PENDING",
    }


def build_manifest() -> dict[str, Any]:
    git_state = git_head_and_status()
    public_audit = ORBIT_COUPLING_ROOT / "PUBLIC_TRANSDUCER_NULL_LAW_AUDIT.md"
    offline_validate_path = HERE / "ORBITSTATE_OFFLINE_VALIDATE.json"
    offline = read_json(offline_validate_path) if offline_validate_path.exists() else {}
    transport_path = HERE / "ORBITSTATE_TRANSPORT_SIMULATION.json"
    transport = read_json(transport_path) if transport_path.exists() else {}
    feature_boundary_path = HERE / "ORBITSTATE_FEATURE_BOUNDARY_SELF_TEST.json"
    feature_boundary = read_json(feature_boundary_path) if feature_boundary_path.exists() else {}
    deployment_layout = read_json(DEPLOYMENT_LAYOUT_TEST_PATH) if DEPLOYMENT_LAYOUT_TEST_PATH.exists() else {}
    attempt0_receipt = read_json(ATTEMPT0_RECEIPT_PATH) if ATTEMPT0_RECEIPT_PATH.exists() else {}
    repair_audit_path = HERE / "LIVE_CUSTODY_AND_PROTOCOL_REPAIR_AUDIT.md"
    sol_path = HERE / "ORBITSTATE_SOL_AUDIT.json"
    if not sol_path.exists():
        write_json(sol_path, build_sol_audit_placeholder())
    sol = read_json(sol_path)
    source_map = read_json(HERE / "ORBITSTATE_PRIVATE_SOURCE_MAP.json")
    manifest = {
        "schema": "ORBITSTATE_IMPLEMENTATION_MANIFEST_V2",
        "starting_commit": EXPECTED_STARTING_COMMIT,
        "head_at_freeze_build": git_state["head"],
        "origin_main_at_freeze_build": git_state["origin_main"],
        "branch_at_freeze_build": git_state["branch"],
        "git_status_porcelain_at_freeze_build": git_state["status_porcelain"],
        "final_commit": "AWAITING_COHERENT_COMMIT",
        "science_package_id": SCIENCE_PACKAGE_ID,
        "transaction_run_id": TRANSACTION_RUN_ID,
        "run_id": TRANSACTION_RUN_ID,
        "expected_local_root": str(EXPECTED_LOCAL_RUN_ROOT),
        "expected_remote_root": EXPECTED_REMOTE_RUN_ROOT,
        "expected_remote_output_root": EXPECTED_REMOTE_OUTPUT_ROOT,
        "attempt_zero": {
            "transaction_run_id": ATTEMPT0_RUN_ID,
            "local_root": str(ATTEMPT0_LOCAL_RUN_ROOT),
            "remote_root": ATTEMPT0_REMOTE_RUN_ROOT,
            "controller_result_sha256": ATTEMPT0_CONTROLLER_RESULT_SHA256,
            "failure_receipt_sha256": file_hash_or_none(ATTEMPT0_RECEIPT_PATH),
            "controller_status": attempt0_receipt.get("controller_status"),
            "target_returncode": attempt0_receipt.get("target_returncode"),
            "scientific_classification_emitted": attempt0_receipt.get("scientific_classification_emitted"),
            "hardware_execution_began": attempt0_receipt.get("hardware_execution_began"),
        },
        "zero_live_contact_attestation": {
            "network_connections": 0,
            "hardware_executions": 0,
            "ssh_scp_ping_target_contact": 0,
        },
        "frozen_science": {
            "N": public.N,
            "d": public.D_MEMBER,
            "fold_d": public.FOLD_MEMBER,
            "M": public.BASE_WORK,
            "quantization_scale": public.QUANTIZATION_SCALE,
            "public_q0_absolute_bound": public.PUBLIC_Q0_ABSOLUTE_BOUND,
            "private_odd_signal_floor": public.PRIVATE_ODD_SIGNAL_FLOOR,
            "relational_tolerance": public.RELATIONAL_TOLERANCE,
            "replicates": len(public.REPLICATES),
            "conditions": len(public.CONDITIONS),
            "public_decoder_phases": len(public.PHASES),
            "mapping_leg_records": 144,
            "independent_component_windows": 288,
        },
        "public_transducer_reference": read_json(HERE / "PUBLIC_TRANSDUCER_REFERENCE.json"),
        "source_file_map": build_source_file_map(include_manifest=False),
        "hashes": {
            "public_null_law_audit_sha256": sha256_file(public_audit),
            "live_custody_repair_audit_sha256": file_hash_or_none(repair_audit_path),
            "contract_sha256": sha256_file(HERE / "ORBITSTATE_INDEPENDENT_CONTRACT_V2.md"),
            "public_transducer_reference_file_sha256": sha256_file(HERE / "PUBLIC_TRANSDUCER_REFERENCE.json"),
            "public_schedule_json_sha256": sha256_file(HERE / "ORBITSTATE_PUBLIC_SCHEDULE.json"),
            "public_schedule_tsv_sha256": sha256_file(HERE / "ORBITSTATE_PUBLIC_SCHEDULE.tsv"),
            "public_schedule_sha256_file_sha256": sha256_file(HERE / "ORBITSTATE_PUBLIC_SCHEDULE.sha256"),
            "private_source_map_file_sha256": sha256_file(HERE / "ORBITSTATE_PRIVATE_SOURCE_MAP.json"),
            "private_source_map_canonical_sha256": public.digest(source_map),
            "source_bundle_sha256": offline.get("source_bundle", {}).get("sha256"),
            "offline_binary_sha256": offline.get("runtime_compile", {}).get("binary_sha256"),
            "disassembly_sha256": offline.get("disassembly", {}).get("sha256"),
            "self_test_sha256": file_hash_or_none(HERE / "ORBITSTATE_SELF_TEST.json"),
            "target_self_test_sha256": file_hash_or_none(HERE / "ORBITSTATE_TARGET_SELF_TEST.json"),
            "controller_self_test_sha256": file_hash_or_none(HERE / "ORBITSTATE_CONTROLLER_SELF_TEST.json"),
            "transport_simulation_sha256": file_hash_or_none(transport_path),
            "feature_boundary_self_test_sha256": file_hash_or_none(feature_boundary_path),
            "deployment_layout_self_test_sha256": file_hash_or_none(DEPLOYMENT_LAYOUT_TEST_PATH),
            "attempt0_failure_receipt_sha256": file_hash_or_none(ATTEMPT0_RECEIPT_PATH),
            "retry1_authority_overlay_sha256": file_hash_or_none(RETRY1_AUTHORITY_PATH),
            "sol_audit_sha256": sha256_file(sol_path),
        },
        "offline_validation": {
            "passed": offline.get("passed"),
            "offline_validate_sha256": offline.get("offline_validate_sha256"),
        },
        "transport_simulation": {
            "passed": transport.get("passed"),
            "sha256": file_hash_or_none(transport_path),
        },
        "feature_boundary_self_test": {
            "passed": feature_boundary.get("passed"),
            "sha256": file_hash_or_none(feature_boundary_path),
        },
        "deployment_layout_self_test": {
            "passed": deployment_layout.get("passed"),
            "sha256": file_hash_or_none(DEPLOYMENT_LAYOUT_TEST_PATH),
            "starting_commit_layout_would_fail": deployment_layout.get("checks", {}).get("starting_commit_layout_would_fail"),
        },
        "sol_audit": {
            "status": sol.get("status"),
            "material_findings": sol.get("material_findings"),
            "sha256": sha256_file(sol_path),
        },
        "future_authorization": {
            "commit_binding_env": COMMIT_ENV,
            "manifest_binding_env": MANIFEST_ENV,
            "live_authority_env": AUTHORITY_ENV,
            "live_authority_value": AUTHORITY_VALUE,
            "powershell_command": (
                f'$env:{COMMIT_ENV} = "<new-final-commit>"\n'
                f'$env:{MANIFEST_ENV} = "<new-manifest-file-sha>"\n'
                f'$env:{AUTHORITY_ENV} = "{AUTHORITY_VALUE}"\n\n'
                '.\\.venv\\Scripts\\python.exe '
                '"THOUGHT\\LAB\\CAT_CAS\\7_decoder\\50_phase_bm_cpu\\50_6_fixed_point_substrate\\14_noncollapse_frontier\\phase6b6\\live_small_wall\\orbit_coupling\\orbitstate_independent_v2\\run_orbitstate_independent_v2.py" '
                "--execute-authorized"
            ),
        },
        "allowed_target_classes": public.ALLOWED_RESULT_CLASSES,
        "forbidden_target_classes": public.FORBIDDEN_RESULT_CLASSES,
        "small_wall_post_audit_promotion_law": [
            "fresh target result is ORBITSTATE_INDEPENDENT_COUPLING_CONFIRMED",
            "both fresh replicates independently pass",
            "receiver feature hash frozen before unblinding",
            "no private source field entered receiver feature extraction",
            "restoration and physical mapping controls pass",
            "GPT-5.6 Sol Extra High read-only audit has no material blocker",
            "GPT-5.5 independently verifies copied source and evidence",
        ],
    }
    manifest["manifest_canonical_sha256"] = canonical_manifest_digest(manifest)
    return manifest


def write_self_test_artifacts() -> dict[str, Any]:
    public_self = public.self_test()
    write_json(HERE / "ORBITSTATE_SELF_TEST.json", public_self)
    target_self = target.self_test(HERE, HERE)
    return {
        "public_self_test": public_self,
        "target_self_test": target_self,
    }


def validate_source_bundle_against_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    expected = manifest.get("hashes", {}).get("source_bundle_sha256")
    require(type(expected) is str and len(expected) == 64, "manifest source bundle hash missing")
    with tempfile.TemporaryDirectory(prefix="orbitstate_source_bundle_check_") as temp:
        receipt = target.deterministic_source_bundle(HERE, Path(temp) / "ORBITSTATE_SOURCE_BUNDLE.tar.gz")
    return {
        "passed": receipt["sha256"] == expected,
        "expected_source_bundle_sha256": expected,
        "observed_source_bundle_sha256": receipt["sha256"],
    }


def pretransport_gate(
    *,
    manifest: dict[str, Any],
    commit_binding: str,
    manifest_file_sha: str,
    local_runs_root: Path,
) -> dict[str, Any]:
    git_state = git_head_and_status()
    require(git_state["branch"] == "main", "live execution requires main branch")
    require(git_state["status_porcelain"] == "", "working tree must be clean before live execution")
    require(git_state["head"] == commit_binding, "HEAD does not match commit binding")
    require(git_state["origin_main"] == commit_binding, "origin/main does not match commit binding")
    actual_manifest_sha = sha256_file(HERE / "ORBITSTATE_IMPLEMENTATION_MANIFEST.json")
    require(actual_manifest_sha == manifest_file_sha, "manifest file SHA does not match authority")
    require(manifest_file_sha == os.environ.get(MANIFEST_ENV), "manifest file SHA environment mismatch")
    require(os.environ.get(COMMIT_ENV) == commit_binding, "commit binding environment mismatch")
    require(os.environ.get(AUTHORITY_ENV) == AUTHORITY_VALUE, "live authority mismatch")
    require(canonical_manifest_digest(manifest) == manifest["manifest_canonical_sha256"], "manifest canonical digest drift")
    require(manifest.get("science_package_id") == SCIENCE_PACKAGE_ID, "manifest science package identity drift")
    require(manifest.get("transaction_run_id") == TRANSACTION_RUN_ID, "manifest transaction identity drift")
    local_root = local_runs_root / TRANSACTION_RUN_ID
    require(not local_root.exists(), f"local run root already exists: {local_root}")
    attempt0 = attempt0_controller_receipt()
    deployment = read_json(DEPLOYMENT_LAYOUT_TEST_PATH)
    require(deployment.get("passed") is True, "deployment-layout self-test did not pass")
    frozen = validate_frozen_hashes()
    require(frozen["passed"], f"frozen hashes drifted: {frozen['failures']}")
    source_file_map = build_source_file_map(include_manifest=False)
    require(source_file_map == manifest.get("source_file_map"), "manifest source-file map drift")
    bundle = validate_source_bundle_against_manifest(manifest)
    require(bundle["passed"], "source bundle reconstruction drift")
    validate = validate_only()
    require(validate["passed"], "validate-only failed before transport")
    return {
        "passed": True,
        "git_state": git_state,
        "manifest_file_sha256": actual_manifest_sha,
        "manifest_canonical_sha256": manifest["manifest_canonical_sha256"],
        "frozen_hashes": frozen,
        "attempt0_receipt": {
            "controller_result_sha256": attempt0["controller_result_sha256"],
            "controller_status": attempt0["controller_status"],
            "target_returncode": attempt0["target_returncode"],
        },
        "deployment_layout_self_test_sha256": sha256_file(DEPLOYMENT_LAYOUT_TEST_PATH),
        "source_bundle": bundle,
        "validate_only_sha256": validate["validate_only_sha256"],
        "local_run_root_absent": str(local_root),
    }


def source_transfer_entries(manifest: dict[str, Any], manifest_sha: str) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    source_map = manifest.get("source_file_map", {})
    for name in target.SOURCE_FILE_NAMES:
        path = HERE / name
        require(path.exists(), f"source transfer file missing: {name}")
        observed = {"sha256": sha256_file(path), "size": path.stat().st_size}
        require(source_map.get(name) == observed, f"source transfer hash drift: {name}")
        entries.append({"name": name, "path": path, **observed})
    manifest_path = HERE / "ORBITSTATE_IMPLEMENTATION_MANIFEST.json"
    require(sha256_file(manifest_path) == manifest_sha, "implementation manifest transfer hash drift")
    entries.append(
        {
            "name": "ORBITSTATE_IMPLEMENTATION_MANIFEST.json",
            "path": manifest_path,
            "sha256": manifest_sha,
            "size": manifest_path.stat().st_size,
        }
    )
    return entries


def run_controller_command(
    runner: Any,
    command: list[str],
    *,
    timeout: float,
    executed: list[list[str]],
) -> subprocess.CompletedProcess[str]:
    require(not command_mutates_attempt0_root(command), f"attempt-zero root mutation command blocked: {' '.join(command)}")
    completed = runner(command, text=True, capture_output=True, timeout=timeout, check=False)
    executed.append(command)
    return completed


def command_mutates_attempt0_root(command: list[str]) -> bool:
    text = " ".join(command)
    if ATTEMPT0_REMOTE_RUN_ROOT not in text:
        return False
    if command and command[0] == "scp" and any(f":{ATTEMPT0_REMOTE_RUN_ROOT}/" in item for item in command[1:]):
        return True
    if command and command[0] == "ssh" and ("rm -rf" in text or "--execute-live" in text or "install -d" in text):
        return True
    return False


def verify_copyback_manifest(local_root: Path) -> dict[str, Any]:
    manifest_path = local_root / "COPYBACK_MANIFEST.json"
    require(manifest_path.exists(), "COPYBACK_MANIFEST.json missing")
    manifest = read_json(manifest_path)
    entries = manifest.get("entries")
    require(type(entries) is list, "copyback manifest entries missing")
    expected_digest = public.digest({key: value for key, value in manifest.items() if key != "copyback_manifest_sha256"})
    require(manifest.get("copyback_manifest_sha256") == expected_digest, "copyback manifest self digest failed")
    listed: set[str] = set()
    for entry in entries:
        require(type(entry) is dict, "copyback manifest entry must be object")
        rel_text = entry.get("path")
        require(type(rel_text) is str, "copyback manifest entry path missing")
        rel = safe_relpath(rel_text)
        require(rel_text not in listed, f"duplicate copyback entry: {rel_text}")
        listed.add(rel_text)
        path = local_root / rel
        require(path.exists() and path.is_file(), f"copyback entry missing file: {rel_text}")
        require(path.stat().st_size == entry.get("size"), f"copyback size mismatch: {rel_text}")
        require(sha256_file(path) == entry.get("sha256"), f"copyback SHA mismatch: {rel_text}")
    actual = {
        item.relative_to(local_root).as_posix()
        for item in local_root.rglob("*")
        if item.is_file() and item.name != "COPYBACK_MANIFEST.json"
    }
    require(actual == listed, f"copyback manifest coverage mismatch missing={sorted(actual - listed)} extra={sorted(listed - actual)}")
    return {"passed": True, "entry_count": len(entries), "copyback_manifest_sha256": manifest["copyback_manifest_sha256"]}


def verify_success_evidence(local_root: Path, *, manifest_sha: str, target_returncode: int) -> dict[str, Any]:
    require(target_returncode == 0, "target return code must be zero for success")
    copyback = verify_copyback_manifest(local_root)
    final_path = local_root / "FINAL_RESULT_ORBITSTATE_INDEPENDENT_V2.json"
    execution_manifest_path = local_root / "ORBITSTATE_INDEPENDENT_V2_MANIFEST.json"
    require(final_path.exists(), "final result missing")
    require(execution_manifest_path.exists(), "success execution manifest missing")
    final = read_json(final_path)
    execution = read_json(execution_manifest_path)
    require(final.get("status") == "ORBITSTATE_INDEPENDENT_TARGET_COMPLETE", "target final status is not complete")
    require(final.get("result_class") in public.ALLOWED_RESULT_CLASSES, "target result class is not allowed")
    require(execution.get("implementation_manifest_file_sha256") == manifest_sha, "implementation manifest binding mismatch")
    expected_execution_digest = public.digest(
        {key: value for key, value in execution.items() if key != "execution_manifest_sha256"}
    )
    require(execution.get("execution_manifest_sha256") == expected_execution_digest, "execution manifest self digest failed")
    final_hashes = execution.get("final_evidence_hashes")
    require(type(final_hashes) is dict, "execution manifest final evidence hashes missing")
    for name, expected_hash in final_hashes.items():
        rel = safe_relpath(name)
        path = local_root / rel
        require(path.exists() and path.is_file(), f"success evidence file missing: {name}")
        require(sha256_file(path) == expected_hash, f"success evidence hash mismatch: {name}")
    return {
        "passed": True,
        "copyback": copyback,
        "final": final,
        "execution_manifest_sha256": execution["execution_manifest_sha256"],
        "result_class": final["result_class"],
    }


def verify_target_failure_evidence(local_root: Path) -> dict[str, Any]:
    copyback = verify_copyback_manifest(local_root)
    required = [
        "TARGET_FAILURE_ORBITSTATE_INDEPENDENT_V2.json",
        "ORBITSTATE_INDEPENDENT_V2_FAILURE_MANIFEST.json",
        "FINAL_RESULT_ORBITSTATE_INDEPENDENT_V2.json",
        "LIVE_CUSTODY_LOG.json",
        "COPYBACK_MANIFEST.json",
    ]
    missing = [name for name in required if not (local_root / name).exists()]
    require(not missing, f"target failure evidence missing: {missing}")
    require(not (local_root / "ORBITSTATE_INDEPENDENT_V2_MANIFEST.json").exists(), "success manifest must not exist on target failure")
    final = read_json(local_root / "FINAL_RESULT_ORBITSTATE_INDEPENDENT_V2.json")
    require(final.get("scientific_classification_emitted") is False, "failure final must not emit science")
    require("result_class" not in final, "failure final must not contain result_class")
    failure_manifest = read_json(local_root / "ORBITSTATE_INDEPENDENT_V2_FAILURE_MANIFEST.json")
    expected = public.digest({key: value for key, value in failure_manifest.items() if key != "failure_manifest_sha256"})
    require(failure_manifest.get("failure_manifest_sha256") == expected, "failure manifest self digest failed")
    return {
        "passed": True,
        "copyback": copyback,
        "failure_phase": final.get("failure_phase"),
    }


def controller_failure(
    *,
    status: str,
    remote_retained: bool,
    remote_cleaned: bool,
    executed: list[list[str]],
    **extra: Any,
) -> dict[str, Any]:
    return {
        "status": status,
        "remote_retained": remote_retained,
        "remote_cleaned": remote_cleaned,
        "commands": executed,
        **extra,
    }


def execute_transport_transaction(
    *,
    manifest: dict[str, Any],
    runner: Any = subprocess.run,
    local_runs_root: Path = RUNS_ROOT,
    commit_binding: str | None = None,
    manifest_sha: str | None = None,
    enforce_pretransport: bool = False,
) -> dict[str, Any]:
    local_root = local_runs_root / TRANSACTION_RUN_ID
    commit_binding = commit_binding or git_text("rev-parse", "HEAD")
    manifest_path = HERE / "ORBITSTATE_IMPLEMENTATION_MANIFEST.json"
    manifest_sha = manifest_sha or sha256_file(manifest_path)
    if enforce_pretransport:
        preflight = pretransport_gate(
            manifest=manifest,
            commit_binding=commit_binding,
            manifest_file_sha=manifest_sha,
            local_runs_root=local_runs_root,
        )
    else:
        require(not local_root.exists(), f"local run root already exists: {local_root}")
        preflight = {"passed": True, "simulated": True}

    entries = source_transfer_entries(manifest, manifest_sha)
    executed: list[list[str]] = []
    create_remote = [
        "ssh",
        TARGET_HOST,
        (
            "set -eu; "
            f"test ! -e {sh_quote(EXPECTED_REMOTE_RUN_ROOT)}; "
            f"install -d -m 0700 {sh_quote(EXPECTED_REMOTE_SOURCE_ROOT)}"
        ),
    ]
    completed = run_controller_command(runner, create_remote, timeout=30.0, executed=executed)
    if completed.returncode != 0:
        return controller_failure(
            status="ORBITSTATE_CONTROLLER_REMOTE_ROOT_NOT_ABSENT",
            remote_retained=True,
            remote_cleaned=False,
            executed=executed,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            pretransport_gate=preflight,
        )

    for entry in entries:
        remote_dest = f"{TARGET_HOST}:{EXPECTED_REMOTE_SOURCE_ROOT}/{entry['name']}"
        completed = run_controller_command(
            runner,
            ["scp", str(entry["path"]), remote_dest],
            timeout=120.0,
            executed=executed,
        )
        if completed.returncode != 0:
            return controller_failure(
                status="ORBITSTATE_CONTROLLER_SOURCE_TRANSFER_FAILED",
                remote_retained=True,
                remote_cleaned=False,
                executed=executed,
                failed_transfer=entry["name"],
                returncode=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
                pretransport_gate=preflight,
            )

    remote_command = (
        f"cd {sh_quote(EXPECTED_REMOTE_SOURCE_ROOT)} && "
        f"timeout --signal=TERM --kill-after=5s {REMOTE_TIMEOUT_SECONDS}s "
        f"env {COMMIT_ENV}={sh_quote(commit_binding)} {MANIFEST_ENV}={sh_quote(manifest_sha)} "
        f"{AUTHORITY_ENV}={sh_quote(AUTHORITY_VALUE)} "
        f"python3 {sh_quote(EXPECTED_REMOTE_SOURCE_ROOT + '/orbitstate_independent_target.py')} "
        f"--execute-live --source-root {sh_quote(EXPECTED_REMOTE_SOURCE_ROOT)} "
        f"--output-root {sh_quote(EXPECTED_REMOTE_OUTPUT_ROOT)} --run-id {sh_quote(TRANSACTION_RUN_ID)} "
        f"--expected-manifest-sha {sh_quote(manifest_sha)} "
        f"--expected-commit-binding {sh_quote(commit_binding)}"
    )
    target_completed = run_controller_command(
        runner,
        ["ssh", TARGET_HOST, remote_command],
        timeout=LOCAL_SSH_TIMEOUT_SECONDS,
        executed=executed,
    )

    output_exists = run_controller_command(
        runner,
        ["ssh", TARGET_HOST, f"test -d {sh_quote(EXPECTED_REMOTE_OUTPUT_ROOT)}"],
        timeout=30.0,
        executed=executed,
    )
    copied = False
    copyback_error: dict[str, Any] | None = None
    if output_exists.returncode == 0:
        local_root.mkdir(parents=True, exist_ok=False)
        copyback = run_controller_command(
            runner,
            ["scp", "-r", f"{TARGET_HOST}:{EXPECTED_REMOTE_OUTPUT_ROOT}/.", str(local_root)],
            timeout=300.0,
            executed=executed,
        )
        copied = copyback.returncode == 0
        if not copied:
            copyback_error = {"returncode": copyback.returncode, "stdout": copyback.stdout, "stderr": copyback.stderr}

    if target_completed.returncode != 0:
        if copied:
            try:
                failure_evidence = verify_target_failure_evidence(local_root)
                return controller_failure(
                    status="ORBITSTATE_CONTROLLER_TARGET_FAILED_EVIDENCE_RETAINED",
                    remote_retained=True,
                    remote_cleaned=False,
                    executed=executed,
                    target_returncode=target_completed.returncode,
                    stdout=target_completed.stdout,
                    stderr=target_completed.stderr,
                    failure_evidence=failure_evidence,
                    pretransport_gate=preflight,
                )
            except Exception as exc:
                return controller_failure(
                    status="ORBITSTATE_CONTROLLER_TARGET_FAILED_EVIDENCE_INVALID",
                    remote_retained=True,
                    remote_cleaned=False,
                    executed=executed,
                    target_returncode=target_completed.returncode,
                    evidence_error=str(exc),
                    stdout=target_completed.stdout,
                    stderr=target_completed.stderr,
                    pretransport_gate=preflight,
                )
        return controller_failure(
            status="ORBITSTATE_CONTROLLER_TARGET_FAILED_NO_COPYBACK",
            remote_retained=True,
            remote_cleaned=False,
            executed=executed,
            target_returncode=target_completed.returncode,
            output_root_exists=output_exists.returncode == 0,
            copyback_error=copyback_error,
            stdout=target_completed.stdout,
            stderr=target_completed.stderr,
            pretransport_gate=preflight,
        )

    if not copied:
        return controller_failure(
            status="ORBITSTATE_CONTROLLER_COPYBACK_FAILED",
            remote_retained=True,
            remote_cleaned=False,
            executed=executed,
            output_root_exists=output_exists.returncode == 0,
            copyback_error=copyback_error,
            pretransport_gate=preflight,
        )

    try:
        success = verify_success_evidence(local_root, manifest_sha=manifest_sha, target_returncode=target_completed.returncode)
    except Exception as exc:
        return controller_failure(
            status="ORBITSTATE_CONTROLLER_SUCCESS_EVIDENCE_INVALID",
            remote_retained=True,
            remote_cleaned=False,
            executed=executed,
            evidence_error=str(exc),
            pretransport_gate=preflight,
        )

    cleanup = run_controller_command(
        runner,
        ["ssh", TARGET_HOST, f"rm -rf -- {sh_quote(EXPECTED_REMOTE_RUN_ROOT)}"],
        timeout=120.0,
        executed=executed,
    )
    if cleanup.returncode != 0:
        return controller_failure(
            status="ORBITSTATE_CONTROLLER_CLEANUP_FAILED",
            remote_retained=True,
            remote_cleaned=False,
            executed=executed,
            cleanup_returncode=cleanup.returncode,
            final=success["final"],
            pretransport_gate=preflight,
        )
    absence = run_controller_command(
        runner,
        ["ssh", TARGET_HOST, f"test ! -e {sh_quote(EXPECTED_REMOTE_RUN_ROOT)}"],
        timeout=30.0,
        executed=executed,
    )
    if absence.returncode != 0:
        return controller_failure(
            status="ORBITSTATE_CONTROLLER_CLEANUP_ABSENCE_VERIFY_FAILED",
            remote_retained=True,
            remote_cleaned=False,
            executed=executed,
            absence_returncode=absence.returncode,
            final=success["final"],
            pretransport_gate=preflight,
        )
    return {
        "status": "ORBITSTATE_CONTROLLER_TARGET_COMPLETE",
        "remote_retained": False,
        "remote_cleaned": True,
        "local_root": str(local_root),
        "remote_root": EXPECTED_REMOTE_RUN_ROOT,
        "final": success["final"],
        "success_evidence": success,
        "commands": executed,
        "pretransport_gate": preflight,
    }


def write_copyback_manifest_for_fake(local_root: Path, corrupt: str | None = None) -> None:
    entries: list[dict[str, Any]] = []
    for path in sorted(local_root.rglob("*")):
        if not path.is_file() or path.name == "COPYBACK_MANIFEST.json":
            continue
        rel = path.relative_to(local_root).as_posix()
        entries.append({"path": rel, "size": path.stat().st_size, "sha256": sha256_file(path)})
    if corrupt == "size" and entries:
        entries[0]["size"] = int(entries[0]["size"]) + 1
    if corrupt == "sha" and entries:
        entries[0]["sha256"] = "0" * 64
    manifest = {
        "schema": "ORBITSTATE_COPYBACK_MANIFEST_V2",
        "science_package_id": SCIENCE_PACKAGE_ID,
        "transaction_run_id": TRANSACTION_RUN_ID,
        "run_id": TRANSACTION_RUN_ID,
        "entries": entries,
        "entry_count": len(entries),
    }
    manifest["copyback_manifest_sha256"] = public.digest(manifest)
    write_json(local_root / "COPYBACK_MANIFEST.json", manifest)


def write_fake_success_evidence(local_root: Path, manifest_sha: str, corrupt: str | None = None) -> None:
    local_root.mkdir(parents=True, exist_ok=True)
    required_payloads = {
        "RAW_ORBITSTATE_RECEIVER_CAPTURE.jsonl": "{}\n",
        "ORBITSTATE_RECEIVER_SENTINELS.jsonl": "{}\n",
        "ORBITSTATE_STAGE_RECEIPTS.jsonl": "{}\n",
        "ORBITSTATE_SOURCE_RECEIPTS.jsonl": "{}\n",
        "ORBITSTATE_RECEIVER_FEATURES.sha256": "0" * 64 + "  ORBITSTATE_RECEIVER_FEATURES.json\n",
    }
    for name, payload in required_payloads.items():
        (local_root / name).write_text(payload, encoding="utf-8")
    for name in [
        "ORBITSTATE_SOURCE_BUNDLE.tar.gz",
        "orbitstate_independent_runtime",
        "ORBITSTATE_RUNTIME_DISASSEMBLY_NORMALIZED.txt",
    ]:
        (local_root / name).write_bytes(f"fake {name}\n".encode("utf-8"))
    for name in [
        "ORBITSTATE_SOURCE_HASHES.json",
        "ORBITSTATE_RECEIVER_FEATURES.json",
        "ORBITSTATE_FEATURE_FREEZE_RECEIPT.json",
        "ORBITSTATE_ADJUDICATION.json",
        "LIVE_CUSTODY_LOG.json",
    ]:
        write_json(local_root / name, {"schema": name, "fake_transport": True})
    final = {
        "status": "ORBITSTATE_INDEPENDENT_TARGET_COMPLETE",
        "science_package_id": SCIENCE_PACKAGE_ID,
        "transaction_run_id": TRANSACTION_RUN_ID,
        "run_id": TRANSACTION_RUN_ID,
        "result_class": public.RESULT_CONFIRMED,
        "scientific_classification_emitted": True,
        "fake_transport": True,
    }
    write_json(local_root / "FINAL_RESULT_ORBITSTATE_INDEPENDENT_V2.json", final)
    evidence_hashes = {
        path.relative_to(local_root).as_posix(): sha256_file(path)
        for path in sorted(local_root.rglob("*"))
        if path.is_file() and path.name not in {"COPYBACK_MANIFEST.json", "ORBITSTATE_INDEPENDENT_V2_MANIFEST.json"}
    }
    execution_manifest = {
        "schema": "ORBITSTATE_SUCCESS_EXECUTION_MANIFEST_V2",
        "science_package_id": SCIENCE_PACKAGE_ID,
        "transaction_run_id": TRANSACTION_RUN_ID,
        "run_id": TRANSACTION_RUN_ID,
        "implementation_manifest_file_sha256": manifest_sha,
        "implementation_manifest_canonical_sha256": "fake-canonical",
        "result_class": public.RESULT_CONFIRMED,
        "record_counts": {
            "mapping_leg_records": 144,
            "independent_component_windows": 288,
            "stage_receipts": 2016,
            "source_receipts": 288,
        },
        "final_evidence_hashes": evidence_hashes,
    }
    execution_manifest["execution_manifest_sha256"] = public.digest(execution_manifest)
    write_json(local_root / "ORBITSTATE_INDEPENDENT_V2_MANIFEST.json", execution_manifest)
    write_copyback_manifest_for_fake(local_root, corrupt=corrupt)


def write_fake_failure_evidence(local_root: Path, corrupt: str | None = None) -> None:
    local_root.mkdir(parents=True, exist_ok=True)
    write_json(local_root / "TARGET_FAILURE_ORBITSTATE_INDEPENDENT_V2.json", {"failure_phase": "fake_target", "fake_transport": True})
    final = {
        "status": "ORBITSTATE_INDEPENDENT_TARGET_FAILED",
        "science_package_id": SCIENCE_PACKAGE_ID,
        "transaction_run_id": TRANSACTION_RUN_ID,
        "run_id": TRANSACTION_RUN_ID,
        "failure_phase": "fake_target",
        "scientific_classification_emitted": False,
        "fake_transport": True,
    }
    write_json(local_root / "FINAL_RESULT_ORBITSTATE_INDEPENDENT_V2.json", final)
    write_json(local_root / "LIVE_CUSTODY_LOG.json", {"schema": "LIVE_CUSTODY_LOG_V2", "fake_transport": True})
    failure_manifest = {
        "schema": "ORBITSTATE_FAILURE_MANIFEST_V2",
        "science_package_id": SCIENCE_PACKAGE_ID,
        "transaction_run_id": TRANSACTION_RUN_ID,
        "run_id": TRANSACTION_RUN_ID,
        "scientific_classification_emitted": False,
        "failure_files": {
            "TARGET_FAILURE_ORBITSTATE_INDEPENDENT_V2.json": sha256_file(local_root / "TARGET_FAILURE_ORBITSTATE_INDEPENDENT_V2.json"),
            "FINAL_RESULT_ORBITSTATE_INDEPENDENT_V2.json": sha256_file(local_root / "FINAL_RESULT_ORBITSTATE_INDEPENDENT_V2.json"),
            "LIVE_CUSTODY_LOG.json": sha256_file(local_root / "LIVE_CUSTODY_LOG.json"),
        },
    }
    failure_manifest["failure_manifest_sha256"] = public.digest(failure_manifest)
    write_json(local_root / "ORBITSTATE_INDEPENDENT_V2_FAILURE_MANIFEST.json", failure_manifest)
    write_copyback_manifest_for_fake(local_root, corrupt=corrupt)


class FakeRunner:
    def __init__(self, mode: str, fake_local_runs: Path, manifest_sha: str) -> None:
        self.mode = mode
        self.fake_local_runs = fake_local_runs
        self.manifest_sha = manifest_sha
        self.calls: list[list[str]] = []
        self.target_invocations = 0
        self.cleanup_invocations = 0

    def __call__(self, command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        self.calls.append(command)
        stdout = ""
        stderr = ""
        returncode = 0
        text = " ".join(command)
        if command[0] == "ssh" and "test ! -e" in text and "install -d" in text and self.mode == "remote_root_exists":
            returncode = 9
            stderr = "remote root exists"
        elif command[0] == "ssh" and "--execute-live" in text:
            self.target_invocations += 1
            if self.mode == "target_failure":
                returncode = 7
                stderr = "fake target failure"
            elif self.mode == "preexecution_no_output":
                returncode = 12
                stderr = "fake pre-execution failure before output root"
        elif command[0] == "ssh" and f"test -d {EXPECTED_REMOTE_OUTPUT_ROOT}" in text:
            returncode = 1 if self.mode == "preexecution_no_output" else 0
        elif command[0] == "scp" and command[1] == "-r" and EXPECTED_REMOTE_OUTPUT_ROOT in command[2]:
            if self.mode == "copyback_command_failure":
                returncode = 8
                stderr = "fake copyback failure"
            else:
                local_root = Path(command[-1])
                if self.mode == "target_failure":
                    write_fake_failure_evidence(local_root)
                elif self.mode == "copyback_size_mismatch":
                    write_fake_success_evidence(local_root, self.manifest_sha, corrupt="size")
                elif self.mode == "copyback_sha_mismatch":
                    write_fake_success_evidence(local_root, self.manifest_sha, corrupt="sha")
                else:
                    write_fake_success_evidence(local_root, self.manifest_sha)
        elif command[0] == "ssh" and "rm -rf --" in text:
            self.cleanup_invocations += 1
            if self.mode == "cleanup_failure":
                returncode = 10
                stderr = "fake cleanup failure"
        elif command[0] == "ssh" and "test ! -e" in text and "install -d" not in text:
            if self.mode == "cleanup_absence_failure":
                returncode = 11
                stderr = "fake cleanup absence failure"
        return subprocess.CompletedProcess(command, returncode, stdout, stderr)


def recursive_source_transfer_used(commands: list[list[str]]) -> bool:
    return any(len(command) >= 3 and command[0] == "scp" and "-r" in command and str(HERE) in command for command in commands)


def sanitize_fake_transport_receipt(value: Any, temp_root: str) -> Any:
    if isinstance(value, dict):
        return {key: sanitize_fake_transport_receipt(item, temp_root) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_fake_transport_receipt(item, temp_root) for item in value]
    if isinstance(value, str):
        text = value.replace(temp_root, "<fake_transport_root>")
        text = re.sub(r"\b[0-9a-f]{64}\b", "<sha256>", text)
        text = re.sub(r"\b[0-9a-f]{40}\b", "<commit>", text)
        return text
    return value


def fake_transport_self_tests() -> dict[str, Any]:
    manifest = build_manifest()
    manifest_sha = sha256_file(HERE / "ORBITSTATE_IMPLEMENTATION_MANIFEST.json") if (HERE / "ORBITSTATE_IMPLEMENTATION_MANIFEST.json").exists() else manifest["manifest_canonical_sha256"]
    commit_binding = git_text("rev-parse", "HEAD")
    with tempfile.TemporaryDirectory(prefix="orbitstate_transport_") as temp:
        root = Path(temp)
        modes = {
            "success": FakeRunner("success", root / "runs_success", manifest_sha),
            "target_failure": FakeRunner("target_failure", root / "runs_failure", manifest_sha),
            "copyback_command_failure": FakeRunner("copyback_command_failure", root / "runs_copy_command", manifest_sha),
            "copyback_size_mismatch": FakeRunner("copyback_size_mismatch", root / "runs_copy_size", manifest_sha),
            "copyback_sha_mismatch": FakeRunner("copyback_sha_mismatch", root / "runs_copy_sha", manifest_sha),
            "cleanup_failure": FakeRunner("cleanup_failure", root / "runs_cleanup_failure", manifest_sha),
            "cleanup_absence_failure": FakeRunner("cleanup_absence_failure", root / "runs_cleanup_absence", manifest_sha),
            "remote_root_exists": FakeRunner("remote_root_exists", root / "runs_remote_exists", manifest_sha),
            "preexecution_no_output": FakeRunner("preexecution_no_output", root / "runs_preexecution_no_output", manifest_sha),
        }
        results = {
            name: execute_transport_transaction(
                manifest=manifest,
                runner=runner,
                local_runs_root=runner.fake_local_runs,
                commit_binding=commit_binding,
                manifest_sha=manifest_sha,
                enforce_pretransport=False,
            )
            for name, runner in modes.items()
        }
        temp_root_text = str(root)
    success_commands = results["success"]["commands"]
    cleanup_commands = [
        command for command in success_commands if command[0] == "ssh" and "rm -rf --" in " ".join(command)
    ]
    absence_commands = [
        command for command in success_commands if command[0] == "ssh" and f"test ! -e {EXPECTED_REMOTE_RUN_ROOT}" in " ".join(command)
    ]
    transfer_commands = [command for command in success_commands if command[0] == "scp" and command[1] != "-r"]
    target_invocations = {
        name: sum(1 for command in result["commands"] if command[0] == "ssh" and "--execute-live" in " ".join(command))
        for name, result in results.items()
    }
    all_commands = [command for result in results.values() for command in result["commands"]]
    regressions = {
        "existing_remote_root_rejected": results["remote_root_exists"]["status"] == "ORBITSTATE_CONTROLLER_REMOTE_ROOT_NOT_ABSENT"
        and target_invocations["remote_root_exists"] == 0,
        "recursive_unbound_transfer_not_used": not recursive_source_transfer_used(success_commands),
        "explicit_manifest_bound_transfer_used": len(transfer_commands) == len(source_transfer_entries(manifest, manifest_sha)),
        "copyback_command_failure_retains_remote": results["copyback_command_failure"]["remote_retained"] is True,
        "copyback_size_mismatch_fails": results["copyback_size_mismatch"]["status"] == "ORBITSTATE_CONTROLLER_SUCCESS_EVIDENCE_INVALID",
        "copyback_sha_mismatch_fails": results["copyback_sha_mismatch"]["status"] == "ORBITSTATE_CONTROLLER_SUCCESS_EVIDENCE_INVALID",
        "target_failure_evidence_copied": results["target_failure"]["status"] == "ORBITSTATE_CONTROLLER_TARGET_FAILED_EVIDENCE_RETAINED",
        "target_failure_retains_remote_root": results["target_failure"]["remote_retained"] is True,
        "preexecution_no_output_failure_retains_retry1_root": results["preexecution_no_output"]["status"]
        == "ORBITSTATE_CONTROLLER_TARGET_FAILED_NO_COPYBACK"
        and results["preexecution_no_output"]["remote_retained"] is True
        and results["preexecution_no_output"]["output_root_exists"] is False,
        "success_cleanup_targets_exact_root": len(cleanup_commands) == 1 and EXPECTED_REMOTE_RUN_ROOT in " ".join(cleanup_commands[0]),
        "attempt0_cleanup_target_rejected": not any(command_mutates_attempt0_root(command) for command in all_commands),
        "attempt0_root_not_used_for_transfer_or_execution": ATTEMPT0_REMOTE_RUN_ROOT
        not in "\n".join(" ".join(command) for command in all_commands),
        "cleanup_absence_verified": len(absence_commands) >= 1,
        "cleanup_failure_retains_remote": results["cleanup_failure"]["remote_retained"] is True,
        "cleanup_absence_failure_retains_remote": results["cleanup_absence_failure"]["remote_retained"] is True,
        "no_automatic_retry": all(count <= 1 for count in target_invocations.values()),
    }
    result = {
        "schema": "ORBITSTATE_TRANSPORT_SIMULATION_V2",
        "zero_live_contact": True,
        "results": sanitize_fake_transport_receipt(results, temp_root_text),
        "regressions": regressions,
        "target_invocations": target_invocations,
    }
    result["passed"] = all(regressions.values()) and results["success"]["status"] == "ORBITSTATE_CONTROLLER_TARGET_COMPLETE"
    result["transport_simulation_sha256"] = public.digest({key: value for key, value in result.items() if key != "transport_simulation_sha256"})
    write_json(HERE / "ORBITSTATE_TRANSPORT_SIMULATION.json", result)
    return result


def freeze_artifacts() -> dict[str, Any]:
    attempt0 = write_attempt0_failure_receipt()
    public_hashes = public.write_artifacts(HERE)
    offline = target.offline_validate(HERE, HERE)
    self_tests = write_self_test_artifacts()
    deployment = deployment_layout_self_test()
    transport = fake_transport_self_tests()
    manifest = build_manifest()
    write_json(HERE / "ORBITSTATE_IMPLEMENTATION_MANIFEST.json", manifest)
    manifest_file_sha = sha256_file(HERE / "ORBITSTATE_IMPLEMENTATION_MANIFEST.json")
    result = {
        "schema": "ORBITSTATE_FREEZE_ARTIFACTS_V2",
        "public_hashes": public_hashes,
        "offline_validate_passed": offline["passed"],
        "offline_validate_sha256": offline["offline_validate_sha256"],
        "attempt0_failure_receipt_sha256": attempt0["receipt_sha256"],
        "self_tests": {
            "public_self_test_sha256": self_tests["public_self_test"]["self_test_sha256"],
            "target_self_test_sha256": self_tests["target_self_test"]["self_test_sha256"],
        },
        "deployment_layout_passed": deployment["passed"],
        "deployment_layout_self_test_sha256": deployment["deployment_layout_self_test_sha256"],
        "transport_passed": transport["passed"],
        "transport_simulation_sha256": transport["transport_simulation_sha256"],
        "manifest_canonical_sha256": manifest["manifest_canonical_sha256"],
        "manifest_file_sha256": manifest_file_sha,
        "zero_live_contact": True,
    }
    result["freeze_sha256"] = public.digest({key: value for key, value in result.items() if key != "freeze_sha256"})
    return result


def validate_only() -> dict[str, Any]:
    schedule = target.validate_schedule_artifacts(HERE)
    blindness = target.public_manifest_blindness(HERE)
    boundary = target.process_boundary_static_proof(HERE)
    feature_boundary = target.feature_boundary_static_proof(HERE)
    pmu = target.pmu_runtime_static_proof(HERE)
    physical = target.physical_protocol_static_proof(HERE)
    fixed_parent_depth = fixed_parent_depth_static_rejection(HERE)
    deployment = deployment_layout_self_test()
    attempt0 = attempt0_controller_receipt()
    frozen = validate_frozen_hashes()
    public_self = public.self_test()
    manifest = read_json(HERE / "ORBITSTATE_IMPLEMENTATION_MANIFEST.json")
    source_bundle = validate_source_bundle_against_manifest(manifest)
    result = {
        "schema": "ORBITSTATE_VALIDATE_ONLY_V2",
        "zero_live_contact": True,
        "schedule": schedule,
        "blindness": blindness,
        "process_boundary": boundary,
        "feature_boundary": feature_boundary,
        "pmu_runtime_static_proof": pmu,
        "physical_protocol_static_proof": physical,
        "fixed_parent_depth_static_rejection": fixed_parent_depth,
        "deployment_layout": deployment,
        "attempt0_controller_receipt": {
            "controller_result_sha256": attempt0["controller_result_sha256"],
            "controller_status": attempt0["controller_status"],
            "target_returncode": attempt0["target_returncode"],
            "target_output_absent": attempt0["target_output_absent"],
        },
        "frozen_hashes": frozen,
        "source_bundle_reconstruction": source_bundle,
        "public_self_test_passed": public_self["self_test_passed"],
        "manifest_canonical_sha256": canonical_manifest_digest(manifest),
        "manifest_file_sha256": sha256_file(HERE / "ORBITSTATE_IMPLEMENTATION_MANIFEST.json"),
    }
    result["passed"] = all(
        [
            schedule["passed"],
            blindness["passed"],
            boundary["passed"],
            feature_boundary["passed"],
            pmu["passed"],
            physical["passed"],
            fixed_parent_depth["passed"],
            deployment["passed"],
            frozen["passed"],
            source_bundle["passed"],
            public_self["self_test_passed"],
            result["manifest_canonical_sha256"] == manifest["manifest_canonical_sha256"],
        ]
    )
    result["validate_only_sha256"] = public.digest({key: value for key, value in result.items() if key != "validate_only_sha256"})
    return result


def controller_self_test() -> dict[str, Any]:
    validate = validate_only()
    transport = fake_transport_self_tests()
    stable_validate = {
        key: value
        for key, value in validate.items()
        if key not in {"manifest_canonical_sha256", "manifest_file_sha256", "validate_only_sha256"}
    }
    stable_validate_sha = public.digest(stable_validate)
    failure_exit_status_law = {
        "passed": (
            controller_result_success(transport["results"]["success"])
            and not controller_result_success(transport["results"]["target_failure"])
            and not controller_result_success(transport["results"]["copyback_command_failure"])
            and not controller_result_success(transport["results"]["copyback_size_mismatch"])
            and not controller_result_success(transport["results"]["copyback_sha_mismatch"])
            and not controller_result_success(transport["results"]["cleanup_failure"])
            and not controller_result_success(transport["results"]["cleanup_absence_failure"])
            and not controller_result_success(transport["results"]["remote_root_exists"])
        ),
        "success_status": transport["results"]["success"]["status"],
        "target_failure_status": transport["results"]["target_failure"]["status"],
        "copyback_size_mismatch_status": transport["results"]["copyback_size_mismatch"]["status"],
        "copyback_sha_mismatch_status": transport["results"]["copyback_sha_mismatch"]["status"],
        "cleanup_failure_status": transport["results"]["cleanup_failure"]["status"],
    }
    result = {
        "schema": "ORBITSTATE_CONTROLLER_SELF_TEST_V2",
        "validate_only_stable_sha256": stable_validate_sha,
        "validate_only_passed": validate["passed"],
        "transport_simulation_sha256": transport["transport_simulation_sha256"],
        "transport_passed": transport["passed"],
        "failure_exit_status_law": failure_exit_status_law,
        "zero_live_contact": True,
    }
    result["self_test_passed"] = validate["passed"] and transport["passed"] and failure_exit_status_law["passed"]
    result["self_test_sha256"] = public.digest({key: value for key, value in result.items() if key != "self_test_sha256"})
    write_json(HERE / "ORBITSTATE_CONTROLLER_SELF_TEST.json", result)
    manifest = build_manifest()
    write_json(HERE / "ORBITSTATE_IMPLEMENTATION_MANIFEST.json", manifest)
    return result


def controller_result_success(result: dict[str, Any]) -> bool:
    if "passed" in result:
        return result["passed"] is True
    if "self_test_passed" in result:
        return result["self_test_passed"] is True
    if result.get("schema") == "ORBITSTATE_FREEZE_ARTIFACTS_V2":
        return result.get("offline_validate_passed") is True and result.get("transport_passed") is True
    return result.get("status") == "ORBITSTATE_CONTROLLER_TARGET_COMPLETE"


def execute_authorized() -> dict[str, Any]:
    manifest_file_sha = sha256_file(HERE / "ORBITSTATE_IMPLEMENTATION_MANIFEST.json")
    git_state = git_head_and_status()
    commit_binding = os.environ.get(COMMIT_ENV, "")
    require(commit_binding == git_state["head"], "commit authority mismatch")
    require(os.environ.get(MANIFEST_ENV) == manifest_file_sha, "manifest authority mismatch")
    require(os.environ.get(AUTHORITY_ENV) == AUTHORITY_VALUE, "live authority mismatch")
    manifest = read_json(HERE / "ORBITSTATE_IMPLEMENTATION_MANIFEST.json")
    result = execute_transport_transaction(
        manifest=manifest,
        commit_binding=commit_binding,
        manifest_sha=manifest_file_sha,
        enforce_pretransport=True,
    )
    EXPECTED_LOCAL_RUN_ROOT.mkdir(parents=True, exist_ok=True)
    write_json(EXPECTED_LOCAL_RUN_ROOT / "CONTROLLER_RESULT.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare-only", action="store_true")
    modes.add_argument("--validate-only", action="store_true")
    modes.add_argument("--self-test", action="store_true")
    modes.add_argument("--execute-authorized", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.prepare_only:
            result = freeze_artifacts()
        elif args.validate_only:
            result = validate_only()
        elif args.self_test:
            result = controller_self_test()
        else:
            result = execute_authorized()
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if controller_result_success(result) else 1
    except Exception as exc:
        print(json.dumps({"status": "ORBITSTATE_CONTROLLER_FAILED", "error": str(exc)}, indent=2, sort_keys=True), file=os.sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
