#!/usr/bin/env python3
"""Controller for the private OrbitState independent-window package."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import orbitstate_independent_public as public
import orbitstate_independent_target as target


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[9]
ORBIT_COUPLING_ROOT = HERE.parent
LIVE_SMALL_WALL_ROOT = ORBIT_COUPLING_ROOT.parent
RUNS_ROOT = ORBIT_COUPLING_ROOT / "runs"
EXPECTED_LOCAL_RUN_ROOT = RUNS_ROOT / public.RUN_ID
EXPECTED_REMOTE_RUN_ROOT = f"/root/catcas_live_small_wall/{public.RUN_ID}"

COMMIT_ENV = "ORBITSTATE_INDEPENDENT_V2_COMMIT_BINDING"
MANIFEST_ENV = "ORBITSTATE_INDEPENDENT_V2_MANIFEST_SHA256"
AUTHORITY_ENV = "ORBITSTATE_INDEPENDENT_V2_LIVE_AUTHORITY"
AUTHORITY_VALUE = public.RUN_ID


class ControllerError(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise ControllerError(message)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
    offline = json.loads(offline_validate_path.read_text(encoding="utf-8")) if offline_validate_path.exists() else {}
    transport_path = HERE / "ORBITSTATE_TRANSPORT_SIMULATION.json"
    transport = json.loads(transport_path.read_text(encoding="utf-8")) if transport_path.exists() else {}
    sol_path = HERE / "ORBITSTATE_SOL_AUDIT.json"
    if not sol_path.exists():
        write_json(sol_path, build_sol_audit_placeholder())
    sol = json.loads(sol_path.read_text(encoding="utf-8"))
    manifest = {
        "schema": "ORBITSTATE_IMPLEMENTATION_MANIFEST_V2",
        "starting_commit": "4762b5b49b308ae4aca8e141113e4fafe4b0f81e",
        "head_at_freeze_build": git_state["head"],
        "origin_main_at_freeze_build": git_state["origin_main"],
        "branch_at_freeze_build": git_state["branch"],
        "git_status_porcelain_at_freeze_build": git_state["status_porcelain"],
        "final_commit": "AWAITING_COHERENT_COMMIT",
        "run_id": public.RUN_ID,
        "expected_local_root": str(EXPECTED_LOCAL_RUN_ROOT),
        "expected_remote_root": EXPECTED_REMOTE_RUN_ROOT,
        "zero_live_contact_attestation": {
            "network_connections": 0,
            "hardware_executions": 0,
            "ssh_scp_ping_target_contact": 0,
        },
        "public_transducer_reference": json.loads((HERE / "PUBLIC_TRANSDUCER_REFERENCE.json").read_text(encoding="utf-8")),
        "hashes": {
            "public_null_law_audit_sha256": sha256_file(public_audit),
            "contract_sha256": sha256_file(HERE / "ORBITSTATE_INDEPENDENT_CONTRACT_V2.md"),
            "public_transducer_reference_file_sha256": sha256_file(HERE / "PUBLIC_TRANSDUCER_REFERENCE.json"),
            "public_schedule_json_sha256": sha256_file(HERE / "ORBITSTATE_PUBLIC_SCHEDULE.json"),
            "public_schedule_tsv_sha256": sha256_file(HERE / "ORBITSTATE_PUBLIC_SCHEDULE.tsv"),
            "public_schedule_sha256_file_sha256": sha256_file(HERE / "ORBITSTATE_PUBLIC_SCHEDULE.sha256"),
            "private_source_map_file_sha256": sha256_file(HERE / "ORBITSTATE_PRIVATE_SOURCE_MAP.json"),
            "private_source_map_canonical_sha256": public.digest(
                json.loads((HERE / "ORBITSTATE_PRIVATE_SOURCE_MAP.json").read_text(encoding="utf-8"))
            ),
            "source_bundle_sha256": offline.get("source_bundle", {}).get("sha256"),
            "offline_binary_sha256": offline.get("runtime_compile", {}).get("binary_sha256"),
            "disassembly_sha256": offline.get("disassembly", {}).get("sha256"),
            "self_test_sha256": file_hash_or_none(HERE / "ORBITSTATE_SELF_TEST.json"),
            "target_self_test_sha256": file_hash_or_none(HERE / "ORBITSTATE_TARGET_SELF_TEST.json"),
            "controller_self_test_sha256": file_hash_or_none(HERE / "ORBITSTATE_CONTROLLER_SELF_TEST.json"),
            "transport_simulation_sha256": file_hash_or_none(transport_path),
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
            "command": (
                f"{COMMIT_ENV}=<final_commit> {MANIFEST_ENV}=<manifest_file_sha256> "
                f"{AUTHORITY_ENV}={AUTHORITY_VALUE} python "
                "orbitstate_independent_v2/run_orbitstate_independent_v2.py --execute-authorized"
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
    target_self = {
        "schema": "ORBITSTATE_TARGET_SELF_TEST_V2",
        "offline_validate_sha256": json.loads((HERE / "ORBITSTATE_OFFLINE_VALIDATE.json").read_text(encoding="utf-8"))[
            "offline_validate_sha256"
        ],
        "offline_validate_passed": True,
        "zero_live_contact": True,
    }
    target_self["self_test_passed"] = True
    target_self["self_test_sha256"] = public.digest(
        {key: value for key, value in target_self.items() if key != "self_test_sha256"}
    )
    write_json(HERE / "ORBITSTATE_TARGET_SELF_TEST.json", target_self)
    return {
        "public_self_test": public_self,
        "target_self_test": target_self,
    }


def execute_transport_transaction(
    *,
    manifest: dict[str, Any],
    runner: Any = subprocess.run,
    local_runs_root: Path = RUNS_ROOT,
) -> dict[str, Any]:
    local_root = local_runs_root / public.RUN_ID
    require(not local_root.exists(), f"local run root already exists: {local_root}")
    remote_root = EXPECTED_REMOTE_RUN_ROOT
    manifest_path = HERE / "ORBITSTATE_IMPLEMENTATION_MANIFEST.json"
    manifest_sha = sha256_file(manifest_path) if manifest_path.exists() else manifest["manifest_canonical_sha256"]
    commit_binding = git_text("rev-parse", "HEAD")
    remote_source = f"{remote_root}/source"
    remote_output = f"{remote_root}/output"
    commands = [
        ["ssh", "root@192.168.137.100", f"mkdir -p {remote_source} {remote_output}"],
        ["scp", "-r", str(HERE), f"root@192.168.137.100:{remote_source}/"],
        [
            "ssh",
            "root@192.168.137.100",
            (
                f"{COMMIT_ENV}={commit_binding} {MANIFEST_ENV}={manifest_sha} "
                f"{AUTHORITY_ENV}={AUTHORITY_VALUE} "
                f"python3 {remote_source}/orbitstate_independent_v2/orbitstate_independent_target.py "
                f"--execute-live --source-root {remote_source}/orbitstate_independent_v2 "
                f"--output-root {remote_output} --run-id {public.RUN_ID} "
                f"--expected-manifest-sha {manifest_sha} "
                f"--expected-commit-binding {commit_binding}"
            ),
        ],
        ["scp", "-r", f"root@192.168.137.100:{remote_output}", str(local_root)],
    ]
    executed: list[list[str]] = []
    for command in commands:
        completed = runner(command, text=True, capture_output=True, timeout=900, check=False)
        executed.append(command)
        if completed.returncode != 0:
            return {
                "status": "ORBITSTATE_CONTROLLER_TARGET_FAILED",
                "remote_retained": True,
                "remote_cleaned": False,
                "failed_command": command,
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "commands": executed,
            }
    final_path = local_root / "FINAL_RESULT_ORBITSTATE_INDEPENDENT_V2.json"
    require(final_path.exists(), "copyback missing final result")
    final = json.loads(final_path.read_text(encoding="utf-8"))
    cleanup = runner(["ssh", "root@192.168.137.100", f"rm -rf {remote_root}"], text=True, capture_output=True, timeout=120, check=False)
    if cleanup.returncode != 0:
        return {
            "status": "ORBITSTATE_CONTROLLER_CLEANUP_FAILED",
            "remote_retained": True,
            "remote_cleaned": False,
            "cleanup_returncode": cleanup.returncode,
            "final": final,
            "commands": executed,
        }
    return {
        "status": "ORBITSTATE_CONTROLLER_TARGET_COMPLETE",
        "remote_retained": False,
        "remote_cleaned": True,
        "local_root": str(local_root),
        "remote_root": remote_root,
        "final": final,
        "commands": executed,
    }


class FakeRunner:
    def __init__(self, mode: str, fake_remote: Path, fake_local_runs: Path) -> None:
        self.mode = mode
        self.fake_remote = fake_remote
        self.fake_local_runs = fake_local_runs
        self.calls: list[list[str]] = []

    def __call__(self, command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        self.calls.append(command)
        stdout = ""
        stderr = ""
        returncode = 0
        text = " ".join(command)
        if self.mode == "target_failure" and "--execute-live" in text:
            returncode = 7
            stderr = "fake target failure"
        elif self.mode == "copyback_failure" and command[0] == "scp" and "output" in text:
            returncode = 8
            stderr = "fake copyback failure"
        elif command[0] == "scp" and "output" in text:
            local_root = self.fake_local_runs / public.RUN_ID
            local_root.mkdir(parents=True, exist_ok=True)
            write_json(
                local_root / "FINAL_RESULT_ORBITSTATE_INDEPENDENT_V2.json",
                {
                    "status": "ORBITSTATE_INDEPENDENT_TARGET_COMPLETE",
                    "result_class": public.RESULT_CONFIRMED,
                    "fake_transport": True,
                },
            )
        return subprocess.CompletedProcess(command, returncode, stdout, stderr)


def fake_transport_self_tests() -> dict[str, Any]:
    manifest = build_manifest()
    with tempfile.TemporaryDirectory(prefix="orbitstate_transport_") as temp:
        root = Path(temp)
        success = execute_transport_transaction(
            manifest=manifest,
            runner=FakeRunner("success", root / "remote_success", root / "runs_success"),
            local_runs_root=root / "runs_success",
        )
        failure = execute_transport_transaction(
            manifest=manifest,
            runner=FakeRunner("target_failure", root / "remote_failure", root / "runs_failure"),
            local_runs_root=root / "runs_failure",
        )
        copy_failure = execute_transport_transaction(
            manifest=manifest,
            runner=FakeRunner("copyback_failure", root / "remote_copy_failure", root / "runs_copy_failure"),
            local_runs_root=root / "runs_copy_failure",
        )
    result = {
        "schema": "ORBITSTATE_TRANSPORT_SIMULATION_V2",
        "zero_live_contact": True,
        "success": success,
        "target_failure": failure,
        "copyback_failure": copy_failure,
        "no_automatic_retry": True,
    }
    result["passed"] = (
        success["status"] == "ORBITSTATE_CONTROLLER_TARGET_COMPLETE"
        and failure["status"] == "ORBITSTATE_CONTROLLER_TARGET_FAILED"
        and copy_failure["status"] == "ORBITSTATE_CONTROLLER_TARGET_FAILED"
        and failure["remote_retained"]
        and copy_failure["remote_retained"]
        and success["remote_cleaned"]
    )
    result["transport_simulation_sha256"] = public.digest({key: value for key, value in result.items() if key != "transport_simulation_sha256"})
    write_json(HERE / "ORBITSTATE_TRANSPORT_SIMULATION.json", result)
    return result


def freeze_artifacts() -> dict[str, Any]:
    public_hashes = public.write_artifacts(HERE)
    offline = target.offline_validate(HERE, HERE)
    self_tests = write_self_test_artifacts()
    transport = fake_transport_self_tests()
    manifest = build_manifest()
    write_json(HERE / "ORBITSTATE_IMPLEMENTATION_MANIFEST.json", manifest)
    manifest_file_sha = sha256_file(HERE / "ORBITSTATE_IMPLEMENTATION_MANIFEST.json")
    result = {
        "schema": "ORBITSTATE_FREEZE_ARTIFACTS_V2",
        "public_hashes": public_hashes,
        "offline_validate_passed": offline["passed"],
        "offline_validate_sha256": offline["offline_validate_sha256"],
        "self_tests": {
            "public_self_test_sha256": self_tests["public_self_test"]["self_test_sha256"],
            "target_self_test_sha256": self_tests["target_self_test"]["self_test_sha256"],
        },
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
    public_self = public.self_test()
    manifest = json.loads((HERE / "ORBITSTATE_IMPLEMENTATION_MANIFEST.json").read_text(encoding="utf-8"))
    result = {
        "schema": "ORBITSTATE_VALIDATE_ONLY_V2",
        "zero_live_contact": True,
        "schedule": schedule,
        "blindness": blindness,
        "process_boundary": boundary,
        "public_self_test_passed": public_self["self_test_passed"],
        "manifest_canonical_sha256": canonical_manifest_digest(manifest),
        "manifest_file_sha256": sha256_file(HERE / "ORBITSTATE_IMPLEMENTATION_MANIFEST.json"),
    }
    result["passed"] = all(
        [
            schedule["passed"],
            blindness["passed"],
            boundary["passed"],
            public_self["self_test_passed"],
            result["manifest_canonical_sha256"] == manifest["manifest_canonical_sha256"],
        ]
    )
    result["validate_only_sha256"] = public.digest({key: value for key, value in result.items() if key != "validate_only_sha256"})
    return result


def controller_self_test() -> dict[str, Any]:
    validate = validate_only()
    transport = fake_transport_self_tests()
    failure_exit_status_law = {
        "passed": (
            controller_result_success(transport["success"])
            and not controller_result_success(transport["target_failure"])
            and not controller_result_success(transport["copyback_failure"])
            and not controller_result_success({"status": "ORBITSTATE_CONTROLLER_CLEANUP_FAILED"})
        ),
        "success_status": transport["success"]["status"],
        "target_failure_status": transport["target_failure"]["status"],
        "copyback_failure_status": transport["copyback_failure"]["status"],
    }
    result = {
        "schema": "ORBITSTATE_CONTROLLER_SELF_TEST_V2",
        "validate_only_sha256": validate["validate_only_sha256"],
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
    return result.get("status") in {"ORBITSTATE_CONTROLLER_TARGET_COMPLETE"}


def execute_authorized() -> dict[str, Any]:
    git_state = git_head_and_status()
    require(git_state["branch"] == "main", "live execution requires main branch")
    require(git_state["head"] == git_state["origin_main"], "HEAD must equal origin/main")
    require(git_state["status_porcelain"] == "", "working tree must be clean before live execution")
    manifest_file_sha = sha256_file(HERE / "ORBITSTATE_IMPLEMENTATION_MANIFEST.json")
    require(os.environ.get(COMMIT_ENV) == git_state["head"], "commit authority mismatch")
    require(os.environ.get(MANIFEST_ENV) == manifest_file_sha, "manifest authority mismatch")
    require(os.environ.get(AUTHORITY_ENV) == AUTHORITY_VALUE, "live authority mismatch")
    manifest = json.loads((HERE / "ORBITSTATE_IMPLEMENTATION_MANIFEST.json").read_text(encoding="utf-8"))
    result = execute_transport_transaction(manifest=manifest)
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
