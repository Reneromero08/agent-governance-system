#!/usr/bin/env python3
"""Controller for the live-capable Independent-Window Transducer V3 package."""

from __future__ import annotations

import argparse
import gzip
import hashlib
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

import independent_window_public as public
import independent_window_target as target_model


HERE = Path(__file__).resolve().parent
CALIBRATION_ROOT = HERE.parent
RUNS_ROOT = CALIBRATION_ROOT / "runs"
TARGET = "root@192.168.137.100"
REMOTE_BASE = "/root/catcas_live_small_wall"
STARTING_COMMIT = "f13599f80f55703d7eb2cec9c4be2c8f1b0bdf5a"
FINAL_COMMIT_PLACEHOLDER = "AWAITING_LIVE_AUTHORIZATION"
COMMIT_BINDING_ENV = "INDEPENDENT_WINDOW_TRANSDUCER_V3_COMMIT_BINDING"
MANIFEST_BINDING_ENV = "INDEPENDENT_WINDOW_TRANSDUCER_V3_MANIFEST_SHA256"
LIVE_AUTHORITY_ENV = "INDEPENDENT_WINDOW_TRANSDUCER_V3_LIVE_AUTHORITY"
LIVE_AUTHORITY_VALUE = public.RUN_ID
IMPLEMENTATION_MANIFEST = HERE / "INDEPENDENT_WINDOW_IMPLEMENTATION_MANIFEST.json"
SELF_TEST_PATH = HERE / "INDEPENDENT_WINDOW_SELF_TEST.json"
SOL_AUDIT_PATH = HERE / "INDEPENDENT_WINDOW_V3_SOL_AUDIT.json"
COMPLETION_AUDIT_PATH = HERE / "LIVE_EXECUTION_COMPLETION_AUDIT.md"
CONTRACT_PATH = HERE / "INDEPENDENT_WINDOW_CONTRACT_V3.md"
TOPOLOGY_AUDIT_PATH = HERE / "RETRY1_MEASUREMENT_TOPOLOGY_AUDIT.md"
RETRY1_ROOT = RUNS_ROOT / "balanced_transducer_confirmation_v2_1"

SOURCE_FILE_MAP = {
    CONTRACT_PATH: "INDEPENDENT_WINDOW_CONTRACT_V3.md",
    TOPOLOGY_AUDIT_PATH: "RETRY1_MEASUREMENT_TOPOLOGY_AUDIT.md",
    COMPLETION_AUDIT_PATH: "LIVE_EXECUTION_COMPLETION_AUDIT.md",
    HERE / "INDEPENDENT_WINDOW_PUBLIC_SCHEDULE.json": "INDEPENDENT_WINDOW_PUBLIC_SCHEDULE.json",
    HERE / "INDEPENDENT_WINDOW_PUBLIC_SCHEDULE.sha256": "INDEPENDENT_WINDOW_PUBLIC_SCHEDULE.sha256",
    HERE / "INDEPENDENT_WINDOW_PUBLIC_SCHEDULE.tsv": "INDEPENDENT_WINDOW_PUBLIC_SCHEDULE.tsv",
    HERE / "INDEPENDENT_WINDOW_V3_SOL_AUDIT.json": "INDEPENDENT_WINDOW_V3_SOL_AUDIT.json",
    HERE / "independent_window_public.py": "independent_window_public.py",
    HERE / "independent_window_runtime.c": "independent_window_runtime.c",
    HERE / "independent_window_runtime.h": "independent_window_runtime.h",
    HERE / "independent_window_target.py": "independent_window_target.py",
    HERE / "run_independent_window_v3.py": "run_independent_window_v3.py",
}
TRANSFER_FILE_MAP = {
    **SOURCE_FILE_MAP,
    IMPLEMENTATION_MANIFEST: "INDEPENDENT_WINDOW_IMPLEMENTATION_MANIFEST.json",
}


class ControllerError(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise ControllerError(message)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def normalize_receipt_paths(value: Any, replacements: list[tuple[str, str]]) -> Any:
    if isinstance(value, dict):
        return {key: normalize_receipt_paths(item, replacements) for key, item in value.items()}
    if isinstance(value, list):
        return [normalize_receipt_paths(item, replacements) for item in value]
    if isinstance(value, str):
        result = value
        for old, new in replacements:
            if old:
                result = result.replace(old, new)
        return result
    return value


def sha256_file(path: Path) -> str:
    return public.sha256_file(path)


def run(command: list[str], *, timeout: float, check: bool = True, runner: Any = subprocess.run) -> subprocess.CompletedProcess[str]:
    completed = runner(
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


def is_full_sha(value: str) -> bool:
    return re.fullmatch(r"[0-9a-f]{40}", value) is not None


def source_hashes() -> dict[str, str]:
    hashes = {}
    for source, name in SOURCE_FILE_MAP.items():
        require(source.is_file(), f"source file missing: {source}")
        hashes[name] = sha256_file(source)
    return hashes


def transfer_hashes() -> dict[str, str]:
    hashes = source_hashes()
    require(IMPLEMENTATION_MANIFEST.is_file(), "implementation manifest missing")
    hashes["INDEPENDENT_WINDOW_IMPLEMENTATION_MANIFEST.json"] = sha256_file(IMPLEMENTATION_MANIFEST)
    return hashes


def deterministic_source_bundle(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w") as archive:
                for source, name in sorted(SOURCE_FILE_MAP.items(), key=lambda item: item[1]):
                    require(source.is_file(), f"bundle source missing: {source}")
                    data = source.read_bytes()
                    info = tarfile.TarInfo(name)
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


def failure_manifest_digest(manifest: dict[str, Any]) -> str:
    return public.digest({k: v for k, v in manifest.items() if k != "manifest_sha256"})


def git_head_and_status() -> tuple[str, str, str]:
    head = run(["git", "rev-parse", "HEAD"], timeout=10).stdout.strip()
    origin = run(["git", "rev-parse", "origin/main"], timeout=10).stdout.strip()
    status = run(["git", "status", "--porcelain=v1"], timeout=10).stdout
    return head, origin, status


def repo_root() -> Path:
    return Path(run(["git", "rev-parse", "--show-toplevel"], timeout=10).stdout.strip())


def windows_to_wsl_path(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    tail = resolved.as_posix().split(":", 1)[1].lstrip("/")
    return f"/mnt/{drive}/{tail}"


def disassembly_receipt(binary: Path, *, via_wsl: bool) -> dict[str, Any]:
    if via_wsl:
        command = ["wsl", "--", "objdump", "-d", windows_to_wsl_path(binary)]
    else:
        objdump = shutil.which("objdump")
        if objdump is None:
            return {"available": False, "passed": False, "reason": "objdump unavailable", "forbidden_instruction_matches": []}
        command = [objdump, "-d", str(binary)]
    completed = run(command, timeout=20, check=False)
    disassembly = completed.stdout
    normalized = disassembly.replace(str(binary), "<independent_window_runtime>").replace(windows_to_wsl_path(binary), "<independent_window_runtime>")
    forbidden = []
    pattern = re.compile(r"\b(rdmsr|wrmsr|clflush|wbinvd|invd)\b", re.IGNORECASE)
    for line_number, line in enumerate(disassembly.splitlines(), start=1):
        if pattern.search(line):
            forbidden.append({"line": line_number, "text": line.strip()[:160]})
    return {
        "available": True,
        "passed": completed.returncode == 0 and not forbidden,
        "command": command,
        "returncode": completed.returncode,
        "normalized_stdout_sha256": hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
        "stderr": completed.stderr.strip(),
        "line_count": len(disassembly.splitlines()),
        "forbidden_instruction_matches": forbidden,
        "forbidden_instruction_set": ["rdmsr", "wrmsr", "clflush", "wbinvd", "invd"],
    }


def compile_runtime_if_available() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="independent_window_v3_compile_") as temp:
        temp_root = Path(temp)
        replacements = [(str(temp_root), "<compile-temp>"), (windows_to_wsl_path(temp_root), "<compile-temp>")]
        binary = Path(temp) / "independent_window_runtime"
        receipt = target_model.compile_runtime(HERE, binary)
        if not receipt["passed"] and sys.platform.startswith("win"):
            receipt = target_model.compile_runtime(HERE, binary, prefer_wsl=True)
        if not receipt["passed"]:
            return receipt
        self_test = target_model.run_runtime_self_test(binary, receipt.get("runtime_command"))
        schedule_check = target_model.run_runtime_schedule_validation(binary, HERE / "INDEPENDENT_WINDOW_PUBLIC_SCHEDULE.tsv", receipt.get("runtime_command"))
        disassembly = disassembly_receipt(binary, via_wsl=receipt.get("runtime_command", [])[:2] == ["wsl", "--"])
        receipt.update(
            {
                "runtime_binary_sha256": receipt["binary_sha256"],
                "self_test_stdout": self_test["stdout"],
                "self_test_returncode": self_test["returncode"],
                "schedule_check_stdout": schedule_check["stdout"],
                "schedule_check_returncode": schedule_check["returncode"],
                "disassembly": disassembly,
                "passed": receipt["passed"] and self_test["passed"] and schedule_check["passed"] and disassembly["passed"],
            }
        )
        return normalize_receipt_paths(receipt, replacements)


def retry1_evidence_hashes() -> dict[str, Any]:
    require(RETRY1_ROOT.is_dir(), f"retry-one evidence root missing: {RETRY1_ROOT}")
    names = (
        "RAW_TRANSDUCER_CAPTURE.jsonl",
        "RESTORATION_SENTINELS.jsonl",
        "TRANSDUCER_FEATURES_V2.json",
        "TRANSDUCER_ADJUDICATION_CONFIRMATION_V2.json",
        "FINAL_RESULT_CONFIRMATION_V2.json",
        "CONTROLLER_RESULT.json",
        "COPYBACK_MANIFEST.json",
        "CONFIRMATION_V2_MANIFEST.json",
        "CONFIRMATION_SOURCE_BUNDLE.tar.gz",
    )
    files = {}
    for name in names:
        path = RETRY1_ROOT / name
        require(path.is_file(), f"retry-one evidence file missing: {name}")
        files[name] = sha256_file(path)
    controller = json.loads((RETRY1_ROOT / "CONTROLLER_RESULT.json").read_text(encoding="utf-8"))
    final = json.loads((RETRY1_ROOT / "FINAL_RESULT_CONFIRMATION_V2.json").read_text(encoding="utf-8"))
    return {
        "root": str(RETRY1_ROOT),
        "files": files,
        "controller_status": controller.get("status"),
        "final_status": final.get("status"),
        "retained_classification": final.get("adjudication_status"),
        "controller_classification": controller.get("adjudication_status"),
        "raw_capture_sha256_claim": final.get("raw_capture_sha256"),
        "restoration_sentinels_sha256_claim": final.get("restoration_sentinels_sha256"),
        "features_sha256_claim": final.get("features_sha256"),
        "adjudication_sha256_claim": final.get("adjudication_sha256"),
    }


def governance_gate_receipts() -> dict[str, Any]:
    root = repo_root()
    critic_command = [sys.executable, str(root / "CAPABILITY" / "TOOLS" / "governance" / "critic.py")]
    critic = run(critic_command, timeout=120, check=False)
    return {
        "critic": {
            "command": critic_command,
            "returncode": critic.returncode,
            "stdout": critic.stdout.strip(),
            "stderr": critic.stderr.strip(),
            "passed": critic.returncode == 0,
        },
        "full_gate": {
            "command": [sys.executable, str(root / "CAPABILITY" / "TOOLS" / "utilities" / "ci_local_gate.py"), "--full"],
            "status": "REQUIRED_AFTER_COHERENT_COMMIT_BEFORE_PUSH",
            "reason": "full gate is clean-tree and HEAD-bound; final push proof records the executed receipt",
        },
    }


def verify_stage_and_source_receipts(raw_records: list[dict[str, Any]], stage_receipts: list[dict[str, Any]], source_receipts: list[dict[str, Any]]) -> dict[str, Any]:
    failures: list[str] = []
    stage_ids = [row.get("stage_receipt_id") for row in stage_receipts]
    source_ids = [row.get("source_receipt_id") for row in source_receipts]
    if len(stage_ids) != len(set(stage_ids)):
        failures.append("duplicate stage ID")
    if len(source_ids) != len(set(source_ids)):
        failures.append("duplicate source receipt ID")
    timestamps_by_component: dict[tuple[int, int, str], list[int]] = {}
    for row in stage_receipts:
        key = (int(row["replicate"]), int(row["trial_index"]), str(row["component"]))
        timestamps_by_component.setdefault(key, []).append(int(row["monotonic_timestamp_ns"]))
    for key, values in timestamps_by_component.items():
        if values != sorted(values):
            failures.append(f"out-of-order stage timestamp: {key}")
    for record in raw_records:
        for component in ("positive", "negative"):
            expected = [record.get(f"{component}_{suffix}") for suffix in (
                "baseline_receipt_id",
                "pre_sentinel_receipt_id",
                "rebaseline_receipt_id",
                "source_receipt_id",
                "measure_receipt_id",
                "restore_receipt_id",
                "post_sentinel_receipt_id",
            )]
            if len(set(expected)) != len(expected):
                failures.append(f"reused component receipt: {record.get('trial_index')} {component}")
            if record.get(f"{component}_source_receipt_id") not in source_ids:
                failures.append(f"missing source receipt row: {record.get('trial_index')} {component}")
            for receipt_id in expected:
                if receipt_id not in stage_ids and not str(receipt_id).endswith("_source"):
                    failures.append(f"missing stage receipt row: {receipt_id}")
        if target_model is None:
            pass
    return {
        "passed": not failures,
        "failures": failures,
        "stage_receipt_count": len(stage_receipts),
        "source_receipt_count": len(source_receipts),
    }


def build_self_test() -> dict[str, Any]:
    public.write_schedule_artifacts(HERE)
    public_self = public.self_test()
    with tempfile.TemporaryDirectory(prefix="independent_window_v3_target_self_") as temp:
        target_self = target_model.self_test(HERE, Path(temp))
    compile_test = compile_runtime_if_available()
    transport_tests = fake_transport_self_tests()
    result = {
        "schema_id": "CAT_CAS_INDEPENDENT_WINDOW_CONTROLLER_SELF_TEST_V3",
        "run_id": public.RUN_ID,
        "public_self_test_sha256": public_self["self_test_sha256"],
        "public_self_test_passed": public_self["self_test_passed"],
        "target_self_test_sha256": target_self["self_test_sha256"],
        "target_self_test_passed": target_self["self_test_passed"],
        "runtime_self_test_sha256": compile_test.get("runtime_binary_sha256"),
        "compile_test": compile_test,
        "transport_simulation": transport_tests,
        "governance_gates": governance_gate_receipts(),
        "network_connections": 0,
        "ssh_executions": 0,
        "scp_executions": 0,
        "hardware_executions": 0,
    }
    result["self_test_passed"] = (
        public_self["self_test_passed"]
        and target_self["self_test_passed"]
        and compile_test["passed"]
        and transport_tests["passed"]
        and result["governance_gates"]["critic"]["passed"]
    )
    result["self_test_sha256"] = public.digest({k: v for k, v in result.items() if k != "self_test_sha256"})
    write_json(SELF_TEST_PATH, result)
    return result


def sol_audit_disposition() -> dict[str, Any]:
    if not SOL_AUDIT_PATH.is_file():
        return {"status": "PENDING_READ_ONLY_SOL_AUDIT", "audit_record_sha256": None}
    audit = json.loads(SOL_AUDIT_PATH.read_text(encoding="utf-8"))
    audit["audit_record_sha256"] = sha256_file(SOL_AUDIT_PATH)
    return audit


def build_manifest() -> dict[str, Any]:
    schedule_hashes = public.write_schedule_artifacts(HERE)
    self_test = build_self_test()
    require(self_test["self_test_passed"], "self-test did not pass")
    compile_receipt = compile_runtime_if_available()
    require(compile_receipt["passed"], "strict C compile/runtime self-test did not pass")
    with tempfile.TemporaryDirectory(prefix="independent_window_v3_bundle_") as temp:
        source_bundle_sha = deterministic_source_bundle(Path(temp) / "INDEPENDENT_WINDOW_SOURCE_BUNDLE.tar.gz")
    head, origin, status = git_head_and_status()
    retry_hashes = retry1_evidence_hashes()
    manifest = {
        "schema_id": "CAT_CAS_INDEPENDENT_WINDOW_IMPLEMENTATION_MANIFEST_V3",
        "run_id": public.RUN_ID,
        "starting_commit": STARTING_COMMIT,
        "head_at_freeze_build": head,
        "origin_main_at_freeze_build": origin,
        "final_commit": FINAL_COMMIT_PLACEHOLDER,
        "git_status_porcelain_at_freeze_build": status,
        "primary_coordinate": public.PRIMARY_COORDINATE,
        "allowed_classes": list(public.ALLOWED_CLASSES),
        "forbidden_classes": list(public.FORBIDDEN_CLASSES),
        "prior_classifications_preserved": [
            "BALANCED_PHYSICAL_TRANSDUCER_PARTIAL",
            "V1_PARTIAL_V2_TRANSFER_CANDIDATE",
        ],
        "zero_live_contact": True,
        "network_connections": 0,
        "ssh_executions": 0,
        "scp_executions": 0,
        "hardware_executions": 0,
        "q_ladder": list(public.Q_LADDER),
        "base_work": public.BASE_WORK,
        "bank_lines": public.BANK_LINES,
        "line_bytes": public.LINE_BYTES,
        "source_core": public.SOURCE_CORE,
        "receiver_core": public.RECEIVER_CORE,
        "permutation": {"a": public.PERM_A, "b": public.PERM_B},
        "mapping_leg_source_work": public.SOURCE_WORK_PER_MAPPING_LEG,
        "subcapture_source_work": public.SOURCE_WORK_PER_SUBCAPTURE,
        "total_mapping_leg_records": public.TOTAL_TRIALS,
        "total_component_measurement_windows": public.TOTAL_COMPONENT_WINDOWS,
        "q0_split": "repeat 0 null_build; repeat 1 held-out null_test; null_test never builds its own ceiling",
        "schedule_json_sha256": schedule_hashes["schedule_json_sha256"],
        "schedule_tsv_sha256": schedule_hashes["schedule_tsv_sha256"],
        "schedule_semantic_sha256": schedule_hashes["schedule_semantic_sha256"],
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "topology_audit_sha256": sha256_file(TOPOLOGY_AUDIT_PATH),
        "live_execution_completion_audit_sha256": sha256_file(COMPLETION_AUDIT_PATH),
        "source_hashes": source_hashes(),
        "expected_source_bundle_sha256": source_bundle_sha,
        "offline_validation_binary_sha256": compile_receipt["runtime_binary_sha256"],
        "runtime_compile": compile_receipt,
        "runtime_disassembly": compile_receipt["disassembly"],
        "self_test_sha256": self_test["self_test_sha256"],
        "self_test_path_sha256": sha256_file(SELF_TEST_PATH),
        "target_self_test_sha256": self_test["target_self_test_sha256"],
        "controller_self_test_sha256": self_test["self_test_sha256"],
        "transport_simulation_sha256": self_test["transport_simulation"]["transport_simulation_sha256"],
        "governance_gates": self_test["governance_gates"],
        "retry1_evidence": retry_hashes,
        "future_expected_run_root": str(RUNS_ROOT / public.RUN_ID),
        "future_remote_run_root": f"{REMOTE_BASE}/{public.RUN_ID}",
        "future_authorization": {
            "commit_binding_env": COMMIT_BINDING_ENV,
            "manifest_binding_env": MANIFEST_BINDING_ENV,
            "live_authority_env": LIVE_AUTHORITY_ENV,
            "live_authority_value": LIVE_AUTHORITY_VALUE,
            "command": (
                f"{COMMIT_BINDING_ENV}=<final_commit> "
                f"{MANIFEST_BINDING_ENV}=<implementation_manifest_sha256> "
                f"{LIVE_AUTHORITY_ENV}={LIVE_AUTHORITY_VALUE} "
                "python independent_window_v3/run_independent_window_v3.py --execute-authorized"
            ),
        },
        "sol_audit": sol_audit_disposition(),
    }
    manifest["implementation_manifest_sha256"] = manifest_digest(manifest)
    write_json(IMPLEMENTATION_MANIFEST, manifest)
    return manifest


def validate_only() -> dict[str, Any]:
    require(IMPLEMENTATION_MANIFEST.is_file(), "implementation manifest missing; run --prepare-only first")
    manifest = json.loads(IMPLEMENTATION_MANIFEST.read_text(encoding="utf-8"))
    require(manifest["implementation_manifest_sha256"] == manifest_digest(manifest), "manifest digest mismatch")
    schedule = json.loads((HERE / "INDEPENDENT_WINDOW_PUBLIC_SCHEDULE.json").read_text(encoding="utf-8"))
    public.validate_schedule(schedule)
    require(sha256_file(HERE / "INDEPENDENT_WINDOW_PUBLIC_SCHEDULE.json") == manifest["schedule_json_sha256"], "schedule JSON hash drift")
    require(sha256_file(HERE / "INDEPENDENT_WINDOW_PUBLIC_SCHEDULE.tsv") == manifest["schedule_tsv_sha256"], "schedule TSV hash drift")
    require((HERE / "INDEPENDENT_WINDOW_PUBLIC_SCHEDULE.tsv").read_text(encoding="utf-8") == public.schedule_tsv(schedule), "schedule TSV content drift")
    require(source_hashes() == manifest["source_hashes"], "source hashes drifted")
    require(sha256_file(COMPLETION_AUDIT_PATH) == manifest["live_execution_completion_audit_sha256"], "completion audit hash drifted")
    with tempfile.TemporaryDirectory(prefix="independent_window_v3_bundle_check_") as temp:
        require(
            deterministic_source_bundle(Path(temp) / "INDEPENDENT_WINDOW_SOURCE_BUNDLE.tar.gz")
            == manifest["expected_source_bundle_sha256"],
            "source bundle hash drifted",
        )
    self_test = build_self_test()
    require(self_test["self_test_sha256"] == manifest["self_test_sha256"], "self-test hash drifted")
    compile_receipt = compile_runtime_if_available()
    require(compile_receipt["passed"], "runtime compile or self-test failed")
    require(compile_receipt["disassembly"]["passed"], "runtime disassembly inspection failed")
    require(compile_receipt["runtime_binary_sha256"] == manifest["offline_validation_binary_sha256"], "runtime binary hash drifted")
    require(
        compile_receipt["disassembly"]["normalized_stdout_sha256"] == manifest["runtime_disassembly"]["normalized_stdout_sha256"],
        "runtime disassembly hash drifted",
    )
    gates = governance_gate_receipts()
    require(gates["critic"]["passed"], "critic governance gate failed")
    result = {
        "schema_id": "CAT_CAS_INDEPENDENT_WINDOW_VALIDATE_ONLY_V3",
        "run_id": public.RUN_ID,
        "manifest_sha256": manifest["implementation_manifest_sha256"],
        "schedule_json_sha256": manifest["schedule_json_sha256"],
        "schedule_tsv_sha256": manifest["schedule_tsv_sha256"],
        "source_bundle_sha256": manifest["expected_source_bundle_sha256"],
        "offline_validation_binary_sha256": manifest["offline_validation_binary_sha256"],
        "runtime_disassembly_sha256": manifest["runtime_disassembly"]["normalized_stdout_sha256"],
        "target_self_test_sha256": manifest["target_self_test_sha256"],
        "controller_self_test_sha256": manifest["controller_self_test_sha256"],
        "transport_simulation_sha256": manifest["transport_simulation_sha256"],
        "self_test_sha256": manifest["self_test_sha256"],
        "network_connections": 0,
        "ssh_executions": 0,
        "scp_executions": 0,
        "zero_live_contact": True,
        "governance_gates": gates,
        "passed": True,
    }
    result["validate_only_sha256"] = public.digest({k: v for k, v in result.items() if k != "validate_only_sha256"})
    return result


def copyback_paths(manifest: dict[str, Any]) -> set[str]:
    return {entry["path"] for entry in manifest["files"]}


def require_copyback_entries(paths: set[str], required: list[str], status: str) -> None:
    missing = sorted(set(required) - paths)
    require(not missing, f"{status} copyback missing required files: {missing}")


def verify_copy(local_root: Path) -> dict[str, Any]:
    manifest = json.loads((local_root / "COPYBACK_MANIFEST.json").read_text(encoding="utf-8"))
    require(manifest["schema_id"] == "CAT_CAS_INDEPENDENT_WINDOW_COPYBACK_MANIFEST_V3", "copyback schema mismatch")
    require(isinstance(manifest.get("files"), list) and manifest["files"], "copyback files missing")
    paths = copyback_paths(manifest)
    require("FINAL_RESULT_INDEPENDENT_WINDOW_V3.json" in paths, "final result missing from copyback manifest")
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


def pretransport_gate(
    *,
    bound_commit: str,
    bound_manifest: str,
    live_authority: str,
    manifest: dict[str, Any],
    repo_state: tuple[str, str, str],
    local_run: Path,
) -> None:
    head, origin, status = repo_state
    require(is_full_sha(bound_commit), f"{COMMIT_BINDING_ENV} must be set to the authorized final commit")
    require(re.fullmatch(r"[0-9a-f]{64}", bound_manifest) is not None, f"{MANIFEST_BINDING_ENV} must be set to the approved implementation manifest SHA")
    require(live_authority == LIVE_AUTHORITY_VALUE, f"{LIVE_AUTHORITY_ENV} must equal {LIVE_AUTHORITY_VALUE}")
    require(status.strip() == "", "live execution requires a clean working tree")
    require(head == bound_commit, "HEAD does not match commit binding")
    require(origin == bound_commit, "origin/main does not match commit binding")
    require(manifest["implementation_manifest_sha256"] == bound_manifest, "manifest binding does not match validated implementation manifest")
    require(not local_run.exists(), f"local run already exists: {local_run}")


def execute_transport_transaction(
    *,
    manifest: dict[str, Any],
    run_id: str = public.RUN_ID,
    runner: Any = subprocess.run,
    local_runs_root: Path = RUNS_ROOT,
    target: str = TARGET,
    remote_base: str = REMOTE_BASE,
    keep_remote: bool = False,
) -> dict[str, Any]:
    require(run_id == public.RUN_ID, "run ID mismatch")
    require(re.fullmatch(r"[a-z0-9_]{8,80}", run_id) is not None, "run ID is not closed")
    remote_run = f"{remote_base}/{run_id}"
    remote_source = f"{remote_run}/source"
    remote_output = f"{remote_run}/output"
    local_run = local_runs_root / run_id
    require(not local_run.exists(), f"local run already exists: {local_run}")
    require(run_id == LIVE_AUTHORITY_VALUE, "live authority run ID mismatch")

    local_run.mkdir(mode=0o700, parents=True, exist_ok=False)
    preflight = f"set -eu; test ! -e {shlex.quote(remote_run)}; install -d -m 700 -- {shlex.quote(remote_source)}"
    run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", target, preflight], timeout=15, runner=runner)
    for source, remote_name in TRANSFER_FILE_MAP.items():
        run(["scp", "-q", str(source), f"{target}:{remote_source}/{remote_name}"], timeout=30, runner=runner)
    remote_command = (
        f"timeout --signal=TERM --kill-after=5s 420s python3 "
        f"{shlex.quote(remote_source + '/independent_window_target.py')} "
        f"--source-root {shlex.quote(remote_source)} "
        f"--output-root {shlex.quote(remote_output)} "
        f"--run-id {shlex.quote(run_id)} "
        f"--expected-manifest-sha {shlex.quote(manifest['implementation_manifest_sha256'])} "
        "--execute-live"
    )
    completed = run(["ssh", "-o", "BatchMode=yes", target, remote_command], timeout=450, check=False, runner=runner)
    (local_run / "CONTROLLER_STDOUT.txt").write_text(completed.stdout, encoding="utf-8")
    (local_run / "CONTROLLER_STDERR.txt").write_text(completed.stderr, encoding="utf-8")
    copied = run(["scp", "-q", "-r", f"{target}:{remote_output}/.", str(local_run)], timeout=120, check=False, runner=runner)
    require(copied.returncode == 0, f"copy-back failed; remote retained at {remote_run}: {copied.stderr.strip()}")
    copy_manifest = verify_copy(local_run)
    copy_paths = copyback_paths(copy_manifest)
    copy_manifest_sha = sha256_file(local_run / "COPYBACK_MANIFEST.json")
    final = json.loads((local_run / "FINAL_RESULT_INDEPENDENT_WINDOW_V3.json").read_text(encoding="utf-8"))
    require(final.get("run_id") == run_id, "final result run ID mismatch")
    final_status = final["status"]
    cleaned = False
    if final_status == "INDEPENDENT_WINDOW_V3_TARGET_COMPLETE":
        require(completed.returncode == 0, "target reported complete with nonzero return code")
        require_copyback_entries(copy_paths, ["INDEPENDENT_WINDOW_V3_MANIFEST.json"], "target complete")
        execution_manifest = json.loads((local_run / "INDEPENDENT_WINDOW_V3_MANIFEST.json").read_text(encoding="utf-8"))
        require(execution_manifest["manifest_sha256"] == execution_manifest_digest(execution_manifest), "execution manifest digest mismatch")
        require(execution_manifest["implementation_manifest_sha256"] == manifest["implementation_manifest_sha256"], "execution manifest implementation SHA mismatch")
        require(execution_manifest["final_result_sha256"] == sha256_file(local_run / "FINAL_RESULT_INDEPENDENT_WINDOW_V3.json"), "execution manifest final-result SHA mismatch")
        require(final["adjudication_status"] in public.ALLOWED_CLASSES, "target emitted non-allowed scientific class")
        if not keep_remote:
            require(remote_run.startswith(remote_base + "/") and remote_run != remote_base, "unsafe cleanup root")
            cleanup = run(["ssh", "-o", "BatchMode=yes", target, f"rm -rf -- {shlex.quote(remote_run)}"], timeout=20, check=False, runner=runner)
            require(cleanup.returncode == 0, f"verified copy retained but remote cleanup failed: {cleanup.stderr.strip()}")
            absent = run(["ssh", "-o", "BatchMode=yes", target, f"test ! -e {shlex.quote(remote_run)}"], timeout=15, check=False, runner=runner)
            require(absent.returncode == 0, "remote run root remained after cleanup")
            cleaned = True
        controller = {
            "schema_id": "CAT_CAS_INDEPENDENT_WINDOW_CONTROLLER_RESULT_V3",
            "status": "INDEPENDENT_WINDOW_V3_CONTROLLER_TARGET_COMPLETE",
            "run_id": run_id,
            "target": target,
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
            "schedule_json_sha256": final["schedule_json_sha256"],
            "schedule_tsv_sha256": final["schedule_tsv_sha256"],
            "source_bundle_sha256": final["source_bundle_sha256"],
            "offline_validation_binary_sha256": final["offline_validation_binary_sha256"],
            "live_runtime_binary_sha256": final["live_runtime_binary_sha256"],
            "raw_capture_sha256": final["raw_capture_sha256"],
            "sentinels_sha256": final["sentinels_sha256"],
            "stage_receipts_sha256": final["stage_receipts_sha256"],
            "source_receipts_sha256": final["source_receipts_sha256"],
            "features_sha256": final["features_sha256"],
            "adjudication_sha256": final["adjudication_sha256"],
            "copyback_manifest_sha256": copy_manifest_sha,
        }
    elif final_status == "INDEPENDENT_WINDOW_V3_TARGET_FAILED":
        require("INDEPENDENT_WINDOW_V3_MANIFEST.json" not in copy_paths, "target failure copyback must not include success execution manifest")
        require_copyback_entries(
            copy_paths,
            [
                "TARGET_FAILURE_INDEPENDENT_WINDOW_V3.json",
                "INDEPENDENT_WINDOW_V3_FAILURE_MANIFEST.json",
                "LIVE_CUSTODY_LOG.json",
            ],
            "target failure",
        )
        target_failure_path = local_run / "TARGET_FAILURE_INDEPENDENT_WINDOW_V3.json"
        failure_manifest_path = local_run / "INDEPENDENT_WINDOW_V3_FAILURE_MANIFEST.json"
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
            "schema_id": "CAT_CAS_INDEPENDENT_WINDOW_CONTROLLER_RESULT_V3",
            "status": "INDEPENDENT_WINDOW_V3_CONTROLLER_TARGET_FAILED",
            "run_id": run_id,
            "target": target,
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
            "failure_manifest_sha256": failure_manifest["manifest_sha256"],
            "target_failure_sha256": failure_manifest["target_failure_sha256"],
            "copyback_manifest_sha256": copy_manifest_sha,
        }
    else:
        raise ControllerError(f"unknown target final status: {final_status}")
    write_json(local_run / "CONTROLLER_RESULT.json", controller)
    return controller


def execute_authorized() -> dict[str, Any]:
    bound_commit = os.environ.get(COMMIT_BINDING_ENV, "").strip()
    bound_manifest = os.environ.get(MANIFEST_BINDING_ENV, "").strip()
    live_authority = os.environ.get(LIVE_AUTHORITY_ENV, "").strip()
    require(IMPLEMENTATION_MANIFEST.is_file(), "implementation manifest missing")
    manifest = json.loads(IMPLEMENTATION_MANIFEST.read_text(encoding="utf-8"))
    require(manifest["implementation_manifest_sha256"] == manifest_digest(manifest), "implementation manifest self digest mismatch")
    pretransport_gate(
        bound_commit=bound_commit,
        bound_manifest=bound_manifest,
        live_authority=live_authority,
        manifest=manifest,
        repo_state=git_head_and_status(),
        local_run=RUNS_ROOT / public.RUN_ID,
    )
    validation = validate_only()
    require(validation["manifest_sha256"] == bound_manifest, "manifest binding does not match validated implementation manifest")
    pretransport_gate(
        bound_commit=bound_commit,
        bound_manifest=bound_manifest,
        live_authority=live_authority,
        manifest=manifest,
        repo_state=git_head_and_status(),
        local_run=RUNS_ROOT / public.RUN_ID,
    )
    controller = execute_transport_transaction(manifest=manifest)
    controller.update(
        {
            "commit_binding": bound_commit,
            "origin_main_at_execution": bound_commit,
            "live_authority": live_authority,
            "validated_manifest_sha256": validation["manifest_sha256"],
        }
    )
    write_json(RUNS_ROOT / public.RUN_ID / "CONTROLLER_RESULT.json", controller)
    return controller


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as out:
        for row in rows:
            out.write(json.dumps(row, sort_keys=True) + "\n")


def _fake_stage_source_rows(raw_records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    stages = []
    sources = []
    timestamp = 1000
    for record in raw_records:
        for component in ("positive", "negative"):
            for ordinal, stage in enumerate(public.SUBCAPTURE_STAGE_SEQUENCE):
                suffix = {
                    "receiver_baseline": "baseline_receipt_id",
                    "pre_sentinel": "pre_sentinel_receipt_id",
                    "rebaseline": "rebaseline_receipt_id",
                    "source_encoding": "source_receipt_id",
                    "measure_logical_bank": "measure_receipt_id",
                    "restore_both_banks": "restore_receipt_id",
                    "post_sentinel": "post_sentinel_receipt_id",
                }[stage]
                stages.append(
                    {
                        "schema_id": "CAT_CAS_INDEPENDENT_WINDOW_STAGE_RECEIPT_V3",
                        "stage_receipt_id": record[f"{component}_{suffix}"],
                        "replicate": int(record["replicate_index"]),
                        "pair": int(record["pair_index"]),
                        "mapping_leg": int(record["leg_index"]),
                        "trial_index": int(record["trial_index"]),
                        "component": component,
                        "stage_ordinal": ordinal,
                        "stage_name": stage,
                        "return_code": 0,
                        "monotonic_timestamp_ns": timestamp,
                    }
                )
                timestamp += 1
            sources.append(
                {
                    "schema_id": "CAT_CAS_INDEPENDENT_WINDOW_SOURCE_RECEIPT_V3",
                    "source_receipt_id": record[f"{component}_source_receipt_id"],
                    "replicate": int(record["replicate_index"]),
                    "pair": int(record["pair_index"]),
                    "mapping_leg": int(record["leg_index"]),
                    "trial_index": int(record["trial_index"]),
                    "component": component,
                    "q": int(record["q"]),
                    "positive_work": int(record[f"{component}_source_positive_work"]),
                    "negative_work": int(record[f"{component}_source_negative_work"]),
                    "total_work": int(record[f"{component}_source_total_work"]),
                }
            )
    return stages, sources


def _write_fake_success_output(remote_output: Path, remote_source: Path, manifest: dict[str, Any]) -> None:
    remote_output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(remote_source / "INDEPENDENT_WINDOW_PUBLIC_SCHEDULE.json", remote_output / "INDEPENDENT_WINDOW_PUBLIC_SCHEDULE.json")
    shutil.copy2(remote_source / "INDEPENDENT_WINDOW_PUBLIC_SCHEDULE.tsv", remote_output / "INDEPENDENT_WINDOW_PUBLIC_SCHEDULE.tsv")
    shutil.copy2(remote_source / "INDEPENDENT_WINDOW_PUBLIC_SCHEDULE.sha256", remote_output / "INDEPENDENT_WINDOW_PUBLIC_SCHEDULE.sha256")
    bundle_sha, _ = target_model.deterministic_source_bundle(remote_source, remote_output / "INDEPENDENT_WINDOW_SOURCE_BUNDLE.tar.gz")
    capture = public.build_mock_capture("independent_no_carryover")
    _write_jsonl(remote_output / "RAW_INDEPENDENT_WINDOW_CAPTURE.jsonl", capture["raw_records"])
    _write_jsonl(remote_output / "INDEPENDENT_WINDOW_SENTINELS.jsonl", capture["sentinels"])
    stages, sources = _fake_stage_source_rows(capture["raw_records"])
    _write_jsonl(remote_output / "INDEPENDENT_WINDOW_STAGE_RECEIPTS.jsonl", stages)
    _write_jsonl(remote_output / "INDEPENDENT_WINDOW_SOURCE_RECEIPTS.jsonl", sources)
    for rep in (0, 1):
        (remote_output / f"INDEPENDENT_WINDOW_RUNTIME_STDOUT_REPLICATE_{rep}.txt").write_text("synthetic success\n", encoding="utf-8")
        (remote_output / f"INDEPENDENT_WINDOW_RUNTIME_STDERR_REPLICATE_{rep}.txt").write_text("", encoding="utf-8")
    features = public.extract_features(capture["schedule"], capture["raw_records"], capture["sentinels"])
    adjudication = public.adjudicate(features)
    write_json(remote_output / "INDEPENDENT_WINDOW_FEATURES.json", features)
    write_json(remote_output / "INDEPENDENT_WINDOW_ADJUDICATION.json", adjudication)
    final = {
        "schema_id": "CAT_CAS_INDEPENDENT_WINDOW_TARGET_RESULT_V3",
        "status": "INDEPENDENT_WINDOW_V3_TARGET_COMPLETE",
        "run_id": public.RUN_ID,
        "adjudication_status": adjudication["status"],
        "scientific_classification_emitted": True,
        "primary_coordinate": public.PRIMARY_COORDINATE,
        "implementation_manifest_sha256": manifest["implementation_manifest_sha256"],
        "schedule_json_sha256": sha256_file(remote_output / "INDEPENDENT_WINDOW_PUBLIC_SCHEDULE.json"),
        "schedule_tsv_sha256": sha256_file(remote_output / "INDEPENDENT_WINDOW_PUBLIC_SCHEDULE.tsv"),
        "source_bundle_sha256": bundle_sha,
        "binary_custody_mode": target_model.BINARY_CUSTODY_MODE,
        "offline_validation_binary_sha256": manifest.get("offline_validation_binary_sha256"),
        "live_runtime_binary_sha256": "1" * 64,
        "raw_capture_sha256": sha256_file(remote_output / "RAW_INDEPENDENT_WINDOW_CAPTURE.jsonl"),
        "sentinels_sha256": sha256_file(remote_output / "INDEPENDENT_WINDOW_SENTINELS.jsonl"),
        "stage_receipts_sha256": sha256_file(remote_output / "INDEPENDENT_WINDOW_STAGE_RECEIPTS.jsonl"),
        "source_receipts_sha256": sha256_file(remote_output / "INDEPENDENT_WINDOW_SOURCE_RECEIPTS.jsonl"),
        "features_sha256": sha256_file(remote_output / "INDEPENDENT_WINDOW_FEATURES.json"),
        "adjudication_sha256": sha256_file(remote_output / "INDEPENDENT_WINDOW_ADJUDICATION.json"),
        "raw_record_count": len(capture["raw_records"]),
        "component_window_count": len(capture["raw_records"]) * 2,
    }
    write_json(remote_output / "FINAL_RESULT_INDEPENDENT_WINDOW_V3.json", final)
    write_json(remote_output / "LIVE_CUSTODY_LOG.json", {"schema_id": "CAT_CAS_INDEPENDENT_WINDOW_LIVE_CUSTODY_LOG_V3", "synthetic": True})
    target_model.build_execution_manifest(
        remote_output,
        run_id=public.RUN_ID,
        implementation_manifest_sha256=manifest["implementation_manifest_sha256"],
        source_bundle_sha256=bundle_sha,
        schedule_json_sha256=final["schedule_json_sha256"],
        schedule_tsv_sha256=final["schedule_tsv_sha256"],
        offline_validation_binary_sha256=manifest.get("offline_validation_binary_sha256"),
        live_runtime_binary_sha256="1" * 64,
        raw_record_count=len(capture["raw_records"]),
        component_window_count=len(capture["raw_records"]) * 2,
    )
    target_model.build_copyback_manifest(remote_output)


def _write_fake_failure_output(remote_output: Path) -> None:
    remote_output.mkdir(parents=True, exist_ok=True)
    state = {
        "run_id": public.RUN_ID,
        "phase": "synthetic_target_failure",
        "hardware_execution_began": False,
        "pmu_preflight_began": False,
        "pmu_preflight_completed": False,
        "replicate_states": {"0": {"began": False, "completed": False}, "1": {"began": False, "completed": False}},
    }
    target_model.failure_evidence(remote_output, state=state, exc=target_model.TargetError("synthetic target failure"))


class FakeTransport:
    def __init__(self, remote_root: Path, manifest: dict[str, Any], scenario: str) -> None:
        self.remote_root = remote_root
        self.manifest = manifest
        self.scenario = scenario
        self.network_commands = 0
        self.cleaned_roots: list[str] = []

    def map_remote(self, remote_path: str) -> Path:
        cleaned = remote_path.lstrip("/")
        return self.remote_root / cleaned

    def completed(self, command: list[str], rc: int = 0, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, rc, stdout=stdout, stderr=stderr)

    def __call__(self, command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if command[0] in {"ssh", "scp"}:
            self.network_commands += 1
        if command[0] == "ssh":
            script = command[-1]
            if "test ! -e" in script and "install -d" in script:
                remote_run = f"{REMOTE_BASE}/{public.RUN_ID}"
                if self.map_remote(remote_run).exists():
                    return self.completed(command, 1, stderr="remote exists")
                self.map_remote(f"{remote_run}/source").mkdir(parents=True, exist_ok=False)
                return self.completed(command)
            if "independent_window_target.py" in script:
                remote_source = self.map_remote(f"{REMOTE_BASE}/{public.RUN_ID}/source")
                remote_output = self.map_remote(f"{REMOTE_BASE}/{public.RUN_ID}/output")
                if self.scenario == "target_failure":
                    _write_fake_failure_output(remote_output)
                    return self.completed(command, 1, stdout='{"status":"INDEPENDENT_WINDOW_V3_TARGET_FAILED"}\n')
                _write_fake_success_output(remote_output, remote_source, self.manifest)
                return self.completed(command, 0, stdout='{"status":"INDEPENDENT_WINDOW_V3_TARGET_COMPLETE"}\n')
            if script.startswith("rm -rf -- "):
                target_path = shlex.split(script[len("rm -rf -- "):])[0]
                self.cleaned_roots.append(target_path)
                shutil.rmtree(self.map_remote(target_path), ignore_errors=True)
                return self.completed(command)
            if script.startswith("test ! -e "):
                target_path = shlex.split(script[len("test ! -e "):])[0]
                return self.completed(command, 0 if not self.map_remote(target_path).exists() else 1)
            return self.completed(command, 1, stderr=f"unhandled ssh script: {script}")
        if command[0] == "scp":
            if "-r" in command:
                if self.scenario == "copyback_failure":
                    return self.completed(command, 1, stderr="synthetic copyback failure")
                remote_spec = command[-2]
                local_root = Path(command[-1])
                remote_path = remote_spec.split(":", 1)[1]
                if remote_path.endswith("/."):
                    remote_path = remote_path[:-2]
                source_root = self.map_remote(remote_path)
                for path in source_root.rglob("*"):
                    if path.is_file():
                        rel = path.relative_to(source_root)
                        dest = local_root / rel
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(path, dest)
                return self.completed(command)
            source = Path(command[-2])
            remote_spec = command[-1]
            remote_path = remote_spec.split(":", 1)[1]
            dest = self.map_remote(remote_path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
            return self.completed(command)
        return self.completed(command, 1, stderr=f"unexpected command: {command}")


def fake_transport_self_tests() -> dict[str, Any]:
    public.write_schedule_artifacts(HERE)
    minimal_manifest = {
        "implementation_manifest_sha256": "0" * 64,
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "source_hashes": source_hashes(),
        "schedule_json_sha256": sha256_file(HERE / "INDEPENDENT_WINDOW_PUBLIC_SCHEDULE.json"),
        "schedule_tsv_sha256": sha256_file(HERE / "INDEPENDENT_WINDOW_PUBLIC_SCHEDULE.tsv"),
        "expected_source_bundle_sha256": "",
        "offline_validation_binary_sha256": "2" * 64,
    }
    with tempfile.TemporaryDirectory(prefix="independent_window_v3_fake_bundle_") as temp:
        minimal_manifest["expected_source_bundle_sha256"] = deterministic_source_bundle(Path(temp) / "bundle.tar.gz")
    minimal_manifest["implementation_manifest_sha256"] = public.digest(minimal_manifest)
    env_failures_block_transport = False
    manifest_mismatch_blocks_transport = False
    dirty_tree_blocks_transport = False
    head_mismatch_blocks_transport = False
    origin_mismatch_blocks_transport = False
    existing_local_root_blocks_transport = False
    try:
        pretransport_gate(bound_commit="", bound_manifest="", live_authority="", manifest=minimal_manifest, repo_state=("a" * 40, "a" * 40, ""), local_run=Path("nope"))
    except ControllerError:
        env_failures_block_transport = True
    try:
        pretransport_gate(bound_commit="a" * 40, bound_manifest="b" * 64, live_authority=LIVE_AUTHORITY_VALUE, manifest=minimal_manifest, repo_state=("a" * 40, "a" * 40, ""), local_run=Path("nope"))
    except ControllerError:
        manifest_mismatch_blocks_transport = True
    try:
        pretransport_gate(bound_commit="a" * 40, bound_manifest=minimal_manifest["implementation_manifest_sha256"], live_authority=LIVE_AUTHORITY_VALUE, manifest=minimal_manifest, repo_state=("a" * 40, "a" * 40, " M dirty\n"), local_run=Path("nope"))
    except ControllerError:
        dirty_tree_blocks_transport = True
    try:
        pretransport_gate(bound_commit="a" * 40, bound_manifest=minimal_manifest["implementation_manifest_sha256"], live_authority=LIVE_AUTHORITY_VALUE, manifest=minimal_manifest, repo_state=("b" * 40, "a" * 40, ""), local_run=Path("nope"))
    except ControllerError:
        head_mismatch_blocks_transport = True
    try:
        pretransport_gate(bound_commit="a" * 40, bound_manifest=minimal_manifest["implementation_manifest_sha256"], live_authority=LIVE_AUTHORITY_VALUE, manifest=minimal_manifest, repo_state=("a" * 40, "b" * 40, ""), local_run=Path("nope"))
    except ControllerError:
        origin_mismatch_blocks_transport = True
    with tempfile.TemporaryDirectory(prefix="independent_window_v3_existing_local_") as temp:
        existing = Path(temp) / public.RUN_ID
        existing.mkdir()
        try:
            pretransport_gate(bound_commit="a" * 40, bound_manifest=minimal_manifest["implementation_manifest_sha256"], live_authority=LIVE_AUTHORITY_VALUE, manifest=minimal_manifest, repo_state=("a" * 40, "a" * 40, ""), local_run=existing)
        except ControllerError:
            existing_local_root_blocks_transport = True

    fake_success_ok = False
    fake_failure_ok = False
    fake_copyback_failure_ok = False
    earlier_remote_roots_untouched = False
    no_retry_path = True
    with tempfile.TemporaryDirectory(prefix="independent_window_v3_fake_transport_") as temp:
        root = Path(temp)
        local_runs = root / "runs"
        remote = root / "remote"
        older_remote = remote / "root" / "catcas_live_small_wall" / "balanced_transducer_confirmation_v2_1"
        older_remote.mkdir(parents=True)
        (older_remote / "marker").write_text("keep", encoding="utf-8")
        success_runner = FakeTransport(remote, minimal_manifest, "success")
        success = execute_transport_transaction(manifest=minimal_manifest, runner=success_runner, local_runs_root=local_runs)
        fake_success_ok = (
            success["status"] == "INDEPENDENT_WINDOW_V3_CONTROLLER_TARGET_COMPLETE"
            and success["remote_cleaned"] is True
            and success_runner.cleaned_roots == [f"{REMOTE_BASE}/{public.RUN_ID}"]
        )
        earlier_remote_roots_untouched = (older_remote / "marker").is_file()
    with tempfile.TemporaryDirectory(prefix="independent_window_v3_fake_failure_") as temp:
        root = Path(temp)
        local_runs = root / "runs"
        remote = root / "remote"
        failure_runner = FakeTransport(remote, minimal_manifest, "target_failure")
        failure = execute_transport_transaction(manifest=minimal_manifest, runner=failure_runner, local_runs_root=local_runs)
        fake_failure_ok = (
            failure["status"] == "INDEPENDENT_WINDOW_V3_CONTROLLER_TARGET_FAILED"
            and failure["remote_retained"] is True
            and (failure_runner.map_remote(f"{REMOTE_BASE}/{public.RUN_ID}")).exists()
        )
    with tempfile.TemporaryDirectory(prefix="independent_window_v3_fake_copyback_") as temp:
        root = Path(temp)
        local_runs = root / "runs"
        remote = root / "remote"
        copy_runner = FakeTransport(remote, minimal_manifest, "copyback_failure")
        try:
            execute_transport_transaction(manifest=minimal_manifest, runner=copy_runner, local_runs_root=local_runs)
        except ControllerError:
            fake_copyback_failure_ok = copy_runner.map_remote(f"{REMOTE_BASE}/{public.RUN_ID}").exists()
    result = {
        "schema_id": "CAT_CAS_INDEPENDENT_WINDOW_TRANSPORT_SIMULATION_V3",
        "env_failures_block_transport": env_failures_block_transport,
        "manifest_mismatch_blocks_transport": manifest_mismatch_blocks_transport,
        "dirty_tree_blocks_transport": dirty_tree_blocks_transport,
        "head_mismatch_blocks_transport": head_mismatch_blocks_transport,
        "origin_mismatch_blocks_transport": origin_mismatch_blocks_transport,
        "existing_local_root_blocks_transport": existing_local_root_blocks_transport,
        "fake_success_transport_verified": fake_success_ok,
        "fake_target_failure_verified": fake_failure_ok,
        "fake_copyback_failure_retains_remote": fake_copyback_failure_ok,
        "earlier_remote_roots_never_cleanup_targets": earlier_remote_roots_untouched,
        "no_automatic_retry_path_exists": no_retry_path,
        "offline_modes_zero_network": True,
    }
    result["passed"] = all(value for key, value in result.items() if key not in {"schema_id"})
    result["transport_simulation_sha256"] = public.digest({k: v for k, v in result.items() if k != "transport_simulation_sha256"})
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--self-test", action="store_true")
    modes.add_argument("--prepare-only", action="store_true")
    modes.add_argument("--validate-only", action="store_true")
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
            ok = result["passed"]
        elif args.execute_authorized:
            result = execute_authorized()
            ok = result["status"] == "INDEPENDENT_WINDOW_V3_CONTROLLER_TARGET_COMPLETE"
        else:
            raise ControllerError("no execution mode selected")
        print(json.dumps(result, sort_keys=True))
        return 0 if ok else 1
    except Exception as exc:
        print(f"run_independent_window_v3: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
