#!/usr/bin/env python3
"""Offline controller for the frozen Independent-Window Transducer V3 package."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import re
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
STARTING_COMMIT = "524b9e580cb13f547fd7e0638dcf25b3b66e1112"
FINAL_COMMIT_PLACEHOLDER = "AWAITING_LIVE_AUTHORIZATION"
COMMIT_BINDING_ENV = "INDEPENDENT_WINDOW_TRANSDUCER_V3_COMMIT_BINDING"
MANIFEST_BINDING_ENV = "INDEPENDENT_WINDOW_TRANSDUCER_V3_MANIFEST_SHA256"
LIVE_AUTHORITY_ENV = "INDEPENDENT_WINDOW_TRANSDUCER_V3_LIVE_AUTHORITY"
LIVE_AUTHORITY_VALUE = public.RUN_ID
IMPLEMENTATION_MANIFEST = HERE / "INDEPENDENT_WINDOW_IMPLEMENTATION_MANIFEST.json"
SELF_TEST_PATH = HERE / "INDEPENDENT_WINDOW_SELF_TEST.json"
SOL_AUDIT_PATH = HERE / "INDEPENDENT_WINDOW_V3_SOL_AUDIT.json"
CONTRACT_PATH = HERE / "INDEPENDENT_WINDOW_CONTRACT_V3.md"
TOPOLOGY_AUDIT_PATH = HERE / "RETRY1_MEASUREMENT_TOPOLOGY_AUDIT.md"
RETRY1_ROOT = RUNS_ROOT / "balanced_transducer_confirmation_v2_1"
SOURCE_FILE_MAP = {
    CONTRACT_PATH: "INDEPENDENT_WINDOW_CONTRACT_V3.md",
    TOPOLOGY_AUDIT_PATH: "RETRY1_MEASUREMENT_TOPOLOGY_AUDIT.md",
    HERE / "INDEPENDENT_WINDOW_PUBLIC_SCHEDULE.json": "INDEPENDENT_WINDOW_PUBLIC_SCHEDULE.json",
    HERE / "INDEPENDENT_WINDOW_PUBLIC_SCHEDULE.sha256": "INDEPENDENT_WINDOW_PUBLIC_SCHEDULE.sha256",
    HERE / "INDEPENDENT_WINDOW_PUBLIC_SCHEDULE.tsv": "INDEPENDENT_WINDOW_PUBLIC_SCHEDULE.tsv",
    HERE / "independent_window_public.py": "independent_window_public.py",
    HERE / "independent_window_runtime.c": "independent_window_runtime.c",
    HERE / "independent_window_runtime.h": "independent_window_runtime.h",
    HERE / "independent_window_target.py": "independent_window_target.py",
    HERE / "run_independent_window_v3.py": "run_independent_window_v3.py",
    SOL_AUDIT_PATH: "INDEPENDENT_WINDOW_V3_SOL_AUDIT.json",
}


class ControllerError(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise ControllerError(message)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


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


def git_head_and_status() -> tuple[str, str, str]:
    head = run(["git", "rev-parse", "HEAD"], timeout=10).stdout.strip()
    origin = run(["git", "rev-parse", "origin/main"], timeout=10).stdout.strip()
    status = run(["git", "status", "--porcelain=v1"], timeout=10).stdout
    return head, origin, status


def is_full_sha(value: str) -> bool:
    return re.fullmatch(r"[0-9a-f]{40}", value) is not None


def source_hashes() -> dict[str, str]:
    hashes = {}
    for source, name in SOURCE_FILE_MAP.items():
        require(source.is_file(), f"source file missing: {source}")
        hashes[name] = sha256_file(source)
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


def disassembly_receipt(binary: Path, *, via_wsl: bool) -> dict[str, Any]:
    if via_wsl:
        command = ["wsl", "--", "objdump", "-d", windows_to_wsl_path(binary)]
    else:
        objdump = shutil.which("objdump")
        if objdump is None:
            return {
                "available": False,
                "passed": False,
                "command": None,
                "reason": "objdump unavailable",
                "forbidden_instruction_matches": [],
            }
        command = [objdump, "-d", str(binary)]
    completed = run(command, timeout=20, check=False)
    disassembly = completed.stdout
    normalized_disassembly = disassembly.replace(str(binary), "<independent_window_runtime>")
    normalized_disassembly = normalized_disassembly.replace(windows_to_wsl_path(binary), "<independent_window_runtime>")
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
        "normalized_stdout_sha256": hashlib.sha256(normalized_disassembly.encode("utf-8")).hexdigest(),
        "stderr": completed.stderr.strip(),
        "line_count": len(disassembly.splitlines()),
        "forbidden_instruction_matches": forbidden,
        "forbidden_instruction_set": ["rdmsr", "wrmsr", "clflush", "wbinvd", "invd"],
    }


def compile_runtime_if_available() -> dict[str, Any]:
    compiler = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    with tempfile.TemporaryDirectory(prefix="independent_window_v3_compile_") as temp:
        binary = Path(temp) / "independent_window_runtime"
        if compiler:
            command = [
                compiler,
                *target_model.STRICT_C_FLAGS,
                "-I",
                str(HERE),
                str(HERE / "independent_window_runtime.c"),
                "-o",
                str(binary),
            ]
            runtime_command = [str(binary)]
            via_wsl = False
        else:
            wsl_check = run(["wsl", "--", "gcc", "--version"], timeout=10, check=False)
            if wsl_check.returncode != 0:
                return {
                    "available": False,
                    "passed": False,
                    "compiler": None,
                    "runtime_binary_sha256": None,
                    "note": "no local C compiler or WSL gcc available",
                }
            command = [
                "wsl",
                "--",
                "gcc",
                *target_model.STRICT_C_FLAGS,
                "-I",
                windows_to_wsl_path(HERE),
                windows_to_wsl_path(HERE / "independent_window_runtime.c"),
                "-o",
                windows_to_wsl_path(binary),
            ]
            runtime_command = ["wsl", "--", windows_to_wsl_path(binary)]
            via_wsl = True
        completed = run(command, timeout=30, check=False)
        if completed.returncode != 0:
            return {
                "available": True,
                "passed": False,
                "compiler": compiler or "wsl:gcc",
                "command": command,
                "returncode": completed.returncode,
                "stdout": completed.stdout.strip(),
                "stderr": completed.stderr.strip(),
                "runtime_binary_sha256": None,
            }
        binary_hash = sha256_file(binary)
        self_test = run(runtime_command + ["--self-test"], timeout=10, check=False)
        schedule_tsv_arg = str(HERE / "INDEPENDENT_WINDOW_PUBLIC_SCHEDULE.tsv")
        if runtime_command[:2] == ["wsl", "--"]:
            schedule_tsv_arg = windows_to_wsl_path(HERE / "INDEPENDENT_WINDOW_PUBLIC_SCHEDULE.tsv")
        schedule_check = run(runtime_command + ["--validate-schedule-tsv", schedule_tsv_arg], timeout=10, check=False)
        disassembly = disassembly_receipt(binary, via_wsl=via_wsl)
        return {
            "available": True,
            "passed": (
                self_test.returncode == 0
                and "INDEPENDENT_WINDOW_V3_RUNTIME_SELF_TEST_OK" in self_test.stdout
                and schedule_check.returncode == 0
                and "INDEPENDENT_WINDOW_V3_SCHEDULE_TSV_OK" in schedule_check.stdout
                and disassembly["passed"]
            ),
            "compiler": compiler or "wsl:gcc",
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
            "runtime_binary_sha256": binary_hash,
            "self_test_stdout": self_test.stdout.strip(),
            "self_test_stderr": self_test.stderr.strip(),
            "schedule_check_stdout": schedule_check.stdout.strip(),
            "schedule_check_stderr": schedule_check.stderr.strip(),
            "disassembly": disassembly,
        }


def windows_to_wsl_path(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    tail = resolved.as_posix().split(":", 1)[1]
    return f"/mnt/{drive}{tail}"


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


def repo_root() -> Path:
    return Path(run(["git", "rev-parse", "--show-toplevel"], timeout=10).stdout.strip())


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


def build_self_test() -> dict[str, Any]:
    public.write_schedule_artifacts(HERE)
    public_self = public.self_test()
    with tempfile.TemporaryDirectory(prefix="independent_window_v3_target_self_") as temp:
        target_receipt = target_model.offline_validate(HERE, Path(temp))
    result = {
        "schema_id": "CAT_CAS_INDEPENDENT_WINDOW_CONTROLLER_SELF_TEST_V3",
        "run_id": public.RUN_ID,
        "public_self_test": public_self,
        "target_offline_validation": {
            "passed": target_receipt["passed"],
            "source_bundle_sha256": target_receipt["source_bundle_sha256"],
            "runtime_binary_sha256": target_receipt["compile"]["binary_sha256"],
            "compile_passed": target_receipt["compile"]["passed"],
            "runtime_self_test_passed": target_receipt["runtime_self_test"]["passed"],
            "runtime_schedule_validation_passed": target_receipt["runtime_schedule_validation"]["passed"],
        },
        "governance_gates": governance_gate_receipts(),
    }
    result["self_test_passed"] = (
        public_self["self_test_passed"]
        and target_receipt["passed"]
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


def manifest_digest(manifest: dict[str, Any]) -> str:
    return public.digest({k: v for k, v in manifest.items() if k != "implementation_manifest_sha256"})


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
        "source_hashes": source_hashes(),
        "expected_source_bundle_sha256": source_bundle_sha,
        "offline_validation_binary_sha256": compile_receipt["runtime_binary_sha256"],
        "runtime_compile": compile_receipt,
        "runtime_disassembly": compile_receipt["disassembly"],
        "self_test_sha256": self_test["self_test_sha256"],
        "self_test_path_sha256": sha256_file(SELF_TEST_PATH),
        "governance_gates": self_test["governance_gates"],
        "retry1_evidence": retry_hashes,
        "future_expected_run_root": str(RUNS_ROOT / public.RUN_ID),
        "future_remote_run_root": f"/root/catcas_live_small_wall/{public.RUN_ID}",
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
    require(
        compile_receipt["runtime_binary_sha256"] == manifest["offline_validation_binary_sha256"],
        "runtime binary hash drifted",
    )
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
        "self_test_sha256": manifest["self_test_sha256"],
        "network_connections": 0,
        "ssh_executions": 0,
        "scp_executions": 0,
        "zero_live_contact": True,
        "runtime_disassembly_sha256": manifest["runtime_disassembly"]["normalized_stdout_sha256"],
        "governance_gates": gates,
        "passed": True,
    }
    result["validate_only_sha256"] = public.digest({k: v for k, v in result.items() if k != "validate_only_sha256"})
    return result


def execute_authorized() -> dict[str, Any]:
    bound_commit = os.environ.get(COMMIT_BINDING_ENV, "").strip()
    bound_manifest = os.environ.get(MANIFEST_BINDING_ENV, "").strip()
    live_authority = os.environ.get(LIVE_AUTHORITY_ENV, "").strip()
    require(is_full_sha(bound_commit), f"{COMMIT_BINDING_ENV} must be set to the authorized final commit")
    require(re.fullmatch(r"[0-9a-f]{64}", bound_manifest) is not None, f"{MANIFEST_BINDING_ENV} must be set to the approved implementation manifest SHA")
    require(live_authority == LIVE_AUTHORITY_VALUE, f"{LIVE_AUTHORITY_ENV} must equal {LIVE_AUTHORITY_VALUE}")
    head, origin, status = git_head_and_status()
    require(status.strip() == "", "live execution requires a clean working tree")
    require(head == bound_commit and origin == bound_commit, "HEAD and origin/main must match the commit binding")
    validation = validate_only()
    require(validation["manifest_sha256"] == bound_manifest, "manifest binding does not match validated implementation manifest")
    head_after, origin_after, status_after = git_head_and_status()
    require(status_after.strip() == "", "live authorization validation must leave a clean working tree")
    require(head_after == bound_commit and origin_after == bound_commit, "commit binding drifted during authorization validation")
    return {
        "schema_id": "CAT_CAS_INDEPENDENT_WINDOW_LIVE_AUTHORIZATION_GATE_V3",
        "run_id": public.RUN_ID,
        "status": "LIVE_TRANSPORT_NOT_INVOKED_BY_OFFLINE_FREEZE_PACKAGE",
        "zero_live_contact": True,
        "network_connections": 0,
        "ssh_executions": 0,
        "scp_executions": 0,
        "commit_binding": bound_commit,
        "manifest_binding": bound_manifest,
        "validated_manifest_sha256": validation["manifest_sha256"],
        "message": "This offline freeze package only verifies the authorization gate. A later live-contact task must explicitly authorize transport.",
    }


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
            ok = False
        else:
            raise ControllerError("no execution mode selected")
        print(json.dumps(result, sort_keys=True))
        return 0 if ok else 1
    except Exception as exc:
        print(f"run_independent_window_v3: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
