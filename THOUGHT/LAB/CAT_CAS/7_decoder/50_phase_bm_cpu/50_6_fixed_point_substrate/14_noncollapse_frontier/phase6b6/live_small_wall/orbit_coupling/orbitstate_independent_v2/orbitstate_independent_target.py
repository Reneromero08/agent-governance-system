#!/usr/bin/env python3
"""Target-side wrapper for the OrbitState independent-window experiment.

Offline modes compile and validate the frozen package without opening network
connections, touching PMU state, or contacting a lab target.  ``--execute-live``
is the future target entry point; it is only called by the controller after the
three exact authorization gates pass.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import platform
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Any

import orbitstate_independent_public as public


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[9]
COMMIT_ENV = "ORBITSTATE_INDEPENDENT_V2_COMMIT_BINDING"
MANIFEST_ENV = "ORBITSTATE_INDEPENDENT_V2_MANIFEST_SHA256"
AUTHORITY_ENV = "ORBITSTATE_INDEPENDENT_V2_LIVE_AUTHORITY"
AUTHORITY_VALUE = public.RUN_ID


class TargetError(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise TargetError(message)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_digest(value: Any) -> str:
    return public.digest(value)


def windows_to_wsl_path(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    parts = [part for part in resolved.parts[1:]]
    return "/mnt/" + drive + "/" + "/".join(parts)


def command_available(name: str) -> bool:
    return shutil.which(name) is not None


def run_command(command: list[str], *, timeout: float, check: bool = True) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(command, text=True, capture_output=True, timeout=timeout, check=False)
    if check and completed.returncode != 0:
        raise TargetError(
            f"command failed rc={completed.returncode}: {' '.join(command)}\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )
    return completed


def source_files(source_root: Path) -> list[Path]:
    names = [
        "ORBITSTATE_INDEPENDENT_CONTRACT_V2.md",
        "PUBLIC_TRANSDUCER_REFERENCE.json",
        "orbitstate_independent_public.py",
        "orbitstate_independent_runtime.c",
        "orbitstate_independent_runtime.h",
        "orbitstate_independent_target.py",
        "run_orbitstate_independent_v2.py",
        "ORBITSTATE_PUBLIC_SCHEDULE.json",
        "ORBITSTATE_PUBLIC_SCHEDULE.tsv",
        "ORBITSTATE_PUBLIC_SCHEDULE.sha256",
        "ORBITSTATE_PRIVATE_SOURCE_MAP.json",
        "ORBITSTATE_PRIVATE_SOURCE_MAP.sha256",
    ]
    return [source_root / name for name in names if (source_root / name).exists()]


def source_hashes(source_root: Path) -> dict[str, str]:
    return {path.name: sha256_file(path) for path in source_files(source_root)}


def deterministic_source_bundle(source_root: Path, output_path: Path) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    files = source_files(source_root)
    raw_tar = io.BytesIO()
    with tarfile.open(fileobj=raw_tar, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for path in sorted(files, key=lambda item: item.name):
            info = archive.gettarinfo(str(path), arcname=path.name)
            info.mtime = 0
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            with path.open("rb") as handle:
                archive.addfile(info, handle)
    raw_tar.seek(0)
    with gzip.GzipFile(filename="", mode="wb", fileobj=output_path.open("wb"), mtime=0) as gz:
        gz.write(raw_tar.getvalue())
    return {"path": str(output_path), "sha256": sha256_file(output_path), "file_count": len(files)}


def validate_schedule_artifacts(source_root: Path) -> dict[str, Any]:
    schedule = json.loads((source_root / "ORBITSTATE_PUBLIC_SCHEDULE.json").read_text(encoding="utf-8"))
    source_map = json.loads((source_root / "ORBITSTATE_PRIVATE_SOURCE_MAP.json").read_text(encoding="utf-8"))
    reference = json.loads((source_root / "PUBLIC_TRANSDUCER_REFERENCE.json").read_text(encoding="utf-8"))
    public.validate_public_schedule(schedule)
    public.validate_private_source_map(schedule, source_map)
    require(reference["public_q0_absolute_bound"] == public.PUBLIC_Q0_ABSOLUTE_BOUND, "public bound drift")
    require(reference["private_odd_signal_floor"] == public.PRIVATE_ODD_SIGNAL_FLOOR, "private signal floor drift")
    expected_tsv = public.public_schedule_tsv(schedule)
    require((source_root / "ORBITSTATE_PUBLIC_SCHEDULE.tsv").read_text(encoding="utf-8") == expected_tsv, "schedule TSV drift")
    private_sha_line = (source_root / "ORBITSTATE_PRIVATE_SOURCE_MAP.sha256").read_text(encoding="utf-8").strip()
    require(private_sha_line.startswith(public.digest(source_map)), "private source map sha file drift")
    return {
        "passed": True,
        "public_schedule_json_sha256": sha256_file(source_root / "ORBITSTATE_PUBLIC_SCHEDULE.json"),
        "public_schedule_tsv_sha256": sha256_file(source_root / "ORBITSTATE_PUBLIC_SCHEDULE.tsv"),
        "public_schedule_canonical_sha256": public.digest(schedule),
        "private_source_map_sha256": public.digest(source_map),
        "public_reference_sha256": public.digest(reference),
        "mapping_leg_records": len(schedule["rows"]),
    }


def public_manifest_blindness(source_root: Path) -> dict[str, Any]:
    schedule_json = (source_root / "ORBITSTATE_PUBLIC_SCHEDULE.json").read_text(encoding="utf-8").lower()
    schedule_tsv = (source_root / "ORBITSTATE_PUBLIC_SCHEDULE.tsv").read_text(encoding="utf-8").lower()
    forbidden = [
        "pre_projection_d",
        "pre_projection_fold",
        "source_off",
        "query_off",
        "post_projection",
        "declaration_sham",
        "query_scramble",
        "equal_orbit_odd_zero",
        "source_polarity_inversion_d",
        "q_theta",
        "positive_work",
        "negative_work",
        "dummy_work",
        "private_source_phase",
        "expected_sign",
        "expected_result",
    ]
    hits = [term for term in forbidden if term in schedule_json or term in schedule_tsv]
    return {"passed": not hits, "forbidden_hits": hits}


def function_block(text: str, name: str) -> str:
    start = text.find(name)
    if start < 0:
        return ""
    brace = text.find("{", start)
    if brace < 0:
        return ""
    depth = 0
    for index in range(brace, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[brace : index + 1]
    return ""


def process_boundary_static_proof(source_root: Path) -> dict[str, Any]:
    runtime = (source_root / "orbitstate_independent_runtime.c").read_text(encoding="utf-8")
    source_block = function_block(runtime, "source_child_loop")
    receiver_block = function_block(runtime, "receiver_execute_rows")
    source_opens = "load_private_record(private_map_path" in source_block
    receiver_does_not_parse = "load_private_record(" not in receiver_block and "fopen(private_map_path" not in receiver_block
    source_after_fork = "fork()" in receiver_block and "source_child_loop" in receiver_block
    return {
        "passed": source_opens and receiver_does_not_parse and source_after_fork,
        "source_child_opens_private_map": source_opens,
        "receiver_parses_private_map": not receiver_does_not_parse,
        "source_child_after_process_separation": source_after_fork,
    }


def compile_runtime(source_root: Path, output_root: Path) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    binary = output_root / "orbitstate_independent_runtime"
    source = source_root / "orbitstate_independent_runtime.c"
    include = source_root
    base_flags = [
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
    ]
    if platform.system().lower().startswith("win"):
        require(command_available("wsl"), "WSL gcc is required for strict Linux runtime compilation on Windows")
        command = [
            "wsl",
            "--",
            "gcc",
            *base_flags,
            "-I",
            windows_to_wsl_path(include),
            windows_to_wsl_path(source),
            "-lm",
            "-o",
            windows_to_wsl_path(binary),
        ]
        runtime_command = ["wsl", "--", windows_to_wsl_path(binary)]
        disassemble_command = ["wsl", "--", "objdump", "-d", windows_to_wsl_path(binary)]
    else:
        gcc = shutil.which("gcc")
        require(gcc is not None, "gcc not available")
        command = [gcc, *base_flags, "-I", str(include), str(source), "-lm", "-o", str(binary)]
        runtime_command = [str(binary)]
        disassemble_command = ["objdump", "-d", str(binary)]
    completed = run_command(command, timeout=60.0)
    return {
        "passed": True,
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "binary": str(binary),
        "binary_sha256": sha256_file(binary),
        "runtime_command": runtime_command,
        "disassemble_command": disassemble_command,
    }


def disassembly_receipt(compile_receipt: dict[str, Any], output_root: Path) -> dict[str, Any]:
    completed = run_command(compile_receipt["disassemble_command"], timeout=30.0)
    text = completed.stdout
    normalized = "\n".join(line.split("\t", 1)[-1].strip() for line in text.splitlines() if line.strip()) + "\n"
    path = output_root / "ORBITSTATE_RUNTIME_DISASSEMBLY_NORMALIZED.txt"
    path.write_text(normalized, encoding="utf-8")
    return {
        "passed": True,
        "path": str(path),
        "sha256": sha256_file(path),
        "line_count": len(normalized.splitlines()),
    }


def run_runtime_self_test(compile_receipt: dict[str, Any]) -> dict[str, Any]:
    completed = run_command([*compile_receipt["runtime_command"], "--self-test"], timeout=30.0)
    data = json.loads(completed.stdout)
    return {
        "passed": data["status"] == "ORBITSTATE_RUNTIME_SELF_TEST_OK",
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "data": data,
    }


def run_runtime_schedule_validation(compile_receipt: dict[str, Any], source_root: Path) -> dict[str, Any]:
    schedule_arg = str(source_root / "ORBITSTATE_PUBLIC_SCHEDULE.tsv")
    if compile_receipt["runtime_command"][0] == "wsl":
        schedule_arg = windows_to_wsl_path(source_root / "ORBITSTATE_PUBLIC_SCHEDULE.tsv")
    completed = run_command([*compile_receipt["runtime_command"], "--validate-schedule-tsv", schedule_arg], timeout=30.0)
    data = json.loads(completed.stdout)
    return {
        "passed": data["status"] == "ORBITSTATE_PUBLIC_SCHEDULE_TSV_OK",
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "data": data,
    }


def baseline_control_audit(public_self: dict[str, Any]) -> dict[str, Any]:
    required_null_controls = {
        "zero_coupling",
        "source_off_leakage",
        "query_off_leakage",
        "postprojection_odd_leakage",
        "sham_first_harmonic_leakage",
        "query_scramble_leakage",
        "equal_orbit_odd_leakage",
    }
    required_randomized_order_controls = {
        "mapping_bias",
        "source_order_bias",
        "subcapture_order_bias",
    }
    required = required_null_controls | required_randomized_order_controls
    expected_rejections = set(public_self.get("expected_rejections", []))
    mocks = public_self.get("mocks", {})
    missing = sorted(required - expected_rejections)
    failed = sorted(name for name in required if mocks.get(name, {}).get("passed") is not True)
    return {
        "passed": not missing and not failed,
        "required_null_controls": sorted(required_null_controls),
        "required_randomized_order_controls": sorted(required_randomized_order_controls),
        "missing_expected_rejections": missing,
        "failed_mocks": failed,
    }


def offline_validate(source_root: Path, output_root: Path) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    schedule_receipt = validate_schedule_artifacts(source_root)
    blindness = public_manifest_blindness(source_root)
    boundary = process_boundary_static_proof(source_root)
    public_self = public.self_test()
    baseline_controls = baseline_control_audit(public_self)
    bundle = deterministic_source_bundle(source_root, output_root / "ORBITSTATE_SOURCE_BUNDLE.tar.gz")
    compile_receipt = compile_runtime(source_root, output_root)
    runtime_self = run_runtime_self_test(compile_receipt)
    schedule_runtime = run_runtime_schedule_validation(compile_receipt, source_root)
    disassembly = disassembly_receipt(compile_receipt, output_root)
    result = {
        "schema": "ORBITSTATE_TARGET_OFFLINE_VALIDATE_V2",
        "zero_live_contact": True,
        "network_connections": 0,
        "hardware_executions": 0,
        "schedule": schedule_receipt,
        "public_manifest_blindness": blindness,
        "process_boundary_static_proof": boundary,
        "public_self_test_sha256": public_self["self_test_sha256"],
        "public_self_test_passed": public_self["self_test_passed"],
        "baseline_control_audit": baseline_controls,
        "source_bundle": bundle,
        "runtime_compile": compile_receipt,
        "runtime_self_test": runtime_self,
        "runtime_schedule_validation": schedule_runtime,
        "disassembly": disassembly,
        "source_hashes": source_hashes(source_root),
    }
    result["passed"] = all(
        [
            schedule_receipt["passed"],
            blindness["passed"],
            boundary["passed"],
            public_self["self_test_passed"],
            baseline_controls["passed"],
            runtime_self["passed"],
            schedule_runtime["passed"],
            compile_receipt["passed"],
            disassembly["passed"],
        ]
    )
    result["offline_validate_sha256"] = canonical_digest({key: value for key, value in result.items() if key != "offline_validate_sha256"})
    write_json(output_root / "ORBITSTATE_OFFLINE_VALIDATE.json", result)
    return result


def execute_live(
    source_root: Path,
    output_root: Path,
    *,
    run_id: str,
    expected_manifest_sha: str,
    expected_commit_binding: str,
) -> dict[str, Any]:
    require(run_id == public.RUN_ID, "run id mismatch")
    manifest_path = source_root / "ORBITSTATE_IMPLEMENTATION_MANIFEST.json"
    require(manifest_path.exists(), "implementation manifest missing")
    require(sha256_file(manifest_path) == expected_manifest_sha, "manifest file hash mismatch")
    require(len(expected_commit_binding) == 40 and all(ch in "0123456789abcdef" for ch in expected_commit_binding), "bad commit binding")
    require(os.environ.get(COMMIT_ENV) == expected_commit_binding, "target commit authority mismatch")
    require(os.environ.get(MANIFEST_ENV) == expected_manifest_sha, "target manifest authority mismatch")
    require(os.environ.get(AUTHORITY_ENV) == AUTHORITY_VALUE, "target live authority mismatch")
    output_root.mkdir(parents=True, exist_ok=True)
    for name in [
        "RAW_ORBITSTATE_RECEIVER_CAPTURE.jsonl",
        "ORBITSTATE_RECEIVER_SENTINELS.jsonl",
        "ORBITSTATE_STAGE_RECEIPTS.jsonl",
        "ORBITSTATE_SOURCE_RECEIPTS.jsonl",
        "ORBITSTATE_RECEIVER_FEATURES.json",
        "ORBITSTATE_RECEIVER_FEATURES.sha256",
        "ORBITSTATE_ADJUDICATION.json",
        "FINAL_RESULT_ORBITSTATE_INDEPENDENT_V2.json",
    ]:
        path = output_root / name
        if path.exists():
            path.unlink()
    compile_receipt = compile_runtime(source_root, output_root)
    runtime_command = compile_receipt["runtime_command"]
    schedule_arg = str(source_root / "ORBITSTATE_PUBLIC_SCHEDULE.tsv")
    private_arg = str(source_root / "ORBITSTATE_PRIVATE_SOURCE_MAP.json")
    output_arg = str(output_root)
    if runtime_command[0] == "wsl":
        schedule_arg = windows_to_wsl_path(source_root / "ORBITSTATE_PUBLIC_SCHEDULE.tsv")
        private_arg = windows_to_wsl_path(source_root / "ORBITSTATE_PRIVATE_SOURCE_MAP.json")
        output_arg = windows_to_wsl_path(output_root)
    for replicate in public.REPLICATES:
        run_command(
            [
                *runtime_command,
                "--run-schedule",
                "--schedule-tsv",
                schedule_arg,
                "--private-map",
                private_arg,
                "--output-root",
                output_arg,
                "--replicate",
                str(replicate),
            ],
            timeout=600.0,
        )
    schedule = json.loads((source_root / "ORBITSTATE_PUBLIC_SCHEDULE.json").read_text(encoding="utf-8"))
    raw = public.read_jsonl(output_root / "RAW_ORBITSTATE_RECEIVER_CAPTURE.jsonl")
    sentinels = public.read_jsonl(output_root / "ORBITSTATE_RECEIVER_SENTINELS.jsonl")
    stage = public.read_jsonl(output_root / "ORBITSTATE_STAGE_RECEIPTS.jsonl")
    features = public.extract_receiver_features(schedule, raw, sentinels, stage)
    write_json(output_root / "ORBITSTATE_RECEIVER_FEATURES.json", features)
    (output_root / "ORBITSTATE_RECEIVER_FEATURES.sha256").write_text(
        features["receiver_features_sha256"] + "  ORBITSTATE_RECEIVER_FEATURES.json\n", encoding="utf-8"
    )
    frozen_feature_sha = features["receiver_features_sha256"]
    require(public.receiver_feature_digest(features) == frozen_feature_sha, "receiver feature freeze verification failed")
    private_map = json.loads((source_root / "ORBITSTATE_PRIVATE_SOURCE_MAP.json").read_text(encoding="utf-8"))
    source_receipts = public.read_jsonl(output_root / "ORBITSTATE_SOURCE_RECEIPTS.jsonl")
    adjudication = public.adjudicate(
        public_schedule=schedule,
        receiver_features=features,
        receiver_features_sha256=frozen_feature_sha,
        private_source_map=private_map,
        source_receipts=source_receipts,
    )
    write_json(output_root / "ORBITSTATE_ADJUDICATION.json", adjudication)
    final = {
        "status": "ORBITSTATE_INDEPENDENT_TARGET_COMPLETE",
        "run_id": run_id,
        "result_class": adjudication["result_class"],
        "compile": compile_receipt,
        "receiver_features_sha256": frozen_feature_sha,
        "adjudication_sha256": sha256_file(output_root / "ORBITSTATE_ADJUDICATION.json"),
    }
    write_json(output_root / "FINAL_RESULT_ORBITSTATE_INDEPENDENT_V2.json", final)
    return final


def self_test(source_root: Path, output_root: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="orbitstate_target_self_") as temp:
        offline = offline_validate(source_root, Path(temp) / "offline")
    result = {
        "schema": "ORBITSTATE_TARGET_SELF_TEST_V2",
        "offline_validate_sha256": offline["offline_validate_sha256"],
        "offline_validate_passed": offline["passed"],
        "zero_live_contact": True,
    }
    result["self_test_passed"] = offline["passed"]
    result["self_test_sha256"] = canonical_digest({key: value for key, value in result.items() if key != "self_test_sha256"})
    write_json(output_root / "ORBITSTATE_TARGET_SELF_TEST.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--self-test", action="store_true")
    modes.add_argument("--offline-validate", action="store_true")
    modes.add_argument("--execute-live", action="store_true")
    parser.add_argument("--source-root", type=Path, default=HERE)
    parser.add_argument("--output-root", type=Path, default=HERE / "_target_out")
    parser.add_argument("--run-id", default=public.RUN_ID)
    parser.add_argument("--expected-manifest-sha")
    parser.add_argument("--expected-commit-binding")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.self_test:
            result = self_test(args.source_root.resolve(), args.output_root.resolve())
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0 if result["self_test_passed"] else 1
        if args.offline_validate:
            result = offline_validate(args.source_root.resolve(), args.output_root.resolve())
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0 if result["passed"] else 1
        require(args.expected_manifest_sha is not None, "--execute-live requires --expected-manifest-sha")
        require(args.expected_commit_binding is not None, "--execute-live requires --expected-commit-binding")
        result = execute_live(
            args.source_root.resolve(),
            args.output_root.resolve(),
            run_id=args.run_id,
            expected_manifest_sha=args.expected_manifest_sha,
            expected_commit_binding=args.expected_commit_binding,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        failure = {
            "status": "ORBITSTATE_INDEPENDENT_TARGET_FAILED",
            "error": str(exc),
            "zero_live_contact": not args.execute_live,
        }
        write_json(args.output_root.resolve() / "FINAL_RESULT_ORBITSTATE_INDEPENDENT_V2.json", failure)
        print(json.dumps(failure, indent=2, sort_keys=True), file=os.sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
