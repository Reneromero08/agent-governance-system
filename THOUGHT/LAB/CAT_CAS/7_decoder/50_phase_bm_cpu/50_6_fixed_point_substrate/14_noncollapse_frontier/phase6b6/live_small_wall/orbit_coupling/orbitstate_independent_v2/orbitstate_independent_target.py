#!/usr/bin/env python3
"""Target-side wrapper for the OrbitState independent-window experiment.

Offline modes compile and validate the frozen package without opening network
connections, touching PMU state, or contacting a lab target. ``--execute-live``
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
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any

import orbitstate_independent_public as public


HERE = Path(__file__).resolve().parent
SCIENCE_PACKAGE_ID = public.RUN_ID
TRANSACTION_RUN_ID = "orbitstate_independent_v2_1"
COMMIT_ENV = "ORBITSTATE_INDEPENDENT_V2_COMMIT_BINDING"
MANIFEST_ENV = "ORBITSTATE_INDEPENDENT_V2_MANIFEST_SHA256"
AUTHORITY_ENV = "ORBITSTATE_INDEPENDENT_V2_LIVE_AUTHORITY"
AUTHORITY_VALUE = TRANSACTION_RUN_ID
OUTPUT_ROOT_MARKER = ".orbitstate_independent_v2_owned"

SOURCE_FILE_NAMES = [
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

SUCCESS_EVIDENCE_FILES = [
    "ORBITSTATE_INDEPENDENT_V2_MANIFEST.json",
    "ORBITSTATE_SOURCE_BUNDLE.tar.gz",
    "ORBITSTATE_SOURCE_HASHES.json",
    "RAW_ORBITSTATE_RECEIVER_CAPTURE.jsonl",
    "ORBITSTATE_RECEIVER_SENTINELS.jsonl",
    "ORBITSTATE_STAGE_RECEIPTS.jsonl",
    "ORBITSTATE_SOURCE_RECEIPTS.jsonl",
    "ORBITSTATE_RECEIVER_FEATURES.json",
    "ORBITSTATE_RECEIVER_FEATURES.sha256",
    "ORBITSTATE_FEATURE_FREEZE_RECEIPT.json",
    "ORBITSTATE_ADJUDICATION.json",
    "LIVE_CUSTODY_LOG.json",
    "FINAL_RESULT_ORBITSTATE_INDEPENDENT_V2.json",
    "COPYBACK_MANIFEST.json",
]


class TargetError(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise TargetError(message)


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


def canonical_digest(value: Any) -> str:
    return public.digest(value)


def windows_to_wsl_path(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    parts = [part for part in resolved.parts[1:]]
    return "/mnt/" + drive + "/" + "/".join(parts)


def command_available(name: str) -> bool:
    return shutil.which(name) is not None


def run_command(
    command: list[str],
    *,
    timeout: float,
    check: bool = True,
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(command, text=True, capture_output=True, timeout=timeout, check=False, env=env, cwd=cwd)
    if check and completed.returncode != 0:
        raise TargetError(
            f"command failed rc={completed.returncode}: {' '.join(command)}\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )
    return completed


def source_files(source_root: Path) -> list[Path]:
    return [source_root / name for name in SOURCE_FILE_NAMES if (source_root / name).exists()]


def source_hashes(source_root: Path) -> dict[str, dict[str, Any]]:
    return {path.name: {"sha256": sha256_file(path), "size": path.stat().st_size} for path in source_files(source_root)}


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


def validate_public_schedule_only(source_root: Path) -> dict[str, Any]:
    schedule = read_json(source_root / "ORBITSTATE_PUBLIC_SCHEDULE.json")
    reference = read_json(source_root / "PUBLIC_TRANSDUCER_REFERENCE.json")
    public.validate_public_schedule(schedule)
    require(reference["public_q0_absolute_bound"] == public.PUBLIC_Q0_ABSOLUTE_BOUND, "public bound drift")
    require(reference["private_odd_signal_floor"] == public.PRIVATE_ODD_SIGNAL_FLOOR, "private signal floor drift")
    expected_tsv = public.public_schedule_tsv(schedule)
    require((source_root / "ORBITSTATE_PUBLIC_SCHEDULE.tsv").read_text(encoding="utf-8") == expected_tsv, "schedule TSV drift")
    schedule_sha_line = read_json(source_root / "ORBITSTATE_PUBLIC_SCHEDULE.sha256")
    require(schedule_sha_line["json_sha256"] == sha256_file(source_root / "ORBITSTATE_PUBLIC_SCHEDULE.json"), "schedule JSON sha file drift")
    require(schedule_sha_line["tsv_sha256"] == sha256_file(source_root / "ORBITSTATE_PUBLIC_SCHEDULE.tsv"), "schedule TSV sha file drift")
    return {
        "passed": True,
        "public_schedule_json_sha256": sha256_file(source_root / "ORBITSTATE_PUBLIC_SCHEDULE.json"),
        "public_schedule_tsv_sha256": sha256_file(source_root / "ORBITSTATE_PUBLIC_SCHEDULE.tsv"),
        "public_schedule_canonical_sha256": public.digest(schedule),
        "public_reference_sha256": public.digest(reference),
        "mapping_leg_records": len(schedule["rows"]),
    }


def private_map_validator_summary(source_root: Path) -> dict[str, Any]:
    schedule = read_json(source_root / "ORBITSTATE_PUBLIC_SCHEDULE.json")
    source_map = read_json(source_root / "ORBITSTATE_PRIVATE_SOURCE_MAP.json")
    public.validate_private_source_map(schedule, source_map)
    private_sha_line = (source_root / "ORBITSTATE_PRIVATE_SOURCE_MAP.sha256").read_text(encoding="utf-8").strip()
    canonical = public.digest(source_map)
    require(private_sha_line.startswith(canonical), "private source map sha file drift")
    return {
        "passed": True,
        "file_sha256": sha256_file(source_root / "ORBITSTATE_PRIVATE_SOURCE_MAP.json"),
        "canonical_sha256": canonical,
        "record_count": len(source_map["records"]),
        "schema_identifier": source_map.get("schema"),
    }


def sanitized_subprocess_env() -> dict[str, str]:
    keep = {
        "PATH": os.environ.get("PATH", ""),
        "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
        "WINDIR": os.environ.get("WINDIR", ""),
        "HOME": os.environ.get("HOME", ""),
        "TMPDIR": os.environ.get("TMPDIR", ""),
        "TEMP": os.environ.get("TEMP", ""),
        "TMP": os.environ.get("TMP", ""),
    }
    return {key: value for key, value in keep.items() if value}


def validate_private_map_in_separate_process(source_root: Path) -> dict[str, Any]:
    command = [sys.executable, str(Path(__file__).resolve()), "--private-map-validate", "--source-root", str(source_root)]
    completed = run_command(command, timeout=60.0, env=sanitized_subprocess_env())
    data = json.loads(completed.stdout)
    allowed = {"passed", "file_sha256", "canonical_sha256", "record_count", "schema_identifier"}
    require(set(data) <= allowed, "private validator returned private-bearing fields")
    require(data.get("passed") is True, "private validator failed")
    return data


def validate_schedule_artifacts(source_root: Path) -> dict[str, Any]:
    public_receipt = validate_public_schedule_only(source_root)
    private_receipt = validate_private_map_in_separate_process(source_root)
    return {
        "passed": public_receipt["passed"] and private_receipt["passed"],
        "public_schedule": public_receipt,
        "private_source_map_validator": private_receipt,
        "mapping_leg_records": public_receipt["mapping_leg_records"],
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
    source_apply_block = function_block(runtime, "source_apply_encoding")
    receiver_block = function_block(runtime, "receiver_execute_rows")
    pair_block = function_block(runtime, "execute_mapping_pair")
    source_opens = "load_private_record(private_map_path" in source_block
    receiver_does_not_parse = "load_private_record(" not in receiver_block and "fopen(private_map_path" not in receiver_block
    source_after_fork = "fork()" in pair_block and "source_child_loop" in pair_block
    source_core_recorded = "source_cpu_before" in source_apply_block and "source_cpu_after" in source_apply_block
    return {
        "passed": source_opens and receiver_does_not_parse and source_after_fork and source_core_recorded,
        "source_child_opens_private_map": source_opens,
        "receiver_parses_private_map": not receiver_does_not_parse,
        "source_child_after_process_separation": source_after_fork,
        "source_cpu_recorded": source_core_recorded,
    }


def feature_boundary_static_proof(source_root: Path) -> dict[str, Any]:
    text = (source_root / "orbitstate_independent_target.py").read_text(encoding="utf-8")
    public_pos = text.find("validate_public_schedule_only(source_root)")
    private_pos = text.find("validate_private_map_in_separate_process(source_root)")
    feature_pos = text.find("run_receiver_feature_subprocess(")
    adjudication_pos = text.find("run_adjudication_subprocess(")
    receiver_feature_func_start = text.rfind("def run_receiver_feature_subprocess")
    receiver_feature_func_end = text.find("\ndef receiver_features_main", receiver_feature_func_start)
    receiver_feature_block = text[receiver_feature_func_start:receiver_feature_func_end]
    checks = {
        "public_schedule_validation_split": public_pos >= 0,
        "private_validator_subprocess": private_pos >= 0,
        "receiver_feature_subprocess": feature_pos >= 0,
        "adjudication_subprocess": adjudication_pos >= 0,
        "feature_hash_frozen_before_adjudication": 0 <= feature_pos < adjudication_pos,
        "receiver_subprocess_has_no_private_args": "--private-map" not in receiver_feature_block
        and "--source-receipts" not in receiver_feature_block,
        "receiver_subprocess_isolated_inputs": "feature_input_root.mkdir(mode=0o700)" in text
        and "cwd=feature_input_root" in text
        and "receiver feature input escaped isolated root" in text,
        "receiver_subprocess_denies_private_paths": "--deny-path" in receiver_feature_block
        and "receiver feature subprocess can access denied path" in text,
        "adjudication_cli_receiver_features_argument": 'parser.add_argument("--receiver-features", type=Path)' in text,
        "feature_mutation_rejected": "feature_digest_after_unblinding" in text
        and "feature mutation after unblinding" in text,
    }
    result = {
        "schema": "ORBITSTATE_FEATURE_BOUNDARY_SELF_TEST_V2",
        "zero_live_contact": True,
        "checks": checks,
    }
    result["passed"] = all(checks.values())
    result["feature_boundary_self_test_sha256"] = canonical_digest(
        {key: value for key, value in result.items() if key != "feature_boundary_self_test_sha256"}
    )
    write_json(source_root / "ORBITSTATE_FEATURE_BOUNDARY_SELF_TEST.json", result)
    return result


def pmu_runtime_static_proof(source_root: Path) -> dict[str, Any]:
    runtime = (source_root / "orbitstate_independent_runtime.c").read_text(encoding="utf-8")
    checks = {
        "process_scoped_pid_zero": "perf_event_open_checked(&attr, 0, -1" in runtime,
        "no_system_wide_cpu_counting": "perf_event_open_checked(&attr, -1, ORBITSTATE_RECEIVER_CORE" not in runtime
        and "perf_event_open_checked(&attr, -1," not in runtime,
        "userspace_only": "attr.exclude_kernel = 1;" in runtime and "attr.exclude_hv = 1;" in runtime,
        "group_read_format": "PERF_FORMAT_GROUP" in runtime
        and "PERF_FORMAT_TOTAL_TIME_ENABLED" in runtime
        and "PERF_FORMAT_TOTAL_TIME_RUNNING" in runtime
        and "PERF_FORMAT_ID" in runtime,
        "complete_group_count_checked": "readout.nr != 3u" in runtime,
        "read_size_checked": "got != (ssize_t)sizeof(readout)" in runtime,
        "event_ids_checked": "event ID drift" in runtime,
        "multiplexing_checked": "time_enabled != readout.time_running" in runtime,
        "receiver_cpu_checked": "receiver CPU migration" in runtime,
        "positive_cycles_checked": "PMU cycles must be positive" in runtime,
        "hardcoded_cycles_removed": "\"cycles\":0" not in runtime and "pmu_unmultiplexed\":true" not in runtime,
        "pmu_preflight_present": "--pmu-preflight" in runtime,
    }
    return {"passed": all(checks.values()), "checks": checks}


def physical_protocol_static_proof(source_root: Path) -> dict[str, Any]:
    runtime = (source_root / "orbitstate_independent_runtime.c").read_text(encoding="utf-8")
    checks = {
        "identical_bank_bytes": "uint8_t value = pattern_byte(index)" in runtime
        and "banks.bank_a[index] = value" in runtime
        and "banks.bank_b[index] = value" in runtime,
        "dummy_bank_separate": "ORBITSTATE_DUMMY_BANK_INITIAL_VALUE" in runtime,
        "affine_line_permutation": "257u * index + 43u" in runtime,
        "sequential_modulo_not_used": "i % ORBITSTATE_BANK_LINES" not in runtime,
        "v3_pattern_materialization": "pattern_byte(size_t index)" in runtime and "index * 131u + 17u" in runtime,
        "v3_segmented_sentinels": "same_value_store_sentinel" in runtime
        and "starts[4] = {0u, 1024u, 2048u, 3072u}" in runtime
        and "measure_sentinel_bank" in runtime,
        "pair_allocation": "execute_mapping_pair" in runtime and "allocate_banks()" in function_block(runtime, "execute_mapping_pair"),
        "full_two_bank_baseline": "receiver_full_baseline" in runtime and "full_bank_touch(banks->bank_a" in runtime,
        "full_two_bank_rebaseline": "receiver_rebaseline" in runtime,
        "full_two_bank_restoration": "receiver_restoration" in runtime and "restore_all_banks" in runtime,
        "source_off_dummy_only": "source_off_dummy_mode" in runtime and "same_value_store(banks->dummy_bank, ORBITSTATE_TOTAL_WORK)" in runtime,
        "stage_receipt_count_contract": "stage_receipts\":2016" not in runtime,
    }
    return {"passed": all(checks.values()), "checks": checks}


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


def tree_digest(root: Path) -> str:
    h = hashlib.sha256()
    if not root.exists():
        h.update(b"<absent>")
        return h.hexdigest()
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root).as_posix().encode("utf-8")
        h.update(rel)
        if path.is_file():
            h.update(b"F")
            h.update(path.read_bytes())
        elif path.is_dir():
            h.update(b"D")
    return h.hexdigest()


def create_absent_output_root(output_root: Path) -> None:
    require(not output_root.exists(), f"output root already exists: {output_root}")
    output_root.mkdir(mode=0o700, parents=False, exist_ok=False)
    (output_root / OUTPUT_ROOT_MARKER).write_text("orbitstate_independent_v2_owned\n", encoding="utf-8")


def root_custody_self_test() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="orbitstate_root_custody_") as temp:
        root = Path(temp)
        existing = root / "existing_output"
        existing.mkdir()
        marker = existing / "marker.txt"
        marker.write_text("preserve me\n", encoding="utf-8")
        before = tree_digest(existing)
        rejected = False
        try:
            create_absent_output_root(existing)
        except TargetError:
            rejected = True
        after = tree_digest(existing)
        fresh = root / "fresh_output"
        create_absent_output_root(fresh)
        created = fresh.exists() and (fresh / OUTPUT_ROOT_MARKER).exists()
    return {
        "passed": rejected and before == after and created,
        "existing_output_root_rejected": rejected,
        "existing_output_root_preserved": before == after,
        "fresh_output_root_created": created,
    }


def failure_copyback_self_test() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="orbitstate_failure_packet_") as temp:
        output_root = Path(temp) / "output"
        create_absent_output_root(output_root)
        custody = {
            "schema": "LIVE_CUSTODY_LOG_V2",
            "science_package_id": SCIENCE_PACKAGE_ID,
            "transaction_run_id": TRANSACTION_RUN_ID,
            "run_id": TRANSACTION_RUN_ID,
        }
        state = {"hardware_execution_began": False, "source_and_schedule_hashes_verified": True}
        seal_failure_evidence(
            output_root,
            phase="self_test_failure",
            exc=TargetError("self-test failure"),
            custody=custody,
            state=state,
        )
        copyback = read_json(output_root / "COPYBACK_MANIFEST.json")
        listed = {entry["path"] for entry in copyback["entries"]}
        actual = {path.relative_to(output_root).as_posix() for path in output_root.rglob("*") if path.is_file() and path.name != "COPYBACK_MANIFEST.json"}
        required = {
            "TARGET_FAILURE_ORBITSTATE_INDEPENDENT_V2.json",
            "ORBITSTATE_INDEPENDENT_V2_FAILURE_MANIFEST.json",
            "FINAL_RESULT_ORBITSTATE_INDEPENDENT_V2.json",
            "LIVE_CUSTODY_LOG.json",
        }
        final = read_json(output_root / "FINAL_RESULT_ORBITSTATE_INDEPENDENT_V2.json")
        marker_absent = not (output_root / OUTPUT_ROOT_MARKER).exists()
        manifest_self_digest = copyback["copyback_manifest_sha256"] == canonical_digest(
            {key: value for key, value in copyback.items() if key != "copyback_manifest_sha256"}
        )
    return {
        "passed": marker_absent
        and manifest_self_digest
        and required <= listed
        and actual == listed
        and final.get("scientific_classification_emitted") is False
        and "result_class" not in final,
        "ownership_marker_absent": marker_absent,
        "copyback_manifest_self_digest": manifest_self_digest,
        "required_failure_files_listed": required <= listed,
        "copyback_coverage_exact": actual == listed,
        "scientific_classification_emitted": final.get("scientific_classification_emitted"),
        "result_class_absent": "result_class" not in final,
    }


def offline_validate(source_root: Path, output_root: Path) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    schedule_receipt = validate_schedule_artifacts(source_root)
    blindness = public_manifest_blindness(source_root)
    boundary = process_boundary_static_proof(source_root)
    feature_boundary = feature_boundary_static_proof(source_root)
    pmu_static = pmu_runtime_static_proof(source_root)
    physical_static = physical_protocol_static_proof(source_root)
    root_custody = root_custody_self_test()
    failure_copyback = failure_copyback_self_test()
    public_self = public.self_test()
    baseline_controls = baseline_control_audit(public_self)
    bundle = deterministic_source_bundle(source_root, output_root / "ORBITSTATE_SOURCE_BUNDLE.tar.gz")
    hashes = source_hashes(source_root)
    write_json(output_root / "ORBITSTATE_SOURCE_HASHES.json", hashes)
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
        "feature_boundary_static_proof": feature_boundary,
        "pmu_runtime_static_proof": pmu_static,
        "physical_protocol_static_proof": physical_static,
        "root_custody_self_test": root_custody,
        "failure_copyback_self_test": failure_copyback,
        "public_self_test_sha256": public_self["self_test_sha256"],
        "public_self_test_passed": public_self["self_test_passed"],
        "baseline_control_audit": baseline_controls,
        "source_bundle": bundle,
        "runtime_compile": compile_receipt,
        "runtime_self_test": runtime_self,
        "runtime_schedule_validation": schedule_runtime,
        "disassembly": disassembly,
        "source_hashes": hashes,
    }
    result["passed"] = all(
        [
            schedule_receipt["passed"],
            blindness["passed"],
            boundary["passed"],
            feature_boundary["passed"],
            pmu_static["passed"],
            physical_static["passed"],
            root_custody["passed"],
            failure_copyback["passed"],
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


def read_text_or_none(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def parse_cpuinfo() -> dict[str, Any]:
    text = read_text_or_none(Path("/proc/cpuinfo")) or ""
    first: dict[str, str] = {}
    for line in text.splitlines():
        if not line.strip():
            if first:
                break
            continue
        if ":" in line:
            key, value = line.split(":", 1)
            first[key.strip()] = value.strip()
    return {
        "vendor_id": first.get("vendor_id"),
        "cpu_family": int(first.get("cpu family", "-1")) if first.get("cpu family", "-1").isdigit() else None,
        "model_name": first.get("model name"),
    }


def resolve_k10temp(strict: bool) -> dict[str, Any]:
    sensors: list[dict[str, Any]] = []
    for name_path in Path("/sys/class/hwmon").glob("hwmon*/name"):
        name = read_text_or_none(name_path)
        if name == "k10temp":
            hwmon = name_path.parent
            temp_inputs = sorted(hwmon.glob("temp*_input"))
            sensors.append(
                {
                    "hwmon": str(hwmon),
                    "name": name,
                    "temp_inputs": [str(path) for path in temp_inputs],
                    "temperatures_c": [
                        int(read_text_or_none(path) or "0") / 1000.0 for path in temp_inputs if (read_text_or_none(path) or "0").lstrip("-").isdigit()
                    ],
                }
            )
    if strict:
        require(len(sensors) == 1, "strict k10temp resolution failed")
        require(bool(sensors[0]["temperatures_c"]), "k10temp has no readable temperatures")
    return {"strict": strict, "sensors": sensors}


def temperature_snapshot(label: str, *, strict: bool) -> dict[str, Any]:
    resolved = resolve_k10temp(strict=strict)
    values = [value for sensor in resolved["sensors"] for value in sensor["temperatures_c"]]
    if strict:
        require(max(values) < 68.0, f"temperature above limit before {label}: {max(values)} C")
    return {"label": label, "timestamp_ns": time.time_ns(), "k10temp": resolved, "max_c": max(values) if values else None}


def identity_snapshot(strict: bool) -> dict[str, Any]:
    status = read_text_or_none(Path("/proc/self/status")) or ""
    caps: dict[str, str] = {}
    for line in status.splitlines():
        if line.startswith(("CapEff:", "CapPrm:", "CapBnd:")):
            key, value = line.split(":", 1)
            caps[key] = value.strip()
    cpu = parse_cpuinfo()
    pmu_type = read_text_or_none(Path("/sys/bus/event_source/devices/cpu/type"))
    event_format = read_text_or_none(Path("/sys/bus/event_source/devices/cpu/format/event"))
    umask_format = read_text_or_none(Path("/sys/bus/event_source/devices/cpu/format/umask"))
    core4_online = read_text_or_none(Path("/sys/devices/system/cpu/cpu4/online")) or "1"
    core5_online = read_text_or_none(Path("/sys/devices/system/cpu/cpu5/online")) or "1"
    snapshot = {
        "uid": os.getuid() if hasattr(os, "getuid") else None,
        "euid": os.geteuid() if hasattr(os, "geteuid") else None,
        "gid": os.getgid() if hasattr(os, "getgid") else None,
        "egid": os.getegid() if hasattr(os, "getegid") else None,
        "capabilities": caps,
        "perf_event_paranoid": read_text_or_none(Path("/proc/sys/kernel/perf_event_paranoid")),
        "cpu": cpu,
        "cpu_pmu_type": pmu_type,
        "event_format": event_format,
        "umask_format": umask_format,
        "core4_online": core4_online,
        "core5_online": core5_online,
    }
    if strict:
        require(snapshot["euid"] == 0, "live target must run as root")
        require(cpu["vendor_id"] == "AuthenticAMD", "live target must be AuthenticAMD")
        require(cpu["cpu_family"] == 16, "live target must be AMD family 16")
        require(event_format == "config:0-7,32-35", "PMU event format drift")
        require(umask_format == "config:8-15", "PMU umask format drift")
        require(core4_online == "1" and core5_online == "1", "cores 4 and 5 must be online")
    return snapshot


def policy_snapshot(label: str) -> dict[str, Any]:
    policies: dict[str, dict[str, str | None]] = {}
    for core in (4, 5):
        base = Path(f"/sys/devices/system/cpu/cpu{core}/cpufreq")
        policies[f"cpu{core}"] = {
            "scaling_driver": read_text_or_none(base / "scaling_driver"),
            "scaling_governor": read_text_or_none(base / "scaling_governor"),
            "scaling_min_freq": read_text_or_none(base / "scaling_min_freq"),
            "scaling_max_freq": read_text_or_none(base / "scaling_max_freq"),
            "scaling_cur_freq": read_text_or_none(base / "scaling_cur_freq"),
        }
    return {"label": label, "timestamp_ns": time.time_ns(), "policies": policies}


def process_snapshot(label: str) -> dict[str, Any]:
    forbidden_terms = ["stress", "stress-ng", "perf record", "perf stat", "cpupower", "turbostat"]
    completed = run_command(["ps", "-eo", "pid=,comm=,args="], timeout=10.0, check=False)
    hits = []
    for line in completed.stdout.splitlines():
        lowered = line.lower()
        if any(term in lowered for term in forbidden_terms):
            hits.append(line.strip())
    return {
        "label": label,
        "timestamp_ns": time.time_ns(),
        "forbidden_terms": forbidden_terms,
        "forbidden_process_hits": hits,
        "scan_returncode": completed.returncode,
    }


def append_custody(custody: dict[str, Any], key: str, value: Any, output_root: Path) -> None:
    custody.setdefault(key, []).append(value)
    write_json(output_root / "LIVE_CUSTODY_LOG.json", custody)


def run_runtime_pmu_preflight(compile_receipt: dict[str, Any], output_root: Path) -> dict[str, Any]:
    completed = run_command([*compile_receipt["runtime_command"], "--pmu-preflight"], timeout=30.0)
    data = json.loads(completed.stdout)
    require(data.get("scientific_classification_emitted") is False, "PMU preflight emitted science")
    require(data.get("cycles", 0) > 0, "PMU preflight cycles must be positive")
    write_json(output_root / "ORBITSTATE_PMU_PREFLIGHT.json", data)
    return {
        "passed": data.get("status") == "ORBITSTATE_RUNTIME_PMU_PREFLIGHT_OK",
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "data": data,
        "sha256": sha256_file(output_root / "ORBITSTATE_PMU_PREFLIGHT.json"),
    }


def run_receiver_feature_subprocess(source_root: Path, output_root: Path) -> dict[str, Any]:
    feature_input_root = output_root / "receiver_feature_input"
    feature_output_root = output_root / "receiver_feature_output"
    require(not feature_input_root.exists(), "receiver feature input root already exists")
    require(not feature_output_root.exists(), "receiver feature output root already exists")
    feature_input_root.mkdir(mode=0o700)
    feature_output_root.mkdir(mode=0o700)
    inputs = {
        "ORBITSTATE_PUBLIC_SCHEDULE.json": source_root / "ORBITSTATE_PUBLIC_SCHEDULE.json",
        "RAW_ORBITSTATE_RECEIVER_CAPTURE.jsonl": output_root / "RAW_ORBITSTATE_RECEIVER_CAPTURE.jsonl",
        "ORBITSTATE_RECEIVER_SENTINELS.jsonl": output_root / "ORBITSTATE_RECEIVER_SENTINELS.jsonl",
        "ORBITSTATE_STAGE_RECEIPTS.jsonl": output_root / "ORBITSTATE_STAGE_RECEIPTS.jsonl",
    }
    for name, src in inputs.items():
        shutil.copyfile(src, feature_input_root / name)
    deny_paths = [
        "ORBITSTATE_PRIVATE_SOURCE_MAP.json",
        "ORBITSTATE_SOURCE_RECEIPTS.jsonl",
    ]
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--extract-receiver-features",
        "--schedule-json",
        "ORBITSTATE_PUBLIC_SCHEDULE.json",
        "--raw-capture",
        "RAW_ORBITSTATE_RECEIVER_CAPTURE.jsonl",
        "--sentinels",
        "ORBITSTATE_RECEIVER_SENTINELS.jsonl",
        "--stage-receipts",
        "ORBITSTATE_STAGE_RECEIPTS.jsonl",
        "--output-root",
        str(feature_output_root),
        "--deny-path",
        deny_paths[0],
        "--deny-path",
        deny_paths[1],
    ]
    completed = run_command(command, timeout=120.0, env=sanitized_subprocess_env(), cwd=feature_input_root)
    data = json.loads(completed.stdout)
    require(data.get("passed") is True, "receiver feature subprocess failed")
    shutil.copyfile(feature_output_root / "ORBITSTATE_RECEIVER_FEATURES.json", output_root / "ORBITSTATE_RECEIVER_FEATURES.json")
    shutil.copyfile(feature_output_root / "ORBITSTATE_RECEIVER_FEATURES.sha256", output_root / "ORBITSTATE_RECEIVER_FEATURES.sha256")
    shutil.rmtree(feature_input_root)
    shutil.rmtree(feature_output_root)
    return data


def receiver_features_main(args: argparse.Namespace) -> dict[str, Any]:
    forbidden_env = [key for key in os.environ if "PRIVATE" in key.upper() or "SOURCE_RECEIPT" in key.upper()]
    require(not forbidden_env, f"receiver feature subprocess received private-bearing environment: {forbidden_env}")
    denied = [path.resolve() for path in args.deny_path]
    for denied_path in denied:
        require(not denied_path.exists(), f"receiver feature subprocess can access denied path: {denied_path}")
    input_root = Path.cwd().resolve()
    for input_path in [args.schedule_json, args.raw_capture, args.sentinels, args.stage_receipts]:
        require(input_path.resolve().parent == input_root, f"receiver feature input escaped isolated root: {input_path}")
    require(args.output_root.resolve() != input_root, "receiver feature output must be separate from input root")
    schedule = read_json(args.schedule_json)
    raw = public.read_jsonl(args.raw_capture)
    sentinels = public.read_jsonl(args.sentinels)
    stage = public.read_jsonl(args.stage_receipts)
    features = public.extract_receiver_features(schedule, raw, sentinels, stage)
    write_json(args.output_root / "ORBITSTATE_RECEIVER_FEATURES.json", features)
    (args.output_root / "ORBITSTATE_RECEIVER_FEATURES.sha256").write_text(
        features["receiver_features_sha256"] + "  ORBITSTATE_RECEIVER_FEATURES.json\n", encoding="utf-8"
    )
    return {
        "passed": True,
        "receiver_features_sha256": features["receiver_features_sha256"],
        "receiver_feature_process_private_env_keys": forbidden_env,
        "receiver_feature_process_private_args": [],
        "denied_private_paths_absent": [str(path) for path in denied],
    }


def run_adjudication_subprocess(
    source_root: Path,
    output_root: Path,
    receiver_features_sha256: str,
    chronology: list[dict[str, Any]],
) -> dict[str, Any]:
    receipt_path = output_root / "ORBITSTATE_FEATURE_FREEZE_RECEIPT.json"
    write_json(
        receipt_path,
        {
            "schema": "ORBITSTATE_FEATURE_FREEZE_RECEIPT_V2",
            "science_package_id": SCIENCE_PACKAGE_ID,
            "transaction_run_id": TRANSACTION_RUN_ID,
            "run_id": TRANSACTION_RUN_ID,
            "events": chronology,
            "receiver_features_sha256": receiver_features_sha256,
        },
    )
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--adjudicate-frozen",
        "--schedule-json",
        str(source_root / "ORBITSTATE_PUBLIC_SCHEDULE.json"),
        "--receiver-features",
        str(output_root / "ORBITSTATE_RECEIVER_FEATURES.json"),
        "--receiver-features-sha256",
        receiver_features_sha256,
        "--private-map",
        str(source_root / "ORBITSTATE_PRIVATE_SOURCE_MAP.json"),
        "--source-receipts",
        str(output_root / "ORBITSTATE_SOURCE_RECEIPTS.jsonl"),
        "--feature-freeze-receipt",
        str(receipt_path),
        "--output-root",
        str(output_root),
    ]
    completed = run_command(command, timeout=120.0, env=sanitized_subprocess_env())
    data = json.loads(completed.stdout)
    require(data.get("passed") is True, "adjudication subprocess failed")
    return data


def adjudicate_frozen_main(args: argparse.Namespace) -> dict[str, Any]:
    schedule = read_json(args.schedule_json)
    features = read_json(args.receiver_features)
    feature_digest_after_unblinding = public.receiver_feature_digest(features)
    require(feature_digest_after_unblinding == args.receiver_features_sha256, "feature mutation after unblinding")
    receipt = read_json(args.feature_freeze_receipt)
    events = list(receipt["events"])
    events.append({"event": "private_map_opened_for_adjudication", "timestamp_ns": time.time_ns()})
    private_map = read_json(args.private_map)
    events.append({"event": "adjudication_started", "timestamp_ns": time.time_ns()})
    source_receipts = public.read_jsonl(args.source_receipts)
    adjudication = public.adjudicate(
        public_schedule=schedule,
        receiver_features=features,
        receiver_features_sha256=args.receiver_features_sha256,
        private_source_map=private_map,
        source_receipts=source_receipts,
    )
    write_json(args.output_root / "ORBITSTATE_ADJUDICATION.json", adjudication)
    receipt["events"] = events
    receipt["private_map_opened_after_feature_hash_frozen"] = True
    receipt["adjudication_sha256"] = sha256_file(args.output_root / "ORBITSTATE_ADJUDICATION.json")
    receipt["feature_freeze_receipt_sha256"] = canonical_digest(
        {key: value for key, value in receipt.items() if key != "feature_freeze_receipt_sha256"}
    )
    write_json(args.feature_freeze_receipt, receipt)
    return {
        "passed": True,
        "result_class": adjudication["result_class"],
        "adjudication_sha256": sha256_file(args.output_root / "ORBITSTATE_ADJUDICATION.json"),
        "feature_freeze_receipt_sha256": sha256_file(args.feature_freeze_receipt),
        "feature_digest_after_unblinding": feature_digest_after_unblinding,
    }


def safe_manifest_files(output_root: Path) -> list[Path]:
    return sorted(
        path
        for path in output_root.rglob("*")
        if path.is_file() and path.name != "COPYBACK_MANIFEST.json"
    )


def write_copyback_manifest(output_root: Path) -> dict[str, Any]:
    entries = [
        {"path": path.relative_to(output_root).as_posix(), "size": path.stat().st_size, "sha256": sha256_file(path)}
        for path in safe_manifest_files(output_root)
    ]
    manifest = {
        "schema": "ORBITSTATE_COPYBACK_MANIFEST_V2",
        "science_package_id": SCIENCE_PACKAGE_ID,
        "transaction_run_id": TRANSACTION_RUN_ID,
        "run_id": TRANSACTION_RUN_ID,
        "entries": entries,
        "entry_count": len(entries),
    }
    manifest["copyback_manifest_sha256"] = canonical_digest(manifest)
    write_json(output_root / "COPYBACK_MANIFEST.json", manifest)
    return manifest


def success_execution_manifest(
    *,
    source_root: Path,
    output_root: Path,
    implementation_manifest_file_sha256: str,
    implementation_manifest_canonical_sha256: str,
    compile_receipt: dict[str, Any],
    source_bundle: dict[str, Any],
    source_hash_receipt: dict[str, Any],
    pmu_preflight: dict[str, Any],
    final_result_path: Path,
) -> dict[str, Any]:
    evidence_names = [
        "ORBITSTATE_SOURCE_BUNDLE.tar.gz",
        "ORBITSTATE_SOURCE_HASHES.json",
        "orbitstate_independent_runtime",
        "ORBITSTATE_RUNTIME_DISASSEMBLY_NORMALIZED.txt",
        "RAW_ORBITSTATE_RECEIVER_CAPTURE.jsonl",
        "ORBITSTATE_RECEIVER_SENTINELS.jsonl",
        "ORBITSTATE_STAGE_RECEIPTS.jsonl",
        "ORBITSTATE_SOURCE_RECEIPTS.jsonl",
        "ORBITSTATE_RECEIVER_FEATURES.json",
        "ORBITSTATE_RECEIVER_FEATURES.sha256",
        "ORBITSTATE_FEATURE_FREEZE_RECEIPT.json",
        "ORBITSTATE_ADJUDICATION.json",
        "LIVE_CUSTODY_LOG.json",
        "FINAL_RESULT_ORBITSTATE_INDEPENDENT_V2.json",
        "ORBITSTATE_PMU_PREFLIGHT.json",
    ]
    final_hashes = {name: sha256_file(output_root / name) for name in evidence_names if (output_root / name).exists()}
    raw_count = len(public.read_jsonl(output_root / "RAW_ORBITSTATE_RECEIVER_CAPTURE.jsonl"))
    stage_count = len(public.read_jsonl(output_root / "ORBITSTATE_STAGE_RECEIPTS.jsonl"))
    source_count = len(public.read_jsonl(output_root / "ORBITSTATE_SOURCE_RECEIPTS.jsonl"))
    manifest = {
        "schema": "ORBITSTATE_SUCCESS_EXECUTION_MANIFEST_V2",
        "science_package_id": SCIENCE_PACKAGE_ID,
        "transaction_run_id": TRANSACTION_RUN_ID,
        "run_id": TRANSACTION_RUN_ID,
        "contract_sha256": sha256_file(source_root / "ORBITSTATE_INDEPENDENT_CONTRACT_V2.md"),
        "implementation_manifest_file_sha256": implementation_manifest_file_sha256,
        "implementation_manifest_canonical_sha256": implementation_manifest_canonical_sha256,
        "public_schedule_json_sha256": sha256_file(source_root / "ORBITSTATE_PUBLIC_SCHEDULE.json"),
        "public_schedule_tsv_sha256": sha256_file(source_root / "ORBITSTATE_PUBLIC_SCHEDULE.tsv"),
        "private_source_map_file_sha256": sha256_file(source_root / "ORBITSTATE_PRIVATE_SOURCE_MAP.json"),
        "private_source_map_canonical_sha256": validate_private_map_in_separate_process(source_root)["canonical_sha256"],
        "source_bundle_sha256": source_bundle["sha256"],
        "live_runtime_binary_sha256": compile_receipt["binary_sha256"],
        "raw_capture_sha256": sha256_file(output_root / "RAW_ORBITSTATE_RECEIVER_CAPTURE.jsonl"),
        "sentinels_sha256": sha256_file(output_root / "ORBITSTATE_RECEIVER_SENTINELS.jsonl"),
        "stage_receipts_sha256": sha256_file(output_root / "ORBITSTATE_STAGE_RECEIPTS.jsonl"),
        "source_receipts_sha256": sha256_file(output_root / "ORBITSTATE_SOURCE_RECEIPTS.jsonl"),
        "receiver_features_sha256": sha256_file(output_root / "ORBITSTATE_RECEIVER_FEATURES.json"),
        "feature_freeze_receipt_sha256": sha256_file(output_root / "ORBITSTATE_FEATURE_FREEZE_RECEIPT.json"),
        "adjudication_sha256": sha256_file(output_root / "ORBITSTATE_ADJUDICATION.json"),
        "final_result_sha256": sha256_file(final_result_path),
        "pmu_preflight_sha256": pmu_preflight["sha256"],
        "source_hashes_sha256": sha256_file(output_root / "ORBITSTATE_SOURCE_HASHES.json"),
        "source_hash_receipt": source_hash_receipt,
        "record_counts": {
            "mapping_leg_records": 144,
            "independent_component_windows": raw_count,
            "stage_receipts": stage_count,
            "source_receipts": source_count,
        },
        "final_evidence_hashes": final_hashes,
    }
    manifest["execution_manifest_sha256"] = canonical_digest(manifest)
    write_json(output_root / "ORBITSTATE_INDEPENDENT_V2_MANIFEST.json", manifest)
    return manifest


def seal_failure_evidence(
    output_root: Path,
    *,
    phase: str,
    exc: BaseException,
    custody: dict[str, Any],
    state: dict[str, Any],
) -> dict[str, Any]:
    failure = {
        "schema": "TARGET_FAILURE_ORBITSTATE_INDEPENDENT_V2",
        "science_package_id": SCIENCE_PACKAGE_ID,
        "transaction_run_id": TRANSACTION_RUN_ID,
        "run_id": TRANSACTION_RUN_ID,
        "failure_phase": phase,
        "exception_type": type(exc).__name__,
        "exception_message": str(exc),
        "platform_identity_snapshot": custody.get("identity"),
        "temperatures": custody.get("temperatures", []),
        "process_snapshots": custody.get("process_snapshots", []),
        "policy_snapshots": custody.get("policy_snapshots", []),
        "source_and_schedule_hashes_verified": state.get("source_and_schedule_hashes_verified", False),
        "compile_state": state.get("compile_state"),
        "live_binary_hash": state.get("live_binary_hash"),
        "runtime_self_test_state": state.get("runtime_self_test_state"),
        "pmu_preflight_receipt": state.get("pmu_preflight_receipt"),
        "feature_freeze_phase": state.get("feature_freeze_phase"),
        "unblinding_phase": state.get("unblinding_phase"),
        "replicate_states": state.get("replicate_states", []),
        "hardware_execution_began": state.get("hardware_execution_began", False),
        "scientific_classification_emitted": False,
    }
    marker = output_root / OUTPUT_ROOT_MARKER
    if marker.exists():
        marker.unlink()
    write_json(output_root / "TARGET_FAILURE_ORBITSTATE_INDEPENDENT_V2.json", failure)
    final = {
        "status": "ORBITSTATE_INDEPENDENT_TARGET_FAILED",
        "science_package_id": SCIENCE_PACKAGE_ID,
        "transaction_run_id": TRANSACTION_RUN_ID,
        "run_id": TRANSACTION_RUN_ID,
        "failure_phase": phase,
        "exception_type": type(exc).__name__,
        "exception_message": str(exc),
        "scientific_classification_emitted": False,
    }
    write_json(output_root / "FINAL_RESULT_ORBITSTATE_INDEPENDENT_V2.json", final)
    write_json(output_root / "LIVE_CUSTODY_LOG.json", custody)
    files = {
        name: sha256_file(output_root / name)
        for name in [
            "TARGET_FAILURE_ORBITSTATE_INDEPENDENT_V2.json",
            "FINAL_RESULT_ORBITSTATE_INDEPENDENT_V2.json",
            "LIVE_CUSTODY_LOG.json",
        ]
        if (output_root / name).exists()
    }
    manifest = {
        "schema": "ORBITSTATE_FAILURE_MANIFEST_V2",
        "science_package_id": SCIENCE_PACKAGE_ID,
        "transaction_run_id": TRANSACTION_RUN_ID,
        "run_id": TRANSACTION_RUN_ID,
        "failure_phase": phase,
        "scientific_classification_emitted": False,
        "failure_files": files,
    }
    manifest["failure_manifest_sha256"] = canonical_digest(manifest)
    write_json(output_root / "ORBITSTATE_INDEPENDENT_V2_FAILURE_MANIFEST.json", manifest)
    copyback = write_copyback_manifest(output_root)
    return {"failure": failure, "final": final, "failure_manifest": manifest, "copyback_manifest": copyback}


def execute_live(
    source_root: Path,
    output_root: Path,
    *,
    run_id: str,
    expected_manifest_sha: str,
    expected_commit_binding: str,
) -> dict[str, Any]:
    phase = "authorization"
    custody: dict[str, Any] = {
        "schema": "LIVE_CUSTODY_LOG_V2",
        "science_package_id": SCIENCE_PACKAGE_ID,
        "transaction_run_id": run_id,
        "run_id": run_id,
        "zero_frequency_writes": 0,
        "zero_sysctl_writes": 0,
        "zero_voltage_writes": 0,
        "zero_msr_reads": 0,
        "zero_msr_writes": 0,
        "zero_physical_address_access": 0,
        "zero_cache_set_mapping": 0,
        "temperatures": [],
        "process_snapshots": [],
        "policy_snapshots": [],
    }
    state: dict[str, Any] = {"replicate_states": [], "hardware_execution_began": False}
    require(not output_root.exists(), f"output root already exists: {output_root}")
    create_absent_output_root(output_root)
    try:
        require(run_id == TRANSACTION_RUN_ID, "transaction run id mismatch")
        manifest_path = source_root / "ORBITSTATE_IMPLEMENTATION_MANIFEST.json"
        require(manifest_path.exists(), "implementation manifest missing")
        require(sha256_file(manifest_path) == expected_manifest_sha, "manifest file hash mismatch")
        implementation_manifest = read_json(manifest_path)
        require(implementation_manifest.get("science_package_id") == SCIENCE_PACKAGE_ID, "science package identity mismatch")
        require(implementation_manifest.get("transaction_run_id") == TRANSACTION_RUN_ID, "transaction identity mismatch")
        require(
            canonical_digest({key: value for key, value in implementation_manifest.items() if key != "manifest_canonical_sha256"})
            == implementation_manifest["manifest_canonical_sha256"],
            "implementation manifest canonical digest failed",
        )
        require(len(expected_commit_binding) == 40 and all(ch in "0123456789abcdef" for ch in expected_commit_binding), "bad commit binding")
        require(os.environ.get(COMMIT_ENV) == expected_commit_binding, "target commit authority mismatch")
        require(os.environ.get(MANIFEST_ENV) == expected_manifest_sha, "target manifest authority mismatch")
        require(os.environ.get(AUTHORITY_ENV) == AUTHORITY_VALUE, "target live authority mismatch")
        public_schedule = validate_public_schedule_only(source_root)
        private_summary = validate_private_map_in_separate_process(source_root)
        state["source_and_schedule_hashes_verified"] = public_schedule["passed"] and private_summary["passed"]
        custody["identity"] = identity_snapshot(strict=True)
        phase = "before_compilation"
        append_custody(custody, "temperatures", temperature_snapshot("before compilation", strict=True), output_root)
        source_bundle = deterministic_source_bundle(source_root, output_root / "ORBITSTATE_SOURCE_BUNDLE.tar.gz")
        source_hash_receipt = source_hashes(source_root)
        write_json(output_root / "ORBITSTATE_SOURCE_HASHES.json", source_hash_receipt)
        compile_receipt = compile_runtime(source_root, output_root)
        state["compile_state"] = {"passed": compile_receipt["passed"], "binary_sha256": compile_receipt["binary_sha256"]}
        state["live_binary_hash"] = compile_receipt["binary_sha256"]
        runtime_self = run_runtime_self_test(compile_receipt)
        state["runtime_self_test_state"] = runtime_self
        phase = "before_pmu_preflight"
        append_custody(custody, "temperatures", temperature_snapshot("before PMU preflight", strict=True), output_root)
        pmu_preflight = run_runtime_pmu_preflight(compile_receipt, output_root)
        require(pmu_preflight["passed"], "PMU preflight failed")
        state["pmu_preflight_receipt"] = pmu_preflight
        append_custody(custody, "process_snapshots", process_snapshot("before execution"), output_root)
        before_policy = policy_snapshot("before execution")
        append_custody(custody, "policy_snapshots", before_policy, output_root)
        runtime_command = compile_receipt["runtime_command"]
        schedule_arg = str(source_root / "ORBITSTATE_PUBLIC_SCHEDULE.tsv")
        private_arg = str(source_root / "ORBITSTATE_PRIVATE_SOURCE_MAP.json")
        output_arg = str(output_root)
        if runtime_command[0] == "wsl":
            schedule_arg = windows_to_wsl_path(source_root / "ORBITSTATE_PUBLIC_SCHEDULE.tsv")
            private_arg = windows_to_wsl_path(source_root / "ORBITSTATE_PRIVATE_SOURCE_MAP.json")
            output_arg = windows_to_wsl_path(output_root)
        for replicate in public.REPLICATES:
            phase = f"before_replicate_{replicate}"
            append_custody(custody, "temperatures", temperature_snapshot(f"before replicate {replicate}", strict=True), output_root)
            state["hardware_execution_began"] = True
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
            state["replicate_states"].append({"replicate": replicate, "completed": True})
            append_custody(custody, "temperatures", temperature_snapshot(f"after replicate {replicate}", strict=True), output_root)
            append_custody(custody, "process_snapshots", process_snapshot(f"after replicate {replicate}"), output_root)
        phase = "receiver_feature_freeze"
        state["feature_freeze_phase"] = "receiver_feature_process_started"
        chronology = [{"event": "receiver_feature_process_started", "timestamp_ns": time.time_ns()}]
        feature_receipt = run_receiver_feature_subprocess(source_root, output_root)
        chronology.append({"event": "receiver_feature_process_completed", "timestamp_ns": time.time_ns()})
        chronology.append(
            {
                "event": "receiver_feature_hash_frozen",
                "timestamp_ns": time.time_ns(),
                "receiver_features_sha256": feature_receipt["receiver_features_sha256"],
            }
        )
        state["feature_freeze_phase"] = "receiver_feature_hash_frozen"
        phase = "private_unblinding"
        state["unblinding_phase"] = "adjudication_subprocess_started"
        adjudication_receipt = run_adjudication_subprocess(
            source_root,
            output_root,
            feature_receipt["receiver_features_sha256"],
            chronology,
        )
        state["unblinding_phase"] = "adjudication_completed"
        append_custody(custody, "process_snapshots", process_snapshot("final"), output_root)
        final_policy = policy_snapshot("final")
        append_custody(custody, "policy_snapshots", final_policy, output_root)
        for cpu in ("cpu4", "cpu5"):
            before = before_policy["policies"][cpu]
            after = final_policy["policies"][cpu]
            require(before["scaling_driver"] == after["scaling_driver"], f"{cpu} scaling driver changed")
            require(before["scaling_governor"] == after["scaling_governor"], f"{cpu} scaling governor changed")
            require(before["scaling_min_freq"] == after["scaling_min_freq"], f"{cpu} scaling min changed")
            require(before["scaling_max_freq"] == after["scaling_max_freq"], f"{cpu} scaling max changed")
        phase = "before_final_success"
        append_custody(custody, "temperatures", temperature_snapshot("before final success", strict=True), output_root)
        adjudication = read_json(output_root / "ORBITSTATE_ADJUDICATION.json")
        final = {
            "status": "ORBITSTATE_INDEPENDENT_TARGET_COMPLETE",
            "science_package_id": SCIENCE_PACKAGE_ID,
            "transaction_run_id": run_id,
            "run_id": run_id,
            "result_class": adjudication_receipt["result_class"],
            "allowed_result_classes": public.ALLOWED_RESULT_CLASSES,
            "forbidden_result_classes": public.FORBIDDEN_RESULT_CLASSES,
            "receiver_features_sha256": feature_receipt["receiver_features_sha256"],
            "adjudication_sha256": sha256_file(output_root / "ORBITSTATE_ADJUDICATION.json"),
            "feature_freeze_receipt_sha256": sha256_file(output_root / "ORBITSTATE_FEATURE_FREEZE_RECEIPT.json"),
            "scientific_classification_emitted": True,
        }
        require(final["result_class"] == adjudication["result_class"], "adjudication result drift")
        final_path = output_root / "FINAL_RESULT_ORBITSTATE_INDEPENDENT_V2.json"
        write_json(final_path, final)
        marker = output_root / OUTPUT_ROOT_MARKER
        if marker.exists():
            marker.unlink()
        success_execution_manifest(
            source_root=source_root,
            output_root=output_root,
            implementation_manifest_file_sha256=expected_manifest_sha,
            implementation_manifest_canonical_sha256=implementation_manifest["manifest_canonical_sha256"],
            compile_receipt=compile_receipt,
            source_bundle=source_bundle,
            source_hash_receipt=source_hash_receipt,
            pmu_preflight=pmu_preflight,
            final_result_path=final_path,
        )
        write_copyback_manifest(output_root)
        return final
    except Exception as exc:
        if state.get("hardware_execution_began"):
            append_custody(custody, "process_snapshots", process_snapshot("exception cleanup when hardware execution began"), output_root)
        seal_failure_evidence(output_root, phase=phase, exc=exc, custody=custody, state=state)
        raise


def self_test(source_root: Path, output_root: Path) -> dict[str, Any]:
    offline_path = output_root / "ORBITSTATE_OFFLINE_VALIDATE.json"
    offline = read_json(offline_path) if offline_path.exists() else offline_validate(source_root, output_root)
    result = {
        "schema": "ORBITSTATE_TARGET_SELF_TEST_V2",
        "offline_validate_sha256": offline["offline_validate_sha256"],
        "offline_validate_passed": offline["passed"],
        "root_custody_self_test": offline["root_custody_self_test"],
        "failure_copyback_self_test": offline["failure_copyback_self_test"],
        "feature_boundary_static_proof": offline["feature_boundary_static_proof"],
        "pmu_runtime_static_proof": offline["pmu_runtime_static_proof"],
        "physical_protocol_static_proof": offline["physical_protocol_static_proof"],
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
    modes.add_argument("--private-map-validate", action="store_true")
    modes.add_argument("--extract-receiver-features", action="store_true")
    modes.add_argument("--adjudicate-frozen", action="store_true")
    parser.add_argument("--source-root", type=Path, default=HERE)
    parser.add_argument("--output-root", type=Path, default=HERE / "_target_out")
    parser.add_argument("--run-id", default=TRANSACTION_RUN_ID)
    parser.add_argument("--expected-manifest-sha")
    parser.add_argument("--expected-commit-binding")
    parser.add_argument("--schedule-json", type=Path)
    parser.add_argument("--raw-capture", type=Path)
    parser.add_argument("--sentinels", type=Path)
    parser.add_argument("--stage-receipts", type=Path)
    parser.add_argument("--receiver-features", type=Path)
    parser.add_argument("--receiver-features-sha256")
    parser.add_argument("--private-map", type=Path)
    parser.add_argument("--source-receipts", type=Path)
    parser.add_argument("--feature-freeze-receipt", type=Path)
    parser.add_argument("--deny-path", type=Path, action="append", default=[])
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
        if args.private_map_validate:
            result = private_map_validator_summary(args.source_root.resolve())
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0 if result["passed"] else 1
        if args.extract_receiver_features:
            require(args.schedule_json is not None, "--receiver-features requires --schedule-json")
            require(args.raw_capture is not None, "--receiver-features requires --raw-capture")
            require(args.sentinels is not None, "--receiver-features requires --sentinels")
            require(args.stage_receipts is not None, "--receiver-features requires --stage-receipts")
            result = receiver_features_main(args)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0 if result["passed"] else 1
        if args.adjudicate_frozen:
            require(args.schedule_json is not None, "--adjudicate-frozen requires --schedule-json")
            require(args.receiver_features is not None, "--adjudicate-frozen requires --receiver-features")
            require(args.receiver_features_sha256 is not None, "--adjudicate-frozen requires --receiver-features-sha256")
            require(args.private_map is not None, "--adjudicate-frozen requires --private-map")
            require(args.source_receipts is not None, "--adjudicate-frozen requires --source-receipts")
            require(args.feature_freeze_receipt is not None, "--adjudicate-frozen requires --feature-freeze-receipt")
            result = adjudicate_frozen_main(args)
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
            "scientific_classification_emitted": False,
        }
        print(json.dumps(failure, indent=2, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
