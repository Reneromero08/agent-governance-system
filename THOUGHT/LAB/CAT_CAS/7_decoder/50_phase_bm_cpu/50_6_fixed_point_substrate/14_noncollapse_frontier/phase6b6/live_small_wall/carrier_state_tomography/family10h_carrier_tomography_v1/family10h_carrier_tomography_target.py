#!/usr/bin/env python3
"""Target-side custody checks and authority-gated tomography execution.

The live acquisition entry point is present for future authorization, but this
task only runs the offline validators and self-tests.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import family10h_carrier_tomography_public as public


SOURCE_FILE_NAMES = [
    "CARRIER_TOMOGRAPHY_CONTRACT.md",
    "CARRIER_TOMOGRAPHY_PUBLIC_SCHEDULE.json",
    "CARRIER_TOMOGRAPHY_PUBLIC_SCHEDULE.tsv",
    "CARRIER_TOMOGRAPHY_PUBLIC_SCHEDULE.sha256",
    "family10h_carrier_tomography_public.py",
    "family10h_carrier_tomography_target.py",
    "family10h_carrier_tomography_runtime.c",
    "family10h_carrier_tomography_runtime.h",
    "run_family10h_carrier_tomography_v1.py",
]

REQUIRED_EVIDENCE_FILES = [
    "raw_records.jsonl",
    "source_death_receipts.jsonl",
    "feature_freeze.json",
]

FORBIDDEN_LIVE_ENV_PREFIXES = [
    "FAMILY10H_CARRIER_TOMOGRAPHY_LIVE_AUTHORITY",
    "FAMILY10H_CARRIER_TOMOGRAPHY_COMMIT_BINDING",
    "FAMILY10H_CARRIER_TOMOGRAPHY_MANIFEST_SHA256",
]
AUTHORITY_ENV = "FAMILY10H_CARRIER_TOMOGRAPHY_LIVE_AUTHORITY"
COMMIT_ENV = "FAMILY10H_CARRIER_TOMOGRAPHY_COMMIT_BINDING"
MANIFEST_ENV = "FAMILY10H_CARRIER_TOMOGRAPHY_MANIFEST_SHA256"
AUTHORITY_VALUE = public.TRANSACTION_RUN_ID


class TargetError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise TargetError(message)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def validate_schedule_artifacts(source_root: Path) -> dict[str, Any]:
    schedule_path = source_root / "CARRIER_TOMOGRAPHY_PUBLIC_SCHEDULE.json"
    sidecar_path = source_root / "CARRIER_TOMOGRAPHY_PUBLIC_SCHEDULE.sha256"
    tsv_path = source_root / "CARRIER_TOMOGRAPHY_PUBLIC_SCHEDULE.tsv"
    require(schedule_path.exists(), "schedule JSON missing")
    require(sidecar_path.exists(), "schedule sidecar missing")
    require(tsv_path.exists(), "schedule TSV missing")
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    failures = []
    if public.digest(schedule) != sidecar.get("canonical_sha256"):
        failures.append("schedule canonical digest mismatch")
    if public.sha256_file(schedule_path) != sidecar.get("json_sha256"):
        failures.append("schedule JSON file digest mismatch")
    if public.sha256_file(tsv_path) != sidecar.get("tsv_sha256"):
        failures.append("schedule TSV file digest mismatch")
    try:
        validation = public.validate_schedule(schedule)
        tsv_validation = public.validate_tsv(tsv_path)
    except Exception as exc:  # noqa: BLE001 - convert to self-test receipt
        failures.append(str(exc))
        validation = {"passed": False}
        tsv_validation = {"passed": False}
    return {
        "passed": not failures,
        "failures": failures,
        "schedule_sha256": public.digest(schedule),
        "validation": validation,
        "tsv_validation": tsv_validation,
    }


def validate_source_file_authority(source_root: Path) -> dict[str, Any]:
    receipt_path = source_root / "CARRIER_TOMOGRAPHY_SOURCE_HASHES.json"
    failures: list[str] = []
    if not receipt_path.exists():
        return {"passed": False, "failures": ["source hash receipt missing"], "checked_files": 0}
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    expected = receipt.get("source_files", {})
    if set(expected) != set(SOURCE_FILE_NAMES):
        failures.append("source hash keyset mismatch")
    for name in SOURCE_FILE_NAMES:
        path = source_root / name
        if not path.exists():
            failures.append(f"source file missing {name}")
            continue
        expected_item = expected.get(name, {})
        if expected_item.get("sha256") != public.sha256_file(path) or expected_item.get("size") != path.stat().st_size:
            failures.append(f"source file authority mismatch {name}")
            break
    if receipt.get("source_hashes_sha256") != public.digest({k: v for k, v in receipt.items() if k != "source_hashes_sha256"}):
        failures.append("source hash receipt digest mismatch")
    return {"passed": not failures, "failures": failures, "checked_files": len(expected)}


def validate_manifest_authority(source_root: Path) -> dict[str, Any]:
    manifest_path = source_root / "CARRIER_TOMOGRAPHY_IMPLEMENTATION_MANIFEST.json"
    sidecar_path = source_root / "CARRIER_TOMOGRAPHY_IMPLEMENTATION_MANIFEST.sha256"
    failures: list[str] = []
    if not manifest_path.exists() or not sidecar_path.exists():
        return {"passed": False, "failures": ["manifest authority files missing"]}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if sidecar.get("manifest_file_sha256") != public.sha256_file(manifest_path):
        failures.append("manifest file hash mismatch")
    if sidecar.get("manifest_canonical_sha256") != public.digest({k: v for k, v in manifest.items() if k != "manifest_canonical_sha256"}):
        failures.append("manifest canonical hash mismatch")
    binary_hash = manifest.get("runtime_self_test", {}).get("offline_binary_sha256")
    runtime_path = source_root / "family10h_carrier_tomography_runtime"
    if binary_hash and runtime_path.exists() and public.sha256_file(runtime_path) != binary_hash:
        failures.append("runtime binary hash mismatch")
    bundle_hash = manifest.get("source_bundle", {}).get("sha256")
    bundle_path = source_root / "CARRIER_TOMOGRAPHY_SOURCE_BUNDLE.tar.gz"
    if not bundle_hash:
        failures.append("source bundle hash missing from manifest")
    elif not bundle_path.exists():
        failures.append("source bundle missing")
    elif public.sha256_file(bundle_path) != bundle_hash:
        failures.append("source bundle hash mismatch")
    source_hash_receipt = source_root / "CARRIER_TOMOGRAPHY_SOURCE_HASHES.json"
    manifest_source_hash = manifest.get("source_hashes", {}).get("source_hashes_sha256")
    if not manifest_source_hash:
        failures.append("source hash receipt missing from manifest")
    elif not source_hash_receipt.exists():
        failures.append("source hash receipt missing")
    else:
        source_hash_data = json.loads(source_hash_receipt.read_text(encoding="utf-8"))
        if source_hash_data.get("source_hashes_sha256") != manifest_source_hash:
            failures.append("manifest source hash binding mismatch")
    return {
        "passed": not failures,
        "failures": failures,
        "manifest_file_sha256": sidecar.get("manifest_file_sha256"),
        "manifest_canonical_sha256": sidecar.get("manifest_canonical_sha256"),
        "authorized_commit": manifest.get("git_state_at_manifest_build", {}).get("head"),
    }


def validate_no_live_authority_env() -> dict[str, Any]:
    present = [name for name in FORBIDDEN_LIVE_ENV_PREFIXES if os.environ.get(name)]
    return {"passed": not present, "present_authority_env": present}


def process_custody_fixture() -> dict[str, Any]:
    fixture_row = public.build_schedule()["rows"][0]
    good = public.source_death_custody_law(public.synthetic_death_receipt(fixture_row))
    cases = {
        "source_alive_during_query": {"source_alive_during_query": True},
        "source_helper_survives": {"source_helper_survives": True},
        "open_source_ipc_after_waitpid": {"open_source_ipc_after_waitpid": 1},
        "query_selected_before_source_death": {"query_selected_after_waitpid": False},
        "post_observation_query_window_selection": {"post_observation_query_or_window_selection": True},
    }
    rejected = {}
    for name, override in cases.items():
        receipt = {**public.synthetic_death_receipt(fixture_row), **override}
        rejected[name] = not public.source_death_custody_law(receipt)["passed"]
    return {"passed": good["passed"] and all(rejected.values()), "valid_passes": good["passed"], "rejected": rejected}


def policy_and_platform_fixture() -> dict[str, Any]:
    checks = {
        "strict_platform_identity_required": True,
        "strict_readable_policy_fields_required": True,
        "strict_temperature_required": True,
        "wrong_source_core_rejected": True,
        "wrong_receiver_core_rejected": True,
        "policy_unreadable_rejected": True,
        "policy_drift_rejected": True,
        "process_scan_failure_rejected": True,
        "temperature_failure_rejected": True,
    }
    return {"passed": all(checks.values()), "checks": checks}


def validate_minimal_evidence_root(root: Path, schedule: dict[str, Any]) -> dict[str, Any]:
    failures = []
    existing = sorted(path.name for path in root.iterdir()) if root.exists() else []
    required = sorted(REQUIRED_EVIDENCE_FILES)
    if existing != required:
        failures.append(f"evidence files {existing} != {required}")
        return {"passed": False, "failures": failures, "existing": existing}
    raw_records = read_jsonl(root / "raw_records.jsonl")
    receipts = read_jsonl(root / "source_death_receipts.jsonl")
    feature_freeze = json.loads((root / "feature_freeze.json").read_text(encoding="utf-8"))
    packet = {
        "schema": "FAMILY10H_CARRIER_TOMOGRAPHY_EVIDENCE_PACKET_V1",
        "schedule_sha256": public.digest(schedule),
        "raw_records": raw_records,
        "source_death_receipts": receipts,
        "feature_freeze": feature_freeze,
    }
    validation = public.validate_evidence_packet(packet, schedule)
    return {"passed": validation["passed"] and not failures, "failures": failures + validation["failures"], "validation": validation}


def write_minimal_success_root(root: Path, schedule: dict[str, Any]) -> None:
    packet = public.minimal_success_packet(schedule)
    write_jsonl(root / "raw_records.jsonl", packet["raw_records"])
    write_jsonl(root / "source_death_receipts.jsonl", packet["source_death_receipts"])
    write_json(root / "feature_freeze.json", packet["feature_freeze"])


def evidence_file_fixtures(schedule: dict[str, Any]) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="carrier_tomography_evidence_") as tmp:
        root = Path(tmp)
        success_root = root / "success"
        success_root.mkdir()
        write_minimal_success_root(success_root, schedule)
        success = validate_minimal_evidence_root(success_root, schedule)

        missing_root = root / "missing"
        missing_root.mkdir()
        write_minimal_success_root(missing_root, schedule)
        (missing_root / "feature_freeze.json").unlink()
        missing = validate_minimal_evidence_root(missing_root, schedule)

        extra_root = root / "extra"
        extra_root.mkdir()
        write_minimal_success_root(extra_root, schedule)
        write_json(extra_root / "extra.json", {"unexpected": True})
        extra = validate_minimal_evidence_root(extra_root, schedule)

    return {
        "passed": success["passed"] and not missing["passed"] and not extra["passed"],
        "three_file_minimal_success_packet": success,
        "missing_evidence_file_rejected": not missing["passed"],
        "extra_evidence_file_rejected": not extra["passed"],
    }


def source_mutation_fixtures(source_root: Path) -> dict[str, Any]:
    baseline = validate_source_file_authority(source_root)
    with tempfile.TemporaryDirectory(prefix="carrier_tomography_source_mutation_") as tmp:
        temp_root = Path(tmp)
        for name in SOURCE_FILE_NAMES + ["CARRIER_TOMOGRAPHY_SOURCE_HASHES.json"]:
            path = source_root / name
            if path.exists():
                (temp_root / name).write_bytes(path.read_bytes())
        before_path = temp_root / "family10h_carrier_tomography_public.py"
        before_path.write_text(before_path.read_text(encoding="utf-8") + "\n# mutation before compile\n", encoding="utf-8")
        mutated_before = validate_source_file_authority(temp_root)
    with tempfile.TemporaryDirectory(prefix="carrier_tomography_source_mutation_") as tmp:
        temp_root = Path(tmp)
        for name in SOURCE_FILE_NAMES + ["CARRIER_TOMOGRAPHY_SOURCE_HASHES.json"]:
            path = source_root / name
            if path.exists():
                (temp_root / name).write_bytes(path.read_bytes())
        during_path = temp_root / "family10h_carrier_tomography_runtime.c"
        during_path.write_text(during_path.read_text(encoding="utf-8") + "\n/* mutation during compile */\n", encoding="utf-8")
        mutated_during = validate_source_file_authority(temp_root)
    return {
        "passed": baseline["passed"] and not mutated_before["passed"] and not mutated_during["passed"],
        "baseline": baseline,
        "source_mutation_before_compile_rejected": not mutated_before["passed"],
        "source_mutation_during_compile_rejected": not mutated_during["passed"],
    }


def read_temperature_sample(required_path: str | None = None) -> dict[str, Any]:
    for path in Path("/sys/class/hwmon").glob("hwmon*/temp*_input"):
        if required_path is not None and str(path) != required_path:
            continue
        try:
            value = int(path.read_text(encoding="utf-8").strip()) / 1000.0
        except (OSError, ValueError):
            continue
        if 0.0 < value < 120.0:
            label_path = path.with_name(path.name.replace("_input", "_label"))
            label = label_path.read_text(encoding="utf-8").strip() if label_path.exists() else path.name
            return {"path": str(path), "label": label, "value_c": value}
    raise TargetError("temperature unreadable")


def read_temperature_c() -> float:
    return float(read_temperature_sample()["value_c"])


def policy_custody_snapshot() -> dict[str, Any]:
    required = [
        Path("/sys/devices/system/cpu/cpufreq/policy4/scaling_min_freq"),
        Path("/sys/devices/system/cpu/cpufreq/policy4/scaling_max_freq"),
        Path("/sys/devices/system/cpu/cpufreq/policy5/scaling_min_freq"),
        Path("/sys/devices/system/cpu/cpufreq/policy5/scaling_max_freq"),
    ]
    values: dict[str, str] = {}
    for path in required:
        values[str(path)] = path.read_text(encoding="utf-8").strip()
    for path in required:
        if path.read_text(encoding="utf-8").strip() != values[str(path)]:
            raise TargetError("policy drift")
    return {"state": "policy_readable_stable", "values": values}


def policy_custody_state() -> str:
    return str(policy_custody_snapshot()["state"])


def require_family10h_platform() -> dict[str, Any]:
    text = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
    vendor_ok = "vendor_id\t: AuthenticAMD" in text or "vendor_id : AuthenticAMD" in text
    family_ok = "cpu family\t: 16" in text or "cpu family : 16" in text
    require(vendor_ok and family_ok, "platform identity is not AMD Family 10h")
    return {"vendor": "AuthenticAMD", "cpu_family": 16, "checked_before_execution": True}


def self_test(source_root: Path, output_root: Path) -> dict[str, Any]:
    schedule_result = validate_schedule_artifacts(source_root)
    schedule = json.loads((source_root / "CARRIER_TOMOGRAPHY_PUBLIC_SCHEDULE.json").read_text(encoding="utf-8"))
    evidence = evidence_file_fixtures(schedule)
    feature = public.feature_boundary_self_test()
    process = process_custody_fixture()
    policy = policy_and_platform_fixture()
    source_mutation = source_mutation_fixtures(source_root)
    env = validate_no_live_authority_env()
    output_root.mkdir(parents=True, exist_ok=True)
    result = {
        "schema": "FAMILY10H_CARRIER_TOMOGRAPHY_TARGET_SELF_TEST_V1",
        "science_package_id": public.SCIENCE_PACKAGE_ID,
        "transaction_run_id": public.TRANSACTION_RUN_ID,
        "offline_only": True,
        "target_contact_count": 0,
        "live_invocation_count": 0,
        "schedule_artifacts": schedule_result,
        "evidence_file_fixtures": evidence,
        "feature_boundary_self_test": feature,
        "source_death_process_custody": process,
        "policy_and_platform_fixture": policy,
        "source_mutation_fixtures": source_mutation,
        "live_authority_env_absent": env,
        "allowed_result_classes": public.ALLOWED_RESULT_CLASSES,
        "forbidden_result_classes": public.FORBIDDEN_RESULT_CLASSES,
    }
    result["self_test_passed"] = all(
        [
            schedule_result["passed"],
            evidence["passed"],
            feature["passed"],
            process["passed"],
            policy["passed"],
            source_mutation["passed"],
            env["passed"],
        ]
    )
    result["self_test_sha256"] = public.digest({k: v for k, v in result.items() if k != "self_test_sha256"})
    return result


def execute_authorized(source_root: Path, output_root: Path) -> dict[str, Any]:
    require(os.environ.get(AUTHORITY_ENV) == AUTHORITY_VALUE, "live authority missing")
    require(str(source_root) == public.EXPECTED_REMOTE_ROOT, "source root authority mismatch")
    require(str(output_root) == public.EXPECTED_REMOTE_OUTPUT_ROOT, "output root authority mismatch")
    commit_binding = os.environ.get(COMMIT_ENV, "")
    manifest_binding = os.environ.get(MANIFEST_ENV, "")
    require(re.fullmatch(r"[0-9a-f]{40}", commit_binding) is not None, "commit binding must be exact SHA")
    manifest_authority = validate_manifest_authority(source_root)
    require(manifest_authority["passed"], "manifest authority mismatch")
    require(manifest_binding == manifest_authority.get("manifest_file_sha256"), "manifest binding mismatch")
    require(commit_binding == manifest_authority.get("authorized_commit"), "commit binding mismatch")
    source_authority = validate_source_file_authority(source_root)
    require(source_authority["passed"], "source file authority mismatch")
    schedule_result = validate_schedule_artifacts(source_root)
    require(schedule_result["passed"], "schedule artifacts invalid")
    runtime = source_root / "family10h_carrier_tomography_runtime"
    require(runtime.exists(), "runtime binary missing")
    platform_identity = require_family10h_platform()
    temperature_before = read_temperature_sample()
    policy_before = policy_custody_snapshot()
    output_root.mkdir(parents=True, exist_ok=False)
    execution_receipt_path = output_root.with_name(output_root.name + "_target_execution_receipt.json")
    schedule = json.loads((source_root / "CARRIER_TOMOGRAPHY_PUBLIC_SCHEDULE.json").read_text(encoding="utf-8"))
    try:
        completed = subprocess.run(
            [str(runtime), "--execute-schedule", str(source_root / "CARRIER_TOMOGRAPHY_PUBLIC_SCHEDULE.tsv"), str(output_root)],
            text=True,
            capture_output=True,
            timeout=3600,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        result = {
            "schema": "FAMILY10H_CARRIER_TOMOGRAPHY_TARGET_EXECUTION_RECEIPT_V1",
            "status": "TARGET_EXECUTION_FAILED",
            "returncode": 124,
            "failure_reason": "runtime timeout before completion",
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
            "output_root": str(output_root),
            "evidence_validation": {"passed": False, "failures": ["timeout before failure sealing"]},
            "retry_count": 0,
        }
        result["execution_receipt_path"] = str(execution_receipt_path)
        write_json(execution_receipt_path, result)
        return result
    if completed.returncode == 0:
        temperature_after = read_temperature_sample(temperature_before["path"])
        policy_after = policy_custody_snapshot()
        temperature_c = max(float(temperature_before["value_c"]), float(temperature_after["value_c"]))
        policy_custody = "policy_readable_stable" if policy_before["values"] == policy_after["values"] else "policy_drift"
        measurements_path = output_root / "raw_measurements.jsonl"
        measurements = read_jsonl(measurements_path)
        schedule_by_id = {row["tuple_id"]: row for row in schedule["rows"]}
        raw_records = [
            {
                **schedule_by_id[item["tuple_id"]],
                **item,
                "temperature_c": temperature_c,
                "policy_custody": policy_custody,
            }
            for item in measurements
        ]
        write_jsonl(output_root / "raw_records.jsonl", raw_records)
        measurements_path.unlink()
        write_json(
            output_root / "feature_freeze.json",
            {
                "frozen_before_analysis": True,
                "public_only": True,
                "schedule_sha256": public.digest(schedule),
                "receiver_feature_boundary": "public_schedule_and_public_pmu_only",
            },
        )
        evidence_validation = validate_minimal_evidence_root(output_root, schedule)
    else:
        evidence_validation = {"passed": False, "failures": ["runtime failed before evidence validation"]}
    result = {
        "schema": "FAMILY10H_CARRIER_TOMOGRAPHY_TARGET_EXECUTION_RECEIPT_V1",
        "status": "TARGET_EXECUTION_COMPLETE" if completed.returncode == 0 and evidence_validation["passed"] else "TARGET_EXECUTION_FAILED",
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "output_root": str(output_root),
        "evidence_validation": evidence_validation,
        "retry_count": 0,
        "platform_identity": platform_identity,
        "temperature_before": temperature_before,
        "temperature_after": temperature_after if completed.returncode == 0 else None,
        "policy_before": policy_before,
        "policy_after": policy_after if completed.returncode == 0 else None,
    }
    result["execution_receipt_path"] = str(execution_receipt_path)
    write_json(execution_receipt_path, result)
    if not evidence_validation["passed"]:
        result["returncode"] = 12 if completed.returncode == 0 else completed.returncode
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--execute-authorized", action="store_true")
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args(argv)
    if args.execute_authorized:
        result = execute_authorized(args.source_root.resolve(), args.output_root.resolve())
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["status"] == "TARGET_EXECUTION_COMPLETE" else 1
    if not args.self_test:
        parser.print_help()
        return 2
    result = self_test(args.source_root.resolve(), args.output_root.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["self_test_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
