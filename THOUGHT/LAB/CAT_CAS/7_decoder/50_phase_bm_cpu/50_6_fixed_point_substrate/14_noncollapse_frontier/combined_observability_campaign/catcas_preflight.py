#!/usr/bin/env python3
"""Read-only engineering/acquisition preflight for the Phase 6 combined campaign."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

from collect_target_engineering_evidence import smoke_checks  # noqa: E402
from generate_campaign_plan import verify  # noqa: E402

ROUTES = {"v4s5": (4, 5), "v2s3": (2, 3)}
COMMIT_RE = re.compile(r"[0-9a-f]{40}")
NCPU = 6


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_object(path: Path, description: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object")
    return value


def valid_commit(value: Any) -> bool:
    return isinstance(value, str) and bool(COMMIT_RE.fullmatch(value)) and set(value) != {"0"}


def safe_bundle_path(root: Path, relative: Any) -> Path | None:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        return None
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError:
        return None
    return path


def verify_bound_files(root: Path, bundle: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    files = bundle.get("files")
    if not isinstance(files, dict) or not files:
        return ["source bundle has no file bindings"]
    expected: set[str] = set()
    for relative, binding in sorted(files.items()):
        if not isinstance(relative, str):
            errors.append("bundle contains a non-string path")
            continue
        expected.add(relative)
        path = safe_bundle_path(root, relative)
        if path is None:
            errors.append(f"invalid bundle path: {relative}")
            continue
        if not isinstance(binding, dict):
            errors.append(f"invalid binding object: {relative}")
            continue
        if not path.is_file():
            errors.append(f"missing bound file: {relative}")
            continue
        if path.stat().st_size != binding.get("size"):
            errors.append(f"size mismatch: {relative}")
        if sha256_file(path) != binding.get("sha256"):
            errors.append(f"sha256 mismatch: {relative}")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    } - {"source_bundle.json", "source_bundle.sha256"}
    if actual != expected:
        errors.append(
            f"bound file set mismatch: missing={sorted(expected - actual)} "
            f"unexpected={sorted(actual - expected)}"
        )
    return errors


def verify_manifest_directory(run: Path) -> list[str]:
    errors: list[str] = []
    manifest_path = run / "run_manifest.json"
    try:
        manifest = load_object(manifest_path, f"{run.name} run manifest")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return [f"{run.name}: invalid run manifest: {exc}"]
    files = manifest.get("files")
    if not isinstance(files, dict):
        return [f"{run.name}: invalid files table"]
    if "run_manifest.json" in files:
        errors.append(f"{run.name}: manifest includes itself")
    root = run.resolve()
    for name, binding in files.items():
        if not isinstance(name, str) or not isinstance(binding, dict):
            errors.append(f"{run.name}: invalid file binding {name!r}")
            continue
        path = (run / name).resolve()
        try:
            path.relative_to(root)
        except ValueError:
            errors.append(f"{run.name}: path escape {name}")
            continue
        if not path.is_file():
            errors.append(f"{run.name}: missing {name}")
            continue
        if path.stat().st_size != binding.get("size"):
            errors.append(f"{run.name}: size mismatch {name}")
        if sha256_file(path) != binding.get("sha256"):
            errors.append(f"{run.name}: sha mismatch {name}")
    return errors


def session_bundles_valid(root: Path) -> tuple[bool, list[str]]:
    errors: list[str] = []
    directories = sorted(path for path in root.iterdir() if path.is_dir()) if root.is_dir() else []
    if len(directories) != 12:
        errors.append(f"expected 12 compiled sessions, found {len(directories)}")
    for directory in directories:
        try:
            manifest = load_object(directory / "session_manifest.json", "session manifest")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{directory.name}: invalid session bundle: {exc}")
            continue
        if manifest.get("schema_id") != "CAT_CAS_PHASE6_COMBINED_SESSION_MANIFEST_V1":
            errors.append(f"{directory.name}: bad manifest schema")
            continue
        files = manifest.get("files")
        if not isinstance(files, dict):
            errors.append(f"{directory.name}: invalid files table")
            continue
        root_resolved = directory.resolve()
        for name, binding in files.items():
            if not isinstance(name, str) or not isinstance(binding, dict):
                errors.append(f"{directory.name}: invalid binding {name!r}")
                continue
            path = (directory / name).resolve()
            try:
                path.relative_to(root_resolved)
            except ValueError:
                errors.append(f"{directory.name}: path escape {name}")
                continue
            if not path.is_file():
                errors.append(f"{directory.name}: missing {name}")
                continue
            if path.stat().st_size != binding.get("size"):
                errors.append(f"{directory.name}: size mismatch {name}")
            if sha256_file(path) != binding.get("sha256"):
                errors.append(f"{directory.name}: sha mismatch {name}")
    return not errors, errors


def test_log_valid(path: Path, description: str) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return [f"cannot read {description}: {exc}"]
    errors: list[str] = []
    if "OK" not in text:
        errors.append(f"{description} lacks unittest OK marker")
    forbidden = ("FAILED", "Traceback (most recent call last)", "AddressSanitizer", "runtime error:")
    if any(marker in text for marker in forbidden):
        errors.append(f"{description} contains failure/sanitizer markers")
    return errors


def validation_report_valid(path: Path, bundle_root: Path | None = None) -> list[str]:
    errors: list[str] = []
    try:
        report = load_object(path, "validation report")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return [f"invalid validation report: {exc}"]
    records = report.get("records")
    if report.get("schema_id") != "CAT_CAS_PHASE6_VALIDATION_EVIDENCE_V1":
        errors.append("unexpected validation report schema")
    if report.get("sessions_expected") != 12:
        errors.append("validation report expected-session count is not 12")
    if report.get("sessions_passed") != 12 or report.get("all_pass") is not True:
        errors.append("validation report does not prove 12/12 pass")
    if report.get("hardware_touched") is not False:
        errors.append("validation report indicates hardware was touched")
    if not isinstance(records, list) or len(records) != 12 or not all(
        isinstance(record, dict) for record in records
    ):
        errors.append("validation report record count/type is not 12 objects")
        records = []
    if bundle_root is None:
        return errors

    runs_root = safe_bundle_path(bundle_root, report.get("runs_root"))
    if runs_root is None or not runs_root.is_dir():
        errors.append("validation report has invalid runs_root")
        return errors
    directories = sorted(item for item in runs_root.iterdir() if item.is_dir())
    if len(directories) != 12:
        errors.append(f"expected 12 raw validation runs, found {len(directories)}")
    actual_records: list[dict[str, Any]] = []
    hardware_touched = False
    passed = 0
    for directory in directories:
        errors.extend(verify_manifest_directory(directory))
        try:
            run = load_object(directory / "run.json", f"{directory.name} validation run")
            manifest_path = directory / "run_manifest.json"
            status_ok = run.get("status") == "VALIDATION_ONLY_HARDWARE_NOT_EXECUTED"
            hardware_ok = run.get("hardware_executed") is False
            claim_ok = run.get("physical_carrier_restoration_claimed") is False
            authorization_ok = run.get("scientific_acquisition_authorized") is False
            restoration_ok = run.get("restoration_authorized") is False
            hardware_touched = hardware_touched or run.get("hardware_executed") is True
            if all((status_ok, hardware_ok, claim_ok, authorization_ok, restoration_ok)):
                passed += 1
            else:
                errors.append(f"{directory.name}: validation-only run contract failed")
            actual_records.append({
                "session_id": directory.name,
                "runner_exit_code": 0 if status_ok else 1,
                "hardware_executed": run.get("hardware_executed"),
                "run_manifest_sha256": sha256_file(manifest_path),
            })
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{directory.name}: invalid validation run: {exc}")
    if report.get("sessions_passed") != passed:
        errors.append("validation report sessions_passed does not match raw runs")
    if report.get("hardware_touched") is not hardware_touched:
        errors.append("validation report hardware_touched does not match raw runs")
    if records != actual_records:
        errors.append("validation report records do not match raw runs")
    return errors



def evidence_valid(bundle_root: Path, bundle: dict[str, Any]) -> tuple[bool, list[str]]:
    errors: list[str] = []
    evidence = bundle.get("evidence")
    if not isinstance(evidence, dict):
        return False, ["source bundle missing evidence map"]
    for key in ("strict_test_log", "sanitizer_test_log", "python_test_log"):
        path = safe_bundle_path(bundle_root, evidence.get(key))
        if path is None or not path.is_file():
            errors.append(f"missing evidence file: {key}")
            continue
        errors.extend(test_log_valid(path, key))
    validation_path = safe_bundle_path(bundle_root, evidence.get("validation_report"))
    if validation_path is None or not validation_path.is_file():
        errors.append("missing validation report")
    else:
        errors.extend(validation_report_valid(validation_path, bundle_root))
    return not errors, errors


def mapping_six_true(value: Any) -> bool:
    return isinstance(value, dict) and set(value) == {str(i) for i in range(NCPU)} and all(
        item is True for item in value.values()
    )


def mapping_six_ints(value: Any) -> bool:
    return isinstance(value, dict) and set(value) == {str(i) for i in range(NCPU)} and all(
        isinstance(item, int) for item in value.values()
    )


def target_engineering_valid(
    bundle_root: Path, bundle: dict[str, Any], min_free_gb: float
) -> tuple[bool, list[str], dict[str, bool], dict[str, Any]]:
    errors: list[str] = []
    checks: dict[str, bool] = {}
    evidence = bundle.get("evidence")
    if not isinstance(evidence, dict):
        return False, ["missing evidence map"], {}, {}
    report_path = safe_bundle_path(bundle_root, evidence.get("target_engineering_report"))
    if report_path is None or not report_path.is_file():
        return False, ["missing target engineering report"], {}, {}
    try:
        report = load_object(report_path, "target engineering report")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return False, [f"invalid target engineering report: {exc}"], {}, {}
    checks["target_report_schema"] = (
        report.get("schema_id") == "CAT_CAS_PHASE6_TARGET_ENGINEERING_EVIDENCE_V1"
    )
    checks["target_commit_matches"] = report.get("executor_commit") == bundle.get("executor_commit")
    runner = bundle_root / "combined_pdn_runner"
    checks["target_runner_hash_matches"] = (
        runner.is_file() and report.get("executor_sha256") == sha256_file(runner)
    )
    source_transfer_path = bundle_root / "source_transfer_bundle.json"
    source_transfer_checksum_path = bundle_root / "source_transfer_bundle.sha256"
    source_transfer_digest = (
        sha256_file(source_transfer_path) if source_transfer_path.is_file() else None
    )
    declared_source_transfer_digest: str | None = None
    if source_transfer_checksum_path.is_file():
        try:
            fields = source_transfer_checksum_path.read_text(encoding="utf-8").split()
            declared_source_transfer_digest = fields[0] if fields else None
        except OSError:
            declared_source_transfer_digest = None
    checks["source_transfer_binding_present"] = (
        isinstance(report.get("source_transfer_bundle_sha256"), str)
        and report.get("source_transfer_bundle_sha256")
        == bundle.get("source_transfer_bundle_sha256")
        == source_transfer_digest
        == declared_source_transfer_digest
    )
    before_snapshot_path = safe_bundle_path(bundle_root, report.get("before_snapshot"))
    before_snapshot: dict[str, Any] = {}
    if before_snapshot_path is None or not before_snapshot_path.is_file():
        errors.append("invalid before snapshot path")
    else:
        try:
            before_snapshot = load_object(before_snapshot_path, "before snapshot")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"invalid before snapshot: {exc}")
    checks["before_snapshot_schema"] = (
        before_snapshot.get("schema_id") == "CAT_CAS_PHASE6_TARGET_SNAPSHOT_V1"
    )
    before = report.get("before_host_state")
    after = report.get("after_host_state")
    checks["before_snapshot_matches_report"] = (
        isinstance(before_snapshot.get("host_state"), dict)
        and before_snapshot.get("host_state") == before
    )
    if not isinstance(before, dict) or not isinstance(after, dict):
        errors.append("target report host states must be objects")
        return False, errors, checks, report
    checks.update({
        "running_as_root": before.get("effective_uid") == 0 and after.get("effective_uid") == 0,
        "same_target_host": bool(before.get("host")) and before.get("host") == after.get("host"),
        "baseline_no_runner_processes": before.get("runner_processes") == [],
        "cpu_count_at_least_6": isinstance(after.get("cpu_count"), int) and after["cpu_count"] >= NCPU,
        "constant_tsc": isinstance(after.get("cpu_flags"), list) and "constant_tsc" in after["cpu_flags"],
        "nonstop_tsc": isinstance(after.get("cpu_flags"), list) and "nonstop_tsc" in after["cpu_flags"],
        "k10temp_available": isinstance(after.get("k10temp_path"), str) and bool(after["k10temp_path"]),
        "msr_readable_cores_0_5": mapping_six_true(after.get("msr_readable")),
        "cpufreq_controls_readable_writable_cores_0_5": mapping_six_true(after.get("cpufreq_controls")),
        "free_space_sufficient": isinstance(after.get("free_bytes"), int) and after["free_bytes"] >= int(min_free_gb * 1024**3),
        "cpufreq_min_snapshot_complete": mapping_six_ints(before.get("cpufreq_min_khz")) and mapping_six_ints(after.get("cpufreq_min_khz")),
        "cpufreq_max_snapshot_complete": mapping_six_ints(before.get("cpufreq_max_khz")) and mapping_six_ints(after.get("cpufreq_max_khz")),
        "cleanup_cpufreq_min_restored": before.get("cpufreq_min_khz") == after.get("cpufreq_min_khz"),
        "cleanup_cpufreq_max_restored": before.get("cpufreq_max_khz") == after.get("cpufreq_max_khz"),
        "cleanup_boost_restored": before.get("boost") == after.get("boost"),
        "cleanup_no_runner_processes": after.get("runner_processes") == [],
    })
    smoke_dir = safe_bundle_path(bundle_root, report.get("smoke_run_dir"))
    late_dir = safe_bundle_path(bundle_root, report.get("late_sender_run_dir"))
    if smoke_dir is None or not smoke_dir.is_dir():
        errors.append("invalid smoke run directory")
    else:
        errors.extend(verify_manifest_directory(smoke_dir))
        try:
            smoke_run, smoke_rows, smoke_result = smoke_checks(smoke_dir)
            checks.update({f"smoke_{key}": value for key, value in smoke_result.items()})
            checks["smoke_executor_commit_matches"] = (
                smoke_run.get("executor_git_commit") == bundle.get("executor_commit")
            )
            checks["smoke_report_matches_files"] = (
                report.get("smoke_run") == smoke_run and report.get("smoke_rows") == smoke_rows
            )
        except (OSError, ValueError, json.JSONDecodeError, csv.Error) as exc:
            errors.append(f"invalid smoke evidence: {exc}")
    if late_dir is None or not late_dir.is_dir():
        errors.append("invalid late-sender run directory")
    else:
        errors.extend(verify_manifest_directory(late_dir))
        try:
            late_run = load_object(late_dir / "run.json", "late-sender run")
            checks.update({
                "late_sender_failed_closed": late_run.get("exit_status") == "FAILED",
                "late_sender_executor_commit_recorded": (
                    valid_commit(late_run.get("executor_git_commit"))
                ),
                "late_sender_executor_commit_matches": (
                    late_run.get("executor_git_commit") == bundle.get("executor_commit")
                ),
                "late_sender_correct_reason": (
                    late_run.get("failure_reason") == "SENDER_EPOCH_ALIGNMENT_FAILURE"
                ),
                "late_sender_mock_did_not_touch_hardware": (
                    late_run.get("hardware_executed") is False
                ),
                "late_sender_mock_execution_class": (
                    late_run.get("execution_class") == "MOCK_HARDWARE_TEST"
                ),
                "late_sender_scientific_acquisition_not_authorized": (
                    late_run.get("scientific_acquisition_authorized") is False
                    and late_run.get("authorization_artifact_sha256") is None
                ),
                "late_sender_host_control_state_restored": (
                    late_run.get("host_control_state_restored") is True
                ),
                "late_sender_automatic_retry_disabled": (
                    late_run.get("automatic_retry") is False
                ),
                "late_sender_physical_carrier_restoration_not_claimed": (
                    late_run.get("physical_carrier_restoration_claimed") is False
                ),
                "late_sender_report_matches_file": report.get("late_sender_run") == late_run,
            })
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"invalid late-sender evidence: {exc}")
    checks["scientific_acquisition_not_started"] = report.get("scientific_acquisition_started") is False
    checks["physical_carrier_restoration_not_claimed"] = (
        report.get("physical_carrier_restoration_claimed") is False
    )
    reported_checks = report.get("checks")
    collector_expected = {
        name: value
        for name, value in checks.items()
        if name.startswith(("smoke_", "cleanup_", "late_sender_"))
        and name not in {
            "smoke_report_matches_files",
            "smoke_executor_commit_matches",
            "late_sender_report_matches_file",
            "late_sender_executor_commit_matches",
        }
    }
    checks["collector_checks_match"] = (
        isinstance(reported_checks, dict) and reported_checks == collector_expected
    )
    checks["target_report_all_pass"] = (
        report.get("all_pass") is True and all(collector_expected.values())
    )
    for name, value in checks.items():
        if value is not True:
            errors.append(f"target engineering check failed: {name}")
    return not errors, errors, checks, report


def authorization_valid(
    path: Path | None, *, bundle_path: Path, bundle: dict[str, Any], output_root: Path
) -> tuple[bool, dict[str, Any], list[str]]:
    if path is None:
        return False, {}, ["acquisition authorization artifact not supplied"]
    try:
        authorization = load_object(path, "authorization artifact")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return False, {}, [f"invalid authorization artifact: {exc}"]
    errors: list[str] = []
    if authorization.get("schema_id") != "CAT_CAS_PHASE6_ACQUISITION_AUTHORIZATION_V1":
        errors.append("unexpected authorization schema")
    if authorization.get("acquisition_authorized") is not True:
        errors.append("acquisition_authorized is not true")
    if authorization.get("restoration_authorized") is not False:
        errors.append("restoration must remain unauthorized")
    if authorization.get("executor_commit") != bundle.get("executor_commit"):
        errors.append("authorization executor commit mismatch")
    if authorization.get("campaign_plan_sha256") != bundle.get("campaign_plan_sha256"):
        errors.append("authorization campaign plan mismatch")
    if authorization.get("source_bundle_sha256") != sha256_file(bundle_path):
        errors.append("authorization source bundle mismatch")
    if authorization.get("authorized_output_root") != str(output_root):
        errors.append("authorization output root mismatch")
    if not authorization.get("authorized_by"):
        errors.append("authorization missing authorized_by")
    return not errors, authorization, errors


def inspect(
    plan_dir: Path,
    bundle_root: Path,
    output_root: Path,
    min_free_gb: float,
    authorization_path: Path | None = None,
) -> dict[str, Any]:
    manifest = load_object(plan_dir / "campaign_manifest.json", "campaign manifest")
    plan = load_object(plan_dir / "campaign_plan.json", "campaign plan")
    bundle_path = bundle_root / "source_bundle.json"
    if not bundle_path.is_file():
        raise ValueError("preflight requires source_bundle.json")
    bundle = load_object(bundle_path, "source bundle")
    binding_path = bundle_root / "COMBINED_CAMPAIGN_BINDING.json"
    binding = load_object(binding_path, "canonical binding") if binding_path.is_file() else {}
    runner = bundle_root / "combined_pdn_runner"
    schedules = bundle_root / "compiled_sessions"
    plan_errors = verify(plan_dir)
    plan_hash = sha256_file(plan_dir / "campaign_plan.json")
    manifest_hash = sha256_file(plan_dir / "campaign_manifest.json")
    sessions = plan.get("sessions")
    route_ok = isinstance(sessions, list) and all(isinstance(item, dict) for item in sessions) and (
        {item.get("route") for item in sessions} == set(ROUTES)
    )
    bound_file_errors = verify_bound_files(bundle_root, bundle)
    sessions_ok, session_errors = session_bundles_valid(schedules)
    evidence_ok, evidence_errors = evidence_valid(bundle_root, bundle)
    target_ok, target_errors, target_checks, target_report = target_engineering_valid(
        bundle_root, bundle, min_free_gb
    )
    authorization_ok, authorization, authorization_errors = authorization_valid(
        authorization_path, bundle_path=bundle_path, bundle=bundle, output_root=output_root
    )
    executor_commit = bundle.get("executor_commit")
    engineering_checks = {
        "bundle_schema_valid": bundle.get("schema_id") == "CAT_CAS_PHASE6_EXECUTOR_SOURCE_BUNDLE_V1",
        "bundle_class_is_non_authorizing": (
            bundle.get("artifact_class") == "ENGINEERING_BUNDLE_NOT_ACQUISITION_AUTHORIZATION"
        ),
        "bundle_files_hash_bound": not bound_file_errors,
        "executor_commit_valid": valid_commit(executor_commit),
        "executor_exists_and_executable": runner.is_file() and os.access(runner, os.X_OK),
        "executor_binary_hash_valid": runner.is_file() and sha256_file(runner) == bundle.get("executor_sha256"),
        "plan_sources_agree": manifest.get("source_commit") == plan.get("source_commit"),
        "plan_manifest_valid": not plan_errors,
        "plan_hash_matches_bundle": plan_hash == bundle.get("campaign_plan_sha256"),
        "campaign_manifest_hash_matches_bundle": manifest_hash == bundle.get("campaign_manifest_sha256"),
        "canonical_plan_binding_valid": (
            isinstance(binding.get("campaign_plan"), dict)
            and plan_hash == binding["campaign_plan"].get("sha256")
            and manifest_hash == binding.get("campaign_manifest_sha256")
        ),
        "route_set_frozen": route_ok,
        "all_twelve_schedules_compiled": sessions_ok,
        "test_and_validation_evidence_valid": evidence_ok,
        "target_engineering_evidence_valid": target_ok,
        "output_path_unused": not output_root.exists(),
        "restoration_not_authorized": (
            plan.get("restoration_authorized") is False
            and manifest.get("restoration_authorized") is False
            and bundle.get("restoration_authorized") is False
        ),
    }
    engineering_ready = all(engineering_checks.values()) and all(target_checks.values())
    acquisition_ready = engineering_ready and authorization_ok
    after = target_report.get("after_host_state") if isinstance(target_report, dict) else {}
    return {
        "schema_id": "CAT_CAS_PHASE6_COMBINED_PREFLIGHT_V4",
        "host": after.get("host") if isinstance(after, dict) else None,
        "bundle_root": str(bundle_root),
        "source_bundle_sha256": sha256_file(bundle_path),
        "executor_commit": executor_commit,
        "plan_source_commit": manifest.get("source_commit"),
        "plan_dir": str(plan_dir),
        "plan_sha256": plan_hash,
        "campaign_manifest_sha256": manifest_hash,
        "output_root": str(output_root),
        "minimum_free_gb": min_free_gb,
        "plan_validation_errors": plan_errors,
        "bound_file_errors": bound_file_errors,
        "session_bundle_errors": session_errors,
        "evidence_errors": evidence_errors,
        "target_engineering_errors": target_errors,
        "authorization_errors": authorization_errors,
        "authorization": authorization,
        "engineering_checks": engineering_checks,
        "target_checks": target_checks,
        "engineering_ready": engineering_ready,
        "acquisition_authorized": authorization_ok,
        "acquisition_ready": acquisition_ready,
        "scientific_acquisition_started": False,
        "physical_carrier_restoration_claimed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-dir", type=Path, required=True)
    parser.add_argument("--bundle-root", "--repo-root", dest="bundle_root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--min-free-gb", type=float, default=20.0)
    parser.add_argument(
        "--engineering-only",
        action="store_true",
        help="return success for engineering readiness even without acquisition authorization",
    )
    args = parser.parse_args()
    try:
        report = inspect(
            args.plan_dir.resolve(),
            args.bundle_root.resolve(),
            args.output_root.resolve(),
            args.min_free_gb,
            args.authorization.resolve() if args.authorization else None,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.engineering_only:
        return 0 if report["engineering_ready"] else 2
    return 0 if report["acquisition_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
