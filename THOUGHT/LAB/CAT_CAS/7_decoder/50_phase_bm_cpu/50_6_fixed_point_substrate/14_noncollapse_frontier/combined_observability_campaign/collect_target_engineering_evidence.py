#!/usr/bin/env python3
"""Collect raw CAT_CAS host state and derive the engineering-smoke evidence report."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import socket
import sys
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

from verify_run_manifests import verify as verify_run_manifests  # noqa: E402

NCPU = 6
MAX_EPOCH_SKEW_SECONDS = 0.005
COMMIT_RE = re.compile(r"[0-9a-f]{40}")
SHA256_RE = re.compile(r"[0-9a-f]{64}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_int(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def read_cpu_flags() -> list[str]:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("flags"):
                return sorted(set(line.split(":", 1)[1].split()))
    except OSError:
        pass
    return []


def k10temp_path() -> str | None:
    for name_path in sorted(Path("/sys/class/hwmon").glob("hwmon*/name")):
        try:
            if name_path.read_text(encoding="utf-8").strip() == "k10temp":
                candidate = name_path.parent / "temp1_input"
                if candidate.is_file():
                    return str(candidate)
        except OSError:
            continue
    return None


def is_runner_process(comm: str, argv0: str) -> bool:
    """Recognize the runner despite Linux's 15-byte ``comm`` truncation."""
    executable = Path(argv0).name if argv0 else ""
    return executable == "combined_pdn_runner" or comm == "combined_pdn_runner"[:15]


def runner_processes() -> list[dict[str, Any]]:
    processes: list[dict[str, Any]] = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            comm = (proc / "comm").read_text(encoding="utf-8").strip()
            raw = (proc / "cmdline").read_bytes().split(b"\0", 1)[0]
            argv0 = raw.decode(errors="replace")
            if is_runner_process(comm, argv0):
                processes.append({"pid": int(proc.name), "comm": comm, "argv0": argv0})
        except (OSError, ValueError):
            continue
    return processes



def host_snapshot(space_path: Path) -> dict[str, Any]:
    cpufreq_min: dict[str, int | None] = {}
    cpufreq_max: dict[str, int | None] = {}
    controls: dict[str, bool] = {}
    msr: dict[str, bool] = {}
    for core in range(NCPU):
        root = Path(f"/sys/devices/system/cpu/cpu{core}/cpufreq")
        min_path = root / "scaling_min_freq"
        max_path = root / "scaling_max_freq"
        cpufreq_min[str(core)] = read_int(min_path)
        cpufreq_max[str(core)] = read_int(max_path)
        controls[str(core)] = os.access(min_path, os.R_OK | os.W_OK) and os.access(
            max_path, os.R_OK | os.W_OK
        )
        msr[str(core)] = os.access(f"/dev/cpu/{core}/msr", os.R_OK)
    stat = os.statvfs(space_path)
    return {
        "host": socket.gethostname(),
        "effective_uid": os.geteuid(),
        "cpu_count": os.cpu_count() or 0,
        "cpu_flags": read_cpu_flags(),
        "k10temp_path": k10temp_path(),
        "msr_readable": msr,
        "cpufreq_controls": controls,
        "cpufreq_min_khz": cpufreq_min,
        "cpufreq_max_khz": cpufreq_max,
        "boost": read_int(Path("/sys/devices/system/cpu/cpufreq/boost")),
        "free_bytes": stat.f_bavail * stat.f_frsize,
        "runner_processes": runner_processes(),
    }


def load_object(path: Path, description: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object")
    return value


def int_field(row: dict[str, str], name: str) -> int:
    try:
        return int(row[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid smoke field {name}") from exc


def smoke_checks(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, str]], dict[str, bool]]:
    run = load_object(run_dir / "run.json", "smoke run")
    with (run_dir / "window_results.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    checks: dict[str, bool] = {
        "run_complete": run.get("exit_status") == "COMPLETE",
        "engineering_smoke_identity": run.get("session_id") == "ENGINEERING_SMOKE_TEST",
        "executor_commit_recorded": (
            isinstance(run.get("executor_git_commit"), str)
            and bool(COMMIT_RE.fullmatch(run["executor_git_commit"]))
            and set(run["executor_git_commit"]) != {"0"}
        ),
        "engineering_smoke_execution_class": (
            run.get("execution_class") == "ENGINEERING_SMOKE_NOT_SCIENTIFIC_ACQUISITION"
        ),
        "hardware_executed": run.get("hardware_executed") is True,
        "scientific_acquisition_not_authorized": (
            run.get("scientific_acquisition_authorized") is False
            and run.get("authorization_artifact_sha256") is None
        ),
        "host_control_state_restored": run.get("host_control_state_restored") is True,
        "physical_carrier_restoration_not_claimed": (
            run.get("physical_carrier_restoration_claimed") is False
        ),
        "automatic_retry_disabled": run.get("automatic_retry") is False,
        "restoration_not_authorized": run.get("restoration_authorized") is False,
        "three_windows": len(rows) == 3,
    }
    driven = [row for row in rows if row.get("drive_on") == "1"]
    sender_off = [row for row in rows if row.get("sender_off_required") == "1"]
    checks["two_driven_windows"] = len(driven) == 2
    checks["one_sender_off_window"] = len(sender_off) == 1
    tsc_hz = run.get("tsc_calibration_hz")
    try:
        skew_limit = int(float(tsc_hz) * MAX_EPOCH_SKEW_SECONDS)
        capture_overrun_limit = int(float(tsc_hz) * 0.020)
    except (TypeError, ValueError, OverflowError):
        skew_limit = -1
        capture_overrun_limit = -1
    driven_ok = skew_limit >= 0 and capture_overrun_limit >= 0
    for row in driven:
        try:
            origin = int_field(row, "slot_start_tsc")
            deadline = int_field(row, "capture_deadline_tsc")
            ready = int_field(row, "sender_ready_tsc")
            sender_epoch = int_field(row, "sender_epoch_tsc")
            first_drive = int_field(row, "first_drive_tsc")
            receiver_epoch = int_field(row, "receiver_epoch_tsc")
            first_sample = int_field(row, "first_sample_tsc")
            last_sample = int_field(row, "last_sample_tsc")
            driven_ok = driven_ok and all(
                (
                    ready < origin,
                    origin <= sender_epoch <= origin + skew_limit,
                    origin <= receiver_epoch <= origin + skew_limit,
                    origin <= first_drive <= deadline,
                    first_sample >= receiver_epoch,
                    first_sample <= last_sample <= deadline + capture_overrun_limit,
                    row.get("sender_started") == "1",
                    row.get("sender_stopped") == "1",
                    row.get("sender_alive_at_capture") == "1",
                    row.get("window_status") == "OK",
                )
            )
        except ValueError:
            driven_ok = False
    checks["driven_timing_and_lifecycle"] = driven_ok and len(driven) == 2
    off_ok = False
    if len(sender_off) == 1:
        row = sender_off[0]
        try:
            origin = int_field(row, "slot_start_tsc")
            deadline = int_field(row, "capture_deadline_tsc")
            receiver_epoch = int_field(row, "receiver_epoch_tsc")
            first_sample = int_field(row, "first_sample_tsc")
            last_sample = int_field(row, "last_sample_tsc")
            off_ok = all(
                (
                    int_field(row, "sender_ready_tsc") == 0,
                    int_field(row, "sender_epoch_tsc") == 0,
                    int_field(row, "first_drive_tsc") == 0,
                    origin <= receiver_epoch <= origin + skew_limit,
                    first_sample >= receiver_epoch,
                    first_sample <= last_sample <= deadline + capture_overrun_limit,
                    row.get("sender_started") == "0",
                    row.get("sender_stopped") == "1",
                    row.get("sender_alive_at_capture") == "0",
                    row.get("computed_I") == "null",
                    row.get("computed_Q") == "null",
                    row.get("window_status") == "OK",
                )
            )
        except ValueError:
            off_ok = False
    checks["sender_off_is_true_off"] = off_ok
    return run, rows, checks



def relative_file(root: Path, path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"evidence path escapes root: {path}") from exc


def validation_report_command(args: argparse.Namespace) -> int:
    runs_root = args.runs_root.resolve()
    manifest_errors = verify_run_manifests(runs_root)
    directories = sorted(path for path in runs_root.iterdir() if path.is_dir())
    records: list[dict[str, Any]] = []
    passed = 0
    errors = list(manifest_errors)
    if len(directories) != 12:
        errors.append(f"expected 12 validation runs, found {len(directories)}")
    for directory in directories:
        try:
            run = load_object(directory / "run.json", f"{directory.name} validation run")
            manifest_path = directory / "run_manifest.json"
            record = {
                "session_id": directory.name,
                "runner_exit_code": 0 if run.get("status") == "VALIDATION_ONLY_HARDWARE_NOT_EXECUTED" else 1,
                "hardware_executed": run.get("hardware_executed"),
                "run_manifest_sha256": sha256_file(manifest_path),
            }
            records.append(record)
            if record["runner_exit_code"] != 0 or record["hardware_executed"] is not False:
                errors.append(f"{directory.name}: validation-only contract failed")
            else:
                passed += 1
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{directory.name}: invalid validation run: {exc}")
    report = {
        "schema_id": "CAT_CAS_PHASE6_VALIDATION_EVIDENCE_V1",
        "runs_root": relative_file(args.evidence_root.resolve(), runs_root),
        "sessions_expected": 12,
        "sessions_passed": passed,
        "all_pass": not errors and len(records) == 12,
        "hardware_touched": any(record.get("hardware_executed") is True for record in records),
        "records": records,
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["all_pass"] else 2


def snapshot_command(args: argparse.Namespace) -> int:
    snapshot = {
        "schema_id": "CAT_CAS_PHASE6_TARGET_SNAPSHOT_V1",
        "host_state": host_snapshot(args.space_path.resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(snapshot, indent=2, sort_keys=True))
    return 0


def finalize_command(args: argparse.Namespace) -> int:
    if not COMMIT_RE.fullmatch(args.executor_commit) or set(args.executor_commit) == {"0"}:
        raise ValueError("invalid executor commit")
    if not SHA256_RE.fullmatch(args.source_transfer_bundle_sha256):
        raise ValueError("invalid source-transfer bundle SHA-256")
    root = args.evidence_root.resolve()
    before = load_object(args.before.resolve(), "before snapshot")
    if before.get("schema_id") != "CAT_CAS_PHASE6_TARGET_SNAPSHOT_V1":
        raise ValueError("unexpected before-snapshot schema")
    before_state = before.get("host_state")
    if not isinstance(before_state, dict):
        raise ValueError("before snapshot missing host_state")
    after_state = host_snapshot(args.space_path.resolve())
    smoke_run, smoke_rows, smoke = smoke_checks(args.smoke_run.resolve())
    late_run = load_object(args.late_sender_run.resolve() / "run.json", "late-sender run")
    cleanup = {
        "cpufreq_min_restored": before_state.get("cpufreq_min_khz") == after_state.get("cpufreq_min_khz"),
        "cpufreq_max_restored": before_state.get("cpufreq_max_khz") == after_state.get("cpufreq_max_khz"),
        "boost_restored": before_state.get("boost") == after_state.get("boost"),
        "no_runner_processes": after_state.get("runner_processes") == [],
    }
    late_sender = {
        "failed_closed": late_run.get("exit_status") == "FAILED",
        "executor_commit_recorded": late_run.get("executor_git_commit") == args.executor_commit,
        "correct_reason": late_run.get("failure_reason") == "SENDER_EPOCH_ALIGNMENT_FAILURE",
        "mock_did_not_touch_hardware": late_run.get("hardware_executed") is False,
        "mock_execution_class": late_run.get("execution_class") == "MOCK_HARDWARE_TEST",
        "scientific_acquisition_not_authorized": (
            late_run.get("scientific_acquisition_authorized") is False
            and late_run.get("authorization_artifact_sha256") is None
        ),
        "host_control_state_restored": late_run.get("host_control_state_restored") is True,
        "automatic_retry_disabled": late_run.get("automatic_retry") is False,
        "physical_carrier_restoration_not_claimed": (
            late_run.get("physical_carrier_restoration_claimed") is False
        ),
    }
    checks = {
        **{f"smoke_{key}": value for key, value in smoke.items()},
        **{f"cleanup_{key}": value for key, value in cleanup.items()},
        **{f"late_sender_{key}": value for key, value in late_sender.items()},
    }
    report = {
        "schema_id": "CAT_CAS_PHASE6_TARGET_ENGINEERING_EVIDENCE_V1",
        "executor_commit": args.executor_commit,
        "source_transfer_bundle_sha256": args.source_transfer_bundle_sha256,
        "executor_sha256": sha256_file(args.runner.resolve()),
        "before_snapshot": relative_file(root, args.before),
        "smoke_run_dir": relative_file(root, args.smoke_run),
        "late_sender_run_dir": relative_file(root, args.late_sender_run),
        "before_host_state": before_state,
        "after_host_state": after_state,
        "smoke_run": smoke_run,
        "smoke_rows": smoke_rows,
        "late_sender_run": late_run,
        "checks": checks,
        "all_pass": all(checks.values()),
        "scientific_acquisition_started": False,
        "physical_carrier_restoration_claimed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["all_pass"] else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    validation = subparsers.add_parser("validation-report")
    validation.add_argument("--evidence-root", type=Path, required=True)
    validation.add_argument("--runs-root", type=Path, required=True)
    validation.add_argument("--output", type=Path, required=True)
    validation.set_defaults(handler=validation_report_command)

    snapshot = subparsers.add_parser("snapshot")
    snapshot.add_argument("--output", type=Path, required=True)
    snapshot.add_argument("--space-path", type=Path, default=Path("."))
    snapshot.set_defaults(handler=snapshot_command)

    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--evidence-root", type=Path, required=True)
    finalize.add_argument("--before", type=Path, required=True)
    finalize.add_argument("--smoke-run", type=Path, required=True)
    finalize.add_argument("--late-sender-run", type=Path, required=True)
    finalize.add_argument("--runner", type=Path, required=True)
    finalize.add_argument("--executor-commit", required=True)
    finalize.add_argument("--source-transfer-bundle-sha256", required=True)
    finalize.add_argument("--output", type=Path, required=True)
    finalize.add_argument("--space-path", type=Path, default=Path("."))
    finalize.set_defaults(handler=finalize_command)
    args = parser.parse_args()
    try:
        return args.handler(args)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
