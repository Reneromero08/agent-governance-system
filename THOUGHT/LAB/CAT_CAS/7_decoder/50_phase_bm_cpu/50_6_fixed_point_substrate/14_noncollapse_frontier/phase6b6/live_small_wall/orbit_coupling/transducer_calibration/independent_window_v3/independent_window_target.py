#!/usr/bin/env python3
"""Target-side wrapper for Independent-Window Transducer V3.

Offline modes compile and validate the frozen package without opening network
connections or touching PMU state. ``--execute-live`` is the lab-side entry
point used only after the controller has completed the three authorization
bindings and transported the bound source package.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any

import independent_window_public as public


HERE = Path(__file__).resolve().parent
TEMPERATURE_VETO_C = 68.0
PHYSICALLY_PLAUSIBLE_TEMP_C = (0.0, 125.0)
POLICY_IDS = (4, 5)
PMU_EVENT_FORMAT = "config:0-7,32-35"
PMU_UMASK_FORMAT = "config:8-15"
CAP_PERFMON_BIT = 38
BINARY_CUSTODY_MODE = "target_compile_bound_by_source_and_compiler_contract"
PROCESS_SCAN_COMMAND = ("ps", "-eo", "pid=,comm=,args=")
FORBIDDEN_PROCESS_MARKERS = (
    "independent_window_runtime",
    "confirmation_v2_runtime",
    "balanced_transducer_runtime",
    "orbit_query_runtime",
    "f10_pmc_first_light_worker",
    "gate_a_worker_live",
    "combined_pdn_runner",
    "run_combined_campaign",
)
STRICT_C_FLAGS = (
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
)
SOURCE_FILES = (
    "INDEPENDENT_WINDOW_CONTRACT_V3.md",
    "RETRY1_MEASUREMENT_TOPOLOGY_AUDIT.md",
    "LIVE_EXECUTION_COMPLETION_AUDIT.md",
    "INDEPENDENT_WINDOW_PUBLIC_SCHEDULE.json",
    "INDEPENDENT_WINDOW_PUBLIC_SCHEDULE.sha256",
    "INDEPENDENT_WINDOW_PUBLIC_SCHEDULE.tsv",
    "INDEPENDENT_WINDOW_V3_SOL_AUDIT.json",
    "independent_window_public.py",
    "independent_window_runtime.c",
    "independent_window_runtime.h",
    "independent_window_target.py",
    "run_independent_window_v3.py",
)
CONTROL_FILES = ("INDEPENDENT_WINDOW_IMPLEMENTATION_MANIFEST.json",)
SUCCESS_REQUIRED_FILES = (
    "INDEPENDENT_WINDOW_V3_MANIFEST.json",
    "INDEPENDENT_WINDOW_PUBLIC_SCHEDULE.json",
    "INDEPENDENT_WINDOW_PUBLIC_SCHEDULE.tsv",
    "INDEPENDENT_WINDOW_PUBLIC_SCHEDULE.sha256",
    "INDEPENDENT_WINDOW_SOURCE_BUNDLE.tar.gz",
    "INDEPENDENT_WINDOW_SOURCE_HASHES.json",
    "INDEPENDENT_WINDOW_RUNTIME_STDOUT_REPLICATE_0.txt",
    "INDEPENDENT_WINDOW_RUNTIME_STDERR_REPLICATE_0.txt",
    "INDEPENDENT_WINDOW_RUNTIME_STDOUT_REPLICATE_1.txt",
    "INDEPENDENT_WINDOW_RUNTIME_STDERR_REPLICATE_1.txt",
    "RAW_INDEPENDENT_WINDOW_CAPTURE.jsonl",
    "INDEPENDENT_WINDOW_SENTINELS.jsonl",
    "INDEPENDENT_WINDOW_STAGE_RECEIPTS.jsonl",
    "INDEPENDENT_WINDOW_SOURCE_RECEIPTS.jsonl",
    "INDEPENDENT_WINDOW_FEATURES.json",
    "INDEPENDENT_WINDOW_ADJUDICATION.json",
    "FINAL_RESULT_INDEPENDENT_WINDOW_V3.json",
    "LIVE_CUSTODY_LOG.json",
    "COPYBACK_MANIFEST.json",
)
FAILURE_REQUIRED_FILES = (
    "TARGET_FAILURE_INDEPENDENT_WINDOW_V3.json",
    "INDEPENDENT_WINDOW_V3_FAILURE_MANIFEST.json",
    "FINAL_RESULT_INDEPENDENT_WINDOW_V3.json",
    "LIVE_CUSTODY_LOG.json",
    "COPYBACK_MANIFEST.json",
)


class TargetError(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise TargetError(message)


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


def digest_file_or_none(path: Path) -> str | None:
    return sha256_file(path) if path.is_file() else None


def manifest_digest(manifest: dict[str, Any]) -> str:
    return public.digest({k: v for k, v in manifest.items() if k != "implementation_manifest_sha256"})


def execution_manifest_digest(manifest: dict[str, Any]) -> str:
    return public.digest({k: v for k, v in manifest.items() if k != "manifest_sha256"})


def failure_manifest_digest(manifest: dict[str, Any]) -> str:
    return public.digest({k: v for k, v in manifest.items() if k != "manifest_sha256"})


def _now_ns() -> int:
    return time.monotonic_ns()


def windows_to_wsl_path(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    tail = resolved.as_posix().split(":", 1)[1].lstrip("/")
    return f"/mnt/{drive}/{tail}"


def source_hashes(source_root: Path) -> dict[str, str]:
    return {
        name: sha256_file(source_root / name)
        for name in SOURCE_FILES
        if (source_root / name).is_file()
    }


def all_transferred_hashes(source_root: Path) -> dict[str, str]:
    hashes = source_hashes(source_root)
    for name in CONTROL_FILES:
        path = source_root / name
        if path.is_file():
            hashes[name] = sha256_file(path)
    return hashes


def deterministic_source_bundle(source_root: Path, output_path: Path) -> tuple[str, dict[str, str]]:
    hashes = source_hashes(source_root)
    require(set(SOURCE_FILES).issubset(hashes), "source bundle file set incomplete")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w") as archive:
                for name in sorted(SOURCE_FILES):
                    data = (source_root / name).read_bytes()
                    info = tarfile.TarInfo(name)
                    info.size = len(data)
                    info.mtime = 0
                    info.mode = 0o644
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    archive.addfile(info, io.BytesIO(data))
    write_json(output_path.parent / "INDEPENDENT_WINDOW_SOURCE_HASHES.json", {"schema_id": "CAT_CAS_INDEPENDENT_WINDOW_SOURCE_HASHES_V3", "files": hashes})
    return sha256_file(output_path), hashes


def validate_schedule_artifacts(source_root: Path, output_root: Path | None = None) -> dict[str, Any]:
    schedule_path = source_root / "INDEPENDENT_WINDOW_PUBLIC_SCHEDULE.json"
    tsv_path = source_root / "INDEPENDENT_WINDOW_PUBLIC_SCHEDULE.tsv"
    sha_path = source_root / "INDEPENDENT_WINDOW_PUBLIC_SCHEDULE.sha256"
    require(schedule_path.is_file(), "schedule JSON missing")
    require(tsv_path.is_file(), "schedule TSV missing")
    require(sha_path.is_file(), "schedule SHA missing")
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    public.validate_schedule(schedule)
    require(sha256_file(schedule_path) == sha_path.read_text(encoding="utf-8").strip(), "schedule SHA mismatch")
    require(tsv_path.read_text(encoding="utf-8") == public.schedule_tsv(schedule), "schedule TSV mismatch")
    if output_root is not None:
        shutil.copy2(schedule_path, output_root / "INDEPENDENT_WINDOW_PUBLIC_SCHEDULE.json")
        shutil.copy2(tsv_path, output_root / "INDEPENDENT_WINDOW_PUBLIC_SCHEDULE.tsv")
        shutil.copy2(sha_path, output_root / "INDEPENDENT_WINDOW_PUBLIC_SCHEDULE.sha256")
    return {
        "schedule": schedule,
        "schedule_json_sha256": sha256_file(schedule_path),
        "schedule_tsv_sha256": sha256_file(tsv_path),
        "schedule_semantic_sha256": schedule["schedule_semantic_sha256"],
        "total_mapping_leg_records": len(schedule["trials"]),
        "total_component_measurement_windows": schedule["total_component_measurement_windows"],
    }


def compile_runtime(source_root: Path, binary_path: Path, *, prefer_wsl: bool = False) -> dict[str, Any]:
    binary_path.parent.mkdir(parents=True, exist_ok=True)
    compiler = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    use_wsl = prefer_wsl or compiler is None
    if use_wsl:
        wsl_check = subprocess.run(
            ["wsl", "--", "gcc", "--version"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
            check=False,
        )
        if wsl_check.returncode != 0:
            return {"available": False, "passed": False, "compiler": None, "binary_sha256": None, "stderr": wsl_check.stderr.strip()}
        command = [
            "wsl",
            "--",
            "gcc",
            *STRICT_C_FLAGS,
            "-I",
            windows_to_wsl_path(source_root),
            windows_to_wsl_path(source_root / "independent_window_runtime.c"),
            "-o",
            windows_to_wsl_path(binary_path),
        ]
        runtime_command = ["wsl", "--", windows_to_wsl_path(binary_path)]
        compiler_name = "wsl:gcc"
    else:
        command = [
            str(compiler),
            *STRICT_C_FLAGS,
            "-I",
            str(source_root),
            str(source_root / "independent_window_runtime.c"),
            "-o",
            str(binary_path),
        ]
        runtime_command = [str(binary_path)]
        compiler_name = str(compiler)
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )
    return {
        "available": True,
        "passed": completed.returncode == 0,
        "compiler": compiler_name,
        "command": command,
        "runtime_command": runtime_command,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "binary_sha256": sha256_file(binary_path) if completed.returncode == 0 and binary_path.is_file() else None,
    }


def run_command(command: list[str], *, timeout: float, check: bool = False) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    if check and completed.returncode != 0:
        raise TargetError(f"command failed ({completed.returncode}): {command!r}\n{completed.stderr.strip()}")
    return completed


def run_runtime_self_test(binary_path: Path, runtime_command: list[str] | None = None) -> dict[str, Any]:
    command = list(runtime_command or [str(binary_path)]) + ["--self-test"]
    completed = run_command(command, timeout=10)
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "passed": completed.returncode == 0 and "INDEPENDENT_WINDOW_V3_RUNTIME_SELF_TEST_OK" in completed.stdout,
    }


def run_runtime_schedule_validation(binary_path: Path, schedule_tsv: Path, runtime_command: list[str] | None = None) -> dict[str, Any]:
    schedule_arg = str(schedule_tsv)
    if runtime_command and runtime_command[:2] == ["wsl", "--"]:
        schedule_arg = windows_to_wsl_path(schedule_tsv)
    command = list(runtime_command or [str(binary_path)]) + ["--validate-schedule-tsv", schedule_arg]
    completed = run_command(command, timeout=10)
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "passed": completed.returncode == 0 and "INDEPENDENT_WINDOW_V3_SCHEDULE_TSV_OK" in completed.stdout,
    }


def parse_pmu_preflight(completed: subprocess.CompletedProcess[str], command: list[str]) -> dict[str, Any]:
    try:
        receipt = json.loads(completed.stdout.strip())
    except json.JSONDecodeError:
        receipt = {
            "schema_id": "CAT_CAS_INDEPENDENT_WINDOW_PMU_PREFLIGHT_V3",
            "status": "INDEPENDENT_WINDOW_V3_PMU_PREFLIGHT_FAILED",
            "scientific_classification_emitted": False,
            "parse_error": "stdout was not valid JSON",
        }
    receipt["command"] = command
    receipt["returncode"] = completed.returncode
    receipt["stderr"] = completed.stderr.strip()
    try:
        invariant_passed = (
            receipt.get("event_count") == 3
            and receipt.get("preflight_opened") is True
            and receipt.get("preflight_read_ok") is True
            and receipt.get("preflight_event_order_ok") is True
            and receipt.get("preflight_unmultiplexed") is True
            and int(receipt.get("preflight_time_enabled", 0)) > 0
            and int(receipt.get("preflight_time_enabled", -1)) == int(receipt.get("preflight_time_running", -2))
            and int(receipt.get("preflight_cpu_before", -1)) == public.RECEIVER_CORE
            and int(receipt.get("preflight_cpu_after", -1)) == public.RECEIVER_CORE
            and int(receipt.get("preflight_cycles", 0)) > 0
            and receipt.get("bytes_unchanged") is True
        )
    except (TypeError, ValueError):
        invariant_passed = False
    receipt["passed"] = (
        completed.returncode == 0
        and receipt.get("status") == "INDEPENDENT_WINDOW_V3_PMU_PREFLIGHT_OK"
        and receipt.get("scientific_classification_emitted") is False
        and invariant_passed
    )
    return receipt


def run_runtime_pmu_preflight(binary_path: Path, *, runtime_command: list[str] | None = None, runner: Any = subprocess.run) -> dict[str, Any]:
    command = list(runtime_command or [str(binary_path)]) + ["--pmu-preflight"]
    completed = runner(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20, check=False)
    return parse_pmu_preflight(completed, command)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line:
            rows.append(json.loads(line))
    return rows


def append_jsonl(output_path: Path, source_paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with output_path.open("w", encoding="utf-8", newline="\n") as out:
        for path in source_paths:
            for line in path.read_text(encoding="utf-8").splitlines():
                if line:
                    out.write(line + "\n")
                    rows.append(json.loads(line))
    return rows


def validate_stage_and_source_receipts(
    raw_records: list[dict[str, Any]],
    stage_receipts: list[dict[str, Any]],
    source_receipts: list[dict[str, Any]],
) -> dict[str, Any]:
    failures: list[str] = []
    expected_stage_count = public.TOTAL_TRIALS * 2 * len(public.SUBCAPTURE_STAGE_SEQUENCE)
    expected_source_count = public.TOTAL_TRIALS * 2
    if len(stage_receipts) != expected_stage_count:
        failures.append(f"stage receipt count {len(stage_receipts)} != {expected_stage_count}")
    if len(source_receipts) != expected_source_count:
        failures.append(f"source receipt count {len(source_receipts)} != {expected_source_count}")
    stage_ids = [str(row.get("stage_receipt_id", "")) for row in stage_receipts]
    source_ids = [str(row.get("source_receipt_id", "")) for row in source_receipts]
    if len(stage_ids) != len(set(stage_ids)):
        failures.append("duplicate stage ID")
    if len(source_ids) != len(set(source_ids)):
        failures.append("duplicate source receipt ID")
    stage_by_key: dict[tuple[int, int, str], list[dict[str, Any]]] = {}
    for row in stage_receipts:
        try:
            key = (int(row["replicate"]), int(row["trial_index"]), str(row["component"]))
            stage_by_key.setdefault(key, []).append(row)
        except (KeyError, TypeError, ValueError):
            failures.append("malformed stage receipt")
    for key, rows in stage_by_key.items():
        rows_sorted = sorted(rows, key=lambda item: int(item.get("stage_ordinal", -1)))
        names = [str(row.get("stage_name")) for row in rows_sorted]
        stamps = [int(row.get("monotonic_timestamp_ns", -1)) for row in rows_sorted]
        if names != list(public.SUBCAPTURE_STAGE_SEQUENCE):
            failures.append(f"component sequence mismatch {key}: {names}")
        if stamps != sorted(stamps):
            failures.append(f"out-of-order stage timestamps {key}")
        if any(int(row.get("return_code", -1)) != 0 for row in rows_sorted):
            failures.append(f"nonzero stage return code {key}")
    stage_id_set = set(stage_ids)
    source_id_set = set(source_ids)
    for record in raw_records:
        for component in ("positive", "negative"):
            suffixes = (
                "baseline_receipt_id",
                "pre_sentinel_receipt_id",
                "rebaseline_receipt_id",
                "source_receipt_id",
                "measure_receipt_id",
                "restore_receipt_id",
                "post_sentinel_receipt_id",
            )
            receipts = [str(record.get(f"{component}_{suffix}", "")) for suffix in suffixes]
            if len(receipts) != len(set(receipts)):
                failures.append(f"reused component receipt {record.get('trial_index')} {component}")
            for suffix, receipt_id in zip(suffixes, receipts):
                if suffix == "source_receipt_id":
                    if receipt_id not in source_id_set:
                        failures.append(f"missing source receipt row {receipt_id}")
                elif receipt_id not in stage_id_set:
                    failures.append(f"missing stage receipt row {receipt_id}")
    return {
        "passed": not failures,
        "failures": failures,
        "stage_receipt_count": len(stage_receipts),
        "source_receipt_count": len(source_receipts),
    }


def _read_required(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise TargetError(f"required readable path unavailable: {path}") from exc


def _read_optional(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _identity_candidates(row: dict[str, Any]) -> list[str]:
    candidates = [str(row["comm"])]
    args = str(row["args"])
    parts = args.strip().split()
    if parts:
        exe = Path(parts[0].strip("[]")).name
        candidates.append(exe)
        stem = Path(exe).stem.lower()
        if stem in {"python", "python3", "python2", "pypy", "pypy3", "bash", "sh"} and len(parts) >= 2:
            candidates.append(Path(parts[1]).name)
    normalized = []
    for item in candidates:
        if item.endswith(".py"):
            normalized.append(item[:-3])
        normalized.append(item)
    return sorted(set(normalized))


def _parse_ps_line(line: str) -> dict[str, Any] | None:
    parts = line.strip().split(None, 2)
    if len(parts) < 2:
        return None
    try:
        pid = int(parts[0])
    except ValueError:
        return None
    return {"pid": pid, "comm": parts[1], "args": parts[2] if len(parts) == 3 else ""}


def process_snapshot(phase: str, *, runner: Any = subprocess.run, timeout: float = 5.0) -> dict[str, Any]:
    completed = runner(list(PROCESS_SCAN_COMMAND), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, check=False)
    if completed.returncode != 0:
        raise TargetError(f"process scan failed during {phase}: {completed.stderr.strip()}")
    rows = [row for row in (_parse_ps_line(line) for line in completed.stdout.splitlines()) if row is not None]
    matches = []
    for row in rows:
        identities = _identity_candidates(row)
        markers = [marker for marker in FORBIDDEN_PROCESS_MARKERS if marker in identities]
        if markers:
            matches.append({"markers": markers, **row})
    return {
        "phase": phase,
        "timestamp_monotonic_ns": _now_ns(),
        "command": list(PROCESS_SCAN_COMMAND),
        "observed_process_count": len(rows),
        "forbidden_markers": list(FORBIDDEN_PROCESS_MARKERS),
        "forbidden_matches": matches,
        "identity_source": "comm_exe_or_interpreter_script_only",
    }


def require_no_forbidden_processes(snapshot: dict[str, Any]) -> None:
    require(not snapshot["forbidden_matches"], f"forbidden process present during {snapshot['phase']}")


def _capability_enabled(value: str | None, bit: int) -> bool:
    if value is None:
        return False
    try:
        return bool(int(value, 16) & (1 << bit))
    except ValueError:
        return False


def privileged_identity_snapshot(status_path: Path = Path("/proc/self/status")) -> dict[str, Any]:
    fields: dict[str, str] = {}
    status_text = _read_required(status_path)
    for line in status_text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if key in {"CapEff", "CapPrm", "CapBnd"}:
            fields[key] = value.strip().split()[0]
    missing = sorted(set(("CapEff", "CapPrm", "CapBnd")) - set(fields))
    if missing:
        raise TargetError(f"missing capability fields: {missing}")
    snapshot = {
        "real_uid": os.getuid(),
        "effective_uid": os.geteuid(),
        "real_gid": os.getgid(),
        "effective_gid": os.getegid(),
        "status_path": str(status_path),
        "capability_fields": fields,
        "cap_eff": fields["CapEff"],
        "cap_prm": fields["CapPrm"],
        "cap_bnd": fields["CapBnd"],
        "cap_perfmon_effective": _capability_enabled(fields["CapEff"], CAP_PERFMON_BIT),
        "cap_perfmon_permitted": _capability_enabled(fields["CapPrm"], CAP_PERFMON_BIT),
        "cap_perfmon_bounding": _capability_enabled(fields["CapBnd"], CAP_PERFMON_BIT),
        "effective_uid_zero_required": True,
        "nonroot_cap_perfmon_path_supported": False,
    }
    snapshot["passed"] = snapshot["effective_uid"] == 0
    return snapshot


def cpuinfo_snapshot() -> dict[str, str]:
    info: dict[str, str] = {}
    path = Path("/proc/cpuinfo")
    if path.is_file():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                info.setdefault(key.strip(), value.strip())
    return info


def pmu_platform_snapshot(
    *,
    cpuinfo: dict[str, str] | None = None,
    identity: dict[str, Any] | None = None,
    event_source_root: Path = Path("/sys/bus/event_source/devices/cpu"),
    cpu_root: Path = Path("/sys/devices/system/cpu"),
    perf_event_paranoid: Path = Path("/proc/sys/kernel/perf_event_paranoid"),
) -> dict[str, Any]:
    identity = privileged_identity_snapshot() if identity is None else identity
    cpuinfo = cpuinfo_snapshot() if cpuinfo is None else cpuinfo
    require(identity["effective_uid"] == 0, "effective UID zero required for target PMU custody")
    vendor = cpuinfo.get("vendor_id")
    family = cpuinfo.get("cpu family")
    require(vendor == "AuthenticAMD", f"unexpected CPU vendor: {vendor}")
    require(str(family) == "16", f"unexpected CPU family: {family}")
    pmu_type = _read_required(event_source_root / "type")
    require(re.fullmatch(r"\d+", pmu_type), "PMU type is not numeric")
    event_format = _read_required(event_source_root / "format" / "event")
    umask_format = _read_required(event_source_root / "format" / "umask")
    require(event_format == PMU_EVENT_FORMAT, f"PMU event format drift: {event_format}")
    require(umask_format == PMU_UMASK_FORMAT, f"PMU umask format drift: {umask_format}")
    core_online = {}
    for core in (public.SOURCE_CORE, public.RECEIVER_CORE):
        online = _read_optional(cpu_root / f"cpu{core}" / "online")
        core_online[str(core)] = "1" if online is None else online
        require(core_online[str(core)] == "1", f"CPU {core} not online")
    paranoid_raw = _read_required(perf_event_paranoid)
    require(re.fullmatch(r"[+-]?\d+", paranoid_raw), "perf_event_paranoid malformed")
    return {
        "vendor_id": vendor,
        "cpu_family": family,
        "effective_uid": identity["effective_uid"],
        "pmu_type": pmu_type,
        "event_format": event_format,
        "umask_format": umask_format,
        "perf_event_paranoid": paranoid_raw,
        "perf_event_paranoid_role": "diagnostic_only_for_root",
        "core_online": core_online,
        "passed": True,
    }


def temperature_observation(phase: str, *, hwmon_root: Path = Path("/sys/class/hwmon")) -> dict[str, Any]:
    candidates = []
    for hwmon in sorted(hwmon_root.glob("hwmon*")):
        name_path = hwmon / "name"
        if name_path.is_file() and _read_required(name_path) == "k10temp":
            candidates.append(hwmon)
    if len(candidates) != 1:
        raise TargetError(f"k10temp resolver found {len(candidates)} candidates during {phase}")
    temp_path = candidates[0] / "temp1_input"
    raw = _read_required(temp_path)
    if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", raw) is None:
        raise TargetError(f"malformed k10temp value during {phase}: {raw!r}")
    temp_c = float(raw) / 1000.0
    low, high = PHYSICALLY_PLAUSIBLE_TEMP_C
    require(low <= temp_c <= high, f"implausible k10temp value during {phase}: {temp_c}")
    return {
        "phase": phase,
        "timestamp_monotonic_ns": _now_ns(),
        "hwmon_path": str(candidates[0]),
        "input_path": str(temp_path),
        "raw_millidegrees_c": raw,
        "temperature_c": temp_c,
        "veto_c": TEMPERATURE_VETO_C,
        "below_veto": temp_c < TEMPERATURE_VETO_C,
    }


def require_temperature_below(observation: dict[str, Any]) -> None:
    require(bool(observation["below_veto"]), f"temperature veto during {observation['phase']}: {observation['temperature_c']}")


def policy_snapshot(phase: str, *, cpufreq_root: Path = Path("/sys/devices/system/cpu/cpufreq")) -> dict[str, Any]:
    policies: dict[str, Any] = {}
    for policy_id in POLICY_IDS:
        path = cpufreq_root / f"policy{policy_id}"
        require(path.is_dir(), f"cpufreq policy missing: {path}")
        affected = _read_optional(path / "affected_cpus")
        related = _read_optional(path / "related_cpus")
        membership = affected or related
        require(bool(membership), f"policy{policy_id} has no CPU membership field")
        members = {int(value) for value in membership.split() if value.isdigit()}
        require(policy_id in members, f"policy{policy_id} membership does not include CPU {policy_id}")
        policies[str(policy_id)] = {
            "policy_id": policy_id,
            "resolved_policy_path": str(path.resolve()),
            "scaling_min_freq": _read_required(path / "scaling_min_freq"),
            "scaling_max_freq": _read_required(path / "scaling_max_freq"),
            "scaling_cur_freq": _read_required(path / "scaling_cur_freq"),
            "affected_cpus": affected,
            "related_cpus": related,
            "cpu_membership": sorted(members),
            "scaling_driver": _read_required(path / "scaling_driver"),
        }
    return {"phase": phase, "timestamp_monotonic_ns": _now_ns(), "policies": policies}


def compare_policy_snapshots(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    failures = []
    for policy_id, left in before["policies"].items():
        right = after["policies"].get(policy_id)
        if right is None:
            failures.append({"policy_id": policy_id, "field": "policy_missing_after"})
            continue
        for field in ("resolved_policy_path", "affected_cpus", "related_cpus", "cpu_membership", "scaling_driver"):
            if left.get(field) != right.get(field):
                failures.append({"policy_id": policy_id, "field": field, "before": left.get(field), "after": right.get(field)})
        for field in ("scaling_min_freq", "scaling_max_freq"):
            if left[field] != right[field]:
                failures.append({"policy_id": policy_id, "field": field, "before": left[field], "after": right[field]})
    return {"passed": not failures, "failures": failures, "before_phase": before.get("phase"), "after_phase": after.get("phase")}


def build_copyback_manifest(output_root: Path) -> dict[str, Any]:
    files = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name != "COPYBACK_MANIFEST.json":
            rel = path.relative_to(output_root).as_posix()
            files.append({"path": rel, "size": path.stat().st_size, "sha256": sha256_file(path)})
    manifest = {"schema_id": "CAT_CAS_INDEPENDENT_WINDOW_COPYBACK_MANIFEST_V3", "run_id": public.RUN_ID, "files": files}
    write_json(output_root / "COPYBACK_MANIFEST.json", manifest)
    return manifest


def build_execution_manifest(output_root: Path, **values: Any) -> dict[str, Any]:
    manifest = {
        "schema_id": "CAT_CAS_INDEPENDENT_WINDOW_EXECUTION_MANIFEST_V3",
        "run_id": values["run_id"],
        "implementation_manifest_sha256": values["implementation_manifest_sha256"],
        "source_bundle_sha256": values["source_bundle_sha256"],
        "schedule_json_sha256": values["schedule_json_sha256"],
        "schedule_tsv_sha256": values["schedule_tsv_sha256"],
        "binary_custody_mode": BINARY_CUSTODY_MODE,
        "offline_validation_binary_sha256": values.get("offline_validation_binary_sha256"),
        "live_runtime_binary_sha256": values["live_runtime_binary_sha256"],
        "raw_capture_sha256": sha256_file(output_root / "RAW_INDEPENDENT_WINDOW_CAPTURE.jsonl"),
        "sentinels_sha256": sha256_file(output_root / "INDEPENDENT_WINDOW_SENTINELS.jsonl"),
        "stage_receipts_sha256": sha256_file(output_root / "INDEPENDENT_WINDOW_STAGE_RECEIPTS.jsonl"),
        "source_receipts_sha256": sha256_file(output_root / "INDEPENDENT_WINDOW_SOURCE_RECEIPTS.jsonl"),
        "features_sha256": sha256_file(output_root / "INDEPENDENT_WINDOW_FEATURES.json"),
        "adjudication_sha256": sha256_file(output_root / "INDEPENDENT_WINDOW_ADJUDICATION.json"),
        "final_result_sha256": sha256_file(output_root / "FINAL_RESULT_INDEPENDENT_WINDOW_V3.json"),
        "raw_record_count": values["raw_record_count"],
        "component_window_count": values["component_window_count"],
        "allowed_classifications": list(public.ALLOWED_CLASSES),
        "forbidden_classifications": list(public.FORBIDDEN_CLASSES),
        "primary_coordinate": public.PRIMARY_COORDINATE,
    }
    manifest["manifest_sha256"] = execution_manifest_digest(manifest)
    write_json(output_root / "INDEPENDENT_WINDOW_V3_MANIFEST.json", manifest)
    return manifest


def custody_log(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_id": "CAT_CAS_INDEPENDENT_WINDOW_LIVE_CUSTODY_LOG_V3",
        "run_id": state.get("run_id", public.RUN_ID),
        "phases_completed": list(state.get("phases_completed", [])),
        "temperature_observations": list(state.get("temperature_observations", [])),
        "process_snapshots": list(state.get("process_snapshots", [])),
        "policy_snapshots": list(state.get("policy_snapshots", [])),
        "policy_comparison": state.get("policy_comparison"),
        "privilege_capability_snapshot": state.get("privilege_capability_snapshot"),
        "pmu_platform_snapshot": state.get("pmu_platform_snapshot"),
        "pmu_preflight_receipt": state.get("pmu_preflight_receipt"),
        "runtime_self_test": state.get("runtime_self_test"),
        "runtime_schedule_validation": state.get("runtime_schedule_validation"),
        "compile": state.get("compile"),
        "hardware_execution_began": bool(state.get("hardware_execution_began", False)),
        "pmu_preflight_began": bool(state.get("pmu_preflight_began", False)),
        "pmu_preflight_completed": bool(state.get("pmu_preflight_completed", False)),
        "replicate_states": state.get("replicate_states", {}),
        "frequency_writes": 0,
        "voltage_writes": 0,
        "sysctl_writes": 0,
        "msr_reads": 0,
        "msr_writes": 0,
        "physical_address_access": False,
        "cache_set_mapping": False,
    }


def failure_evidence(output_root: Path, *, state: dict[str, Any], exc: BaseException) -> list[str]:
    output_root.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    target_failure = {
        "schema_id": "CAT_CAS_INDEPENDENT_WINDOW_TARGET_FAILURE_V3",
        "status": "INDEPENDENT_WINDOW_V3_TARGET_FAILED",
        "run_id": state.get("run_id", public.RUN_ID),
        "failure_phase": state.get("phase", "unknown"),
        "exception_type": type(exc).__name__,
        "exception_message": str(exc),
        "hardware_execution_began": bool(state.get("hardware_execution_began", False)),
        "pmu_preflight_began": bool(state.get("pmu_preflight_began", False)),
        "pmu_preflight_completed": bool(state.get("pmu_preflight_completed", False)),
        "replicate_states": state.get("replicate_states", {}),
        "source_schedule_hashes_verified": bool(state.get("source_schedule_hashes_verified", False)),
        "compile_command": state.get("compile", {}).get("command") if isinstance(state.get("compile"), dict) else None,
        "live_binary_hash": state.get("compile", {}).get("binary_sha256") if isinstance(state.get("compile"), dict) else None,
        "runtime_self_test_state": state.get("runtime_self_test"),
        "pmu_preflight_receipt": state.get("pmu_preflight_receipt"),
        "temperature_observations": state.get("temperature_observations", []),
        "process_snapshots": state.get("process_snapshots", []),
        "policy_snapshots": state.get("policy_snapshots", []),
        "privilege_capability_snapshot": state.get("privilege_capability_snapshot"),
        "scientific_classification_emitted": False,
    }
    try:
        write_json(output_root / "TARGET_FAILURE_INDEPENDENT_WINDOW_V3.json", target_failure)
        write_json(output_root / "LIVE_CUSTODY_LOG.json", custody_log(state))
        final = {
            "schema_id": "CAT_CAS_INDEPENDENT_WINDOW_TARGET_RESULT_V3",
            "status": "INDEPENDENT_WINDOW_V3_TARGET_FAILED",
            "run_id": state.get("run_id", public.RUN_ID),
            "failure_sha256": sha256_file(output_root / "TARGET_FAILURE_INDEPENDENT_WINDOW_V3.json"),
            "scientific_classification_emitted": False,
        }
        write_json(output_root / "FINAL_RESULT_INDEPENDENT_WINDOW_V3.json", final)
        failure_manifest = {
            "schema_id": "CAT_CAS_INDEPENDENT_WINDOW_FAILURE_MANIFEST_V3",
            "run_id": state.get("run_id", public.RUN_ID),
            "target_failure_sha256": sha256_file(output_root / "TARGET_FAILURE_INDEPENDENT_WINDOW_V3.json"),
            "final_result_sha256": sha256_file(output_root / "FINAL_RESULT_INDEPENDENT_WINDOW_V3.json"),
            "live_custody_log_sha256": sha256_file(output_root / "LIVE_CUSTODY_LOG.json"),
            "scientific_classification_emitted": False,
        }
        failure_manifest["manifest_sha256"] = failure_manifest_digest(failure_manifest)
        write_json(output_root / "INDEPENDENT_WINDOW_V3_FAILURE_MANIFEST.json", failure_manifest)
        build_copyback_manifest(output_root)
    except Exception as inner:  # pragma: no cover - best effort sealing path
        errors.append(str(inner))
    return errors


def offline_validate(source_root: Path, output_root: Path) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    schedule_receipt = validate_schedule_artifacts(source_root)
    replacements: list[tuple[str, str]] = []
    with tempfile.TemporaryDirectory(prefix="independent_window_v3_target_") as temp:
        temp_root = Path(temp)
        replacements = [(str(temp_root), "<target-temp>"), (windows_to_wsl_path(temp_root), "<target-temp>")]
        bundle_sha, hashes = deterministic_source_bundle(source_root, temp_root / "INDEPENDENT_WINDOW_SOURCE_BUNDLE.tar.gz")
        compile_receipt = compile_runtime(source_root, temp_root / "independent_window_runtime")
        if not compile_receipt["passed"] and sys.platform.startswith("win"):
            compile_receipt = compile_runtime(source_root, temp_root / "independent_window_runtime", prefer_wsl=True)
        runtime_self = {"passed": False, "skipped": True}
        schedule_tsv_check = {"passed": False, "skipped": True}
        if compile_receipt["passed"]:
            runtime_self = run_runtime_self_test(temp_root / "independent_window_runtime", compile_receipt.get("runtime_command"))
            schedule_tsv_check = run_runtime_schedule_validation(
                temp_root / "independent_window_runtime",
                source_root / "INDEPENDENT_WINDOW_PUBLIC_SCHEDULE.tsv",
                compile_receipt.get("runtime_command"),
            )
    public_self = public.self_test()
    result = {
        "schema_id": "CAT_CAS_INDEPENDENT_WINDOW_TARGET_OFFLINE_VALIDATION_V3",
        "run_id": public.RUN_ID,
        "schedule": {k: v for k, v in schedule_receipt.items() if k != "schedule"},
        "source_hashes": hashes,
        "source_bundle_sha256": bundle_sha,
        "compile": compile_receipt,
        "runtime_self_test": runtime_self,
        "runtime_schedule_validation": schedule_tsv_check,
        "public_self_test_sha256": public_self["self_test_sha256"],
        "public_self_test_passed": public_self["self_test_passed"],
        "network_connections": 0,
        "ssh_executions": 0,
        "scp_executions": 0,
        "hardware_executions": 0,
    }
    result["passed"] = (
        result["public_self_test_passed"]
        and result["compile"]["passed"]
        and result["runtime_self_test"]["passed"]
        and result["runtime_schedule_validation"]["passed"]
    )
    stable_result = normalize_receipt_paths({k: v for k, v in result.items() if k != "validation_sha256"}, replacements)
    result["validation_sha256"] = public.digest(stable_result)
    write_json(output_root / "INDEPENDENT_WINDOW_TARGET_OFFLINE_VALIDATION.json", result)
    return result


def _run_replicate(binary: Path, runtime_command: list[str], schedule_tsv: Path, output_root: Path, replicate: int) -> subprocess.CompletedProcess[str]:
    command = list(runtime_command) + [
        "--capture",
        "--schedule-tsv",
        str(schedule_tsv),
        "--output-root",
        str(output_root),
        "--replicate",
        str(replicate),
    ]
    return run_command(command, timeout=180)


def output_tree_digest(root: Path) -> str:
    rows = []
    if root.exists():
        for path in sorted(root.rglob("*")):
            if path.is_file():
                rel = path.relative_to(root).as_posix()
                rows.append({"path": rel, "size": path.stat().st_size, "sha256": sha256_file(path)})
    return public.digest(rows)


def execute_live(source_root: Path, output_root: Path, *, run_id: str, expected_manifest_sha: str) -> dict[str, Any]:
    state: dict[str, Any] = {
        "run_id": run_id,
        "phase": "start",
        "phases_completed": [],
        "temperature_observations": [],
        "process_snapshots": [],
        "policy_snapshots": [],
        "replicate_states": {"0": {"began": False, "completed": False}, "1": {"began": False, "completed": False}},
    }
    output_root_created = False
    try:
        require(run_id == public.RUN_ID, "run ID mismatch")
        require(re.fullmatch(r"[a-z0-9_]{8,80}", run_id) is not None, "run ID is not closed")
        state["phase"] = "create_output_root"
        require(not output_root.exists(), f"output root already exists: {output_root}")
        output_root.mkdir(mode=0o700, parents=True, exist_ok=False)
        output_root_created = True
        state["phases_completed"].append(state["phase"])

        state["phase"] = "process_preflight"
        process = process_snapshot("process_preflight")
        require_no_forbidden_processes(process)
        state["process_snapshots"].append(process)
        state["phases_completed"].append(state["phase"])

        state["phase"] = "policy_before"
        policy_before = policy_snapshot("before")
        state["policy_snapshots"].append(policy_before)
        state["phases_completed"].append(state["phase"])

        state["phase"] = "privilege_snapshot"
        identity = privileged_identity_snapshot()
        require(identity["passed"], "effective UID zero required")
        state["privilege_capability_snapshot"] = identity
        state["phases_completed"].append(state["phase"])

        state["phase"] = "pmu_platform_snapshot"
        pmu_platform = pmu_platform_snapshot(identity=identity)
        state["pmu_platform_snapshot"] = pmu_platform
        state["phases_completed"].append(state["phase"])

        state["phase"] = "temperature_before_compilation"
        temp = temperature_observation("before_compilation")
        require_temperature_below(temp)
        state["temperature_observations"].append(temp)
        state["phases_completed"].append(state["phase"])

        state["phase"] = "source_manifest_schedule_verification"
        implementation_manifest_path = source_root / "INDEPENDENT_WINDOW_IMPLEMENTATION_MANIFEST.json"
        require(implementation_manifest_path.is_file(), "implementation manifest missing")
        implementation_manifest = json.loads(implementation_manifest_path.read_text(encoding="utf-8"))
        require(implementation_manifest["implementation_manifest_sha256"] == manifest_digest(implementation_manifest), "implementation manifest self digest mismatch")
        require(implementation_manifest["implementation_manifest_sha256"] == expected_manifest_sha, "expected manifest SHA mismatch")
        schedule_receipt = validate_schedule_artifacts(source_root, output_root)
        require(source_hashes(source_root) == implementation_manifest["source_hashes"], "source hashes drifted")
        require(schedule_receipt["schedule_json_sha256"] == implementation_manifest["schedule_json_sha256"], "schedule JSON SHA mismatch")
        require(schedule_receipt["schedule_tsv_sha256"] == implementation_manifest["schedule_tsv_sha256"], "schedule TSV SHA mismatch")
        state["source_schedule_hashes_verified"] = True
        state["phases_completed"].append(state["phase"])

        state["phase"] = "source_bundle"
        source_bundle_sha, source_hash_receipt = deterministic_source_bundle(source_root, output_root / "INDEPENDENT_WINDOW_SOURCE_BUNDLE.tar.gz")
        require(source_bundle_sha == implementation_manifest["expected_source_bundle_sha256"], "source bundle SHA mismatch")
        state["phases_completed"].append(state["phase"])

        state["phase"] = "strict_compile"
        compile_receipt = compile_runtime(source_root, output_root / "independent_window_runtime")
        state["compile"] = compile_receipt
        require(compile_receipt["passed"], f"strict target compilation failed: {compile_receipt.get('stderr')}")
        state["phases_completed"].append(state["phase"])

        state["phase"] = "runtime_self_test"
        runtime_self = run_runtime_self_test(output_root / "independent_window_runtime", compile_receipt["runtime_command"])
        state["runtime_self_test"] = runtime_self
        require(runtime_self["passed"], "runtime self-test failed")
        state["phases_completed"].append(state["phase"])

        state["phase"] = "runtime_schedule_validation"
        schedule_validation = run_runtime_schedule_validation(
            output_root / "independent_window_runtime",
            output_root / "INDEPENDENT_WINDOW_PUBLIC_SCHEDULE.tsv",
            compile_receipt["runtime_command"],
        )
        state["runtime_schedule_validation"] = schedule_validation
        require(schedule_validation["passed"], "runtime schedule validation failed")
        state["phases_completed"].append(state["phase"])

        state["phase"] = "runtime_pmu_preflight"
        state["pmu_preflight_began"] = True
        pmu_preflight = run_runtime_pmu_preflight(output_root / "independent_window_runtime", runtime_command=compile_receipt["runtime_command"])
        state["pmu_preflight_receipt"] = pmu_preflight
        state["pmu_preflight_completed"] = True
        require(pmu_preflight["passed"], "runtime PMU preflight failed")
        state["phases_completed"].append(state["phase"])

        batch_roots = []
        for replicate in (0, 1):
            state["phase"] = f"temperature_before_replicate_{replicate}"
            temp = temperature_observation(f"before_replicate_{replicate}")
            require_temperature_below(temp)
            state["temperature_observations"].append(temp)
            state["phases_completed"].append(state["phase"])

            state["phase"] = f"execute_replicate_{replicate}"
            state["hardware_execution_began"] = True
            state["replicate_states"][str(replicate)]["began"] = True
            batch_root = output_root / f"replicate_{replicate}"
            require(not batch_root.exists(), f"replicate output root already exists: {batch_root}")
            completed = _run_replicate(
                output_root / "independent_window_runtime",
                compile_receipt["runtime_command"],
                output_root / "INDEPENDENT_WINDOW_PUBLIC_SCHEDULE.tsv",
                batch_root,
                replicate,
            )
            (output_root / f"INDEPENDENT_WINDOW_RUNTIME_STDOUT_REPLICATE_{replicate}.txt").write_text(completed.stdout, encoding="utf-8")
            (output_root / f"INDEPENDENT_WINDOW_RUNTIME_STDERR_REPLICATE_{replicate}.txt").write_text(completed.stderr, encoding="utf-8")
            require(completed.returncode == 0, f"runtime replicate {replicate} failed")
            state["replicate_states"][str(replicate)]["completed"] = True
            batch_roots.append(batch_root)
            state["phases_completed"].append(state["phase"])

            state["phase"] = f"post_replicate_{replicate}_temperature_process"
            temp = temperature_observation(f"after_replicate_{replicate}")
            require_temperature_below(temp)
            state["temperature_observations"].append(temp)
            process = process_snapshot(f"after_replicate_{replicate}")
            require_no_forbidden_processes(process)
            state["process_snapshots"].append(process)
            state["phases_completed"].append(state["phase"])

        state["phase"] = "combine_and_extract"
        raw_records = append_jsonl(output_root / "RAW_INDEPENDENT_WINDOW_CAPTURE.jsonl", [root / "RAW_INDEPENDENT_WINDOW_CAPTURE.jsonl" for root in batch_roots])
        sentinels = append_jsonl(output_root / "INDEPENDENT_WINDOW_SENTINELS.jsonl", [root / "INDEPENDENT_WINDOW_SENTINELS.jsonl" for root in batch_roots])
        stage_receipts = append_jsonl(output_root / "INDEPENDENT_WINDOW_STAGE_RECEIPTS.jsonl", [root / "INDEPENDENT_WINDOW_STAGE_RECEIPTS.jsonl" for root in batch_roots])
        source_receipts = append_jsonl(output_root / "INDEPENDENT_WINDOW_SOURCE_RECEIPTS.jsonl", [root / "INDEPENDENT_WINDOW_SOURCE_RECEIPTS.jsonl" for root in batch_roots])
        require(len(raw_records) == public.TOTAL_TRIALS, "combined raw record count mismatch")
        require(len(raw_records) * 2 == public.TOTAL_COMPONENT_WINDOWS, "combined component window count mismatch")
        require(len(stage_receipts) == public.TOTAL_TRIALS * 2 * len(public.SUBCAPTURE_STAGE_SEQUENCE), "stage receipt count mismatch")
        require(len(source_receipts) == public.TOTAL_TRIALS * 2, "source receipt count mismatch")
        receipt_validation = validate_stage_and_source_receipts(raw_records, stage_receipts, source_receipts)
        require(receipt_validation["passed"], f"stage/source receipt validation failed: {receipt_validation['failures'][:4]}")
        features = public.extract_features(schedule_receipt["schedule"], raw_records, sentinels)
        require(features["integrity"]["schedule_matched"], "runtime capture failed public integrity laws")
        adjudication = public.adjudicate(features)
        require(adjudication["status"] in public.ALLOWED_CLASSES, "adjudication emitted non-allowed status")
        write_json(output_root / "INDEPENDENT_WINDOW_FEATURES.json", features)
        write_json(output_root / "INDEPENDENT_WINDOW_ADJUDICATION.json", adjudication)
        state["phases_completed"].append(state["phase"])

        state["phase"] = "policy_after"
        policy_after = policy_snapshot("after")
        state["policy_snapshots"].append(policy_after)
        policy_comparison = compare_policy_snapshots(policy_before, policy_after)
        state["policy_comparison"] = policy_comparison
        require(policy_comparison["passed"], "policy min/max or identity drift")
        state["phases_completed"].append(state["phase"])

        state["phase"] = "final_temperature_process"
        temp = temperature_observation("final")
        require_temperature_below(temp)
        state["temperature_observations"].append(temp)
        process = process_snapshot("final")
        require_no_forbidden_processes(process)
        state["process_snapshots"].append(process)
        state["phases_completed"].append(state["phase"])

        final = {
            "schema_id": "CAT_CAS_INDEPENDENT_WINDOW_TARGET_RESULT_V3",
            "status": "INDEPENDENT_WINDOW_V3_TARGET_COMPLETE",
            "run_id": run_id,
            "adjudication_status": adjudication["status"],
            "scientific_classification_emitted": True,
            "primary_coordinate": public.PRIMARY_COORDINATE,
            "allowed_classifications": list(public.ALLOWED_CLASSES),
            "forbidden_classifications": list(public.FORBIDDEN_CLASSES),
            "implementation_manifest_sha256": implementation_manifest["implementation_manifest_sha256"],
            "schedule_json_sha256": schedule_receipt["schedule_json_sha256"],
            "schedule_tsv_sha256": schedule_receipt["schedule_tsv_sha256"],
            "source_bundle_sha256": source_bundle_sha,
            "binary_custody_mode": BINARY_CUSTODY_MODE,
            "offline_validation_binary_sha256": implementation_manifest.get("offline_validation_binary_sha256"),
            "live_runtime_binary_sha256": compile_receipt["binary_sha256"],
            "raw_capture_sha256": sha256_file(output_root / "RAW_INDEPENDENT_WINDOW_CAPTURE.jsonl"),
            "sentinels_sha256": sha256_file(output_root / "INDEPENDENT_WINDOW_SENTINELS.jsonl"),
            "stage_receipts_sha256": sha256_file(output_root / "INDEPENDENT_WINDOW_STAGE_RECEIPTS.jsonl"),
            "source_receipts_sha256": sha256_file(output_root / "INDEPENDENT_WINDOW_SOURCE_RECEIPTS.jsonl"),
            "features_sha256": sha256_file(output_root / "INDEPENDENT_WINDOW_FEATURES.json"),
            "adjudication_sha256": sha256_file(output_root / "INDEPENDENT_WINDOW_ADJUDICATION.json"),
            "raw_record_count": len(raw_records),
            "component_window_count": len(raw_records) * 2,
        }
        write_json(output_root / "FINAL_RESULT_INDEPENDENT_WINDOW_V3.json", final)
        write_json(output_root / "LIVE_CUSTODY_LOG.json", custody_log(state))
        execution_manifest = build_execution_manifest(
            output_root,
            run_id=run_id,
            implementation_manifest_sha256=implementation_manifest["implementation_manifest_sha256"],
            source_bundle_sha256=source_bundle_sha,
            schedule_json_sha256=schedule_receipt["schedule_json_sha256"],
            schedule_tsv_sha256=schedule_receipt["schedule_tsv_sha256"],
            offline_validation_binary_sha256=implementation_manifest.get("offline_validation_binary_sha256"),
            live_runtime_binary_sha256=compile_receipt["binary_sha256"],
            raw_record_count=len(raw_records),
            component_window_count=len(raw_records) * 2,
        )
        copyback = build_copyback_manifest(output_root)
        result = {
            "schema_id": "CAT_CAS_INDEPENDENT_WINDOW_TARGET_EXECUTION_V3",
            "final": final,
            "execution_manifest": execution_manifest,
            "copyback_manifest": copyback,
            "passed": True,
        }
        return result
    except Exception as exc:
        if output_root_created:
            failure_evidence(output_root, state=state, exc=exc)
        raise


def self_test(source_root: Path, output_root: Path) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    offline = offline_validate(source_root, output_root / "offline")
    capture = public.build_mock_capture("independent_no_carryover")
    features = public.extract_features(capture["schedule"], capture["raw_records"], capture["sentinels"])
    adjudication = public.adjudicate(features)
    capture_schema_mock = {
        "raw_records_128": len(capture["raw_records"]) == public.TOTAL_TRIALS,
        "component_windows_256": len(capture["raw_records"]) * 2 == public.TOTAL_COMPONENT_WINDOWS,
        "stage_sequence_present": all(
            row.get("positive_stage_sequence") == list(public.SUBCAPTURE_STAGE_SEQUENCE)
            and row.get("negative_stage_sequence") == list(public.SUBCAPTURE_STAGE_SEQUENCE)
            for row in capture["raw_records"]
        ),
        "source_work_8192": all(row.get("mapping_leg_source_work") == public.SOURCE_WORK_PER_MAPPING_LEG for row in capture["raw_records"]),
        "adjudication_confirms_independent_mock": adjudication["status"] == public.CLASS_CONFIRMED,
    }

    def pmu_stdout(**overrides: Any) -> str:
        receipt = {
            "schema_id": "CAT_CAS_INDEPENDENT_WINDOW_PMU_PREFLIGHT_V3",
            "status": "INDEPENDENT_WINDOW_V3_PMU_PREFLIGHT_OK",
            "scientific_classification_emitted": False,
            "event_count": 3,
            "bytes_unchanged": True,
            "preflight_opened": True,
            "preflight_read_ok": True,
            "preflight_event_order_ok": True,
            "preflight_unmultiplexed": True,
            "preflight_time_enabled": 100,
            "preflight_time_running": 100,
            "preflight_cpu_before": public.RECEIVER_CORE,
            "preflight_cpu_after": public.RECEIVER_CORE,
            "preflight_cycles": 100,
            "preflight_change_to_dirty": 0,
            "preflight_probe_dirty": 0,
        }
        receipt.update(overrides)
        return json.dumps(receipt, sort_keys=True)

    def pmu_runner(stdout: str, returncode: int = 0, stderr: str = ""):
        return lambda *args, **kwargs: subprocess.CompletedProcess(args[0], returncode, stdout=stdout, stderr=stderr)

    good_pmu = run_runtime_pmu_preflight(Path("/synthetic/independent_window_runtime"), runner=pmu_runner(pmu_stdout()))["passed"]
    partial_pmu_rejected = not run_runtime_pmu_preflight(
        Path("/synthetic/independent_window_runtime"),
        runner=pmu_runner(pmu_stdout(event_count=2)),
    )["passed"]
    multiplexed_rejected = not run_runtime_pmu_preflight(
        Path("/synthetic/independent_window_runtime"),
        runner=pmu_runner(pmu_stdout(preflight_unmultiplexed=False, preflight_time_running=50)),
    )["passed"]
    wrong_cpu_rejected = not run_runtime_pmu_preflight(
        Path("/synthetic/independent_window_runtime"),
        runner=pmu_runner(pmu_stdout(preflight_cpu_after=4)),
    )["passed"]

    failure_root = output_root / "failure_case"
    state = {
        "run_id": public.RUN_ID,
        "phase": "synthetic_failure",
        "hardware_execution_began": False,
        "pmu_preflight_began": False,
        "pmu_preflight_completed": False,
        "replicate_states": {"0": {"began": False, "completed": False}, "1": {"began": False, "completed": False}},
    }
    failure_errors = failure_evidence(failure_root, state=state, exc=TargetError("synthetic failure"))
    failure_final = json.loads((failure_root / "FINAL_RESULT_INDEPENDENT_WINDOW_V3.json").read_text(encoding="utf-8"))
    failure_manifest = json.loads((failure_root / "INDEPENDENT_WINDOW_V3_FAILURE_MANIFEST.json").read_text(encoding="utf-8"))
    failure_sealed = (
        not failure_errors
        and all((failure_root / name).is_file() for name in FAILURE_REQUIRED_FILES)
        and failure_final.get("scientific_classification_emitted") is False
        and "adjudication_status" not in failure_final
        and failure_manifest["manifest_sha256"] == failure_manifest_digest(failure_manifest)
    )
    existing_root = output_root / "existing_root_regression"
    existing_root.mkdir(parents=True, exist_ok=True)
    (existing_root / "marker.txt").write_text("do-not-touch\n", encoding="utf-8", newline="\n")
    before_existing = output_tree_digest(existing_root)
    existing_root_rejected_without_write = False
    try:
        execute_live(
            source_root,
            existing_root,
            run_id=public.RUN_ID,
            expected_manifest_sha="0" * 64,
        )
    except Exception:
        existing_root_rejected_without_write = output_tree_digest(existing_root) == before_existing
    c_existing_root = output_root / "c_existing_root_regression"
    c_existing_root.mkdir(parents=True, exist_ok=True)
    (c_existing_root / "marker.txt").write_text("do-not-touch\n", encoding="utf-8", newline="\n")
    before_c_existing = output_tree_digest(c_existing_root)
    c_compile = compile_runtime(source_root, output_root / "c_regression_runtime" / "independent_window_runtime")
    if not c_compile["passed"] and sys.platform.startswith("win"):
        c_compile = compile_runtime(
            source_root,
            output_root / "c_regression_runtime" / "independent_window_runtime",
            prefer_wsl=True,
        )
    c_existing_capture_root_rejected_without_write = False
    if c_compile["passed"]:
        schedule_arg = str(source_root / "INDEPENDENT_WINDOW_PUBLIC_SCHEDULE.tsv")
        output_arg = str(c_existing_root)
        if c_compile.get("runtime_command", [])[:2] == ["wsl", "--"]:
            schedule_arg = windows_to_wsl_path(source_root / "INDEPENDENT_WINDOW_PUBLIC_SCHEDULE.tsv")
            output_arg = windows_to_wsl_path(c_existing_root)
        c_reject = run_command(
            list(c_compile["runtime_command"])
            + ["--capture", "--schedule-tsv", schedule_arg, "--output-root", output_arg, "--replicate", "0"],
            timeout=10,
        )
        c_existing_capture_root_rejected_without_write = (
            c_reject.returncode != 0 and output_tree_digest(c_existing_root) == before_c_existing
        )
    result = {
        "schema_id": "CAT_CAS_INDEPENDENT_WINDOW_TARGET_SELF_TEST_V3",
        "offline_validation_sha256": offline["validation_sha256"],
        "offline_validation_passed": offline["passed"],
        "capture_schema_mock": capture_schema_mock,
        "successful_pmu_preflight_emits_no_scientific_classification": good_pmu,
        "partial_pmu_group_fails_closed": partial_pmu_rejected,
        "multiplexed_pmu_preflight_fails_closed": multiplexed_rejected,
        "wrong_cpu_affinity_fails_closed": wrong_cpu_rejected,
        "failure_evidence_sealed_without_scientific_classification": failure_sealed,
        "existing_target_output_root_rejected_without_write": existing_root_rejected_without_write,
        "existing_c_capture_root_rejected_without_write": c_existing_capture_root_rejected_without_write,
        "binary_custody_mode": BINARY_CUSTODY_MODE,
        "strict_c_flags": list(STRICT_C_FLAGS),
        "network_connections": 0,
        "ssh_executions": 0,
        "scp_executions": 0,
        "hardware_executions": 0,
        "frequency_writes": 0,
        "voltage_writes": 0,
        "sysctl_writes": 0,
        "msr_reads": 0,
        "msr_writes": 0,
    }
    result["self_test_passed"] = (
        offline["passed"]
        and all(capture_schema_mock.values())
        and good_pmu
        and partial_pmu_rejected
        and multiplexed_rejected
        and wrong_cpu_rejected
        and failure_sealed
        and existing_root_rejected_without_write
        and c_existing_capture_root_rejected_without_write
    )
    result["self_test_sha256"] = public.digest({k: v for k, v in result.items() if k != "self_test_sha256"})
    write_json(output_root / "INDEPENDENT_WINDOW_TARGET_SELF_TEST.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=HERE)
    parser.add_argument("--output-root", type=Path, default=HERE)
    parser.add_argument("--run-id", default=public.RUN_ID)
    parser.add_argument("--expected-manifest-sha")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--self-test", action="store_true")
    modes.add_argument("--offline-validate", action="store_true")
    modes.add_argument("--execute-live", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.self_test:
            with tempfile.TemporaryDirectory(prefix="independent_window_v3_target_self_") as temp:
                result = self_test(args.source_root.resolve(), Path(temp))
            ok = result["self_test_passed"]
        elif args.offline_validate:
            result = offline_validate(args.source_root.resolve(), args.output_root.resolve())
            ok = result["passed"]
        else:
            require(args.expected_manifest_sha is not None, "--execute-live requires --expected-manifest-sha")
            result = execute_live(
                args.source_root.resolve(),
                args.output_root.resolve(),
                run_id=args.run_id,
                expected_manifest_sha=args.expected_manifest_sha,
            )
            ok = result["final"]["status"] == "INDEPENDENT_WINDOW_V3_TARGET_COMPLETE"
        print(json.dumps(result, sort_keys=True))
        return 0 if ok else 1
    except Exception as exc:
        print(f"independent_window_target: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
