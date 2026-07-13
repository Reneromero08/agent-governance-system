#!/usr/bin/env python3
"""Lab-side target wrapper for the frozen Confirmation V2 package."""

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
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import confirmation_v2_public as public


TEMPERATURE_VETO_C = 68.0
PHYSICALLY_PLAUSIBLE_TEMP_C = (0.0, 125.0)
FORBIDDEN_PROCESS_MARKERS = (
    "confirmation_v2_runtime",
    "balanced_transducer_runtime",
    "orbit_query_runtime",
    "f10_pmc_first_light_worker",
    "gate_a_worker_live",
    "combined_pdn_runner",
    "run_combined_campaign",
)
PROCESS_SCAN_COMMAND = ("ps", "-eo", "pid=,comm=,args=")
POLICY_IDS = (4, 5)
PMU_EVENT_FORMAT = "config:0-7,32-35"
PMU_UMASK_FORMAT = "config:8-15"
PERF_EVENT_PARANOID_ALLOWED_MAX = 2
CPU_EVENT_SOURCE_TYPE_MIN = 0
BINARY_CUSTODY_MODE = "target_compile_bound_by_source_and_compiler_contract"
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
    "CONFIRMATION_CONTRACT_V2.md",
    "ADJUDICATION_LAW_AUDIT.md",
    "CONFIRMATION_PUBLIC_TRIAL_SCHEDULE.json",
    "CONFIRMATION_PUBLIC_TRIAL_SCHEDULE.sha256",
    "CONFIRMATION_PUBLIC_TRIAL_SCHEDULE.tsv",
    "confirmation_v2_public.py",
    "confirmation_v2_runtime.c",
    "confirmation_v2_runtime.h",
    "confirmation_v2_target.py",
    "run_confirmation_v2.py",
    "balanced_transducer_adjudication_v2.py",
    "balanced_transducer_public.py",
)


class TargetError(RuntimeError):
    pass


@contextmanager
def tempfile_dir(parent: Path, name: str):
    path = parent / name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def require(value: bool, message: str) -> None:
    if not value:
        raise TargetError(message)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    return public.sha256_file(path)


def manifest_digest(manifest: dict[str, Any]) -> str:
    return public.digest({k: v for k, v in manifest.items() if k != "implementation_manifest_sha256"})


def source_hashes(source_root: Path) -> dict[str, str]:
    return {
        name: sha256_file(source_root / name)
        for name in SOURCE_FILES
        if (source_root / name).is_file()
    }


def build_source_bundle(source_root: Path, output_root: Path) -> tuple[str, dict[str, str]]:
    hashes = source_hashes(source_root)
    require(set(SOURCE_FILES).issubset(hashes), "source bundle file set incomplete")
    bundle_path = output_root / "CONFIRMATION_SOURCE_BUNDLE.tar.gz"
    with bundle_path.open("wb") as raw:
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
    write_json(output_root / "CONFIRMATION_SOURCE_HASHES.json", {"schema_id": "CAT_CAS_CONFIRMATION_V2_SOURCE_HASHES", "files": hashes})
    return sha256_file(bundle_path), hashes


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def prepare_schedule_artifacts(source_root: Path, output_root: Path) -> tuple[dict[str, Any], Path, str, str]:
    schedule_path = source_root / "CONFIRMATION_PUBLIC_TRIAL_SCHEDULE.json"
    tsv_path = source_root / "CONFIRMATION_PUBLIC_TRIAL_SCHEDULE.tsv"
    sha_path = source_root / "CONFIRMATION_PUBLIC_TRIAL_SCHEDULE.sha256"
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    public.validate_schedule(schedule)
    require(sha256_file(schedule_path) == read_text(sha_path), "schedule sha file mismatch")
    require(tsv_path.read_text(encoding="utf-8") == public.schedule_tsv(schedule), "schedule TSV mismatch")
    shutil.copy2(schedule_path, output_root / "CONFIRMATION_PUBLIC_TRIAL_SCHEDULE.json")
    shutil.copy2(tsv_path, output_root / "CONFIRMATION_PUBLIC_TRIAL_SCHEDULE.tsv")
    shutil.copy2(sha_path, output_root / "CONFIRMATION_PUBLIC_TRIAL_SCHEDULE.sha256")
    return schedule, output_root / "CONFIRMATION_PUBLIC_TRIAL_SCHEDULE.tsv", sha256_file(schedule_path), sha256_file(tsv_path)


def runtime_command(binary: Path, schedule_tsv: Path, batch_root: Path, replicate: int) -> list[str]:
    return [
        str(binary),
        "--schedule-tsv",
        str(schedule_tsv),
        "--output-root",
        str(batch_root),
        "--replicate",
        str(replicate),
    ]


def compile_runtime(source_root: Path, binary: Path) -> tuple[list[str], str]:
    command = [
        "cc",
        *STRICT_C_FLAGS,
        "-I",
        str(source_root),
        str(source_root / "confirmation_v2_runtime.c"),
        "-o",
        str(binary),
    ]
    subprocess.run(command, check=True, timeout=30)
    return command, sha256_file(binary)


def run_runtime_self_test(binary: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [str(binary), "--self-test"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=10,
        check=False,
    )
    return {
        "command": [str(binary), "--self-test"],
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "passed": completed.returncode == 0 and "CONFIRMATION_V2_RUNTIME_SELF_TEST_OK" in completed.stdout,
    }


def combine_batch_jsonl(output_root: Path, batch_roots: list[Path], name: str) -> list[dict[str, Any]]:
    combined: list[dict[str, Any]] = []
    with (output_root / name).open("w", encoding="utf-8", newline="\n") as out:
        for batch in batch_roots:
            for line in (batch / name).read_text(encoding="utf-8").splitlines():
                if line:
                    out.write(line + "\n")
                    combined.append(json.loads(line))
    return combined


def build_copyback_manifest(output_root: Path) -> dict[str, Any]:
    files = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name != "COPYBACK_MANIFEST.json":
            rel = path.relative_to(output_root).as_posix()
            files.append({"path": rel, "size": path.stat().st_size, "sha256": sha256_file(path)})
    manifest = {"schema_id": "CAT_CAS_CONFIRMATION_V2_COPYBACK_MANIFEST", "files": files}
    write_json(output_root / "COPYBACK_MANIFEST.json", manifest)
    return manifest


def build_execution_manifest(
    output_root: Path,
    *,
    run_id: str,
    implementation_manifest_sha256: str | None,
    source_bundle_sha: str,
    schedule_json_sha: str,
    schedule_tsv_sha: str,
    live_runtime_binary_sha: str,
    offline_validation_binary_sha: str | None,
    raw_count: int,
    sentinel_count: int,
) -> dict[str, Any]:
    manifest = {
        "schema_id": "CAT_CAS_CONFIRMATION_V2_MANIFEST",
        "run_id": run_id,
        "implementation_manifest_sha256": implementation_manifest_sha256,
        "source_bundle_sha256": source_bundle_sha,
        "schedule_json_sha256": schedule_json_sha,
        "schedule_tsv_sha256": schedule_tsv_sha,
        "binary_custody_mode": BINARY_CUSTODY_MODE,
        "offline_validation_binary_sha256": offline_validation_binary_sha,
        "live_runtime_binary_sha256": live_runtime_binary_sha,
        "raw_capture_sha256": sha256_file(output_root / "RAW_TRANSDUCER_CAPTURE.jsonl"),
        "restoration_sentinels_sha256": sha256_file(output_root / "RESTORATION_SENTINELS.jsonl"),
        "features_sha256": sha256_file(output_root / "TRANSDUCER_FEATURES_V2.json"),
        "adjudication_sha256": sha256_file(output_root / "TRANSDUCER_ADJUDICATION_CONFIRMATION_V2.json"),
        "final_result_sha256": sha256_file(output_root / "FINAL_RESULT_CONFIRMATION_V2.json"),
        "raw_record_count": raw_count,
        "sentinel_record_count": sentinel_count,
        "allowed_classifications": list(public.ALLOWED_CLASSES),
        "forbidden_classifications": list(public.FORBIDDEN_CLASSES),
        "primary_coordinate": public.PRIMARY_COORDINATE,
    }
    manifest["manifest_sha256"] = public.digest({k: v for k, v in manifest.items() if k != "manifest_sha256"})
    write_json(output_root / "CONFIRMATION_V2_MANIFEST.json", manifest)
    return manifest


def _now_ns() -> int:
    return time.monotonic_ns()


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


def _process_marker_matches(row: dict[str, Any]) -> list[str]:
    identities = _identity_candidates(row)
    matches = []
    for marker in FORBIDDEN_PROCESS_MARKERS:
        if marker in identities:
            matches.append(marker)
    return matches


def process_snapshot(
    phase: str,
    *,
    runner: Any = subprocess.run,
    timeout: float = 5.0,
) -> dict[str, Any]:
    completed = runner(
        list(PROCESS_SCAN_COMMAND),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        raise TargetError(f"process scan failed during {phase}: {completed.stderr.strip()}")
    rows = [row for row in (_parse_ps_line(line) for line in completed.stdout.splitlines()) if row is not None]
    matches = []
    for row in rows:
        markers = _process_marker_matches(row)
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


def cpuinfo_snapshot() -> dict[str, str]:
    info: dict[str, str] = {}
    path = Path("/proc/cpuinfo")
    if path.is_file():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                info.setdefault(key.strip(), value.strip())
    return info


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
    if temp_c < low or temp_c > high:
        raise TargetError(f"implausible k10temp value during {phase}: {temp_c}")
    observation = {
        "phase": phase,
        "timestamp_monotonic_ns": _now_ns(),
        "hwmon_path": str(candidates[0]),
        "input_path": str(temp_path),
        "raw_millidegrees_c": raw,
        "temperature_c": temp_c,
        "veto_c": TEMPERATURE_VETO_C,
        "below_veto": temp_c < TEMPERATURE_VETO_C,
    }
    return observation


def require_temperature_below(observation: dict[str, Any]) -> None:
    require(bool(observation["below_veto"]), f"temperature veto during {observation['phase']}: {observation['temperature_c']}")


def policy_snapshot(
    phase: str,
    *,
    cpufreq_root: Path = Path("/sys/devices/system/cpu/cpufreq"),
    policy_ids: tuple[int, ...] = POLICY_IDS,
) -> dict[str, Any]:
    policies: dict[str, Any] = {}
    for policy_id in policy_ids:
        path = cpufreq_root / f"policy{policy_id}"
        if not path.is_dir():
            raise TargetError(f"cpufreq policy missing: {path}")
        affected = _read_optional(path / "affected_cpus")
        related = _read_optional(path / "related_cpus")
        membership = affected or related
        if not membership:
            raise TargetError(f"policy{policy_id} has no CPU membership field")
        members = {int(value) for value in membership.split() if value.isdigit()}
        if policy_id not in members:
            raise TargetError(f"policy{policy_id} membership does not include CPU {policy_id}")
        driver = _read_required(path / "scaling_driver")
        policies[str(policy_id)] = {
            "policy_id": policy_id,
            "resolved_policy_path": str(path.resolve()),
            "scaling_min_freq": _read_required(path / "scaling_min_freq"),
            "scaling_max_freq": _read_required(path / "scaling_max_freq"),
            "scaling_cur_freq": _read_required(path / "scaling_cur_freq"),
            "affected_cpus": affected,
            "related_cpus": related,
            "cpu_membership": sorted(members),
            "scaling_driver": driver,
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
    return {
        "passed": not failures,
        "frequency_writes": 0,
        "failures": failures,
        "scaling_cur_freq_diagnostic_only": True,
    }


def pmu_platform_snapshot(
    *,
    cpuinfo: dict[str, str] | None = None,
    event_source_root: Path = Path("/sys/bus/event_source/devices/cpu"),
    cpu_root: Path = Path("/sys/devices/system/cpu"),
    perf_event_paranoid: Path = Path("/proc/sys/kernel/perf_event_paranoid"),
) -> dict[str, Any]:
    info = cpuinfo or cpuinfo_snapshot()
    format_dir = event_source_root / "format"
    formats = {
        path.name: _read_required(path)
        for path in sorted(format_dir.iterdir())
        if path.is_file()
    }
    cores = {}
    for core in (4, 5):
        online_path = cpu_root / f"cpu{core}" / "online"
        cores[str(core)] = {"online_path": str(online_path), "online": _read_required(online_path)}
    event_source_type = _read_required(event_source_root / "type")
    paranoid_raw = _read_required(perf_event_paranoid)
    snapshot = {
        "timestamp_monotonic_ns": _now_ns(),
        "cpuinfo": {
            "vendor_id": info.get("vendor_id"),
            "cpu_family": info.get("cpu family"),
            "model": info.get("model"),
            "model_name": info.get("model name"),
            "flags": info.get("flags"),
        },
        "event_source_cpu_type": event_source_type,
        "format_dir": str(format_dir),
        "formats": formats,
        "perf_event_paranoid": paranoid_raw,
        "perf_event_paranoid_allowed_max": PERF_EVENT_PARANOID_ALLOWED_MAX,
        "cores": cores,
        "msr_reads": 0,
        "msr_writes": 0,
    }
    failures = []
    if snapshot["cpuinfo"]["vendor_id"] != "AuthenticAMD":
        failures.append("CPU vendor mismatch")
    if snapshot["cpuinfo"]["cpu_family"] != "16":
        failures.append("CPU family mismatch")
    if formats.get("event") != PMU_EVENT_FORMAT:
        failures.append("PMU event format mismatch")
    if formats.get("umask") != PMU_UMASK_FORMAT:
        failures.append("PMU umask format mismatch")
    try:
        if int(event_source_type) < CPU_EVENT_SOURCE_TYPE_MIN:
            failures.append("PMU event source type negative")
    except ValueError:
        failures.append("PMU event source type is not numeric")
    try:
        if int(paranoid_raw) > PERF_EVENT_PARANOID_ALLOWED_MAX:
            failures.append("perf_event_paranoid too restrictive")
    except ValueError:
        failures.append("perf_event_paranoid is not numeric")
    for core, core_state in cores.items():
        if core_state["online"] != "1":
            failures.append(f"CPU core {core} is not online")
    snapshot["passed"] = not failures
    snapshot["failures"] = failures
    if failures:
        raise TargetError("; ".join(failures))
    return snapshot


def write_custody_log(output_root: Path, custody: dict[str, Any]) -> None:
    write_json(output_root / "LIVE_CUSTODY_LOG.json", custody)


def append_temperature(custody: dict[str, Any], phase: str) -> dict[str, Any]:
    observation = temperature_observation(phase)
    custody["temperature_observations"].append(observation)
    require_temperature_below(observation)
    return observation


def append_process_snapshot(custody: dict[str, Any], phase: str, *, reject_forbidden: bool) -> dict[str, Any]:
    snapshot = process_snapshot(phase)
    custody["process_snapshots"].append(snapshot)
    if reject_forbidden:
        require_no_forbidden_processes(snapshot)
    return snapshot


def failure_evidence(
    output_root: Path,
    *,
    state: dict[str, Any],
    exc: BaseException,
) -> list[dict[str, str]]:
    errors: list[dict[str, str]] = []
    try:
        if state.get("hardware_execution_began"):
            try:
                append_process_snapshot(state["custody"], "exception_cleanup", reject_forbidden=False)
            except Exception as cleanup_exc:  # failure evidence must not mask the original error
                state["custody"].setdefault("cleanup_errors", []).append(
                    {"type": type(cleanup_exc).__name__, "message": str(cleanup_exc)}
                )
        failure = {
            "schema_id": "CAT_CAS_CONFIRMATION_V2_TARGET_FAILURE",
            "status": "CONFIRMATION_V2_TARGET_FAILED",
            "run_id": state.get("run_id"),
            "failure_phase": state.get("phase", "unknown"),
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "temperature_observations": state["custody"].get("temperature_observations", []),
            "process_snapshots": state["custody"].get("process_snapshots", []),
            "policy_snapshots": state["custody"].get("policy_snapshots", []),
            "source_and_schedule_hashes_verified_so_far": state.get("verified_hashes", {}),
            "hardware_execution_began": bool(state.get("hardware_execution_began")),
            "replicates": state.get("replicates", {}),
            "scientific_classification_emitted": False,
        }
        write_json(output_root / "TARGET_FAILURE_CONFIRMATION_V2.json", failure)
        failure_manifest = {
            "schema_id": "CAT_CAS_CONFIRMATION_V2_FAILURE_MANIFEST",
            "run_id": state.get("run_id"),
            "target_failure_sha256": sha256_file(output_root / "TARGET_FAILURE_CONFIRMATION_V2.json"),
            "scientific_classification_emitted": False,
        }
        failure_manifest["manifest_sha256"] = public.digest({k: v for k, v in failure_manifest.items() if k != "manifest_sha256"})
        write_json(output_root / "CONFIRMATION_V2_FAILURE_MANIFEST.json", failure_manifest)
        final = {
            "schema_id": "CAT_CAS_BALANCED_TRANSDUCER_CONFIRMATION_TARGET_RESULT_V2",
            "status": "CONFIRMATION_V2_TARGET_FAILED",
            "run_id": state.get("run_id"),
            "failure_sha256": sha256_file(output_root / "TARGET_FAILURE_CONFIRMATION_V2.json"),
            "scientific_classification_emitted": False,
        }
        write_json(output_root / "FINAL_RESULT_CONFIRMATION_V2.json", final)
        write_custody_log(output_root, state["custody"])
        build_copyback_manifest(output_root)
    except Exception as evidence_exc:
        errors.append({"type": type(evidence_exc).__name__, "message": str(evidence_exc)})
    return errors


def execute(source_root: Path, output_root: Path, *, run_id: str, expected_manifest_sha: str | None = None) -> dict[str, Any]:
    require(not output_root.exists(), f"output root already exists: {output_root}")
    output_root.mkdir(mode=0o700, parents=True, exist_ok=False)
    start_ns = time.monotonic_ns()
    custody: dict[str, Any] = {
        "schema_id": "CAT_CAS_CONFIRMATION_V2_LIVE_CUSTODY_LOG",
        "run_id": run_id,
        "process_snapshots": [],
        "temperature_observations": [],
        "policy_snapshots": [],
        "events": [],
        "frequency_writes": 0,
        "voltage_writes": 0,
        "msr_reads": 0,
        "msr_writes": 0,
    }
    state: dict[str, Any] = {
        "run_id": run_id,
        "phase": "init",
        "custody": custody,
        "verified_hashes": {},
        "hardware_execution_began": False,
        "replicates": {str(rep): {"began": False, "completed": False} for rep in public.REPLICATES},
    }
    try:
        state["phase"] = "identity"
        require(run_id == public.RUN_ID, "run ID mismatch")

        state["phase"] = "process_preflight"
        append_process_snapshot(custody, "before_execution", reject_forbidden=True)
        write_custody_log(output_root, custody)

        state["phase"] = "policy_before"
        policy_before = policy_snapshot("before")
        custody["policy_snapshots"].append(policy_before)
        write_custody_log(output_root, custody)

        state["phase"] = "pmu_platform"
        cpuinfo = cpuinfo_snapshot()
        pmu = pmu_platform_snapshot(cpuinfo=cpuinfo)
        custody["pmu_platform"] = pmu
        write_custody_log(output_root, custody)

        state["phase"] = "temperature_before_compilation"
        append_temperature(custody, "before_compilation")
        write_custody_log(output_root, custody)

        state["phase"] = "source_schedule_verification"
        schedule, schedule_tsv, schedule_json_sha, schedule_tsv_sha = prepare_schedule_artifacts(source_root, output_root)
        source_bundle_sha, source_hashes_map = build_source_bundle(source_root, output_root)
        state["verified_hashes"].update(
            {
                "schedule_json_sha256": schedule_json_sha,
                "schedule_tsv_sha256": schedule_tsv_sha,
                "source_bundle_sha256": source_bundle_sha,
            }
        )
        implementation_manifest_sha: str | None = None
        offline_validation_binary_sha: str | None = None
        if expected_manifest_sha:
            manifest_path = source_root / "CONFIRMATION_V2_IMPLEMENTATION_MANIFEST.json"
            require(manifest_path.is_file(), "implementation manifest missing")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            recomputed_manifest_sha = manifest_digest(manifest)
            require(manifest["implementation_manifest_sha256"] == expected_manifest_sha, "implementation manifest self SHA mismatch")
            require(recomputed_manifest_sha == expected_manifest_sha, "implementation manifest recomputed SHA mismatch")
            require("expected_runtime_binary_sha256" not in manifest, "obsolete expected runtime binary field present")
            require(manifest["binary_custody"]["mode"] == BINARY_CUSTODY_MODE, "binary custody mode mismatch")
            require(source_bundle_sha == manifest["expected_source_bundle_sha256"], "source bundle SHA mismatch")
            require(schedule_json_sha == manifest["schedule_json_sha256"], "schedule JSON SHA mismatch")
            require(schedule_tsv_sha == manifest["schedule_tsv_sha256"], "schedule TSV SHA mismatch")
            require(source_hashes_map == manifest["source_hashes"], "source hash manifest mismatch")
            implementation_manifest_sha = expected_manifest_sha
            offline_validation_binary_sha = manifest["binary_custody"]["offline_validation_binary_sha256"]
            state["verified_hashes"]["implementation_manifest_sha256"] = expected_manifest_sha

        state["phase"] = "compile_runtime"
        binary = source_root / "confirmation_v2_runtime"
        compile_command, live_runtime_binary_sha = compile_runtime(source_root, binary)
        runtime_self_test = run_runtime_self_test(binary)
        require(runtime_self_test["passed"], "target runtime binary self-test failed")
        custody["events"].append(
            {
                "phase": "compile_runtime",
                "compile_command": compile_command,
                "binary_custody_mode": BINARY_CUSTODY_MODE,
                "offline_validation_binary_sha256": offline_validation_binary_sha,
                "live_runtime_binary_sha256": live_runtime_binary_sha,
                "runtime_self_test": runtime_self_test,
            }
        )
        write_custody_log(output_root, custody)

        batch_roots = []
        runtime_results = []
        for rep in public.REPLICATES:
            state["phase"] = f"process_before_replicate_{rep}"
            append_process_snapshot(custody, f"before_replicate_{rep}", reject_forbidden=True)
            write_custody_log(output_root, custody)

            state["phase"] = f"temperature_before_replicate_{rep}"
            append_temperature(custody, f"before_replicate_{rep}")
            write_custody_log(output_root, custody)

            state["phase"] = f"runtime_replicate_{rep}"
            state["hardware_execution_began"] = True
            state["replicates"][str(rep)]["began"] = True
            batch_root = output_root / f"batch_{rep}"
            command = runtime_command(binary, schedule_tsv, batch_root, rep)
            completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120, check=False)
            (output_root / f"CONFIRMATION_RUNTIME_STDOUT_REPLICATE_{rep}.txt").write_text(completed.stdout, encoding="utf-8")
            (output_root / f"CONFIRMATION_RUNTIME_STDERR_REPLICATE_{rep}.txt").write_text(completed.stderr, encoding="utf-8")
            require(completed.returncode == 0, f"runtime replicate {rep} failed")
            state["replicates"][str(rep)]["completed"] = True
            runtime_results.append({"replicate": rep, "returncode": completed.returncode, "command": command})
            batch_roots.append(batch_root)

            state["phase"] = f"temperature_after_replicate_{rep}"
            append_temperature(custody, f"after_replicate_{rep}")
            write_custody_log(output_root, custody)

            state["phase"] = f"process_after_replicate_{rep}"
            append_process_snapshot(custody, f"after_replicate_{rep}", reject_forbidden=True)
            write_custody_log(output_root, custody)

        state["phase"] = "process_post_replicates"
        append_process_snapshot(custody, "after_replicates", reject_forbidden=True)
        write_custody_log(output_root, custody)

        state["phase"] = "policy_after"
        policy_after = policy_snapshot("after")
        custody["policy_snapshots"].append(policy_after)
        policy_comparison = compare_policy_snapshots(policy_before, policy_after)
        custody["policy_comparison"] = policy_comparison
        require(policy_comparison["passed"], "CPU policy changed during confirmation")
        write_custody_log(output_root, custody)

        state["phase"] = "adjudication"
        raw_records = combine_batch_jsonl(output_root, batch_roots, "RAW_TRANSDUCER_CAPTURE.jsonl")
        sentinels = combine_batch_jsonl(output_root, batch_roots, "RESTORATION_SENTINELS.jsonl")
        require(len(raw_records) == public.TOTAL_TRIALS, "raw record count mismatch")
        require(len(sentinels) == public.TOTAL_TRIALS, "sentinel record count mismatch")
        features = public.extract_features(schedule, raw_records, sentinels)
        adjudication = public.adjudicate(features)
        write_json(output_root / "TRANSDUCER_FEATURES_V2.json", features)
        write_json(output_root / "TRANSDUCER_ADJUDICATION_CONFIRMATION_V2.json", adjudication)

        state["phase"] = "temperature_before_final_success"
        append_temperature(custody, "before_final_success")
        write_custody_log(output_root, custody)

        state["phase"] = "final_success"
        final = {
            "schema_id": "CAT_CAS_BALANCED_TRANSDUCER_CONFIRMATION_TARGET_RESULT_V2",
            "status": "CONFIRMATION_V2_TARGET_COMPLETE",
            "run_id": run_id,
            "adjudication_status": adjudication["status"],
            "primary_coordinate": public.PRIMARY_COORDINATE,
            "schedule_json_sha256": schedule_json_sha,
            "schedule_tsv_sha256": schedule_tsv_sha,
            "source_bundle_sha256": source_bundle_sha,
            "source_hashes": source_hashes_map,
            "compile_command": compile_command,
            "binary_custody_mode": BINARY_CUSTODY_MODE,
            "offline_validation_binary_sha256": offline_validation_binary_sha,
            "live_runtime_binary_sha256": live_runtime_binary_sha,
            "runtime_binary_sha256": live_runtime_binary_sha,
            "runtime_binary_self_test": runtime_self_test,
            "raw_capture_sha256": sha256_file(output_root / "RAW_TRANSDUCER_CAPTURE.jsonl"),
            "restoration_sentinels_sha256": sha256_file(output_root / "RESTORATION_SENTINELS.jsonl"),
            "features_sha256": features["features_sha256"],
            "adjudication_sha256": adjudication["adjudication_sha256"],
            "raw_record_count": len(raw_records),
            "sentinel_record_count": len(sentinels),
            "runtime_results": runtime_results,
            "cpuinfo": cpuinfo,
            "pmu_platform": pmu,
            "temperature_observations": custody["temperature_observations"],
            "process_snapshots": custody["process_snapshots"],
            "policy_snapshots": custody["policy_snapshots"],
            "policy_comparison": policy_comparison,
            "frequency_writes": 0,
            "voltage_writes": 0,
            "msr_reads": 0,
            "msr_writes": 0,
            "physical_address_access": False,
            "cache_set_mapping": False,
            "runtime_duration_ns": time.monotonic_ns() - start_ns,
        }
        write_json(output_root / "FINAL_RESULT_CONFIRMATION_V2.json", final)
        execution_manifest = build_execution_manifest(
            output_root,
            run_id=run_id,
            implementation_manifest_sha256=implementation_manifest_sha,
            source_bundle_sha=source_bundle_sha,
            schedule_json_sha=schedule_json_sha,
            schedule_tsv_sha=schedule_tsv_sha,
            live_runtime_binary_sha=live_runtime_binary_sha,
            offline_validation_binary_sha=offline_validation_binary_sha,
            raw_count=len(raw_records),
            sentinel_count=len(sentinels),
        )
        manifest = build_copyback_manifest(output_root)
        write_custody_log(output_root, custody)
        return {"final": final, "execution_manifest": execution_manifest, "copyback_manifest": manifest}
    except Exception as exc:
        evidence_errors = failure_evidence(output_root, state=state, exc=exc)
        if evidence_errors:
            raise TargetError(f"{exc}; failure evidence sealing errors: {evidence_errors}") from exc
        raise


def self_test(source_root: Path, output_root: Path) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    schedule_hashes = public.write_schedule_artifacts(source_root)
    public_self = public.self_test()
    ideal = public.run_case("ideal_direct")
    zero = public.run_case("zero_transfer")

    def completed(stdout: str, returncode: int = 0, stderr: str = "") -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(list(PROCESS_SCAN_COMMAND), returncode, stdout=stdout, stderr=stderr)

    synthetic_forbidden_detected = False
    interpreter_forbidden_detected = False
    wrapper_source_argument_not_false_positive = False
    try:
        snap = process_snapshot(
            "self_test_forbidden",
            runner=lambda *args, **kwargs: completed("123 confirmation_v2 /tmp/confirmation_v2_runtime --self-test\n"),
        )
        require_no_forbidden_processes(snap)
    except Exception:
        synthetic_forbidden_detected = True
    try:
        snap = process_snapshot(
            "self_test_interpreter_forbidden",
            runner=lambda *args, **kwargs: completed("124 python3 python3 /tmp/run_combined_campaign.py --arg\n"),
        )
        require_no_forbidden_processes(snap)
    except Exception:
        interpreter_forbidden_detected = True
    try:
        snap = process_snapshot(
            "self_test_wrapper_argument",
            runner=lambda *args, **kwargs: completed("125 python3 python3 /tmp/confirmation_v2_target.py /tmp/confirmation_v2_runtime.c\n"),
        )
        require_no_forbidden_processes(snap)
        wrapper_source_argument_not_false_positive = True
    except Exception:
        wrapper_source_argument_not_false_positive = False

    failed_process_scan_fails_closed = False
    try:
        process_snapshot("self_test_failed_ps", runner=lambda *args, **kwargs: completed("", returncode=1, stderr="ps failed"))
    except Exception:
        failed_process_scan_fails_closed = True

    def write_hwmon(root: Path, entries: list[tuple[str, str]]) -> None:
        for index, (name, value) in enumerate(entries):
            hwmon = root / f"hwmon{index}"
            hwmon.mkdir(parents=True)
            (hwmon / "name").write_text(name + "\n", encoding="utf-8")
            if value != "MISSING":
                (hwmon / "temp1_input").write_text(value + "\n", encoding="utf-8")

    missing_k10temp_fails_closed = False
    ambiguous_k10temp_fails_closed = False
    malformed_temperature_fails_closed = False
    temperature_veto_blocks = False
    post_replicate_temperature_veto_blocks = False
    with tempfile_dir(output_root, "temp_tests") as temp_root:
        try:
            write_hwmon(temp_root / "missing", [("acpitz", "42000")])
            temperature_observation("missing", hwmon_root=temp_root / "missing")
        except Exception:
            missing_k10temp_fails_closed = True
        try:
            write_hwmon(temp_root / "ambiguous", [("k10temp", "42000"), ("k10temp", "43000")])
            temperature_observation("ambiguous", hwmon_root=temp_root / "ambiguous")
        except Exception:
            ambiguous_k10temp_fails_closed = True
        try:
            write_hwmon(temp_root / "malformed", [("k10temp", "not-a-number")])
            temperature_observation("malformed", hwmon_root=temp_root / "malformed")
        except Exception:
            malformed_temperature_fails_closed = True
        try:
            write_hwmon(temp_root / "veto", [("k10temp", "68000")])
            require_temperature_below(temperature_observation("before_replicate_0", hwmon_root=temp_root / "veto"))
        except Exception:
            temperature_veto_blocks = True
        try:
            require_temperature_below({"phase": "after_replicate_0", "temperature_c": 68.0, "below_veto": False})
        except Exception:
            post_replicate_temperature_veto_blocks = True

    policy_identity_drift_rejected = False
    policy_minmax_drift_rejected = False
    policy_cur_freq_variation_allowed = False
    before = {
        "policies": {
            "4": {"resolved_policy_path": "/p4", "affected_cpus": "4", "related_cpus": "4", "cpu_membership": [4], "scaling_driver": "acpi", "scaling_min_freq": "100", "scaling_max_freq": "200", "scaling_cur_freq": "150"},
            "5": {"resolved_policy_path": "/p5", "affected_cpus": "5", "related_cpus": "5", "cpu_membership": [5], "scaling_driver": "acpi", "scaling_min_freq": "100", "scaling_max_freq": "200", "scaling_cur_freq": "150"},
        }
    }
    after_cur = json.loads(json.dumps(before))
    after_cur["policies"]["4"]["scaling_cur_freq"] = "175"
    policy_cur_freq_variation_allowed = compare_policy_snapshots(before, after_cur)["passed"]
    after_identity = json.loads(json.dumps(before))
    after_identity["policies"]["4"]["affected_cpus"] = "4 6"
    policy_identity_drift_rejected = not compare_policy_snapshots(before, after_identity)["passed"]
    after_minmax = json.loads(json.dumps(before))
    after_minmax["policies"]["5"]["scaling_max_freq"] = "201"
    policy_minmax_drift_rejected = not compare_policy_snapshots(before, after_minmax)["passed"]

    missing_policy_membership_rejected = False
    missing_policy_driver_rejected = False
    with tempfile_dir(output_root, "policy_tests") as policy_root:
        for policy_id in POLICY_IDS:
            policy = policy_root / f"policy{policy_id}"
            policy.mkdir(parents=True)
            (policy / "scaling_min_freq").write_text("100\n", encoding="utf-8")
            (policy / "scaling_max_freq").write_text("200\n", encoding="utf-8")
            (policy / "scaling_cur_freq").write_text("150\n", encoding="utf-8")
            (policy / "affected_cpus").write_text(f"{policy_id}\n", encoding="utf-8")
        try:
            policy_snapshot("missing_driver", cpufreq_root=policy_root)
        except Exception:
            missing_policy_driver_rejected = True
        for policy_id in POLICY_IDS:
            (policy_root / f"policy{policy_id}" / "scaling_driver").write_text("acpi\n", encoding="utf-8")
        (policy_root / "policy4" / "affected_cpus").unlink()
        try:
            policy_snapshot("missing_membership", cpufreq_root=policy_root)
        except Exception:
            missing_policy_membership_rejected = True

    def fake_pmu_root(
        root: Path,
        *,
        event: str = PMU_EVENT_FORMAT,
        umask: str = PMU_UMASK_FORMAT,
        event_type: str = "4",
        paranoid_value: str = "2",
        core4_online: str = "1",
        core5_online: str = "1",
    ) -> dict[str, Path]:
        event_root = root / "event_source" / "cpu"
        fmt = event_root / "format"
        fmt.mkdir(parents=True)
        (event_root / "type").write_text(event_type + "\n", encoding="utf-8")
        (fmt / "event").write_text(event + "\n", encoding="utf-8")
        (fmt / "umask").write_text(umask + "\n", encoding="utf-8")
        cpu_root = root / "cpu"
        for core in (4, 5):
            core_dir = cpu_root / f"cpu{core}"
            core_dir.mkdir(parents=True)
            value = core4_online if core == 4 else core5_online
            (core_dir / "online").write_text(value + "\n", encoding="utf-8")
        paranoid_path = root / "perf_event_paranoid"
        paranoid_path.write_text(paranoid_value + "\n", encoding="utf-8")
        return {"event": event_root, "cpu": cpu_root, "paranoid": paranoid_path}

    pmu_event_format_drift_rejected = False
    pmu_umask_format_drift_rejected = False
    pmu_type_drift_rejected = False
    paranoid_drift_rejected = False
    cpu_vendor_drift_rejected = False
    cpu_family_drift_rejected = False
    core_offline_rejected = False
    with tempfile_dir(output_root, "pmu_tests") as pmu_root:
        try:
            paths = fake_pmu_root(pmu_root / "event_bad", event="config:0-7")
            pmu_platform_snapshot(
                cpuinfo={"vendor_id": "AuthenticAMD", "cpu family": "16"},
                event_source_root=paths["event"],
                cpu_root=paths["cpu"],
                perf_event_paranoid=paths["paranoid"],
            )
        except Exception:
            pmu_event_format_drift_rejected = True
        try:
            paths = fake_pmu_root(pmu_root / "umask_bad", umask="config:8-14")
            pmu_platform_snapshot(
                cpuinfo={"vendor_id": "AuthenticAMD", "cpu family": "16"},
                event_source_root=paths["event"],
                cpu_root=paths["cpu"],
                perf_event_paranoid=paths["paranoid"],
            )
        except Exception:
            pmu_umask_format_drift_rejected = True
        try:
            paths = fake_pmu_root(pmu_root / "type_bad", event_type="not-numeric")
            pmu_platform_snapshot(
                cpuinfo={"vendor_id": "AuthenticAMD", "cpu family": "16"},
                event_source_root=paths["event"],
                cpu_root=paths["cpu"],
                perf_event_paranoid=paths["paranoid"],
            )
        except Exception:
            pmu_type_drift_rejected = True
        try:
            paths = fake_pmu_root(pmu_root / "paranoid_bad", paranoid_value="3")
            pmu_platform_snapshot(
                cpuinfo={"vendor_id": "AuthenticAMD", "cpu family": "16"},
                event_source_root=paths["event"],
                cpu_root=paths["cpu"],
                perf_event_paranoid=paths["paranoid"],
            )
        except Exception:
            paranoid_drift_rejected = True
        try:
            paths = fake_pmu_root(pmu_root / "vendor_bad")
            pmu_platform_snapshot(
                cpuinfo={"vendor_id": "GenuineIntel", "cpu family": "16"},
                event_source_root=paths["event"],
                cpu_root=paths["cpu"],
                perf_event_paranoid=paths["paranoid"],
            )
        except Exception:
            cpu_vendor_drift_rejected = True
        try:
            paths = fake_pmu_root(pmu_root / "family_bad")
            pmu_platform_snapshot(
                cpuinfo={"vendor_id": "AuthenticAMD", "cpu family": "17"},
                event_source_root=paths["event"],
                cpu_root=paths["cpu"],
                perf_event_paranoid=paths["paranoid"],
            )
        except Exception:
            cpu_family_drift_rejected = True
        try:
            paths = fake_pmu_root(pmu_root / "core_bad", core5_online="0")
            pmu_platform_snapshot(
                cpuinfo={"vendor_id": "AuthenticAMD", "cpu family": "16"},
                event_source_root=paths["event"],
                cpu_root=paths["cpu"],
                perf_event_paranoid=paths["paranoid"],
            )
        except Exception:
            core_offline_rejected = True

    synthetic_files = {
        "RAW_TRANSDUCER_CAPTURE.jsonl": "{}\n",
        "RESTORATION_SENTINELS.jsonl": "{}\n",
        "TRANSDUCER_FEATURES_V2.json": json.dumps({"features_sha256": "synthetic"}, sort_keys=True) + "\n",
        "TRANSDUCER_ADJUDICATION_CONFIRMATION_V2.json": json.dumps({"adjudication_sha256": "synthetic"}, sort_keys=True) + "\n",
        "FINAL_RESULT_CONFIRMATION_V2.json": json.dumps({"status": "synthetic"}, sort_keys=True) + "\n",
    }
    for name, text in synthetic_files.items():
        (output_root / name).write_text(text, encoding="utf-8")
    execution_manifest = build_execution_manifest(
        output_root,
        run_id=public.RUN_ID,
        implementation_manifest_sha256="0" * 64,
        source_bundle_sha="1" * 64,
        schedule_json_sha=schedule_hashes["schedule_json_sha256"],
        schedule_tsv_sha=schedule_hashes["schedule_tsv_sha256"],
        live_runtime_binary_sha="2" * 64,
        offline_validation_binary_sha="3" * 64,
        raw_count=public.TOTAL_TRIALS,
        sentinel_count=public.TOTAL_TRIALS,
    )
    copyback_manifest = build_copyback_manifest(output_root)
    copyback_paths = {entry["path"] for entry in copyback_manifest["files"]}
    copyback_seals_final = "FINAL_RESULT_CONFIRMATION_V2.json" in copyback_paths
    copyback_seals_execution_manifest = "CONFIRMATION_V2_MANIFEST.json" in copyback_paths
    execution_manifest_seals_final = (
        execution_manifest["final_result_sha256"] == sha256_file(output_root / "FINAL_RESULT_CONFIRMATION_V2.json")
    )
    binary_hash_semantics_coherent = (
        execution_manifest["binary_custody_mode"] == BINARY_CUSTODY_MODE
        and execution_manifest["offline_validation_binary_sha256"] == "3" * 64
        and execution_manifest["live_runtime_binary_sha256"] == "2" * 64
    )
    target_binary_self_test_required = True

    failure_root = output_root / "failure_case"
    failure_root.mkdir(parents=True, exist_ok=True)
    failure_state = {
        "run_id": public.RUN_ID,
        "phase": "self_test_failure",
        "custody": {
            "temperature_observations": [{"phase": "before_compilation", "temperature_c": 42.0}],
            "process_snapshots": [],
            "policy_snapshots": [],
        },
        "verified_hashes": {"schedule_json_sha256": schedule_hashes["schedule_json_sha256"]},
        "hardware_execution_began": False,
        "replicates": {"0": {"began": False, "completed": False}, "1": {"began": False, "completed": False}},
    }
    failure_errors = failure_evidence(failure_root, state=failure_state, exc=TargetError("synthetic failure"))
    failure_final = json.loads((failure_root / "FINAL_RESULT_CONFIRMATION_V2.json").read_text(encoding="utf-8"))
    failure_evidence_sealing_succeeded = failure_errors == []
    failure_evidence_written_without_classification = (
        (failure_root / "TARGET_FAILURE_CONFIRMATION_V2.json").is_file()
        and (failure_root / "LIVE_CUSTODY_LOG.json").is_file()
        and (failure_root / "COPYBACK_MANIFEST.json").is_file()
        and failure_final.get("scientific_classification_emitted") is False
        and "adjudication_status" not in failure_final
    )
    result = {
        "schema_id": "CAT_CAS_CONFIRMATION_V2_TARGET_SELF_TEST",
        "schedule_hashes": schedule_hashes,
        "public_self_test_sha256": public_self["self_test_sha256"],
        "public_self_test_passed": public_self["self_test_passed"],
        "ideal_status": ideal["status"],
        "zero_status": zero["status"],
        "copyback_seals_final_result": copyback_seals_final,
        "copyback_seals_execution_manifest": copyback_seals_execution_manifest,
        "execution_manifest_seals_final_result": execution_manifest_seals_final,
        "synthetic_forbidden_process_detected": synthetic_forbidden_detected,
        "interpreter_forbidden_process_detected": interpreter_forbidden_detected,
        "wrapper_source_argument_not_false_positive": wrapper_source_argument_not_false_positive,
        "failed_process_scan_fails_closed": failed_process_scan_fails_closed,
        "missing_k10temp_fails_closed": missing_k10temp_fails_closed,
        "ambiguous_k10temp_fails_closed": ambiguous_k10temp_fails_closed,
        "malformed_temperature_fails_closed": malformed_temperature_fails_closed,
        "temperature_at_or_above_veto_blocks": temperature_veto_blocks,
        "post_replicate_temperature_veto_blocks_final_success": post_replicate_temperature_veto_blocks,
        "policy_identity_drift_rejected": policy_identity_drift_rejected,
        "policy_minmax_drift_rejected": policy_minmax_drift_rejected,
        "missing_policy_membership_rejected": missing_policy_membership_rejected,
        "missing_policy_driver_rejected": missing_policy_driver_rejected,
        "scaling_cur_freq_variation_allowed": policy_cur_freq_variation_allowed,
        "pmu_event_format_drift_rejected": pmu_event_format_drift_rejected,
        "pmu_umask_format_drift_rejected": pmu_umask_format_drift_rejected,
        "pmu_type_drift_rejected": pmu_type_drift_rejected,
        "perf_event_paranoid_drift_rejected": paranoid_drift_rejected,
        "cpu_vendor_drift_rejected": cpu_vendor_drift_rejected,
        "cpu_family_drift_rejected": cpu_family_drift_rejected,
        "core_offline_rejected": core_offline_rejected,
        "binary_hash_semantics_coherent": binary_hash_semantics_coherent,
        "target_binary_self_test_required": target_binary_self_test_required,
        "failure_evidence_sealing_succeeded": failure_evidence_sealing_succeeded,
        "failure_evidence_written_without_scientific_classification": failure_evidence_written_without_classification,
        "network_connections": 0,
        "ssh_executions": 0,
        "scp_executions": 0,
        "hardware_executions": 0,
        "frequency_writes": 0,
        "voltage_writes": 0,
        "msr_reads": 0,
        "msr_writes": 0,
    }
    result["self_test_passed"] = (
        public_self["self_test_passed"]
        and ideal["status"] == public.law_v2.V2_CLASS_CONFIRMED
        and zero["status"] == public.law_v2.V2_CLASS_NOT_ESTABLISHED
        and copyback_seals_final
        and copyback_seals_execution_manifest
        and execution_manifest_seals_final
        and synthetic_forbidden_detected
        and interpreter_forbidden_detected
        and wrapper_source_argument_not_false_positive
        and failed_process_scan_fails_closed
        and missing_k10temp_fails_closed
        and ambiguous_k10temp_fails_closed
        and malformed_temperature_fails_closed
        and temperature_veto_blocks
        and post_replicate_temperature_veto_blocks
        and policy_identity_drift_rejected
        and policy_minmax_drift_rejected
        and missing_policy_membership_rejected
        and missing_policy_driver_rejected
        and policy_cur_freq_variation_allowed
        and pmu_event_format_drift_rejected
        and pmu_umask_format_drift_rejected
        and pmu_type_drift_rejected
        and paranoid_drift_rejected
        and cpu_vendor_drift_rejected
        and cpu_family_drift_rejected
        and core_offline_rejected
        and binary_hash_semantics_coherent
        and target_binary_self_test_required
        and failure_evidence_sealing_succeeded
        and failure_evidence_written_without_classification
    )
    result["self_test_sha256"] = public.digest({k: v for k, v in result.items() if k != "self_test_sha256"})
    write_json(output_root / "CONFIRMATION_TARGET_SELF_TEST.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", default=public.RUN_ID)
    parser.add_argument("--expected-manifest-sha")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.self_test:
            result = self_test(args.source_root.resolve(), args.output_root.resolve())
            ok = result["self_test_passed"]
        else:
            result = execute(
                args.source_root.resolve(),
                args.output_root.resolve(),
                run_id=args.run_id,
                expected_manifest_sha=args.expected_manifest_sha,
            )
            ok = result["final"]["status"] == "CONFIRMATION_V2_TARGET_COMPLETE"
        print(json.dumps(result, sort_keys=True))
        return 0 if ok else 1
    except Exception as exc:
        print(f"confirmation_v2_target: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
