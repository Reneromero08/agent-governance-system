#!/usr/bin/env python3
"""Remote-side target for one bounded public balanced-transducer calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import balanced_transducer_public as public_model


TEMPERATURE_VETO_C = 68.0
SOURCE_NAMES = (
    "BALANCED_TRANSDUCER_CONTRACT.md",
    "FIRST_LIGHT_TRANSDUCER_AUDIT.md",
    "PUBLIC_TRIAL_SCHEDULE.json",
    "PUBLIC_TRIAL_SCHEDULE.sha256",
    "PUBLIC_TRIAL_SCHEDULE.tsv",
    "balanced_transducer_target.py",
    "balanced_transducer_public.py",
    "balanced_transducer_runtime.c",
    "balanced_transducer_runtime.h",
)
FORBIDDEN_PROCESS_MARKERS = (
    "balanced_transducer_runtime",
    "orbit_query_runtime",
    "f10_pmc_first_light_worker",
    "gate_a_worker_live",
    "combined_pdn_runner",
    "run_combined_campaign",
)


class BalancedTargetError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BalancedTargetError(message)


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("xb") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("xb") as handle:
        handle.write(value.encode("utf-8"))
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def prepare_schedule_artifacts(source_root: Path, output_root: Path) -> tuple[dict[str, Any], Path, str]:
    schedule_source = source_root / "PUBLIC_TRIAL_SCHEDULE.json"
    schedule_hash_source = source_root / "PUBLIC_TRIAL_SCHEDULE.sha256"
    schedule_tsv_source = source_root / "PUBLIC_TRIAL_SCHEDULE.tsv"
    schedule = json.loads(schedule_source.read_text(encoding="utf-8"))
    public_model.validate_schedule(schedule)
    expected_schedule_hash = read_text(schedule_hash_source)
    require(sha256_file(schedule_source) == expected_schedule_hash, "source schedule hash mismatch")
    require(
        schedule["schedule_sha256"] == public_model.digest({k: v for k, v in schedule.items() if k != "schedule_sha256"}),
        "schedule embedded hash mismatch",
    )
    require(
        schedule_tsv_source.read_text(encoding="utf-8") == public_model.schedule_tsv(schedule),
        "schedule TSV does not match JSON schedule",
    )
    schedule_path = output_root / "PUBLIC_TRIAL_SCHEDULE.json"
    schedule_hash_path = output_root / "PUBLIC_TRIAL_SCHEDULE.sha256"
    schedule_tsv_path = output_root / "PUBLIC_TRIAL_SCHEDULE.tsv"
    shutil.copy2(schedule_source, schedule_path)
    shutil.copy2(schedule_hash_source, schedule_hash_path)
    shutil.copy2(schedule_tsv_source, schedule_tsv_path)
    return schedule, schedule_tsv_path, sha256_file(schedule_path)


def runtime_command(binary: Path, schedule_tsv_path: Path, batch_root: Path, replicate: int) -> list[str]:
    return [
        str(binary),
        "--schedule-tsv",
        str(schedule_tsv_path),
        "--output-root",
        str(batch_root),
        "--replicate",
        str(replicate),
    ]


def temperature_path() -> Path:
    for root in sorted(Path("/sys/class/hwmon").glob("hwmon*")):
        name_path = root / "name"
        if name_path.is_file() and read_text(name_path) == "k10temp":
            temp = root / "temp1_input"
            require(temp.is_file(), "k10temp temp1_input missing")
            return temp
    raise BalancedTargetError("k10temp hwmon path not found")


def read_temperature_c(path: Path) -> float:
    raw = read_text(path)
    require(raw.isdecimal(), f"temperature is not decimal: {raw!r}")
    return int(raw, 10) / 1000.0


def process_snapshot() -> dict[str, Any]:
    completed = subprocess.run(
        ["ps", "-eo", "pid=,comm=,args="],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=5,
        check=False,
    )
    matches: list[str] = []
    for raw in completed.stdout.splitlines():
        line = raw.strip()
        if any(marker in line for marker in FORBIDDEN_PROCESS_MARKERS):
            matches.append(line)
    return {"observed_process_count": len(completed.stdout.splitlines()), "forbidden_matches": matches}


def policy_snapshot() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for core in (4, 5):
        policy = Path(f"/sys/devices/system/cpu/cpufreq/policy{core}")
        try:
            result[f"policy{core}"] = {
                "resolved_path": str(policy.resolve(strict=True)),
                "scaling_min_freq": int(read_text(policy / "scaling_min_freq")),
                "scaling_max_freq": int(read_text(policy / "scaling_max_freq")),
                "scaling_cur_freq": int(read_text(policy / "scaling_cur_freq")),
            }
        except (OSError, ValueError) as exc:
            result[f"policy{core}"] = {"error": str(exc)}
    return result


def policy_limits_restored(before: dict[str, Any], after: dict[str, Any]) -> bool:
    for key in ("policy4", "policy5"):
        left = before.get(key, {})
        right = after.get(key, {})
        for field in ("resolved_path", "scaling_min_freq", "scaling_max_freq"):
            if left.get(field) != right.get(field):
                return False
    return True


def cpuinfo_snapshot() -> dict[str, str]:
    wanted = ("model name", "cpu family", "model", "stepping", "vendor_id")
    out: dict[str, str] = {}
    for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, value = [part.strip() for part in line.split(":", 1)]
        if key in wanted and key not in out:
            out[key] = value
    return out


def pmu_sysfs_snapshot() -> dict[str, Any]:
    cpu_pmu = Path("/sys/bus/event_source/devices/cpu")
    fmt: dict[str, str] = {}
    for path in sorted((cpu_pmu / "format").glob("*")):
        if path.is_file():
            fmt[path.name] = read_text(path)
    return {
        "cpu_pmu_type": read_text(cpu_pmu / "type") if (cpu_pmu / "type").is_file() else None,
        "format": fmt,
        "perf_event_paranoid": read_text(Path("/proc/sys/kernel/perf_event_paranoid")),
        "cpu_online": read_text(Path("/sys/devices/system/cpu/online")),
    }


def require_cpu_and_pmu(cpuinfo: dict[str, str], pmu: dict[str, Any]) -> None:
    require(cpuinfo.get("vendor_id") == "AuthenticAMD", f"unexpected CPU vendor: {cpuinfo.get('vendor_id')}")
    require(cpuinfo.get("cpu family") == "16", f"unexpected CPU family: {cpuinfo.get('cpu family')}")
    fmt = pmu.get("format", {})
    require(fmt.get("event") == "config:0-7,32-35", f"unexpected PMU event format: {fmt.get('event')}")
    require(fmt.get("umask") == "config:8-15", f"unexpected PMU umask format: {fmt.get('umask')}")


def source_digest(source_root: Path) -> tuple[str, dict[str, str]]:
    hashes = {name: sha256_file(source_root / name) for name in SOURCE_NAMES}
    return sha256_bytes(canonical_bytes(hashes)), hashes


def compile_runtime(source_root: Path, binary: Path) -> tuple[str, dict[str, str], list[str], str]:
    bundle, hashes = source_digest(source_root)
    command = [
        "cc",
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
        "-I",
        str(source_root),
        str(source_root / "balanced_transducer_runtime.c"),
        "-o",
        str(binary),
    ]
    subprocess.run(command, check=True, timeout=30)
    return bundle, hashes, command, sha256_file(binary)


def build_file_manifest(output_root: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name != "FILE_MANIFEST.json":
            files.append(
                {
                    "path": path.relative_to(output_root).as_posix(),
                    "size": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return {"schema_id": "CAT_CAS_BALANCED_TRANSDUCER_FILE_MANIFEST_V1", "files": files}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def combine_batch_jsonl(output_root: Path, batch_roots: list[Path], name: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for root in batch_roots:
        records.extend(load_jsonl(root / name))
    records.sort(key=lambda item: (int(item["replicate_index"]), int(item["trial_index"])))
    write_text(output_root / name, "".join(json.dumps(record, sort_keys=True) + "\n" for record in records))
    return records


def execute(source_root: Path, output_root: Path) -> dict[str, Any]:
    require(source_root.is_dir(), f"source root missing: {source_root}")
    require(not output_root.exists(), f"output root already exists: {output_root}")
    for name in SOURCE_NAMES:
        require((source_root / name).is_file(), f"source missing: {name}")
    output_root.mkdir(mode=0o700, parents=True, exist_ok=False)

    final: dict[str, Any] = {
        "schema_id": "CAT_CAS_BALANCED_TRANSDUCER_TARGET_RESULT_V1",
        "status": "BALANCED_TRANSDUCER_TARGET_FAILED",
        "failure": None,
        "frequency_writes": 0,
        "voltage_writes": 0,
        "msr_reads": 0,
        "msr_writes": 0,
        "physical_address_access": False,
        "cache_set_mapping": False,
    }
    try:
        temp_path = temperature_path()
        pre_temperature = read_temperature_c(temp_path)
        require(pre_temperature < TEMPERATURE_VETO_C, f"temperature veto before run: {pre_temperature} C")
        pre_processes = process_snapshot()
        require(not pre_processes["forbidden_matches"], "forbidden CAT_CAS process present before calibration")
        policy_before = policy_snapshot()
        cpuinfo = cpuinfo_snapshot()
        pmu = pmu_sysfs_snapshot()
        require_cpu_and_pmu(cpuinfo, pmu)
        schedule, schedule_tsv_path, schedule_hash = prepare_schedule_artifacts(source_root, output_root)

        binary = source_root / "balanced_transducer_runtime"
        source_bundle, source_hashes, compile_command, binary_hash = compile_runtime(source_root, binary)
        source_bundle_doc = {
            "schema_id": "CAT_CAS_BALANCED_TRANSDUCER_SOURCE_BUNDLE_V1",
            "source_bundle_sha256": source_bundle,
            "source_hashes": source_hashes,
            "contract_sha256": source_hashes["BALANCED_TRANSDUCER_CONTRACT.md"],
            "public_trial_schedule_sha256": schedule_hash,
            "runtime_binary_sha256": binary_hash,
            "compile_command": compile_command,
        }
        write_json(output_root / "SOURCE_BUNDLE.json", source_bundle_doc)

        runtime_results: list[dict[str, Any]] = []
        batch_roots: list[Path] = []
        start_ns = time.monotonic_ns()
        for replicate in public_model.REPLICATES:
            replicate_pre_temp = read_temperature_c(temp_path)
            require(
                replicate_pre_temp < TEMPERATURE_VETO_C,
                f"temperature veto before replicate {replicate}: {replicate_pre_temp} C",
            )
            batch_root = output_root / f"batch_{replicate}"
            command = runtime_command(binary, schedule_tsv_path, batch_root, replicate)
            completed = subprocess.run(
                command,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=120,
                check=False,
            )
            replicate_post_temp = read_temperature_c(temp_path)
            require(
                replicate_post_temp < TEMPERATURE_VETO_C,
                f"temperature veto after replicate {replicate}: {replicate_post_temp} C",
            )
            write_text(output_root / f"RUNTIME_STDOUT_{replicate}.txt", completed.stdout)
            write_text(output_root / f"RUNTIME_STDERR_{replicate}.txt", completed.stderr)
            runtime_results.append(
                {
                    "replicate": replicate,
                    "command": command,
                    "returncode": completed.returncode,
                    "stdout_sha256": sha256_bytes(completed.stdout.encode("utf-8")),
                    "stderr_sha256": sha256_bytes(completed.stderr.encode("utf-8")),
                    "pre_temperature_c": replicate_pre_temp,
                    "post_temperature_c": replicate_post_temp,
                    "batch_root": str(batch_root),
                }
            )
            require(completed.returncode == 0, f"runtime replicate {replicate} failed")
            batch_roots.append(batch_root)
        finish_ns = time.monotonic_ns()

        raw_records = combine_batch_jsonl(output_root, batch_roots, "RAW_TRANSDUCER_CAPTURE.jsonl")
        sentinels = combine_batch_jsonl(output_root, batch_roots, "RESTORATION_SENTINELS.jsonl")
        features = public_model.extract_features(schedule, raw_records, sentinels)
        write_json(output_root / "TRANSDUCER_FEATURES.json", features)
        adjudication = public_model.adjudicate(features)
        write_json(output_root / "TRANSDUCER_ADJUDICATION.json", adjudication)

        post_temperature = read_temperature_c(temp_path)
        require(post_temperature < TEMPERATURE_VETO_C, f"temperature veto after run: {post_temperature} C")
        policy_after = policy_snapshot()
        post_processes = process_snapshot()
        require(not post_processes["forbidden_matches"], "forbidden CAT_CAS process present after calibration")
        final.update(
            {
                "status": "BALANCED_TRANSDUCER_TARGET_COMPLETE",
                "source_bundle_sha256": source_bundle,
                "source_hashes": source_hashes,
                "runtime_binary_sha256": binary_hash,
                "compile_command": compile_command,
                "runtime_results": runtime_results,
                "runtime_duration_ns": finish_ns - start_ns,
                "raw_record_count": len(raw_records),
                "sentinel_record_count": len(sentinels),
                "public_trial_schedule_sha256": schedule_hash,
                "source_bundle_file_sha256": sha256_file(output_root / "SOURCE_BUNDLE.json"),
                "raw_capture_sha256": sha256_file(output_root / "RAW_TRANSDUCER_CAPTURE.jsonl"),
                "restoration_sentinels_sha256": sha256_file(output_root / "RESTORATION_SENTINELS.jsonl"),
                "features_sha256": sha256_file(output_root / "TRANSDUCER_FEATURES.json"),
                "adjudication_sha256": sha256_file(output_root / "TRANSDUCER_ADJUDICATION.json"),
                "adjudication_status": adjudication["status"],
                "primary_coordinate": adjudication["primary_coordinate"],
                "eligible_coordinates": adjudication["eligible_coordinates"],
                "temperature": {
                    "path": str(temp_path),
                    "veto_c": TEMPERATURE_VETO_C,
                    "preflight_c": pre_temperature,
                    "post_c": post_temperature,
                    "below_veto": pre_temperature < TEMPERATURE_VETO_C and post_temperature < TEMPERATURE_VETO_C,
                },
                "policy_before": policy_before,
                "policy_after": policy_after,
                "cpu_frequency_policy_restored": policy_limits_restored(policy_before, policy_after),
                "preflight_processes": pre_processes,
                "process_cleanup": post_processes,
                "cpuinfo": cpuinfo,
                "kernel": os.uname().release,
                "pmu_sysfs": pmu,
                "claim_ceiling": "public balanced transducer only; no OrbitState coupling candidate and no Small Wall crossing claim",
            }
        )
    except Exception as exc:  # retain closed target result on failure
        final["failure"] = f"{type(exc).__name__}: {exc}"
    write_json(output_root / "FINAL_RESULT.json", final)
    write_json(output_root / "FILE_MANIFEST.json", build_file_manifest(output_root))
    return final


def self_test(source_root: Path, output_root: Path) -> dict[str, Any]:
    require(not output_root.exists(), f"self-test output exists: {output_root}")
    output_root.mkdir(mode=0o700, parents=True)
    schedule, schedule_tsv_path, schedule_hash = prepare_schedule_artifacts(source_root, output_root)
    public_self_test = public_model.self_test()
    source_bundle, source_hashes, command, binary_hash = compile_runtime(
        source_root,
        output_root / "balanced_transducer_runtime_check",
    )
    completed = subprocess.run(
        [str(output_root / "balanced_transducer_runtime_check"), "--self-test"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=20,
        check=False,
    )
    result = {
        "schema_id": "CAT_CAS_BALANCED_TRANSDUCER_TARGET_SELF_TEST_V1",
        "source_bundle_sha256": source_bundle,
        "source_hashes": source_hashes,
        "compile_command": command,
        "public_trial_schedule_sha256": schedule_hash,
        "public_trial_count": len(schedule["trials"]),
        "example_runtime_command": runtime_command(
            output_root / "balanced_transducer_runtime_check",
            schedule_tsv_path,
            output_root / "batch_0",
            0,
        ),
        "runtime_binary_sha256": binary_hash,
        "runtime_self_test_returncode": completed.returncode,
        "runtime_self_test_stdout": completed.stdout,
        "runtime_self_test_stderr": completed.stderr,
        "public_self_test": public_self_test,
        "target_self_test_passed": public_self_test["self_test_passed"] and completed.returncode == 0,
    }
    write_json(output_root / "TARGET_SELF_TEST.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        result = self_test(args.source_root.resolve(), args.output_root.resolve())
        print(json.dumps(result, sort_keys=True))
        return 0 if result["target_self_test_passed"] else 1
    result = execute(args.source_root.resolve(), args.output_root.resolve())
    print(json.dumps({"status": result["status"], "failure": result.get("failure")}, sort_keys=True))
    return 0 if result["status"] == "BALANCED_TRANSDUCER_TARGET_COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
