#!/usr/bin/env python3
"""Compile and run one bounded Family 10h PMU first-light discriminator."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any


TEMPERATURE_VETO_C = 68.0
SOURCE_NAMES = ("f10_pmc_first_light_target.py", "f10_pmc_first_light_worker.c")
MODES: dict[str, dict[str, Any]] = {
    "pmu-first-light": {
        "worker_args": [],
        "result_file": "F10_PMC_FIRST_LIGHT_RESULT.json",
        "complete_status": "F10_PMC_FIRST_LIGHT_TARGET_COMPLETE",
        "failed_status": "F10_PMC_FIRST_LIGHT_TARGET_FAILED",
        "claim_ceiling": "Family 10h PMU first-light discriminator only; no coherence holonomy, OrbitState coupling, fold-odd recovery, or Small Wall crossing claim",
    },
    "coherence-operators": {
        "worker_args": ["--coherence-operators"],
        "result_file": "F10_COHERENCE_OPERATOR_RESULT.json",
        "complete_status": "F10_COHERENCE_OPERATOR_TARGET_COMPLETE",
        "failed_status": "F10_COHERENCE_OPERATOR_TARGET_FAILED",
        "claim_ceiling": "Controlled coherence-operator PMU discriminator only; no path memory, coherence holonomy, OrbitState coupling, fold-odd recovery, or Small Wall crossing claim",
    },
    "path-dependence": {
        "worker_args": ["--path-dependence"],
        "result_file": "F10_PATH_DEPENDENCE_PILOT_RESULT.json",
        "complete_status": "F10_PATH_DEPENDENCE_TARGET_COMPLETE",
        "failed_status": "F10_PATH_DEPENDENCE_TARGET_FAILED",
        "claim_ceiling": "Path-dependence pilot only; no coherence holonomy, OrbitState coupling, fold-odd recovery, or Small Wall crossing claim",
    },
    "path-dual-observe": {
        "worker_args": ["--path-dual-observe"],
        "result_file": "F10_PATH_DUAL_OBSERVE_RESULT.json",
        "complete_status": "F10_PATH_DUAL_OBSERVE_TARGET_COMPLETE",
        "failed_status": "F10_PATH_DUAL_OBSERVE_TARGET_FAILED",
        "claim_ceiling": "Dual-observed path pilot only; no coherence holonomy, OrbitState coupling, fold-odd recovery, or Small Wall crossing claim",
    },
    "path-rw-observe": {
        "worker_args": ["--path-rw-observe"],
        "result_file": "F10_PATH_RW_OBSERVE_RESULT.json",
        "complete_status": "F10_PATH_RW_OBSERVE_TARGET_COMPLETE",
        "failed_status": "F10_PATH_RW_OBSERVE_TARGET_FAILED",
        "claim_ceiling": "Read/store path pilot only; no coherence holonomy, OrbitState coupling, fold-odd recovery, or Small Wall crossing claim",
    },
}
FORBIDDEN_PROCESS_MARKERS = (
    "f10_pmc_first_light_worker",
    "gate_a_worker_live",
    "combined_pdn_runner",
    "run_combined_campaign",
    "explicit_slot_runtime",
)


class PmcFirstLightError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PmcFirstLightError(message)


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("xb") as handle:
        handle.write(canonical_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def temperature_path() -> Path:
    roots = sorted(Path("/sys/class/hwmon").glob("hwmon*"))
    for root in roots:
        name_path = root / "name"
        if name_path.is_file() and read_text(name_path) == "k10temp":
            temp = root / "temp1_input"
            require(temp.is_file(), "k10temp temp1_input missing")
            return temp
    raise PmcFirstLightError("k10temp hwmon path not found")


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
    return {
        "observed_process_count": len(completed.stdout.splitlines()),
        "forbidden_matches": matches,
    }


def source_digest(source_root: Path) -> tuple[str, dict[str, str]]:
    hashes = {name: sha256_file(source_root / name) for name in SOURCE_NAMES}
    return hashlib.sha256(canonical_bytes(hashes)).hexdigest(), hashes


def compile_worker(source_root: Path, binary: Path) -> tuple[str, list[str]]:
    bundle_sha256, _hashes = source_digest(source_root)
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
        str(source_root / "f10_pmc_first_light_worker.c"),
        "-o",
        str(binary),
    ]
    subprocess.run(command, check=True, timeout=30)
    return bundle_sha256, command


def pmu_sysfs_snapshot() -> dict[str, Any]:
    cpu_pmu = Path("/sys/bus/event_source/devices/cpu")
    fmt: dict[str, str] = {}
    events: dict[str, str] = {}
    caps: dict[str, str] = {}
    for path in sorted((cpu_pmu / "format").glob("*")):
        if path.is_file():
            fmt[path.name] = read_text(path)
    for path in sorted((cpu_pmu / "events").glob("*")):
        if path.is_file():
            events[path.name] = read_text(path)
    for path in sorted((cpu_pmu / "caps").glob("*")):
        if path.is_file():
            caps[path.name] = read_text(path)
    return {
        "cpu_pmu_type": read_text(cpu_pmu / "type"),
        "format": fmt,
        "events": events,
        "caps": caps,
        "perf_event_paranoid": read_text(Path("/proc/sys/kernel/perf_event_paranoid")),
        "cpu_online": read_text(Path("/sys/devices/system/cpu/online")),
    }


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


def build_file_manifest(output_root: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for path in sorted(output_root.rglob("*")):
        if not path.is_file() or path.name == "FILE_MANIFEST.json":
            continue
        files.append(
            {
                "path": path.relative_to(output_root).as_posix(),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {"schema_id": "CAT_CAS_F10_PMC_FIRST_LIGHT_FILE_MANIFEST_V1", "files": files}


def execute(source_root: Path, output_root: Path, mode: str) -> dict[str, Any]:
    require(mode in MODES, f"unsupported mode: {mode}")
    mode_config = MODES[mode]
    require(source_root.is_dir(), f"source root missing: {source_root}")
    require(not output_root.exists(), f"output root already exists: {output_root}")
    for name in SOURCE_NAMES:
        require((source_root / name).is_file(), f"source missing: {name}")
    output_root.mkdir(mode=0o700, parents=True, exist_ok=False)
    binary = source_root / "f10_pmc_first_light_worker"
    temp_path = temperature_path()
    preflight_processes = process_snapshot()
    require(not preflight_processes["forbidden_matches"], "forbidden CAT_CAS process present before PMU transaction")
    pre_temperature = read_temperature_c(temp_path)
    require(pre_temperature < TEMPERATURE_VETO_C, f"preflight temperature veto: {pre_temperature} C")
    source_bundle_sha256, compile_command = compile_worker(source_root, binary)
    worker_command = [str(binary), *mode_config["worker_args"], "--output-root", str(output_root)]
    start = time.monotonic_ns()
    completed = subprocess.run(
        worker_command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )
    finish = time.monotonic_ns()
    post_temperature = read_temperature_c(temp_path)
    process_cleanup = process_snapshot()
    result_path = output_root / str(mode_config["result_file"])
    result_available = result_path.is_file()
    result = json.loads(result_path.read_text(encoding="utf-8")) if result_available else None
    final = {
        "schema_id": "CAT_CAS_F10_PMC_FIRST_LIGHT_TARGET_RESULT_V1",
        "mode": mode,
        "status": mode_config["complete_status"] if completed.returncode == 0 and result_available else mode_config["failed_status"],
        "source_bundle_sha256": source_bundle_sha256,
        "source_hashes": source_digest(source_root)[1],
        "compile_command": compile_command,
        "worker_command": worker_command,
        "cpuinfo": cpuinfo_snapshot(),
        "kernel": os.uname().release,
        "pmu_sysfs": pmu_sysfs_snapshot(),
        "temperature": {
            "path": str(temp_path),
            "veto_c": TEMPERATURE_VETO_C,
            "preflight_c": pre_temperature,
            "post_c": post_temperature,
            "below_veto": pre_temperature < TEMPERATURE_VETO_C and post_temperature < TEMPERATURE_VETO_C,
        },
        "preflight_processes": preflight_processes,
        "process_cleanup": process_cleanup,
        "runtime": {
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "duration_ns": finish - start,
        },
        "worker_result_available": result_available,
        "worker_status": None if result is None else result.get("status"),
        "worker_result_file": str(mode_config["result_file"]),
        "frequency_writes": 0,
        "voltage_writes": 0,
        "msr_reads": 0,
        "msr_writes": 0,
        "claim_ceiling": mode_config["claim_ceiling"],
    }
    write_json(output_root / "FINAL_RESULT.json", final)
    write_json(output_root / "FILE_MANIFEST.json", build_file_manifest(output_root))
    print(json.dumps(final, sort_keys=True))
    return final


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mode", choices=sorted(MODES), default="pmu-first-light")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = execute(args.source_root, args.output_root, args.mode)
    except (PmcFirstLightError, OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        print(f"f10_pmc_first_light_target: {exc}", file=__import__("sys").stderr)
        return 1
    return 0 if result["status"].endswith("_TARGET_COMPLETE") else 1


if __name__ == "__main__":
    raise SystemExit(main())
