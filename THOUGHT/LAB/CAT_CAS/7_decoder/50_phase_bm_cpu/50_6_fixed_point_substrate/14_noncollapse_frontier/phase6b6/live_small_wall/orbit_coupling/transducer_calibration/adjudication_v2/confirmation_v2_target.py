#!/usr/bin/env python3
"""Lab-side target wrapper for the frozen Confirmation V2 package."""

from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

import confirmation_v2_public as public


TEMPERATURE_VETO_C = 68.0
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
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
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
        str(source_root / "confirmation_v2_runtime.c"),
        "-o",
        str(binary),
    ]
    subprocess.run(command, check=True, timeout=30)
    return command, sha256_file(binary)


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
    runtime_binary_sha: str,
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
        "runtime_binary_sha256": runtime_binary_sha,
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


def process_snapshot() -> dict[str, Any]:
    return {"forbidden_matches": [], "observed_process_count": -1, "note": "live process scan occurs on lab device only"}


def cpuinfo_snapshot() -> dict[str, str]:
    info: dict[str, str] = {}
    path = Path("/proc/cpuinfo")
    if path.is_file():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                info.setdefault(key.strip(), value.strip())
    return info


def require_cpu(cpuinfo: dict[str, str]) -> None:
    require(cpuinfo.get("vendor_id") == "AuthenticAMD", "CPU vendor mismatch")
    require(cpuinfo.get("cpu family") == "16", "CPU family mismatch")


def temperature_c() -> float:
    for path in Path("/sys/class/hwmon").glob("hwmon*/temp1_input"):
        try:
            return float(path.read_text(encoding="utf-8").strip()) / 1000.0
        except (OSError, ValueError):
            continue
    return -1.0


def execute(source_root: Path, output_root: Path, *, run_id: str, expected_manifest_sha: str | None = None) -> dict[str, Any]:
    require(run_id == public.RUN_ID, "run ID mismatch")
    require(not output_root.exists(), f"output root already exists: {output_root}")
    output_root.mkdir(mode=0o700, parents=True, exist_ok=False)
    start_ns = time.monotonic_ns()
    cpuinfo = cpuinfo_snapshot()
    require_cpu(cpuinfo)
    temp_before = temperature_c()
    require(temp_before < 0.0 or temp_before < TEMPERATURE_VETO_C, f"temperature veto before run: {temp_before}")
    schedule, schedule_tsv, schedule_json_sha, schedule_tsv_sha = prepare_schedule_artifacts(source_root, output_root)
    source_bundle_sha, source_hashes_map = build_source_bundle(source_root, output_root)
    implementation_manifest_sha: str | None = None
    if expected_manifest_sha:
        manifest_path = source_root / "CONFIRMATION_V2_IMPLEMENTATION_MANIFEST.json"
        require(manifest_path.is_file(), "implementation manifest missing")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        recomputed_manifest_sha = manifest_digest(manifest)
        require(manifest["implementation_manifest_sha256"] == expected_manifest_sha, "implementation manifest self SHA mismatch")
        require(recomputed_manifest_sha == expected_manifest_sha, "implementation manifest recomputed SHA mismatch")
        require(source_bundle_sha == manifest["expected_source_bundle_sha256"], "source bundle SHA mismatch")
        require(schedule_json_sha == manifest["schedule_json_sha256"], "schedule JSON SHA mismatch")
        require(schedule_tsv_sha == manifest["schedule_tsv_sha256"], "schedule TSV SHA mismatch")
        require(source_hashes_map == manifest["source_hashes"], "source hash manifest mismatch")
        implementation_manifest_sha = expected_manifest_sha
    binary = source_root / "confirmation_v2_runtime"
    compile_command, runtime_binary_sha = compile_runtime(source_root, binary)
    batch_roots = []
    runtime_results = []
    for rep in public.REPLICATES:
        batch_root = output_root / f"batch_{rep}"
        command = runtime_command(binary, schedule_tsv, batch_root, rep)
        completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120, check=False)
        (output_root / f"CONFIRMATION_RUNTIME_STDOUT_REPLICATE_{rep}.txt").write_text(completed.stdout, encoding="utf-8")
        (output_root / f"CONFIRMATION_RUNTIME_STDERR_REPLICATE_{rep}.txt").write_text(completed.stderr, encoding="utf-8")
        require(completed.returncode == 0, f"runtime replicate {rep} failed")
        runtime_results.append({"replicate": rep, "returncode": completed.returncode, "command": command})
        batch_roots.append(batch_root)
    raw_records = combine_batch_jsonl(output_root, batch_roots, "RAW_TRANSDUCER_CAPTURE.jsonl")
    sentinels = combine_batch_jsonl(output_root, batch_roots, "RESTORATION_SENTINELS.jsonl")
    require(len(raw_records) == public.TOTAL_TRIALS, "raw record count mismatch")
    require(len(sentinels) == public.TOTAL_TRIALS, "sentinel record count mismatch")
    features = public.extract_features(schedule, raw_records, sentinels)
    adjudication = public.adjudicate(features)
    write_json(output_root / "TRANSDUCER_FEATURES_V2.json", features)
    write_json(output_root / "TRANSDUCER_ADJUDICATION_CONFIRMATION_V2.json", adjudication)
    temp_after = temperature_c()
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
        "runtime_binary_sha256": runtime_binary_sha,
        "raw_capture_sha256": sha256_file(output_root / "RAW_TRANSDUCER_CAPTURE.jsonl"),
        "restoration_sentinels_sha256": sha256_file(output_root / "RESTORATION_SENTINELS.jsonl"),
        "features_sha256": features["features_sha256"],
        "adjudication_sha256": adjudication["adjudication_sha256"],
        "raw_record_count": len(raw_records),
        "sentinel_record_count": len(sentinels),
        "runtime_results": runtime_results,
        "cpuinfo": cpuinfo,
        "temperature": {"pre_c": temp_before, "post_c": temp_after, "veto_c": TEMPERATURE_VETO_C},
        "frequency_writes": 0,
        "voltage_writes": 0,
        "msr_reads": 0,
        "msr_writes": 0,
        "physical_address_access": False,
        "cache_set_mapping": False,
        "process_cleanup": process_snapshot(),
        "runtime_duration_ns": time.monotonic_ns() - start_ns,
    }
    write_json(output_root / "LIVE_CUSTODY_LOG.json", {"schema_id": "CAT_CAS_CONFIRMATION_V2_LIVE_CUSTODY_LOG", "run_id": run_id, "events": runtime_results})
    write_json(output_root / "FINAL_RESULT_CONFIRMATION_V2.json", final)
    execution_manifest = build_execution_manifest(
        output_root,
        run_id=run_id,
        implementation_manifest_sha256=implementation_manifest_sha,
        source_bundle_sha=source_bundle_sha,
        schedule_json_sha=schedule_json_sha,
        schedule_tsv_sha=schedule_tsv_sha,
        runtime_binary_sha=runtime_binary_sha,
        raw_count=len(raw_records),
        sentinel_count=len(sentinels),
    )
    manifest = build_copyback_manifest(output_root)
    return {"final": final, "execution_manifest": execution_manifest, "copyback_manifest": manifest}


def self_test(source_root: Path, output_root: Path) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    schedule_hashes = public.write_schedule_artifacts(source_root)
    public_self = public.self_test()
    ideal = public.run_case("ideal_direct")
    zero = public.run_case("zero_transfer")
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
        runtime_binary_sha="2" * 64,
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
