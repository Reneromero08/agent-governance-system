#!/usr/bin/env python3
"""Remote-side bounded OrbitState physical-query first-light target."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import orbit_query_public as public_model


TEMPERATURE_VETO_C = 68.0
SOURCE_NAMES = (
    "orbit_query_target.py",
    "orbit_query_public.py",
    "orbit_query_model.py",
    "orbit_query_runtime.c",
    "orbit_query_runtime.h",
)
FORBIDDEN_PROCESS_MARKERS = (
    "orbit_query_runtime",
    "f10_pmc_first_light_worker",
    "gate_a_worker_live",
    "combined_pdn_runner",
    "run_combined_campaign",
)


class OrbitTargetError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise OrbitTargetError(message)


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


def temperature_path() -> Path:
    for root in sorted(Path("/sys/class/hwmon").glob("hwmon*")):
        name_path = root / "name"
        if name_path.is_file() and read_text(name_path) == "k10temp":
            temp = root / "temp1_input"
            require(temp.is_file(), "k10temp temp1_input missing")
            return temp
    raise OrbitTargetError("k10temp hwmon path not found")


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


def source_digest(source_root: Path) -> tuple[str, dict[str, str]]:
    hashes = {name: sha256_file(source_root / name) for name in SOURCE_NAMES}
    return sha256_bytes(canonical_bytes(hashes)), hashes


def compile_worker(source_root: Path, binary: Path) -> tuple[str, dict[str, str], list[str]]:
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
        str(source_root / "orbit_query_runtime.c"),
        "-lm",
        "-o",
        str(binary),
    ]
    subprocess.run(command, check=True, timeout=30)
    return bundle, hashes, command


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_raw(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def combine_replicates_blind(output_root: Path, replicate_roots: list[Path]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    groups: list[dict[str, Any]] = []
    raw_records: list[dict[str, Any]] = []
    for root in replicate_roots:
        manifest = load_json(root / "CAPTURE_MANIFEST.json")
        groups.extend(manifest["groups"])
        raw_records.extend(load_raw(root / "RAW_CAPTURE.jsonl"))
    combined_manifest = {
        "schema_id": "CAT_CAS_ORBIT_QUERY_CAPTURE_MANIFEST_V1",
        "modulus": public_model.N,
        "phase_names": list(public_model.PHASE_NAMES),
        "decoder": "Z=(2/K)*sum(response_k*exp(i*theta_k))",
        "fresh_process_replicates": len(replicate_roots),
        "groups": groups,
    }
    public_model.assert_receiver_manifest_blind(combined_manifest)
    write_json(output_root / "CAPTURE_MANIFEST.json", combined_manifest)
    raw_payload = "".join(json.dumps(record, sort_keys=True) + "\n" for record in raw_records)
    write_text(output_root / "RAW_CAPTURE.jsonl", raw_payload)
    return combined_manifest, raw_records


def combine_unblinding_after_freeze(
    output_root: Path,
    replicate_roots: list[Path],
    expected_features_hash: str,
    expected_manifest_hash: str,
    expected_raw_hash: str,
) -> dict[str, Any]:
    features_path = output_root / "FEATURES_FROZEN.json"
    hash_path = output_root / "FEATURES_FROZEN.sha256"
    receipt_path = output_root / "FEATURE_FREEZE_RECEIPT.json"
    require(hash_path.is_file(), "features hash missing before unblind")
    require(receipt_path.is_file(), "freeze receipt missing before unblind")
    stored_hash = read_text(hash_path)
    current_hash = sha256_file(features_path)
    require(stored_hash == expected_features_hash, "stored features hash drift before unblind")
    require(current_hash == expected_features_hash, "current features hash drift before unblind")
    receipt = load_json(receipt_path)
    require(receipt["schema_id"] == "CAT_CAS_ORBIT_QUERY_FEATURE_FREEZE_RECEIPT_V1", "freeze receipt schema drift")
    require(receipt["features_frozen_sha256"] == expected_features_hash, "freeze receipt feature hash drift")
    require(receipt["capture_manifest_sha256"] == expected_manifest_hash, "freeze receipt manifest hash drift")
    require(receipt["raw_capture_sha256"] == expected_raw_hash, "freeze receipt raw hash drift")
    require(sha256_file(output_root / "CAPTURE_MANIFEST.json") == expected_manifest_hash, "current manifest hash drift before unblind")
    require(sha256_file(output_root / "RAW_CAPTURE.jsonl") == expected_raw_hash, "current raw hash drift before unblind")
    require(receipt["unblinding_loaded"] is False, "freeze receipt already marked unblind-loaded")
    unblind_groups: list[dict[str, Any]] = []
    for root in replicate_roots:
        unblind = load_json(root / "UNBLINDING_MAP.json")
        unblind_groups.extend(unblind["groups"])
    combined_unblind = {
        "schema_id": "CAT_CAS_ORBIT_QUERY_UNBLINDING_MAP_V1",
        "fresh_process_replicates": len(replicate_roots),
        "groups": unblind_groups,
    }
    write_json(output_root / "UNBLINDING_MAP.json", combined_unblind)
    return combined_unblind


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
    return {"schema_id": "CAT_CAS_ORBIT_QUERY_FILE_MANIFEST_V1", "files": files}


def execute(source_root: Path, output_root: Path) -> dict[str, Any]:
    require(source_root.is_dir(), f"source root missing: {source_root}")
    require(not output_root.exists(), f"output root already exists: {output_root}")
    for name in SOURCE_NAMES:
        require((source_root / name).is_file(), f"source missing: {name}")
    output_root.mkdir(mode=0o700, parents=True, exist_ok=False)

    final: dict[str, Any] = {
        "schema_id": "CAT_CAS_ORBIT_QUERY_TARGET_RESULT_V1",
        "status": "ORBIT_QUERY_TARGET_FAILED",
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
        require(not pre_processes["forbidden_matches"], "forbidden CAT_CAS process present before run")
        policy_before = policy_snapshot()
        binary = source_root / "orbit_query_runtime"
        source_bundle, source_hashes, compile_command = compile_worker(source_root, binary)
        replicate_roots: list[Path] = []
        runtime_results: list[dict[str, Any]] = []
        start_ns = time.monotonic_ns()
        for replicate in range(2):
            replicate_pre_temperature = read_temperature_c(temp_path)
            require(
                replicate_pre_temperature < TEMPERATURE_VETO_C,
                f"temperature veto before replicate {replicate}: {replicate_pre_temperature} C",
            )
            replicate_root = output_root / f"batch_{replicate}"
            command = [str(binary), "--output-root", str(replicate_root), "--replicate", str(replicate)]
            completed = subprocess.run(
                command,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=90,
                check=False,
            )
            replicate_post_temperature = read_temperature_c(temp_path)
            require(
                replicate_post_temperature < TEMPERATURE_VETO_C,
                f"temperature veto after replicate {replicate}: {replicate_post_temperature} C",
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
                    "pre_temperature_c": replicate_pre_temperature,
                    "post_temperature_c": replicate_post_temperature,
                }
            )
            require(completed.returncode == 0, f"runtime replicate {replicate} failed")
            replicate_roots.append(replicate_root)
        finish_ns = time.monotonic_ns()
        manifest, raw_records = combine_replicates_blind(output_root, replicate_roots)
        features = public_model.extract_features(manifest, raw_records)
        write_json(output_root / "FEATURES_FROZEN.json", features)
        manifest_hash = sha256_file(output_root / "CAPTURE_MANIFEST.json")
        raw_hash = sha256_file(output_root / "RAW_CAPTURE.jsonl")
        features_hash = sha256_file(output_root / "FEATURES_FROZEN.json")
        write_text(output_root / "FEATURES_FROZEN.sha256", features_hash + "\n")
        freeze_receipt = {
            "schema_id": "CAT_CAS_ORBIT_QUERY_FEATURE_FREEZE_RECEIPT_V1",
            "capture_manifest_sha256": manifest_hash,
            "raw_capture_sha256": raw_hash,
            "features_frozen_sha256": features_hash,
            "feature_model_sha256": sha256_file(source_root / "orbit_query_public.py"),
            "unblinding_loaded": False,
        }
        write_json(output_root / "FEATURE_FREEZE_RECEIPT.json", freeze_receipt)
        unblind = combine_unblinding_after_freeze(output_root, replicate_roots, features_hash, manifest_hash, raw_hash)
        private_model = importlib.import_module("orbit_query_model")
        adjudication = private_model.adjudicate(features, unblind)
        write_json(output_root / "ADJUDICATION.json", adjudication)
        post_temperature = read_temperature_c(temp_path)
        require(post_temperature < TEMPERATURE_VETO_C, f"temperature veto after run: {post_temperature} C")
        policy_after = policy_snapshot()
        post_processes = process_snapshot()
        require(not post_processes["forbidden_matches"], "forbidden CAT_CAS process present after run")
        final.update(
            {
                "status": "ORBIT_QUERY_TARGET_COMPLETE",
                "source_bundle_sha256": source_bundle,
                "source_hashes": source_hashes,
                "compile_command": compile_command,
                "runtime_results": runtime_results,
                "runtime_duration_ns": finish_ns - start_ns,
                "feature_record_count": len(features["features"]),
                "raw_record_count": len(raw_records),
                "features_frozen_sha256": features_hash,
                "capture_manifest_sha256": manifest_hash,
                "raw_capture_sha256": raw_hash,
                "feature_freeze_receipt_sha256": sha256_file(output_root / "FEATURE_FREEZE_RECEIPT.json"),
                "unblinding_read_after_features_sha256": True,
                "adjudication_status": adjudication["status"],
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
                "claim_ceiling": "ORBITSTATE_PHYSICAL_QUERY_COUPLING_CANDIDATE; SMALL_WALL_CROSSED forbidden",
            }
        )
    except Exception as exc:  # final result must survive failed-closed target runs
        final["failure"] = f"{type(exc).__name__}: {exc}"
    write_json(output_root / "FINAL_RESULT.json", final)
    manifest = build_file_manifest(output_root)
    write_json(output_root / "FILE_MANIFEST.json", manifest)
    return final


def self_test(source_root: Path, output_root: Path) -> dict[str, Any]:
    require(not output_root.exists(), f"self-test output exists: {output_root}")
    output_root.mkdir(mode=0o700, parents=True)
    source_bundle, source_hashes, command = compile_worker(source_root, output_root / "orbit_query_runtime_check")
    completed = subprocess.run(
        [str(output_root / "orbit_query_runtime_check"), "--self-test"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=20,
        check=False,
    )
    result = {
        "schema_id": "CAT_CAS_ORBIT_QUERY_TARGET_SELF_TEST_V1",
        "source_bundle_sha256": source_bundle,
        "source_hashes": source_hashes,
        "compile_command": command,
        "runtime_self_test_returncode": completed.returncode,
        "runtime_self_test_stdout": completed.stdout,
        "runtime_self_test_stderr": completed.stderr,
        "target_self_test_passed": completed.returncode == 0,
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
    return 0 if result["status"] == "ORBIT_QUERY_TARGET_COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
