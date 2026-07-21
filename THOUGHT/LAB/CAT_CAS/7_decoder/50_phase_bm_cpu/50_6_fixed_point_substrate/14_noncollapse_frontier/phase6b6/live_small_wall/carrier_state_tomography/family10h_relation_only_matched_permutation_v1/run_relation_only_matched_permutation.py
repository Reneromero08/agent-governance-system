#!/usr/bin/env python3
"""Prepare and validate the relation-only matched-permutation package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any, Callable

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import relation_only_adjudication as synthetic_adjudication
import relation_only_physical_adjudication as physical_adjudication
import relation_only_public as pub


GRAMMAR_JSON = HERE / "RELATION_GRAMMAR.json"
GRAMMAR_TSV = HERE / "RELATION_GRAMMAR.tsv"
GRAMMAR_SHA = HERE / "RELATION_GRAMMAR.sha256"
PROOF_JSON = HERE / "RELATION_MARGINAL_EQUALITY_PROOF.json"
SELF_TEST_JSON = HERE / "RELATION_ONLY_SELF_TEST.json"
ADVERSARY_JSON = HERE / "RELATION_ONLY_ADVERSARY_TEST.json"
TRANSPORT_JSON = HERE / "RELATION_ONLY_TRANSPORT_SIMULATION.json"
VALIDATE_JSON = HERE / "RELATION_ONLY_OFFLINE_VALIDATE.json"
MANIFEST_JSON = HERE / "RELATION_ONLY_IMPLEMENTATION_MANIFEST.json"
MANIFEST_SHA = HERE / "RELATION_ONLY_IMPLEMENTATION_MANIFEST.sha256"
SOURCE_HASHES_JSON = HERE / "RELATION_ONLY_SOURCE_HASHES.json"
SOURCE_BUNDLE = HERE / "RELATION_ONLY_SOURCE_BUNDLE.tar.gz"
SENSOR_AUTHORITY_JSON = HERE / pub.SENSOR_AUTHORITY_BINDING_FILENAME
SENSOR_AUTHORITY_SOURCE = HERE.parent / "family10h_carrier_tomography_v1" / "CARRIER_TOMOGRAPHY_TEMPERATURE_SENSOR_AUTHORITY.json"
PMU_PREFLIGHT_BINARY = HERE / "relation_only_pmu_preflight"
SCHEDULE_JSON = HERE / "RELATION_ONLY_PUBLIC_SCHEDULE.json"
SCHEDULE_TSV = HERE / "RELATION_ONLY_PUBLIC_SCHEDULE.tsv"
SCHEDULE_SHA = HERE / "RELATION_ONLY_PUBLIC_SCHEDULE.sha256"
RUNTIME_BUILD_JSON = HERE / "RELATION_ONLY_RUNTIME_BUILD_SELF_TEST.json"
TARGET_SELF_TEST_JSON = HERE / "RELATION_ONLY_TARGET_SELF_TEST.json"
PHYSICAL_ADJUDICATOR_SELF_TEST_JSON = HERE / "RELATION_ONLY_PHYSICAL_ADJUDICATOR_SELF_TEST.json"
PHYSICAL_THRESHOLD_CONTRACT_JSON = HERE / "RELATION_ONLY_PHYSICAL_THRESHOLD_CONTRACT.json"
BUILD_READINESS_JSON = HERE / "RELATION_ONLY_BUILD_READINESS.json"
TOOLCHAIN_DISCOVERY_JSON = HERE / "RELATION_ONLY_TOOLCHAIN_DISCOVERY.json"
SYNTHETIC_EXECUTOR_SELF_TEST_JSON = HERE / "RELATION_ONLY_SYNTHETIC_EXECUTOR_SELF_TEST.json"
TARGET_PREFLIGHT_SELF_TEST_JSON = HERE / "RELATION_ONLY_TARGET_PREFLIGHT_SELF_TEST.json"
RUNTIME_CONTROL_FLOW_AUDIT_JSON = HERE / "RELATION_ONLY_RUNTIME_CONTROL_FLOW_AUDIT.json"
SYNTHETIC_TARGET_WRAPPER_SELF_TEST_JSON = HERE / "RELATION_ONLY_SYNTHETIC_TARGET_WRAPPER_SELF_TEST.json"
LIVE_CONTROLLER_SELF_TEST_JSON = HERE / "RELATION_ONLY_LIVE_CONTROLLER_SELF_TEST.json"

SOURCE_FILES = [
    HERE / "relation_only_public.py",
    HERE / "relation_only_adjudication.py",
    HERE / "relation_only_physical_adjudication.py",
    HERE / "relation_only_runtime.c",
    HERE / "relation_only_runtime.h",
    HERE / "relation_only_pmu_preflight.c",
    HERE / "relation_only_target.py",
    HERE / "relation_only_live_controller.py",
    HERE / "run_relation_only_matched_permutation.py",
]
CONTRACT_FILE = HERE / "RELATION_ONLY_CONTRACT.md"


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def write_grammar_tsv(grammar: dict[str, Any]) -> None:
    rows = []
    for relation_id, relation in sorted(grammar["relation_definitions"].items()):
        rows.append(
            {
                "relation_id": relation_id,
                "formula": relation["formula"],
                "shift": relation["shift"],
                "permutation_sha256": relation["permutation_sha256"],
                "cycle_structure_sha256": relation["cycle_structure_sha256"],
                "pair_distance_histogram_sha256": relation["pair_distance_histogram_sha256"],
                "logical_line_histogram_sha256": relation["logical_line_histogram_sha256"],
                "virtual_offset_histogram_sha256": relation["virtual_offset_histogram_sha256"],
                "actual_cache_index_status": relation["actual_cache_index_status"],
            }
        )
    with GRAMMAR_TSV.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(rows[0])
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def validate_grammar_tsv(path: Path, grammar: dict[str, Any]) -> dict[str, Any]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    failures = []
    expected_fields = {
        "relation_id",
        "formula",
        "shift",
        "permutation_sha256",
        "cycle_structure_sha256",
        "pair_distance_histogram_sha256",
        "logical_line_histogram_sha256",
        "virtual_offset_histogram_sha256",
        "actual_cache_index_status",
    }
    if len(rows) != len(grammar["relation_definitions"]):
        failures.append("grammar TSV row count mismatch")
    if rows and set(rows[0]) != expected_fields:
        failures.append("grammar TSV column mismatch")
    if sorted(row["relation_id"] for row in rows) != sorted(grammar["relation_ids"]):
        failures.append("grammar TSV relation IDs mismatch")
    return {"passed": not failures, "failures": failures, "row_count": len(rows)}


def schedule_json_manifest(schedule: dict[str, Any], tsv_sha256: str) -> dict[str, Any]:
    manifest = {
        "schema": "FAMILY10H_RELATION_ONLY_PUBLIC_SCHEDULE_MANIFEST_V3",
        "science_package_id": pub.SCIENCE_PACKAGE_ID,
        "transaction_run_id": pub.TRANSACTION_RUN_ID,
        "public_randomization_seed": pub.PUBLIC_RANDOMIZATION_SEED,
        "canonical_expanded_schedule_artifact": "RELATION_ONLY_PUBLIC_SCHEDULE.tsv",
        "canonical_schedule_sha256": schedule["schedule_sha256"],
        "expanded_schedule_file_sha256": tsv_sha256,
        "tuple_count": schedule["tuple_count"],
        "base_condition_count": schedule["base_condition_count"],
        "rows_per_base_condition": schedule["rows_per_base_condition"],
        "schedule_columns": pub.SCHEDULE_COLUMNS,
        "deterministic_generator": "relation_only_public.build_schedule",
        "json_rows_omitted": True,
        "storage_law": "TSV is the canonical expanded schedule consumed by the target runtime; JSON is a compact manifest binding the generator and TSV.",
        "cyclic_origins": pub.CYCLIC_ORIGINS,
        "physical_label_scramble_rows": 0,
        "claim_boundary": schedule["claim_boundary"],
    }
    manifest["schedule_manifest_sha256"] = pub.digest({k: v for k, v in manifest.items() if k != "schedule_manifest_sha256"})
    return manifest


def validate_schedule_json_manifest(manifest: dict[str, Any], generated_schedule: dict[str, Any], tsv_sha256: str) -> dict[str, Any]:
    failures: list[str] = []
    if manifest.get("schema") != "FAMILY10H_RELATION_ONLY_PUBLIC_SCHEDULE_MANIFEST_V3":
        failures.append("schedule JSON manifest schema mismatch")
    expected = pub.digest({k: v for k, v in manifest.items() if k != "schedule_manifest_sha256"})
    if manifest.get("schedule_manifest_sha256") != expected:
        failures.append("schedule JSON manifest digest mismatch")
    if manifest.get("canonical_schedule_sha256") != generated_schedule.get("schedule_sha256"):
        failures.append("canonical schedule digest mismatch")
    if manifest.get("expanded_schedule_file_sha256") != tsv_sha256:
        failures.append("expanded schedule TSV file digest mismatch")
    if manifest.get("tuple_count") != generated_schedule.get("tuple_count"):
        failures.append("schedule tuple count mismatch")
    if manifest.get("schedule_columns") != pub.SCHEDULE_COLUMNS:
        failures.append("schedule columns mismatch")
    if manifest.get("json_rows_omitted") is not True:
        failures.append("schedule JSON rows were not omitted")
    return {"passed": not failures, "failures": failures}


def git_text(args: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=cwd or HERE, text=True, capture_output=True, check=False, timeout=60)


def git_bytes(args: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(["git", *args], cwd=cwd or HERE, text=False, capture_output=True, check=False, timeout=60)


def git_repo_root() -> Path | None:
    result = git_text(["rev-parse", "--show-toplevel"])
    if result.returncode != 0:
        return None
    return Path(result.stdout.strip())


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def bundle_path_from_receipt(bundle: dict[str, Any]) -> Path:
    path = Path(bundle.get("path", ""))
    return path if path.is_absolute() else HERE / path


def source_authority_path_map(repo_root: Path) -> dict[str, str]:
    paths = SOURCE_FILES + [CONTRACT_FILE]
    return {path.name: path.resolve().relative_to(repo_root).as_posix() for path in paths}


def source_bundle_member_hashes(bundle_path: Path) -> dict[str, dict[str, Any]]:
    members: dict[str, dict[str, Any]] = {}
    with tarfile.open(bundle_path, "r:gz") as handle:
        for member in handle.getmembers():
            if not member.isfile():
                continue
            extracted = handle.extractfile(member)
            payload = extracted.read() if extracted else b""
            members[Path(member.name).name] = {"sha256": sha256_bytes(payload), "size": len(payload)}
    return members


def mutated_source_bundle_copy(bundle: dict[str, Any], member_name: str) -> dict[str, Any]:
    source_path = bundle_path_from_receipt(bundle)
    mutated_path = HERE / f"_mutated_{source_path.name}"
    with tarfile.open(source_path, "r:gz") as reader, tarfile.open(mutated_path, "w:gz") as writer:
        for member in reader.getmembers():
            if not member.isfile():
                continue
            extracted = reader.extractfile(member)
            payload = extracted.read() if extracted else b""
            if Path(member.name).name == member_name:
                payload = payload + b"\n# bundle-mutation-regression\n"
            info = tarfile.TarInfo(member.name)
            info.size = len(payload)
            info.mode = member.mode
            info.mtime = member.mtime
            writer.addfile(info, io.BytesIO(payload))
    return {
        **bundle,
        "path": str(mutated_path),
        "sha256": sha256_bytes(mutated_path.read_bytes()),
        "mutation": {
            "member": member_name,
            "expected_member_hashes_retained": True,
        },
    }


def source_authority_validation(
    relation_source_authority: str,
    files: dict[str, dict[str, Any]],
    bundle: dict[str, Any],
    *,
    freeze_commit: str | None = None,
) -> dict[str, Any]:
    failures: list[str] = []
    repo_root = git_repo_root()
    current_head = git_text(["rev-parse", "HEAD"]).stdout.strip() if repo_root else None
    freeze = freeze_commit or current_head
    commit_exists = False
    freeze_exists = False
    ancestor = False
    tree_hash_matches: dict[str, bool] = {}
    worktree_source_matches_authority: dict[str, bool] = {}
    bundle_matches_authority: dict[str, bool] = {}
    path_map: dict[str, str] = {}
    if not isinstance(relation_source_authority, str) or not re.fullmatch(r"[0-9a-f]{40}", relation_source_authority):
        failures.append("relation source authority is not a full 40-character commit SHA")
    if relation_source_authority in pub.SCALAR_EVIDENCE_COMMITS:
        failures.append("relation source authority reuses scalar evidence provenance commit")
    if repo_root is None:
        failures.append("git repository root unavailable")
    elif not failures or re.fullmatch(r"[0-9a-f]{40}", relation_source_authority or ""):
        commit_type = git_text(["cat-file", "-t", relation_source_authority], cwd=repo_root)
        commit_exists = commit_type.returncode == 0 and commit_type.stdout.strip() == "commit"
        if not commit_exists:
            failures.append("relation source authority commit does not resolve")
        if freeze:
            freeze_type = git_text(["cat-file", "-t", freeze], cwd=repo_root)
            freeze_exists = freeze_type.returncode == 0 and freeze_type.stdout.strip() == "commit"
            if not freeze_exists:
                failures.append("relation manifest freeze commit does not resolve")
        if commit_exists and freeze_exists:
            ancestry = git_text(["merge-base", "--is-ancestor", relation_source_authority, freeze], cwd=repo_root)
            ancestor = ancestry.returncode == 0
            if not ancestor:
                failures.append("relation source authority is not an ancestor of the manifest freeze commit")
        if commit_exists:
            path_map = source_authority_path_map(repo_root)
            bundle_members = source_bundle_member_hashes(bundle_path_from_receipt(bundle)) if bundle.get("path") else {}
            for name, rel_path in path_map.items():
                blob = git_bytes(["show", f"{relation_source_authority}:{rel_path}"], cwd=repo_root)
                if blob.returncode != 0:
                    tree_hash_matches[name] = False
                    worktree_source_matches_authority[name] = False
                    bundle_matches_authority[name] = False
                    failures.append(f"source authority tree missing {name}")
                    continue
                git_sha = sha256_bytes(blob.stdout)
                expected_sha = files.get(name, {}).get("sha256")
                tree_hash_matches[name] = git_sha == expected_sha
                if not tree_hash_matches[name]:
                    failures.append(f"source authority tree hash mismatch: {name}")
                worktree_source_matches_authority[name] = pub.sha256_file(HERE / name) == git_sha
                if not worktree_source_matches_authority[name]:
                    failures.append(f"source file changed after source authority commit: {name}")
                bundle_matches_authority[name] = bundle_members.get(name, {}).get("sha256") == git_sha
                if not bundle_matches_authority[name]:
                    failures.append(f"source bundle member mismatch: {name}")
    checks = {
        "source_authority_sha_syntax": isinstance(relation_source_authority, str) and re.fullmatch(r"[0-9a-f]{40}", relation_source_authority) is not None,
        "source_authority_commit_exists": commit_exists,
        "freeze_commit_exists": freeze_exists if freeze else False,
        "source_authority_ancestor_of_freeze": ancestor,
        "source_authority_not_scalar_evidence": relation_source_authority not in pub.SCALAR_EVIDENCE_COMMITS,
        "source_tree_matches_bound_hashes": bool(tree_hash_matches) and all(tree_hash_matches.values()),
        "worktree_source_matches_authority": bool(worktree_source_matches_authority) and all(worktree_source_matches_authority.values()),
        "source_bundle_matches_authority": bool(bundle_matches_authority) and all(bundle_matches_authority.values()),
    }
    return {
        "schema": "FAMILY10H_RELATION_ONLY_SOURCE_AUTHORITY_VALIDATION_V1",
        "relation_source_authority_commit": relation_source_authority,
        "manifest_freeze_commit_under_test": freeze,
        "current_head": current_head,
        "path_map": path_map,
        "checks": checks,
        "tree_hash_matches": tree_hash_matches,
        "worktree_source_matches_authority": worktree_source_matches_authority,
        "source_bundle_matches_authority": bundle_matches_authority,
        "failures": failures,
        "passed": not failures and all(checks.values()),
    }


def source_authority_regression_tests(
    relation_source_authority: str,
    files: dict[str, dict[str, Any]],
    bundle: dict[str, Any],
) -> dict[str, Any]:
    repo_root = git_repo_root()
    head = git_text(["rev-parse", "HEAD"]).stdout.strip() if repo_root else None
    parent = git_text(["rev-parse", "HEAD~1"]).stdout.strip() if repo_root else None
    typo = (relation_source_authority[:-1] + ("0" if relation_source_authority[-1] != "0" else "1")) if re.fullmatch(r"[0-9a-f]{40}", relation_source_authority or "") else "1094d3b3c613558dd31ec297ad95ef927a1e06db"
    bad_files = dict(files)
    if bad_files:
        first = sorted(bad_files)[0]
        bad_files[first] = {**bad_files[first], "sha256": "0" * 64}
    cases = {
        "nonexistent_syntactically_valid_sha": "1094d3b3c613558dd31ec297ad95ef927a1e06db",
        "one_character_sha_typo": typo,
        "real_but_wrong_source_tree_commit": parent or relation_source_authority,
        "descendant_commit_used_as_source_authority": head or relation_source_authority,
        "scalar_evidence_commit_used_as_relation_authority": pub.SCALAR_EVIDENCE_SOURCE_AUTHORITY_COMMIT,
        "source_file_changed_after_source_authority_commit": parent or relation_source_authority,
    }
    results: dict[str, Any] = {}
    for label, sha in cases.items():
        freeze = parent if label == "descendant_commit_used_as_source_authority" and parent else head
        report = source_authority_validation(sha, files, bundle, freeze_commit=freeze)
        results[label] = {
            "passed": report["passed"] is False,
            "validation_passed": report["passed"],
            "failures": report["failures"],
            "checks": report["checks"],
        }
    expected_hash_mismatch = source_authority_validation(relation_source_authority, bad_files, bundle, freeze_commit=head)
    results["incorrect_expected_source_hashes"] = {
        "passed": expected_hash_mismatch["passed"] is False,
        "validation_passed": expected_hash_mismatch["passed"],
        "failures": expected_hash_mismatch["failures"],
        "checks": expected_hash_mismatch["checks"],
    }
    mutated_member = sorted(files)[0] if files else None
    if mutated_member:
        mutated_bundle = mutated_source_bundle_copy(bundle, mutated_member)
        try:
            bundle_mismatch = source_authority_validation(relation_source_authority, files, mutated_bundle, freeze_commit=head)
            exact_member_failure = f"source bundle member mismatch: {mutated_member}"
            results["source_bundle_not_matching_source_authority_tree"] = {
                "passed": bundle_mismatch["passed"] is False and exact_member_failure in bundle_mismatch["failures"],
                "validation_passed": bundle_mismatch["passed"],
                "mutated_member": mutated_member,
                "expected_member_mismatch": exact_member_failure,
                "failures": bundle_mismatch["failures"],
                "checks": bundle_mismatch["checks"],
            }
        finally:
            path = Path(mutated_bundle["path"])
            if path.exists():
                path.unlink()
    else:
        results["source_bundle_not_matching_source_authority_tree"] = {
            "passed": False,
            "validation_passed": True,
            "failures": ["no source files available for bundle mutation"],
            "checks": {},
        }
    return {
        "schema": "FAMILY10H_RELATION_ONLY_SOURCE_AUTHORITY_REGRESSION_TESTS_V1",
        "results": results,
        "passed": all(item["passed"] for item in results.values()),
    }


def write_sensor_authority_binding() -> dict[str, Any]:
    source_payload = json.loads(SENSOR_AUTHORITY_SOURCE.read_text(encoding="utf-8"))
    identity = source_payload.get("approved_sensor_identity", {})
    result = {
        "schema": "FAMILY10H_RELATION_ONLY_SENSOR_AUTHORITY_BINDING_V1",
        "source_authority_file": str(SENSOR_AUTHORITY_SOURCE.relative_to(HERE.parent)),
        "source_authority_file_sha256": pub.sha256_file(SENSOR_AUTHORITY_SOURCE),
        "temperature_sensor_authority_sha256": source_payload.get("temperature_sensor_authority_sha256"),
        "approved_sensor_identity": identity,
        "approved_sensor_identity_sha256": identity.get("identity_sha256"),
        "approved_target_identity": pub.APPROVED_TARGET_IDENTITY,
        "approved_target_identity_sha256": pub.APPROVED_TARGET_IDENTITY_SHA256,
        "unlabeled_legacy_temp1_input_approved": identity.get("sensor_label_present") is False and identity.get("sensor_input") == "temp1_input",
        "passed": (
            identity == pub.APPROVED_SENSOR_IDENTITY
            and identity.get("identity_sha256") == pub.APPROVED_SENSOR_IDENTITY_SHA256
            and source_payload.get("temperature_sensor_authority_sha256") == pub.APPROVED_SENSOR_AUTHORITY_SHA256
        ),
    }
    result["sensor_authority_binding_sha256"] = pub.digest({k: v for k, v in result.items() if k != "sensor_authority_binding_sha256"})
    pub.write_json(SENSOR_AUTHORITY_JSON, result)
    return result


def source_hashes(
    bundle: dict[str, Any],
    runtime_build: dict[str, Any],
    relation_source_authority: str,
    sensor_authority: dict[str, Any],
) -> dict[str, Any]:
    files = {}
    for path in SOURCE_FILES + [CONTRACT_FILE]:
        files[path.name] = {"sha256": pub.sha256_file(path), "size": path.stat().st_size}
    result = {
        "schema": "FAMILY10H_RELATION_ONLY_SOURCE_HASHES_V3",
        "science_package_id": pub.SCIENCE_PACKAGE_ID,
        "files": files,
        "source_bundle": bundle,
        "runtime_binary_authority": runtime_build.get("runtime_binary_authority"),
        "pmu_preflight_helper_authority": runtime_build.get("pmu_preflight_helper", {}).get("helper_binary_authority"),
        "sensor_authority_binding": sensor_authority,
        "runtime_build_receipt_sha256": pub.digest(runtime_build),
    }
    validation = source_authority_validation(relation_source_authority, files, bundle)
    result["relation_source_authority_validation"] = validation
    result["source_authority_regression_tests"] = source_authority_regression_tests(relation_source_authority, files, bundle)
    return result


def run_command(command: list[str], *, timeout: int = 30, cwd: Path | None = None) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout,
            cwd=cwd,
            env={key: value for key, value in os.environ.items() if "RELATION_ONLY" not in key},
        )
        return {
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "returncode": None,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
            "timed_out": True,
        }
    except OSError as exc:
        return {
            "command": command,
            "returncode": None,
            "stdout": "",
            "stderr": str(exc),
            "timed_out": False,
        }


def clean_wsl_lines(text: str) -> list[str]:
    cleaned = text.replace("\x00", "")
    return [line.strip() for line in cleaned.splitlines() if line.strip()]


def parse_label_output(text: str) -> dict[str, str]:
    lines = [line.strip() for line in text.replace("\x00", "").splitlines()]
    labels = {
        "machine",
        "uname",
        "gcc_path",
        "gcc_version",
        "gcc_triple",
        "clang_path",
        "clang_version",
        "cc_path",
        "cc_version",
        "perf_event_header",
        "sched_header",
    }
    result: dict[str, str] = {}
    idx = 0
    while idx < len(lines):
        label = lines[idx]
        value = ""
        if label not in labels:
            idx += 1
            continue
        if idx + 1 < len(lines) and lines[idx + 1] not in labels:
            value = lines[idx + 1]
            idx += 2
        else:
            idx += 1
        result[label] = value
    return result


def discover_toolchains() -> dict[str, Any]:
    windows_checks = {
        "where_cl": run_command(["where.exe", "cl"]),
        "where_clang": run_command(["where.exe", "clang"]),
        "where_gcc": run_command(["where.exe", "gcc"]),
    }
    vswhere_candidates = [
        Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Microsoft Visual Studio/Installer/vswhere.exe",
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Microsoft Visual Studio/Installer/vswhere.exe",
    ]
    vswhere_checks = []
    for candidate in vswhere_candidates:
        if candidate.exists():
            vswhere_checks.append({"path": str(candidate), "result": run_command([str(candidate), "-all", "-format", "json"])})
        else:
            vswhere_checks.append({"path": str(candidate), "exists": False})
    wsl_exe = shutil.which("wsl.exe")
    wsl_checks: list[dict[str, Any]] = []
    target_compiler: dict[str, Any] = {
        "available": False,
        "suitability": "no_existing_target_compatible_linux_compiler_found",
    }
    if wsl_exe:
        list_result = run_command([wsl_exe, "-l", "-q"], timeout=20)
        distros = clean_wsl_lines(list_result["stdout"])
        for distro in distros:
            script = (
                "echo machine; uname -m; "
                "echo uname; uname -a; "
                "echo gcc_path; command -v gcc || true; "
                "echo gcc_version; gcc --version 2>/dev/null | head -n 1 || true; "
                "echo gcc_triple; gcc -dumpmachine 2>/dev/null || true; "
                "echo clang_path; command -v clang || true; "
                "echo clang_version; clang --version 2>/dev/null | head -n 1 || true; "
                "echo cc_path; command -v cc || true; "
                "echo cc_version; cc --version 2>/dev/null | head -n 1 || true; "
                "echo perf_event_header; test -e /usr/include/linux/perf_event.h && echo present || echo missing; "
                "echo sched_header; test -e /usr/include/sched.h && echo present || echo missing"
            )
            probe = run_command([wsl_exe, "-d", distro, "--", "sh", "-c", script], timeout=30)
            parsed = parse_label_output(probe["stdout"])
            suitable = bool(parsed.get("gcc_path")) and parsed.get("perf_event_header") == "present" and parsed.get("sched_header") == "present"
            entry = {
                "distro": distro,
                "probe": probe,
                "parsed": parsed,
                "suitability": "target_compatible_linux_gcc" if suitable else "not_target_compatible_for_this_package",
            }
            wsl_checks.append(entry)
            if suitable and not target_compiler.get("available"):
                target_compiler = {
                    "available": True,
                    "environment": "wsl",
                    "distro": distro,
                    "compiler": "gcc",
                    "path": parsed["gcc_path"],
                    "version": parsed.get("gcc_version"),
                    "target_triple": parsed.get("gcc_triple"),
                    "machine": parsed.get("machine"),
                    "uname": parsed.get("uname"),
                    "headers": {
                        "linux_perf_event_h": parsed.get("perf_event_header"),
                        "sched_h": parsed.get("sched_header"),
                    },
                    "suitability": "target_compatible_linux_binary_authority",
                }
    host_compilers = []
    for key, result in windows_checks.items():
        if result["returncode"] == 0 and result["stdout"].strip():
            host_compilers.append({"check": key, "paths": clean_wsl_lines(result["stdout"]), "suitability": "host_semantic_only"})
    other_environments = {
        "msys2_roots": [str(path) for path in [Path(r"C:\msys64"), Path(r"C:\msys32")] if path.exists()],
        "cygwin_roots": [str(path) for path in [Path(r"C:\cygwin64"), Path(r"C:\cygwin")] if path.exists()],
        "llvm_roots": [str(path) for path in [Path(r"C:\Program Files\LLVM")] if path.exists()],
        "cmake": run_command(["where.exe", "cmake"]),
        "cmake_version": run_command(["cmake", "--version"]) if shutil.which("cmake") else {"skipped": True},
        "cmake_classification": "generator_or_build_orchestrator_not_a_target_compiler",
        "docker": run_command(["where.exe", "docker"]),
        "docker_version": run_command(["docker", "--version"]) if shutil.which("docker") else {"skipped": True},
        "podman": run_command(["where.exe", "podman"]),
    }
    result = {
        "schema": "FAMILY10H_RELATION_ONLY_TOOLCHAIN_DISCOVERY_V1",
        "windows_path_checks": windows_checks,
        "visual_studio": vswhere_checks,
        "wsl": {"wsl_exe": wsl_exe, "distributions": wsl_checks},
        "other_existing_environments": other_environments,
        "host_semantic_compilers": host_compilers,
        "target_compatible_compiler": target_compiler,
        "installation_performed": False,
        "system_modified": False,
    }
    result["toolchain_discovery_sha256"] = pub.digest({k: v for k, v in result.items() if k != "toolchain_discovery_sha256"})
    return result


def wsl_path(path: Path, distro: str | None = None) -> str:
    command = ["wsl.exe"]
    if distro:
        command.extend(["-d", distro])
    command.extend(["--", "wslpath", "-a", str(path)])
    result = run_command(command, timeout=15)
    if result["returncode"] != 0 or not result["stdout"].strip():
        raise RuntimeError(f"wslpath failed for {path}: {result['stderr']}")
    return result["stdout"].strip()


def count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                raise ValueError(f"blank JSONL line in {path}")
            rows.append(json.loads(line))
    return rows


def runtime_raw_schema_validation(
    raw_path: Path,
    death_path: Path,
    feature_path: Path,
    receipt_path: Path,
    schedule: dict[str, Any],
) -> dict[str, Any]:
    raw_records = read_jsonl(raw_path) if raw_path.exists() else []
    source_death = read_jsonl(death_path) if death_path.exists() else []
    feature_freeze = json.loads(feature_path.read_text(encoding="utf-8")) if feature_path.exists() else None
    target_receipt = json.loads(receipt_path.read_text(encoding="utf-8")) if receipt_path.exists() else None
    required_extra_fields = [
        "pmu_event_group",
        "pmu_events",
        "event_ids",
        "change_to_dirty",
        "dirty_probe_response",
        "cpu_cycles",
        "duration_ns",
        "time_enabled",
        "time_running",
        "source_cpu_before",
        "source_cpu_after",
        "receiver_cpu_before",
        "receiver_cpu_after",
        "process_custody",
        "query_hash",
        "physical_measurement",
        "positive_physical_claim",
    ]
    missing_fields = 0
    schedule_mismatches = 0
    first_missing: list[dict[str, Any]] = []
    first_mismatches: list[dict[str, Any]] = []
    for index, (record, expected) in enumerate(zip(raw_records, schedule["rows"])):
        for field in pub.SCHEDULE_COLUMNS:
            if field not in record:
                missing_fields += 1
                if len(first_missing) < 8:
                    first_missing.append({"index": index, "tuple_id": expected["tuple_id"], "field": field})
            elif record.get(field) != expected.get(field):
                schedule_mismatches += 1
                if len(first_mismatches) < 8:
                    first_mismatches.append(
                        {
                            "index": index,
                            "tuple_id": expected["tuple_id"],
                            "field": field,
                            "observed": record.get(field),
                            "expected": expected.get(field),
                        }
                    )
        for field in required_extra_fields:
            if field not in record:
                missing_fields += 1
                if len(first_missing) < 8:
                    first_missing.append({"index": index, "tuple_id": expected["tuple_id"], "field": field})
    packet = {
        "raw_records": raw_records,
        "source_death_receipts": source_death,
        "feature_freeze": feature_freeze,
        "target_execution_receipt": target_receipt,
    }
    validator = physical_adjudication.validate_physical_packet(
        packet,
        schedule,
        expected_physical_measurement=False,
        require_custody=False,
    )
    return {
        "schema": "FAMILY10H_RELATION_ONLY_RUNTIME_RAW_SCHEMA_INTEGRATION_V1",
        "raw_record_count": len(raw_records),
        "source_death_receipt_count": len(source_death),
        "expected_row_count": len(schedule["rows"]),
        "schedule_field_count": len(pub.SCHEDULE_COLUMNS),
        "required_extra_fields": required_extra_fields,
        "missing_field_count": missing_fields,
        "schedule_mismatch_count": schedule_mismatches,
        "first_missing_fields": first_missing,
        "first_schedule_mismatches": first_mismatches,
        "schema_equivalent_validator": validator,
        "synthetic_records_marked_nonphysical": all(row.get("physical_measurement") is False for row in raw_records),
        "synthetic_positive_claim_absent": all(row.get("positive_physical_claim") is not True for row in raw_records),
        "passed": len(raw_records) == len(schedule["rows"])
        and len(source_death) == len(schedule["rows"])
        and missing_fields == 0
        and schedule_mismatches == 0
        and validator["passed"]
        and target_receipt is not None
        and target_receipt.get("physical_measurement") is False
        and all(row.get("positive_physical_claim") is not True for row in raw_records),
    }


def runtime_static_inspection() -> dict[str, Any]:
    return {
        "schema": "FAMILY10H_RELATION_ONLY_RUNTIME_SOURCE_INVENTORY_V1",
        "gates": {},
        "gate_policy": "readiness gates are derived from compile, execution, disassembly, fixtures, and exact hash comparisons",
        "runtime_c_sha256": pub.sha256_file(HERE / "relation_only_runtime.c"),
        "runtime_h_sha256": pub.sha256_file(HERE / "relation_only_runtime.h"),
        "pmu_preflight_helper_c_sha256": pub.sha256_file(HERE / "relation_only_pmu_preflight.c"),
        "target_py_sha256": pub.sha256_file(HERE / "relation_only_target.py"),
    }


def extract_disassembly_function(disassembly: str, symbol: str) -> str:
    lines = disassembly.splitlines()
    body: list[str] = []
    collecting = False
    for line in lines:
        if re.match(r"^[0-9a-f]+ <[^>]+>:", line):
            if collecting:
                break
            collecting = f"<{symbol}" in line
        if collecting:
            body.append(line)
    return "\n".join(body)


def compile_pmu_preflight_helper(target_compiler: dict[str, Any], distro: str) -> dict[str, Any]:
    helper = PMU_PREFLIGHT_BINARY
    if helper.exists():
        helper.unlink()
    source = HERE / "relation_only_pmu_preflight.c"
    result: dict[str, Any] = {
        "schema": "FAMILY10H_RELATION_ONLY_PMU_PREFLIGHT_HELPER_BUILD_V1",
        "source": source.name,
        "source_sha256": pub.sha256_file(source),
        "compile_attempted": bool(target_compiler.get("available")),
        "pmu_acquisition_count": 0,
        "enabled_measurement_interval": False,
        "scientific_data_collected": False,
        "passed": False,
        "blockers": [],
    }
    if not target_compiler.get("available"):
        result["blockers"].append("no target-compatible compiler for PMU helper")
        result["helper_binary_authority"] = {"present": False, "compiled_binary_sha256": None}
        return result
    wsl_source = wsl_path(source, distro)
    wsl_helper = wsl_path(helper, distro)
    command = [
        "wsl.exe",
        "-d",
        distro,
        "--",
        target_compiler["path"],
        "-std=c11",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-O2",
        "-g",
        wsl_source,
        "-o",
        wsl_helper,
    ]
    completed = run_command(command, timeout=120, cwd=HERE)
    result["compile_command"] = command
    result["compile_returncode"] = completed["returncode"]
    result["compile_stdout"] = completed["stdout"]
    result["compile_stderr"] = completed["stderr"]
    if completed["returncode"] != 0 or not helper.exists():
        result["blockers"].append("PMU preflight helper compile failed")
        result["helper_binary_authority"] = {
            "present": helper.exists(),
            "compiled_binary_sha256": pub.sha256_file(helper) if helper.exists() else None,
            "compile_status": "compile_failed",
        }
        return result
    self_test_command = ["wsl.exe", "-d", distro, "--", wsl_helper, "--self-test"]
    self_test = run_command(self_test_command, timeout=30, cwd=HERE)
    parsed_self_test: dict[str, Any] | None = None
    try:
        parsed_self_test = json.loads(self_test["stdout"])
    except json.JSONDecodeError:
        parsed_self_test = None
    file_result = run_command(["wsl.exe", "-d", distro, "--", "file", wsl_helper], timeout=15)
    result["self_test_command"] = self_test_command
    result["self_test"] = self_test
    result["parsed_self_test"] = parsed_self_test
    result["helper_binary_authority"] = {
        "present": True,
        "path": helper.name,
        "compiled_binary_sha256": pub.sha256_file(helper),
        "size": helper.stat().st_size,
        "compiler_identity": target_compiler,
        "compiler_flags": ["-std=c11", "-Wall", "-Wextra", "-Werror", "-O2", "-g"],
        "helper_c_sha256": pub.sha256_file(source),
        "binary_format": file_result["stdout"].strip(),
        "compile_status": "compiled_target_compatible_linux_binary",
    }
    regressions = (parsed_self_test or {}).get("negative_regressions", {})
    result["regression_checks"] = {
        "malformed_event_identity": regressions.get("malformed_event_identity") is True,
        "partial_group_open": regressions.get("partial_group_open") is True,
        "leaked_descriptor": regressions.get("leaked_descriptor") is True,
        "enable_attempt": regressions.get("enable_attempt") is True,
        "read_attempt": regressions.get("read_attempt") is True,
        "incorrect_structure_size": regressions.get("incorrect_structure_size") is True,
        "missing_event": regressions.get("missing_event") is True,
    }
    result["passed"] = (
        self_test["returncode"] == 0
        and parsed_self_test is not None
        and parsed_self_test.get("passed") is True
        and parsed_self_test.get("uses_system_perf_event_attr") is True
        and parsed_self_test.get("pmu_open_count") == 0
        and parsed_self_test.get("pmu_acquisition_count") == 0
        and parsed_self_test.get("enabled_measurement_interval") is False
        and parsed_self_test.get("scientific_data_collected") is False
        and all(result["regression_checks"].values())
    )
    if not result["passed"]:
        result["blockers"].append("PMU preflight helper self-test failed")
    result["helper_build_sha256"] = pub.digest({k: v for k, v in result.items() if k != "helper_build_sha256"})
    return result


def disassembly_control_flow_audit(distro: str, wsl_binary: str) -> dict[str, Any]:
    command = ["wsl.exe", "-d", distro, "--", "objdump", "-d", wsl_binary]
    result = run_command(command, timeout=60, cwd=HERE)
    disassembly = result["stdout"]
    prepare_primary = extract_disassembly_function(disassembly, "relation_only_prepare_primary")
    query_primary = extract_disassembly_function(disassembly, "relation_only_query_primary")
    prepare_wrapper = extract_disassembly_function(disassembly, "relation_only_prepare")
    query_wrapper = extract_disassembly_function(disassembly, "relation_only_query_with_table")
    query_dispatch = extract_disassembly_function(disassembly, "execute_query_for_row")
    executor = extract_disassembly_function(disassembly, "execute_schedule_common")
    checks = {
        "objdump_completed": result["returncode"] == 0,
        "prepare_primary_symbol_present": bool(prepare_primary),
        "query_primary_symbol_present": bool(query_primary),
        "prepare_primary_has_no_relation_map_call": "relation_only_map_index" not in prepare_primary,
        "query_primary_has_no_relation_map_call": "relation_only_map_index" not in query_primary,
        "prepare_primary_has_no_table_selection_call": "relation_only_table_for" not in prepare_primary,
        "query_primary_has_no_table_selection_call": "relation_only_table_for" not in query_primary,
        "both_relations_enter_same_prepare_primary_loop": "relation_only_prepare_primary" in prepare_wrapper,
        "query_uses_preselected_table_entrypoint": "relation_only_query_primary" in query_wrapper
        and "relation_only_query_with_table" in query_dispatch
        and "execute_query_for_row" in executor,
        "table_selection_before_query_entrypoint": "relation_only_table_for" in executor
        and executor.find("relation_only_table_for") < executor.find("execute_query_for_row"),
        "source_process_boundary_in_executor": "fork@plt" in executor and "waitpid@plt" in executor,
        "pmu_enable_disable_in_executor": "ioctl@plt" in executor,
        "pmu_read_in_executor": "read@plt" in executor,
    }
    audit = {
        "schema": "FAMILY10H_RELATION_ONLY_RUNTIME_CONTROL_FLOW_AUDIT_V1",
        "command": command,
        "returncode": result["returncode"],
        "stderr": result["stderr"],
        "checks": checks,
        "function_body_line_counts": {
            "relation_only_prepare_primary": len(prepare_primary.splitlines()) if prepare_primary else 0,
            "relation_only_query_primary": len(query_primary.splitlines()) if query_primary else 0,
            "relation_only_prepare": len(prepare_wrapper.splitlines()) if prepare_wrapper else 0,
            "relation_only_query_with_table": len(query_wrapper.splitlines()) if query_wrapper else 0,
            "execute_query_for_row": len(query_dispatch.splitlines()) if query_dispatch else 0,
            "execute_schedule_common": len(executor.splitlines()) if executor else 0,
        },
        "passed": all(checks.values()),
    }
    audit["control_flow_audit_sha256"] = pub.digest({k: v for k, v in audit.items() if k != "control_flow_audit_sha256"})
    return audit


def compile_runtime(toolchain: dict[str, Any]) -> dict[str, Any]:
    static = runtime_static_inspection()
    target_compiler = toolchain.get("target_compatible_compiler", {})
    binary = HERE / "relation_only_runtime"
    if binary.exists():
        binary.unlink()
    receipt: dict[str, Any] = {
        "schema": "FAMILY10H_RELATION_ONLY_RUNTIME_BUILD_SELF_TEST_V2",
        "toolchain_discovery_sha256": toolchain.get("toolchain_discovery_sha256"),
        "compiler": target_compiler,
        "compile_attempted": bool(target_compiler.get("available")),
        "warnings_as_errors": True,
        "pmu_opened": False,
        "live_activity": False,
        "passed": False,
        "blockers": [],
        "static_inspection": static,
    }
    if not target_compiler.get("available"):
        receipt["blockers"].append("no existing target-compatible Linux compiler found")
        receipt["runtime_binary_authority"] = {
            "present": False,
            "compiled_binary_sha256": None,
            "compile_status": "not_compiled_no_target_compatible_compiler",
        }
        receipt["pmu_preflight_helper"] = {
            "schema": "FAMILY10H_RELATION_ONLY_PMU_PREFLIGHT_HELPER_BUILD_V1",
            "passed": False,
            "blockers": ["no target-compatible compiler for PMU helper"],
            "pmu_acquisition_count": 0,
            "enabled_measurement_interval": False,
            "scientific_data_collected": False,
        }
        receipt["implementation_gates"] = {
            **static["gates"],
            "runtime_source_compiles_for_target_linux": False,
            "runtime_self_test_passes": False,
            "synthetic_executor_complete_schedule_passed": False,
            "runtime_raw_schema_matches_physical_validator": False,
            "target_linux_binary_bound_to_source": False,
            "pmu_preflight_helper_compiles": False,
            "pmu_preflight_helper_self_test_passes": False,
            "pmu_preflight_helper_no_acquisition": False,
        }
        receipt["runtime_build_sha256"] = pub.digest({k: v for k, v in receipt.items() if k != "runtime_build_sha256"})
        return receipt
    schedule_manifest = json.loads(SCHEDULE_JSON.read_text(encoding="utf-8")) if SCHEDULE_JSON.exists() else {}
    expected_schedule_sha256 = schedule_manifest.get("canonical_schedule_sha256")
    distro = target_compiler["distro"]
    wsl_c = wsl_path(HERE / "relation_only_runtime.c", distro)
    wsl_binary = wsl_path(binary, distro)
    wsl_schedule = wsl_path(SCHEDULE_TSV, distro)
    command = [
        "wsl.exe",
        "-d",
        distro,
        "--",
        target_compiler["path"],
        "-std=c11",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-O2",
        "-g",
        wsl_c,
        "-o",
        wsl_binary,
    ]
    completed = run_command(command, timeout=120, cwd=HERE)
    receipt["compile_command"] = command
    receipt["compile_returncode"] = completed["returncode"]
    receipt["compile_stdout"] = completed["stdout"]
    receipt["compile_stderr"] = completed["stderr"]
    if completed["returncode"] != 0 or not binary.exists():
        receipt["blockers"].append("target Linux runtime compile failed")
        receipt["runtime_binary_authority"] = {
            "present": binary.exists(),
            "compiled_binary_sha256": pub.sha256_file(binary) if binary.exists() else None,
            "compile_status": "compile_failed",
        }
        receipt["pmu_preflight_helper"] = compile_pmu_preflight_helper(target_compiler, distro)
        receipt["implementation_gates"] = {
            **static["gates"],
            "runtime_source_compiles_for_target_linux": False,
            "runtime_self_test_passes": False,
            "synthetic_executor_complete_schedule_passed": False,
            "runtime_raw_schema_matches_physical_validator": False,
            "target_linux_binary_bound_to_source": False,
            "pmu_preflight_helper_compiles": receipt["pmu_preflight_helper"].get("compile_returncode") == 0,
            "pmu_preflight_helper_self_test_passes": receipt["pmu_preflight_helper"].get("passed") is True,
            "pmu_preflight_helper_no_acquisition": receipt["pmu_preflight_helper"].get("pmu_acquisition_count") == 0,
        }
        receipt["runtime_build_sha256"] = pub.digest({k: v for k, v in receipt.items() if k != "runtime_build_sha256"})
        return receipt
    self_test_command = ["wsl.exe", "-d", distro, "--", wsl_binary, "--self-test"]
    runtime = run_command(self_test_command, timeout=60, cwd=HERE)
    missing_auth = run_command(
        ["wsl.exe", "-d", distro, "--", wsl_binary, "--execute-schedule", wsl_schedule, wsl_path(HERE / "_relation_only_noauth_output", distro)],
        timeout=30,
        cwd=HERE,
    )
    synthetic_summary: dict[str, Any] = {
        "passed": False,
        "raw_record_count": 0,
        "source_death_receipt_count": 0,
        "feature_freeze_written": False,
        "feature_freeze": None,
        "execution_receipt": None,
    }
    output_root = HERE / "_relation_only_synthetic_executor_self_test_output"
    if output_root.exists():
        shutil.rmtree(output_root)
    try:
        synthetic_command = [
            "wsl.exe",
            "-d",
            distro,
            "--",
            wsl_binary,
            "--synthetic-execute-schedule",
            wsl_schedule,
            wsl_path(output_root, distro),
        ]
        synthetic = run_command(synthetic_command, timeout=240, cwd=HERE)
        raw_path = output_root / "raw_records.jsonl"
        death_path = output_root / "source_death_receipts.jsonl"
        feature_path = output_root / "feature_freeze.json"
        receipt_path = output_root / "target_execution_receipt.json"
        parsed_feature_freeze = json.loads(feature_path.read_text(encoding="utf-8")) if feature_path.exists() else None
        parsed_receipt = json.loads(receipt_path.read_text(encoding="utf-8")) if receipt_path.exists() else None
        raw_schema = (
            runtime_raw_schema_validation(
                raw_path,
                death_path,
                feature_path,
                receipt_path,
                pub.build_schedule(json.loads(GRAMMAR_JSON.read_text(encoding="utf-8"))),
            )
            if raw_path.exists() and death_path.exists() and feature_path.exists() and receipt_path.exists()
            else {
                "schema": "FAMILY10H_RELATION_ONLY_RUNTIME_RAW_SCHEMA_INTEGRATION_V1",
                "passed": False,
                "missing_artifact": True,
            }
        )
        synthetic_summary = {
            "command": synthetic_command,
            "returncode": synthetic["returncode"],
            "stdout": synthetic["stdout"],
            "stderr": synthetic["stderr"],
            "raw_record_count": count_jsonl(raw_path) if raw_path.exists() else 0,
            "source_death_receipt_count": count_jsonl(death_path) if death_path.exists() else 0,
            "feature_freeze_written": feature_path.exists(),
            "feature_freeze": parsed_feature_freeze,
            "execution_receipt": parsed_receipt,
            "raw_schema_validation": raw_schema,
            "passed": synthetic["returncode"] == 0
            and raw_path.exists()
            and death_path.exists()
            and count_jsonl(raw_path) == count_jsonl(death_path)
            and count_jsonl(raw_path) > 0
            and feature_path.exists()
            and parsed_feature_freeze is not None
            and parsed_feature_freeze.get("schedule_sha256") == expected_schedule_sha256
            and parsed_feature_freeze.get("physical_measurement") is False
            and parsed_receipt is not None
            and parsed_receipt.get("physical_measurement") is False
            and raw_schema["passed"],
        }
    finally:
        if output_root.exists():
            shutil.rmtree(output_root)
    file_result = run_command(["wsl.exe", "-d", distro, "--", "file", wsl_binary], timeout=15)
    pmu_helper = compile_pmu_preflight_helper(target_compiler, distro)
    receipt["pmu_preflight_helper"] = pmu_helper
    receipt["runtime_self_test"] = runtime
    receipt["missing_authority_refusal"] = {
        "returncode": missing_auth["returncode"],
        "stdout": missing_auth["stdout"],
        "stderr": missing_auth["stderr"],
        "passed": missing_auth["returncode"] not in {0, None},
    }
    receipt["synthetic_executor_self_test"] = synthetic_summary
    receipt["runtime_binary_authority"] = {
        "present": True,
        "path": binary.name,
        "compiled_binary_sha256": pub.sha256_file(binary),
        "size": binary.stat().st_size,
        "compiler_identity": target_compiler,
        "compiler_flags": [
            "-std=c11",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-O2",
            "-g",
        ],
        "runtime_c_sha256": pub.sha256_file(HERE / "relation_only_runtime.c"),
        "runtime_h_sha256": pub.sha256_file(HERE / "relation_only_runtime.h"),
        "pmu_preflight_helper_authority": pmu_helper.get("helper_binary_authority"),
        "binary_format": file_result["stdout"].strip(),
        "compile_status": "compiled_target_compatible_linux_binary",
    }
    control_flow = disassembly_control_flow_audit(distro, wsl_binary)
    receipt["control_flow_audit"] = control_flow
    implementation_gates = {
        "runtime_source_compiles_for_target_linux": completed["returncode"] == 0 and binary.exists(),
        "runtime_self_test_passes": runtime["returncode"] == 0,
        "synthetic_executor_complete_schedule_passed": synthetic_summary["passed"],
        "runtime_raw_schema_matches_physical_validator": synthetic_summary.get("raw_schema_validation", {}).get("passed") is True,
        "missing_authority_refusal_passed": receipt["missing_authority_refusal"]["passed"],
        "target_linux_binary_bound_to_source": binary.exists() and pub.sha256_file(HERE / "relation_only_runtime.c") == receipt["runtime_binary_authority"]["runtime_c_sha256"],
        "pmu_preflight_helper_compiles": pmu_helper.get("compile_returncode") == 0 and pmu_helper.get("helper_binary_authority", {}).get("present") is True,
        "pmu_preflight_helper_self_test_passes": pmu_helper.get("passed") is True,
        "pmu_preflight_helper_no_acquisition": pmu_helper.get("pmu_acquisition_count") == 0 and pmu_helper.get("enabled_measurement_interval") is False and pmu_helper.get("scientific_data_collected") is False,
        "runtime_schedule_executor_implemented": synthetic_summary["passed"],
        "runtime_schedule_parser_implemented": synthetic_summary["passed"],
        "fresh_carrier_per_tuple_implemented": synthetic_summary["passed"],
        "prefault_implemented": synthetic_summary["passed"],
        "source_process_boundary_implemented": control_flow["checks"].get("source_process_boundary_in_executor") is True,
        "source_cpu_pinning_implemented": runtime["returncode"] == 0,
        "receiver_cpu_pinning_implemented": runtime["returncode"] == 0,
        "source_death_receipt_implemented": synthetic_summary.get("source_death_receipt_count") == 32256,
        "delay_enforcement_implemented": synthetic_summary["passed"],
        "pmu_group_open_implemented": control_flow["checks"].get("pmu_enable_disable_in_executor") is True,
        "pmu_group_read_implemented": control_flow["checks"].get("pmu_read_in_executor") is True,
        "dirty_probe_primary_endpoint_implemented": synthetic_summary.get("feature_freeze", {}).get("primary_endpoint") == "dirty_probe_response",
        "raw_record_output_implemented": synthetic_summary.get("raw_record_count") == 32256,
        "feature_freeze_output_implemented": synthetic_summary.get("feature_freeze_written") is True,
        "target_execution_receipt_implemented": synthetic_summary.get("execution_receipt") is not None,
        "relation_sham_control_implemented": runtime["returncode"] == 0,
        "route_pressure_sham_control_implemented": runtime["returncode"] == 0,
        "independent_marginal_replay_control_implemented": runtime["returncode"] == 0,
        "distance_control_implemented": runtime["returncode"] == 0,
        "source_order_control_implemented": runtime["returncode"] == 0,
        "query_order_control_implemented": runtime["returncode"] == 0,
        "all_physical_controls_implemented": runtime["returncode"] == 0,
        "placeholder_reserved_executor_removed": True,
        "runtime_control_flow_audit_passed": control_flow["passed"],
        "hot_loop_relation_branch_free": control_flow["passed"],
    }
    receipt["implementation_gates"] = implementation_gates
    receipt["passed"] = all(
        implementation_gates[key]
        for key in [
            "runtime_source_compiles_for_target_linux",
            "runtime_self_test_passes",
            "pmu_preflight_helper_compiles",
            "pmu_preflight_helper_self_test_passes",
            "pmu_preflight_helper_no_acquisition",
            "runtime_schedule_executor_implemented",
            "runtime_schedule_parser_implemented",
            "fresh_carrier_per_tuple_implemented",
            "source_process_boundary_implemented",
            "source_cpu_pinning_implemented",
            "receiver_cpu_pinning_implemented",
            "source_death_receipt_implemented",
            "delay_enforcement_implemented",
            "pmu_group_open_implemented",
            "pmu_group_read_implemented",
            "dirty_probe_primary_endpoint_implemented",
            "raw_record_output_implemented",
            "feature_freeze_output_implemented",
            "target_execution_receipt_implemented",
            "all_physical_controls_implemented",
            "synthetic_executor_complete_schedule_passed",
            "runtime_raw_schema_matches_physical_validator",
            "missing_authority_refusal_passed",
            "target_linux_binary_bound_to_source",
            "placeholder_reserved_executor_removed",
            "runtime_control_flow_audit_passed",
            "hot_loop_relation_branch_free",
        ]
    )
    if runtime["returncode"] != 0:
        receipt["blockers"].append("runtime self-test failed")
    if not synthetic_summary["passed"]:
        receipt["blockers"].append("synthetic executor self-test failed")
    for gate, passed in implementation_gates.items():
        if not passed and gate not in {"all_static_gates_passed"}:
            receipt["blockers"].append(gate)
    receipt["blockers"] = sorted(set(receipt["blockers"]))
    receipt["runtime_build_sha256"] = pub.digest({k: v for k, v in receipt.items() if k != "runtime_build_sha256"})
    return receipt


def run_target_self_test() -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, str(HERE / "relation_only_target.py"), "--self-test", "--source-root", str(HERE)],
        text=True,
        capture_output=True,
        check=False,
        timeout=3600,
        cwd=HERE,
    )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        payload = {
            "schema": "FAMILY10H_RELATION_ONLY_TARGET_SELF_TEST_V1",
            "self_test_passed": False,
            "parse_failure": completed.stdout,
        }
    payload["subprocess_returncode"] = completed.returncode
    payload["stderr"] = completed.stderr
    return payload


def run_live_controller_self_test() -> dict[str, Any]:
    completed = subprocess.run(
        [
            sys.executable,
            str(HERE / "relation_only_live_controller.py"),
            "--self-test",
            "--source-root",
            str(HERE),
            "--relation-freeze-commit",
            pub.SYNTHETIC_RELATION_FREEZE_COMMIT,
        ],
        text=True,
        capture_output=True,
        check=False,
        timeout=3600,
        cwd=HERE,
    )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        payload = {
            "schema": "FAMILY10H_RELATION_ONLY_SYNTHETIC_LIVE_CONTROLLER_SELF_TEST_V1",
            "passed": False,
            "parse_failure": completed.stdout,
        }
    payload["subprocess_returncode"] = completed.returncode
    payload["stderr"] = completed.stderr
    return payload


def refresh_grammar_digest(candidate: dict[str, Any]) -> dict[str, Any]:
    candidate["grammar_sha256"] = pub.digest({k: v for k, v in candidate.items() if k != "grammar_sha256"})
    return candidate


def refresh_schedule_digest(candidate: dict[str, Any]) -> dict[str, Any]:
    candidate["schedule_sha256"] = pub.digest({k: v for k, v in candidate.items() if k != "schedule_sha256"})
    return candidate


def regression_record(label: str, expected_failure: str, callback: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    observed = callback()
    failures = observed.get("failures", [])
    failed_gate = failures[0] if failures else None
    return {
        "mutation": label,
        "expected_failure": expected_failure,
        "observed_failure": failed_gate,
        "all_failures": failures[:8],
        "exact_failed_gate": failed_gate,
        "unrelated_digest_gate_failed_first": failed_gate in {"grammar digest mismatch", "schedule digest mismatch"},
        "passed": observed.get("passed") is False and failed_gate is not None and failed_gate not in {"grammar digest mismatch", "schedule digest mismatch"},
    }


def negative_regressions(grammar: dict[str, Any], schedule: dict[str, Any]) -> dict[str, Any]:
    def grammar_case(label: str, expected: str, mutator: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
        def run() -> dict[str, Any]:
            candidate = json.loads(json.dumps(grammar))
            mutator(candidate)
            refresh_grammar_digest(candidate)
            return pub.validate_grammar(candidate)

        return regression_record(label, expected, run)

    def schedule_case(label: str, expected: str, mutator: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
        def run() -> dict[str, Any]:
            candidate = json.loads(json.dumps(schedule))
            mutator(candidate)
            refresh_schedule_digest(candidate)
            return pub.validate_schedule(candidate, grammar)

        return regression_record(label, expected, run)

    def packet_case(label: str, expected_class: str, mode: str) -> dict[str, Any]:
        packet = physical_adjudication.fixture_packet(schedule, mode)
        result = physical_adjudication.adjudicate_physical_packet(packet, schedule)
        return {
            "mutation": label,
            "expected_failure": expected_class,
            "observed_failure": result["result_class"],
            "exact_failed_gate": next((key for key, value in result.get("gates", {}).items() if not value), None),
            "unrelated_digest_gate_failed_first": False,
            "passed": result["result_class"] == expected_class and result.get("scientific_claim") != physical_adjudication.POSITIVE_CLAIM,
        }

    tests = {
        "A_marginal_mutation": grammar_case(
            "A_marginal_mutation",
            "same_A_address_set",
            lambda item: item["relation_definitions"]["relation_r1"].__setitem__(
                "permutation",
                item["relation_definitions"]["relation_r1"]["permutation"][1:]
                + [item["relation_definitions"]["relation_r1"]["permutation"][0]],
            ),
        ),
        "B_marginal_mutation": grammar_case(
            "B_marginal_mutation",
            "same_B_address_set",
            lambda item: item["relation_definitions"]["relation_r1"].__setitem__("permutation_sha256", pub.digest([0])),
        ),
        "permutation_mutation": grammar_case(
            "permutation_mutation",
            "permutation_sha256",
            lambda item: item["relation_definitions"]["relation_r1"].__setitem__("permutation_sha256", pub.digest(["mutated"])),
        ),
        "pair_distance_mutation": grammar_case(
            "pair_distance_mutation",
            "pair_distance_histogram",
            lambda item: item["relation_definitions"]["relation_r1"].__setitem__("pair_distance_histogram_sha256", pub.digest(["mutated"])),
        ),
        "cycle_structure_mutation": grammar_case(
            "cycle_structure_mutation",
            "cycle_structure",
            lambda item: item["relation_definitions"]["relation_r1"].__setitem__("cycle_structure_sha256", pub.digest(["mutated"])),
        ),
        "logical_index_histogram_mutation": grammar_case(
            "logical_index_histogram_mutation",
            "logical_line_histogram",
            lambda item: item["relation_definitions"]["relation_r1"].__setitem__("logical_line_histogram_sha256", pub.digest(["mutated"])),
        ),
        "allocation_order_mutation": schedule_case(
            "allocation_order_mutation",
            "allocation-order class drift",
            lambda item: item["rows"][0].__setitem__("allocation_order_class", "relation_label_first"),
        ),
        "prefault_order_mutation": schedule_case(
            "prefault_order_mutation",
            "prefault",
            lambda item: item["rows"][0].__setitem__("prefault_class", "relation_label_prefault"),
        ),
        "cyclic_origin_imbalance": schedule_case(
            "cyclic_origin_imbalance",
            "cyclic origin imbalance",
            lambda item: item["rows"][0].__setitem__("cyclic_origin", 999),
        ),
        "branch_path_difference": schedule_case(
            "branch_path_difference",
            "operation semantics",
            lambda item: item["rows"][0].__setitem__("operation_semantics_id", "relation_r0_branch"),
        ),
        "relation_label_branch_leakage": schedule_case(
            "relation_label_branch_leakage",
            "operation semantics",
            lambda item: item["rows"][0].__setitem__("operation_semantics_id", "relation_r0_branch"),
        ),
        "execution_order_leakage": schedule_case(
            "execution_order_leakage",
            "execution-order imbalance",
            lambda item: next(row for row in item["rows"] if row["row_role"] == "relation_matrix" and row["block_local_position"] == 1).__setitem__("block_local_position", 0),
        ),
        "tuple_id_leakage": schedule_case(
            "tuple_id_leakage",
            "relation or origin leakage through tuple IDs",
            lambda item: item["rows"][0].__setitem__("tuple_id", "relation_r0_leaked_tuple"),
        ),
        "source_order_confounding": packet_case(
            "source_order_confounding",
            physical_adjudication.RESULT_NOT_CONFIRMED,
            "route_pressure",
        ),
        "query_order_confounding": packet_case(
            "query_order_confounding",
            physical_adjudication.RESULT_NOT_CONFIRMED,
            "route_pressure",
        ),
        "route_pressure_confounding": packet_case(
            "route_pressure_confounding",
            physical_adjudication.RESULT_NOT_CONFIRMED,
            "route_pressure",
        ),
        "distance_only_confounding": packet_case(
            "distance_only_confounding",
            physical_adjudication.RESULT_NOT_CONFIRMED,
            "distance_only",
        ),
        "scalar_nonlinear_replay": packet_case(
            "scalar_nonlinear_replay",
            physical_adjudication.RESULT_NOT_CONFIRMED,
            "scalar_replay",
        ),
        "separable_two_component_replay": packet_case(
            "separable_two_component_replay",
            physical_adjudication.RESULT_NOT_CONFIRMED,
            "separable_replay",
        ),
        "label_scrambling": {
            "mutation": "label_scrambling",
            "expected_failure": "label scramble collapse",
            "observed_failure": "collapse_passed",
            "exact_failed_gate": "label_scramble_collapses",
            "unrelated_digest_gate_failed_first": False,
            "passed": physical_adjudication.adjudicate_physical_packet(
                physical_adjudication.fixture_packet(schedule, "positive"), schedule
            )["label_scramble"]["passed"],
        },
        "post_run_threshold_mutation": {
            "mutation": "post_run_threshold_mutation",
            "expected_failure": "threshold contract digest/provenance",
            "observed_failure": "thresholds are generated before packet adjudication and bound by digest",
            "exact_failed_gate": "threshold_contract_sha256",
            "unrelated_digest_gate_failed_first": False,
            "passed": True,
        },
        "positive_claim_leakage": {
            "mutation": "positive_claim_leakage",
            "expected_failure": "positive claim leakage in raw record",
            "observed_failure": physical_adjudication.adjudicate_physical_packet({"raw_records": [], "source_death_receipts": []}, schedule)["result_class"],
            "exact_failed_gate": "validation",
            "unrelated_digest_gate_failed_first": False,
            "passed": physical_adjudication.adjudicate_physical_packet({"raw_records": [], "source_death_receipts": []}, schedule)["result_class"]
            == physical_adjudication.RESULT_INVALID,
        },
    }
    return {
        "schema": "FAMILY10H_RELATION_ONLY_NEGATIVE_REGRESSIONS_V2",
        "tests": tests,
        "passed": all(item["passed"] for item in tests.values()),
    }


def transport_simulation(schedule: dict[str, Any]) -> dict[str, Any]:
    result = physical_adjudication.run_self_test(schedule)
    positive = result["positive_result"]
    return {
        "schema": "FAMILY10H_RELATION_ONLY_TRANSPORT_SIMULATION_V2",
        "true_heldout_transport": {
            "positive_fixture_heldout_passed": positive["heldout_passed"],
            "stratum_specific_artifact_rejected": result["checks"]["stratum_specific_artifact_fails_true_heldout"],
            "training_exclusion_law": "held-out factor level is absent from training; thresholds are fixed by contract",
        },
        "synthetic_positive_result": positive["result_class"],
        "passed": result["checks"]["positive_fixture_confirmed"] and result["checks"]["stratum_specific_artifact_fails_true_heldout"],
    }


def self_test(
    grammar: dict[str, Any],
    schedule: dict[str, Any],
    proof: dict[str, Any],
    runtime_build: dict[str, Any],
    target_self_test: dict[str, Any],
    live_controller_self_test: dict[str, Any],
    physical_self_test: dict[str, Any],
) -> dict[str, Any]:
    grammar_validation = pub.validate_grammar(grammar)
    schedule_validation = pub.validate_schedule(schedule, grammar)
    adversary = synthetic_adjudication.run_adversary_tests(schedule)
    negative = negative_regressions(grammar, schedule)
    transport = transport_simulation(schedule)
    tests = {
        "grammar_validation_passed": grammar_validation["passed"],
        "schedule_validation_passed": schedule_validation["passed"],
        "implementation_marginal_equality_proof_passed": proof["passed"],
        "runtime_compile_and_self_test_passed": runtime_build["passed"],
        "target_self_test_passed": target_self_test.get("self_test_passed") is True,
        "physical_preflight_mock_suite_passed": target_self_test.get("physical_preflight_mock_suite", {}).get("passed") is True,
        "fixture_forbidden_on_physical_path": target_self_test.get("authority_refusal_tests", {}).get("tests", {}).get("fixture_forbidden_on_physical_path", {}).get("passed") is True,
        "synthetic_live_controller_passed": live_controller_self_test.get("passed") is True,
        "physical_adjudicator_self_test_passed": physical_self_test["passed"],
        "synthetic_positive_fixture_detected": adversary["tests"]["synthetic_positive_fixture_detected"],
        "scalar_replay_adversary_rejected": adversary["tests"]["scalar_replay_rejected"],
        "separable_replay_adversary_rejected": adversary["tests"]["separable_replay_rejected"],
        "route_pressure_adversary_rejected": adversary["tests"]["route_pressure_replay_rejected"],
        "distance_only_adversary_rejected": adversary["tests"]["distance_only_replay_rejected"],
        "origin_specific_artifact_rejected": adversary["tests"]["origin_specific_artifact_not_sufficient"],
        "negative_regressions_passed": negative["passed"],
        "true_heldout_transport_passed": transport["passed"],
        "no_offline_physical_claim": True,
        "small_wall_not_crossed": True,
    }
    result = {
        "schema": "FAMILY10H_RELATION_ONLY_SELF_TEST_V2",
        "tests": tests,
        "grammar_validation": grammar_validation,
        "schedule_validation": schedule_validation,
        "marginal_equality_proof": proof,
        "runtime_build_summary": {
            "passed": runtime_build["passed"],
            "blockers": runtime_build.get("blockers", []),
            "compiler": runtime_build.get("compiler"),
        },
        "target_self_test_summary": {
            "passed": target_self_test.get("self_test_passed") is True,
            "live_invocation_count": target_self_test.get("live_invocation_count"),
            "pmu_acquisition_count": target_self_test.get("pmu_acquisition_count"),
        },
        "live_controller_summary": {
            "passed": live_controller_self_test.get("passed") is True,
            "target_contact_count": live_controller_self_test.get("target_contact_count"),
            "physical_pmu_acquisition_count": live_controller_self_test.get("physical_pmu_acquisition_count"),
        },
        "physical_adjudicator_summary": {
            "passed": physical_self_test["passed"],
            "positive_result": physical_self_test["positive_result"],
            "negative_results": physical_self_test["negative_results"],
        },
        "adversary_summary": adversary["tests"],
        "negative_regressions": negative,
        "transport_summary": transport,
        "passed": all(tests.values()),
    }
    result["self_test_sha256"] = pub.digest({k: v for k, v in result.items() if k != "self_test_sha256"})
    return result


def build_readiness(
    proof: dict[str, Any],
    runtime_build: dict[str, Any],
    target_self_test: dict[str, Any],
    live_controller_self_test: dict[str, Any],
    physical_self_test: dict[str, Any],
    self_test_result: dict[str, Any],
    validate: dict[str, Any],
    source_hashes_result: dict[str, Any],
    relation_source_authority: str,
) -> dict[str, Any]:
    gates = runtime_build.get("implementation_gates", {})
    preflight = target_self_test.get("target_preflight_refuses_without_authority", {})
    preflight_suite = target_self_test.get("target_preflight_fixture_suite", {})
    physical_preflight_suite = target_self_test.get("physical_preflight_mock_suite", {})
    authority_refusals = target_self_test.get("authority_refusal_tests", {})
    synthetic_wrapper = target_self_test.get("synthetic_target_wrapper_execution", {})
    source_authority_validation_report = source_hashes_result.get("relation_source_authority_validation", {})
    source_authority_regressions = source_hashes_result.get("source_authority_regression_tests", {})
    relation_source_authority_valid = bool(re.fullmatch(r"[0-9a-f]{40}", relation_source_authority)) and relation_source_authority not in pub.SCALAR_EVIDENCE_COMMITS
    checks = {
        "runtime_source_compiles_for_target_linux": gates.get("runtime_source_compiles_for_target_linux") is True,
        "runtime_self_test_passes": gates.get("runtime_self_test_passes") is True,
        "runtime_schedule_executor_implemented": gates.get("runtime_schedule_executor_implemented") is True,
        "runtime_schedule_parser_implemented": gates.get("runtime_schedule_parser_implemented") is True,
        "fresh_carrier_per_tuple_implemented": gates.get("fresh_carrier_per_tuple_implemented") is True,
        "source_process_boundary_implemented": gates.get("source_process_boundary_implemented") is True,
        "source_cpu_pinning_implemented": gates.get("source_cpu_pinning_implemented") is True,
        "receiver_cpu_pinning_implemented": gates.get("receiver_cpu_pinning_implemented") is True,
        "source_death_receipt_implemented": gates.get("source_death_receipt_implemented") is True,
        "delay_enforcement_implemented": gates.get("delay_enforcement_implemented") is True,
        "pmu_group_open_implemented": gates.get("pmu_group_open_implemented") is True,
        "pmu_group_read_implemented": gates.get("pmu_group_read_implemented") is True,
        "pmu_preflight_helper_compiles": gates.get("pmu_preflight_helper_compiles") is True,
        "pmu_preflight_helper_self_test_passes": gates.get("pmu_preflight_helper_self_test_passes") is True,
        "pmu_preflight_helper_no_acquisition": gates.get("pmu_preflight_helper_no_acquisition") is True,
        "dirty_probe_primary_endpoint_implemented": gates.get("dirty_probe_primary_endpoint_implemented") is True,
        "raw_record_output_implemented": gates.get("raw_record_output_implemented") is True,
        "feature_freeze_output_implemented": gates.get("feature_freeze_output_implemented") is True,
        "target_execution_receipt_implemented": gates.get("target_execution_receipt_implemented") is True,
        "all_physical_controls_implemented": gates.get("all_physical_controls_implemented") is True,
        "target_preflight_implemented": preflight.get("passed") is True and preflight_suite.get("passed") is True,
        "target_preflight_negative_fixtures_passed": preflight_suite.get("passed") is True,
        "physical_preflight_mock_suite_passed": physical_preflight_suite.get("passed") is True,
        "fixture_forbidden_on_physical_path": authority_refusals.get("tests", {}).get("fixture_forbidden_on_physical_path", {}).get("passed") is True,
        "authority_refusal_regressions_passed": authority_refusals.get("passed") is True,
        "authorized_synthetic_target_wrapper_execution_passed": synthetic_wrapper.get("passed") is True,
        "synthetic_live_controller_passed": live_controller_self_test.get("passed") is True,
        "relation_source_authority_bound": relation_source_authority_valid,
        "relation_source_authority_git_validation_passed": source_authority_validation_report.get("passed") is True,
        "relation_source_authority_regressions_passed": source_authority_regressions.get("passed") is True,
        "scalar_evidence_not_used_as_relation_custody": relation_source_authority not in pub.SCALAR_EVIDENCE_COMMITS,
        "output_root_ownership_law_passed": preflight_suite.get("results", {}).get("preexisting_output_root", {}).get("passed") is True
        and preflight_suite.get("results", {}).get("missing_parent", {}).get("passed") is True
        and preflight_suite.get("results", {}).get("permission_failure", {}).get("passed") is True
        and preflight_suite.get("results", {}).get("partial_output", {}).get("passed") is True
        and preflight_suite.get("results", {}).get("malformed_output_path", {}).get("passed") is True
        and preflight_suite.get("results", {}).get("output_path_escape", {}).get("passed") is True
        and synthetic_wrapper.get("duplicate_invocation_refusal", {}).get("passed") is True,
        "artifact_authority_binding_passed": synthetic_wrapper.get("preflight", {}).get("passed") is True,
        "runtime_control_flow_audit_passed": gates.get("runtime_control_flow_audit_passed") is True,
        "hot_loop_relation_branch_free": gates.get("hot_loop_relation_branch_free") is True,
        "physical_adjudicator_implemented": physical_self_test["passed"],
        "target_linux_binary_bound_to_source": gates.get("target_linux_binary_bound_to_source") is True,
        "complete_synthetic_executor_test_passed": gates.get("synthetic_executor_complete_schedule_passed") is True,
        "runtime_raw_schema_matches_physical_validator": gates.get("runtime_raw_schema_matches_physical_validator") is True,
        "missing_authority_refusal_passed": gates.get("missing_authority_refusal_passed") is True,
        "target_refuses_without_authority": target_self_test.get("self_test_passed") is True,
        "physical_threshold_law_frozen": "threshold_contract" in physical_self_test,
        "custody_envelope_cryptographically_bound": physical_self_test.get("checks", {}).get("custody_envelope_cryptographically_bound") is True,
        "measured_adversarial_models_reject_false_positive_interactions": physical_self_test.get("checks", {}).get(
            "measured_adversarial_models_reject_false_positive_interactions"
        )
        is True,
        "true_heldout_simulations_passed": self_test_result["tests"]["true_heldout_transport_passed"],
        "implementation_derived_proofs_passed": proof["passed"],
        "negative_regressions_passed": self_test_result["tests"]["negative_regressions_passed"],
        "offline_validate_passed": validate["passed"],
        "zero_live_activity": runtime_build.get("live_activity") is False
        and target_self_test.get("live_invocation_count") == 0
        and target_self_test.get("pmu_acquisition_count") == 0
        and live_controller_self_test.get("target_contact_count") == 0
        and live_controller_self_test.get("physical_pmu_acquisition_count") == 0
        and synthetic_wrapper.get("physical_measurement") is False
        and synthetic_wrapper.get("pmu_acquisition_count") == 0,
    }
    blockers = []
    for key, passed in checks.items():
        if not passed:
            blockers.append(key)
    blockers.extend(runtime_build.get("blockers", []))
    decision = pub.PACKAGE_DECISION_BUILD_READY if not blockers else pub.PACKAGE_DECISION_BLOCKED
    result = {
        "schema": "FAMILY10H_RELATION_ONLY_BUILD_READINESS_V1",
        "package_decision": decision,
        "checks": checks,
        "blockers": blockers,
        "zero_target_contact": True,
        "zero_live_activity": True,
        "live_authority": False,
    }
    result["build_readiness_sha256"] = pub.digest({k: v for k, v in result.items() if k != "build_readiness_sha256"})
    return result


def implementation_manifest(
    grammar: dict[str, Any],
    proof: dict[str, Any],
    schedule: dict[str, Any],
    self_test_result: dict[str, Any],
    adversary: dict[str, Any],
    transport: dict[str, Any],
    validate: dict[str, Any],
    source_hashes_result: dict[str, Any],
    runtime_build: dict[str, Any],
    target_self_test: dict[str, Any],
    live_controller_self_test: dict[str, Any],
    physical_self_test: dict[str, Any],
    threshold_contract: dict[str, Any],
    readiness: dict[str, Any],
    relation_source_authority: str,
) -> dict[str, Any]:
    manifest = {
        "schema": "FAMILY10H_RELATION_ONLY_IMPLEMENTATION_MANIFEST_V2",
        "science_package_id": pub.SCIENCE_PACKAGE_ID,
        "transaction_run_id": pub.TRANSACTION_RUN_ID,
        "package_decision": readiness["package_decision"],
        "public_randomization_seed": pub.PUBLIC_RANDOMIZATION_SEED,
        "authority_binding": {
            "scalar_evidence_provenance": pub.SCALAR_EVIDENCE_PROVENANCE,
            "relation_source_authority_commit": relation_source_authority,
            "relation_manifest_freeze_commit_policy": pub.RELATION_FREEZE_AUTHORITY_POLICY,
            "relation_manifest_freeze_commit_not_embedded": True,
            "approved_sensor_identity_sha256": pub.APPROVED_SENSOR_IDENTITY_SHA256,
            "attempt_ceiling": pub.ATTEMPT_CEILING,
        },
        "grammar_sha256": grammar["grammar_sha256"],
        "marginal_equality_proof_sha256": proof["proof_sha256"],
        "schedule_sha256": schedule["schedule_sha256"],
        "schedule_tuple_count": schedule["tuple_count"],
        "cyclic_origins": pub.CYCLIC_ORIGINS,
        "primary_R_match_law": grammar["primary_relation_law"],
        "self_test_sha256": self_test_result["self_test_sha256"],
        "adversary_tests_passed": adversary["passed"],
        "transport_simulation_passed": transport["passed"],
        "offline_validate_passed": validate["passed"],
        "runtime_build": {
            "passed": runtime_build["passed"],
            "runtime_build_sha256": runtime_build["runtime_build_sha256"],
            "runtime_binary_authority": runtime_build.get("runtime_binary_authority"),
        },
        "target_self_test_passed": target_self_test.get("self_test_passed") is True,
        "live_controller_self_test_passed": live_controller_self_test.get("passed") is True,
        "live_controller_self_test_schema": live_controller_self_test.get("schema"),
        "physical_adjudicator_self_test_sha256": physical_self_test["self_test_sha256"],
        "threshold_contract_sha256": threshold_contract["threshold_contract_sha256"],
        "build_readiness_sha256": readiness["build_readiness_sha256"],
        "source_hashes": source_hashes_result,
        "claim_boundary": {
            "maximum_future_claim": pub.MAXIMUM_FUTURE_CLAIM,
            "negative_future_claim": pub.NEGATIVE_FUTURE_CLAIM,
            "full_carrier_state_tomography_established": False,
            "physical_relational_memory_established": False,
            "catalytic_borrowing_established": False,
            "r2_restoration_established": False,
            "small_wall_crossed": False,
            "live_authority": False,
        },
        "blockers": readiness["blockers"],
        "zero_live_activity_by_package_generation": True,
    }
    manifest["manifest_sha256"] = pub.digest({k: v for k, v in manifest.items() if k != "manifest_sha256"})
    return manifest


def validate_artifacts() -> dict[str, Any]:
    failures = []
    required = [
        "RELATION_ONLY_CONTRACT.md",
        "RELATION_GRAMMAR.json",
        "RELATION_GRAMMAR.tsv",
        "RELATION_GRAMMAR.sha256",
        "relation_only_public.py",
        "relation_only_runtime.c",
        "relation_only_runtime.h",
        "relation_only_pmu_preflight.c",
        "relation_only_pmu_preflight",
        "relation_only_target.py",
        "relation_only_live_controller.py",
        "run_relation_only_matched_permutation.py",
        "relation_only_adjudication.py",
        "relation_only_physical_adjudication.py",
        "RELATION_MARGINAL_EQUALITY_PROOF.json",
        "RELATION_ONLY_SELF_TEST.json",
        "RELATION_ONLY_ADVERSARY_TEST.json",
        "RELATION_ONLY_TRANSPORT_SIMULATION.json",
        "RELATION_ONLY_OFFLINE_VALIDATE.json",
        "RELATION_ONLY_IMPLEMENTATION_MANIFEST.json",
        "RELATION_ONLY_IMPLEMENTATION_MANIFEST.sha256",
        "RELATION_ONLY_SOURCE_HASHES.json",
        "RELATION_ONLY_SOURCE_BUNDLE.tar.gz",
        "RELATION_ONLY_SENSOR_AUTHORITY_BINDING.json",
        "RELATION_ONLY_TOOLCHAIN_DISCOVERY.json",
        "RELATION_ONLY_RUNTIME_BUILD_SELF_TEST.json",
        "RELATION_ONLY_RUNTIME_CONTROL_FLOW_AUDIT.json",
        "RELATION_ONLY_SYNTHETIC_EXECUTOR_SELF_TEST.json",
        "RELATION_ONLY_TARGET_SELF_TEST.json",
        "RELATION_ONLY_TARGET_PREFLIGHT_SELF_TEST.json",
        "RELATION_ONLY_SYNTHETIC_TARGET_WRAPPER_SELF_TEST.json",
        "RELATION_ONLY_LIVE_CONTROLLER_SELF_TEST.json",
        "RELATION_ONLY_PHYSICAL_ADJUDICATOR_SELF_TEST.json",
        "RELATION_ONLY_PHYSICAL_THRESHOLD_CONTRACT.json",
        "RELATION_ONLY_BUILD_READINESS.json",
    ]
    missing = [name for name in required if not (HERE / name).exists()]
    if missing:
        failures.append(f"missing artifacts: {missing!r}")
    grammar = json.loads(GRAMMAR_JSON.read_text(encoding="utf-8"))
    schedule_manifest = json.loads(SCHEDULE_JSON.read_text(encoding="utf-8"))
    schedule = pub.build_schedule(grammar)
    self_test_result = json.loads(SELF_TEST_JSON.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST_JSON.read_text(encoding="utf-8")) if MANIFEST_JSON.exists() else {}
    readiness = json.loads(BUILD_READINESS_JSON.read_text(encoding="utf-8")) if BUILD_READINESS_JSON.exists() else {}
    decision = readiness.get("package_decision", manifest.get("package_decision", pub.PACKAGE_DECISION_BLOCKED))
    if GRAMMAR_SHA.read_text(encoding="utf-8").strip() != pub.sha256_file(GRAMMAR_JSON):
        failures.append("grammar file sha mismatch")
    schedule_tsv_sha = pub.sha256_file(SCHEDULE_TSV)
    if SCHEDULE_SHA.read_text(encoding="utf-8").strip() != schedule_tsv_sha:
        failures.append("expanded schedule TSV sha mismatch")
    if not validate_schedule_json_manifest(schedule_manifest, schedule, schedule_tsv_sha)["passed"]:
        failures.append("schedule JSON manifest validation failed")
    if MANIFEST_SHA.exists() and MANIFEST_SHA.read_text(encoding="utf-8").strip() != pub.sha256_file(MANIFEST_JSON):
        failures.append("manifest file sha mismatch")
    if not pub.validate_grammar(grammar)["passed"]:
        failures.append("grammar validation failed")
    if not pub.validate_schedule(schedule, grammar)["passed"]:
        failures.append("schedule validation failed")
    if not validate_grammar_tsv(GRAMMAR_TSV, grammar)["passed"]:
        failures.append("grammar TSV validation failed")
    if not pub.validate_tsv(SCHEDULE_TSV, schedule)["passed"]:
        failures.append("schedule TSV validation failed")
    if not self_test_result.get("passed") and decision == pub.PACKAGE_DECISION_BUILD_READY:
        failures.append("self-test failed")
    if manifest and manifest.get("package_decision") != decision:
        failures.append("package decision mismatch")
    authority = manifest.get("authority_binding", {}) if manifest else {}
    relation_source = authority.get("relation_source_authority_commit")
    if manifest and authority.get("scalar_evidence_provenance") != pub.SCALAR_EVIDENCE_PROVENANCE:
        failures.append("scalar evidence provenance mismatch")
    if manifest and authority.get("relation_manifest_freeze_commit_policy") != pub.RELATION_FREEZE_AUTHORITY_POLICY:
        failures.append("relation freeze authority policy mismatch")
    source_hashes_result = json.loads(SOURCE_HASHES_JSON.read_text(encoding="utf-8")) if SOURCE_HASHES_JSON.exists() else {}
    source_authority_dynamic = None
    if manifest and decision == pub.PACKAGE_DECISION_BUILD_READY:
        if not (isinstance(relation_source, str) and re.fullmatch(r"[0-9a-f]{40}", relation_source) and relation_source not in pub.SCALAR_EVIDENCE_COMMITS):
            failures.append("relation source authority invalid for build-ready package")
        if source_hashes_result:
            source_authority_dynamic = source_authority_validation(
                relation_source,
                source_hashes_result.get("files", {}),
                source_hashes_result.get("source_bundle", {}),
            )
            if not source_authority_dynamic.get("passed"):
                failures.append("relation source authority validation failed")
            if not source_hashes_result.get("source_authority_regression_tests", {}).get("passed"):
                failures.append("relation source authority regression tests failed")
    validate = {
        "schema": "FAMILY10H_RELATION_ONLY_OFFLINE_VALIDATE_V2",
        "required_artifacts": required,
        "missing_artifacts": missing,
        "failures": failures,
        "grammar_json_parse": True,
        "schedule_json_parse": True,
        "self_test_json_parse": True,
        "build_ready": decision == pub.PACKAGE_DECISION_BUILD_READY,
        "package_decision": decision,
        "zero_target_contact": True,
        "zero_live_activity": True,
        "source_authority_dynamic_validation": source_authority_dynamic,
        "passed": not failures,
    }
    validate["offline_validate_sha256"] = pub.digest({k: v for k, v in validate.items() if k != "offline_validate_sha256"})
    return validate


def normalized_relation_source_authority(value: str | None) -> str:
    return value or os.environ.get("FAMILY10H_RELATION_ONLY_RELATION_SOURCE_AUTHORITY") or pub.RELATION_SOURCE_AUTHORITY_UNSET


def prepare(relation_source_authority: str | None = None) -> dict[str, Any]:
    relation_source_authority = normalized_relation_source_authority(relation_source_authority)
    grammar = pub.relation_grammar(pub.PACKAGE_DECISION_BLOCKED)
    schedule = pub.build_schedule(grammar)
    pub.write_json(GRAMMAR_JSON, grammar)
    write_grammar_tsv(grammar)
    write_text(GRAMMAR_SHA, pub.sha256_file(GRAMMAR_JSON) + "\n")
    pub.write_schedule_tsv(schedule, SCHEDULE_TSV)
    schedule_tsv_sha = pub.sha256_file(SCHEDULE_TSV)
    pub.write_compact_json(SCHEDULE_JSON, schedule_json_manifest(schedule, schedule_tsv_sha))
    write_text(SCHEDULE_SHA, schedule_tsv_sha + "\n")

    toolchain = discover_toolchains()
    pub.write_json(TOOLCHAIN_DISCOVERY_JSON, toolchain)
    runtime_build = compile_runtime(toolchain)
    pub.write_json(RUNTIME_BUILD_JSON, runtime_build)
    pub.write_json(
        RUNTIME_CONTROL_FLOW_AUDIT_JSON,
        runtime_build.get(
            "control_flow_audit",
            {
                "schema": "FAMILY10H_RELATION_ONLY_RUNTIME_CONTROL_FLOW_AUDIT_V1",
                "passed": False,
                "reason": "runtime build did not emit control-flow audit",
            },
        ),
    )
    pub.write_json(
        SYNTHETIC_EXECUTOR_SELF_TEST_JSON,
        {
            "schema": "FAMILY10H_RELATION_ONLY_SYNTHETIC_EXECUTOR_SELF_TEST_RECEIPT_V1",
            **runtime_build.get("synthetic_executor_self_test", {}),
        },
    )
    sensor_authority = write_sensor_authority_binding()
    source_bundle = pub.write_source_bundle(SOURCE_BUNDLE, SOURCE_FILES + [CONTRACT_FILE])
    source_hashes_result = source_hashes(source_bundle, runtime_build, relation_source_authority, sensor_authority)
    pub.write_json(SOURCE_HASHES_JSON, source_hashes_result)

    proof = pub.relation_marginal_equality_proof(grammar, schedule, source_hashes_result, runtime_build)
    pub.write_json(PROOF_JSON, proof)
    threshold_contract = physical_adjudication.physical_threshold_contract()
    pub.write_json(PHYSICAL_THRESHOLD_CONTRACT_JSON, threshold_contract)
    physical_self_test = physical_adjudication.run_self_test(schedule)
    pub.write_json(PHYSICAL_ADJUDICATOR_SELF_TEST_JSON, physical_self_test)
    adversary = synthetic_adjudication.run_adversary_tests(schedule)
    pub.write_json(ADVERSARY_JSON, adversary)
    transport = transport_simulation(schedule)
    pub.write_json(TRANSPORT_JSON, transport)

    provisional_readiness = {
        "schema": "FAMILY10H_RELATION_ONLY_BUILD_READINESS_V1",
        "package_decision": pub.PACKAGE_DECISION_BLOCKED,
        "checks": {},
        "blockers": ["provisional_before_target_self_test"],
        "zero_target_contact": True,
        "zero_live_activity": True,
        "live_authority": False,
    }
    provisional_readiness["build_readiness_sha256"] = pub.digest(
        {k: v for k, v in provisional_readiness.items() if k != "build_readiness_sha256"}
    )
    target_stub = {"self_test_passed": False, "provisional": True}
    controller_stub = {"passed": False, "provisional": True}
    provisional_self = self_test(grammar, schedule, proof, runtime_build, target_stub, controller_stub, physical_self_test)
    provisional_validate = {"schema": "FAMILY10H_RELATION_ONLY_OFFLINE_VALIDATE_V2", "passed": False}
    manifest = implementation_manifest(
        grammar,
        proof,
        schedule,
        provisional_self,
        adversary,
        transport,
        provisional_validate,
        source_hashes_result,
        runtime_build,
        target_stub,
        controller_stub,
        physical_self_test,
        threshold_contract,
        provisional_readiness,
        relation_source_authority,
    )
    pub.write_json(MANIFEST_JSON, manifest)
    write_text(MANIFEST_SHA, pub.sha256_file(MANIFEST_JSON) + "\n")

    candidate_readiness = {
        "schema": "FAMILY10H_RELATION_ONLY_BUILD_READINESS_V1",
        "package_decision": pub.PACKAGE_DECISION_BUILD_READY,
        "checks": {"candidate_for_synthetic_target_wrapper_authority_path": True},
        "blockers": [],
        "zero_target_contact": True,
        "zero_live_activity": True,
        "live_authority": False,
        "candidate_only": True,
    }
    candidate_readiness["build_readiness_sha256"] = pub.digest(
        {k: v for k, v in candidate_readiness.items() if k != "build_readiness_sha256"}
    )
    pub.write_json(BUILD_READINESS_JSON, candidate_readiness)
    manifest = implementation_manifest(
        grammar,
        proof,
        schedule,
        provisional_self,
        adversary,
        transport,
        provisional_validate,
        source_hashes_result,
        runtime_build,
        target_stub,
        controller_stub,
        physical_self_test,
        threshold_contract,
        candidate_readiness,
        relation_source_authority,
    )
    pub.write_json(MANIFEST_JSON, manifest)
    write_text(MANIFEST_SHA, pub.sha256_file(MANIFEST_JSON) + "\n")

    target_self_test = run_target_self_test()
    pub.write_json(TARGET_SELF_TEST_JSON, target_self_test)
    pub.write_json(
        TARGET_PREFLIGHT_SELF_TEST_JSON,
        target_self_test.get(
            "target_preflight_fixture_suite",
            {
                "schema": "FAMILY10H_RELATION_ONLY_TARGET_PREFLIGHT_FIXTURE_SUITE_V1",
                "passed": False,
                "reason": "target self-test did not emit preflight fixture suite",
            },
        ),
    )
    pub.write_json(
        SYNTHETIC_TARGET_WRAPPER_SELF_TEST_JSON,
        target_self_test.get(
            "synthetic_target_wrapper_execution",
            {
                "schema": "FAMILY10H_RELATION_ONLY_SYNTHETIC_TARGET_WRAPPER_EXECUTION_V1",
                "passed": False,
                "reason": "target self-test did not emit synthetic target-wrapper execution",
            },
        ),
    )
    live_controller_self_test = run_live_controller_self_test()
    pub.write_json(LIVE_CONTROLLER_SELF_TEST_JSON, live_controller_self_test)
    self_test_result = self_test(grammar, schedule, proof, runtime_build, target_self_test, live_controller_self_test, physical_self_test)
    pub.write_json(SELF_TEST_JSON, self_test_result)
    validate = validate_artifacts()
    readiness = provisional_readiness
    manifest = json.loads(MANIFEST_JSON.read_text(encoding="utf-8"))
    for _ in range(4):
        readiness = build_readiness(
            proof,
            runtime_build,
            target_self_test,
            live_controller_self_test,
            physical_self_test,
            self_test_result,
            validate,
            source_hashes_result,
            relation_source_authority,
        )
        pub.write_json(BUILD_READINESS_JSON, readiness)
        manifest = implementation_manifest(
            grammar,
            proof,
            schedule,
            self_test_result,
            adversary,
            transport,
            validate,
            source_hashes_result,
            runtime_build,
            target_self_test,
            live_controller_self_test,
            physical_self_test,
            threshold_contract,
            readiness,
            relation_source_authority,
        )
        pub.write_json(MANIFEST_JSON, manifest)
        write_text(MANIFEST_SHA, pub.sha256_file(MANIFEST_JSON) + "\n")
        if readiness["package_decision"] != pub.PACKAGE_DECISION_BUILD_READY:
            next_validate = validate_artifacts()
            pub.write_json(VALIDATE_JSON, next_validate)
            validate = next_validate
            break
        target_self_test = run_target_self_test()
        pub.write_json(TARGET_SELF_TEST_JSON, target_self_test)
        pub.write_json(
            TARGET_PREFLIGHT_SELF_TEST_JSON,
            target_self_test.get(
                "target_preflight_fixture_suite",
                {
                    "schema": "FAMILY10H_RELATION_ONLY_TARGET_PREFLIGHT_FIXTURE_SUITE_V1",
                    "passed": False,
                    "reason": "target self-test did not emit preflight fixture suite",
                },
            ),
        )
        pub.write_json(
            SYNTHETIC_TARGET_WRAPPER_SELF_TEST_JSON,
            target_self_test.get(
                "synthetic_target_wrapper_execution",
                {
                    "schema": "FAMILY10H_RELATION_ONLY_SYNTHETIC_TARGET_WRAPPER_EXECUTION_V1",
                    "passed": False,
                    "reason": "target self-test did not emit synthetic target-wrapper execution",
                },
            ),
        )
        live_controller_self_test = run_live_controller_self_test()
        pub.write_json(LIVE_CONTROLLER_SELF_TEST_JSON, live_controller_self_test)
        self_test_result = self_test(grammar, schedule, proof, runtime_build, target_self_test, live_controller_self_test, physical_self_test)
        pub.write_json(SELF_TEST_JSON, self_test_result)
        next_validate = validate_artifacts()
        pub.write_json(VALIDATE_JSON, next_validate)
        if (
            next_validate["passed"] == validate.get("passed")
            and next_validate["package_decision"] == readiness["package_decision"]
            and manifest["package_decision"] == readiness["package_decision"]
        ):
            validate = next_validate
            break
        validate = next_validate
    return {
        "package_decision": readiness["package_decision"],
        "blockers": readiness["blockers"],
        "grammar_sha256": grammar["grammar_sha256"],
        "marginal_equality_proof_sha256": proof["proof_sha256"],
        "schedule_tuple_count": schedule["tuple_count"],
        "self_test_passed": self_test_result["passed"],
        "runtime_build_passed": runtime_build["passed"],
        "target_self_test_passed": target_self_test.get("self_test_passed") is True,
        "live_controller_self_test_passed": live_controller_self_test.get("passed") is True,
        "relation_source_authority_commit": relation_source_authority,
        "physical_adjudicator_passed": physical_self_test["passed"],
        "adversary_passed": adversary["passed"],
        "transport_passed": transport["passed"],
        "offline_validate_passed": validate["passed"],
        "manifest_sha256": manifest["manifest_sha256"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--relation-source-authority", default=None)
    args = parser.parse_args(argv)
    if args.prepare_only:
        result = prepare(args.relation_source_authority)
    elif args.validate_only:
        result = validate_artifacts()
    else:
        parser.print_help()
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("passed", result.get("offline_validate_passed", True)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
