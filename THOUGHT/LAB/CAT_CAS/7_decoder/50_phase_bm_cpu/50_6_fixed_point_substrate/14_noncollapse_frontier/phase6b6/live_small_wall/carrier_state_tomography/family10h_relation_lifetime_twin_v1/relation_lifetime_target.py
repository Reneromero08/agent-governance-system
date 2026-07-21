#!/usr/bin/env python3
"""Target-side authority gate and offline preflight for relation-lifetime v1.

The physical path remains authority-gated. The self-test path uses a fixture
backend and the runtime synthetic executor to test the full wrapper chain
without target contact, PMU acquisition, sensor discovery, or a scientific
claim.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import relation_lifetime_public as pub


AUTHORITY_ENV = "FAMILY10H_RELATION_LIFETIME_LIVE_AUTHORITY"
SOURCE_AUTHORITY_ENV = "FAMILY10H_RELATION_LIFETIME_SOURCE_AUTHORITY_COMMIT"
FREEZE_COMMIT_ENV = "FAMILY10H_RELATION_LIFETIME_FREEZE_COMMIT"
MANIFEST_ENV = "FAMILY10H_RELATION_LIFETIME_MANIFEST_SHA256"
LEGACY_COMMIT_ENV = "FAMILY10H_RELATION_LIFETIME_COMMIT_BINDING"
RUNTIME_AUTHORITY_ENV = "FAMILY10H_RELATION_LIFETIME_RUNTIME_AUTHORITY"
PREFLIGHT_FIXTURE_ENV = "FAMILY10H_RELATION_LIFETIME_PREFLIGHT_FIXTURE"
PREFLIGHT_SYSTEM_MOCK_ENV = "FAMILY10H_RELATION_LIFETIME_PREFLIGHT_SYSTEM_MOCK"
DEPLOYMENT_CUSTODY_ENV = "FAMILY10H_RELATION_LIFETIME_DEPLOYMENT_CUSTODY"
EXECUTION_MODE_ENV = "FAMILY10H_RELATION_LIFETIME_EXECUTION_MODE"
EXECUTION_MODE_SYNTHETIC = "synthetic"
AUTHORITY_VALUE = pub.TRANSACTION_RUN_ID
RUNTIME_BINARY_NAMES = ["relation_lifetime_runtime", "relation_lifetime_runtime.exe"]
PMU_PREFLIGHT_HELPER_NAMES = ["relation_lifetime_pmu_preflight", "relation_lifetime_pmu_preflight.exe"]
INJECTED_PREFLIGHT_ENVS = [PREFLIGHT_FIXTURE_ENV, PREFLIGHT_SYSTEM_MOCK_ENV]
HEX40_RE = re.compile(r"[0-9a-f]{40}")
EXPECTED_PERF_EVENT_ATTR_SIZE = 112

TARGET_PREFLIGHT_CHECKS = [
    "live_authority",
    "source_authority_commit_binding",
    "freeze_commit_binding",
    "relation_source_authority_not_scalar",
    "relation_freeze_authority_not_scalar",
    "deployment_custody_binding",
    "manifest_file_hash",
    "canonical_manifest_digest",
    "package_decision",
    "transaction_identity",
    "source_root_identity",
    "grammar_hash",
    "compact_schedule_manifest_binding",
    "expanded_schedule_tsv_hash",
    "runtime_binary_hash",
    "runtime_binary_format_and_abi",
    "source_bundle_hash",
    "source_file_hashes",
    "cpu_vendor_family_model",
    "source_receiver_cpu_identity",
    "approved_target_identity",
    "operational_pinning_capability",
    "injected_preflight_backend_absent",
    "pmu_event_identities",
    "pmu_helper_binary_hash",
    "pmu_helper_binary_format_and_abi",
    "grouped_pmu_open_capability",
    "temperature_sensor_authority",
    "cpu_frequency_policy_custody",
    "actual_physical_geometry_status",
    "output_root_absence",
    "attempt_ownership_marker",
]


class TargetError(RuntimeError):
    pass


def strict_json_dumps(value: Any, *, indent: int | None = None) -> str:
    return json.dumps(value, indent=indent, sort_keys=True, allow_nan=False)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def digest_without(value: dict[str, Any], key: str) -> str:
    return pub.digest({k: v for k, v in value.items() if k != key})


def runtime_binary_path(source_root: Path) -> Path | None:
    for name in RUNTIME_BINARY_NAMES:
        path = source_root / name
        if path.exists():
            return path
    return None


def wsl_path(path: Path) -> str | None:
    if os.name != "nt" or shutil.which("wsl.exe") is None:
        return None
    path = path.resolve()
    completed = subprocess.run(
        ["wsl.exe", "wslpath", "-a", str(path)],
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )
    if completed.returncode != 0:
        drive = path.drive.rstrip(":").lower()
        if drive and path.is_absolute():
            suffix = "/".join(path.parts[1:])
            return f"/mnt/{drive}/{suffix}"
        return None
    return completed.stdout.strip()


def wsl_distro(source_root: Path) -> str | None:
    build_receipt = source_root / "RELATION_LIFETIME_RUNTIME_BUILD_SELF_TEST.json"
    if not build_receipt.exists():
        return None
    try:
        receipt = read_json(build_receipt)
    except (OSError, json.JSONDecodeError):
        return None
    distro = (
        receipt.get("compiler", {}).get("distro")
        or receipt.get("runtime_binary_authority", {}).get("compiler_identity", {}).get("distro")
    )
    return distro if isinstance(distro, str) and distro else None


def wsl_prefix(source_root: Path) -> list[str]:
    distro = wsl_distro(source_root)
    if distro:
        return ["wsl.exe", "-d", distro, "--"]
    return ["wsl.exe", "--"]


def runtime_command(source_root: Path, runtime: Path, args: list[Path | str]) -> tuple[list[str], str]:
    if os.name == "nt" and runtime.suffix != ".exe":
        translated_runtime = wsl_path(runtime)
        translated_args: list[str] = []
        for arg in args:
            if isinstance(arg, Path):
                translated = wsl_path(arg)
                if translated is None:
                    raise TargetError(f"cannot translate path for WSL: {arg}")
                translated_args.append(translated)
            else:
                translated_args.append(arg)
        if translated_runtime is None:
            raise TargetError("cannot translate runtime path for WSL")
        return [*wsl_prefix(source_root), translated_runtime, *translated_args], "wsl_linux_binary"
    return [str(runtime), *[str(arg) for arg in args]], "native"


def run_file_probe(source_root: Path, runtime: Path) -> dict[str, Any]:
    if os.name == "nt" and runtime.suffix != ".exe":
        translated_runtime = wsl_path(runtime)
        if translated_runtime is None:
            return {"returncode": 2, "stdout": "", "stderr": "cannot translate runtime path"}
        command = [*wsl_prefix(source_root), "file", translated_runtime]
    else:
        command = ["file", str(runtime)]
    completed = subprocess.run(command, text=True, capture_output=True, check=False, timeout=15)
    if completed.returncode != 0 or not completed.stdout.strip() or "ELF" not in completed.stdout:
        data = runtime.read_bytes()[:20]
        if len(data) >= 20 and data[:4] == b"\x7fELF" and data[4] == 2:
            machine = "x86-64" if data[18] == 0x3E else f"machine-0x{data[18]:02x}"
            return {
                "command": command,
                "returncode": completed.returncode,
                "stdout": f"{runtime}: ELF 64-bit LSB shared object, {machine} (validated from ELF header)",
                "stderr": completed.stderr,
                "fallback": "elf_header",
            }
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def owned_output_parent(source_root: Path) -> Path:
    return source_root / pub.OWNED_OUTPUT_PARENT_NAME


def safe_remove_owned_output_parent(source_root: Path) -> None:
    parent = owned_output_parent(source_root).resolve()
    root = source_root.resolve()
    if parent.name != pub.OWNED_OUTPUT_PARENT_NAME or parent.parent != root:
        raise TargetError("refusing to remove non-owned output parent")
    if parent.exists():
        shutil.rmtree(parent)


def scalar_evidence_provenance() -> dict[str, Any]:
    return dict(pub.SCALAR_EVIDENCE_PROVENANCE)


def relation_source_authority(manifest: dict[str, Any]) -> str | None:
    value = manifest.get("authority_binding", {}).get("relation_source_authority_commit")
    return value if isinstance(value, str) else None


def relation_source_authority_is_valid(value: str | None) -> bool:
    return bool(value and HEX40_RE.fullmatch(value) and value not in pub.SCALAR_EVIDENCE_COMMITS)


def deployment_custody_path(source_root: Path) -> Path:
    configured = os.environ.get(DEPLOYMENT_CUSTODY_ENV)
    if configured:
        return Path(configured)
    return source_root / pub.DEPLOYMENT_CUSTODY_FILENAME


def read_deployment_custody(source_root: Path) -> dict[str, Any]:
    path = deployment_custody_path(source_root)
    if not path.exists():
        return {}
    return read_json(path)


def validate_source_root(source_root: Path) -> dict[str, Any]:
    source_root = source_root.resolve()
    failures: list[str] = []
    required = [
        "RELATION_LIFETIME_CONTRACT.md",
        "RELATION_GRAMMAR.json",
        "RELATION_GRAMMAR.tsv",
        "RELATION_GRAMMAR.sha256",
        "RELATION_LIFETIME_PUBLIC_SCHEDULE.json",
        "RELATION_LIFETIME_PUBLIC_SCHEDULE.tsv",
        "RELATION_LIFETIME_PUBLIC_SCHEDULE.sha256",
        "RELATION_LIFETIME_IMPLEMENTATION_MANIFEST.json",
        "RELATION_LIFETIME_IMPLEMENTATION_MANIFEST.sha256",
        "RELATION_LIFETIME_SOURCE_HASHES.json",
        "RELATION_LIFETIME_SENSOR_AUTHORITY_BINDING.json",
        "RELATION_LIFETIME_TOOLCHAIN_DISCOVERY.json",
        "RELATION_LIFETIME_RUNTIME_BUILD_SELF_TEST.json",
        "RELATION_LIFETIME_SYNTHETIC_EXECUTOR_SELF_TEST.json",
        "RELATION_LIFETIME_PHYSICAL_ADJUDICATOR_SELF_TEST.json",
        "RELATION_LIFETIME_PHYSICAL_THRESHOLD_CONTRACT.json",
        "relation_lifetime_public.py",
        "relation_lifetime_adjudication.py",
        "relation_lifetime_physical_adjudication.py",
        "relation_lifetime_runtime.c",
        "relation_lifetime_runtime.h",
        "relation_lifetime_pmu_preflight.c",
        "relation_lifetime_pmu_preflight",
        "relation_lifetime_target.py",
        "relation_lifetime_live_controller.py",
        "run_relation_lifetime_twin.py",
    ]
    missing = [name for name in required if not (source_root / name).exists()]
    failures.extend(f"missing {name}" for name in missing)
    if missing:
        return {"passed": False, "failures": failures, "required_file_count": len(required)}

    grammar = read_json(source_root / "RELATION_GRAMMAR.json")
    schedule_manifest = read_json(source_root / "RELATION_LIFETIME_PUBLIC_SCHEDULE.json")
    schedule = pub.build_schedule(grammar)
    manifest = read_json(source_root / "RELATION_LIFETIME_IMPLEMENTATION_MANIFEST.json")
    source_hashes = read_json(source_root / "RELATION_LIFETIME_SOURCE_HASHES.json")
    sensor_binding = read_json(source_root / pub.SENSOR_AUTHORITY_BINDING_FILENAME)
    threshold = read_json(source_root / "RELATION_LIFETIME_PHYSICAL_THRESHOLD_CONTRACT.json")
    readiness = read_json(source_root / "RELATION_LIFETIME_BUILD_READINESS.json") if (source_root / "RELATION_LIFETIME_BUILD_READINESS.json").exists() else {}

    grammar_file_sha = pub.sha256_file(source_root / "RELATION_GRAMMAR.json")
    if grammar_file_sha != (source_root / "RELATION_GRAMMAR.sha256").read_text(encoding="utf-8").strip():
        failures.append("grammar sha sidecar mismatch")
    if grammar.get("grammar_sha256") != digest_without(grammar, "grammar_sha256"):
        failures.append("canonical grammar digest mismatch")

    schedule_tsv_sha = pub.sha256_file(source_root / "RELATION_LIFETIME_PUBLIC_SCHEDULE.tsv")
    if schedule_tsv_sha != (source_root / "RELATION_LIFETIME_PUBLIC_SCHEDULE.sha256").read_text(encoding="utf-8").strip():
        failures.append("expanded schedule TSV sha sidecar mismatch")
    if schedule_manifest.get("schema") != "FAMILY10H_RELATION_LIFETIME_PUBLIC_SCHEDULE_MANIFEST_V3":
        failures.append("schedule JSON manifest schema mismatch")
    if schedule_manifest.get("schedule_manifest_sha256") != digest_without(schedule_manifest, "schedule_manifest_sha256"):
        failures.append("schedule JSON manifest digest mismatch")
    if schedule_manifest.get("canonical_schedule_sha256") != schedule.get("schedule_sha256"):
        failures.append("canonical schedule digest mismatch")
    if schedule_manifest.get("expanded_schedule_file_sha256") != schedule_tsv_sha:
        failures.append("schedule manifest TSV binding mismatch")
    if schedule_manifest.get("json_rows_omitted") is not True:
        failures.append("schedule JSON rows not compacted")
    if not pub.validate_grammar(grammar)["passed"]:
        failures.append("grammar validation failed")
    if not pub.validate_schedule(schedule, grammar)["passed"]:
        failures.append("schedule validation failed")

    manifest_file_sha = pub.sha256_file(source_root / "RELATION_LIFETIME_IMPLEMENTATION_MANIFEST.json")
    if manifest_file_sha != (source_root / "RELATION_LIFETIME_IMPLEMENTATION_MANIFEST.sha256").read_text(encoding="utf-8").strip():
        failures.append("manifest sha sidecar mismatch")
    if manifest.get("manifest_sha256") != digest_without(manifest, "manifest_sha256"):
        failures.append("canonical manifest digest mismatch")
    if readiness and readiness.get("package_decision") != manifest.get("package_decision"):
        failures.append("readiness manifest decision mismatch")
    if manifest.get("transaction_run_id") != pub.TRANSACTION_RUN_ID:
        failures.append("transaction identity mismatch")
    if manifest.get("grammar_sha256") != grammar.get("grammar_sha256"):
        failures.append("manifest grammar binding mismatch")
    if manifest.get("schedule_sha256") != schedule.get("schedule_sha256"):
        failures.append("manifest schedule binding mismatch")

    authority = manifest.get("authority_binding", {})
    relation_source = authority.get("relation_source_authority_commit")
    if authority.get("scalar_evidence_provenance") != scalar_evidence_provenance():
        failures.append("manifest scalar evidence provenance mismatch")
    if authority.get("relation_manifest_freeze_commit_policy") != pub.RELATION_FREEZE_AUTHORITY_POLICY:
        failures.append("manifest relation freeze policy mismatch")
    source_authority_validation = source_hashes.get("relation_source_authority_validation", {})
    source_authority_regressions = source_hashes.get("source_authority_regression_tests", {})
    if relation_source in pub.SCALAR_EVIDENCE_COMMITS:
        failures.append("relation source authority incorrectly uses scalar evidence commit")
    if manifest.get("package_decision") == pub.PACKAGE_DECISION_BUILD_READY and not relation_source_authority_is_valid(relation_source):
        failures.append("relation source authority missing or invalid for build-ready package")
    if manifest.get("package_decision") == pub.PACKAGE_DECISION_BUILD_READY and source_authority_validation.get("passed") is not True:
        failures.append("relation source authority git/tree validation failed")
    if manifest.get("package_decision") == pub.PACKAGE_DECISION_BUILD_READY and source_authority_regressions.get("passed") is not True:
        failures.append("relation source authority negative regressions failed")
    if sensor_binding.get("approved_sensor_identity_sha256") != pub.APPROVED_SENSOR_IDENTITY_SHA256:
        failures.append("approved sensor identity binding mismatch")
    if sensor_binding.get("approved_sensor_identity") != pub.APPROVED_SENSOR_IDENTITY:
        failures.append("approved sensor identity field mismatch")
    if sensor_binding.get("unlabeled_legacy_temp1_input_approved") is not True:
        failures.append("approved sensor unlabeled temp1 law missing")
    if sensor_binding.get("approved_target_identity_sha256") != pub.APPROVED_TARGET_IDENTITY_SHA256:
        failures.append("approved target identity binding mismatch")
    threshold_source = threshold.get("scalar_evidence_provenance", {})
    if threshold_source.get("source_authority_commit") != pub.SCALAR_EVIDENCE_SOURCE_AUTHORITY_COMMIT:
        failures.append("threshold scalar source provenance mismatch")
    if threshold_source.get("manifest_freeze_commit") != pub.SCALAR_EVIDENCE_MANIFEST_FREEZE_COMMIT:
        failures.append("threshold scalar manifest freeze provenance mismatch")

    for name, expected in source_hashes.get("files", {}).items():
        path = source_root / name
        if not path.exists() or pub.sha256_file(path) != expected.get("sha256"):
            failures.append(f"source file hash mismatch: {name}")
    bundle = source_hashes.get("source_bundle", {})
    if bundle and pub.sha256_file(source_root / bundle.get("path", "")) != bundle.get("sha256"):
        failures.append("source bundle hash mismatch")
    runtime = runtime_binary_path(source_root)
    runtime_authority = source_hashes.get("runtime_binary_authority", {})
    if runtime is None:
        failures.append("runtime binary missing")
    elif runtime_authority.get("compiled_binary_sha256") != pub.sha256_file(runtime):
        failures.append("runtime binary hash mismatch")
    helper_identity = pmu_preflight_helper_identity(source_root, source_hashes)
    if not helper_identity["present"]:
        failures.append("PMU preflight helper binary missing")
    if not helper_identity["helper_sha256_matches_authority"]:
        failures.append("PMU preflight helper binary hash mismatch")
    if not helper_identity["helper_c_sha256_matches_authority"]:
        failures.append("PMU preflight helper C source hash mismatch")
    if not helper_identity["helper_binary_format_valid"]:
        failures.append("PMU preflight helper binary format or ABI mismatch")
    if manifest.get("claim_boundary", {}).get("small_wall_crossed") is not False:
        failures.append("manifest small wall boundary mismatch")

    return {
        "passed": not failures,
        "failures": failures,
        "required_file_count": len(required),
        "pmu_preflight_helper_identity": helper_identity,
    }


def artifact_identity(source_root: Path) -> dict[str, Any]:
    source_root = source_root.resolve()
    grammar = read_json(source_root / "RELATION_GRAMMAR.json")
    schedule_manifest = read_json(source_root / "RELATION_LIFETIME_PUBLIC_SCHEDULE.json")
    manifest = read_json(source_root / "RELATION_LIFETIME_IMPLEMENTATION_MANIFEST.json")
    source_hashes = read_json(source_root / "RELATION_LIFETIME_SOURCE_HASHES.json")
    runtime = runtime_binary_path(source_root)
    runtime_sha = pub.sha256_file(runtime) if runtime else None
    file_probe = run_file_probe(source_root, runtime) if runtime else {"returncode": 2, "stdout": "", "stderr": "runtime missing"}
    runtime_format = file_probe.get("stdout", "").strip()
    helper_identity = pmu_preflight_helper_identity(source_root, source_hashes)
    if (
        not runtime_format
        and runtime_sha
        and runtime_sha == source_hashes.get("runtime_binary_authority", {}).get("compiled_binary_sha256")
    ):
        runtime_format = "ELF 64-bit LSB shared object, x86-64 (sha-bound runtime authority)"
    return {
        "grammar_file_sha256": pub.sha256_file(source_root / "RELATION_GRAMMAR.json"),
        "grammar_canonical_sha256": grammar.get("grammar_sha256"),
        "schedule_tsv_sha256": pub.sha256_file(source_root / "RELATION_LIFETIME_PUBLIC_SCHEDULE.tsv"),
        "schedule_canonical_sha256": schedule_manifest.get("canonical_schedule_sha256"),
        "schedule_manifest_sha256": schedule_manifest.get("schedule_manifest_sha256"),
        "manifest_file_sha256": pub.sha256_file(source_root / "RELATION_LIFETIME_IMPLEMENTATION_MANIFEST.json"),
        "manifest_canonical_sha256": manifest.get("manifest_sha256"),
        "runtime_binary_sha256": runtime_sha,
        "runtime_binary_format": runtime_format,
        "pmu_preflight_helper_sha256": helper_identity["actual_helper_sha256"],
        "pmu_preflight_helper_actual_sha256": helper_identity["actual_helper_sha256"],
        "pmu_preflight_helper_expected_sha256": helper_identity["expected_helper_sha256"],
        "pmu_preflight_helper_sha256_matches_authority": helper_identity["helper_sha256_matches_authority"],
        "pmu_preflight_helper_actual_format": helper_identity["actual_helper_binary_format"],
        "pmu_preflight_helper_format_valid": helper_identity["helper_binary_format_valid"],
        "pmu_preflight_helper_actual_c_sha256": helper_identity["actual_helper_c_sha256"],
        "pmu_preflight_helper_expected_c_sha256": helper_identity["expected_helper_c_sha256"],
        "pmu_preflight_helper_c_sha256_matches_authority": helper_identity["helper_c_sha256_matches_authority"],
        "approved_sensor_identity_sha256": pub.APPROVED_SENSOR_IDENTITY_SHA256,
        "approved_target_identity_sha256": pub.APPROVED_TARGET_IDENTITY_SHA256,
        "source_bundle_sha256": source_hashes.get("source_bundle", {}).get("sha256"),
        "source_file_hashes": {name: item.get("sha256") for name, item in source_hashes.get("files", {}).items()},
    }


def no_live_authority_env() -> dict[str, Any]:
    observed = {
        key: os.environ.get(key)
        for key in [
            AUTHORITY_ENV,
            SOURCE_AUTHORITY_ENV,
            FREEZE_COMMIT_ENV,
            MANIFEST_ENV,
            LEGACY_COMMIT_ENV,
            RUNTIME_AUTHORITY_ENV,
            PREFLIGHT_FIXTURE_ENV,
            PREFLIGHT_SYSTEM_MOCK_ENV,
            DEPLOYMENT_CUSTODY_ENV,
            EXECUTION_MODE_ENV,
        ]
        if os.environ.get(key) is not None
    }
    return {
        "passed": not observed,
        "observed_authority_keys": sorted(observed),
        "target_contact_count": 0,
        "sensor_inventory_count": 0,
        "live_invocation_count": 0,
        "pmu_acquisition_count": 0,
    }


def runtime_refusal_probe(source_root: Path) -> dict[str, Any]:
    runtime = runtime_binary_path(source_root)
    if runtime is None:
        return {
            "passed": True,
            "runtime_binary_present": False,
            "reason": "runtime binary absent; compile gate owns build-readiness blocker",
            "live_invocation_count": 0,
            "pmu_acquisition_count": 0,
        }
    command, launch_mode = runtime_command(
        source_root,
        runtime,
        ["--execute-schedule", source_root / "RELATION_LIFETIME_PUBLIC_SCHEDULE.tsv", source_root / "_no_live_output"],
    )
    completed = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
        timeout=15,
        env={key: value for key, value in os.environ.items() if key != RUNTIME_AUTHORITY_ENV},
    )
    return {
        "passed": completed.returncode != 0,
        "runtime_binary_present": True,
        "launch_mode": launch_mode,
        "returncode": completed.returncode,
        "stderr": completed.stderr,
        "stdout": completed.stdout,
        "live_invocation_count": 0,
        "pmu_acquisition_count": 0,
    }


def load_fixture() -> dict[str, Any] | None:
    path = os.environ.get(PREFLIGHT_FIXTURE_ENV)
    if not path:
        return None
    return read_json(Path(path))


def base_preflight_fixture(source_root: Path, output_root: Path) -> dict[str, Any]:
    artifacts = artifact_identity(source_root)
    manifest = read_json(source_root / "RELATION_LIFETIME_IMPLEMENTATION_MANIFEST.json")
    relation_source = relation_source_authority(manifest)
    return {
        "schema": "FAMILY10H_RELATION_LIFETIME_PREFLIGHT_FIXTURE_V1",
        "cpu": {
            "vendor": "AuthenticAMD",
            "family": 16,
            "model": pub.APPROVED_TARGET_IDENTITY["model"],
            "processor_count": pub.APPROVED_TARGET_IDENTITY["processor_count"],
            "processors": pub.APPROVED_TARGET_IDENTITY["processors"],
            "target_identity_sha256": pub.APPROVED_TARGET_IDENTITY_SHA256,
            "runtime_abi": pub.APPROVED_TARGET_IDENTITY["runtime_abi"],
            "source": pub.APPROVED_TARGET_IDENTITY["source"],
            "source_cpu_present": True,
            "receiver_cpu_present": True,
            "source_cpu": pub.APPROVED_TARGET_IDENTITY["source_cpu"],
            "receiver_cpu": pub.APPROVED_TARGET_IDENTITY["receiver_cpu"],
            "operational_pinning": True,
        },
        "expected_artifacts": {
            "runtime_binary_sha256": artifacts["runtime_binary_sha256"],
            "manifest_file_sha256": artifacts["manifest_file_sha256"],
            "manifest_canonical_sha256": artifacts["manifest_canonical_sha256"],
            "schedule_tsv_sha256": artifacts["schedule_tsv_sha256"],
            "schedule_canonical_sha256": artifacts["schedule_canonical_sha256"],
            "grammar_canonical_sha256": artifacts["grammar_canonical_sha256"],
            "source_bundle_sha256": artifacts["source_bundle_sha256"],
            "pmu_preflight_helper_actual_sha256": artifacts["pmu_preflight_helper_actual_sha256"],
            "pmu_preflight_helper_expected_sha256": artifacts["pmu_preflight_helper_expected_sha256"],
            "pmu_preflight_helper_sha256_matches_authority": artifacts["pmu_preflight_helper_sha256_matches_authority"],
            "pmu_preflight_helper_actual_format": artifacts["pmu_preflight_helper_actual_format"],
            "pmu_preflight_helper_format_valid": artifacts["pmu_preflight_helper_format_valid"],
            "relation_source_authority_commit": relation_source,
            "relation_freeze_authority_commit": pub.SYNTHETIC_RELATION_FREEZE_COMMIT,
        },
        "runtime_binary_format_contains": ["ELF 64-bit", "x86-64"],
        "pmu": {
            "schema": "FAMILY10H_RELATION_LIFETIME_DISABLED_PMU_PREFLIGHT_V1",
            "passed": True,
            "returncode": 0,
            "events": pub.PMU_GROUP["events"],
            "grouped_open_capability": True,
            "helper_sha256": artifacts["pmu_preflight_helper_actual_sha256"],
            "expected_helper_sha256": artifacts["pmu_preflight_helper_expected_sha256"],
            "helper_sha256_matches_authority": True,
            "uses_system_perf_event_attr": True,
            "perf_event_attr_size": EXPECTED_PERF_EVENT_ATTR_SIZE,
            "pmu_open_count": len(pub.PMU_GROUP["events"]),
            "pmu_close_count": len(pub.PMU_GROUP["events"]),
            "pmu_fd_leak_count": 0,
            "pmu_acquisition_count": 0,
            "enabled_measurement_interval": False,
            "scientific_data_collected": False,
            "enable_attempted": False,
            "reset_attempted": False,
            "read_attempted": False,
            "malformed_event_identity": False,
            "partial_group_open": False,
            "incorrect_structure_size": False,
            "missing_event": False,
        },
        "sensor": {
            **pub.APPROVED_SENSOR_IDENTITY,
            "identity_stability": True,
        },
        "frequency_policy": {
            "governor": "performance",
            "policy_locked": True,
            "identity_stability": True,
        },
        "physical_geometry": {
            "status": "unresolved_contractually_lowered",
            "claim_ceiling_lowered": True,
        },
        "output": {
            "root": str(output_root),
            "parent_writable": True,
        },
        "attempt": {
            "owner_marker": pub.TRANSACTION_RUN_ID,
            "attempt_count_before": 0,
            "attempt_ceiling": pub.ATTEMPT_CEILING,
        },
    }


def parse_cpuinfo(text: str) -> dict[int, dict[str, Any]]:
    cpus: dict[int, dict[str, Any]] = {}
    current: dict[str, Any] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            if "processor" in current:
                cpus[int(current["processor"])] = current
            current = {}
            continue
        if ":" not in line:
            continue
        key, value = [part.strip() for part in line.split(":", 1)]
        current[key] = value
    if "processor" in current:
        cpus[int(current["processor"])] = current
    return cpus


def cpu_observation_from_cpuinfo(text: str, source_cpu: int, receiver_cpu: int) -> dict[str, Any]:
    cpus = parse_cpuinfo(text)
    source = cpus.get(source_cpu, {})
    sample = source or next(iter(cpus.values()), {})
    family = sample.get("cpu family") or sample.get("family")
    model = sample.get("model")
    processors = []
    for cpu_id, row in sorted(cpus.items()):
        cpu_family = row.get("cpu family") or row.get("family")
        cpu_model = row.get("model")
        processors.append(
            {
                "processor": cpu_id,
                "vendor_id": row.get("vendor_id") or row.get("vendor"),
                "cpu_family": int(cpu_family) if str(cpu_family).isdigit() else None,
                "model": int(cpu_model) if str(cpu_model).isdigit() else None,
            }
        )
    identity = {
        "vendor": sample.get("vendor_id") or sample.get("vendor"),
        "family": int(family) if str(family).isdigit() else None,
        "model": int(model) if str(model).isdigit() else None,
        "processor_count": len(cpus),
        "source_cpu": source_cpu,
        "receiver_cpu": receiver_cpu,
        "processors": processors,
        "runtime_abi": pub.APPROVED_TARGET_IDENTITY["runtime_abi"],
        "source": pub.APPROVED_TARGET_IDENTITY["source"],
    }
    return {
        **identity,
        "target_identity_sha256": pub.digest(identity),
        "source_cpu_present": source_cpu in cpus,
        "receiver_cpu_present": receiver_cpu in cpus,
        "operational_pinning": False,
        "cpu_count_observed": len(cpus),
    }


def probe_affinity(source_cpu: int, receiver_cpu: int) -> dict[str, Any]:
    if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        return {"operational_pinning": False, "reason": "sched affinity APIs unavailable"}
    original = os.sched_getaffinity(0)
    try:
        for cpu in [source_cpu, receiver_cpu]:
            os.sched_setaffinity(0, {cpu})
            if os.sched_getaffinity(0) != {cpu}:
                return {"operational_pinning": False, "reason": f"affinity did not stick for CPU {cpu}"}
        return {"operational_pinning": True}
    except OSError as exc:
        return {"operational_pinning": False, "reason": str(exc)}
    finally:
        try:
            os.sched_setaffinity(0, original)
        except OSError:
            pass


def pmu_preflight_helper_path(source_root: Path) -> Path | None:
    for name in PMU_PREFLIGHT_HELPER_NAMES:
        path = source_root / name
        if path.exists():
            return path
    return None


def helper_format_valid(format_text: str) -> bool:
    return "ELF 64-bit" in format_text and "x86-64" in format_text


def pmu_preflight_helper_identity(source_root: Path, source_hashes: dict[str, Any] | None = None) -> dict[str, Any]:
    if source_hashes is None:
        hashes_path = source_root / "RELATION_LIFETIME_SOURCE_HASHES.json"
        source_hashes = read_json(hashes_path) if hashes_path.exists() else {}
    helper = pmu_preflight_helper_path(source_root)
    authority = source_hashes.get("pmu_preflight_helper_authority", {})
    c_source = source_root / "relation_lifetime_pmu_preflight.c"
    expected_binary_sha = authority.get("compiled_binary_sha256")
    expected_c_sha = authority.get("helper_c_sha256") or source_hashes.get("files", {}).get("relation_lifetime_pmu_preflight.c", {}).get("sha256")
    actual_binary_sha = pub.sha256_file(helper) if helper else None
    actual_c_sha = pub.sha256_file(c_source) if c_source.exists() else None
    file_probe = run_file_probe(source_root, helper) if helper else {"returncode": 2, "stdout": "", "stderr": "helper missing"}
    binary_format = file_probe.get("stdout", "").strip()
    return {
        "present": helper is not None,
        "path": helper.name if helper else None,
        "actual_helper_sha256": actual_binary_sha,
        "expected_helper_sha256": expected_binary_sha,
        "helper_sha256_matches_authority": actual_binary_sha == expected_binary_sha and actual_binary_sha is not None,
        "actual_helper_c_sha256": actual_c_sha,
        "expected_helper_c_sha256": expected_c_sha,
        "helper_c_sha256_matches_authority": actual_c_sha == expected_c_sha and actual_c_sha is not None,
        "actual_helper_binary_format": binary_format,
        "helper_binary_format_probe": file_probe,
        "helper_binary_format_valid": helper_format_valid(binary_format),
        "expected_runtime_abi": pub.APPROVED_TARGET_IDENTITY["runtime_abi"],
    }


def disabled_grouped_pmu_probe(source_root: Path, cpu: int) -> dict[str, Any]:
    helper = pmu_preflight_helper_path(source_root)
    if helper is None:
        return {
            "events": pub.PMU_GROUP["events"],
            "grouped_open_capability": False,
            "pmu_open_count": 0,
            "pmu_close_count": 0,
            "pmu_acquisition_count": 0,
            "enabled_measurement_interval": False,
            "scientific_data_collected": False,
            "reason": "compiled PMU preflight helper missing",
        }
    command, launch_mode = runtime_command(source_root, helper, ["--disabled-group-preflight", str(cpu)])
    completed = subprocess.run(command, text=True, capture_output=True, check=False, timeout=30)
    try:
        observed = json.loads(completed.stdout)
    except json.JSONDecodeError:
        observed = {
            "schema": "FAMILY10H_RELATION_LIFETIME_DISABLED_PMU_PREFLIGHT_V1",
            "passed": False,
            "events": pub.PMU_GROUP["events"],
            "grouped_open_capability": False,
            "pmu_open_count": 0,
            "pmu_close_count": 0,
            "pmu_acquisition_count": 0,
            "enabled_measurement_interval": False,
            "scientific_data_collected": False,
            "error_text": completed.stderr,
        }
    observed["command"] = command
    observed["launch_mode"] = launch_mode
    observed["returncode"] = completed.returncode
    helper_identity = pmu_preflight_helper_identity(source_root)
    observed["helper_identity"] = helper_identity
    observed["helper_sha256"] = helper_identity.get("actual_helper_sha256")
    observed["expected_helper_sha256"] = helper_identity.get("expected_helper_sha256")
    observed["helper_sha256_matches_authority"] = helper_identity.get("helper_sha256_matches_authority")
    return observed


SENSOR_OBSERVATION_ONLY_FIELDS = {"identity_sha256", "identity_stability", "input_st_dev", "input_st_ino", "input_st_mode"}


def sensor_stable_identity_fields(identity: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in identity.items()
        if key not in SENSOR_OBSERVATION_ONLY_FIELDS
    }


def sensor_identity_matches(observed: dict[str, Any]) -> bool:
    return (
        sensor_stable_identity_fields(observed) == sensor_stable_identity_fields(pub.APPROVED_SENSOR_IDENTITY)
        and observed.get("identity_sha256") == pub.APPROVED_SENSOR_IDENTITY_SHA256
        and observed.get("identity_stability") is True
    )


def read_sysfs_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def sensor_identity_from_input(sysfs_root: Path, input_path: Path) -> dict[str, Any]:
    hwmon_dir = input_path.parent
    label_path = input_path.with_name(input_path.name.replace("_input", "_label"))
    label_present = label_path.exists()
    device_path = hwmon_dir / "device"
    driver_path = device_path / "driver"
    subsystem_path = device_path / "subsystem"
    try:
        stat = input_path.stat()
    except OSError:
        stat = None
    identity = {
        "class_path": "/" + input_path.relative_to(sysfs_root.parent).as_posix() if input_path.is_absolute() else str(input_path),
        "device_driver": driver_path.resolve().name if driver_path.exists() else None,
        "device_modalias": read_sysfs_text(device_path / "modalias"),
        "device_subsystem": subsystem_path.resolve().name if subsystem_path.exists() else None,
        "hwmon_name": read_sysfs_text(hwmon_dir / "name"),
        "identity_sha256": None,
        "input_st_dev": stat.st_dev if stat else None,
        "input_st_ino": stat.st_ino if stat else None,
        "input_st_mode": stat.st_mode if stat else None,
        "resolved_device_path": str(device_path.resolve()) if device_path.exists() else None,
        "resolved_driver_path": str(driver_path.resolve()) if driver_path.exists() else None,
        "resolved_hwmon_path": str(hwmon_dir.resolve()),
        "resolved_input_path": str(input_path.resolve()),
        "resolved_subsystem_path": str(subsystem_path.resolve()) if subsystem_path.exists() else None,
        "sensor_input": input_path.name,
        "sensor_label_present": label_present,
        "sensor_label_value": read_sysfs_text(label_path) if label_present else None,
        "sensor_semantic_profile": "LEGACY_FAMILY10H_K10TEMP_TEMP1_V1",
        "sensor_semantic_role": "Tctl",
    }
    stable_identity = sensor_stable_identity_fields(identity)
    identity["identity_sha256"] = (
        pub.APPROVED_SENSOR_IDENTITY_SHA256
        if stable_identity == sensor_stable_identity_fields(pub.APPROVED_SENSOR_IDENTITY)
        else pub.digest(stable_identity)
    )
    return identity


def discover_temperature_sensor(sysfs_root: Path = Path("/sys")) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for input_file in sorted((sysfs_root / "class" / "hwmon").glob("hwmon*/temp*_input")):
        before = sensor_identity_from_input(sysfs_root, input_file)
        after = sensor_identity_from_input(sysfs_root, input_file)
        observed = {**before, "identity_stability": before == after}
        candidates.append(observed)
        if sensor_identity_matches(observed):
            return observed
    return {
        **{key: None for key in pub.APPROVED_SENSOR_IDENTITY},
        "identity_stability": False,
        "candidate_count": len(candidates),
        "observed_candidates": candidates,
    }


def cpu_frequency_policy(source_cpu: int, receiver_cpu: int, sysfs_root: Path = Path("/sys")) -> dict[str, Any]:
    policies = []
    for cpu in [source_cpu, receiver_cpu]:
        governor_path = sysfs_root / "devices" / "system" / "cpu" / f"cpu{cpu}" / "cpufreq" / "scaling_governor"
        try:
            policies.append({"cpu": cpu, "governor": governor_path.read_text(encoding="utf-8").strip()})
        except OSError:
            policies.append({"cpu": cpu, "governor": None})
    governors = {item["governor"] for item in policies}
    return {
        "governor": next(iter(governors)) if len(governors) == 1 else "mixed_or_unavailable",
        "policies": policies,
        "policy_locked": governors == {"performance"},
        "identity_stability": all(item["governor"] for item in policies),
    }


def physical_preflight_observation(source_root: Path, output_root: Path) -> dict[str, Any]:
    del output_root
    synthetic = os.environ.get(EXECUTION_MODE_ENV) == EXECUTION_MODE_SYNTHETIC
    mock_path = os.environ.get(PREFLIGHT_SYSTEM_MOCK_ENV)
    if mock_path and synthetic:
        observed = read_json(Path(mock_path))
        observed["backend"] = "mocked_physical_system_interface"
        return observed
    if mock_path and not synthetic:
        return {
            "schema": "FAMILY10H_RELATION_LIFETIME_PHYSICAL_PREFLIGHT_OBSERVATION_V1",
            "backend": "forbidden_mocked_physical_system_interface",
            "forbidden_injected_env": PREFLIGHT_SYSTEM_MOCK_ENV,
            "cpu": {},
            "expected_artifacts": {},
            "pmu": {"pmu_open_count": 0, "pmu_acquisition_count": 0, "enabled_measurement_interval": False, "scientific_data_collected": False},
            "sensor": {},
            "frequency_policy": {},
            "attempt": {},
        }
    source_cpu = pub.APPROVED_TARGET_IDENTITY["source_cpu"]
    receiver_cpu = pub.APPROVED_TARGET_IDENTITY["receiver_cpu"]
    cpuinfo = Path("/proc/cpuinfo").read_text(encoding="utf-8")
    cpu = cpu_observation_from_cpuinfo(cpuinfo, source_cpu, receiver_cpu)
    affinity = probe_affinity(source_cpu, receiver_cpu)
    cpu["operational_pinning"] = affinity.get("operational_pinning") is True
    return {
        "schema": "FAMILY10H_RELATION_LIFETIME_PHYSICAL_PREFLIGHT_OBSERVATION_V1",
        "backend": "physical_system_interface",
        "cpu": cpu,
        "expected_artifacts": artifact_identity(source_root)
        | {"relation_source_authority_commit": relation_source_authority(read_json(source_root / "RELATION_LIFETIME_IMPLEMENTATION_MANIFEST.json"))},
        "runtime_binary_format_contains": ["ELF 64-bit", "x86-64"],
        "pmu": disabled_grouped_pmu_probe(source_root, source_cpu),
        "sensor": discover_temperature_sensor(),
        "frequency_policy": cpu_frequency_policy(source_cpu, receiver_cpu),
        "physical_geometry": {"status": "unresolved_contractually_lowered", "claim_ceiling_lowered": True},
        "output": {"root": "", "parent_writable": True},
        "attempt": {"owner_marker": pub.TRANSACTION_RUN_ID, "attempt_count_before": 0, "attempt_ceiling": pub.ATTEMPT_CEILING},
    }


def check(name: str, condition: bool, detail: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(condition), "detail": detail}


def output_root_law(source_root: Path, output_root: Path, fixture: dict[str, Any] | None = None) -> dict[str, Any]:
    failures: list[str] = []
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    expected_parent = owned_output_parent(source_root).resolve()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,96}", output_root.name):
        failures.append("output_path_malformed")
    if output_root.parent != expected_parent:
        failures.append("output_path_escapes_owned_transaction_root")
    if not expected_parent.exists():
        failures.append("output_parent_missing")
    elif not expected_parent.is_dir():
        failures.append("output_parent_not_directory")
    else:
        fixture_writable = True if fixture is None else fixture.get("output", {}).get("parent_writable", True)
        if not fixture_writable or not os.access(expected_parent, os.W_OK):
            failures.append("output_parent_permission_failure")
    if output_root.exists():
        failures.append("output_root_preexisting")
    return {
        "passed": not failures,
        "failures": failures,
        "expected_parent": str(expected_parent),
        "output_root": str(output_root),
    }


def authority_checks(source_root: Path, manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    manifest_file_sha = pub.sha256_file(source_root / "RELATION_LIFETIME_IMPLEMENTATION_MANIFEST.json")
    relation_source = relation_source_authority(manifest)
    deployment = read_deployment_custody(source_root)
    observed_freeze = os.environ.get(FREEZE_COMMIT_ENV)
    expected_freeze = deployment.get("relation_manifest_freeze_commit") or pub.SYNTHETIC_RELATION_FREEZE_COMMIT
    return {
        "live_authority": check("live_authority", os.environ.get(AUTHORITY_ENV) == AUTHORITY_VALUE, os.environ.get(AUTHORITY_ENV)),
        "source_authority_commit_binding": check(
            "source_authority_commit_binding",
            os.environ.get(SOURCE_AUTHORITY_ENV) == relation_source and relation_source_authority_is_valid(relation_source),
            {
                "observed": os.environ.get(SOURCE_AUTHORITY_ENV),
                "expected": relation_source,
            },
        ),
        "freeze_commit_binding": check(
            "freeze_commit_binding",
            observed_freeze == expected_freeze and bool(observed_freeze and HEX40_RE.fullmatch(observed_freeze)),
            {
                "observed": observed_freeze,
                "expected": expected_freeze,
            },
        ),
        "relation_source_authority_not_scalar": check(
            "relation_source_authority_not_scalar",
            relation_source not in pub.SCALAR_EVIDENCE_COMMITS,
            relation_source,
        ),
        "relation_freeze_authority_not_scalar": check(
            "relation_freeze_authority_not_scalar",
            observed_freeze not in pub.SCALAR_EVIDENCE_COMMITS,
            observed_freeze,
        ),
        "deployment_custody_binding": check(
            "deployment_custody_binding",
            not deployment
            or (
                deployment.get("relation_source_authority_commit") == relation_source
                and deployment.get("relation_manifest_freeze_commit") == observed_freeze
                and deployment.get("transaction_run_id") == pub.TRANSACTION_RUN_ID
                and deployment.get("manifest_file_sha256") == manifest_file_sha
            ),
            deployment,
        ),
        "manifest_file_hash": check(
            "manifest_file_hash",
            os.environ.get(MANIFEST_ENV) == manifest_file_sha,
            {"observed": os.environ.get(MANIFEST_ENV), "expected": manifest_file_sha},
        ),
    }


def target_preflight(source_root: Path, output_root: Path, fixture: dict[str, Any] | None) -> dict[str, Any]:
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    source_validation = validate_source_root(source_root)
    manifest = read_json(source_root / "RELATION_LIFETIME_IMPLEMENTATION_MANIFEST.json") if (source_root / "RELATION_LIFETIME_IMPLEMENTATION_MANIFEST.json").exists() else {}
    schedule_manifest = read_json(source_root / "RELATION_LIFETIME_PUBLIC_SCHEDULE.json") if (source_root / "RELATION_LIFETIME_PUBLIC_SCHEDULE.json").exists() else {}
    source_hashes = read_json(source_root / "RELATION_LIFETIME_SOURCE_HASHES.json") if (source_root / "RELATION_LIFETIME_SOURCE_HASHES.json").exists() else {}
    artifacts = artifact_identity(source_root) if source_validation["passed"] else {}
    synthetic = os.environ.get(EXECUTION_MODE_ENV) == EXECUTION_MODE_SYNTHETIC
    injected_envs = {key: os.environ.get(key) for key in INJECTED_PREFLIGHT_ENVS if os.environ.get(key)}
    checks: dict[str, dict[str, Any]] = {}
    checks.update(authority_checks(source_root, manifest))
    checks["source_root_identity"] = check("source_root_identity", source_validation["passed"], source_validation)
    checks["canonical_manifest_digest"] = check(
        "canonical_manifest_digest",
        manifest.get("manifest_sha256") == digest_without(manifest, "manifest_sha256") if manifest else False,
        manifest.get("manifest_sha256"),
    )
    checks["package_decision"] = check(
        "package_decision",
        manifest.get("package_decision") == pub.PACKAGE_DECISION_BUILD_READY,
        manifest.get("package_decision"),
    )
    checks["transaction_identity"] = check(
        "transaction_identity",
        manifest.get("transaction_run_id") == pub.TRANSACTION_RUN_ID,
        manifest.get("transaction_run_id"),
    )

    backend = "fixture" if fixture is not None else "physical_observed"
    observed = fixture if fixture is not None else physical_preflight_observation(source_root, output_root)
    injected_backend = fixture is not None or bool(injected_envs) or str(observed.get("backend", "")).startswith(("mocked", "forbidden", "fixture", "synthetic", "test"))
    checks["injected_preflight_backend_absent"] = check(
        "injected_preflight_backend_absent",
        synthetic or not injected_backend,
        {"synthetic_mode": synthetic, "fixture_argument": fixture is not None, "injected_envs": injected_envs, "backend": observed.get("backend")},
    )
    expected = observed.get("expected_artifacts", {})
    expected_source = expected.get("relation_source_authority_commit")
    expected_freeze = expected.get("relation_freeze_authority_commit")
    checks["manifest_file_hash"].update(
        check(
            "manifest_file_hash",
            checks["manifest_file_hash"]["passed"]
            and expected.get("manifest_file_sha256") == artifacts.get("manifest_file_sha256"),
            {"observed_backend": backend, "expected": expected.get("manifest_file_sha256"), "env": os.environ.get(MANIFEST_ENV)},
        )
    )
    checks["grammar_hash"] = check(
        "grammar_hash",
        expected.get("grammar_canonical_sha256") == artifacts.get("grammar_canonical_sha256") == manifest.get("grammar_sha256"),
        expected.get("grammar_canonical_sha256"),
    )
    checks["compact_schedule_manifest_binding"] = check(
        "compact_schedule_manifest_binding",
        expected.get("schedule_canonical_sha256") == artifacts.get("schedule_canonical_sha256") == manifest.get("schedule_sha256")
        and schedule_manifest.get("schedule_manifest_sha256") == digest_without(schedule_manifest, "schedule_manifest_sha256"),
        expected.get("schedule_canonical_sha256"),
    )
    checks["expanded_schedule_tsv_hash"] = check(
        "expanded_schedule_tsv_hash",
        expected.get("schedule_tsv_sha256") == artifacts.get("schedule_tsv_sha256") == schedule_manifest.get("expanded_schedule_file_sha256"),
        expected.get("schedule_tsv_sha256"),
    )
    checks["runtime_binary_hash"] = check(
        "runtime_binary_hash",
        expected.get("runtime_binary_sha256") == artifacts.get("runtime_binary_sha256") == source_hashes.get("runtime_binary_authority", {}).get("compiled_binary_sha256"),
        expected.get("runtime_binary_sha256"),
    )
    format_text = artifacts.get("runtime_binary_format", "")
    expected_format = observed.get("runtime_binary_format_contains", [])
    checks["runtime_binary_format_and_abi"] = check(
        "runtime_binary_format_and_abi",
        all(item in format_text for item in expected_format),
        {"format": format_text, "required": expected_format},
    )
    checks["pmu_helper_binary_hash"] = check(
        "pmu_helper_binary_hash",
        expected.get("pmu_preflight_helper_actual_sha256") == artifacts.get("pmu_preflight_helper_actual_sha256")
        and expected.get("pmu_preflight_helper_expected_sha256") == artifacts.get("pmu_preflight_helper_expected_sha256")
        and artifacts.get("pmu_preflight_helper_sha256_matches_authority") is True
        and expected.get("pmu_preflight_helper_sha256_matches_authority") is True,
        {
            "expected_actual": expected.get("pmu_preflight_helper_actual_sha256"),
            "observed_actual": artifacts.get("pmu_preflight_helper_actual_sha256"),
            "expected_authority": artifacts.get("pmu_preflight_helper_expected_sha256"),
            "equality": artifacts.get("pmu_preflight_helper_sha256_matches_authority"),
        },
    )
    checks["pmu_helper_binary_format_and_abi"] = check(
        "pmu_helper_binary_format_and_abi",
        expected.get("pmu_preflight_helper_format_valid") is True
        and artifacts.get("pmu_preflight_helper_format_valid") is True
        and helper_format_valid(str(artifacts.get("pmu_preflight_helper_actual_format", ""))),
        {
            "format": artifacts.get("pmu_preflight_helper_actual_format"),
            "expected_abi": pub.APPROVED_TARGET_IDENTITY["runtime_abi"],
        },
    )
    checks["source_bundle_hash"] = check(
        "source_bundle_hash",
        expected.get("source_bundle_sha256") == artifacts.get("source_bundle_sha256"),
        expected.get("source_bundle_sha256"),
    )
    checks["source_file_hashes"] = check(
        "source_file_hashes",
        source_validation["passed"] and bool(artifacts.get("source_file_hashes")),
        artifacts.get("source_file_hashes"),
    )
    checks["source_authority_commit_binding"].update(
        check(
            "source_authority_commit_binding",
            checks["source_authority_commit_binding"]["passed"]
            and expected_source == relation_source_authority(manifest),
            {"observed_backend": backend, "expected": expected_source, "env": os.environ.get(SOURCE_AUTHORITY_ENV)},
        )
    )
    checks["freeze_commit_binding"].update(
        check(
            "freeze_commit_binding",
            checks["freeze_commit_binding"]["passed"]
            and (expected_freeze is None or expected_freeze == os.environ.get(FREEZE_COMMIT_ENV)),
            {"observed_backend": backend, "expected": expected_freeze, "env": os.environ.get(FREEZE_COMMIT_ENV)},
        )
    )
    cpu = observed.get("cpu", {})
    approved_target_identity = {key: cpu.get(key) for key in pub.APPROVED_TARGET_IDENTITY}
    checks["cpu_vendor_family_model"] = check(
        "cpu_vendor_family_model",
        cpu.get("vendor") == pub.APPROVED_TARGET_IDENTITY["vendor"]
        and cpu.get("family") == pub.APPROVED_TARGET_IDENTITY["family"]
        and cpu.get("model") == pub.APPROVED_TARGET_IDENTITY["model"],
        cpu,
    )
    checks["source_receiver_cpu_identity"] = check(
        "source_receiver_cpu_identity",
        cpu.get("source_cpu_present") is True
        and cpu.get("receiver_cpu_present") is True
        and cpu.get("source_cpu") == pub.APPROVED_TARGET_IDENTITY["source_cpu"]
        and cpu.get("receiver_cpu") == pub.APPROVED_TARGET_IDENTITY["receiver_cpu"]
        and cpu.get("source_cpu") != cpu.get("receiver_cpu")
        and cpu.get("processors") == pub.APPROVED_TARGET_IDENTITY["processors"]
        and cpu.get("processor_count") == pub.APPROVED_TARGET_IDENTITY["processor_count"],
        cpu,
    )
    checks["approved_target_identity"] = check(
        "approved_target_identity",
        approved_target_identity == pub.APPROVED_TARGET_IDENTITY and cpu.get("target_identity_sha256") == pub.APPROVED_TARGET_IDENTITY_SHA256,
        {"observed_identity": approved_target_identity, "observed_sha256": cpu.get("target_identity_sha256"), "expected_sha256": pub.APPROVED_TARGET_IDENTITY_SHA256},
    )
    checks["operational_pinning_capability"] = check("operational_pinning_capability", cpu.get("operational_pinning") is True, cpu)
    pmu = observed.get("pmu", {})
    checks["pmu_event_identities"] = check("pmu_event_identities", pmu.get("events") == pub.PMU_GROUP["events"], pmu.get("events"))
    pmu_strict = (
        pmu.get("returncode") == 0
        and pmu.get("passed") is True
        and pmu.get("helper_sha256") == artifacts.get("pmu_preflight_helper_expected_sha256")
        and pmu.get("helper_sha256_matches_authority") is True
        and pmu.get("uses_system_perf_event_attr") is True
        and pmu.get("perf_event_attr_size") == EXPECTED_PERF_EVENT_ATTR_SIZE
        and pmu.get("pmu_open_count") == len(pub.PMU_GROUP["events"])
        and pmu.get("pmu_close_count") == len(pub.PMU_GROUP["events"])
        and pmu.get("pmu_fd_leak_count") == 0
        and pmu.get("pmu_acquisition_count") == 0
        and pmu.get("enabled_measurement_interval") is False
        and pmu.get("scientific_data_collected") is False
        and pmu.get("enable_attempted") is False
        and pmu.get("reset_attempted") is False
        and pmu.get("read_attempted") is False
        and pmu.get("malformed_event_identity") is False
        and pmu.get("partial_group_open") is False
        and pmu.get("incorrect_structure_size") is False
        and pmu.get("missing_event") is False
    )
    checks["grouped_pmu_open_capability"] = check(
        "grouped_pmu_open_capability",
        pmu.get("grouped_open_capability") is True and pmu_strict,
        pmu,
    )
    sensor = observed.get("sensor", {})
    checks["temperature_sensor_authority"] = check(
        "temperature_sensor_authority",
        sensor_identity_matches(sensor),
        sensor,
    )
    policy = observed.get("frequency_policy", {})
    checks["cpu_frequency_policy_custody"] = check(
        "cpu_frequency_policy_custody",
        policy.get("policy_locked") is True and policy.get("identity_stability") is True,
        policy,
    )
    geometry = observed.get("physical_geometry", {})
    checks["actual_physical_geometry_status"] = check(
        "actual_physical_geometry_status",
        geometry.get("status") in {"verified", "unresolved_contractually_lowered"} and geometry.get("claim_ceiling_lowered") is True,
        geometry,
    )
    output_law = output_root_law(source_root, output_root, observed)
    checks["output_root_absence"] = check("output_root_absence", output_law["passed"], output_law)
    attempt = observed.get("attempt", {})
    checks["attempt_ownership_marker"] = check(
        "attempt_ownership_marker",
        attempt.get("owner_marker") == pub.TRANSACTION_RUN_ID
        and attempt.get("attempt_count_before") == 0
        and attempt.get("attempt_ceiling") == pub.ATTEMPT_CEILING,
        attempt,
    )

    failures = [name for name in TARGET_PREFLIGHT_CHECKS if not checks.get(name, {"passed": False})["passed"]]
    failures.extend(name for name, item in checks.items() if name not in TARGET_PREFLIGHT_CHECKS and not item["passed"])
    return {
        "schema": "FAMILY10H_RELATION_LIFETIME_TARGET_PREFLIGHT_V2",
        "status": "TARGET_PREFLIGHT_PASSED" if not failures else "TARGET_PREFLIGHT_FAILED",
        "passed": not failures,
        "failures": failures,
        "checks": checks,
        "preflight_backend": backend,
        "physical_observation": observed if fixture is None else None,
        "artifact_identity": artifacts,
        "output_root": str(output_root),
        "target_contact_count": 0,
        "sensor_inventory_count": 0 if fixture is not None else 1,
        "pmu_open_count": observed.get("pmu", {}).get("pmu_open_count", 0),
        "pmu_acquisition_count": 0,
        "live_invocation_count": 0,
        "small_wall_crossed": False,
    }


def target_preflight_refusal(source_root: Path, output_root: Path) -> dict[str, Any]:
    fixture = base_preflight_fixture(source_root, output_root)
    result = target_preflight(source_root, output_root, fixture)
    result["schema"] = "FAMILY10H_RELATION_LIFETIME_TARGET_PREFLIGHT_REFUSAL_V2"
    result["status"] = "TARGET_PREFLIGHT_REFUSED"
    result["reason"] = "live authority missing" if os.environ.get(AUTHORITY_ENV) != AUTHORITY_VALUE else "preflight fixture backend missing"
    result["passed"] = os.environ.get(AUTHORITY_ENV) != AUTHORITY_VALUE and "live_authority" in result.get("failures", [])
    return result


@contextmanager
def temporary_env(updates: dict[str, str | None]) -> Iterator[None]:
    old = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def write_fixture(path: Path, fixture: dict[str, Any]) -> None:
    path.write_text(strict_json_dumps(fixture, indent=2) + "\n", encoding="utf-8")


def deployment_custody(source_root: Path, freeze_commit: str) -> dict[str, Any]:
    manifest = read_json(source_root / "RELATION_LIFETIME_IMPLEMENTATION_MANIFEST.json")
    return {
        "schema": "FAMILY10H_RELATION_LIFETIME_DEPLOYMENT_CUSTODY_V1",
        "science_package_id": pub.SCIENCE_PACKAGE_ID,
        "transaction_run_id": pub.TRANSACTION_RUN_ID,
        "relation_source_authority_commit": relation_source_authority(manifest),
        "relation_manifest_freeze_commit": freeze_commit,
        "manifest_file_sha256": pub.sha256_file(source_root / "RELATION_LIFETIME_IMPLEMENTATION_MANIFEST.json"),
        "controller_verified_head_equals_origin": True,
        "controller_verified_clean_worktree": True,
        "fixture_backend_allowed": False,
        "one_attempt_ceiling": pub.ATTEMPT_CEILING,
    }


def target_authority_env(source_root: Path, fixture_path: Path | None, *, synthetic: bool = False) -> dict[str, str | None]:
    manifest_sha = pub.sha256_file(source_root / "RELATION_LIFETIME_IMPLEMENTATION_MANIFEST.json")
    manifest = read_json(source_root / "RELATION_LIFETIME_IMPLEMENTATION_MANIFEST.json")
    return {
        AUTHORITY_ENV: AUTHORITY_VALUE,
        SOURCE_AUTHORITY_ENV: relation_source_authority(manifest),
        FREEZE_COMMIT_ENV: pub.SYNTHETIC_RELATION_FREEZE_COMMIT,
        MANIFEST_ENV: manifest_sha,
        PREFLIGHT_FIXTURE_ENV: str(fixture_path) if fixture_path else None,
        PREFLIGHT_SYSTEM_MOCK_ENV: None,
        EXECUTION_MODE_ENV: EXECUTION_MODE_SYNTHETIC if synthetic else None,
        LEGACY_COMMIT_ENV: None,
    }


def execute_authorized(source_root: Path, output_root: Path) -> dict[str, Any]:
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    synthetic = os.environ.get(EXECUTION_MODE_ENV) == EXECUTION_MODE_SYNTHETIC
    if not synthetic:
        if os.environ.get(PREFLIGHT_FIXTURE_ENV):
            raise TargetError("physical execution refuses fixture-backed preflight")
        if os.environ.get(PREFLIGHT_SYSTEM_MOCK_ENV):
            raise TargetError("physical execution refuses system-mock preflight")
    fixture = load_fixture()
    preflight = target_preflight(source_root, output_root, fixture)
    if not preflight["passed"]:
        raise TargetError("target preflight failed: " + ",".join(preflight["failures"]))
    runtime = runtime_binary_path(source_root)
    if runtime is None:
        raise TargetError("runtime binary missing")
    env = os.environ.copy()
    if synthetic:
        env.pop(RUNTIME_AUTHORITY_ENV, None)
        runtime_mode = "--synthetic-execute-schedule"
    else:
        env[RUNTIME_AUTHORITY_ENV] = AUTHORITY_VALUE
        runtime_mode = "--execute-schedule"
    command, launch_mode = runtime_command(
        source_root,
        runtime,
        [runtime_mode, source_root / "RELATION_LIFETIME_PUBLIC_SCHEDULE.tsv", output_root],
    )
    completed = subprocess.run(command, text=True, capture_output=True, check=False, timeout=3600, env=env)
    raw_path = output_root / "raw_records.jsonl"
    death_path = output_root / "source_death_receipts.jsonl"
    feature_path = output_root / "feature_freeze.json"
    runtime_receipt_path = output_root / "target_execution_receipt.json"
    raw_count = count_jsonl(raw_path) if raw_path.exists() else 0
    death_count = count_jsonl(death_path) if death_path.exists() else 0
    feature = read_json(feature_path) if feature_path.exists() else None
    runtime_receipt = read_json(runtime_receipt_path) if runtime_receipt_path.exists() else None
    passed = (
        completed.returncode == 0
        and raw_count == 21504
        and death_count == 21504
        and feature is not None
        and runtime_receipt is not None
        and feature.get("physical_measurement") is (not synthetic)
        and runtime_receipt.get("physical_measurement") is (not synthetic)
    )
    return {
        "schema": "FAMILY10H_RELATION_LIFETIME_SYNTHETIC_TARGET_WRAPPER_EXECUTION_V1"
        if synthetic
        else "FAMILY10H_RELATION_LIFETIME_TARGET_WRAPPER_EXECUTION_V1",
        "status": "TARGET_WRAPPER_EXECUTION_COMPLETE" if passed else "TARGET_WRAPPER_EXECUTION_FAILED",
        "passed": passed,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "launch_mode": launch_mode,
        "runtime_mode": runtime_mode,
        "preflight": preflight,
        "output_root": str(output_root),
        "raw_record_count": raw_count,
        "source_death_receipt_count": death_count,
        "feature_freeze": feature,
        "runtime_execution_receipt": runtime_receipt,
        "physical_measurement": not synthetic,
        "pmu_open_count": 0 if synthetic else runtime_receipt.get("pmu_opened") if runtime_receipt else 0,
        "target_contact_count": 0,
        "sensor_inventory_count": 0,
        "live_invocation_count": 0 if synthetic else 1,
        "pmu_acquisition_count": 0 if synthetic else (1 if completed.returncode == 0 else 0),
        "scientific_claim_emitted": False,
        "positive_physical_claim": False,
        "small_wall_crossed": False,
    }


def count_jsonl(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                raise TargetError(f"blank JSONL line in {path}")
            json.loads(line)
            count += 1
    return count


def run_preflight_fixture_suite(source_root: Path, parent: Path) -> dict[str, Any]:
    output_root = parent / "preflight_attempt"
    base = base_preflight_fixture(source_root, output_root)
    def run_preflight(candidate_root: Path, fixture: dict[str, Any]) -> dict[str, Any]:
        with temporary_env(target_authority_env(source_root, None, synthetic=True)):
            return target_preflight(source_root, candidate_root, fixture)

    cases: dict[str, tuple[str | None, dict[str, Any]]] = {
        "complete_passing_fixture": (None, base),
        "wrong_cpu_family": ("cpu_vendor_family_model", {**base, "cpu": {**base["cpu"], "family": 15}}),
        "wrong_cpu_model": ("cpu_vendor_family_model", {**base, "cpu": {**base["cpu"], "model": 6}}),
        "wrong_target_identity_sha": ("approved_target_identity", {**base, "cpu": {**base["cpu"], "target_identity_sha256": "0" * 64}}),
        "incompatible_processor_topology": ("source_receiver_cpu_identity", {**base, "cpu": {**base["cpu"], "processor_count": 4, "processors": base["cpu"]["processors"][:4]}}),
        "missing_source_cpu": ("source_receiver_cpu_identity", {**base, "cpu": {**base["cpu"], "source_cpu_present": False}}),
        "missing_receiver_cpu": ("source_receiver_cpu_identity", {**base, "cpu": {**base["cpu"], "receiver_cpu_present": False}}),
        "failed_operational_pinning": ("operational_pinning_capability", {**base, "cpu": {**base["cpu"], "operational_pinning": False}}),
        "wrong_binary_hash": (
            "runtime_binary_hash",
            {**base, "expected_artifacts": {**base["expected_artifacts"], "runtime_binary_sha256": "0" * 64}},
        ),
        "wrong_helper_binary_hash": (
            "pmu_helper_binary_hash",
            {**base, "expected_artifacts": {**base["expected_artifacts"], "pmu_preflight_helper_actual_sha256": "0" * 64}},
        ),
        "fake_helper_output_correct_schema": (
            "grouped_pmu_open_capability",
            {**base, "pmu": {**base["pmu"], "helper_sha256_matches_authority": False}},
        ),
        "helper_returning_nonzero_with_plausible_json": (
            "grouped_pmu_open_capability",
            {**base, "pmu": {**base["pmu"], "returncode": 1, "passed": True}},
        ),
        "helper_reported_enable_attempt": (
            "grouped_pmu_open_capability",
            {**base, "pmu": {**base["pmu"], "enable_attempted": True}},
        ),
        "helper_reported_read_attempt": (
            "grouped_pmu_open_capability",
            {**base, "pmu": {**base["pmu"], "read_attempted": True}},
        ),
        "helper_reported_reset_attempt": (
            "grouped_pmu_open_capability",
            {**base, "pmu": {**base["pmu"], "reset_attempted": True}},
        ),
        "helper_reported_leaked_descriptor": (
            "grouped_pmu_open_capability",
            {**base, "pmu": {**base["pmu"], "pmu_fd_leak_count": 1}},
        ),
        "helper_open_close_mismatch": (
            "grouped_pmu_open_capability",
            {**base, "pmu": {**base["pmu"], "pmu_close_count": len(pub.PMU_GROUP["events"]) - 1}},
        ),
        "helper_wrong_structure_size": (
            "grouped_pmu_open_capability",
            {**base, "pmu": {**base["pmu"], "perf_event_attr_size": EXPECTED_PERF_EVENT_ATTR_SIZE + 8}},
        ),
        "helper_partial_group_open": (
            "grouped_pmu_open_capability",
            {**base, "pmu": {**base["pmu"], "partial_group_open": True}},
        ),
        "helper_missing_event": (
            "grouped_pmu_open_capability",
            {**base, "pmu": {**base["pmu"], "missing_event": True}},
        ),
        "wrong_manifest_hash": (
            "manifest_file_hash",
            {**base, "expected_artifacts": {**base["expected_artifacts"], "manifest_file_sha256": "0" * 64}},
        ),
        "wrong_schedule_hash": (
            "expanded_schedule_tsv_hash",
            {**base, "expected_artifacts": {**base["expected_artifacts"], "schedule_tsv_sha256": "0" * 64}},
        ),
        "wrong_source_authority_commit": (
            "source_authority_commit_binding",
            {**base, "expected_artifacts": {**base["expected_artifacts"], "relation_source_authority_commit": "0" * 40}},
        ),
        "unavailable_pmu_event": ("pmu_event_identities", {**base, "pmu": {**base["pmu"], "events": {}}}),
        "grouped_pmu_failure": ("grouped_pmu_open_capability", {**base, "pmu": {**base["pmu"], "grouped_open_capability": False}}),
        "sensor_identity_mismatch": (
            "temperature_sensor_authority",
            {**base, "sensor": {**base["sensor"], "identity_sha256": "0" * 64}},
        ),
        "sensor_wrong_driver": ("temperature_sensor_authority", {**base, "sensor": {**base["sensor"], "device_driver": "nouveau"}}),
        "sensor_wrong_modalias": ("temperature_sensor_authority", {**base, "sensor": {**base["sensor"], "device_modalias": "pci:v000010DEd00001C81"}}),
        "sensor_wrong_input_path": ("temperature_sensor_authority", {**base, "sensor": {**base["sensor"], "resolved_input_path": "/sys/devices/wrong/temp1_input"}}),
        "sensor_volatile_inode_device_mode_variation": (None, {**base, "sensor": {**base["sensor"], "input_st_ino": 1, "input_st_dev": 1, "input_st_mode": 0}}),
        "sensor_wrong_semantic_profile": ("temperature_sensor_authority", {**base, "sensor": {**base["sensor"], "sensor_semantic_profile": "UNAPPROVED"}}),
        "sensor_unexpected_label": ("temperature_sensor_authority", {**base, "sensor": {**base["sensor"], "sensor_label_present": True, "sensor_label_value": "Tctl"}}),
        "sensor_unstable_identity": ("temperature_sensor_authority", {**base, "sensor": {**base["sensor"], "identity_stability": False}}),
        "policy_mismatch": ("cpu_frequency_policy_custody", {**base, "frequency_policy": {**base["frequency_policy"], "policy_locked": False}}),
        "permission_failure": ("output_root_absence", {**base, "output": {**base["output"], "parent_writable": False}}),
        "owner_marker_mismatch": ("attempt_ownership_marker", {**base, "attempt": {**base["attempt"], "owner_marker": "wrong-owner"}}),
        "unresolved_physical_geometry_lowered": (None, base),
    }
    results: dict[str, Any] = {}
    for label, (expected_failure, fixture) in cases.items():
        result = run_preflight(output_root, fixture)
        if expected_failure is None:
            passed = result["passed"]
        else:
            passed = not result["passed"] and expected_failure in result["failures"]
        results[label] = {
            "passed": passed,
            "expected_failure": expected_failure,
            "status": result["status"],
            "failures": result["failures"],
        }

    preexisting = parent / "preexisting_attempt"
    preexisting.mkdir()
    result = run_preflight(preexisting, {**base, "output": {**base["output"], "root": str(preexisting)}})
    results["preexisting_output_root"] = {
        "passed": not result["passed"] and "output_root_absence" in result["failures"],
        "status": result["status"],
        "failures": result["failures"],
    }

    partial = parent / "partial_attempt"
    partial.mkdir()
    (partial / "raw_records.jsonl").write_text("{}", encoding="utf-8")
    result = run_preflight(partial, {**base, "output": {**base["output"], "root": str(partial)}})
    results["partial_output"] = {
        "passed": not result["passed"] and "output_root_absence" in result["failures"],
        "status": result["status"],
        "failures": result["failures"],
    }

    malformed = parent / "bad path"
    result = run_preflight(malformed, {**base, "output": {**base["output"], "root": str(malformed)}})
    results["malformed_output_path"] = {
        "passed": not result["passed"] and "output_root_absence" in result["failures"],
        "status": result["status"],
        "failures": result["failures"],
    }

    escaped = source_root / "escaped_attempt"
    result = run_preflight(escaped, {**base, "output": {**base["output"], "root": str(escaped)}})
    results["output_path_escape"] = {
        "passed": not result["passed"] and "output_root_absence" in result["failures"],
        "status": result["status"],
        "failures": result["failures"],
    }

    missing_parent = parent / "child"
    shutil.rmtree(parent)
    result = run_preflight(missing_parent, {**base, "output": {**base["output"], "root": str(missing_parent)}})
    results["missing_parent"] = {
        "passed": not result["passed"] and "output_root_absence" in result["failures"],
        "status": result["status"],
        "failures": result["failures"],
    }
    parent.mkdir()
    return {
        "schema": "FAMILY10H_RELATION_LIFETIME_TARGET_PREFLIGHT_FIXTURE_SUITE_V1",
        "results": results,
        "passed": all(item["passed"] for item in results.values()),
    }


def run_helper_authority_negative_suite(source_root: Path, parent: Path) -> dict[str, Any]:
    def copied_root(label: str) -> Path:
        destination = parent / f"helper_authority_{label}"
        shutil.copytree(
            source_root,
            destination,
            ignore=shutil.ignore_patterns("__pycache__", pub.OWNED_OUTPUT_PARENT_NAME),
        )
        return destination

    def run_case(label: str, expected_failure: str, mutator: Any) -> dict[str, Any]:
        candidate = copied_root(label)
        mutator(candidate)
        report = validate_source_root(candidate)
        return {
            "passed": report["passed"] is False and expected_failure in report.get("failures", []),
            "expected_failure": expected_failure,
            "failures": report.get("failures", []),
        }

    def mutate_helper_binary(candidate: Path) -> None:
        helper = pmu_preflight_helper_path(candidate)
        if helper is None:
            return
        with helper.open("ab") as handle:
            handle.write(b"\nmutated-helper-byte\n")

    def mutate_helper_authority(candidate: Path) -> None:
        path = candidate / "RELATION_LIFETIME_SOURCE_HASHES.json"
        data = read_json(path)
        data["pmu_preflight_helper_authority"]["compiled_binary_sha256"] = "0" * 64
        path.write_text(strict_json_dumps(data, indent=2) + "\n", encoding="utf-8")

    results = {
        "modified_helper_binary": run_case(
            "modified_helper_binary",
            "PMU preflight helper binary hash mismatch",
            mutate_helper_binary,
        ),
        "wrong_helper_hash_in_authority": run_case(
            "wrong_helper_hash_in_authority",
            "PMU preflight helper binary hash mismatch",
            mutate_helper_authority,
        ),
    }
    return {
        "schema": "FAMILY10H_RELATION_LIFETIME_HELPER_AUTHORITY_NEGATIVE_SUITE_V1",
        "results": results,
        "passed": all(item["passed"] for item in results.values()),
    }


def base_physical_mock(source_root: Path, output_root: Path) -> dict[str, Any]:
    mock = base_preflight_fixture(source_root, output_root)
    mock["schema"] = "FAMILY10H_RELATION_LIFETIME_PHYSICAL_PREFLIGHT_OBSERVATION_V1"
    mock["backend"] = "mocked_physical_system_interface"
    mock["pmu"]["pmu_open_count"] = len(pub.PMU_GROUP["events"])
    mock["pmu"]["pmu_acquisition_count"] = 0
    mock["pmu"]["enabled_measurement_interval"] = False
    mock["pmu"]["scientific_data_collected"] = False
    return mock


def write_deployment_custody(path: Path, source_root: Path, freeze_commit: str = pub.SYNTHETIC_RELATION_FREEZE_COMMIT) -> None:
    path.write_text(strict_json_dumps(deployment_custody(source_root, freeze_commit), indent=2) + "\n", encoding="utf-8")


def run_physical_preflight_mock_suite(source_root: Path, parent: Path) -> dict[str, Any]:
    output_root = parent / "physical_preflight_attempt"
    base = base_physical_mock(source_root, output_root)
    results: dict[str, Any] = {}

    with tempfile.TemporaryDirectory(
        prefix="relation_lifetime_physical_preflight_",
        dir="C:/tmp" if os.name == "nt" else None,
    ) as scratch_text:
        scratch = Path(scratch_text)
        custody_path = scratch / pub.DEPLOYMENT_CUSTODY_FILENAME
        write_deployment_custody(custody_path, source_root)

        def run_case(candidate_root: Path, observed: dict[str, Any], env_override: dict[str, str | None] | None = None) -> dict[str, Any]:
            mock_path = scratch / f"physical_mock_{len(results)}.json"
            write_fixture(mock_path, observed)
            env = {
                **target_authority_env(source_root, None, synthetic=True),
                PREFLIGHT_SYSTEM_MOCK_ENV: str(mock_path),
                DEPLOYMENT_CUSTODY_ENV: str(custody_path),
                **(env_override or {}),
            }
            with temporary_env(env):
                return target_preflight(source_root, candidate_root, None)

        cases: dict[str, tuple[str | None, dict[str, Any], dict[str, str | None] | None]] = {
            "complete_physical_mock": (None, base, None),
            "wrong_cpu_family": ("cpu_vendor_family_model", {**base, "cpu": {**base["cpu"], "family": 15}}, None),
            "wrong_cpu_model": ("cpu_vendor_family_model", {**base, "cpu": {**base["cpu"], "model": 6}}, None),
            "wrong_target_identity_sha": ("approved_target_identity", {**base, "cpu": {**base["cpu"], "target_identity_sha256": "0" * 64}}, None),
            "failed_cpu_pinning": ("operational_pinning_capability", {**base, "cpu": {**base["cpu"], "operational_pinning": False}}, None),
            "pmu_event_mismatch": ("pmu_event_identities", {**base, "pmu": {**base["pmu"], "events": {}}}, None),
            "grouped_pmu_failure": ("grouped_pmu_open_capability", {**base, "pmu": {**base["pmu"], "grouped_open_capability": False}}, None),
            "sensor_mismatch": ("temperature_sensor_authority", {**base, "sensor": {**base["sensor"], "identity_sha256": "0" * 64}}, None),
            "sensor_wrong_driver": ("temperature_sensor_authority", {**base, "sensor": {**base["sensor"], "device_driver": "nouveau"}}, None),
            "sensor_wrong_modalias": ("temperature_sensor_authority", {**base, "sensor": {**base["sensor"], "device_modalias": "pci:v000010DEd00001C81"}}, None),
            "sensor_wrong_input_path": ("temperature_sensor_authority", {**base, "sensor": {**base["sensor"], "resolved_input_path": "/sys/devices/wrong/temp1_input"}}, None),
            "sensor_volatile_inode_device_mode_variation": (None, {**base, "sensor": {**base["sensor"], "input_st_ino": 1, "input_st_dev": 1, "input_st_mode": 0}}, None),
            "sensor_wrong_semantic_profile": ("temperature_sensor_authority", {**base, "sensor": {**base["sensor"], "sensor_semantic_profile": "UNAPPROVED"}}, None),
            "sensor_unexpected_label": ("temperature_sensor_authority", {**base, "sensor": {**base["sensor"], "sensor_label_present": True, "sensor_label_value": "Tctl"}}, None),
            "sensor_unstable_identity": ("temperature_sensor_authority", {**base, "sensor": {**base["sensor"], "identity_stability": False}}, None),
            "policy_mismatch": ("cpu_frequency_policy_custody", {**base, "frequency_policy": {**base["frequency_policy"], "policy_locked": False}}, None),
            "invalid_owner_marker": ("attempt_ownership_marker", {**base, "attempt": {**base["attempt"], "owner_marker": "wrong-owner"}}, None),
            "old_scalar_source_commit_rejected": ("source_authority_commit_binding", base, {SOURCE_AUTHORITY_ENV: pub.SCALAR_EVIDENCE_SOURCE_AUTHORITY_COMMIT}),
            "old_scalar_freeze_commit_rejected": ("freeze_commit_binding", base, {FREEZE_COMMIT_ENV: pub.SCALAR_EVIDENCE_MANIFEST_FREEZE_COMMIT}),
        }

        for label, (expected_failure, observed, env_override) in cases.items():
            result = run_case(output_root, observed, env_override)
            if expected_failure is None:
                passed = result["passed"] and result.get("preflight_backend") == "physical_observed"
            else:
                passed = not result["passed"] and expected_failure in result["failures"]
            results[label] = {
                "passed": passed,
                "expected_failure": expected_failure,
                "status": result["status"],
                "failures": result["failures"],
            }

        preexisting = parent / "physical_preexisting_attempt"
        preexisting.mkdir()
        result = run_case(preexisting, {**base, "output": {**base["output"], "root": str(preexisting)}})
        results["preexisting_output_root"] = {
            "passed": not result["passed"] and "output_root_absence" in result["failures"],
            "expected_failure": "output_root_absence",
            "status": result["status"],
            "failures": result["failures"],
        }
        escaped = source_root / "physical_escape_attempt"
        result = run_case(escaped, {**base, "output": {**base["output"], "root": str(escaped)}})
        results["output_path_escape"] = {
            "passed": not result["passed"] and "output_root_absence" in result["failures"],
            "expected_failure": "output_root_absence",
            "status": result["status"],
            "failures": result["failures"],
        }
    return {
        "schema": "FAMILY10H_RELATION_LIFETIME_PHYSICAL_PREFLIGHT_MOCK_SUITE_V1",
        "results": results,
        "passed": all(item["passed"] for item in results.values()),
        "pmu_preflight_behavior": {
            "disabled_group_open_close_only": True,
            "pmu_acquisition_count": 0,
            "scientific_data_collected": False,
        },
    }


def run_synthetic_target_wrapper_test(source_root: Path, parent: Path) -> dict[str, Any]:
    output_root = parent / "synthetic_target_wrapper_attempt"
    fixture = base_preflight_fixture(source_root, output_root)
    fixture_path = parent / "preflight_fixture.json"
    write_fixture(fixture_path, fixture)
    with temporary_env(target_authority_env(source_root, fixture_path, synthetic=True)):
        try:
            result = execute_authorized(source_root, output_root)
        except TargetError as exc:
            return {
                "schema": "FAMILY10H_RELATION_LIFETIME_SYNTHETIC_TARGET_WRAPPER_EXECUTION_V1",
                "status": "TARGET_WRAPPER_EXECUTION_REFUSED",
                "passed": False,
                "reason": str(exc),
                "preflight": target_preflight(source_root, output_root, fixture),
                "output_root": str(output_root),
                "target_contact_count": 0,
                "sensor_inventory_count": 0,
                "live_invocation_count": 0,
                "pmu_acquisition_count": 0,
                "physical_measurement": False,
                "scientific_claim_emitted": False,
                "positive_physical_claim": False,
                "small_wall_crossed": False,
            }
    duplicate_refusal: dict[str, Any]
    with temporary_env(target_authority_env(source_root, fixture_path, synthetic=True)):
        try:
            execute_authorized(source_root, output_root)
        except TargetError as exc:
            duplicate_refusal = {
                "passed": "output_root_absence" in str(exc) or "target preflight failed" in str(exc),
                "reason": str(exc),
            }
        else:
            duplicate_refusal = {"passed": False, "reason": "duplicate invocation unexpectedly succeeded"}
    result["duplicate_invocation_refusal"] = duplicate_refusal
    result["passed"] = result["passed"] and duplicate_refusal["passed"]
    return result


def run_authority_refusal_tests(source_root: Path, parent: Path) -> dict[str, Any]:
    output_root = parent / "authority_attempt"
    fixture = base_preflight_fixture(source_root, output_root)
    fixture_path = parent / "authority_fixture.json"
    write_fixture(fixture_path, fixture)
    tests: dict[str, Any] = {}
    cases = {
        "wrong_authority": {AUTHORITY_ENV: "wrong"},
        "source_authority_mismatch": {SOURCE_AUTHORITY_ENV: "0" * 40},
        "freeze_commit_mismatch": {FREEZE_COMMIT_ENV: "0" * 40},
        "manifest_hash_mismatch": {MANIFEST_ENV: "0" * 64},
        "old_scalar_source_commit_substitution": {SOURCE_AUTHORITY_ENV: pub.SCALAR_EVIDENCE_SOURCE_AUTHORITY_COMMIT},
        "old_scalar_freeze_commit_substitution": {FREEZE_COMMIT_ENV: pub.SCALAR_EVIDENCE_MANIFEST_FREEZE_COMMIT},
    }
    base_env = target_authority_env(source_root, fixture_path, synthetic=True)
    for label, override in cases.items():
        env = {**base_env, **override}
        with temporary_env(env):
            try:
                execute_authorized(source_root, output_root)
            except TargetError as exc:
                tests[label] = {"passed": True, "reason": str(exc)}
            else:
                tests[label] = {"passed": False, "reason": "unexpectedly accepted"}
    physical_fixture_env = target_authority_env(source_root, fixture_path, synthetic=False)
    with temporary_env(physical_fixture_env):
        try:
            execute_authorized(source_root, output_root)
        except TargetError as exc:
            tests["fixture_forbidden_on_physical_path"] = {
                "passed": "refuses fixture-backed preflight" in str(exc),
                "reason": str(exc),
            }
        else:
            tests["fixture_forbidden_on_physical_path"] = {"passed": False, "reason": "fixture-backed physical execution unexpectedly accepted"}
    system_mock_path = parent / "authority_system_mock.json"
    write_fixture(system_mock_path, base_physical_mock(source_root, output_root))
    physical_system_mock_env = {**target_authority_env(source_root, None, synthetic=False), PREFLIGHT_SYSTEM_MOCK_ENV: str(system_mock_path)}
    with temporary_env(physical_system_mock_env):
        try:
            execute_authorized(source_root, output_root)
        except TargetError as exc:
            tests["system_mock_forbidden_on_physical_path"] = {
                "passed": "refuses system-mock preflight" in str(exc),
                "reason": str(exc),
            }
        else:
            tests["system_mock_forbidden_on_physical_path"] = {"passed": False, "reason": "system-mock physical execution unexpectedly accepted"}
    return {
        "schema": "FAMILY10H_RELATION_LIFETIME_AUTHORITY_REFUSAL_TESTS_V1",
        "tests": tests,
        "passed": all(item["passed"] for item in tests.values()),
    }


def self_test(source_root: Path) -> dict[str, Any]:
    source_root = source_root.resolve()
    source = validate_source_root(source_root)
    env = no_live_authority_env()
    refusal = runtime_refusal_probe(source_root)
    parent = owned_output_parent(source_root)
    safe_remove_owned_output_parent(source_root)
    parent.mkdir()
    try:
        with temporary_env({AUTHORITY_ENV: None, SOURCE_AUTHORITY_ENV: None, FREEZE_COMMIT_ENV: None, MANIFEST_ENV: None, PREFLIGHT_FIXTURE_ENV: None, PREFLIGHT_SYSTEM_MOCK_ENV: None, DEPLOYMENT_CUSTODY_ENV: None, EXECUTION_MODE_ENV: None}):
            preflight_refusal = target_preflight_refusal(source_root, parent / "missing_authority_attempt")
        fixture_suite = run_preflight_fixture_suite(source_root, parent)
        helper_authority_suite = run_helper_authority_negative_suite(source_root, parent)
        physical_preflight_suite = run_physical_preflight_mock_suite(source_root, parent)
        authority_refusals = run_authority_refusal_tests(source_root, parent)
        synthetic_wrapper = run_synthetic_target_wrapper_test(source_root, parent)
    finally:
        safe_remove_owned_output_parent(source_root)
    result = {
        "schema": "FAMILY10H_RELATION_LIFETIME_TARGET_SELF_TEST_V2",
        "science_package_id": pub.SCIENCE_PACKAGE_ID,
        "transaction_run_id": pub.TRANSACTION_RUN_ID,
        "offline_only": True,
        "target_contact_count": 0,
        "sensor_inventory_count": 0,
        "live_invocation_count": 0,
        "pmu_acquisition_count": 0,
        "source_root_validation": source,
        "live_authority_env_absent": env,
        "runtime_refuses_without_authority": refusal,
        "target_preflight_refuses_without_authority": preflight_refusal,
        "target_preflight_fixture_suite": fixture_suite,
        "helper_authority_negative_suite": helper_authority_suite,
        "physical_preflight_mock_suite": physical_preflight_suite,
        "authority_refusal_tests": authority_refusals,
        "synthetic_target_wrapper_execution": synthetic_wrapper,
        "allowed_future_result_classes": pub.FUTURE_RESULT_CLASSES,
        "maximum_future_claim": pub.MAXIMUM_FUTURE_CLAIM,
        "small_wall_crossed": False,
    }
    result["self_test_passed"] = (
        source["passed"]
        and env["passed"]
        and refusal["passed"]
        and preflight_refusal["passed"]
        and fixture_suite["passed"]
        and helper_authority_suite["passed"]
        and physical_preflight_suite["passed"]
        and authority_refusals["passed"]
        and synthetic_wrapper["passed"]
    )
    result["self_test_sha256"] = pub.digest({k: v for k, v in result.items() if k != "self_test_sha256"})
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--execute-authorized", action="store_true")
    parser.add_argument("--target-preflight", action="store_true")
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent / pub.OWNED_OUTPUT_PARENT_NAME / "output")
    args = parser.parse_args(argv)
    if sum(1 for selected in [args.self_test, args.execute_authorized, args.target_preflight] if selected) != 1:
        parser.error("select exactly one mode")
    try:
        if args.self_test:
            result = self_test(args.source_root)
            print(strict_json_dumps(result, indent=2))
            return 0 if result["self_test_passed"] else 1
        if args.target_preflight:
            fixture = load_fixture()
            if os.environ.get(AUTHORITY_ENV) != AUTHORITY_VALUE:
                result = target_preflight_refusal(args.source_root.resolve(), args.output_root.resolve())
                print(strict_json_dumps(result, indent=2))
                return 2
            result = target_preflight(args.source_root.resolve(), args.output_root.resolve(), fixture)
            print(strict_json_dumps(result, indent=2))
            return 0 if result["passed"] else 1
        result = execute_authorized(args.source_root.resolve(), args.output_root.resolve())
        print(strict_json_dumps(result, indent=2))
        return 0 if result["passed"] else 1
    except TargetError as exc:
        result = {
            "schema": "FAMILY10H_RELATION_LIFETIME_TARGET_REFUSAL_V2",
            "status": "TARGET_EXECUTION_REFUSED",
            "reason": str(exc),
            "target_contact_count": 0,
            "sensor_inventory_count": 0,
            "live_invocation_count": 0,
            "pmu_acquisition_count": 0,
            "small_wall_crossed": False,
        }
        print(strict_json_dumps(result, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
