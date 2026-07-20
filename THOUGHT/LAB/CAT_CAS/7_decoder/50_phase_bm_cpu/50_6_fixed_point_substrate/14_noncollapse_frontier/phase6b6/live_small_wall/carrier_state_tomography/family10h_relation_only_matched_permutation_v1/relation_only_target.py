#!/usr/bin/env python3
"""Target-side authority gate and offline preflight for relation-only v1.

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
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import relation_only_public as pub


AUTHORITY_ENV = "FAMILY10H_RELATION_ONLY_LIVE_AUTHORITY"
SOURCE_AUTHORITY_ENV = "FAMILY10H_RELATION_ONLY_SOURCE_AUTHORITY_COMMIT"
FREEZE_COMMIT_ENV = "FAMILY10H_RELATION_ONLY_FREEZE_COMMIT"
MANIFEST_ENV = "FAMILY10H_RELATION_ONLY_MANIFEST_SHA256"
LEGACY_COMMIT_ENV = "FAMILY10H_RELATION_ONLY_COMMIT_BINDING"
RUNTIME_AUTHORITY_ENV = "FAMILY10H_RELATION_ONLY_RUNTIME_AUTHORITY"
PREFLIGHT_FIXTURE_ENV = "FAMILY10H_RELATION_ONLY_PREFLIGHT_FIXTURE"
EXECUTION_MODE_ENV = "FAMILY10H_RELATION_ONLY_EXECUTION_MODE"
EXECUTION_MODE_SYNTHETIC = "synthetic"
AUTHORITY_VALUE = pub.TRANSACTION_RUN_ID
RUNTIME_BINARY_NAMES = ["relation_only_runtime", "relation_only_runtime.exe"]

TARGET_PREFLIGHT_CHECKS = [
    "live_authority",
    "source_authority_commit_binding",
    "freeze_commit_binding",
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
    "operational_pinning_capability",
    "pmu_event_identities",
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
    completed = subprocess.run(
        ["wsl.exe", "wslpath", "-a", str(path)],
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def wsl_distro(source_root: Path) -> str | None:
    build_receipt = source_root / "RELATION_ONLY_RUNTIME_BUILD_SELF_TEST.json"
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


def source_authority() -> dict[str, Any]:
    return {
        "source_authority_commit": pub.SOURCE_AUTHORITY_COMMIT,
        "manifest_freeze_commit": pub.MANIFEST_FREEZE_COMMIT,
        "postrun_seal_commit": pub.POSTRUN_SEAL_COMMIT,
        "approved_sensor_identity_sha256": pub.APPROVED_SENSOR_IDENTITY_SHA256,
        "attempt_ceiling": pub.ATTEMPT_CEILING,
    }


def validate_source_root(source_root: Path) -> dict[str, Any]:
    source_root = source_root.resolve()
    failures: list[str] = []
    required = [
        "RELATION_ONLY_CONTRACT.md",
        "RELATION_GRAMMAR.json",
        "RELATION_GRAMMAR.tsv",
        "RELATION_GRAMMAR.sha256",
        "RELATION_ONLY_PUBLIC_SCHEDULE.json",
        "RELATION_ONLY_PUBLIC_SCHEDULE.tsv",
        "RELATION_ONLY_PUBLIC_SCHEDULE.sha256",
        "RELATION_ONLY_IMPLEMENTATION_MANIFEST.json",
        "RELATION_ONLY_IMPLEMENTATION_MANIFEST.sha256",
        "RELATION_ONLY_SOURCE_HASHES.json",
        "RELATION_ONLY_TOOLCHAIN_DISCOVERY.json",
        "RELATION_ONLY_RUNTIME_BUILD_SELF_TEST.json",
        "RELATION_ONLY_SYNTHETIC_EXECUTOR_SELF_TEST.json",
        "RELATION_ONLY_PHYSICAL_ADJUDICATOR_SELF_TEST.json",
        "RELATION_ONLY_PHYSICAL_THRESHOLD_CONTRACT.json",
        "relation_only_public.py",
        "relation_only_adjudication.py",
        "relation_only_physical_adjudication.py",
        "relation_only_runtime.c",
        "relation_only_runtime.h",
        "relation_only_target.py",
        "run_relation_only_matched_permutation.py",
    ]
    missing = [name for name in required if not (source_root / name).exists()]
    failures.extend(f"missing {name}" for name in missing)
    if missing:
        return {"passed": False, "failures": failures, "required_file_count": len(required)}

    grammar = read_json(source_root / "RELATION_GRAMMAR.json")
    schedule_manifest = read_json(source_root / "RELATION_ONLY_PUBLIC_SCHEDULE.json")
    schedule = pub.build_schedule(grammar)
    manifest = read_json(source_root / "RELATION_ONLY_IMPLEMENTATION_MANIFEST.json")
    source_hashes = read_json(source_root / "RELATION_ONLY_SOURCE_HASHES.json")
    threshold = read_json(source_root / "RELATION_ONLY_PHYSICAL_THRESHOLD_CONTRACT.json")
    readiness = read_json(source_root / "RELATION_ONLY_BUILD_READINESS.json") if (source_root / "RELATION_ONLY_BUILD_READINESS.json").exists() else {}

    grammar_file_sha = pub.sha256_file(source_root / "RELATION_GRAMMAR.json")
    if grammar_file_sha != (source_root / "RELATION_GRAMMAR.sha256").read_text(encoding="utf-8").strip():
        failures.append("grammar sha sidecar mismatch")
    if grammar.get("grammar_sha256") != digest_without(grammar, "grammar_sha256"):
        failures.append("canonical grammar digest mismatch")

    schedule_tsv_sha = pub.sha256_file(source_root / "RELATION_ONLY_PUBLIC_SCHEDULE.tsv")
    if schedule_tsv_sha != (source_root / "RELATION_ONLY_PUBLIC_SCHEDULE.sha256").read_text(encoding="utf-8").strip():
        failures.append("expanded schedule TSV sha sidecar mismatch")
    if schedule_manifest.get("schema") != "FAMILY10H_RELATION_ONLY_PUBLIC_SCHEDULE_MANIFEST_V3":
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

    manifest_file_sha = pub.sha256_file(source_root / "RELATION_ONLY_IMPLEMENTATION_MANIFEST.json")
    if manifest_file_sha != (source_root / "RELATION_ONLY_IMPLEMENTATION_MANIFEST.sha256").read_text(encoding="utf-8").strip():
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
    if authority != source_authority():
        failures.append("manifest authority binding mismatch")
    threshold_source = threshold.get("source_evidence", {})
    if threshold_source.get("source_authority_commit") != pub.SOURCE_AUTHORITY_COMMIT:
        failures.append("threshold source authority mismatch")
    if threshold_source.get("manifest_freeze_commit") != pub.MANIFEST_FREEZE_COMMIT:
        failures.append("threshold manifest freeze mismatch")

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
    if manifest.get("claim_boundary", {}).get("small_wall_crossed") is not False:
        failures.append("manifest small wall boundary mismatch")

    return {"passed": not failures, "failures": failures, "required_file_count": len(required)}


def artifact_identity(source_root: Path) -> dict[str, Any]:
    source_root = source_root.resolve()
    grammar = read_json(source_root / "RELATION_GRAMMAR.json")
    schedule_manifest = read_json(source_root / "RELATION_ONLY_PUBLIC_SCHEDULE.json")
    manifest = read_json(source_root / "RELATION_ONLY_IMPLEMENTATION_MANIFEST.json")
    source_hashes = read_json(source_root / "RELATION_ONLY_SOURCE_HASHES.json")
    runtime = runtime_binary_path(source_root)
    runtime_sha = pub.sha256_file(runtime) if runtime else None
    file_probe = run_file_probe(source_root, runtime) if runtime else {"returncode": 2, "stdout": "", "stderr": "runtime missing"}
    return {
        "grammar_file_sha256": pub.sha256_file(source_root / "RELATION_GRAMMAR.json"),
        "grammar_canonical_sha256": grammar.get("grammar_sha256"),
        "schedule_tsv_sha256": pub.sha256_file(source_root / "RELATION_ONLY_PUBLIC_SCHEDULE.tsv"),
        "schedule_canonical_sha256": schedule_manifest.get("canonical_schedule_sha256"),
        "schedule_manifest_sha256": schedule_manifest.get("schedule_manifest_sha256"),
        "manifest_file_sha256": pub.sha256_file(source_root / "RELATION_ONLY_IMPLEMENTATION_MANIFEST.json"),
        "manifest_canonical_sha256": manifest.get("manifest_sha256"),
        "runtime_binary_sha256": runtime_sha,
        "runtime_binary_format": file_probe.get("stdout", "").strip(),
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
        ["--execute-schedule", source_root / "RELATION_ONLY_PUBLIC_SCHEDULE.tsv", source_root / "_no_live_output"],
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
    return {
        "schema": "FAMILY10H_RELATION_ONLY_PREFLIGHT_FIXTURE_V1",
        "cpu": {
            "vendor": "AuthenticAMD",
            "family": 16,
            "model": 6,
            "source_cpu_present": True,
            "receiver_cpu_present": True,
            "source_cpu": 4,
            "receiver_cpu": 5,
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
            "source_authority_commit": pub.SOURCE_AUTHORITY_COMMIT,
            "manifest_freeze_commit": pub.MANIFEST_FREEZE_COMMIT,
        },
        "runtime_binary_format_contains": ["ELF 64-bit", "x86-64"],
        "pmu": {
            "events": pub.PMU_GROUP["events"],
            "grouped_open_capability": True,
        },
        "sensor": {
            "identity_sha256": pub.APPROVED_SENSOR_IDENTITY_SHA256,
            "hwmon_name": "k10temp",
            "sensor_label": "Tctl",
            "resolved_device_identity": "fixture-family10h-k10temp",
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
    manifest_file_sha = pub.sha256_file(source_root / "RELATION_ONLY_IMPLEMENTATION_MANIFEST.json")
    return {
        "live_authority": check("live_authority", os.environ.get(AUTHORITY_ENV) == AUTHORITY_VALUE, os.environ.get(AUTHORITY_ENV)),
        "source_authority_commit_binding": check(
            "source_authority_commit_binding",
            os.environ.get(SOURCE_AUTHORITY_ENV) == manifest.get("authority_binding", {}).get("source_authority_commit"),
            {
                "observed": os.environ.get(SOURCE_AUTHORITY_ENV),
                "expected": manifest.get("authority_binding", {}).get("source_authority_commit"),
            },
        ),
        "freeze_commit_binding": check(
            "freeze_commit_binding",
            os.environ.get(FREEZE_COMMIT_ENV) == manifest.get("authority_binding", {}).get("manifest_freeze_commit"),
            {
                "observed": os.environ.get(FREEZE_COMMIT_ENV),
                "expected": manifest.get("authority_binding", {}).get("manifest_freeze_commit"),
            },
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
    manifest = read_json(source_root / "RELATION_ONLY_IMPLEMENTATION_MANIFEST.json") if (source_root / "RELATION_ONLY_IMPLEMENTATION_MANIFEST.json").exists() else {}
    schedule_manifest = read_json(source_root / "RELATION_ONLY_PUBLIC_SCHEDULE.json") if (source_root / "RELATION_ONLY_PUBLIC_SCHEDULE.json").exists() else {}
    source_hashes = read_json(source_root / "RELATION_ONLY_SOURCE_HASHES.json") if (source_root / "RELATION_ONLY_SOURCE_HASHES.json").exists() else {}
    artifacts = artifact_identity(source_root) if source_validation["passed"] else {}
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

    if fixture is None:
        checks["fixture_backend_present"] = check("fixture_backend_present", False, "preflight fixture backend missing")
    else:
        expected = fixture.get("expected_artifacts", {})
        checks["manifest_file_hash"].update(
            check(
                "manifest_file_hash",
                checks["manifest_file_hash"]["passed"]
                and expected.get("manifest_file_sha256") == artifacts.get("manifest_file_sha256"),
                {"fixture": expected.get("manifest_file_sha256"), "env": os.environ.get(MANIFEST_ENV)},
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
        expected_format = fixture.get("runtime_binary_format_contains", [])
        checks["runtime_binary_format_and_abi"] = check(
            "runtime_binary_format_and_abi",
            all(item in format_text for item in expected_format),
            {"format": format_text, "required": expected_format},
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
                and expected.get("source_authority_commit") == pub.SOURCE_AUTHORITY_COMMIT,
                {"fixture": expected.get("source_authority_commit"), "env": os.environ.get(SOURCE_AUTHORITY_ENV)},
            )
        )
        checks["freeze_commit_binding"].update(
            check(
                "freeze_commit_binding",
                checks["freeze_commit_binding"]["passed"]
                and expected.get("manifest_freeze_commit") == pub.MANIFEST_FREEZE_COMMIT,
                {"fixture": expected.get("manifest_freeze_commit"), "env": os.environ.get(FREEZE_COMMIT_ENV)},
            )
        )
        cpu = fixture.get("cpu", {})
        checks["cpu_vendor_family_model"] = check(
            "cpu_vendor_family_model",
            cpu.get("vendor") == "AuthenticAMD" and cpu.get("family") == 16 and isinstance(cpu.get("model"), int),
            cpu,
        )
        checks["source_receiver_cpu_identity"] = check(
            "source_receiver_cpu_identity",
            cpu.get("source_cpu_present") is True and cpu.get("receiver_cpu_present") is True and cpu.get("source_cpu") != cpu.get("receiver_cpu"),
            cpu,
        )
        checks["operational_pinning_capability"] = check("operational_pinning_capability", cpu.get("operational_pinning") is True, cpu)
        pmu = fixture.get("pmu", {})
        checks["pmu_event_identities"] = check("pmu_event_identities", pmu.get("events") == pub.PMU_GROUP["events"], pmu.get("events"))
        checks["grouped_pmu_open_capability"] = check("grouped_pmu_open_capability", pmu.get("grouped_open_capability") is True, pmu)
        sensor = fixture.get("sensor", {})
        checks["temperature_sensor_authority"] = check(
            "temperature_sensor_authority",
            sensor.get("identity_sha256") == pub.APPROVED_SENSOR_IDENTITY_SHA256
            and bool(sensor.get("hwmon_name"))
            and bool(sensor.get("sensor_label"))
            and sensor.get("identity_stability") is True,
            sensor,
        )
        policy = fixture.get("frequency_policy", {})
        checks["cpu_frequency_policy_custody"] = check(
            "cpu_frequency_policy_custody",
            policy.get("policy_locked") is True and policy.get("identity_stability") is True,
            policy,
        )
        geometry = fixture.get("physical_geometry", {})
        checks["actual_physical_geometry_status"] = check(
            "actual_physical_geometry_status",
            geometry.get("status") in {"verified", "unresolved_contractually_lowered"} and geometry.get("claim_ceiling_lowered") is True,
            geometry,
        )
        output_law = output_root_law(source_root, output_root, fixture)
        checks["output_root_absence"] = check("output_root_absence", output_law["passed"], output_law)
        attempt = fixture.get("attempt", {})
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
        "schema": "FAMILY10H_RELATION_ONLY_TARGET_PREFLIGHT_V2",
        "status": "TARGET_PREFLIGHT_PASSED" if not failures else "TARGET_PREFLIGHT_FAILED",
        "passed": not failures,
        "failures": failures,
        "checks": checks,
        "artifact_identity": artifacts,
        "output_root": str(output_root),
        "target_contact_count": 0,
        "sensor_inventory_count": 0,
        "pmu_open_count": 0,
        "pmu_acquisition_count": 0,
        "live_invocation_count": 0,
        "small_wall_crossed": False,
    }


def target_preflight_refusal(source_root: Path, output_root: Path) -> dict[str, Any]:
    result = target_preflight(source_root, output_root, None)
    result["schema"] = "FAMILY10H_RELATION_ONLY_TARGET_PREFLIGHT_REFUSAL_V2"
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


def target_authority_env(source_root: Path, fixture_path: Path | None, *, synthetic: bool = False) -> dict[str, str | None]:
    manifest_sha = pub.sha256_file(source_root / "RELATION_ONLY_IMPLEMENTATION_MANIFEST.json")
    return {
        AUTHORITY_ENV: AUTHORITY_VALUE,
        SOURCE_AUTHORITY_ENV: pub.SOURCE_AUTHORITY_COMMIT,
        FREEZE_COMMIT_ENV: pub.MANIFEST_FREEZE_COMMIT,
        MANIFEST_ENV: manifest_sha,
        PREFLIGHT_FIXTURE_ENV: str(fixture_path) if fixture_path else None,
        EXECUTION_MODE_ENV: EXECUTION_MODE_SYNTHETIC if synthetic else None,
        LEGACY_COMMIT_ENV: None,
    }


def execute_authorized(source_root: Path, output_root: Path) -> dict[str, Any]:
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    fixture = load_fixture()
    preflight = target_preflight(source_root, output_root, fixture)
    if not preflight["passed"]:
        raise TargetError("target preflight failed: " + ",".join(preflight["failures"]))
    runtime = runtime_binary_path(source_root)
    if runtime is None:
        raise TargetError("runtime binary missing")
    synthetic = os.environ.get(EXECUTION_MODE_ENV) == EXECUTION_MODE_SYNTHETIC
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
        [runtime_mode, source_root / "RELATION_ONLY_PUBLIC_SCHEDULE.tsv", output_root],
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
        and raw_count == 32256
        and death_count == 32256
        and feature is not None
        and runtime_receipt is not None
        and feature.get("physical_measurement") is (not synthetic)
        and runtime_receipt.get("physical_measurement") is (not synthetic)
    )
    return {
        "schema": "FAMILY10H_RELATION_ONLY_SYNTHETIC_TARGET_WRAPPER_EXECUTION_V1"
        if synthetic
        else "FAMILY10H_RELATION_ONLY_TARGET_WRAPPER_EXECUTION_V1",
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
        "missing_source_cpu": ("source_receiver_cpu_identity", {**base, "cpu": {**base["cpu"], "source_cpu_present": False}}),
        "missing_receiver_cpu": ("source_receiver_cpu_identity", {**base, "cpu": {**base["cpu"], "receiver_cpu_present": False}}),
        "failed_operational_pinning": ("operational_pinning_capability", {**base, "cpu": {**base["cpu"], "operational_pinning": False}}),
        "wrong_binary_hash": (
            "runtime_binary_hash",
            {**base, "expected_artifacts": {**base["expected_artifacts"], "runtime_binary_sha256": "0" * 64}},
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
            {**base, "expected_artifacts": {**base["expected_artifacts"], "source_authority_commit": "0" * 40}},
        ),
        "unavailable_pmu_event": ("pmu_event_identities", {**base, "pmu": {**base["pmu"], "events": {}}}),
        "grouped_pmu_failure": ("grouped_pmu_open_capability", {**base, "pmu": {**base["pmu"], "grouped_open_capability": False}}),
        "sensor_identity_mismatch": (
            "temperature_sensor_authority",
            {**base, "sensor": {**base["sensor"], "identity_sha256": "0" * 64}},
        ),
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
        "schema": "FAMILY10H_RELATION_ONLY_TARGET_PREFLIGHT_FIXTURE_SUITE_V1",
        "results": results,
        "passed": all(item["passed"] for item in results.values()),
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
                "schema": "FAMILY10H_RELATION_ONLY_SYNTHETIC_TARGET_WRAPPER_EXECUTION_V1",
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
    return {
        "schema": "FAMILY10H_RELATION_ONLY_AUTHORITY_REFUSAL_TESTS_V1",
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
        with temporary_env({AUTHORITY_ENV: None, SOURCE_AUTHORITY_ENV: None, FREEZE_COMMIT_ENV: None, MANIFEST_ENV: None, PREFLIGHT_FIXTURE_ENV: None, EXECUTION_MODE_ENV: None}):
            preflight_refusal = target_preflight_refusal(source_root, parent / "missing_authority_attempt")
        fixture_suite = run_preflight_fixture_suite(source_root, parent)
        authority_refusals = run_authority_refusal_tests(source_root, parent)
        synthetic_wrapper = run_synthetic_target_wrapper_test(source_root, parent)
    finally:
        safe_remove_owned_output_parent(source_root)
    result = {
        "schema": "FAMILY10H_RELATION_ONLY_TARGET_SELF_TEST_V2",
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
            "schema": "FAMILY10H_RELATION_ONLY_TARGET_REFUSAL_V2",
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
