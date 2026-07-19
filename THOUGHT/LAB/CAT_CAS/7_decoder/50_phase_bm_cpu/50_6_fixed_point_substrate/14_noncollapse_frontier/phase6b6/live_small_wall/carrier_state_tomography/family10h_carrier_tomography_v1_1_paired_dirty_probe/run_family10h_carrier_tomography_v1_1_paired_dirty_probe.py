#!/usr/bin/env python3
"""Controller for the v1.1 paired dirty-probe q-readout confirmation."""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import tarfile
import time
from pathlib import Path
from typing import Any

import family10h_carrier_tomography_public as public
import family10h_carrier_tomography_target as target


HERE = Path(__file__).resolve().parent
CARRIER_ROOT = HERE.parent
RUN_ROOT = CARRIER_ROOT / "runs" / public.TRANSACTION_RUN_ID
ATTEMPT_DIR = RUN_ROOT / "attempt_1"
TARGET_HOST = "root@192.168.137.100"
REMOTE_BASE = "/root/catcas_live_small_wall"
REMOTE_ROOT = public.EXPECTED_REMOTE_ROOT
REMOTE_ARCHIVE = f"{REMOTE_BASE}/{public.TRANSACTION_RUN_ID}_attempt1_remote_root.tar.gz"
REMOTE_PACKAGE = f"{REMOTE_BASE}/{public.TRANSACTION_RUN_ID}_source_package.tar.gz"
LOCAL_TMP_PACKAGE = Path("C:/tmp") / f"{public.TRANSACTION_RUN_ID}_source_package.tar.gz"
LOCAL_TMP_ARCHIVE = Path("C:/tmp") / f"{public.TRANSACTION_RUN_ID}_remote_root.tar.gz"
LOCAL_ARCHIVE = ATTEMPT_DIR / "ATTEMPT_1_REMOTE_ROOT.tar.gz"
OWNER_MARKER = f".{public.TRANSACTION_RUN_ID}_owner"
MANIFEST = HERE / "CARRIER_TOMOGRAPHY_IMPLEMENTATION_MANIFEST.json"
MANIFEST_SIDECAR = HERE / "CARRIER_TOMOGRAPHY_IMPLEMENTATION_MANIFEST.sha256"
SOURCE_HASHES = HERE / "CARRIER_TOMOGRAPHY_SOURCE_HASHES.json"
SOURCE_BUNDLE = HERE / "CARRIER_TOMOGRAPHY_SOURCE_BUNDLE.tar.gz"
OFFLINE_VALIDATE = HERE / "CARRIER_TOMOGRAPHY_OFFLINE_VALIDATE.json"
SOFTWARE_FIXTURE_ADJUDICATION = HERE / "PAIRED_DIRTY_PROBE_V1_0_FIXTURE_ADJUDICATION.json"
SOFTWARE_FIXTURE_SELF_TEST = HERE / "PAIRED_DIRTY_PROBE_V1_0_FIXTURE_SELF_TEST.json"
APPROVED_SENSOR_AUTHORITY = (
    CARRIER_ROOT
    / "family10h_carrier_tomography_v1"
    / "CARRIER_TOMOGRAPHY_TEMPERATURE_SENSOR_AUTHORITY.json"
)
V1_0_ATTEMPT3_DIR = CARRIER_ROOT / "runs" / "family10h_carrier_tomography_v1_0" / "attempt_3"
LIVE_NONCE = "84a4eb1ff15f89a51f6dfe0bfb8c8df61c53bcce3a7b6108475af0b3489ec1c2"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_path(path: Path) -> str:
    return public.sha256_file(path)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run_local(command: list[str], *, cwd: Path | None = None, timeout: int = 120) -> dict[str, Any]:
    started = time.monotonic()
    completed = subprocess.run(command, cwd=str(cwd) if cwd else None, text=True, capture_output=True, timeout=timeout, check=False)
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }


def run_remote(script: str, *, timeout: int = 120) -> dict[str, Any]:
    return run_local(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", TARGET_HOST, script], timeout=timeout)


def run_scp(source: str, dest: str, *, timeout: int = 120) -> dict[str, Any]:
    return run_local(["scp", "-O", "-o", "BatchMode=yes", source, dest], timeout=timeout)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def git_value(args: list[str]) -> str:
    completed = run_local(["git", *args], cwd=HERE, timeout=30)
    require(completed["returncode"] == 0, f"git {' '.join(args)} failed: {completed['stderr']}")
    return completed["stdout"].strip()


def repo_root() -> Path:
    return Path(git_value(["rev-parse", "--show-toplevel"]))


def status_short() -> str:
    return run_local(["git", "status", "--short"], cwd=HERE, timeout=30)["stdout"]


def source_file_hashes() -> dict[str, Any]:
    source_files = {}
    for name in target.SOURCE_FILE_NAMES:
        path = HERE / name
        require(path.exists(), f"source file missing {name}")
        source_files[name] = {"sha256": sha256_path(path), "size": path.stat().st_size}
    receipt = {
        "schema": "FAMILY10H_PAIRED_DIRTY_PROBE_V1_1_SOURCE_HASHES_V1",
        "science_package_id": public.SCIENCE_PACKAGE_ID,
        "transaction_run_id": public.TRANSACTION_RUN_ID,
        "source_files": source_files,
        "runtime_binary_authority": target.runtime_binary_authority(HERE),
    }
    receipt["source_hashes_sha256"] = public.digest({k: v for k, v in receipt.items() if k != "source_hashes_sha256"})
    return receipt


def write_source_bundle() -> dict[str, Any]:
    with SOURCE_BUNDLE.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w") as archive:
                for name in sorted(target.SOURCE_FILE_NAMES):
                    path = HERE / name
                    info = archive.gettarinfo(str(path), arcname=name)
                    info.mtime = 0
                    info.uid = 0
                    info.gid = 0
                    info.mode = 0o644
                    info.uname = ""
                    info.gname = ""
                    with path.open("rb") as handle:
                        archive.addfile(info, handle)
    return {
        "path": str(SOURCE_BUNDLE),
        "sha256": sha256_path(SOURCE_BUNDLE),
        "size": SOURCE_BUNDLE.stat().st_size,
        "file_count": len(target.SOURCE_FILE_NAMES),
        "files": sorted(target.SOURCE_FILE_NAMES),
    }


def approved_temperature_identity() -> dict[str, Any]:
    authority = read_json(APPROVED_SENSOR_AUTHORITY)
    identity = authority.get("approved_sensor_identity")
    require(isinstance(identity, dict), "approved temperature identity missing")
    require(identity.get("identity_sha256") == public.temperature_identity_digest(identity), "approved temperature identity digest mismatch")
    return identity


def import_adjudicator() -> Any:
    path = HERE / "paired_dirty_probe_prospective_adjudication.py"
    spec = importlib.util.spec_from_file_location("paired_dirty_probe_prospective_adjudication", path)
    require(spec is not None and spec.loader is not None, "cannot load v1.1 adjudicator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_fixture_adjudication() -> dict[str, Any]:
    adjudicator = import_adjudicator()
    module_public = adjudicator.load_public_module()
    schedule = module_public.load_schedule_from_artifacts()
    packet = module_public.minimal_success_packet(schedule)
    for record in packet["raw_records"]:
        if record["matrix_block"] == "persistence_matrix" and record["query"] == "query_A":
            record["dirty_probe_response"] = 10000 + int(record["q"])
        elif record["matrix_block"] == "persistence_matrix" and record["query"] == "query_B":
            record["dirty_probe_response"] = 10000 - int(record["q"])
        else:
            record["dirty_probe_response"] = 10000
    provenance = {
        "source": "synthetic_v1_1_software_fixture",
        "archive_negative_fixture": str(V1_0_ATTEMPT3_DIR),
        "archive_negative_fixture_run_id": "family10h_carrier_tomography_v1_0",
    }
    validation = module_public.validate_evidence_packet(packet, schedule)
    require(validation["passed"], "v1.1 synthetic fixture packet validation failed")
    result = adjudicator.adjudicate_packet(packet, schedule)
    result["validation"] = validation
    result["source_evidence"] = provenance
    self_test = adjudicator.run_self_tests(
        module_public,
        packet,
        schedule,
        V1_0_ATTEMPT3_DIR,
        archive_run_id="family10h_carrier_tomography_v1_0",
    )
    write_json(SOFTWARE_FIXTURE_ADJUDICATION, result)
    write_json(SOFTWARE_FIXTURE_SELF_TEST, self_test)
    return {
        "fixture_adjudication_path": str(SOFTWARE_FIXTURE_ADJUDICATION),
        "fixture_adjudication_sha256": sha256_path(SOFTWARE_FIXTURE_ADJUDICATION),
        "fixture_self_test_path": str(SOFTWARE_FIXTURE_SELF_TEST),
        "fixture_self_test_sha256": sha256_path(SOFTWARE_FIXTURE_SELF_TEST),
        "fixture_result_class": result.get("result_class"),
        "fixture_claim": result.get("scientific_claim"),
        "fixture_self_test_passed": self_test.get("passed") is True,
    }


def prepare_source_authority() -> dict[str, Any]:
    source_hashes = source_file_hashes()
    write_json(SOURCE_HASHES, source_hashes)
    source_bundle = write_source_bundle()
    fixture = run_fixture_adjudication()
    schedule = public.load_schedule_from_artifacts()
    schedule_validation = public.validate_schedule(schedule)
    source_authority = target.validate_source_file_authority(HERE)
    runtime_authority = target.validate_runtime_binary_authority(HERE, expected=source_hashes["runtime_binary_authority"])
    validation = {
        "schema": "FAMILY10H_PAIRED_DIRTY_PROBE_V1_1_OFFLINE_VALIDATE_V1",
        "science_package_id": public.SCIENCE_PACKAGE_ID,
        "transaction_run_id": public.TRANSACTION_RUN_ID,
        "target_contact_count": 0,
        "sensor_inventory_count": 0,
        "live_invocation_count": 0,
        "pmu_acquisition_count": 0,
        "schedule_validation": schedule_validation,
        "source_authority": source_authority,
        "runtime_authority": runtime_authority,
        "software_fixture": fixture,
        "threshold_source": "PROSPECTIVE_PAIRED_DIRTY_PROBE_CONFIRMATION_CONTRACT.md copied as CARRIER_TOMOGRAPHY_CONTRACT.md",
    }
    validation["passed"] = (
        schedule_validation.get("passed") is True
        and source_authority.get("passed") is True
        and runtime_authority.get("passed") is True
        and fixture.get("fixture_self_test_passed") is True
    )
    validation["offline_validate_sha256"] = public.digest({k: v for k, v in validation.items() if k != "offline_validate_sha256"})
    write_json(OFFLINE_VALIDATE, validation)
    return {
        "source_hashes": str(SOURCE_HASHES),
        "source_hashes_sha256": sha256_path(SOURCE_HASHES),
        "source_bundle": source_bundle,
        "offline_validate": str(OFFLINE_VALIDATE),
        "offline_validate_sha256": sha256_path(OFFLINE_VALIDATE),
        "passed": validation["passed"],
    }


def build_live_challenge(source_authority_commit: str) -> dict[str, Any]:
    source_hashes = read_json(SOURCE_HASHES)
    source_bundle_sha = sha256_path(SOURCE_BUNDLE)
    schedule_hashes = read_json(HERE / "CARRIER_TOMOGRAPHY_PUBLIC_SCHEDULE.sha256")
    return {
        "schema": "FAMILY10H_PAIRED_DIRTY_PROBE_V1_1_LIVE_AUTHORITY_CHALLENGE_V1",
        "science_package_id": public.SCIENCE_PACKAGE_ID,
        "transaction_run_id": public.TRANSACTION_RUN_ID,
        "source_authority_commit": source_authority_commit,
        "source_hashes_sha256": source_hashes["source_hashes_sha256"],
        "source_bundle_sha256": source_bundle_sha,
        "runtime_binary_sha256": source_hashes["runtime_binary_authority"]["sha256"],
        "schedule_canonical_sha256": schedule_hashes["canonical_sha256"],
        "schedule_json_sha256": schedule_hashes["json_sha256"],
        "schedule_tsv_sha256": schedule_hashes["tsv_sha256"],
        "controller_nonce_sha256": sha256_bytes(LIVE_NONCE.encode("ascii")),
        "primary_endpoint": "dirty_probe_response",
        "primary_contrast": "D_single = dirty_probe_response(query_A) - dirty_probe_response(query_B)",
        "maximum_claim": "PUBLIC_POST_SOURCE_SCALAR_CARRIER_Q_READOUT_CONFIRMED",
    }


def prepare_freeze_manifest(source_authority_commit: str) -> dict[str, Any]:
    require(re.fullmatch(r"[0-9a-f]{40}", source_authority_commit) is not None, "source authority commit malformed")
    source_hashes = read_json(SOURCE_HASHES)
    source_bundle = {
        "path": str(SOURCE_BUNDLE),
        "sha256": sha256_path(SOURCE_BUNDLE),
        "size": SOURCE_BUNDLE.stat().st_size,
        "file_count": len(target.SOURCE_FILE_NAMES),
        "files": sorted(target.SOURCE_FILE_NAMES),
    }
    offline_validate = read_json(OFFLINE_VALIDATE)
    challenge = build_live_challenge(source_authority_commit)
    manifest = {
        "schema": "FAMILY10H_PAIRED_DIRTY_PROBE_V1_1_IMPLEMENTATION_MANIFEST_V1",
        "science_package_id": public.SCIENCE_PACKAGE_ID,
        "transaction_run_id": public.TRANSACTION_RUN_ID,
        "package_decision": public.PACKAGE_DECISION_FROZEN,
        "claim_ceiling": "PUBLIC_POST_SOURCE_SCALAR_CARRIER_Q_READOUT_CONFIRMED",
        "forbidden_claims": [
            "SMALL_WALL_CROSSED",
            "CATALYTIC_BORROWING_ESTABLISHED",
            "PHYSICAL_RELATIONAL_MEMORY_ESTABLISHED",
            "RELATIONAL_CARRIER_ESTABLISHED",
            "FULL_CARRIER_STATE_TOMOGRAPHY_ESTABLISHED",
        ],
        "git_state_at_manifest_build": {
            "branch": git_value(["rev-parse", "--abbrev-ref", "HEAD"]),
            "head": source_authority_commit,
            "working_tree_status_at_source_freeze": "clean before freeze-manifest generation",
        },
        "contact_counter_attestation": {
            "target_contact_count": 0,
            "sensor_inventory_count": 0,
            "live_invocation_count": 0,
            "pmu_acquisition_count": 0,
        },
        "source_hashes": source_hashes,
        "source_bundle": source_bundle,
        "runtime_binary_authority": source_hashes["runtime_binary_authority"],
        "offline_validate": {
            "passed": offline_validate.get("passed") is True,
            "sha256": sha256_path(OFFLINE_VALIDATE),
            "path": str(OFFLINE_VALIDATE),
        },
        "temperature_sensor_authority": {
            "approved_sensor_identity": approved_temperature_identity(),
            "controller_challenge": challenge,
            "authority_source": "committed v1.0 C6 sensor-authority receipt reused as preapproved sensor identity",
        },
        "primary_law": {
            "endpoint": "dirty_probe_response",
            "contrast": "D_single = dirty_probe_response(query_A) - dirty_probe_response(query_B)",
            "matched_key": ["session", "replicate", "delay_label", "mapping", "source_order", "q", "source_off_control"],
            "mapping_policy": "logical_query_identity_no_map_sign_inversion",
            "secondary_channels": ["change_to_dirty", "cpu_cycles", "duration_ns"],
            "secondary_channel_substitution_allowed": False,
        },
        "thresholds": {
            "source_off_abs_max": 128,
            "source_off_mean_abs": 32,
            "q0_abs_mean_max": 64,
            "q1536_abs_mean_min": 2000,
            "model_min_r2": 0.98,
            "heldout_max_relative_rmse": 0.10,
            "slope_max_relative_disagreement": 0.10,
            "intercept_abs_max": 64,
            "odd_symmetry_max_relative_error": 0.10,
            "signal_to_null_min_ratio": 20,
            "classifier_min_exact_accuracy": 0.95,
            "classifier_min_sign_accuracy": 1.0,
        },
    }
    manifest["manifest_canonical_sha256"] = public.digest({k: v for k, v in manifest.items() if k != "manifest_canonical_sha256"})
    write_json(MANIFEST, manifest)
    sidecar = {
        "schema": "FAMILY10H_PAIRED_DIRTY_PROBE_V1_1_MANIFEST_SHA256_V1",
        "manifest_canonical_sha256": manifest["manifest_canonical_sha256"],
        "manifest_file_sha256": sha256_path(MANIFEST),
    }
    write_json(MANIFEST_SIDECAR, sidecar)
    return validate_package()


def validate_package() -> dict[str, Any]:
    failures: list[str] = []
    schedule = public.validate_schedule(public.load_schedule_from_artifacts())
    if schedule.get("passed") is not True:
        failures.append("schedule validation failed")
    source_authority = target.validate_source_file_authority(HERE)
    if source_authority.get("passed") is not True:
        failures.append("source authority failed: " + ",".join(source_authority.get("failures", [])))
    manifest_authority = target.validate_manifest_authority(HERE) if MANIFEST.exists() and MANIFEST_SIDECAR.exists() else {"passed": False, "failures": ["manifest missing"]}
    if MANIFEST.exists() and MANIFEST_SIDECAR.exists() and manifest_authority.get("passed") is not True:
        failures.append("manifest authority failed: " + ",".join(manifest_authority.get("failures", [])))
    offline = read_json(OFFLINE_VALIDATE) if OFFLINE_VALIDATE.exists() else {"passed": False}
    if offline.get("passed") is not True:
        failures.append("offline validation receipt failed")
    return {
        "schema": "FAMILY10H_PAIRED_DIRTY_PROBE_V1_1_VALIDATE_RESULT_V1",
        "passed": not failures,
        "failures": failures,
        "schedule": schedule,
        "source_authority": source_authority,
        "manifest_authority": manifest_authority,
        "offline_validate": offline,
    }


def package_tarball() -> dict[str, Any]:
    if LOCAL_TMP_PACKAGE.exists():
        LOCAL_TMP_PACKAGE.unlink()
    include_files = sorted(path for path in HERE.iterdir() if path.is_file() and path.name != "__pycache__")
    with LOCAL_TMP_PACKAGE.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w") as archive:
                for path in include_files:
                    info = archive.gettarinfo(str(path), arcname=f"{public.TRANSACTION_RUN_ID}/{path.name}")
                    info.mtime = 0
                    info.uid = 0
                    info.gid = 0
                    info.mode = 0o755 if path.name == target.RUNTIME_BINARY_NAME else 0o644
                    info.uname = ""
                    info.gname = ""
                    with path.open("rb") as handle:
                        archive.addfile(info, handle)
    return {
        "path": str(LOCAL_TMP_PACKAGE),
        "sha256": sha256_path(LOCAL_TMP_PACKAGE),
        "size": LOCAL_TMP_PACKAGE.stat().st_size,
        "file_count": len(include_files),
        "files": [path.name for path in include_files],
    }


def normalize_member(name: str) -> str:
    normalized = name.replace("\\", "/").lstrip("/")
    prefix = f"{public.TRANSACTION_RUN_ID}/"
    return normalized[len(prefix):] if normalized.startswith(prefix) else normalized


def inventory_archive() -> dict[str, Any]:
    files = []
    required = {
        "output/raw_records.jsonl",
        "output/source_death_receipts.jsonl",
        "output/feature_freeze.json",
        "output_target_execution_receipt.json",
    }
    matches = {key: [] for key in required}
    with tarfile.open(LOCAL_ARCHIVE, "r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            extracted = archive.extractfile(member)
            require(extracted is not None, f"cannot extract {member.name}")
            data = extracted.read()
            relative = normalize_member(member.name)
            item = {"relative_path": relative, "archive_member": member.name, "sha256": sha256_bytes(data), "size": len(data)}
            files.append(item)
            if relative in matches:
                matches[relative].append(item)
    failures = []
    for relative, items in matches.items():
        if len(items) != 1:
            failures.append(f"required member match count {relative} = {len(items)}")
    archive_record = {
        "relative_path": LOCAL_ARCHIVE.name,
        "sha256": sha256_path(LOCAL_ARCHIVE),
        "size": LOCAL_ARCHIVE.stat().st_size,
    }
    inventory = {
        "schema": "FAMILY10H_PAIRED_DIRTY_PROBE_V1_1_EVIDENCE_INVENTORY_V1",
        "attempt": 1,
        "transaction_run_id": public.TRANSACTION_RUN_ID,
        "remote_root": REMOTE_ROOT,
        "remote_root_archive": archive_record,
        "files": [archive_record, *files],
        "required_member_matches": {relative: items for relative, items in matches.items()},
        "failures": failures,
        "passed": not failures,
    }
    write_json(ATTEMPT_DIR / "ATTEMPT_1_EVIDENCE_INVENTORY.json", inventory)
    return inventory


def copyback_receipt(started_at: str, make_archive: dict[str, Any], scp_archive: dict[str, Any], inventory: dict[str, Any]) -> dict[str, Any]:
    snapshot_files = [
        {"relative_path": item["relative_path"], "sha256": item["sha256"], "size": item["size"]}
        for item in inventory["files"]
        if item["relative_path"] != LOCAL_ARCHIVE.name
    ]
    receipt = {
        "schema": "FAMILY10H_PAIRED_DIRTY_PROBE_V1_1_ATTEMPT1_COPYBACK_RECEIPT_V1",
        "attempt": 1,
        "started_at": started_at,
        "ended_at": utc_now(),
        "target_host": TARGET_HOST,
        "remote_root": REMOTE_ROOT,
        "remote_archive": REMOTE_ARCHIVE,
        "local_tmp_archive": str(LOCAL_TMP_ARCHIVE),
        "local_archive": str(LOCAL_ARCHIVE),
        "local_archive_sha256": sha256_path(LOCAL_ARCHIVE),
        "local_archive_size": LOCAL_ARCHIVE.stat().st_size,
        "make_archive": make_archive,
        "scp_archive": scp_archive,
        "snapshot_root": None,
        "snapshot_file_count": len(snapshot_files),
        "snapshot_files": snapshot_files,
        "passed": inventory.get("passed") is True,
    }
    write_json(ATTEMPT_DIR / "ATTEMPT_1_COPYBACK_RECEIPT.json", receipt)
    return receipt


def execute_live(source_authority_commit: str, freeze_commit: str) -> dict[str, Any]:
    require(RUN_ROOT.exists() is False, f"local run root already exists: {RUN_ROOT}")
    require(re.fullmatch(r"[0-9a-f]{40}", source_authority_commit) is not None, "source authority commit malformed")
    require(re.fullmatch(r"[0-9a-f]{40}", freeze_commit) is not None, "freeze commit malformed")
    require(git_value(["rev-parse", "HEAD"]) == freeze_commit, "local HEAD does not equal freeze commit")
    require(git_value(["rev-parse", "origin/codex/family10h-tomography-repair"]) == freeze_commit, "origin branch does not equal freeze commit")
    require(status_short() == "", "worktree must be clean before live invocation")
    validation = validate_package()
    require(validation["passed"], "package validation failed before live: " + ",".join(validation["failures"]))
    manifest_sidecar = read_json(MANIFEST_SIDECAR)
    package = package_tarball()
    ATTEMPT_DIR.mkdir(parents=True)
    write_json(ATTEMPT_DIR / "ATTEMPT_1_DEPLOYMENT_RECEIPT.json", {
        "schema": "FAMILY10H_PAIRED_DIRTY_PROBE_V1_1_ATTEMPT1_DEPLOYMENT_RECEIPT_V1",
        "attempt": 1,
        "created_at": utc_now(),
        "source_authority_commit": source_authority_commit,
        "freeze_commit": freeze_commit,
        "package": package,
        "manifest_file_sha256": manifest_sidecar["manifest_file_sha256"],
        "manifest_canonical_sha256": manifest_sidecar["manifest_canonical_sha256"],
        "target_host": TARGET_HOST,
        "remote_root": REMOTE_ROOT,
        "remote_package": REMOTE_PACKAGE,
        "remote_archive": REMOTE_ARCHIVE,
        "live_invocation_authorized_count": 1,
    })
    absence = run_remote(f"set -eu; test ! -e {REMOTE_ROOT}; test ! -e {REMOTE_ARCHIVE}; test ! -e {REMOTE_PACKAGE}", timeout=30)
    require(absence["returncode"] == 0, "remote canonical paths are not absent before deployment")
    create_base = run_remote(f"set -eu; mkdir -p {REMOTE_BASE}", timeout=30)
    require(create_base["returncode"] == 0, "remote base create failed")
    upload = run_scp(str(LOCAL_TMP_PACKAGE), f"{TARGET_HOST}:{REMOTE_PACKAGE}", timeout=120)
    require(upload["returncode"] == 0, "package upload failed")
    extract = run_remote(
        f"set -eu; tar -xzf {REMOTE_PACKAGE} -C {REMOTE_BASE}; "
        f"test -d {REMOTE_ROOT}; printf '%s\\n' "
        f"'source_authority_commit={source_authority_commit}' 'freeze_commit={freeze_commit}' "
        f"'transaction_run_id={public.TRANSACTION_RUN_ID}' > {REMOTE_ROOT}/{OWNER_MARKER}; "
        f"test ! -e {public.EXPECTED_REMOTE_OUTPUT_ROOT}",
        timeout=120,
    )
    require(extract["returncode"] == 0, "remote package extract failed")
    command_script = (
        f"set -eu; cd {REMOTE_ROOT}; test -f {OWNER_MARKER}; "
        "unset FAMILY10H_CARRIER_TOMOGRAPHY_RUNTIME_AUTHORITY; "
        f"FAMILY10H_CARRIER_TOMOGRAPHY_LIVE_AUTHORITY={public.TRANSACTION_RUN_ID} "
        f"FAMILY10H_CARRIER_TOMOGRAPHY_COMMIT_BINDING={source_authority_commit} "
        f"FAMILY10H_CARRIER_TOMOGRAPHY_MANIFEST_SHA256={manifest_sidecar['manifest_file_sha256']} "
        f"FAMILY10H_CARRIER_TOMOGRAPHY_TEMPERATURE_AUTHORITY_NONCE={LIVE_NONCE} "
        f"chrt -f 80 python3 -B family10h_carrier_tomography_target.py --execute-authorized "
        f"--source-root {REMOTE_ROOT} --output-root {public.EXPECTED_REMOTE_OUTPUT_ROOT}"
    )
    started_at = utc_now()
    started_receipt = {
        "schema": "FAMILY10H_PAIRED_DIRTY_PROBE_V1_1_ATTEMPT1_LIVE_INVOCATION_STARTED_V1",
        "attempt": 1,
        "started_at": started_at,
        "target_host": TARGET_HOST,
        "remote_root": REMOTE_ROOT,
        "remote_output_root": public.EXPECTED_REMOTE_OUTPUT_ROOT,
        "command": ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", TARGET_HOST, command_script],
        "live_authority": public.TRANSACTION_RUN_ID,
        "source_authority_commit": source_authority_commit,
        "freeze_commit": freeze_commit,
        "manifest_sha256": manifest_sidecar["manifest_file_sha256"],
        "temperature_authority_nonce_sha256": sha256_bytes(LIVE_NONCE.encode("ascii")),
        "live_invocation_count": 1,
    }
    write_json(ATTEMPT_DIR / "ATTEMPT_1_LIVE_INVOCATION_STARTED.json", started_receipt)
    live = run_remote(command_script, timeout=3900)
    (ATTEMPT_DIR / "ATTEMPT_1_TARGET_STDOUT.txt").write_text(live["stdout"], encoding="utf-8")
    (ATTEMPT_DIR / "ATTEMPT_1_TARGET_STDERR.txt").write_text(live["stderr"], encoding="utf-8")
    completed_receipt = {
        "schema": "FAMILY10H_PAIRED_DIRTY_PROBE_V1_1_ATTEMPT1_LIVE_INVOCATION_COMPLETED_V1",
        "attempt": 1,
        "started_at": started_at,
        "ended_at": utc_now(),
        "elapsed_seconds": live["elapsed_seconds"],
        "target_host": TARGET_HOST,
        "remote_root": REMOTE_ROOT,
        "remote_output_root": public.EXPECTED_REMOTE_OUTPUT_ROOT,
        "command": started_receipt["command"],
        "returncode": live["returncode"],
        "stdout_path": str(ATTEMPT_DIR / "ATTEMPT_1_TARGET_STDOUT.txt"),
        "stderr_path": str(ATTEMPT_DIR / "ATTEMPT_1_TARGET_STDERR.txt"),
        "stdout_sha256": sha256_bytes(live["stdout"].encode("utf-8")),
        "stderr_sha256": sha256_bytes(live["stderr"].encode("utf-8")),
        "stdout_size": len(live["stdout"].encode("utf-8")),
        "stderr_size": len(live["stderr"].encode("utf-8")),
        "live_invocation_count": 1,
        "pmu_acquisition_count": 1 if live["returncode"] == 0 else 0,
    }
    write_json(ATTEMPT_DIR / "ATTEMPT_1_LIVE_INVOCATION_COMPLETED.json", completed_receipt)
    copy_started = utc_now()
    make_archive = run_remote(
        f"set -eu; test -d {REMOTE_ROOT}; rm -f {REMOTE_ARCHIVE}; "
        f"tar -czf {REMOTE_ARCHIVE} -C {REMOTE_BASE} {public.TRANSACTION_RUN_ID}; "
        f"sha256sum {REMOTE_ARCHIVE}; du -b {REMOTE_ARCHIVE}",
        timeout=180,
    )
    require(make_archive["returncode"] == 0, "remote archive creation failed")
    if LOCAL_TMP_ARCHIVE.exists():
        LOCAL_TMP_ARCHIVE.unlink()
    scp_archive = run_scp(f"{TARGET_HOST}:{REMOTE_ARCHIVE}", str(LOCAL_TMP_ARCHIVE), timeout=180)
    require(scp_archive["returncode"] == 0, "remote archive copy-back failed")
    shutil.move(str(LOCAL_TMP_ARCHIVE), str(LOCAL_ARCHIVE))
    inventory = inventory_archive()
    copyback = copyback_receipt(copy_started, make_archive, scp_archive, inventory)
    require(copyback["passed"] is True, "copy-back inventory failed")
    cleanup = run_remote(
        f"set -eu; test -f {REMOTE_ROOT}/{OWNER_MARKER}; rm -rf {REMOTE_ROOT} {REMOTE_ARCHIVE} {REMOTE_PACKAGE}; "
        f"test ! -e {REMOTE_ROOT}; test ! -e {REMOTE_ARCHIVE}; test ! -e {REMOTE_PACKAGE}",
        timeout=120,
    )
    cleanup_receipt = {
        "schema": "FAMILY10H_PAIRED_DIRTY_PROBE_V1_1_ATTEMPT1_CLEANUP_RECEIPT_V1",
        "attempt": 1,
        "created_at": utc_now(),
        "target_host": TARGET_HOST,
        "remote_root": REMOTE_ROOT,
        "remote_archive": REMOTE_ARCHIVE,
        "remote_package": REMOTE_PACKAGE,
        "cleanup": cleanup,
        "passed": cleanup["returncode"] == 0,
    }
    write_json(ATTEMPT_DIR / "ATTEMPT_1_CLEANUP_RECEIPT.json", cleanup_receipt)
    return {
        "schema": "FAMILY10H_PAIRED_DIRTY_PROBE_V1_1_LIVE_CONTROLLER_RESULT_V1",
        "passed": live["returncode"] == 0 and inventory["passed"] and cleanup_receipt["passed"],
        "live_returncode": live["returncode"],
        "pmu_acquisition_count": completed_receipt["pmu_acquisition_count"],
        "archive_sha256": sha256_path(LOCAL_ARCHIVE),
        "archive_size": LOCAL_ARCHIVE.stat().st_size,
        "cleanup_passed": cleanup_receipt["passed"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-source-authority", action="store_true")
    parser.add_argument("--prepare-freeze-manifest", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--execute-live", action="store_true")
    parser.add_argument("--source-authority-commit", default="")
    parser.add_argument("--freeze-commit", default="")
    args = parser.parse_args()
    selected = [args.prepare_source_authority, args.prepare_freeze_manifest, args.validate_only, args.execute_live]
    if sum(1 for item in selected if item) != 1:
        parser.error("select exactly one mode")
    if args.prepare_source_authority:
        result = prepare_source_authority()
    elif args.prepare_freeze_manifest:
        result = prepare_freeze_manifest(args.source_authority_commit)
    elif args.validate_only:
        result = validate_package()
    else:
        result = execute_live(args.source_authority_commit, args.freeze_commit)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
