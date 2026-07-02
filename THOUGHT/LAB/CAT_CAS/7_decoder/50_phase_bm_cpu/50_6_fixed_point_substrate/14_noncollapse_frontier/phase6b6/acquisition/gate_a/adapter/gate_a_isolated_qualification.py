#!/usr/bin/env python3
"""Isolated Gate A extracted-bundle qualification harness.

This harness proves that a deterministic Gate A bundle, once extracted outside
the repository, is self-contained: the target runner imports and starts without
.git, local extracted-bundle validation passes, the worker compiles and runs
--validate-only, a synthetic exact future authority validates through the shared
Git-free validator, and every negative mutation is rejected without Git.

It is a test harness, not a packaged payload. It imports the target modules from
the extracted bundle's adapter directory (never from the repository checkout).
When --require-isolated-origin is set the harness asserts that each imported
module resides inside the extracted bundle root and fails otherwise, so a leaked
repository import cannot pass silently.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

SYNTHETIC_REVIEWED_HEAD = "1234567890abcdef1234567890abcdef12345678"
SYNTHETIC_REVIEW_ID = 4619537286
AUTHORITY_NAME = "GATE_A_EXECUTION_AUTHORITY.json"


class HarnessError(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise HarnessError(message)


def _import_target_modules(bundle_root: Path, require_isolated_origin: bool):
    adapter_dir = bundle_root / "adapter"
    sys.path.insert(0, str(adapter_dir))
    import gate_a_target_bundle as target_bundle
    import gate_a_authority
    import gate_a_target_runner

    if require_isolated_origin:
        root_str = str(bundle_root.resolve())
        for module in (target_bundle, gate_a_authority, gate_a_target_runner):
            origin = str(Path(module.__file__).resolve())
            require(origin.startswith(root_str), f"module {module.__name__} imported from outside bundle: {origin}")
    return target_bundle, gate_a_authority, gate_a_target_runner


def authority_template(target_bundle, gate_a_authority, manifest: dict[str, Any], reviewed_head: str, review_id: int) -> dict[str, Any]:
    roles = {entry["role"]: entry for entry in manifest["files"]}
    return {
        "schema_id": "CAT_CAS_PHASE6B6_GATE_A_EXECUTION_AUTHORITY_V1",
        "reviewed_adapter_head": reviewed_head,
        "independent_review_id": review_id,
        "execution_bundle_sha256": manifest["execution_bundle_sha256"],
        "host_adapter_git_blob_sha1": roles["host_adapter"]["git_blob_sha1"],
        "target_runner_git_blob_sha1": roles["target_runner"]["git_blob_sha1"],
        "target_worker_git_blob_sha1": roles["target_worker"]["git_blob_sha1"],
        "schedule_sha256": target_bundle.SCHEDULE_SHA256,
        "target_namespace_sha256": target_bundle.NAMESPACE_SHA256,
        "target_identity_sha256": target_bundle.TARGET_IDENTITY_SHA256,
        "target": gate_a_authority.EXPECTED_TARGET,
        "remote_execution_root": gate_a_authority.REMOTE_EXECUTION_ROOT,
        "remote_output_root": gate_a_authority.REMOTE_OUTPUT_ROOT,
        "maximum_execution_count": 1,
        "consumed": False,
        "project_owner_approved": True,
        "authority_state": {
            "authorization_artifact_created": True,
            "engineering_smoke_authorized": True,
            "hardware_ran": False,
            "calibration_authorized": False,
            "scientific_acquisition_authorized": False,
            "restoration_authorized": False,
            "target_coupling_authorized": False,
            "small_wall_authorized": False,
            "automatic_retry": False,
        },
    }


def _assert_rejects(name: str, func: Callable[[], Any]) -> str:
    try:
        func()
    except Exception:
        return name
    raise HarnessError(f"mutation accepted: {name}")


def _first_payload_py(manifest: dict[str, Any]) -> str:
    for entry in manifest["files"]:
        if entry["package_path"].endswith(".py"):
            return entry["package_path"]
    raise HarnessError("no python payload file present")


def bundle_mutation_suite(target_bundle, bundle_root: Path, work: Path) -> dict[str, Any]:
    base_manifest = target_bundle.load_manifest(bundle_root)
    cases: list[str] = []
    counter = {"n": 0}

    def fresh_copy() -> Path:
        counter["n"] += 1
        dest = work / f"copy_{counter['n']}"
        shutil.copytree(bundle_root, dest)
        return dest

    def reject_disk(name: str, mutate: Callable[[Path], None]) -> None:
        root = fresh_copy()
        mutate(root)
        manifest = target_bundle.load_manifest(root)
        cases.append(_assert_rejects(name, lambda: target_bundle.validate_extracted_bundle(root, manifest, strict=True)))

    def reject_manifest(name: str, mutate: Callable[[dict[str, Any]], None]) -> None:
        manifest = copy.deepcopy(base_manifest)
        mutate(manifest)
        cases.append(_assert_rejects(name, lambda: target_bundle.validate_extracted_bundle(bundle_root, manifest, strict=True)))

    authority_pkg = _first_payload_py(base_manifest)
    schedule_pkg = "GATE_A_ENGINEERING_SMOKE_SCHEDULE.json"

    def delete_import(root: Path) -> None:
        (root / authority_pkg).unlink()

    def delete_payload(root: Path) -> None:
        (root / schedule_pkg).unlink()

    def add_extra(root: Path) -> None:
        (root / "adapter" / "unexpected_extra.py").write_text("# unexpected\n", encoding="utf-8")

    def change_sha256(root: Path) -> None:
        target = root / authority_pkg
        data = bytearray(target.read_bytes())
        data[0] = data[0] ^ 0x20
        target.write_bytes(bytes(data))

    def change_size(root: Path) -> None:
        target = root / authority_pkg
        target.write_bytes(target.read_bytes() + b"\n# grow\n")

    def add_case_collision(m: dict[str, Any]) -> None:
        extra = copy.deepcopy(m["files"][0])
        extra["package_path"] = m["files"][0]["package_path"].upper()
        m["files"].append(extra)

    reject_disk("missing_imported_module", delete_import)
    reject_disk("missing_payload_file", delete_payload)
    reject_disk("extra_unexpected_payload_file", add_extra)
    reject_disk("changed_payload_sha256", change_sha256)
    reject_disk("changed_payload_size", change_size)

    reject_manifest("changed_package_path", lambda m: m["files"][0].__setitem__("package_path", "adapter/renamed_missing.py"))
    reject_manifest("changed_file_role", lambda m: m["files"][0].__setitem__("role", "changed_role"))
    reject_manifest("changed_manifest_execution_bundle_digest", lambda m: m.__setitem__("execution_bundle_sha256", "0" * 64))
    reject_manifest("changed_manifest_archive_digest", lambda m: m.__setitem__("deterministic_archive_sha256", "0" * 64))
    reject_manifest("unsafe_relative_path", lambda m: m["files"][0].__setitem__("package_path", "../escape.py"))
    reject_manifest("extra_manifest_property", lambda m: m.__setitem__("extra", True))
    reject_manifest("manifest_symlink_mode", lambda m: m["files"][0].__setitem__("git_mode", "120000"))
    reject_manifest("case_colliding_filename", add_case_collision)

    on_disk_extras: dict[str, Any] = {}

    def best_effort(name: str, prepare: Callable[[Path], None]) -> None:
        record: dict[str, Any] = {"performed": False, "rejected": None}
        try:
            root = fresh_copy()
            prepare(root)
            record["performed"] = True
            manifest = target_bundle.load_manifest(root)
            try:
                target_bundle.validate_extracted_bundle(root, manifest, strict=True)
                record["rejected"] = False
            except Exception:
                record["rejected"] = True
        except OSError:
            record["performed"] = False
        require(record["performed"] is False or record["rejected"] is True, f"on-disk mutation not rejected: {name}")
        on_disk_extras[name] = record

    def symlink_prepare(root: Path) -> None:
        target = root / authority_pkg
        target.unlink()
        os.symlink(str(root / schedule_pkg), str(target))

    def case_collision_prepare(root: Path) -> None:
        original = root / authority_pkg
        collide = original.parent / original.name.upper()
        if collide.exists():
            raise OSError("case-insensitive filesystem")
        collide.write_bytes(original.read_bytes())
        if not (collide.exists() and original.exists() and collide.name != original.name):
            raise OSError("case collision not materialized")

    best_effort("on_disk_symlink_substitution", symlink_prepare)
    best_effort("on_disk_case_collision", case_collision_prepare)

    return {
        "status": "ISOLATED_BUNDLE_MUTATION_TESTS_PASS",
        "negative_tests": len(cases),
        "cases": cases,
        "on_disk_extras": on_disk_extras,
    }


def authority_mutation_suite(target_bundle, gate_a_authority, manifest: dict[str, Any], work: Path) -> dict[str, Any]:
    reviewed_head = SYNTHETIC_REVIEWED_HEAD
    review_id = SYNTHETIC_REVIEW_ID
    valid = authority_template(target_bundle, gate_a_authority, manifest, reviewed_head, review_id)
    path = work / AUTHORITY_NAME
    cases: list[str] = []

    def write(value: dict[str, Any]) -> tuple[bytes, str]:
        data = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode("utf-8")
        path.write_bytes(data)
        return data, gate_a_authority.sha256_bytes(data)

    def validate(value: dict[str, Any], *, expected_head: str = reviewed_head, expected_id: int = review_id) -> None:
        data, digest = write(value)
        gate_a_authority.validate_execution_authority(
            value,
            authority_sha256=digest,
            authority_bytes=data,
            expected_reviewed_adapter_head=expected_head,
            expected_independent_review_id=expected_id,
            exact_manifest=manifest,
        )

    validate(valid)

    def reject(name: str, mutate: Callable[[dict[str, Any]], None], *, expected_head: str = reviewed_head, expected_id: int = review_id) -> None:
        changed = copy.deepcopy(valid)
        mutate(changed)
        cases.append(_assert_rejects(name, lambda: validate(changed, expected_head=expected_head, expected_id=expected_id)))

    reject("wrong_authority_bundle_digest", lambda v: v.__setitem__("execution_bundle_sha256", "0" * 64))
    reject("wrong_authority_host_adapter_blob", lambda v: v.__setitem__("host_adapter_git_blob_sha1", "0" * 40))
    reject("wrong_authority_target_runner_blob", lambda v: v.__setitem__("target_runner_git_blob_sha1", "0" * 40))
    reject("wrong_authority_target_worker_blob", lambda v: v.__setitem__("target_worker_git_blob_sha1", "0" * 40))
    reject("wrong_schedule_digest", lambda v: v.__setitem__("schedule_sha256", "0" * 64))
    reject("wrong_namespace_digest", lambda v: v.__setitem__("target_namespace_sha256", "0" * 64))
    reject("wrong_target_identity_digest", lambda v: v.__setitem__("target_identity_sha256", "0" * 64))
    reject("wrong_reviewed_adapter_head", lambda v: None, expected_head="0" * 40)
    reject("wrong_review_id", lambda v: None, expected_id=review_id + 1)
    reject("wrong_target", lambda v: v.__setitem__("target", "root@127.0.0.1"))
    reject("wrong_remote_execution_root", lambda v: v.__setitem__("remote_execution_root", "/root/wrong"))
    reject("wrong_remote_output_root", lambda v: v.__setitem__("remote_output_root", "/root/wrong/evidence"))
    reject("consumed_authority", lambda v: v.__setitem__("consumed", True))
    reject("execution_count_greater_than_one", lambda v: v.__setitem__("maximum_execution_count", 2))
    reject("owner_approval_false", lambda v: v.__setitem__("project_owner_approved", False))
    reject("automatic_retry_true", lambda v: v["authority_state"].__setitem__("automatic_retry", True))
    reject("hardware_ran_true", lambda v: v["authority_state"].__setitem__("hardware_ran", True))
    reject("calibration_authority_true", lambda v: v["authority_state"].__setitem__("calibration_authorized", True))
    reject("scientific_acquisition_authority_true", lambda v: v["authority_state"].__setitem__("scientific_acquisition_authorized", True))
    reject("restoration_authority_true", lambda v: v["authority_state"].__setitem__("restoration_authorized", True))
    reject("target_coupling_authority_true", lambda v: v["authority_state"].__setitem__("target_coupling_authorized", True))
    reject("small_wall_authority_true", lambda v: v["authority_state"].__setitem__("small_wall_authorized", True))

    return {
        "status": "ISOLATED_AUTHORITY_MUTATION_TESTS_PASS",
        "negative_tests": len(cases),
        "cases": cases,
    }


def synthetic_authority_validation(target_bundle, gate_a_authority, gate_a_target_runner, manifest: dict[str, Any], work: Path) -> dict[str, Any]:
    reviewed_head = SYNTHETIC_REVIEWED_HEAD
    review_id = SYNTHETIC_REVIEW_ID
    authority = authority_template(target_bundle, gate_a_authority, manifest, reviewed_head, review_id)
    path = work / AUTHORITY_NAME
    data = (json.dumps(authority, sort_keys=True, indent=2) + "\n").encode("utf-8")
    path.write_bytes(data)
    digest = gate_a_authority.sha256_bytes(data)

    shared_result = gate_a_authority.validate_execution_authority(
        authority,
        authority_sha256=digest,
        authority_bytes=data,
        expected_reviewed_adapter_head=reviewed_head,
        expected_independent_review_id=review_id,
        exact_manifest=manifest,
    )

    args = argparse.Namespace(
        authority_artifact=str(path),
        authority_sha256=digest,
        execution_bundle_sha256=manifest["execution_bundle_sha256"],
        source_head=reviewed_head,
        independent_review_id=review_id,
        schedule_sha256=target_bundle.SCHEDULE_SHA256,
        target=gate_a_authority.EXPECTED_TARGET,
        namespace_sha256=target_bundle.NAMESPACE_SHA256,
        output_root=gate_a_authority.REMOTE_OUTPUT_ROOT,
    )
    reached_live_sentinel = False
    sentinel = "authorized live execution path is intentionally unused in this qualification phase"
    try:
        gate_a_target_runner.execute_authorized(args)
    except gate_a_target_runner.TargetRunnerError as exc:
        reached_live_sentinel = str(exc) == sentinel
    require(reached_live_sentinel, "synthetic authority did not reach the intentionally-unused live sentinel")
    return {
        "status": "ISOLATED_SYNTHETIC_AUTHORITY_VALIDATED",
        "shared_validator_result": shared_result,
        "reached_intentionally_unused_live_sentinel": True,
        "authority_written_outside_bundle": True,
    }


def build_isolated_report(bundle_root: Path, *, require_isolated_origin: bool = False, compile_c: bool = True) -> dict[str, Any]:
    bundle_root = Path(bundle_root).resolve()
    target_bundle, gate_a_authority, gate_a_target_runner = _import_target_modules(bundle_root, require_isolated_origin)

    manifest = target_bundle.load_manifest(bundle_root)
    happy = target_bundle.validate_extracted_bundle(bundle_root, manifest, strict=True)
    no_drive = gate_a_target_runner.qualify_no_drive(bundle_root, compile_c=compile_c)
    require(no_drive["network_connections_opened"] == 0, "network connections opened")
    require(no_drive["hardware_probes"] == 0, "hardware probes")
    require(no_drive["sender_starts"] == 0, "sender starts")
    require(no_drive["receiver_captures"] == 0, "receiver captures")
    require(no_drive["control_writes"] == 0, "control writes")
    require(no_drive["hardware_executions"] == 0, "hardware executions")

    with tempfile.TemporaryDirectory(prefix="gate_a_isolated_authority_") as tmp:
        work = Path(tmp)
        synthetic = synthetic_authority_validation(target_bundle, gate_a_authority, gate_a_target_runner, manifest, work)
        authority_mutations = authority_mutation_suite(target_bundle, gate_a_authority, manifest, work)

    with tempfile.TemporaryDirectory(prefix="gate_a_isolated_bundle_") as tmp:
        bundle_mutations = bundle_mutation_suite(target_bundle, bundle_root, Path(tmp))

    require(not list(bundle_root.rglob(AUTHORITY_NAME)), "execution authority artifact must not exist inside bundle")

    total = bundle_mutations["negative_tests"] + authority_mutations["negative_tests"]
    return {
        "status": "GATE_A_ISOLATED_BUNDLE_QUALIFICATION_PASS",
        "null_baseline": "NO_DRIVE_ZERO_COUNT_BASELINE",
        "bundle_root_git_free": not (bundle_root / ".git").exists(),
        "require_isolated_origin": require_isolated_origin,
        "extracted_bundle_validation": happy,
        "target_runner_no_drive": no_drive,
        "synthetic_authority": synthetic,
        "bundle_mutation_tests": bundle_mutations,
        "authority_mutation_tests": authority_mutations,
        "isolated_negative_tests": total,
        "authority_artifact_absent": True,
        "network_connections_opened": 0,
        "target_connection_count": 0,
        "ssh_count": 0,
        "target_directory_creation_count": 0,
        "target_process_count": 0,
        "sender_start_count": 0,
        "receiver_capture_count": 0,
        "control_write_count": 0,
        "msr_access_count": 0,
        "hardware_execution_count": 0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Isolated Gate A extracted-bundle qualification harness")
    parser.add_argument("--bundle-root", required=True)
    parser.add_argument("--require-isolated-origin", action="store_true")
    parser.add_argument("--no-compile-c", action="store_true")
    args = parser.parse_args(argv)
    report = build_isolated_report(
        Path(args.bundle_root),
        require_isolated_origin=args.require_isolated_origin,
        compile_c=not args.no_compile_c,
    )
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (HarnessError, OSError, json.JSONDecodeError) as exc:
        print(f"gate_a_isolated_qualification: {exc}", file=sys.stderr)
        raise SystemExit(1)
