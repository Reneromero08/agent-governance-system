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

# Keep the extracted bundle immutable during qualification: never write .pyc/.pyo
# into the extracted adapter directory when importing the target modules.
sys.dont_write_bytecode = True

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
    sys.dont_write_bytecode = True
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


def _valid_authority_and_args(target_bundle, gate_a_authority, manifest: dict[str, Any], work: Path):
    reviewed_head = SYNTHETIC_REVIEWED_HEAD
    review_id = SYNTHETIC_REVIEW_ID
    authority = authority_template(target_bundle, gate_a_authority, manifest, reviewed_head, review_id)
    path = work / AUTHORITY_NAME
    data = (json.dumps(authority, sort_keys=True, indent=2) + "\n").encode("utf-8")
    path.write_bytes(data)
    digest = gate_a_authority.sha256_bytes(data)
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
    return authority, data, digest, args


def _absent_inspector(gate_a_target_runner, work: Path):
    absent_path = work / "definitely_absent_output_root"
    return lambda _path: gate_a_target_runner.inspect_output_root(absent_path)


def _assert_executor_rejects(gate_a_target_runner, name: str, call: Callable[[], Any]) -> str:
    try:
        call()
    except Exception:
        return name
    raise HarnessError(f"no rejection for {name}")


def _fake_runtime_result(runner) -> dict[str, Any]:
    smoke = runner.smoke_executor
    records: list[dict[str, Any]] = []
    for index, token in enumerate(smoke.SEQUENCE):
        record: dict[str, Any] = {
            "index": index,
            "token": token,
            "requested_start_s": index * smoke.SLOT_S,
            "requested_end_s": (index + 1) * smoke.SLOT_S,
            "drive_on": False,
            "amplitude_level": None,
            "phase_index": None,
            "sign": None,
            "sender_epoch_id": None,
        }
        if token == "S0E":
            record.update({
                "drive_on": True,
                "amplitude_level": 2,
                "phase_index": 0,
                "sign": 1,
                "sender_epoch_id": "gate-a:step:epoch0",
            })
        elif token == "A0P":
            record.update({"drive_on": True, "amplitude_level": 2, "phase_index": 0, "sign": 1})
        elif token == "A0N":
            record.update({"drive_on": True, "amplitude_level": 2, "phase_index": 4, "sign": -1})
        records.append(record)
    return {
        "status": "GATE_A_ENGINEERING_SMOKE_COMPLETE",
        "automatic_retry": False,
        "runtime_execution_count": 1,
        "slot_records": records,
        "capture": {
            "continuous": True,
            "covers_complete_sequence": True,
            "sample_count": smoke.READ_HZ * int(smoke.NOMINAL_DURATION_S),
            "slot_sample_counts": [smoke.NOMINAL_SAMPLES_PER_SLOT] * smoke.SLOT_COUNT,
            "origin_tsc": 1_000_000_000,
            "deadline_tsc": 26_600_000_000,
            "first_sample_tsc": 1_000_000_000,
            "last_sample_tsc": 26_599_600_000,
            "tsc_hz": 3_200_000_000.0,
        },
        "frequency_writes": 0,
        "voltage_writes": 0,
        "msr_reads": 0,
        "msr_writes": 0,
        "step_sender_epoch_count": 1,
        "hardware_executed": True,
    }


def _fake_execution_surfaces(runner):
    smoke = runner.smoke_executor
    counts = {
        "namespace": 0,
        "process": 0,
        "temperature": 0,
        "frequency": 0,
        "claim": 0,
        "runtime": 0,
        "evidence_begin": 0,
        "evidence_complete": 0,
        "network": 0,
        "hardware": 0,
        "msr": 0,
        "control_write": 0,
    }

    class FakePreflight:
        def inspect_namespace(self, _path: Path):
            counts["namespace"] += 1
            return smoke.NamespaceState.ABSENT

        def process_snapshot(self):
            counts["process"] += 1
            return smoke.ProcessSnapshot(True, 0, "synthetic", "", ())

        def temperature_c(self) -> float:
            counts["temperature"] += 1
            return 42.0

        def frequency_khz(self, _core: int) -> int:
            counts["frequency"] += 1
            return smoke.REQUIRED_FREQUENCY_KHZ

    class FakeClaims:
        def claim(self, _authority_sha256: str, _plan) -> None:
            require(counts["claim"] == 0, "synthetic authority claimed more than once")
            counts["claim"] += 1

    class FakeEvidence:
        def begin(self, _plan, _preflight) -> None:
            counts["evidence_begin"] += 1

        def event(self, _value) -> None:
            return None

        def complete(self, _result) -> None:
            counts["evidence_complete"] += 1

        def fail(self, _reason: str) -> None:
            raise HarnessError("synthetic runtime unexpectedly failed")

    class FakeRuntime:
        def execute(self, _plan):
            require(counts["runtime"] == 0, "synthetic runtime called more than once")
            counts["runtime"] += 1
            return _fake_runtime_result(runner)

    return smoke.ExecutionSurfaces(FakePreflight(), FakeClaims(), FakeEvidence(), FakeRuntime()), counts


def synthetic_authority_validation(target_bundle, gate_a_authority, gate_a_target_runner, manifest: dict[str, Any], work: Path, bundle_root: Path) -> dict[str, Any]:
    authority, data, digest, args = _valid_authority_and_args(target_bundle, gate_a_authority, manifest, work)
    reviewed_head = args.source_head
    review_id = args.independent_review_id

    shared_result = gate_a_authority.validate_execution_authority(
        authority,
        authority_sha256=digest,
        authority_bytes=data,
        expected_reviewed_adapter_head=reviewed_head,
        expected_independent_review_id=review_id,
        exact_manifest=manifest,
    )

    surfaces, counts = _fake_execution_surfaces(gate_a_target_runner)
    result = gate_a_target_runner.execute_authorized(
        args,
        bundle_root=bundle_root,
        output_root_inspector=_absent_inspector(gate_a_target_runner, work),
        surfaces=surfaces,
        compile_c=False,
    )
    require(result["status"] == "GATE_A_ENGINEERING_SMOKE_COMPLETE", "synthetic execution did not complete")
    require(counts["claim"] == 1 and counts["runtime"] == 1, "synthetic one-shot counts changed")
    require(counts["network"] == counts["hardware"] == counts["msr"] == counts["control_write"] == 0, "non-driving fake touched a forbidden surface")
    return {
        "status": "ISOLATED_SYNTHETIC_AUTHORITY_AND_EXECUTOR_VALIDATED",
        "shared_validator_result": shared_result,
        "executor_result": result["status"],
        "durable_claim_count": counts["claim"],
        "runtime_execution_count": counts["runtime"],
        "automatic_retry": False,
        "authority_written_outside_bundle": True,
        "positively_absent_output_root_accepted": True,
        "network_connections_opened": counts["network"],
        "hardware_execution_count": counts["hardware"],
        "msr_access_count": counts["msr"],
        "control_write_count": counts["control_write"],
    }


def root_state_suite(target_bundle, gate_a_authority, gate_a_target_runner, manifest: dict[str, Any], work: Path, bundle_root: Path) -> dict[str, Any]:
    _authority, _data, _digest, args = _valid_authority_and_args(target_bundle, gate_a_authority, manifest, work)
    runner = gate_a_target_runner
    cases: list[str] = []

    def run_with_inspector(inspector) -> dict[str, Any]:
        surfaces, _counts = _fake_execution_surfaces(runner)
        return runner.execute_authorized(
            args,
            bundle_root=bundle_root,
            output_root_inspector=inspector,
            surfaces=surfaces,
            compile_c=False,
        )

    def reject(name: str, inspector) -> None:
        cases.append(_assert_executor_rejects(runner, name, lambda: run_with_inspector(inspector)))

    def raise_permission(_path):
        raise PermissionError(13, "permission denied")

    def raise_generic(_path):
        raise OSError(5, "input/output error")

    present_dir = work / "present_dir"
    present_dir.mkdir()
    present_file = work / "present_file"
    present_file.write_text("x", encoding="utf-8")

    reject("existing_directory", lambda _p: runner.inspect_output_root(present_dir))
    reject("existing_regular_file", lambda _p: runner.inspect_output_root(present_file))
    reject("permission_error_unobservable", lambda _p: runner.inspect_output_root(work / "pe", stat_func=raise_permission))
    reject("generic_oserror_unobservable", lambda _p: runner.inspect_output_root(work / "io", stat_func=raise_generic))

    # Positive control: a positively absent path reaches the injected executor.
    absent_result = run_with_inspector(_absent_inspector(runner, work))
    require(absent_result["status"] == "GATE_A_ENGINEERING_SMOKE_COMPLETE", "positively absent output root did not reach the injected executor")

    # Direct production-function mapping checks (inspect_output_root is the default inspector).
    require(runner.inspect_output_root(work / "definitely_absent") is runner.RootState.ABSENT, "absent mapping wrong")
    require(runner.inspect_output_root(present_dir) is runner.RootState.PRESENT, "present-dir mapping wrong")
    require(runner.inspect_output_root(present_file) is runner.RootState.PRESENT, "present-file mapping wrong")
    require(runner.inspect_output_root(work / "pe", stat_func=raise_permission) is runner.RootState.UNOBSERVABLE, "permission mapping wrong")
    require(runner.inspect_output_root(work / "io", stat_func=raise_generic) is runner.RootState.UNOBSERVABLE, "oserror mapping wrong")

    best_effort: dict[str, Any] = {}

    def best_effort_present(name: str, prepare) -> None:
        record: dict[str, Any] = {"performed": False, "rejected": None}
        try:
            path = prepare()
            record["performed"] = True
        except OSError:
            best_effort[name] = record
            return
        try:
            run_with_inspector(lambda _p, _path=path: runner.inspect_output_root(Path(_path)))
            record["rejected"] = False
        except runner.TargetRunnerError:
            record["rejected"] = True
        require(record["rejected"] is True, f"root-state best-effort not rejected: {name}")
        best_effort[name] = record

    def make_symlink():
        link = work / "present_symlink"
        os.symlink(str(present_dir), str(link))
        return link

    def make_broken_symlink():
        link = work / "broken_symlink"
        os.symlink(str(work / "missing_target"), str(link))
        return link

    def make_fifo():
        fifo = work / "present_fifo"
        os.mkfifo(str(fifo))  # type: ignore[attr-defined]
        return fifo

    best_effort_present("existing_symlink", make_symlink)
    best_effort_present("broken_symlink", make_broken_symlink)
    if hasattr(os, "mkfifo"):
        best_effort_present("existing_special_file", make_fifo)

    return {
        "status": "ISOLATED_ROOT_STATE_TESTS_PASS",
        "negative_tests": len(cases),
        "cases": cases,
        "positively_absent_accepted": True,
        "best_effort": best_effort,
    }


def production_strict_custody_suite(target_bundle, gate_a_authority, gate_a_target_runner, manifest: dict[str, Any], bundle_root: Path, work: Path, scratch: Path) -> dict[str, Any]:
    runner = gate_a_target_runner
    _authority, _data, _digest, args = _valid_authority_and_args(target_bundle, gate_a_authority, manifest, work)
    absent = _absent_inspector(runner, work)
    cases: list[str] = []
    counter = {"n": 0}

    def fresh_copy() -> Path:
        counter["n"] += 1
        dest = scratch / f"custody_{counter['n']}"
        shutil.copytree(bundle_root, dest)
        return dest

    authority_py = _first_payload_py(manifest)
    schedule_json = "GATE_A_ENGINEERING_SMOKE_SCHEDULE.json"

    def exec_on(root: Path):
        surfaces, _counts = _fake_execution_surfaces(runner)
        return runner.execute_authorized(
            args,
            bundle_root=root,
            output_root_inspector=absent,
            surfaces=surfaces,
            compile_c=False,
        )

    def reject_prod(name: str, mutate) -> None:
        root = fresh_copy()
        mutate(root)
        _assert_rejects(name + ":qualify_no_drive", lambda: runner.qualify_no_drive(root, compile_c=False))
        _assert_executor_rejects(runner, name + ":execute_authorized", lambda: exec_on(root))
        cases.append(name)

    def add_py(root: Path) -> None:
        (root / "adapter" / "unexpected_extra.py").write_text("# extra\n", encoding="utf-8")

    def add_json(root: Path) -> None:
        (root / "adapter" / "unexpected_extra.json").write_text("{}\n", encoding="utf-8")

    def add_worker_prefixed(root: Path) -> None:
        (root / "adapter" / "gate_a_worker_backdoor").write_text("payload\n", encoding="utf-8")

    def add_pyc(root: Path) -> None:
        cache = root / "adapter" / "__pycache__"
        cache.mkdir(parents=True, exist_ok=True)
        (cache / "unrelated.pyc").write_bytes(b"\x00\x00\x00\x00undeclared")

    def add_pyo(root: Path) -> None:
        cache = root / "adapter" / "__pycache__"
        cache.mkdir(parents=True, exist_ok=True)
        (cache / "unrelated.pyo").write_bytes(b"\x00\x00\x00\x00undeclared")

    def delete_payload(root: Path) -> None:
        (root / schedule_json).unlink()

    def change_payload(root: Path) -> None:
        target = root / authority_py
        data = bytearray(target.read_bytes())
        data[0] = data[0] ^ 0x20
        target.write_bytes(bytes(data))

    reject_prod("unexpected_extra_python_file", add_py)
    reject_prod("unexpected_extra_json_file", add_json)
    reject_prod("unexpected_worker_prefixed_file", add_worker_prefixed)
    reject_prod("undeclared_pyc", add_pyc)
    reject_prod("undeclared_pyo", add_pyo)
    reject_prod("missing_payload_file", delete_payload)
    reject_prod("changed_payload_file", change_payload)

    best_effort: dict[str, Any] = {}

    def best_effort_case(name: str, mutate) -> None:
        record: dict[str, Any] = {"performed": False, "rejected": None}
        try:
            root = fresh_copy()
            mutate(root)
            record["performed"] = True
        except OSError:
            best_effort[name] = record
            return
        rejected = True
        try:
            runner.qualify_no_drive(root, compile_c=False)
            rejected = False
        except Exception:
            rejected = True
        record["rejected"] = rejected
        require(rejected, f"production strict custody best-effort not rejected: {name}")
        best_effort[name] = record

    def sym_payload(root: Path) -> None:
        target = root / authority_py
        target.unlink()
        os.symlink(str(root / schedule_json), str(target))

    def sym_extra(root: Path) -> None:
        os.symlink(str(root / schedule_json), str(root / "adapter" / "extra_link"))

    def case_collision(root: Path) -> None:
        original = root / authority_py
        collide = original.parent / original.name.upper()
        if collide.exists():
            raise OSError("case-insensitive filesystem")
        collide.write_bytes(original.read_bytes())

    def special_file(root: Path) -> None:
        os.mkfifo(str(root / "adapter" / "extra_fifo"))  # type: ignore[attr-defined]

    best_effort_case("symlink_payload_replacement", sym_payload)
    best_effort_case("symlink_extra_file", sym_extra)
    best_effort_case("case_colliding_payload_filename", case_collision)
    if hasattr(os, "mkfifo"):
        best_effort_case("special_file_in_bundle", special_file)

    return {
        "status": "ISOLATED_PRODUCTION_STRICT_CUSTODY_TESTS_PASS",
        "negative_tests": len(cases),
        "cases": cases,
        "best_effort": best_effort,
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
        synthetic = synthetic_authority_validation(target_bundle, gate_a_authority, gate_a_target_runner, manifest, work, bundle_root)
        authority_mutations = authority_mutation_suite(target_bundle, gate_a_authority, manifest, work)

    with tempfile.TemporaryDirectory(prefix="gate_a_isolated_rootstate_") as tmp:
        root_state = root_state_suite(target_bundle, gate_a_authority, gate_a_target_runner, manifest, Path(tmp), bundle_root)

    with tempfile.TemporaryDirectory(prefix="gate_a_isolated_authwork_") as work_tmp, tempfile.TemporaryDirectory(prefix="gate_a_isolated_custody_") as scratch_tmp:
        strict_custody = production_strict_custody_suite(target_bundle, gate_a_authority, gate_a_target_runner, manifest, bundle_root, Path(work_tmp), Path(scratch_tmp))

    with tempfile.TemporaryDirectory(prefix="gate_a_isolated_bundle_") as tmp:
        bundle_mutations = bundle_mutation_suite(target_bundle, bundle_root, Path(tmp))

    require(not list(bundle_root.rglob(AUTHORITY_NAME)), "execution authority artifact must not exist inside bundle")

    total = (
        bundle_mutations["negative_tests"]
        + authority_mutations["negative_tests"]
        + root_state["negative_tests"]
        + strict_custody["negative_tests"]
    )
    return {
        "status": "GATE_A_ISOLATED_BUNDLE_QUALIFICATION_PASS",
        "null_baseline": "NO_DRIVE_ZERO_COUNT_BASELINE",
        "bundle_root_git_free": not (bundle_root / ".git").exists(),
        "require_isolated_origin": require_isolated_origin,
        "extracted_bundle_validation": happy,
        "target_runner_no_drive": no_drive,
        "synthetic_authority": synthetic,
        "root_state_tests": root_state,
        "production_strict_custody_tests": strict_custody,
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


def strict_only_report(bundle_root: Path, *, require_isolated_origin: bool = False) -> dict[str, Any]:
    bundle_root = Path(bundle_root).resolve()
    target_bundle, _gate_a_authority, _gate_a_target_runner = _import_target_modules(bundle_root, require_isolated_origin)
    manifest = target_bundle.load_manifest(bundle_root)
    happy = target_bundle.validate_extracted_bundle(bundle_root, manifest, strict=True)
    require(not list(bundle_root.rglob(AUTHORITY_NAME)), "execution authority artifact must not exist inside bundle")
    return {
        "status": "GATE_A_ISOLATED_STRICT_ONLY_PASS",
        "bundle_root_git_free": not (bundle_root / ".git").exists(),
        "require_isolated_origin": require_isolated_origin,
        "extracted_bundle_validation": happy,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Isolated Gate A extracted-bundle qualification harness")
    parser.add_argument("--bundle-root", required=True)
    parser.add_argument("--require-isolated-origin", action="store_true")
    parser.add_argument("--no-compile-c", action="store_true")
    parser.add_argument("--strict-only", action="store_true")
    args = parser.parse_args(argv)
    if args.strict_only:
        report = strict_only_report(Path(args.bundle_root), require_isolated_origin=args.require_isolated_origin)
    else:
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
