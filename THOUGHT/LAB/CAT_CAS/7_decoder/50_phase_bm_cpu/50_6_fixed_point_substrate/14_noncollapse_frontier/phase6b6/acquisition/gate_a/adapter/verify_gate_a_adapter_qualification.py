#!/usr/bin/env python3
"""Dedicated Gate A adapter no-drive qualification verifier."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

import build_gate_a_execution_bundle as bundle
import gate_a_hardware_adapter as adapter
import gate_a_target_runner

HERE = Path(__file__).resolve().parent
GATE_A = HERE.parent
RESULT = HERE / "GATE_A_ADAPTER_QUALIFICATION_RESULT.json"
CANDIDATE_V2 = HERE / "GATE_A_ENGINEERING_SMOKE_AUTHORITY_CANDIDATE_V2.json"
MANIFEST = HERE / "GATE_A_EXECUTION_BUNDLE_MANIFEST.json"
AUTHORITY_SCHEMA = HERE / "schemas" / "gate_a_execution_authority.schema.json"
AUTHORITY_NAME = "GATE_A_EXECUTION_AUTHORITY.json"
CONTRACT = HERE / "GATE_A_ADAPTER_QUALIFICATION_CONTRACT.json"


class VerifyError(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise VerifyError(message)


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"object required: {path}")
    return value


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def committed_sha256(path: Path, treeish: str = "HEAD") -> str:
    blob = run(["git", "rev-parse", f"{treeish}:{git_rel(path)}"], cwd=bundle.repo_root()).stdout.strip()
    data = subprocess.run(["git", "cat-file", "blob", blob], cwd=bundle.repo_root(), stdout=subprocess.PIPE, check=True).stdout
    return hashlib.sha256(data).hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")).hexdigest()


def run(argv: list[str], *, cwd: Path = HERE, check: bool = True, input_text: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, cwd=cwd, input=input_text, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=check)


def git_blob(path: Path) -> str:
    return run(["git", "hash-object", str(path)]).stdout.strip()


def git_rel(path: Path) -> str:
    return path.resolve().relative_to(bundle.repo_root().resolve()).as_posix()


def validate_schema_closed(schema: dict[str, Any]) -> None:
    require(schema["additionalProperties"] is False, "authority schema top level open")
    require(set(schema["required"]) == set(schema["properties"]), "authority schema required/properties mismatch")
    state = schema["properties"]["authority_state"]
    require(state["additionalProperties"] is False, "authority state schema open")
    require(set(state["required"]) == set(state["properties"]), "authority state required/properties mismatch")


def validate_contract(contract: dict[str, Any]) -> None:
    require(contract["schema_id"] == "CAT_CAS_PHASE6B6_GATE_A_ADAPTER_QUALIFICATION_CONTRACT_V1", "contract schema mismatch")
    require(contract["base_main_commit"] == bundle.BASE_MAIN, "contract base mismatch")
    require(contract["reviewed_gate_a_plan_head"] == bundle.REVIEWED_PLAN_HEAD, "contract reviewed plan mismatch")
    require(contract["gate_a_plan_review"] == bundle.PLAN_REVIEW_ID, "contract review mismatch")
    require(contract["schedule_sha256"] == bundle.SCHEDULE_SHA256, "contract schedule mismatch")
    require(contract["target_namespace_sha256"] == bundle.NAMESPACE_SHA256, "contract namespace mismatch")
    require(contract["target_identity_stdout_sha256"] == bundle.TARGET_IDENTITY_SHA256, "contract target identity mismatch")
    require(contract["target_geometry"]["automatic_retry"] is False, "contract retry boundary mismatch")
    require(contract["target_geometry"]["maximum_execution_count"] == 1, "contract execution count mismatch")
    for key, value in contract["authority_false_state"].items():
        require(value is False, f"contract authority flag must be false: {key}")
    required_tests = set(contract["required_negative_tests"])
    for expected in ("worktree-byte mutation behavior", "index-byte mutation detection"):
        require(expected in required_tests, f"contract negative test missing: {expected}")


def static_forbidden_surface_scan() -> dict[str, Any]:
    blocked_regexes = [
        ("shell_true", "shell=True"),
        ("os_system", "os.system"),
        ("popen", "popen"),
        ("eval_call", "eval("),
        ("exec_call", "exec("),
        ("wrmsr", "wrmsr"),
        ("msr_tools", "msr-tools"),
        ("voltage_write_phrase", "voltage write"),
        ("frequency_write_phrase", "frequency write"),
        ("automatic_retry_phrase", "automatic retry"),
    ]
    control_fragments = [
        ("min_freq", "scaling_" + "min_" + "freq"),
        ("max_freq", "scaling_" + "max_" + "freq"),
        ("boost_path", "/" + "cpufreq" + "/" + "boost"),
        ("cpu_device", "/" + "dev" + "/" + "cpu" + "/"),
    ]
    implementation_files = [
        HERE / "gate_a_authority.py",
        HERE / "gate_a_hardware_adapter.py",
        HERE / "gate_a_target_runner.py",
        HERE / "gate_a_worker.c",
        HERE / "build_gate_a_execution_bundle.py",
    ]
    matches: list[dict[str, str]] = []
    for path in implementation_files:
        text = path.read_text(encoding="utf-8")
        for name, needle in blocked_regexes + control_fragments:
            if needle in text:
                matches.append({"file": path.name, "pattern": name})
    require(not matches, f"forbidden implementation surface present: {matches}")
    return {"status": "FORBIDDEN_SURFACE_SCAN_PASS", "matches": 0}


def assert_rejects(name: str, func: Callable[[], None]) -> str:
    try:
        func()
    except Exception:
        return name
    raise VerifyError(f"mutation accepted: {name}")


def target_args(manifest: dict[str, Any], reviewed_head: str, review_id: int, output_root: str | None = None) -> Any:
    return type("Args", (), {
        "authority_artifact": None,
        "authority_sha256": None,
        "execution_bundle_sha256": manifest["execution_bundle_sha256"],
        "source_head": reviewed_head,
        "independent_review_id": review_id,
        "schedule_sha256": adapter.SCHEDULE_SHA256,
        "target": "root@192.168.137.100",
        "namespace_sha256": adapter.NAMESPACE_SHA256,
        "output_root": output_root or adapter.REMOTE_OUTPUT_ROOT,
    })()


def authority_template(manifest: dict[str, Any], reviewed_head: str, review_id: int) -> dict[str, Any]:
    files_by_role = {entry["role"]: entry for entry in manifest["files"]}
    return {
        "schema_id": "CAT_CAS_PHASE6B6_GATE_A_EXECUTION_AUTHORITY_V1",
        "reviewed_adapter_head": reviewed_head,
        "independent_review_id": review_id,
        "execution_bundle_sha256": manifest["execution_bundle_sha256"],
        "host_adapter_git_blob_sha1": files_by_role["host_adapter"]["git_blob_sha1"],
        "target_runner_git_blob_sha1": files_by_role["target_runner"]["git_blob_sha1"],
        "target_worker_git_blob_sha1": files_by_role["target_worker"]["git_blob_sha1"],
        "schedule_sha256": adapter.SCHEDULE_SHA256,
        "target_namespace_sha256": adapter.NAMESPACE_SHA256,
        "target_identity_sha256": adapter.TARGET_IDENTITY_SHA256,
        "target": "root@192.168.137.100",
        "remote_execution_root": adapter.REMOTE_EXECUTION_ROOT,
        "remote_output_root": adapter.REMOTE_OUTPUT_ROOT,
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


def validate_authority_host(path: Path, digest: str, manifest: dict[str, Any], reviewed_head: str, review_id: int) -> dict[str, Any]:
    authority = load(path)
    return adapter.validate_future_authority(
        authority,
        authority_sha256=digest,
        authority_bytes=path.read_bytes(),
        expected_reviewed_adapter_head=reviewed_head,
        expected_independent_review_id=review_id,
        expected_manifest=manifest,
    )


def validate_authority_both(path: Path, digest: str, manifest: dict[str, Any], reviewed_head: str, review_id: int) -> tuple[dict[str, Any], dict[str, Any]]:
    host_result = validate_authority_host(path, digest, manifest, reviewed_head, review_id)
    runner_result = gate_a_target_runner.validate_authority(path, digest, target_args(manifest, reviewed_head, review_id))
    return host_result, runner_result


def manifest_mutation_tests(manifest: dict[str, Any]) -> dict[str, Any]:
    cases: list[str] = []

    def mutate_file(field: str, value: Any) -> Callable[[], None]:
        def inner() -> None:
            changed = copy.deepcopy(manifest)
            changed["files"][0][field] = value
            adapter.validate_bundle_manifest(changed)
        return inner

    def remove_file() -> None:
        changed = copy.deepcopy(manifest)
        changed["files"].pop()
        adapter.validate_bundle_manifest(changed)

    def add_file() -> None:
        changed = copy.deepcopy(manifest)
        extra = copy.deepcopy(changed["files"][0])
        extra["package_path"] = "adapter/extra.txt"
        changed["files"].append(extra)
        adapter.validate_bundle_manifest(changed)

    def duplicate_path() -> None:
        changed = copy.deepcopy(manifest)
        changed["files"].append(copy.deepcopy(changed["files"][0]))
        adapter.validate_bundle_manifest(changed)

    def case_collision() -> None:
        changed = copy.deepcopy(manifest)
        extra = copy.deepcopy(changed["files"][0])
        extra["package_path"] = changed["files"][0]["package_path"].upper()
        changed["files"].append(extra)
        adapter.validate_bundle_manifest(changed)

    def unsafe_path() -> None:
        changed = copy.deepcopy(manifest)
        changed["files"][0]["package_path"] = "../gate_a_hardware_adapter.py"
        adapter.validate_bundle_manifest(changed)

    def extra_top_level() -> None:
        changed = copy.deepcopy(manifest)
        changed["extra"] = True
        adapter.validate_bundle_manifest(changed)

    for name, func in (
        ("manifest_execution_bundle_sha256_changed", lambda: changed_top(manifest, "execution_bundle_sha256", "0" * 64)),
        ("manifest_deterministic_archive_sha256_changed", lambda: changed_top(manifest, "deterministic_archive_sha256", "0" * 64)),
        ("manifest_per_file_sha256_changed", mutate_file("sha256", "0" * 64)),
        ("manifest_byte_size_changed", mutate_file("byte_size", 1)),
        ("manifest_source_repository_path_changed", mutate_file("source_repository_path", "THOUGHT/changed.py")),
        ("manifest_package_path_changed", mutate_file("package_path", "adapter/changed.py")),
        ("manifest_role_changed", mutate_file("role", "changed_role")),
        ("manifest_git_blob_sha1_changed", mutate_file("git_blob_sha1", "0" * 40)),
        ("manifest_git_mode_changed", mutate_file("git_mode", "100755")),
        ("manifest_missing_file", remove_file),
        ("manifest_extra_file", add_file),
        ("manifest_duplicate_package_path", duplicate_path),
        ("manifest_case_collision", case_collision),
        ("manifest_unsafe_relative_path", unsafe_path),
        ("manifest_symlink_mode", mutate_file("git_mode", "120000")),
        ("manifest_submodule_mode", mutate_file("git_mode", "160000")),
        ("manifest_extra_top_level_property", extra_top_level),
    ):
        cases.append(assert_rejects(name, func))
    return {"status": "MANIFEST_MUTATION_TESTS_PASS", "negative_tests": len(cases), "cases": cases}


def changed_top(source: dict[str, Any], key: str, value: Any) -> None:
    changed = copy.deepcopy(source)
    changed[key] = value
    adapter.validate_bundle_manifest(changed)


def authority_mutation_tests(manifest: dict[str, Any]) -> dict[str, Any]:
    reviewed_head = "1234567890abcdef1234567890abcdef12345678"
    review_id = 4618767711
    cases: list[str] = []
    equivalence_result: dict[str, Any] | None = None

    with tempfile.TemporaryDirectory(prefix="gate_a_authority_mutations_") as tmp:
        path = Path(tmp) / AUTHORITY_NAME

        def write_authority(value: dict[str, Any]) -> str:
            path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
            return sha256_file(path)

        valid = authority_template(manifest, reviewed_head, review_id)
        digest = write_authority(valid)
        host_result, runner_result = validate_authority_both(path, digest, manifest, reviewed_head, review_id)
        require(host_result == runner_result, "host/target authority validation result mismatch")
        equivalence_result = host_result

        def reject_both(name: str, changed: dict[str, Any], expected_head: str = reviewed_head, expected_review_id: int = review_id, digest_override: str | None = None) -> str:
            actual_digest = write_authority(changed)
            use_digest = digest_override or actual_digest
            def host_case() -> None:
                validate_authority_host(path, use_digest, manifest, expected_head, expected_review_id)
            def runner_case() -> None:
                gate_a_target_runner.validate_authority(path, use_digest, target_args(manifest, expected_head, expected_review_id))
            assert_rejects(name + ":host", host_case)
            assert_rejects(name + ":target", runner_case)
            return name

        def mutate(name: str, mutator: Callable[[dict[str, Any]], None], expected_head: str = reviewed_head, expected_review_id: int = review_id) -> None:
            changed = copy.deepcopy(valid)
            mutator(changed)
            cases.append(reject_both(name, changed, expected_head, expected_review_id))

        mutate("authority_extra_top_level_field", lambda v: v.__setitem__("extra", True))
        mutate("authority_missing_top_level_field", lambda v: v.pop("target"))
        mutate("authority_extra_state_field", lambda v: v["authority_state"].__setitem__("extra", True))
        mutate("authority_missing_state_field", lambda v: v["authority_state"].pop("hardware_ran"))
        mutate("authority_wrong_schema_id", lambda v: v.__setitem__("schema_id", "WRONG"))
        mutate("authority_wrong_reviewed_adapter_head", lambda v: v.__setitem__("reviewed_adapter_head", "0" * 40))
        mutate("authority_wrong_independent_review_id", lambda v: v.__setitem__("independent_review_id", review_id + 1))
        mutate("authority_review_id_zero", lambda v: v.__setitem__("independent_review_id", 0), expected_review_id=0)
        mutate("authority_wrong_bundle_digest", lambda v: v.__setitem__("execution_bundle_sha256", "0" * 64))
        mutate("authority_wrong_host_adapter_blob", lambda v: v.__setitem__("host_adapter_git_blob_sha1", "0" * 40))
        mutate("authority_wrong_target_runner_blob", lambda v: v.__setitem__("target_runner_git_blob_sha1", "0" * 40))
        mutate("authority_wrong_target_worker_blob", lambda v: v.__setitem__("target_worker_git_blob_sha1", "0" * 40))
        mutate("authority_wrong_schedule_digest", lambda v: v.__setitem__("schedule_sha256", "0" * 64))
        mutate("authority_wrong_namespace_digest", lambda v: v.__setitem__("target_namespace_sha256", "0" * 64))
        mutate("authority_wrong_target_identity_digest", lambda v: v.__setitem__("target_identity_sha256", "0" * 64))
        mutate("authority_wrong_target", lambda v: v.__setitem__("target", "root@127.0.0.1"))
        mutate("authority_wrong_remote_execution_root", lambda v: v.__setitem__("remote_execution_root", "/root/wrong"))
        mutate("authority_wrong_remote_output_root", lambda v: v.__setitem__("remote_output_root", "/root/wrong/evidence"))
        mutate("authority_maximum_execution_count_gt_one", lambda v: v.__setitem__("maximum_execution_count", 2))
        mutate("authority_consumed_true", lambda v: v.__setitem__("consumed", True))
        mutate("authority_project_owner_approved_false", lambda v: v.__setitem__("project_owner_approved", False))
        mutate("authority_artifact_created_false", lambda v: v["authority_state"].__setitem__("authorization_artifact_created", False))
        mutate("authority_engineering_smoke_authorized_false", lambda v: v["authority_state"].__setitem__("engineering_smoke_authorized", False))
        mutate("authority_hardware_ran_true", lambda v: v["authority_state"].__setitem__("hardware_ran", True))
        mutate("authority_automatic_retry_true", lambda v: v["authority_state"].__setitem__("automatic_retry", True))
        mutate("authority_calibration_authorized_true", lambda v: v["authority_state"].__setitem__("calibration_authorized", True))
        mutate("authority_scientific_acquisition_authorized_true", lambda v: v["authority_state"].__setitem__("scientific_acquisition_authorized", True))
        mutate("authority_restoration_authorized_true", lambda v: v["authority_state"].__setitem__("restoration_authorized", True))
        mutate("authority_target_coupling_authorized_true", lambda v: v["authority_state"].__setitem__("target_coupling_authorized", True))
        mutate("authority_small_wall_authorized_true", lambda v: v["authority_state"].__setitem__("small_wall_authorized", True))
        cases.append(reject_both("authority_file_sha256_mismatch", valid, digest_override="0" * 64))

    return {
        "status": "AUTHORITY_MUTATION_TESTS_PASS",
        "negative_tests": len(cases),
        "host_target_equivalence": equivalence_result,
        "cases": cases,
    }


def other_mutation_tests(ctx: adapter.AdapterContext, manifest: dict[str, Any]) -> dict[str, Any]:
    cases: list[str] = []

    def wrong_schedule() -> None:
        changed = copy.deepcopy(ctx.schedule)
        changed["slot_sequence"][6] = "I"
        adapter.validate_schedule(changed)

    def wrong_target() -> None:
        changed = copy.deepcopy(ctx.schedule)
        changed["target"]["ssh_target"] = "root@127.0.0.1"
        adapter.validate_schedule(changed)

    def wrong_namespace() -> None:
        changed = copy.deepcopy(ctx.namespace)
        changed["binding_sha256"] = "0" * 64
        adapter.validate_namespace(changed)

    def off_or_sham_drive_mutation() -> None:
        changed = copy.deepcopy(ctx.schedule)
        changed["slot_definitions"]["D0"]["executed"]["drive_on"] = True
        adapter.validate_schedule(changed)

    def step_sender_epoch_mutation() -> None:
        changed = copy.deepcopy(ctx.schedule)
        changed["slot_definitions"]["S0E"]["executed"]["sender_epoch_id"] = "gate-a:step:epoch1"
        adapter.validate_schedule(changed)

    def extra_namespace_property() -> None:
        changed = copy.deepcopy(ctx.namespace)
        changed["extra"] = True
        adapter.validate_namespace(changed)

    def worktree_byte_mutation_behavior() -> None:
        target = HERE / "gate_a_worker.c"
        original = target.read_bytes()
        try:
            target.write_bytes(original + b"\n/* mutation */\n")
            bundle.assert_clean_for_paths([target])
        finally:
            target.write_bytes(original)

    def index_byte_mutation_detection() -> None:
        target = HERE / "gate_a_worker.c"
        original = target.read_bytes()
        mutated_blob = run(["git", "hash-object", "-w", "--stdin"], input_text=(original + b"\n/* staged mutation */\n").decode("utf-8")).stdout.strip()
        try:
            run(["git", "update-index", "--cacheinfo", "100644", mutated_blob, git_rel(target)], cwd=bundle.repo_root())
            changed = bundle.render_manifest(":")
            require(changed == manifest, "index mutation changed bundle identity")
        finally:
            run(["git", "add", git_rel(target)], cwd=bundle.repo_root())

    def existing_output_root_rejection() -> None:
        with tempfile.TemporaryDirectory(prefix="gate_a_existing_output_") as tmp:
            args = target_args(manifest, "1234567890abcdef1234567890abcdef12345678", 4618767711, tmp)
            gate_a_target_runner.execute_authorized(args)

    def cleanup_without_receipt() -> None:
        args = type("Args", (), {"copy_back_receipt": None})()
        gate_a_target_runner.cleanup_after_verified_copy(args)

    for name, func in (
        ("schedule_slot_sequence_rejection", wrong_schedule),
        ("schedule_target_rejection", wrong_target),
        ("namespace_digest_rejection", wrong_namespace),
        ("slot_drive_mutation_rejection", off_or_sham_drive_mutation),
        ("step_sender_epoch_mutation_rejection", step_sender_epoch_mutation),
        ("namespace_extra_property_rejection", extra_namespace_property),
        ("worktree_byte_mutation_behavior", worktree_byte_mutation_behavior),
        ("index_byte_mutation_detection", index_byte_mutation_detection),
        ("existing_output_root_rejection", existing_output_root_rejection),
        ("cleanup_without_copy_back_receipt_rejection", cleanup_without_receipt),
    ):
        cases.append(assert_rejects(name, func))

    for flag in (
        "--deploy",
        "--connect",
        "--start-sender",
        "--start-capture",
        "--write-control",
        "--cleanup-after-verified-copy",
    ):
        proc = run([sys.executable, str(HERE / "gate_a_hardware_adapter.py"), flag], check=False)
        require(proc.returncode != 0, f"adapter bypass accepted: {flag}")
        cases.append(f"authority_bypass_rejection:{flag}")
    return {"status": "OTHER_MUTATION_TESTS_PASS", "negative_tests": len(cases), "cases": cases}


def mutation_tests() -> dict[str, Any]:
    ctx = adapter.load_context()
    manifest = copy.deepcopy(ctx.manifest)
    manifest_result = manifest_mutation_tests(manifest)
    authority_result = authority_mutation_tests(manifest)
    other_result = other_mutation_tests(ctx, manifest)
    total = manifest_result["negative_tests"] + authority_result["negative_tests"] + other_result["negative_tests"]
    return {
        "status": "MUTATION_TESTS_PASS",
        "negative_tests": total,
        "manifest": manifest_result,
        "authority": authority_result,
        "other": other_result,
    }


def validate_records() -> dict[str, Any]:
    manifest = load(MANIFEST)
    result = load(RESULT)
    candidate = load(CANDIDATE_V2)
    schema = load(AUTHORITY_SCHEMA)
    contract = load(CONTRACT)
    require(set(manifest) == bundle.MANIFEST_KEYS, "manifest record key set mismatch")
    bundle.validate_manifest(manifest)
    validate_schema_closed(schema)
    validate_contract(contract)
    require(set(candidate) == {
        "schema_id",
        "status",
        "base_main_commit",
        "reviewed_gate_a_plan_head",
        "gate_a_plan_review",
        "plan_reviewed",
        "adapter_implemented",
        "hosted_adapter_qualification_complete",
        "target_adapter_qualification_complete",
        "execution_bundle_ready",
        "execution_bundle_target_qualified",
        "project_owner_execution_approval_recorded",
        "authorization_artifact_created",
        "engineering_smoke_authorized",
        "hardware_ran",
        "schedule_sha256",
        "target_namespace_sha256",
        "target_identity_stdout_sha256",
        "host_adapter_git_blob_sha1",
        "target_runner_git_blob_sha1",
        "target_worker_git_blob_sha1",
        "execution_bundle_sha256",
        "deterministic_archive_sha256",
        "bundle_manifest_sha256",
        "next_boundary",
        "authority_false_state",
    }, "candidate V2 key set mismatch")
    require(set(result) == {
        "schema_id",
        "status",
        "adapter_implementation_complete",
        "hosted_nonexecuting_qualification_complete",
        "target_nonexecuting_qualification_complete",
        "execution_bundle_ready",
        "execution_bundle_target_qualified",
        "authority_artifact_created",
        "engineering_smoke_authorized",
        "hardware_ran",
        "no_target_connection_occurred",
        "no_ssh_occurred",
        "no_sender_ran",
        "no_receiver_capture_ran",
        "no_control_write_occurred",
        "next_boundary",
        "execution_bundle_sha256",
        "deterministic_archive_sha256",
        "bundle_manifest_sha256",
        "authority_false_state",
    }, "qualification result key set mismatch")
    manifest_digest = committed_sha256(MANIFEST)
    result_digest = committed_sha256(RESULT)
    candidate_digest = committed_sha256(CANDIDATE_V2)
    require(result["execution_bundle_sha256"] == manifest["execution_bundle_sha256"], "result bundle digest mismatch")
    require(result["deterministic_archive_sha256"] == manifest["deterministic_archive_sha256"], "result archive digest mismatch")
    require(result["bundle_manifest_sha256"] == manifest_digest, "result manifest digest mismatch")
    require(candidate["execution_bundle_sha256"] == manifest["execution_bundle_sha256"], "candidate bundle digest mismatch")
    require(candidate["deterministic_archive_sha256"] == manifest["deterministic_archive_sha256"], "candidate archive digest mismatch")
    require(candidate["bundle_manifest_sha256"] == manifest_digest, "candidate manifest digest mismatch")
    files_by_role = {entry["role"]: entry for entry in manifest["files"]}
    require(candidate["host_adapter_git_blob_sha1"] == files_by_role["host_adapter"]["git_blob_sha1"], "candidate adapter blob mismatch")
    require(candidate["target_runner_git_blob_sha1"] == files_by_role["target_runner"]["git_blob_sha1"], "candidate runner blob mismatch")
    require(candidate["target_worker_git_blob_sha1"] == files_by_role["target_worker"]["git_blob_sha1"], "candidate worker blob mismatch")
    require(result["status"] == "TARGET_NONEXECUTING_QUALIFICATION_REQUIRED", "result boundary mismatch")
    require(candidate["status"] == "CANDIDATE__BLOCKED_PENDING_TARGET_NONEXECUTING_QUALIFICATION", "candidate status mismatch")
    for key, value in result["authority_false_state"].items():
        require(value is False, f"result authority flag must be false: {key}")
    require(not list(HERE.rglob(AUTHORITY_NAME)), "execution authority artifact must not exist")
    return {
        "status": "RECORDS_VALID",
        "manifest_sha256": manifest_digest,
        "result_sha256": result_digest,
        "candidate_v2_sha256": candidate_digest,
        "execution_bundle_sha256": manifest["execution_bundle_sha256"],
    }


def main() -> int:
    context = adapter.load_context()
    no_drive = adapter.qualify_no_drive(context)
    require(no_drive["transport"] == "NO_DRIVE", "adapter transport not no-drive")
    treeish = os.environ.get("GATE_A_BUNDLE_TREEISH", "HEAD")
    manifest_a = bundle.render_manifest(treeish)
    manifest_b = bundle.render_manifest(treeish)
    require(manifest_a == manifest_b, "bundle double-build mismatch")
    records = validate_records()
    mutations = mutation_tests()
    scan = static_forbidden_surface_scan()
    runtime_path = HERE.parents[2] / "runtime" / "explicit_slot_runtime.py"
    require("SOFTWARE_ENTRY_ONLY_AUTHORITY: real hardware execution is not authorized" in runtime_path.read_text(encoding="utf-8"), "runtime hardware rejection marker missing")
    result = {
        "status": "GATE_A_ADAPTER_QUALIFICATION_PASS",
        "null_baseline": "NO_DRIVE_ZERO_COUNT_BASELINE",
        "adapter_no_drive": no_drive["status"],
        "bundle_double_build_equivalence": True,
        "mutation_tests": mutations,
        "forbidden_surface_scan": scan,
        "records": records,
        "authority_artifact_absent": True,
        "target_connection_count": 0,
        "sender_start_count": 0,
        "control_write_count": 0,
        "msr_access_count": 0,
        "hardware_execution_count": 0,
    }
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (VerifyError, adapter.AdapterError, bundle.BundleError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        print(f"verify_gate_a_adapter_qualification: {exc}", file=sys.stderr)
        raise SystemExit(1)
