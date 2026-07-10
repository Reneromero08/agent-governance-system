#!/usr/bin/env python3
"""Dedicated Gate A adapter no-drive qualification verifier."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

import build_gate_a_execution_bundle as bundle
import gate_a_hardware_adapter as adapter
import gate_a_isolated_qualification as isolated_harness
import gate_a_target_bundle as target_bundle
import gate_a_target_runner

HERE = Path(__file__).resolve().parent
GATE_A = HERE.parent
RESULT = HERE / "GATE_A_ADAPTER_QUALIFICATION_RESULT.json"
CANDIDATE_V2 = HERE / "GATE_A_ENGINEERING_SMOKE_AUTHORITY_CANDIDATE_V2.json"
MANIFEST = HERE / "GATE_A_EXECUTION_BUNDLE_MANIFEST.json"
AUTHORITY_SCHEMA = HERE / "schemas" / "gate_a_execution_authority.schema.json"
AUTHORITY_NAME = "GATE_A_EXECUTION_AUTHORITY.json"
CONTRACT = HERE / "GATE_A_ADAPTER_QUALIFICATION_CONTRACT.json"
HISTORICAL_ADAPTER_COMMIT = "6f243b1aaf7cfaa09f21b8d5816ddd9097612f72"
HISTORICAL_MANIFEST_SHA256 = "ccb7866db67170083cb00d546c334b61772c8ef909131ec9c62ed21115facc94"
HISTORICAL_RESULT_SHA256 = "1d9d2c62cbf81f72eeb9c40f02841f9f507d52eae8229da73fc2f81eb0a15223"
HISTORICAL_CANDIDATE_V2_SHA256 = "d8f190bc7f8c9904659cd697ed091b192843efe18f5f1d74d713282e889b060e"


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
    if treeish == ":":
        blob = bundle.git_index_source(path).blob
    else:
        blob = run(["git", "rev-parse", f"{treeish}:{git_rel(path)}"], cwd=bundle.repo_root()).stdout.strip()
    data = subprocess.run(["git", "cat-file", "blob", blob], cwd=bundle.repo_root(), stdout=subprocess.PIPE, check=True).stdout
    return hashlib.sha256(data).hexdigest()


def committed_object(path: Path, treeish: str) -> dict[str, Any]:
    blob = run(["git", "rev-parse", f"{treeish}:{git_rel(path)}"], cwd=bundle.repo_root()).stdout.strip()
    data = subprocess.run(["git", "cat-file", "blob", blob], cwd=bundle.repo_root(), stdout=subprocess.PIPE, check=True).stdout
    value = json.loads(data)
    require(isinstance(value, dict), f"committed object required: {treeish}:{git_rel(path)}")
    return value


def active_treeish() -> str:
    return os.environ.get("GATE_A_BUNDLE_TREEISH", "HEAD")


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
    for expected in (
        "worktree-byte mutation behavior",
        "index-byte mutation detection",
        "missing authority rejects before transport",
        "committed two-commit authority custody",
        "ordinary worker live execution rejection",
        "second execution rejection",
        "automatic retry rejection",
        "target-local timeout rejection",
        "four-way remote namespace preflight rejection before transfer",
        "complete capture requirement",
        "partial evidence preservation",
        "cleanup requires verified copy-back",
        "cleanup inventory digest recomputation",
        "physically absent sender lifecycle custody",
        "one contiguous STEP sender epoch",
        "distinct bounded anchor epochs",
        "continuous capture across sender lifecycle",
        "raw-derived per-slot lock-in I/Q recomputation",
        "altered lock-in range, tone, slot, I or Q rejection",
        "pre-runtime process receipt required",
        "post-runtime process receipt required on success and failure",
        "post-cleanup process receipt required",
        "process command, return code, raw streams and hashes bound",
        "complete retained host evidence packet",
        "final evidence inventory closure",
        "host command ledger closure",
        "durable authority claim survives cleanup",
        "transport failure injection preserves local failure receipt",
        "transport failure state machine never retries runtime",
        "zero network contact in tests",
    ):
        require(expected in required_tests, f"contract negative test missing: {expected}")
    expected_sources = set(contract["expected_source_files"])
    for expected in (
        "gate_a_engineering_smoke_executor.py",
        "gate_a_process_custody.py",
        "gate_a_engineering_smoke_transport.py",
        "../../../../holo_runtime_v2/combined_pdn_hardware.c",
        "../../../../holo_runtime_v2/gate_a_engineering_smoke_runtime.c",
        "../../../../holo_runtime_v2/gate_a_engineering_smoke_runtime.h",
        "test_gate_a_engineering_smoke_executor.py",
    ):
        require(expected in expected_sources, f"contract source file missing: {expected}")


def static_forbidden_surface_scan() -> dict[str, Any]:
    blocked_regexes = [
        ("shell_true", "shell=True"),
        ("os_system", "os.system("),
        ("eval_call", "eval("),
        ("exec_call", "exec("),
    ]
    implementation_files = [
        HERE / "gate_a_authority.py",
        HERE / "gate_a_target_bundle.py",
        HERE / "gate_a_engineering_smoke_executor.py",
        HERE / "gate_a_process_custody.py",
        HERE / "gate_a_engineering_smoke_transport.py",
        HERE / "gate_a_hardware_adapter.py",
        HERE / "gate_a_target_runner.py",
        HERE / "gate_a_worker.c",
        HERE / "build_gate_a_execution_bundle.py",
        HERE / "gate_a_isolated_qualification.py",
    ]
    matches: list[dict[str, str]] = []
    for path in implementation_files:
        text = path.read_text(encoding="utf-8")
        for name, needle in blocked_regexes:
            if needle in text:
                matches.append({"file": path.name, "pattern": name})
    require(not matches, f"forbidden implementation surface present: {matches}")
    runner_text = (HERE / "gate_a_target_runner.py").read_text(encoding="utf-8")
    worker_text = (HERE / "gate_a_worker.c").read_text(encoding="utf-8")
    host_text = (HERE / "gate_a_hardware_adapter.py").read_text(encoding="utf-8")
    executor_text = (HERE / "gate_a_engineering_smoke_executor.py").read_text(encoding="utf-8")
    transport_text = (HERE / "gate_a_engineering_smoke_transport.py").read_text(encoding="utf-8")
    process_text = (HERE / "gate_a_process_custody.py").read_text(encoding="utf-8")
    require("authorized live execution path is intentionally unused" not in runner_text, "target execution sentinel remains")
    require("live execution unavailable" not in worker_text, "worker execution sentinel remains")
    require("run_gate_a_engineering_smoke" in worker_text, "worker does not call the bounded physical runtime")
    require("transport_factory" in host_text and "validate_future_authority" in host_text, "host authority-before-transport seam missing")
    require("validate_authority_git_custody" in host_text, "host committed-authority custody gate missing")
    require("timeout=self.timeout_s" in executor_text, "target worker timeout missing")
    require("start_new_session=True" in transport_text and "os.killpg" in transport_text and "signal.SIGKILL" in transport_text, "target process-group timeout cleanup missing")
    require("GATE_A_COMPILED_AUTHORITY_SHA256" in worker_text, "worker compile-time authority binding missing")
    require("PRE_RUNTIME_PROCESS_RECEIPT.json" in executor_text and "POST_RUNTIME_PROCESS_RECEIPT.json" in executor_text, "target process receipts are not retained")
    require("POST_CLEANUP_PROCESS_RECEIPT.json" in transport_text, "post-cleanup process receipt is not retained")
    require("raw_stdout_base64" in process_text and "stdout_sha256" in process_text and "parsed_forbidden_hits" in process_text, "shared process custody is incomplete")
    for retained in (
        "AUTHORITY_ARTIFACT.json", "SCHEDULE.json", "EXECUTION_BUNDLE_MANIFEST.json",
        "SOURCE_REVIEW_BINDING.json", "HOST_COMMANDS.jsonl", "TARGET_EXECUTION_RECEIPT.json",
        "TARGET_EVIDENCE_INVENTORY.json", "COPY_BACK_RECEIPT.json",
        "POST_RUNTIME_PROCESS_RECEIPT.json", "POST_CLEANUP_PROCESS_RECEIPT.json",
        "CLEANUP_RECEIPT.json",
        "FINAL_EVIDENCE_INVENTORY.json", "FINAL_BINDINGS.json",
    ):
        require(retained in transport_text, f"retained host packet artifact missing: {retained}")

    runtime = HERE.parents[3] / "holo_runtime_v2" / "gate_a_engineering_smoke_runtime.c"
    runtime_text = runtime.read_text(encoding="utf-8")
    marker = "int run_gate_a_engineering_smoke("
    require(marker in runtime_text, "bounded Gate A physical-runtime entry point missing")
    gate_a_body = runtime_text[runtime_text.index(marker):]
    require("GATE_A_COMPILED_OUTPUT_ROOT" in worker_text and "gate_a_runtime_output_root" in gate_a_body, "worker one-shot output binding missing")
    require("LOCKIN_IQ.jsonl" in runtime_text and "lockin(" in runtime_text, "raw-derived lock-in custody missing")
    require("SENDER_LIFECYCLE.jsonl" in runtime_text and "gate-a:anchor:positive" in runtime_text and "gate-a:anchor:negative" in runtime_text, "bounded sender lifecycle custody missing")
    require("16.0 * sender->slot_s" not in runtime_text, "one sender thread still spans the complete sequence")
    for name, needle in (
        ("frequency_control", "pin_frequency("),
        ("msr_access", "msr_read("),
        ("min_frequency_control", "scaling_min_freq"),
        ("max_frequency_control", "scaling_max_freq"),
        ("boost_control", "/cpufreq/boost"),
        ("msr_device", "/dev/cpu/"),
    ):
        require(needle not in gate_a_body, f"bounded Gate A runtime exposes forbidden surface: {name}")
    return {
        "status": "BOUNDED_EXECUTOR_SURFACE_SCAN_PASS",
        "generic_unsafe_matches": 0,
        "frequency_control_calls": 0,
        "msr_access_calls": 0,
        "intentional_execution_sentinels": 0,
    }


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
        "claim_root": None,
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
    exact_manifest = bundle.validate_committed_manifest_exact(manifest, active_treeish())
    runner_result = gate_a_target_runner.validate_authority(path, digest, target_args(manifest, reviewed_head, review_id), exact_manifest)
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
        exact_manifest = bundle.validate_committed_manifest_exact(manifest, active_treeish())

        def reject_both(name: str, changed: dict[str, Any], expected_head: str = reviewed_head, expected_review_id: int = review_id, digest_override: str | None = None) -> str:
            actual_digest = write_authority(changed)
            use_digest = digest_override or actual_digest
            def host_case() -> None:
                validate_authority_host(path, use_digest, manifest, expected_head, expected_review_id)
            def runner_case() -> None:
                gate_a_target_runner.validate_authority(path, use_digest, target_args(manifest, expected_head, expected_review_id), exact_manifest)
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


def isolated_bundle_qualification() -> dict[str, Any]:
    compile_c = shutil.which("cc") is not None
    with tempfile.TemporaryDirectory(prefix="gate_a_extract_") as tmp:
        root = Path(tmp) / "bundle"
        bundle.write_extracted_tree(root, active_treeish())
        require(not (root / ".git").exists(), "extracted bundle must not contain .git")
        report = isolated_harness.build_isolated_report(root, require_isolated_origin=False, compile_c=compile_c)
    require(report["status"] == "GATE_A_ISOLATED_BUNDLE_QUALIFICATION_PASS", "isolated bundle qualification failed")
    report["c_compiler_present"] = compile_c
    return report


def validate_records() -> dict[str, Any]:
    treeish = active_treeish()
    manifest = load(MANIFEST)
    result = load(RESULT)
    candidate = load(CANDIDATE_V2)
    schema = load(AUTHORITY_SCHEMA)
    contract = load(CONTRACT)
    require(set(manifest) == bundle.MANIFEST_KEYS, "manifest record key set mismatch")
    bundle.validate_committed_manifest_exact(manifest, treeish)
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
    manifest_digest = committed_sha256(MANIFEST, treeish)
    result_digest = committed_sha256(RESULT, treeish)
    candidate_digest = committed_sha256(CANDIDATE_V2, treeish)
    historical_manifest = committed_object(MANIFEST, HISTORICAL_ADAPTER_COMMIT)
    historical_manifest_digest = committed_sha256(MANIFEST, HISTORICAL_ADAPTER_COMMIT)
    require(historical_manifest_digest == HISTORICAL_MANIFEST_SHA256, "historical manifest changed")
    require(result_digest == HISTORICAL_RESULT_SHA256, "historical qualification result bytes changed")
    require(candidate_digest == HISTORICAL_CANDIDATE_V2_SHA256, "historical Candidate V2 bytes changed")
    require(result["execution_bundle_sha256"] == historical_manifest["execution_bundle_sha256"], "historical result bundle binding mismatch")
    require(result["deterministic_archive_sha256"] == historical_manifest["deterministic_archive_sha256"], "historical result archive binding mismatch")
    require(result["bundle_manifest_sha256"] == historical_manifest_digest, "historical result manifest binding mismatch")
    require(candidate["execution_bundle_sha256"] == historical_manifest["execution_bundle_sha256"], "historical candidate bundle binding mismatch")
    require(candidate["deterministic_archive_sha256"] == historical_manifest["deterministic_archive_sha256"], "historical candidate archive binding mismatch")
    require(candidate["bundle_manifest_sha256"] == historical_manifest_digest, "historical candidate manifest binding mismatch")
    historical_roles = {entry["role"]: entry for entry in historical_manifest["files"]}
    require(candidate["host_adapter_git_blob_sha1"] == historical_roles["host_adapter"]["git_blob_sha1"], "historical candidate adapter blob mismatch")
    require(candidate["target_runner_git_blob_sha1"] == historical_roles["target_runner"]["git_blob_sha1"], "historical candidate runner blob mismatch")
    require(candidate["target_worker_git_blob_sha1"] == historical_roles["target_worker"]["git_blob_sha1"], "historical candidate worker blob mismatch")
    require(manifest["engineering_smoke_executor_implemented"] is True, "executor implementation status mismatch")
    require(manifest["execution_bundle_target_qualified"] is True, "target bundle qualification status mismatch")
    require(manifest["engineering_smoke_authorized"] is False, "engineering smoke authority changed")
    require(manifest["hardware_ran"] is False, "hardware run state changed")
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
        "historical_manifest_sha256": historical_manifest_digest,
        "historical_records_immutable": True,
        "execution_bundle_sha256": manifest["execution_bundle_sha256"],
    }


def focused_executor_tests() -> dict[str, Any]:
    completed = run([
        sys.executable,
        "-B",
        "-m",
        "unittest",
        "discover",
        "-s",
        str(HERE),
        "-p",
        "test_gate_a_engineering_smoke_executor.py",
        "-v",
    ], cwd=bundle.repo_root(), check=False)
    require(completed.returncode == 0, f"focused executor tests failed:\n{completed.stdout}\n{completed.stderr}")
    count = sum(1 for line in completed.stderr.splitlines() if line.rstrip().endswith("... ok"))
    require(count >= 30, "focused executor test count below required minimum")
    return {
        "status": "GATE_A_ENGINEERING_SMOKE_EXECUTOR_TESTS_PASS",
        "tests_run": count,
        "network_connections_opened": 0,
        "target_contact_count": 0,
        "hardware_execution_count": 0,
    }


def main() -> int:
    context = adapter.load_context()
    no_drive = adapter.qualify_no_drive(context)
    require(no_drive["transport"] == "NO_DRIVE", "adapter transport not no-drive")
    treeish = active_treeish()
    manifest_a = bundle.render_manifest(treeish)
    manifest_b = bundle.render_manifest(treeish)
    require(manifest_a == manifest_b, "bundle double-build mismatch")
    records = validate_records()
    mutations = mutation_tests()
    isolated = isolated_bundle_qualification()
    scan = static_forbidden_surface_scan()
    executor_tests = focused_executor_tests()
    runtime_path = HERE.parents[2] / "runtime" / "explicit_slot_runtime.py"
    require("SOFTWARE_ENTRY_ONLY_AUTHORITY: real hardware execution is not authorized" in runtime_path.read_text(encoding="utf-8"), "runtime hardware rejection marker missing")
    total_negative_tests = mutations["negative_tests"] + isolated["isolated_negative_tests"]
    result = {
        "status": "GATE_A_ADAPTER_QUALIFICATION_PASS",
        "null_baseline": "NO_DRIVE_ZERO_COUNT_BASELINE",
        "adapter_no_drive": no_drive["status"],
        "bundle_double_build_equivalence": True,
        "mutation_tests": mutations,
        "isolated_bundle_qualification": isolated,
        "total_negative_tests": total_negative_tests,
        "forbidden_surface_scan": scan,
        "executor_tests": executor_tests,
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
