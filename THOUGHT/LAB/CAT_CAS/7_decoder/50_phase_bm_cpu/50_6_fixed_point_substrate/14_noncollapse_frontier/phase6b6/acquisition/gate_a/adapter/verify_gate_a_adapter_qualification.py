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


def mutation_tests() -> dict[str, Any]:
    ctx = adapter.load_context()
    manifest = copy.deepcopy(ctx.manifest)
    negative: list[str] = []

    def wrong_reviewed_head() -> None:
        changed = copy.deepcopy(manifest)
        changed["reviewed_gate_a_plan_head"] = "0" * 40
        adapter.validate_bundle_manifest(changed)

    def wrong_review_id() -> None:
        changed = copy.deepcopy(manifest)
        changed["gate_a_plan_review"] = 1
        adapter.validate_bundle_manifest(changed)

    def wrong_bundle_digest() -> None:
        changed = copy.deepcopy(manifest)
        changed["execution_bundle_sha256"] = "0" * 64
        bundle.validate_manifest(changed)

    def wrong_blob(role: str) -> Callable[[], None]:
        def inner() -> None:
            changed = copy.deepcopy(manifest)
            for entry in changed["files"]:
                if entry["role"] == role:
                    entry["git_blob_sha1"] = "0" * 40
            bundle.validate_manifest(changed)
        return inner

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

    def extra_property() -> None:
        changed = copy.deepcopy(ctx.namespace)
        changed["extra"] = True
        adapter.validate_namespace(changed)

    def case_collision() -> None:
        changed = copy.deepcopy(manifest)
        duplicate = copy.deepcopy(changed["files"][0])
        duplicate["package_path"] = changed["files"][0]["package_path"].upper()
        changed["files"].append(duplicate)
        adapter.validate_bundle_manifest(changed)

    def symlink_rejection() -> None:
        changed = copy.deepcopy(manifest)
        changed["files"][0]["git_mode"] = "120000"
        adapter.validate_bundle_manifest(changed)

    def git_mode_mutation() -> None:
        changed = copy.deepcopy(manifest)
        changed["files"][0]["git_mode"] = "100755"
        adapter.validate_bundle_manifest(changed)

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
            args = type("Args", (), {
                "authority_artifact": None,
                "authority_sha256": None,
                "execution_bundle_sha256": manifest["execution_bundle_sha256"],
                "source_head": "0" * 40,
                "schedule_sha256": adapter.SCHEDULE_SHA256,
                "target": "root@192.168.137.100",
                "namespace_sha256": adapter.NAMESPACE_SHA256,
                "output_root": tmp,
            })()
            import gate_a_target_runner
            gate_a_target_runner.execute_authorized(args)

    def cleanup_without_receipt() -> None:
        args = type("Args", (), {"copy_back_receipt": None})()
        import gate_a_target_runner
        gate_a_target_runner.cleanup_after_verified_copy(args)

    for name, func in (
        ("wrong_reviewed_head_rejection", wrong_reviewed_head),
        ("wrong_review_id_rejection", wrong_review_id),
        ("wrong_bundle_digest_rejection", wrong_bundle_digest),
        ("wrong_adapter_blob_rejection", wrong_blob("host_adapter")),
        ("wrong_target_runner_blob_rejection", wrong_blob("target_runner")),
        ("wrong_worker_blob_rejection", wrong_blob("target_worker")),
        ("wrong_schedule_rejection", wrong_schedule),
        ("wrong_target_rejection", wrong_target),
        ("wrong_namespace_rejection", wrong_namespace),
        ("off_or_sham_drive_mutation_rejection", off_or_sham_drive_mutation),
        ("step_sender_epoch_mutation_rejection", step_sender_epoch_mutation),
        ("extra_property_rejection", extra_property),
        ("case_collision_rejection", case_collision),
        ("symlink_rejection", symlink_rejection),
        ("git_mode_mutation_rejection", git_mode_mutation),
        ("worktree_byte_mutation_behavior", worktree_byte_mutation_behavior),
        ("index_byte_mutation_detection", index_byte_mutation_detection),
        ("existing_output_root_rejection", existing_output_root_rejection),
        ("cleanup_without_copy_back_receipt_rejection", cleanup_without_receipt),
    ):
        negative.append(assert_rejects(name, func))

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
        negative.append(f"authority_bypass_rejection:{flag}")

    authority = {
        "schema_id": "CAT_CAS_PHASE6B6_GATE_A_EXECUTION_AUTHORITY_V1",
        "reviewed_adapter_head": run(["git", "rev-parse", "HEAD"]).stdout.strip(),
        "independent_review_id": 1,
        "execution_bundle_sha256": manifest["execution_bundle_sha256"],
        "host_adapter_git_blob_sha1": manifest["files"][0]["git_blob_sha1"],
        "target_runner_git_blob_sha1": manifest["files"][1]["git_blob_sha1"],
        "target_worker_git_blob_sha1": manifest["files"][2]["git_blob_sha1"],
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

    with tempfile.TemporaryDirectory(prefix="gate_a_authority_mutations_") as tmp:
        path = Path(tmp) / AUTHORITY_NAME
        def write_and_validate(changed: dict[str, Any]) -> None:
            path.write_text(json.dumps(changed, sort_keys=True, indent=2) + "\n", encoding="utf-8")
            changed["_path"] = str(path)
            adapter.validate_future_authority(changed, sha256_file(path), manifest["execution_bundle_sha256"])

        for name, mutate in (
            ("consumed_authority_rejection", lambda v: v.__setitem__("consumed", True)),
            ("execution_count_greater_than_one_rejection", lambda v: v.__setitem__("maximum_execution_count", 2)),
            ("automatic_retry_rejection", lambda v: v["authority_state"].__setitem__("automatic_retry", True)),
            ("calibration_authority_rejection", lambda v: v["authority_state"].__setitem__("calibration_authorized", True)),
            ("scientific_acquisition_authority_rejection", lambda v: v["authority_state"].__setitem__("scientific_acquisition_authorized", True)),
            ("restoration_authority_rejection", lambda v: v["authority_state"].__setitem__("restoration_authorized", True)),
            ("target_coupling_authority_rejection", lambda v: v["authority_state"].__setitem__("target_coupling_authorized", True)),
            ("small_wall_authority_rejection", lambda v: v["authority_state"].__setitem__("small_wall_authorized", True)),
        ):
            def case(mutate: Callable[[dict[str, Any]], None] = mutate) -> None:
                changed = copy.deepcopy(authority)
                mutate(changed)
                write_and_validate(changed)
            negative.append(assert_rejects(name, case))

    expected_negative_count = 33
    require(len(negative) == expected_negative_count, f"negative test count mismatch: {len(negative)}")
    return {"status": "MUTATION_TESTS_PASS", "negative_tests": len(negative), "cases": negative}


def validate_records() -> dict[str, Any]:
    manifest = load(MANIFEST)
    result = load(RESULT)
    candidate = load(CANDIDATE_V2)
    schema = load(AUTHORITY_SCHEMA)
    contract = load(CONTRACT)
    bundle.validate_manifest(manifest)
    validate_schema_closed(schema)
    validate_contract(contract)
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
