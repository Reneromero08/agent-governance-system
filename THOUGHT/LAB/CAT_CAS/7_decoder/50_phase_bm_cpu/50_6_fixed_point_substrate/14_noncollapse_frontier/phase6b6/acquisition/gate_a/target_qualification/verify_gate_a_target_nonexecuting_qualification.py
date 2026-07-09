#!/usr/bin/env python3
"""Verify the committed Gate A target non-executing qualification evidence.

This verifier runs with no network access and never contacts the target. It
validates the owner authorization, the qualification contract, the result and
its closed schema, Candidate V3, and the committed evidence package (final
bindings, copy-back receipt, cleanup receipt, evidence inventory closure, the
copied-back target evidence, exactly-one qualification execution, before/after
tree equality, worker validate-only status, and all-zero hardware counters). It
also proves no execution authority artifact exists and runs mutation tests that
reject every forbidden deviation.
"""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

HERE = Path(__file__).resolve().parent
GATE_A = HERE.parent
ADAPTER = GATE_A / "adapter"
PHASE6B6 = HERE.parents[2]
EVID = PHASE6B6 / "evidence" / "gate_a_target_nonexecuting_qualification_6f243b1a_bundle_abc9e50a"

AUTHORIZATION = HERE / "GATE_A_TARGET_NONEXECUTING_QUALIFICATION_AUTHORIZATION.json"
CONTRACT = HERE / "GATE_A_TARGET_NONEXECUTING_QUALIFICATION_CONTRACT.json"
RESULT = HERE / "GATE_A_TARGET_NONEXECUTING_QUALIFICATION_RESULT.json"
CANDIDATE_V3 = HERE / "GATE_A_ENGINEERING_SMOKE_AUTHORITY_CANDIDATE_V3.json"
SCHEMA = HERE / "schemas" / "gate_a_target_nonexecuting_qualification_result.schema.json"

INTEGRATED_MAIN = "6f243b1aaf7cfaa09f21b8d5816ddd9097612f72"
REVIEWED_HEAD = "653d225594af088dc5a72b1655b8b6192b019fc3"
REVIEW_ID = 4621557139
EXECUTION_BUNDLE = "abc9e50a517d764c553adc5096378992028b29a8f62480a9ae217ebbd5202bba"
DETERMINISTIC_ARCHIVE = "04eaf73336f373865f4e837baca9ff4fe893d3b1b16dd8b8288af1259ff96f9c"
MANIFEST_FILE_SHA = "ccb7866db67170083cb00d546c334b61772c8ef909131ec9c62ed21115facc94"
PREDECESSOR_ID_SHA = "10618a70ceb3413d7507c22254d595d63632bb7ad9243dbe3dc6ebbaf13e19a4"
EXPECTED_HOSTNAME = "catcas"
EXPECTED_ARCH = "x86_64"
EXPECTED_CPU = "AMD Phenom(tm) II X6 1090T Processor"
EXPECTED_OWNER = "Raúl Romero"
PREDECESSOR_RESULT_SHA = "1d9d2c62cbf81f72eeb9c40f02841f9f507d52eae8229da73fc2f81eb0a15223"
PREDECESSOR_CANDIDATE_V2_SHA = "d8f190bc7f8c9904659cd697ed091b192843efe18f5f1d74d713282e889b060e"
PREDECESSOR_DIGEST_MEANING = (
    "PR #36 adapter-package result and Candidate V2 digests, not PR #37 target qualification result "
    "or Candidate V3 committed file hashes"
)
DIGEST_SEMANTICS = EVID / "DIGEST_SEMANTICS.json"

EXPECTED_BLOBS = {
    "gate_a_hardware_adapter.py": "08ea193e341f2d1934fe1955f224b2566c5945e5",
    "gate_a_target_runner.py": "7a17a6a514944888b594d37389e7d6105fbe97c4",
    "gate_a_worker.c": "a1022c0109e0c4406d84f19a9777d8a0008696fb",
    "gate_a_target_bundle.py": "369613f6153c71dddba9aaf1e4086a04519d8a37",
    "gate_a_authority.py": "13d657941e20919281a18c4d31dbd174addcfae3",
    "build_gate_a_execution_bundle.py": "2d67bf6e8ce293ddc12cfe562302d7b00619cff7",
    "verify_gate_a_adapter_qualification.py": "62817afd128fcc2d6a9f25c654fca3174b2a5643",
    "gate_a_isolated_qualification.py": "1dbeb06a6dceb6f1155d447835b3ef265571d19d",
}

FORBIDDEN_COMMAND_SUBSTRINGS = [
    "--probe-only",
    "--execute-authorized",
    "--cleanup-after-verified-copy",
    "rdmsr",
    "wrmsr",
    "cpupower",
    "turbostat",
    "modprobe",
    "insmod",
    "rmmod",
    "apt-get",
    "apt ",
    "combined_pdn_runner",
    "run_combined_campaign",
    "--hardware",
    " git ",
    "git clone",
    "git checkout",
    "git bundle",
]


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


def git_blob(path: Path) -> str:
    return subprocess.run(["git", "hash-object", str(path)], stdout=subprocess.PIPE, check=True, text=True).stdout.strip()


def validate_schema_closed(schema: dict[str, Any]) -> None:
    require(schema["additionalProperties"] is False, "schema top level open")
    require(set(schema["required"]) == set(schema["properties"]), "schema required/properties mismatch")
    for key in ("target", "bindings", "authority_false_state"):
        sub = schema["properties"][key]
        require(sub["additionalProperties"] is False, f"schema {key} open")
        require(set(sub["required"]) == set(sub["properties"]), f"schema {key} required/properties mismatch")


def validate_authorization(auth: dict[str, Any]) -> None:
    require(auth["schema_id"] == "CAT_CAS_PHASE6B6_GATE_A_TARGET_NONEXECUTING_QUALIFICATION_AUTHORIZATION_V1", "authorization schema mismatch")
    require(auth["decision"] == "AUTHORIZED_FOR_GATE_A_TARGET_NONEXECUTING_QUALIFICATION_ONLY", "authorization decision mismatch")
    require(auth["project_owner"] == EXPECTED_OWNER, "authorization owner identity mismatch")
    require(auth["integrated_main"] == INTEGRATED_MAIN, "authorization integrated main mismatch")
    require(auth["maximum_target_qualification_executions"] == 1, "authorization max executions must be 1")
    require(auth["automatic_retry"] is False, "authorization retry must be false")
    for t in ("ssh_authorized", "copy_authorized", "target_filesystem_staging_authorized", "compile_validate_only_authorized", "no_drive_runner_authorized"):
        require(auth[t] is True, f"authorization {t} must be true")
    for f in ("probe_authorized", "engineering_smoke_authorized", "hardware_execution_authorized", "calibration_authorized",
              "scientific_acquisition_authorized", "restoration_authorized", "target_coupling_authorized", "small_wall_authorized",
              "execution_authority_artifact_creation_authorized"):
        require(auth[f] is False, f"authorization {f} must be false (narrow scope)")


def validate_predecessor_adapter_package_digests(digests: dict[str, Any], context: str) -> None:
    require(
        digests["predecessor_adapter_qualification_result_sha256"] == PREDECESSOR_RESULT_SHA,
        f"{context} predecessor adapter result digest mismatch",
    )
    require(
        digests["predecessor_candidate_v2_sha256"] == PREDECESSOR_CANDIDATE_V2_SHA,
        f"{context} predecessor Candidate V2 digest mismatch",
    )
    require(digests["meaning"] == PREDECESSOR_DIGEST_MEANING, f"{context} predecessor digest meaning mismatch")


def validate_contract(contract: dict[str, Any]) -> None:
    require(contract["schema_id"] == "CAT_CAS_PHASE6B6_GATE_A_TARGET_NONEXECUTING_QUALIFICATION_CONTRACT_V1", "contract schema mismatch")
    require(contract["integrated_main"] == INTEGRATED_MAIN, "contract integrated main mismatch")
    require(contract["reviewed_adapter_head"] == REVIEWED_HEAD, "contract reviewed head mismatch")
    require(contract["maximum_target_qualification_executions"] == 1, "contract max executions")
    require(contract["automatic_retry"] is False, "contract retry")
    require(contract["git_on_target_forbidden"] is True, "contract git-on-target must be forbidden")
    blobs = contract["integrated_blobs"]
    require(blobs["host_adapter"] == EXPECTED_BLOBS["gate_a_hardware_adapter.py"], "contract host adapter blob mismatch")
    require(blobs["target_runner"] == EXPECTED_BLOBS["gate_a_target_runner.py"], "contract target runner blob mismatch")
    require(blobs["target_worker"] == EXPECTED_BLOBS["gate_a_worker.c"], "contract target worker blob mismatch")


CONST_TRUE = [
    "project_owner_target_qualification_approval_recorded", "target_connection_occurred", "ssh_occurred",
    "bundle_transferred", "bundle_transfer_verified", "target_identity_verified", "strict_bundle_validation_before",
    "target_no_drive_qualification_complete", "worker_validate_only_complete", "strict_bundle_validation_after",
    "bundle_tree_unchanged", "copy_back_verified", "cleanup_verified", "target_nonexecuting_qualification_complete",
    "execution_bundle_target_qualified",
]
CONST_FALSE = [
    "project_owner_execution_approval_recorded", "authorization_artifact_created", "engineering_smoke_authorized",
    "hardware_ran", "automatic_retry",
]
CONST_ZERO = [
    "probe_count", "execute_authorized_count", "network_connection_count", "hardware_probe_count",
    "sender_start_count", "receiver_capture_count", "control_write_count", "msr_access_count", "hardware_execution_count",
]


def validate_result(result: dict[str, Any]) -> None:
    require(result["schema_id"] == "CAT_CAS_PHASE6B6_GATE_A_TARGET_NONEXECUTING_QUALIFICATION_RESULT_V1", "result schema mismatch")
    require(result["status"] == "TARGET_NONEXECUTING_QUALIFICATION_COMPLETE__ENGINEERING_SMOKE_STILL_UNAUTHORIZED", "result status mismatch")
    require(result["integrated_main"] == INTEGRATED_MAIN, "result integrated main mismatch")
    require(result["reviewed_adapter_head"] == REVIEWED_HEAD, "result reviewed head mismatch")
    require(result["review_id"] == REVIEW_ID, "result review id mismatch")
    require(result["target_qualification_execution_count"] == 1, "result execution count must be 1")
    for k in CONST_TRUE:
        require(result[k] is True, f"result {k} must be true")
    for k in CONST_FALSE:
        require(result[k] is False, f"result {k} must be false")
    for k in CONST_ZERO:
        require(result[k] == 0, f"result {k} must be 0")
    tgt = result["target"]
    require(tgt["ssh_target"] == "root@192.168.137.100", "result target mismatch")
    require(tgt["hostname"] == EXPECTED_HOSTNAME, "result hostname mismatch")
    require(tgt["architecture"] == EXPECTED_ARCH, "result arch mismatch")
    require(tgt["cpu_model"] == EXPECTED_CPU, "result cpu mismatch")
    require(tgt["remote_execution_root"] == "/root/catcas_phase6b6_gate_a_smoke_9c416379", "result exec root mismatch")
    require(tgt["execution_root_final_state"] == "ABSENT", "result exec root not absent")
    require(tgt["transfer_stage_final_state"] == "ABSENT", "result transfer stage not absent")
    b = result["bindings"]
    require(b["execution_bundle_sha256"] == EXECUTION_BUNDLE, "result execution bundle digest mismatch")
    require(b["deterministic_archive_sha256"] == DETERMINISTIC_ARCHIVE, "result archive digest mismatch")
    require(b["bundle_manifest_sha256"] == MANIFEST_FILE_SHA, "result manifest digest mismatch")
    require(b["predecessor_target_identity_stdout_sha256"] == PREDECESSOR_ID_SHA, "result predecessor identity mismatch")
    require(b["deployment_archive_host_sha256"] == b["deployment_archive_target_sha256"], "transfer digest mismatch")
    require(b["current_target_identity_before_sha256"] == b["current_target_identity_after_sha256"], "identity before/after mismatch")
    require(b["before_tree_canonical_sha256"] == b["after_tree_canonical_sha256"], "before/after tree mismatch")
    for name, blob in EXPECTED_BLOBS.items():
        key = {
            "gate_a_hardware_adapter.py": "host_adapter_git_blob_sha1",
            "gate_a_target_runner.py": "target_runner_git_blob_sha1",
            "gate_a_worker.c": "target_worker_git_blob_sha1",
            "gate_a_target_bundle.py": "target_bundle_validator_git_blob_sha1",
            "gate_a_authority.py": "authority_validator_git_blob_sha1",
            "build_gate_a_execution_bundle.py": "bundle_builder_git_blob_sha1",
            "verify_gate_a_adapter_qualification.py": "qualification_verifier_git_blob_sha1",
            "gate_a_isolated_qualification.py": "isolated_qualification_harness_git_blob_sha1",
        }[name]
        require(b[key] == blob, f"result blob {key} mismatch")
    for k, v in result["authority_false_state"].items():
        require(v is False, f"result authority_false_state {k} must be false")
    validate_predecessor_adapter_package_digests(result["predecessor_adapter_package_digests"], "result")
    require(result["next_boundary"] == "GATE_A_ENGINEERING_SMOKE_AUTHORITY_REVIEW_AND_OWNER_DECISION", "result next boundary mismatch")


def validate_candidate_v3(cand: dict[str, Any]) -> None:
    require(cand["schema_id"] == "CAT_CAS_PHASE6B6_GATE_A_ENGINEERING_SMOKE_AUTHORITY_CANDIDATE_V3", "candidate v3 schema mismatch")
    require(cand["status"] == "CANDIDATE__BLOCKED_PENDING_INDEPENDENT_REVIEW_AND_PROJECT_OWNER_EXECUTION_DECISION", "candidate v3 status mismatch")
    for t in ("plan_reviewed", "adapter_implemented", "hosted_adapter_qualification_complete",
              "target_adapter_qualification_complete", "execution_bundle_ready", "execution_bundle_target_qualified",
              "project_owner_target_qualification_approval_recorded"):
        require(cand[t] is True, f"candidate v3 {t} must be true")
    for f in ("project_owner_execution_approval_recorded", "authorization_artifact_created", "engineering_smoke_authorized",
              "hardware_ran", "automatic_retry", "calibration_authorized", "scientific_acquisition_authorized",
              "restoration_authorized", "target_coupling_authorized", "small_wall_authorized"):
        require(cand[f] is False, f"candidate v3 {f} must be false")
    require(cand["next_boundary"] == "GATE_A_ENGINEERING_SMOKE_AUTHORITY_REVIEW_AND_OWNER_DECISION", "candidate v3 next boundary mismatch")
    require(cand["execution_bundle_sha256"] == EXECUTION_BUNDLE, "candidate v3 execution bundle mismatch")
    validate_predecessor_adapter_package_digests(cand["predecessor_adapter_package_digests"], "candidate v3")


def validate_qualification_json(q: dict[str, Any]) -> None:
    require(q["status"] == "GATE_A_TARGET_RUNNER_NO_DRIVE_QUALIFIED", "qual status mismatch")
    require(q["git_free"] is True, "qual not git-free")
    require(q["compiled"] is True, "qual not compiled")
    lbv = q["local_bundle_validation"]
    require(lbv["status"] == "GATE_A_TARGET_BUNDLE_VALIDATED", "qual lbv status mismatch")
    require(lbv["strict"] is True, "qual lbv not strict")
    require(lbv["execution_bundle_sha256"] == EXECUTION_BUNDLE, "qual lbv execution bundle mismatch")
    require(lbv["deterministic_archive_sha256"] == DETERMINISTIC_ARCHIVE, "qual lbv archive mismatch")
    require(q["worker_validate_only"]["status"] == "GATE_A_WORKER_VALIDATE_ONLY_OK", "qual worker validate-only mismatch")
    for c in ("network_connections_opened", "hardware_probes", "sender_starts", "receiver_captures", "control_writes", "msr_accesses", "hardware_executions"):
        require(q[c] == 0, f"qual counter {c} nonzero")


def validate_digest_semantics(semantics: dict[str, Any], final_bindings: dict[str, Any]) -> None:
    require(
        semantics["schema_id"] == "CAT_CAS_PHASE6B6_GATE_A_TARGET_NONEXECUTING_DIGEST_SEMANTICS_V1",
        "digest semantics schema mismatch",
    )
    require(semantics["review_comments"] == [4929354450, 4929446568], "digest semantics review comments mismatch")
    require(
        semantics["predecessor_adapter_qualification_result_sha256"] == PREDECESSOR_RESULT_SHA,
        "digest semantics predecessor adapter result mismatch",
    )
    require(
        semantics["predecessor_candidate_v2_sha256"] == PREDECESSOR_CANDIDATE_V2_SHA,
        "digest semantics predecessor Candidate V2 mismatch",
    )
    require(semantics["predecessor_digest_meaning"] == PREDECESSOR_DIGEST_MEANING, "digest semantics meaning mismatch")
    require(
        semantics["target_qualification_result_committed_sha256"] == sha256_file(RESULT),
        "digest semantics target result committed hash mismatch",
    )
    require(
        semantics["candidate_v3_committed_sha256"] == sha256_file(CANDIDATE_V3),
        "digest semantics Candidate V3 committed hash mismatch",
    )
    require(
        semantics["final_bindings_committed_sha256"] == sha256_file(EVID / "FINAL_BINDINGS.json"),
        "digest semantics FINAL_BINDINGS committed hash mismatch",
    )
    final_digests = final_bindings["predecessor_adapter_package_digests"]
    validate_predecessor_adapter_package_digests(final_digests, "final bindings")
    require(final_bindings["owner_identity"] == EXPECTED_OWNER, "final bindings owner identity mismatch")
    require(
        final_bindings["target_qualification_result_committed_sha256"] == semantics["target_qualification_result_committed_sha256"],
        "final bindings target result committed hash mismatch",
    )
    require(
        final_bindings["candidate_v3_committed_sha256"] == semantics["candidate_v3_committed_sha256"],
        "final bindings Candidate V3 committed hash mismatch",
    )
    require(final_bindings["digest_semantics_path"] == "DIGEST_SEMANTICS.json", "final bindings digest semantics path mismatch")


def require_no_unaccented_owner() -> None:
    forbidden_owner = "Ra" + "ul Romero"
    for root in (HERE, EVID):
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            require(forbidden_owner not in text, f"unaccented owner identity in scoped artifact: {path}")


def validate_evidence(result: dict[str, Any]) -> dict[str, Any]:
    require(EVID.is_dir(), f"evidence dir missing: {EVID}")
    final_bindings = load(EVID / "FINAL_BINDINGS.json")
    semantics = load(DIGEST_SEMANTICS)
    validate_digest_semantics(semantics, final_bindings)
    require(final_bindings["overall_status"] == "SUCCESS", "final bindings not SUCCESS")
    require(final_bindings["qualification_execution_count"] == 1, "final bindings execution count != 1")
    require(final_bindings["qualification_exit_code"] == 0, "final bindings exit code != 0")
    require(final_bindings["bundle_tree_unchanged"] is True, "final bindings tree changed")
    require(final_bindings["copy_back_verified"] is True, "final bindings copy-back unverified")
    require(final_bindings["cleanup_verified"] is True, "final bindings cleanup unverified")

    cb = load(EVID / "COPY_BACK_RECEIPT.json")
    require(cb["schema_id"] == "CAT_CAS_PHASE6B6_GATE_A_COPY_BACK_RECEIPT_V1", "copy-back schema mismatch")
    require(cb["retained_evidence_custody_verified"] is True, "copy-back custody unverified")
    require(cb["inventory_verified"] is True, "copy-back inventory unverified")
    require(cb["unexpected_entries"] == [], "copy-back unexpected entries present")

    cl = load(EVID / "CLEANUP_RECEIPT.json")
    require(cl["schema_id"] == "CAT_CAS_PHASE6B6_GATE_A_CLEANUP_RECEIPT_V1", "cleanup schema mismatch")
    for k in ("exact_execution_root_removed", "exact_transfer_stage_removed", "execution_root_absence_proven", "transfer_stage_absence_proven"):
        require(cl[k] is True, f"cleanup {k} not true")
    require(cl["forbidden_processes_remaining"] == [], "cleanup forbidden processes remain")

    # evidence inventory closure
    inv = load(EVID / "EVIDENCE_INVENTORY.json")
    listed = {e["path"] for e in inv["files"]}
    require("DIGEST_SEMANTICS.json" in listed, "digest semantics missing from evidence inventory")
    for e in inv["files"]:
        fp = EVID / e["path"]
        require(fp.is_file(), f"evidence inventory missing file: {e['path']}")
        require(fp.stat().st_size == e["size"], f"evidence size mismatch: {e['path']}")
        require(sha256_file(fp) == e["sha256"], f"evidence sha256 mismatch: {e['path']}")
    actual = {p.relative_to(EVID).as_posix() for p in EVID.rglob("*") if p.is_file() and p.name not in ("EVIDENCE_INVENTORY.json", "FINAL_BINDINGS.json")}
    extra = actual - listed
    require(not extra, f"unexpected evidence files: {sorted(extra)}")

    # copied-back target evidence closure
    cb_dir = EVID / "copy_back" / "target_evidence"
    require(cb_dir.is_dir(), "copied-back target evidence dir missing")
    tinv = json.loads((cb_dir / "TARGET_EVIDENCE_INVENTORY.json").read_text(encoding="utf-8"))
    require(isinstance(tinv, list), "target evidence inventory must be a list")
    tinv_names = {e["path"] for e in tinv}
    for e in tinv:
        fp = cb_dir / e["path"]
        require(fp.is_file(), f"target evidence missing: {e['path']}")
        require(fp.stat().st_size == e["size"], f"target evidence size mismatch: {e['path']}")
        require(sha256_file(fp) == e["sha256"], f"target evidence sha256 mismatch: {e['path']}")
    cb_actual = {p.name for p in cb_dir.iterdir() if p.is_file()}
    require(tinv_names <= cb_actual, f"copied-back evidence missing inventory entries: {tinv_names - cb_actual}")
    extra_cb = cb_actual - tinv_names
    require(extra_cb <= {"TARGET_EVIDENCE_INVENTORY.json"}, f"copied-back evidence unexpected files: {sorted(extra_cb)}")

    # qualification result from copied-back evidence
    qjson = load(cb_dir / "TARGET_QUALIFICATION_RESULT.json")
    validate_qualification_json(qjson)

    # before/after tree equality
    tree_before = json.loads((cb_dir / "TARGET_TREE_BEFORE.json").read_text(encoding="utf-8"))
    tree_after = json.loads((cb_dir / "TARGET_TREE_AFTER.json").read_text(encoding="utf-8"))
    require(tree_before == tree_after, "copied-back before/after tree differ")

    # identity semantic equality
    id_before = load(cb_dir / "TARGET_IDENTITY_BEFORE.json")
    id_after = load(cb_dir / "TARGET_IDENTITY_AFTER.json")
    for key in ("hostname", "architecture", "cpu_model"):
        require(id_before[key] == id_after[key], f"identity {key} changed")
    require(id_before["hostname"] == EXPECTED_HOSTNAME, "identity hostname mismatch")
    require(id_before["architecture"] == EXPECTED_ARCH, "identity arch mismatch")
    require(id_before["cpu_model"] == EXPECTED_CPU, "identity cpu mismatch")

    # exactly one qualification execution; no forbidden commands
    commands = [json.loads(l) for l in (EVID / "COMMANDS.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    qual_cmds = [c for c in commands if str(c.get("argv")).find("--qualify-no-drive") != -1 and "gate_a_target_runner.py" in str(c.get("argv"))]
    require(len(qual_cmds) == 1, f"expected exactly one qualification execution, found {len(qual_cmds)}")
    for c in commands:
        argv_s = " ".join(c.get("argv", [])) if isinstance(c.get("argv"), list) else str(c.get("argv"))
        for bad in FORBIDDEN_COMMAND_SUBSTRINGS:
            require(bad not in argv_s, f"forbidden command substring {bad!r} in {argv_s[:120]}")

    return {"final_bindings": final_bindings, "digest_semantics": semantics, "copy_back": cb, "cleanup": cl,
            "qualification": qjson, "tree_before": tree_before, "tree_after": tree_after}


def no_authority_artifact() -> None:
    hits = list(PHASE6B6.rglob("GATE_A_EXECUTION_AUTHORITY.json"))
    require(not hits, f"execution authority artifact present: {hits}")
    tracked = subprocess.run(["git", "ls-files", "*GATE_A_EXECUTION_AUTHORITY.json"], stdout=subprocess.PIPE, text=True).stdout.strip()
    require(tracked == "", f"execution authority artifact tracked: {tracked}")


def verify_adapter_blobs() -> None:
    for name, blob in EXPECTED_BLOBS.items():
        got = git_blob(ADAPTER / name)
        require(got == blob, f"adapter blob {name} mismatch: {got} != {blob}")
    manifest_sha = sha256_file(ADAPTER / "GATE_A_EXECUTION_BUNDLE_MANIFEST.json")
    require(manifest_sha == MANIFEST_FILE_SHA, f"adapter manifest file sha mismatch: {manifest_sha}")


def assert_rejects(name: str, func: Callable[[], None]) -> str:
    try:
        func()
    except Exception:
        return name
    raise VerifyError(f"mutation accepted: {name}")


def mutation_tests(
    result: dict[str, Any],
    auth: dict[str, Any],
    cand: dict[str, Any],
    qjson: dict[str, Any],
    semantics: dict[str, Any],
    final_bindings: dict[str, Any],
) -> dict[str, Any]:
    cases: list[str] = []

    def m_result(mut: Callable[[dict[str, Any]], None]) -> Callable[[], None]:
        def inner() -> None:
            r = copy.deepcopy(result)
            mut(r)
            validate_result(r)
        return inner

    def m_qual(mut: Callable[[dict[str, Any]], None]) -> Callable[[], None]:
        def inner() -> None:
            q = copy.deepcopy(qjson)
            mut(q)
            validate_qualification_json(q)
        return inner

    def m_auth(mut: Callable[[dict[str, Any]], None]) -> Callable[[], None]:
        def inner() -> None:
            a = copy.deepcopy(auth)
            mut(a)
            validate_authorization(a)
        return inner

    def m_cand(mut: Callable[[dict[str, Any]], None]) -> Callable[[], None]:
        def inner() -> None:
            c = copy.deepcopy(cand)
            mut(c)
            validate_candidate_v3(c)
        return inner

    def m_semantics(mut: Callable[[dict[str, Any]], None]) -> Callable[[], None]:
        def inner() -> None:
            s = copy.deepcopy(semantics)
            mut(s)
            validate_digest_semantics(s, final_bindings)
        return inner

    checks = [
        ("second_qualification_execution", m_result(lambda r: r.__setitem__("target_qualification_execution_count", 2))),
        ("automatic_retry_enabled_result", m_result(lambda r: r.__setitem__("automatic_retry", True))),
        ("automatic_retry_enabled_auth", m_auth(lambda a: a.__setitem__("automatic_retry", True))),
        ("probe_invocation", m_result(lambda r: r.__setitem__("probe_count", 1))),
        ("execute_authorized_invocation", m_result(lambda r: r.__setitem__("execute_authorized_count", 1))),
        ("wrong_target", m_result(lambda r: r["target"].__setitem__("ssh_target", "root@127.0.0.1"))),
        ("wrong_execution_root", m_result(lambda r: r["target"].__setitem__("remote_execution_root", "/root/wrong"))),
        ("wrong_bundle_digest", m_result(lambda r: r["bindings"].__setitem__("execution_bundle_sha256", "0" * 64))),
        ("wrong_transfer_digest", m_result(lambda r: r["bindings"].__setitem__("deployment_archive_target_sha256", "0" * 64))),
        ("wrong_identity", m_result(lambda r: r["target"].__setitem__("cpu_model", "Intel"))),
        ("wrong_worker_result", m_qual(lambda q: q["worker_validate_only"].__setitem__("status", "BAD"))),
        ("nonzero_hardware_probe", m_qual(lambda q: q.__setitem__("hardware_probes", 1))),
        ("nonzero_sender", m_qual(lambda q: q.__setitem__("sender_starts", 1))),
        ("nonzero_receiver", m_qual(lambda q: q.__setitem__("receiver_captures", 1))),
        ("nonzero_control_write", m_qual(lambda q: q.__setitem__("control_writes", 1))),
        ("nonzero_msr", m_qual(lambda q: q.__setitem__("msr_accesses", 1))),
        ("nonzero_hardware_execution", m_qual(lambda q: q.__setitem__("hardware_executions", 1))),
        ("changed_after_tree", m_result(lambda r: r["bindings"].__setitem__("after_tree_canonical_sha256", "0" * 64))),
        ("copy_back_unverified", m_result(lambda r: r.__setitem__("copy_back_verified", False))),
        ("cleanup_unverified", m_result(lambda r: r.__setitem__("cleanup_verified", False))),
        ("execution_approval_true", m_result(lambda r: r.__setitem__("project_owner_execution_approval_recorded", True))),
        ("authority_artifact_created_true", m_result(lambda r: r.__setitem__("authorization_artifact_created", True))),
        ("engineering_smoke_authorized_true", m_result(lambda r: r.__setitem__("engineering_smoke_authorized", True))),
        ("hardware_ran_true", m_result(lambda r: r.__setitem__("hardware_ran", True))),
        ("auth_engineering_smoke_true", m_auth(lambda a: a.__setitem__("engineering_smoke_authorized", True))),
        ("auth_hardware_execution_true", m_auth(lambda a: a.__setitem__("hardware_execution_authorized", True))),
        ("auth_max_executions_two", m_auth(lambda a: a.__setitem__("maximum_target_qualification_executions", 2))),
        ("project_owner_missing_accent", m_auth(lambda a: a.__setitem__("project_owner", "Ra" + "ul Romero"))),
        ("candidate_execution_approval_true", m_cand(lambda c: c.__setitem__("project_owner_execution_approval_recorded", True))),
        ("candidate_engineering_smoke_true", m_cand(lambda c: c.__setitem__("engineering_smoke_authorized", True))),
        ("candidate_hardware_ran_true", m_cand(lambda c: c.__setitem__("hardware_ran", True))),
        ("wrong_predecessor_adapter_result_digest", m_semantics(lambda s: s.__setitem__("predecessor_adapter_qualification_result_sha256", "0" * 64))),
        ("wrong_predecessor_candidate_v2_digest", m_semantics(lambda s: s.__setitem__("predecessor_candidate_v2_sha256", "0" * 64))),
        ("predecessor_digest_mislabeled_as_target_result", m_semantics(lambda s: s.__setitem__("target_qualification_result_committed_sha256", PREDECESSOR_RESULT_SHA))),
        ("missing_target_result_committed_hash", m_semantics(lambda s: s.pop("target_qualification_result_committed_sha256"))),
        ("missing_candidate_v3_committed_hash", m_semantics(lambda s: s.pop("candidate_v3_committed_sha256"))),
        ("stale_final_bindings_hash", m_semantics(lambda s: s.__setitem__("final_bindings_committed_sha256", "0" * 64))),
    ]
    for name, func in checks:
        cases.append(assert_rejects(name, func))

    # evidence-level mutations: missing / extra evidence, cleanup-before-copy-back
    def missing_evidence() -> None:
        inv = load(EVID / "EVIDENCE_INVENTORY.json")
        fp = EVID / inv["files"][0]["path"]
        data = fp.read_bytes()
        try:
            fp.unlink()
            validate_evidence(result)
        finally:
            fp.write_bytes(data)

    def extra_evidence() -> None:
        extra = EVID / "target" / "logs" / "UNEXPECTED_EXTRA.txt"
        try:
            extra.write_text("x", encoding="utf-8")
            validate_evidence(result)
        finally:
            if extra.exists():
                extra.unlink()

    def missing_digest_semantics() -> None:
        data = DIGEST_SEMANTICS.read_bytes()
        try:
            DIGEST_SEMANTICS.unlink()
            validate_evidence(result)
        finally:
            DIGEST_SEMANTICS.write_bytes(data)

    def stale_inventory_hash() -> None:
        readme = EVID / "README.md"
        data = readme.read_bytes()
        try:
            readme.write_bytes(data + b"\n")
            validate_evidence(result)
        finally:
            readme.write_bytes(data)

    def cleanup_before_copy_back() -> None:
        # a result asserting cleanup verified while copy-back unverified is contradictory
        r = copy.deepcopy(result)
        r["copy_back_verified"] = False
        r["cleanup_verified"] = True
        validate_result(r)
        require(not (r["cleanup_verified"] and not r["copy_back_verified"]), "cleanup must not precede verified copy-back")

    cases.append(assert_rejects("missing_evidence", missing_evidence))
    cases.append(assert_rejects("extra_evidence", extra_evidence))
    cases.append(assert_rejects("missing_digest_semantics", missing_digest_semantics))
    cases.append(assert_rejects("stale_inventory_hash", stale_inventory_hash))
    cases.append(assert_rejects("cleanup_before_copy_back", cleanup_before_copy_back))

    return {"status": "TARGET_NONEXEC_MUTATION_TESTS_PASS", "negative_tests": len(cases), "cases": cases}


def main() -> int:
    auth = load(AUTHORIZATION)
    contract = load(CONTRACT)
    result = load(RESULT)
    cand = load(CANDIDATE_V3)
    schema = load(SCHEMA)

    validate_schema_closed(schema)
    validate_authorization(auth)
    validate_contract(contract)
    validate_result(result)
    validate_candidate_v3(cand)
    verify_adapter_blobs()
    evidence = validate_evidence(result)
    require_no_unaccented_owner()
    no_authority_artifact()
    mutations = mutation_tests(
        result,
        auth,
        cand,
        evidence["qualification"],
        evidence["digest_semantics"],
        evidence["final_bindings"],
    )

    out = {
        "status": "GATE_A_TARGET_NONEXECUTING_QUALIFICATION_VERIFIED",
        "integrated_main": INTEGRATED_MAIN,
        "reviewed_adapter_head": REVIEWED_HEAD,
        "target_nonexecuting_qualification_complete": True,
        "execution_bundle_target_qualified": True,
        "project_owner_execution_approval_recorded": False,
        "engineering_smoke_authorized": False,
        "hardware_ran": False,
        "qualification_execution_count": 1,
        "mutation_tests": mutations,
        "evidence_inventory_sha256": sha256_file(EVID / "EVIDENCE_INVENTORY.json"),
        "next_boundary": "GATE_A_ENGINEERING_SMOKE_AUTHORITY_REVIEW_AND_OWNER_DECISION",
    }
    print(json.dumps(out, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (VerifyError, subprocess.CalledProcessError, json.JSONDecodeError, KeyError) as exc:
        print(f"verify_gate_a_target_nonexecuting_qualification: {exc}", file=sys.stderr)
        raise SystemExit(1)
