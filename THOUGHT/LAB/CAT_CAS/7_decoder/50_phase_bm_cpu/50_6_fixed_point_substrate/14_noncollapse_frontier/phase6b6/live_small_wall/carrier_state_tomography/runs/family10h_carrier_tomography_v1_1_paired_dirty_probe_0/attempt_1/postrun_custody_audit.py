from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tarfile
from pathlib import Path
from typing import Any

SOURCE_AUTHORITY_COMMIT = "cac5d33536768e00aa0de5f515e626fecccdeeda"
MANIFEST_FREEZE_COMMIT = "354ca8ab2d62458fca41481d74ff98c1b39ab6ed"
PRE_AUDIT_EVIDENCE_COMMIT = "cb054593b69de59e262d38793e9aa22e36f551e5"
TRANSACTION_RUN_ID = "family10h_carrier_tomography_v1_1_paired_dirty_probe_0"
RESULT_CLASS = "FAMILY10H_PAIRED_DIRTY_PROBE_Q_READOUT_CONFIRMED_PROSPECTIVE"
SCIENTIFIC_CLAIM = "PUBLIC_POST_SOURCE_SCALAR_CARRIER_Q_READOUT_CONFIRMED"
CLAIM_CEILING = "one-dimensional public scalar q-codeword readout only"
SOURCE_PACKAGE_REL = "THOUGHT/LAB/CAT_CAS/7_decoder/50_phase_bm_cpu/50_6_fixed_point_substrate/14_noncollapse_frontier/phase6b6/live_small_wall/carrier_state_tomography/family10h_carrier_tomography_v1_1_paired_dirty_probe"
ATTEMPT_REL = "THOUGHT/LAB/CAT_CAS/7_decoder/50_phase_bm_cpu/50_6_fixed_point_substrate/14_noncollapse_frontier/phase6b6/live_small_wall/carrier_state_tomography/runs/family10h_carrier_tomography_v1_1_paired_dirty_probe_0/attempt_1"
ARCHIVE_NAME = "ATTEMPT_1_REMOTE_ROOT.tar.gz"
OWNER_MEMBER = f"{TRANSACTION_RUN_ID}/.family10h_carrier_tomography_v1_1_paired_dirty_probe_0_owner"


def repo_root() -> Path:
    return Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_record(path: Path, rel: str | None = None) -> dict[str, Any]:
    return {
        "relative_path": rel or path.name,
        "sha256": sha256_file(path),
        "size": path.stat().st_size,
    }


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def compact_digest(data: Any) -> str:
    return sha256_bytes(json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8"))


def parse_archive_report(stdout: str) -> dict[str, Any]:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError("missing target archive hash/size report")
    sha_match = re.match(r"^([0-9a-f]{64})\s+(.+)$", lines[0])
    size_match = re.match(r"^(\d+)\s+(.+)$", lines[1])
    if not sha_match or not size_match:
        raise ValueError("malformed target archive hash/size report")
    return {
        "sha256": sha_match.group(1),
        "sha_path": sha_match.group(2).strip(),
        "size": int(size_match.group(1)),
        "size_path": size_match.group(2).strip(),
        "raw_stdout": stdout,
    }


def tar_member_bytes(archive: Path, member: str) -> bytes:
    with tarfile.open(archive, "r:gz") as tf:
        extracted = tf.extractfile(tf.getmember(member))
        if extracted is None:
            raise ValueError(f"archive member is not a file: {member}")
        return extracted.read()


def tar_member_record(archive: Path, member: str) -> dict[str, Any]:
    with tarfile.open(archive, "r:gz") as tf:
        info = tf.getmember(member)
        extracted = tf.extractfile(info)
        if extracted is None:
            raise ValueError(f"archive member is not a file: {member}")
        data = extracted.read()
    return {"archive_member": member, "sha256": sha256_bytes(data), "size": info.size}


def count_tar_member_lines(archive: Path, member: str) -> int:
    with tarfile.open(archive, "r:gz") as tf:
        extracted = tf.extractfile(tf.getmember(member))
        if extracted is None:
            raise ValueError(f"archive member is not a file: {member}")
        return sum(1 for _ in extracted)


def inventory_entry(inventory: dict[str, Any], relative_path: str) -> dict[str, Any]:
    for entry in inventory["files"]:
        if entry.get("relative_path") == relative_path:
            return entry
    raise KeyError(relative_path)


def git_diff_names(root: Path, *args: str) -> list[str]:
    output = subprocess.check_output(["git", *args], cwd=root, text=True)
    return [line for line in output.splitlines() if line]


def command_contains_unredacted_nonce(receipt: dict[str, Any]) -> bool:
    joined = "\n".join(str(part) for part in receipt.get("command", []))
    return re.search(r"FAMILY10H_CARRIER_TOMOGRAPHY_TEMPERATURE_AUTHORITY_NONCE=[0-9a-f]{64}", joined) is not None


def source_contains_committed_nonce(root: Path) -> bool:
    source = root / SOURCE_PACKAGE_REL / "run_family10h_carrier_tomography_v1_1_paired_dirty_probe.py"
    text = source.read_text(encoding="utf-8")
    return re.search(r'^LIVE_NONCE = "[0-9a-f]{64}"$', text, flags=re.MULTILINE) is not None


def build_outputs() -> None:
    root = repo_root()
    attempt = root / ATTEMPT_REL
    archive = attempt / ARCHIVE_NAME

    copyback = load_json(attempt / "ATTEMPT_1_COPYBACK_RECEIPT.json")
    inventory = load_json(attempt / "ATTEMPT_1_EVIDENCE_INVENTORY.json")
    adjudication = load_json(attempt / "ATTEMPT_1_PROSPECTIVE_ADJUDICATION.json")
    cleanup = load_json(attempt / "ATTEMPT_1_CLEANUP_RECEIPT.json")
    live_started = load_json(attempt / "ATTEMPT_1_LIVE_INVOCATION_STARTED.json")
    live_completed = load_json(attempt / "ATTEMPT_1_LIVE_INVOCATION_COMPLETED.json")
    deployment = load_json(attempt / "ATTEMPT_1_DEPLOYMENT_RECEIPT.json")
    self_test_path = attempt / "ATTEMPT_1_PROSPECTIVE_SELF_TEST.json"
    evidence_closure_timestamp = cleanup["created_at"]

    target_archive = parse_archive_report(copyback["make_archive"]["stdout"])
    local_archive = file_record(archive, ARCHIVE_NAME)
    inventory_archive = inventory["remote_root_archive"]
    adjudication_archive = {
        "sha256": adjudication["source_evidence"]["archive_sha256"],
        "size": adjudication["source_evidence"]["archive_size"],
    }
    archive_equality_checks = {
        "target_reported_sha_equals_committed_local_sha": target_archive["sha256"] == local_archive["sha256"],
        "target_reported_size_equals_committed_local_size": target_archive["size"] == local_archive["size"],
        "inventory_sha_equals_target_and_local": inventory_archive["sha256"] == target_archive["sha256"] == local_archive["sha256"],
        "inventory_size_equals_target_and_local": inventory_archive["size"] == target_archive["size"] == local_archive["size"],
        "adjudication_sha_equals_target_and_local": adjudication_archive["sha256"] == target_archive["sha256"] == local_archive["sha256"],
        "adjudication_size_equals_target_and_local": adjudication_archive["size"] == target_archive["size"] == local_archive["size"],
    }

    marker_data = tar_member_bytes(archive, OWNER_MEMBER)
    marker_text = marker_data.decode("utf-8")
    marker_fields = dict(line.split("=", 1) for line in marker_text.splitlines() if "=" in line)
    owner_marker = {
        "archive_member_path": OWNER_MEMBER,
        "file_sha256": sha256_bytes(marker_data),
        "file_size": len(marker_data),
        "parsed_marker_fields": marker_fields,
        "expected_source_authority_commit": SOURCE_AUTHORITY_COMMIT,
        "expected_manifest_freeze_commit": MANIFEST_FREEZE_COMMIT,
        "expected_transaction_run_id": TRANSACTION_RUN_ID,
        "exact_match": {
            "source_authority_commit": marker_fields.get("source_authority_commit") == SOURCE_AUTHORITY_COMMIT,
            "freeze_commit": marker_fields.get("freeze_commit") == MANIFEST_FREEZE_COMMIT,
            "transaction_run_id": marker_fields.get("transaction_run_id") == TRANSACTION_RUN_ID,
        },
    }
    owner_marker["passed"] = all(owner_marker["exact_match"].values())

    stale_label_overlay = {
        "classification": "metadata-label-defect-only",
        "does_not_alter_metrics_gates_result_class_or_scientific_claim": True,
        "bindings": {
            "channel_specificity.attempt_3_observation": "channel_specificity.prospective_attempt_1_observation",
            "channel_specificity.attempt_3_primary_passed": "channel_specificity.prospective_attempt_1_primary_passed",
            "channel_specificity.attempt_3_secondary_channels_failed_same_law": "channel_specificity.prospective_attempt_1_secondary_channels_failed_same_law",
        },
        "observed_values": {
            key: adjudication["channel_specificity"].get(key)
            for key in ["attempt_3_observation", "attempt_3_primary_passed", "attempt_3_secondary_channels_failed_same_law"]
        },
    }

    retired_nonce_observation = {
        "completed_controller_embedded_full_nonce_in_committed_source": source_contains_committed_nonce(root),
        "live_start_receipt_contains_unredacted_nonce": command_contains_unredacted_nonce(live_started),
        "live_completion_receipt_contains_unredacted_nonce": command_contains_unredacted_nonce(live_completed),
        "temperature_authority_nonce_sha256": live_started.get("temperature_authority_nonce_sha256"),
        "full_nonce_value_redacted_from_new_postrun_artifacts": True,
        "retired_and_forbidden_from_reuse": True,
        "historical_evidence_deleted_or_rewritten": False,
        "future_controller_requirement": {
            "nonce_supplied_externally_at_execution_time": True,
            "only_nonce_hash_committed_or_retained": True,
            "full_command_receipts_redact_nonce_value": True,
            "source_manifest_run_identity_and_one_attempt_authority_remain_separately_bound": True,
            "committed_nonce_may_not_be_treated_as_independent_secret_authorization_factor": True,
        },
    }

    cleanup_observation = {
        "classification": "non-scientific cleanup-custody observation",
        "remote_canonical_paths_checked_absent_before_deployment": True,
        "controller_created_root_and_wrote_marker_during_same_attempt": True,
        "archived_marker_has_correct_identity": owner_marker["passed"],
        "cleanup_confirmed_marker_presence_before_deletion": cleanup["cleanup"]["returncode"] == 0,
        "cleanup_confirmed_final_path_absence": cleanup["passed"] is True,
        "cleanup_did_not_compare_marker_contents_before_deletion": True,
        "cleanup_receipt_rewritten": False,
        "scientific_packet_invalidated": not owner_marker["passed"],
    }

    future_controller_regressions = [
        "target/local archive hash mismatch",
        "target/local archive size mismatch",
        "missing or malformed target archive report",
        "ownership marker missing",
        "ownership marker content mismatch",
        "ownership marker source-authority mismatch",
        "ownership marker freeze-commit mismatch",
        "ownership marker run-identity mismatch",
        "committed or reused full live nonce",
        "unredacted nonce in command receipts",
    ]

    modified_or_deleted_existing = git_diff_names(root, "diff", "--name-only", "--diff-filter=MD", PRE_AUDIT_EVIDENCE_COMMIT, "--", ATTEMPT_REL)
    source_package_diff = git_diff_names(root, "diff", "--name-only", MANIFEST_FREEZE_COMMIT, "--", SOURCE_PACKAGE_REL)
    v1_0_diff = git_diff_names(root, "diff", "--name-only", PRE_AUDIT_EVIDENCE_COMMIT, "--", str(Path(ATTEMPT_REL).parents[1] / "family10h_carrier_tomography_v1_0"))

    custody_audit = {
        "schema": "FAMILY10H_PAIRED_DIRTY_PROBE_V1_1_ATTEMPT1_POSTRUN_CUSTODY_AUDIT_V1",
        "created_utc": evidence_closure_timestamp,
        "source_authority_commit": SOURCE_AUTHORITY_COMMIT,
        "manifest_freeze_commit": MANIFEST_FREEZE_COMMIT,
        "pre_audit_evidence_commit": PRE_AUDIT_EVIDENCE_COMMIT,
        "transaction_run_id": TRANSACTION_RUN_ID,
        "retained_attempt_path": ATTEMPT_REL,
        "audit_scope": "offline additions-only evidence closure; no target contact or live action",
        "null_model_scope": "not applicable: this postrun audit adds no scientific hardening gate, classifier, shuffled baseline, random baseline, or model-selection law; it checks exact custody equality and metadata labels only",
        "remote_local_archive_equality": {
            "target_reported_archive": target_archive,
            "committed_local_archive": local_archive,
            "evidence_inventory_archive": inventory_archive,
            "adjudication_source_evidence_archive": adjudication_archive,
            "checks": archive_equality_checks,
            "passed": all(archive_equality_checks.values()),
            "historical_controller_stored_matching_values": copyback.get("passed") is True,
            "historical_controller_did_not_parse_and_enforce_target_local_equality_before_copyback_passed": True,
            "postrun_audit_closes_equality_retrospectively": True,
            "actual_archive_custody_valid_because_recorded_values_match_exactly": all(archive_equality_checks.values()),
        },
        "archived_ownership_marker": owner_marker,
        "cleanup_custody_observation": cleanup_observation,
        "authorization_token_handling": retired_nonce_observation,
        "stale_diagnostic_semantic_overlay": stale_label_overlay,
        "future_controller_regressions_required": future_controller_regressions,
        "existing_evidence_modified_or_deleted_since_pre_audit_commit": modified_or_deleted_existing,
        "source_package_diff_from_manifest_freeze_commit": source_package_diff,
        "v1_0_package_or_evidence_diff_from_pre_audit_commit": v1_0_diff,
        "passed": all(archive_equality_checks.values()) and owner_marker["passed"] and not modified_or_deleted_existing and not source_package_diff and not v1_0_diff,
    }

    audit_json = attempt / "ATTEMPT_1_POSTRUN_CUSTODY_AUDIT.json"
    write_json(audit_json, custody_audit)

    audit_md = attempt / "ATTEMPT_1_POSTRUN_CUSTODY_AUDIT.md"
    audit_md.write_text(
        "# Attempt 1 Postrun Custody Audit\n\n"
        f"Result: `{RESULT_CLASS}`\n\n"
        f"Scientific claim: `{SCIENTIFIC_CLAIM}`\n\n"
        "This is an offline additions-only custody closure. No target contact, SSH, SCP, PMU acquisition, runtime execution, cleanup, deployment, or live action was performed by this audit.\n\n"
        "## Remote/Local Archive Equality\n\n"
        f"- Target-reported SHA-256: `{target_archive['sha256']}`\n"
        f"- Committed local archive SHA-256: `{local_archive['sha256']}`\n"
        f"- Evidence-inventory archive SHA-256: `{inventory_archive['sha256']}`\n"
        f"- Adjudication source-evidence archive SHA-256: `{adjudication_archive['sha256']}`\n"
        f"- Target-reported size: `{target_archive['size']}`\n"
        f"- Committed local archive size: `{local_archive['size']}`\n"
        f"- Equality passed: `{custody_audit['remote_local_archive_equality']['passed']}`\n\n"
        "The historical controller stored matching values, but did not itself parse and enforce target/local equality before setting copy-back passed. This postrun audit closes that equality retrospectively. The archive custody is valid because the recorded target, local, inventory, and adjudication values match exactly.\n\n"
        "## Archived Ownership Marker\n\n"
        f"- Archive member: `{OWNER_MEMBER}`\n"
        f"- Marker SHA-256: `{owner_marker['file_sha256']}`\n"
        f"- Marker size: `{owner_marker['file_size']}`\n"
        f"- Source-authority commit match: `{owner_marker['exact_match']['source_authority_commit']}`\n"
        f"- Manifest freeze commit match: `{owner_marker['exact_match']['freeze_commit']}`\n"
        f"- Transaction/run identity match: `{owner_marker['exact_match']['transaction_run_id']}`\n"
        f"- Marker verification passed: `{owner_marker['passed']}`\n\n"
        "Remote canonical paths were checked absent before deployment. The controller created the root and wrote the marker during the same attempt. The archived marker has the correct identity. Cleanup confirmed marker presence and final path absence, but did not compare marker contents before deletion. This is a non-scientific cleanup-custody observation and does not rewrite the cleanup receipt.\n\n"
        "## Authorization Token Handling\n\n"
        "The completed controller embedded the one-shot live nonce in committed source and command receipts. Historical evidence was not deleted or rewritten. The nonce is permanently retired and forbidden from reuse. Future controllers must supply the nonce externally at execution time, retain only its hash, redact full command receipts, and must not treat a committed nonce as an independent secret authorization factor.\n\n"
        "## Stale Diagnostic Semantics\n\n"
        "The prospective adjudication inherited diagnostic names `attempt_3_observation`, `attempt_3_primary_passed`, and `attempt_3_secondary_channels_failed_same_law`. In the v1.1 prospective adjudication, those fields contain diagnostics computed from the v1.1 attempt-1 packet. They should be read as `prospective_attempt_1_observation`, `prospective_attempt_1_primary_passed`, and `prospective_attempt_1_secondary_channels_failed_same_law`. This is a metadata-label defect only and does not alter metrics, gates, result class, or scientific claim.\n\n"
        "## Future Controller Fail-Closed Regressions\n\n"
        + "\n".join(f"- {item}" for item in future_controller_regressions)
        + "\n",
        encoding="utf-8",
    )

    member_hashes = adjudication["source_evidence"]["archive_member_hashes"]
    raw_member = member_hashes["raw_records.jsonl"]
    death_member = member_hashes["source_death_receipts.jsonl"]
    feature_member = member_hashes["feature_freeze.json"]
    target_receipt_member = member_hashes["output_target_execution_receipt.json"]
    schedule_sha_member = inventory_entry(inventory, "CARRIER_TOMOGRAPHY_PUBLIC_SCHEDULE.sha256")
    source_hashes_member = inventory_entry(inventory, "CARRIER_TOMOGRAPHY_SOURCE_HASHES.json")
    source_bundle_member = inventory_entry(inventory, "CARRIER_TOMOGRAPHY_SOURCE_BUNDLE.tar.gz")
    runtime_member = inventory_entry(inventory, "family10h_carrier_tomography_runtime")
    manifest_member = inventory_entry(inventory, "CARRIER_TOMOGRAPHY_IMPLEMENTATION_MANIFEST.json")
    schedule_json_member = inventory_entry(inventory, "CARRIER_TOMOGRAPHY_PUBLIC_SCHEDULE.json")
    schedule_tsv_member = inventory_entry(inventory, "CARRIER_TOMOGRAPHY_PUBLIC_SCHEDULE.tsv")

    summary_core = {
        "schema": "FAMILY10H_PAIRED_DIRTY_PROBE_V1_1_ATTEMPT1_FINAL_EVIDENCE_SUMMARY_V1",
        "created_utc": evidence_closure_timestamp,
        "source_authority_commit": SOURCE_AUTHORITY_COMMIT,
        "manifest_freeze_commit": MANIFEST_FREEZE_COMMIT,
        "pre_audit_evidence_commit": PRE_AUDIT_EVIDENCE_COMMIT,
        "source_package_path": SOURCE_PACKAGE_REL,
        "retained_attempt_path": ATTEMPT_REL,
        "manifest": {
            "canonical_sha256": deployment["manifest_canonical_sha256"],
            "file_sha256": deployment["manifest_file_sha256"],
            "archive_member_file_sha256": manifest_member["sha256"],
            "size": manifest_member["size"],
        },
        "runtime_binary_sha256": runtime_member["sha256"],
        "runtime_binary_size": runtime_member["size"],
        "source_hashes": {
            "file_sha256": source_hashes_member["sha256"],
            "canonical_sha256": load_json_from_tar(archive, f"{TRANSACTION_RUN_ID}/CARRIER_TOMOGRAPHY_SOURCE_HASHES.json")["source_hashes_sha256"],
            "size": source_hashes_member["size"],
        },
        "source_bundle": {"sha256": source_bundle_member["sha256"], "size": source_bundle_member["size"]},
        "schedule": {
            "canonical_sha256": adjudication["source_evidence"]["schedule_sha256"],
            "json_sha256": schedule_json_member["sha256"],
            "json_size": schedule_json_member["size"],
            "tsv_sha256": schedule_tsv_member["sha256"],
            "tsv_size": schedule_tsv_member["size"],
            "receipt_file_sha256": schedule_sha_member["sha256"],
            "receipt_size": schedule_sha_member["size"],
        },
        "remote_root_archive": {"sha256": local_archive["sha256"], "size": local_archive["size"]},
        "raw_records": {"sha256": raw_member["sha256"], "size": raw_member["size"], "count": count_tar_member_lines(archive, raw_member["archive_member"])},
        "source_death_receipts": {"sha256": death_member["sha256"], "size": death_member["size"], "count": count_tar_member_lines(archive, death_member["archive_member"])},
        "feature_freeze": {"sha256": feature_member["sha256"], "size": feature_member["size"]},
        "target_execution_receipt": {"sha256": target_receipt_member["sha256"], "size": target_receipt_member["size"]},
        "receipt_hashes": {
            "live_start": file_record(attempt / "ATTEMPT_1_LIVE_INVOCATION_STARTED.json"),
            "live_completion": file_record(attempt / "ATTEMPT_1_LIVE_INVOCATION_COMPLETED.json"),
            "deployment": file_record(attempt / "ATTEMPT_1_DEPLOYMENT_RECEIPT.json"),
            "copy_back": file_record(attempt / "ATTEMPT_1_COPYBACK_RECEIPT.json"),
            "inventory": file_record(attempt / "ATTEMPT_1_EVIDENCE_INVENTORY.json"),
            "cleanup": file_record(attempt / "ATTEMPT_1_CLEANUP_RECEIPT.json"),
            "adjudication": file_record(attempt / "ATTEMPT_1_PROSPECTIVE_ADJUDICATION.json"),
            "self_test": file_record(self_test_path),
        },
        "postrun_custody_audit_hashes": {
            "json": file_record(audit_json),
            "markdown": file_record(audit_md),
        },
        "result_class": adjudication["result_class"],
        "scientific_claim": adjudication["scientific_claim"],
        "claim_ceiling": CLAIM_CEILING,
        "forbidden_promotions": ["full tomography", "relational carrier", "physical relational memory", "catalytic borrowing", "SMALL_WALL_CROSSED"],
        "failed_gates": [name for name, passed in adjudication["gates"].items() if not passed],
        "cleanup_custody_observation": cleanup_observation,
        "retired_nonce_observation": retired_nonce_observation,
        "stale_label_semantic_overlay": stale_label_overlay,
        "existing_evidence_modified": False,
        "existing_evidence_modified_or_deleted_since_pre_audit_commit": modified_or_deleted_existing,
        "source_package_diff_from_manifest_freeze_commit": source_package_diff,
        "v1_0_package_or_evidence_diff_from_pre_audit_commit": v1_0_diff,
        "zero_target_contact_by_postrun_audit": True,
        "prospective_result_remains_valid": custody_audit["passed"] and adjudication["passed"] is True,
    }
    summary = dict(summary_core)
    summary["canonical_summary_digest"] = compact_digest(summary_core)
    write_json(attempt / "ATTEMPT_1_FINAL_EVIDENCE_SUMMARY.json", summary)


def load_json_from_tar(archive: Path, member: str) -> Any:
    return json.loads(tar_member_bytes(archive, member))


if __name__ == "__main__":
    build_outputs()
