#!/usr/bin/env python3
"""Prospective physical adjudicator for relation-lifetime matched permutations.

The functions here are offline and fixture-driven until a separately authorized
live transaction provides raw evidence. All negative and invalid paths fail
closed with no positive scientific claim.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import random
import re
import tarfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import relation_lifetime_public as pub


RESULT_CONFIRMED = "FAMILY10H_RELATION_LIFETIME_COORDINATE_CONFIRMED_PROSPECTIVE"
RESULT_NOT_CONFIRMED = "FAMILY10H_RELATION_LIFETIME_COORDINATE_NOT_CONFIRMED_PROSPECTIVE"
RESULT_CANDIDATE = "FAMILY10H_RELATION_LIFETIME_COORDINATE_CANDIDATE_PROSPECTIVE"
RESULT_INVALID = "FAMILY10H_RELATION_LIFETIME_COORDINATE_CUSTODY_INVALID"

POSITIVE_CLAIM = pub.MAXIMUM_FUTURE_CLAIM
NEGATIVE_CLAIM = pub.NEGATIVE_FUTURE_CLAIM
FIXTURE_SOURCE_SHA = "4" * 40
FIXTURE_FREEZE_SHA = "5" * 40
_FIXTURE_ARCHIVE_CACHE: dict[tuple[str, str], dict[str, Any]] = {}


def physical_threshold_contract() -> dict[str, Any]:
    contract = {
        "schema": "FAMILY10H_RELATION_LIFETIME_PHYSICAL_THRESHOLD_CONTRACT_V1",
        "threshold_status": "prospective_physical_thresholds_frozen_before_relation_lifetime_acquisition",
        "scalar_evidence_provenance": {
            "basis": "sealed_family10h_v1_1_scalar_q_readout_attempt_1_and_pre_run_logic",
            **pub.SCALAR_EVIDENCE_PROVENANCE,
            "archive_sha256": "0f92bcd4c00ee78b7e78e84c86bf375ee1caf4ca8c52ae49166ea809f16ff041",
            "relation_lifetime_target_data_used": False,
            "not_relation_source_authority": True,
        },
        "absolute_thresholds": {
            "r_match_abs_min": 512.0,
            "alive_R_match_abs_min": 512.0,
            "dead_R_match_abs_max": 128.0,
            "delta_R_lifetime_abs_min": 512.0,
            "control_null_abs_max": 128.0,
            "source_off_abs_max": 128.0,
            "scalar_D_single_drift_abs_max": 128.0,
            "scalar_D_single_lifetime_delta_abs_max": 128.0,
        },
        "relative_thresholds": {
            "q_slope_relative_disagreement_max": 0.25,
            "q_intercept_abs_max": 128.0,
            "heldout_relative_error_max": 0.25,
        },
        "resampling_thresholds": {
            "bootstrap_sign_fraction_min": 0.95,
            "label_scramble_abs_max": 128.0,
            "matched_permutation_null_abs_max": 128.0,
        },
        "provenance_law": {
            "r_match_abs_min": "four times the sealed v1.1 source-off absolute null ceiling",
            "control_null_abs_max": "sealed v1.1 source-off absolute null ceiling",
            "scalar_D_single_drift_abs_max": "one sealed v1.1 source-off absolute null ceiling",
            "no_post_run_revision": True,
            "synthetic_thresholds_are_not_physical_thresholds": True,
        },
    }
    contract["threshold_contract_sha256"] = pub.digest(
        {k: v for k, v in contract.items() if k != "threshold_contract_sha256"}
    )
    return contract


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((value - m) ** 2 for value in values) / (len(values) - 1))


def quantile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * fraction
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def signed_distribution(values: list[float]) -> dict[str, Any]:
    abs_values = [abs(value) for value in values]
    signs = Counter(1 if value > 0 else -1 if value < 0 else 0 for value in values)
    nonzero = signs[1] + signs[-1]
    return {
        "count": len(values),
        "mean": mean(values),
        "abs_mean": mean(abs_values),
        "max_abs": max(abs_values) if abs_values else 0.0,
        "q50_abs": quantile(abs_values, 0.50),
        "q90_abs": quantile(abs_values, 0.90),
        "q95_abs": quantile(abs_values, 0.95),
        "sign_counts": {str(key): signs[key] for key in [-1, 0, 1]},
        "sign_balance": abs(signs[1] - signs[-1]) / nonzero if nonzero else 0.0,
    }


def claim_boundary(observed: bool) -> dict[str, Any]:
    return {
        "reproducible_relation_match_coordinate_observed": observed,
        "maximum_future_claim": POSITIVE_CLAIM,
        "full_carrier_state_tomography_established": False,
        "physical_relational_memory_established": False,
        "catalytic_borrowing_established": False,
        "r2_restoration_established": False,
        "small_wall_crossed": False,
    }


def fail_closed_result(result_class: str, validation: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "FAMILY10H_RELATION_LIFETIME_PHYSICAL_ADJUDICATION_V1",
        "result_class": result_class,
        "passed": False,
        "scientific_claim": NEGATIVE_CLAIM,
        "claim_boundary": claim_boundary(False),
        "validation": validation,
    }


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def strict_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")


def jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(strict_json_bytes(row) + b"\n" for row in rows)


def parse_jsonl_bytes(data: bytes) -> list[Any]:
    rows = []
    for line in data.decode("utf-8").splitlines():
        if not line:
            raise ValueError("blank JSONL line")
        rows.append(json.loads(line))
    return rows


def tar_member_bytes(archive_bytes: bytes) -> dict[str, bytes]:
    members: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:*") as tf:
        for member in tf.getmembers():
            if not member.isfile():
                continue
            extracted = tf.extractfile(member)
            if extracted is None:
                continue
            members[member.name] = extracted.read()
    return members


def find_member(members: dict[str, bytes], suffix: str) -> str | None:
    matches = [name for name in sorted(members) if name == suffix or name.endswith("/" + suffix)]
    return matches[0] if len(matches) == 1 else None


def evidence_inventory_for_members(members: dict[str, bytes], member_map: dict[str, str]) -> dict[str, dict[str, Any]]:
    inventory: dict[str, dict[str, Any]] = {}
    for label, member in member_map.items():
        data = members[member]
        inventory[label] = {
            "member": member,
            "sha256": sha256_bytes(data),
            "size_bytes": len(data),
        }
    return inventory


def custody_envelope_from_archive_bytes(archive_bytes: bytes, *, archive_path: str | None = None, embed_archive_bytes: bool = False) -> dict[str, Any]:
    members = tar_member_bytes(archive_bytes)
    required_suffixes = {
        "raw_records": "raw_records.jsonl",
        "source_death_receipts": "source_death_receipts.jsonl",
        "feature_freeze": "feature_freeze.json",
        "target_execution_receipt": "target_execution_receipt.json",
        "manifest": "RELATION_LIFETIME_IMPLEMENTATION_MANIFEST.json",
        "runtime_binary": "relation_lifetime_runtime",
        "deployment_custody": pub.DEPLOYMENT_CUSTODY_FILENAME,
    }
    member_map: dict[str, str] = {}
    for label, suffix in required_suffixes.items():
        member = find_member(members, suffix)
        if member is None:
            raise ValueError(f"required archive member missing or ambiguous: {suffix}")
        member_map[label] = member
    manifest = json.loads(members[member_map["manifest"]].decode("utf-8"))
    deployment = json.loads(members[member_map["deployment_custody"]].decode("utf-8"))
    envelope = {
        "schema": "FAMILY10H_RELATION_LIFETIME_CRYPTOGRAPHIC_CUSTODY_ENVELOPE_V2",
        "target_execution_receipt_sha256": sha256_bytes(members[member_map["target_execution_receipt"]]),
        "runtime_sha256": sha256_bytes(members[member_map["runtime_binary"]]),
        "manifest_sha256": sha256_bytes(members[member_map["manifest"]]),
        "source_authority_sha": manifest.get("authority_binding", {}).get("relation_source_authority_commit"),
        "freeze_sha": deployment.get("relation_manifest_freeze_commit"),
        "copied_back_archive_sha256": sha256_bytes(archive_bytes),
        "copied_back_archive_size": len(archive_bytes),
        "evidence_inventory": evidence_inventory_for_members(members, member_map),
    }
    if archive_path is not None:
        envelope["copied_back_archive_path"] = archive_path
    if embed_archive_bytes:
        envelope["copied_back_archive_bytes_b64"] = base64.b64encode(archive_bytes).decode("ascii")
    return envelope


def custody_envelope_from_archive_path(archive_path: str | Path) -> dict[str, Any]:
    path = Path(archive_path)
    return custody_envelope_from_archive_bytes(path.read_bytes(), archive_path=str(path), embed_archive_bytes=False)


def fixture_archive_packet_material(packet: dict[str, Any], schedule: dict[str, Any], mode: str) -> dict[str, Any]:
    cache_key = (schedule["schedule_sha256"], mode)
    cached = _FIXTURE_ARCHIVE_CACHE.get(cache_key)
    if cached is not None:
        return json.loads(json.dumps(cached))
    feature_freeze = packet["feature_freeze"]
    target_receipt = packet["target_execution_receipt"]
    manifest = {
        "schema": "FAMILY10H_RELATION_LIFETIME_FIXTURE_MANIFEST_V1",
        "authority_binding": {"relation_source_authority_commit": FIXTURE_SOURCE_SHA},
        "schedule_sha256": schedule["schedule_sha256"],
    }
    deployment = {
        "schema": "FAMILY10H_RELATION_LIFETIME_DEPLOYMENT_CUSTODY_V1",
        "relation_source_authority_commit": FIXTURE_SOURCE_SHA,
        "relation_manifest_freeze_commit": FIXTURE_FREEZE_SHA,
        "manifest_file_sha256": sha256_bytes(strict_json_bytes(manifest)),
        "transaction_run_id": pub.TRANSACTION_RUN_ID,
    }
    archive_members = {
        "RELATION_LIFETIME_IMPLEMENTATION_MANIFEST.json": strict_json_bytes(manifest),
        pub.DEPLOYMENT_CUSTODY_FILENAME: strict_json_bytes(deployment),
        "relation_lifetime_runtime": b"fixture-runtime-binary",
        f"{pub.OWNED_OUTPUT_PARENT_NAME}/attempt_1/raw_records.jsonl": jsonl_bytes(packet["raw_records"]),
        f"{pub.OWNED_OUTPUT_PARENT_NAME}/attempt_1/source_death_receipts.jsonl": jsonl_bytes(packet["source_death_receipts"]),
        f"{pub.OWNED_OUTPUT_PARENT_NAME}/attempt_1/feature_freeze.json": strict_json_bytes(feature_freeze),
        f"{pub.OWNED_OUTPUT_PARENT_NAME}/attempt_1/target_execution_receipt.json": strict_json_bytes(target_receipt),
    }
    archive_io = io.BytesIO()
    with tarfile.open(fileobj=archive_io, mode="w") as tf:
        for name, data in sorted(archive_members.items()):
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            info.mtime = 0
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            tf.addfile(info, io.BytesIO(data))
    archive_bytes = archive_io.getvalue()
    material = {
        "custody_envelope": custody_envelope_from_archive_bytes(archive_bytes, embed_archive_bytes=True),
        "target_execution_receipt": target_receipt,
    }
    _FIXTURE_ARCHIVE_CACHE[cache_key] = json.loads(json.dumps(material))
    return material


def archive_bytes_from_custody(custody: dict[str, Any]) -> tuple[bytes | None, str | None]:
    path = custody.get("copied_back_archive_path")
    if isinstance(path, str):
        candidate = Path(path)
        if candidate.exists():
            return candidate.read_bytes(), None
        return None, "copied-back archive path missing"
    encoded = custody.get("copied_back_archive_bytes_b64")
    if isinstance(encoded, str):
        try:
            return base64.b64decode(encoded.encode("ascii"), validate=True), None
        except Exception:
            return None, "copied-back archive bytes malformed"
    return None, "copied-back archive bytes or path missing"


def validate_custody_envelope(packet: dict[str, Any], custody: dict[str, Any], schedule: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    archive_bytes, archive_error = archive_bytes_from_custody(custody)
    if archive_error is not None or archive_bytes is None:
        return [archive_error or "copied-back archive unavailable"]
    archive_sha = sha256_bytes(archive_bytes)
    archive_size = len(archive_bytes)
    if custody.get("copied_back_archive_sha256") != archive_sha:
        failures.append("copied-back archive sha mismatch")
    if custody.get("copied_back_archive_size") != archive_size:
        failures.append("copied-back archive size mismatch")
    try:
        members = tar_member_bytes(archive_bytes)
    except Exception as exc:
        return [f"copied-back archive unreadable: {exc}"]
    required_suffixes = {
        "raw_records": "raw_records.jsonl",
        "source_death_receipts": "source_death_receipts.jsonl",
        "feature_freeze": "feature_freeze.json",
        "target_execution_receipt": "target_execution_receipt.json",
        "manifest": "RELATION_LIFETIME_IMPLEMENTATION_MANIFEST.json",
        "runtime_binary": "relation_lifetime_runtime",
        "deployment_custody": pub.DEPLOYMENT_CUSTODY_FILENAME,
    }
    inventory = custody.get("evidence_inventory")
    if not isinstance(inventory, dict):
        return failures + ["evidence inventory missing"]
    member_map: dict[str, str] = {}
    for label, suffix in required_suffixes.items():
        entry = inventory.get(label)
        member = entry.get("member") if isinstance(entry, dict) else None
        if not isinstance(member, str):
            member = find_member(members, suffix)
        if member is None or member not in members:
            failures.append(f"{label} archive member missing")
            continue
        member_map[label] = member
        data = members[member]
        if not isinstance(entry, dict):
            failures.append(f"{label} inventory entry missing")
            continue
        if entry.get("sha256") != sha256_bytes(data):
            failures.append(f"{label} inventory hash mismatch")
        if entry.get("size_bytes") != len(data):
            failures.append(f"{label} inventory size mismatch")
    if len(member_map) != len(required_suffixes):
        return failures
    try:
        archived_raw = parse_jsonl_bytes(members[member_map["raw_records"]])
        archived_death = parse_jsonl_bytes(members[member_map["source_death_receipts"]])
        archived_feature = json.loads(members[member_map["feature_freeze"]].decode("utf-8"))
        archived_target = json.loads(members[member_map["target_execution_receipt"]].decode("utf-8"))
        archived_manifest = json.loads(members[member_map["manifest"]].decode("utf-8"))
        archived_deployment = json.loads(members[member_map["deployment_custody"]].decode("utf-8"))
    except Exception as exc:
        return failures + [f"archive member parse failure: {exc}"]
    if archived_raw != packet.get("raw_records"):
        failures.append("raw records archive member mismatch")
    if archived_death != packet.get("source_death_receipts"):
        failures.append("source-death archive member mismatch")
    if archived_feature != packet.get("feature_freeze"):
        failures.append("feature-freeze archive member mismatch")
    if archived_target != packet.get("target_execution_receipt"):
        failures.append("target receipt archive member mismatch")
    if custody.get("target_execution_receipt_sha256") != sha256_bytes(members[member_map["target_execution_receipt"]]):
        failures.append("target execution receipt sha mismatch")
    if custody.get("runtime_sha256") != sha256_bytes(members[member_map["runtime_binary"]]):
        failures.append("runtime binary sha mismatch")
    if custody.get("manifest_sha256") != sha256_bytes(members[member_map["manifest"]]):
        failures.append("manifest file sha mismatch")
    source_sha = archived_manifest.get("authority_binding", {}).get("relation_source_authority_commit")
    if custody.get("source_authority_sha") != source_sha:
        failures.append("source authority sha mismatch")
    if archived_deployment.get("relation_source_authority_commit") != source_sha:
        failures.append("deployment source authority mismatch")
    if custody.get("freeze_sha") != archived_deployment.get("relation_manifest_freeze_commit"):
        failures.append("freeze sha mismatch")
    if archived_manifest.get("schedule_sha256") not in {None, schedule.get("schedule_sha256")}:
        failures.append("manifest schedule binding mismatch")
    return failures


def validate_physical_packet(
    packet: dict[str, Any],
    schedule: dict[str, Any],
    *,
    expected_physical_measurement: bool = True,
    require_custody: bool = True,
) -> dict[str, Any]:
    failures: list[str] = []
    raw_records = packet.get("raw_records")
    source_death = packet.get("source_death_receipts")
    feature_freeze = packet.get("feature_freeze")
    custody = packet.get("custody_envelope")
    if not isinstance(raw_records, list):
        return {"passed": False, "failures": ["raw_records missing"]}
    if not isinstance(source_death, list):
        return {"passed": False, "failures": ["source_death_receipts missing"]}
    schedule_rows = schedule["rows"]
    if len(raw_records) != len(schedule_rows):
        failures.append("raw record count mismatch")
    if len(source_death) != len(schedule_rows):
        failures.append("source-death receipt count mismatch")
    schedule_by_id = {row["tuple_id"]: row for row in schedule_rows}
    if len(schedule_by_id) != len(schedule_rows):
        failures.append("schedule tuple_id duplicate")
    trusted_schedule_fields = list(pub.SCHEDULE_COLUMNS)
    seen = set()
    for index, record in enumerate(raw_records):
        if not isinstance(record, dict):
            failures.append("raw record malformed")
            break
        if index >= len(schedule_rows):
            failures.append("raw record count exceeds schedule")
            break
        ordered_expected = schedule_rows[index]
        tuple_id = record.get("tuple_id")
        expected = schedule_by_id.get(tuple_id)
        if expected is None:
            failures.append("unexpected tuple_id")
            continue
        if tuple_id != ordered_expected["tuple_id"]:
            failures.append("raw record execution order mismatch")
        if tuple_id in seen:
            failures.append("duplicate raw tuple_id")
        seen.add(tuple_id)
        for field in trusted_schedule_fields:
            if record.get(field) != expected.get(field):
                failures.append(f"{field} mismatch")
        for metric in ["dirty_probe_response", "change_to_dirty", "cpu_cycles", "duration_ns"]:
            if type(record.get(metric)) not in {int, float}:
                failures.append(f"{metric} not numeric")
        if record.get("physical_measurement") is not expected_physical_measurement:
            failures.append("physical measurement flag mismatch")
        expected_alive = expected.get("source_lifetime") == "alive_during_query"
        expected_custody = "source_alive_during_query" if expected_alive else "source_dead_before_query"
        if record.get("process_custody") != expected_custody:
            failures.append("process custody mismatch")
        if record.get("source_alive_at_query_start") is not expected_alive:
            failures.append("source alive-at-query-start proof mismatch")
        if record.get("source_alive_at_query_end") is not expected_alive:
            failures.append("source alive-at-query-end proof mismatch")
        if expected_alive and record.get("source_exit_monotonic_ns", 0) < record.get("query_end_monotonic_ns", 1):
            failures.append("alive source exited before query ended")
        if (not expected_alive) and record.get("source_exit_monotonic_ns", 1) > record.get("query_start_monotonic_ns", 0):
            failures.append("dead source had not exited before query")
        if record.get("expected_pmu_group") != pub.PMU_GROUP["name"]:
            failures.append("expected PMU event group mismatch")
        if record.get("pmu_event_group") != pub.PMU_GROUP["name"]:
            failures.append("PMU event group mismatch")
        if record.get("pmu_events") != pub.PMU_GROUP["events"]:
            failures.append("PMU event identity mismatch")
        event_ids = record.get("event_ids")
        if not isinstance(event_ids, dict) or set(event_ids) != set(pub.PMU_GROUP["events"]):
            failures.append("PMU event IDs missing")
        elif len(set(event_ids.values())) != len(event_ids) or not all(isinstance(value, int) and value > 0 for value in event_ids.values()):
            failures.append("PMU event IDs invalid or non-distinct")
        time_enabled = record.get("time_enabled")
        time_running = record.get("time_running")
        if type(time_enabled) not in {int, float} or time_enabled <= 0:
            failures.append("time_enabled invalid")
        if type(time_running) not in {int, float} or time_running <= 0 or (type(time_enabled) in {int, float} and time_running > time_enabled):
            failures.append("time_running invalid")
        if record.get("source_cpu_before") != expected["source_cpu_expected"] or record.get("source_cpu_after") != expected["source_cpu_expected"]:
            failures.append("source CPU custody mismatch")
        if record.get("receiver_cpu_before") != expected["receiver_cpu_expected"] or record.get("receiver_cpu_after") != expected["receiver_cpu_expected"]:
            failures.append("receiver CPU custody mismatch")
        if record.get("positive_physical_claim") is True:
            failures.append("positive claim leakage in raw record")
        if len(failures) > 32:
            break
    death_by_id = {row.get("tuple_id"): row for row in source_death if isinstance(row, dict)}
    if len(death_by_id) != len(source_death):
        failures.append("source-death receipt duplicate or malformed")
    for index, expected in enumerate(schedule_rows):
        receipt = source_death[index] if index < len(source_death) else None
        if not isinstance(receipt, dict):
            failures.append("source-death receipt missing")
            break
        if receipt.get("tuple_id") != expected["tuple_id"]:
            failures.append("source-death receipt execution order mismatch")
        if receipt.get("tuple_id") != expected["tuple_id"] or receipt.get("execution_ordinal") != expected["execution_ordinal"]:
            failures.append("source-death tuple identity mismatch")
        if receipt.get("source_pid") != receipt.get("waitpid_pid"):
            failures.append("source PID/waitpid mismatch")
        if receipt.get("waitpid_status") != "exited_0":
            failures.append("source wait status mismatch")
        expected_alive = expected.get("source_lifetime") == "alive_during_query"
        expected_custody = "source_alive_during_query" if expected_alive else "source_dead_before_query"
        for field in ["source_lifetime", "lifetime_pair_id", "lifetime_execution_order", "lifetime_hold_ns"]:
            if receipt.get(field) != expected.get(field):
                failures.append(f"source lifecycle {field} mismatch")
        if receipt.get("process_custody") != expected_custody:
            failures.append("source lifecycle custody mismatch")
        if receipt.get("source_alive_at_query_start") is not expected_alive:
            failures.append("source lifecycle alive-start mismatch")
        if receipt.get("source_alive_at_query_end") is not expected_alive:
            failures.append("source lifecycle alive-end mismatch")
        if receipt.get("query_selected_after_waitpid") is not (not expected_alive):
            failures.append("query/waitpid lifetime order mismatch")
        if receipt.get("source_alive_during_query") is not expected_alive:
            failures.append("source lifecycle alive-during-query mismatch")
        if expected_alive and receipt.get("source_exit_monotonic_ns", 0) < receipt.get("query_end_monotonic_ns", 1):
            failures.append("alive lifecycle exited before query ended")
        if (not expected_alive) and receipt.get("source_exit_monotonic_ns", 1) > receipt.get("query_start_monotonic_ns", 0):
            failures.append("dead lifecycle had not exited before query")
        if receipt.get("source_helper_survives") is not False:
            failures.append("source helper survived query")
        if receipt.get("open_source_ipc_after_waitpid") != 0:
            failures.append("open source IPC after waitpid")
        if receipt.get("post_observation_query_or_window_selection") is not False:
            failures.append("post-observation selection")
        if receipt.get("source_cpu_before") != expected["source_cpu_expected"] or receipt.get("source_cpu_after") != expected["source_cpu_expected"]:
            failures.append("source-death CPU custody mismatch")
        if receipt.get("physical_measurement") is not expected_physical_measurement:
            failures.append("source-death physical measurement mismatch")
        if len(failures) > 48:
            break
    if not isinstance(feature_freeze, dict):
        failures.append("feature_freeze missing")
    else:
        if feature_freeze.get("physical_measurement") is not expected_physical_measurement:
            failures.append("feature freeze physical measurement mismatch")
        if feature_freeze.get("raw_record_count") != len(schedule_rows):
            failures.append("feature freeze raw-record count mismatch")
        if feature_freeze.get("primary_endpoint") != "dirty_probe_response":
            failures.append("feature freeze primary endpoint mismatch")
        if feature_freeze.get("secondary_endpoints") != ["change_to_dirty", "cpu_cycles", "duration_ns"]:
            failures.append("feature freeze secondary endpoints mismatch")
        if feature_freeze.get("schedule_sha256") != schedule.get("schedule_sha256"):
            failures.append("feature freeze schedule binding mismatch")
        if feature_freeze.get("post_observation_feature_selection") is not False:
            failures.append("post-observation feature selection")
    if not require_custody:
        pass
    elif not isinstance(custody, dict):
        failures.append("custody envelope missing")
    else:
        for field in [
            "target_execution_receipt_sha256",
            "runtime_sha256",
            "manifest_sha256",
            "copied_back_archive_sha256",
        ]:
            value = custody.get(field)
            if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
                failures.append(f"{field} missing or malformed")
        for field in ["source_authority_sha", "freeze_sha"]:
            value = custody.get(field)
            if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{40}", value):
                failures.append(f"{field} missing or malformed")
        if not isinstance(custody.get("copied_back_archive_size"), int) or custody.get("copied_back_archive_size") <= 0:
            failures.append("copied-back archive size missing")
        failures.extend(validate_custody_envelope(packet, custody, schedule))
    return {
        "passed": not failures,
        "failures": failures[:64],
        "raw_record_count": len(raw_records),
        "source_death_receipt_count": len(source_death),
        "expected_physical_measurement": expected_physical_measurement,
        "custody_required": require_custody,
    }


def relation_cells_by_block(rows: list[dict[str, Any]]) -> dict[str, dict[tuple[str, str], float]]:
    by_block: dict[str, dict[tuple[str, str], float]] = defaultdict(dict)
    for row in rows:
        if row.get("row_role") == "relation_matrix":
            by_block[row["block_id"]][(row["r_prepare"], row["r_query"])] = float(row["dirty_probe_response"])
    return by_block


def r_match_block_records(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    failures: list[str] = []
    first_by_block: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("row_role") == "relation_matrix" and row.get("block_id") not in first_by_block:
            first_by_block[row["block_id"]] = row
    for block_id, cells in relation_cells_by_block(rows).items():
        missing = [cell for cell in pub.RELATION_CELLS if cell not in cells]
        if missing:
            failures.append(f"{block_id} missing relation cells {missing!r}")
            continue
        r_match = 0.5 * (
            (cells[("relation_r0", "relation_r0")] + cells[("relation_r1", "relation_r1")])
            - (cells[("relation_r0", "relation_r1")] + cells[("relation_r1", "relation_r0")])
        )
        first = first_by_block[block_id]
        records.append(
            {
                "block_id": block_id,
                "lifetime_pair_id": first["lifetime_pair_id"],
                "source_lifetime": first["source_lifetime"],
                "R_match": r_match,
                "session": first["session"],
                "replicate": first["replicate"],
                "mapping": first["mapping"],
                "delay_label": first["delay_label"],
                "source_order": first["source_order"],
                "query_order": first["query_order"],
                "q": first["q"],
                "cyclic_origin": first["cyclic_origin"],
            }
        )
    return records, failures


def r_match_values(rows: list[dict[str, Any]]) -> tuple[list[float], list[str]]:
    records, failures = r_match_block_records(rows)
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_pair[record["lifetime_pair_id"]].append(record)
    values: list[float] = []
    for pair_id, pair_records in by_pair.items():
        by_lifetime = {record["source_lifetime"]: record for record in pair_records}
        if set(by_lifetime) != {"alive_during_query", "dead_before_query"} or len(pair_records) != 2:
            failures.append(f"{pair_id} missing alive/dead lifetime twin")
            continue
        alive = by_lifetime["alive_during_query"]
        dead = by_lifetime["dead_before_query"]
        for factor in ["session", "replicate", "mapping", "delay_label", "source_order", "query_order", "q", "cyclic_origin"]:
            if alive[factor] != dead[factor]:
                failures.append(f"{pair_id} lifetime twin {factor} mismatch")
        values.append(alive["R_match"] - dead["R_match"])
    return values, failures

def lifetime_component_values(rows: list[dict[str, Any]]) -> tuple[list[float], list[float], list[float], list[str]]:
    records, failures = r_match_block_records(rows)
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_pair[record["lifetime_pair_id"]].append(record)
    alive_values: list[float] = []
    dead_values: list[float] = []
    deltas: list[float] = []
    for pair_id, pair_records in by_pair.items():
        by_lifetime = {record["source_lifetime"]: record for record in pair_records}
        if set(by_lifetime) != {"alive_during_query", "dead_before_query"} or len(pair_records) != 2:
            failures.append(f"{pair_id} missing alive/dead lifetime twin")
            continue
        alive = by_lifetime["alive_during_query"]
        dead = by_lifetime["dead_before_query"]
        alive_values.append(alive["R_match"])
        dead_values.append(dead["R_match"])
        deltas.append(alive["R_match"] - dead["R_match"])
    return alive_values, dead_values, deltas, failures


def effect_summary(rows: list[dict[str, Any]], threshold_contract: dict[str, Any]) -> dict[str, Any]:
    alive_values, dead_values, values, failures = lifetime_component_values(rows)
    effect = mean(values)
    floor = threshold_contract["absolute_thresholds"].get("delta_R_lifetime_abs_min", threshold_contract["absolute_thresholds"]["r_match_abs_min"])
    alive_floor = threshold_contract["absolute_thresholds"].get("alive_R_match_abs_min", floor)
    dead_ceiling = threshold_contract["absolute_thresholds"].get("dead_R_match_abs_max", 128.0)
    distribution = signed_distribution(values)
    alive_distribution = signed_distribution(alive_values)
    dead_distribution = signed_distribution(dead_values)
    return {
        "passed": not failures
        and abs(effect) >= floor
        and distribution["abs_mean"] >= floor
        and abs(mean(alive_values)) >= alive_floor
        and alive_distribution["abs_mean"] >= alive_floor
        and abs(mean(dead_values)) <= dead_ceiling
        and dead_distribution["abs_mean"] <= dead_ceiling,
        "failures": failures,
        "block_count": len(values),
        "R_match_mean": effect,
        "R_match_abs_mean": distribution["abs_mean"],
        "R_match_abs_of_mean": abs(effect),
        "Delta_R_lifetime_mean": effect,
        "Delta_R_lifetime_abs_mean": distribution["abs_mean"],
        "alive_R_match_mean": mean(alive_values),
        "dead_R_match_mean": mean(dead_values),
        "alive_R_match_abs_mean": alive_distribution["abs_mean"],
        "dead_R_match_abs_mean": dead_distribution["abs_mean"],
        "R_match_std": stdev(values),
        "R_match_min": min(values) if values else 0.0,
        "R_match_max": max(values) if values else 0.0,
        "R_match_max_abs": distribution["max_abs"],
        "R_match_q50_abs": distribution["q50_abs"],
        "R_match_q90_abs": distribution["q90_abs"],
        "R_match_q95_abs": distribution["q95_abs"],
        "sign_counts": distribution["sign_counts"],
        "sign_balance": distribution["sign_balance"],
        "threshold": floor,
        "alive_threshold": alive_floor,
        "dead_threshold": dead_ceiling,
        "sign": 1 if effect > 0 else -1 if effect < 0 else 0,
    }

def group_rows(rows: list[dict[str, Any]], factor: str) -> dict[Any, list[dict[str, Any]]]:
    grouped: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row[factor]].append(row)
    return dict(grouped)


def strata_report(rows: list[dict[str, Any]], threshold_contract: dict[str, Any], factors: list[str]) -> dict[str, Any]:
    report = {}
    for factor in factors:
        by_level = {}
        for level, level_rows in sorted(group_rows(rows, factor).items(), key=lambda item: repr(item[0])):
            by_level[str(level)] = effect_summary(level_rows, threshold_contract)
        report[factor] = {
            "levels": by_level,
            "passed": all(item["passed"] for item in by_level.values()),
        }
    return report


def scalar_equivalence(rows: list[dict[str, Any]], threshold_contract: dict[str, Any]) -> dict[str, Any]:
    by_relation_query_q: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for row in rows:
        if row.get("row_role") != "scalar_control":
            continue
        by_relation_query_q[(row["r_prepare"], row["query"], int(row["q"]))].append(float(row["dirty_probe_response"]))
    relation_d: dict[str, dict[int, float]] = defaultdict(dict)
    for relation in pub.RELATIONS:
        for q in pub.Q_VALUES:
            a = by_relation_query_q.get((relation, "query_A", q), [])
            b = by_relation_query_q.get((relation, "query_B", q), [])
            if a and b:
                relation_d[relation][q] = mean(a) - mean(b)
    slopes = {}
    intercepts = {}
    for relation, qmap in relation_d.items():
        xs = [float(q) for q in sorted(qmap)]
        ys = [qmap[int(q)] for q in xs]
        xbar = mean(xs)
        ybar = mean(ys)
        denom = sum((x - xbar) ** 2 for x in xs)
        slope = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys)) / denom if denom else 0.0
        slopes[relation] = slope
        intercepts[relation] = ybar - slope * xbar
    slope_values = list(slopes.values())
    max_slope = max((abs(value) for value in slope_values), default=0.0)
    slope_disagreement = (max(slope_values) - min(slope_values)) / max_slope if max_slope else 0.0
    d_drift = max(
        (abs(relation_d["relation_r0"].get(q, 0.0) - relation_d["relation_r1"].get(q, 0.0)) for q in pub.Q_VALUES),
        default=0.0,
    )
    passed = (
        len(relation_d) == 2
        and d_drift <= threshold_contract["absolute_thresholds"]["scalar_D_single_drift_abs_max"]
        and slope_disagreement <= threshold_contract["relative_thresholds"]["q_slope_relative_disagreement_max"]
        and all(abs(value) <= threshold_contract["relative_thresholds"]["q_intercept_abs_max"] for value in intercepts.values())
    )
    return {
        "passed": passed,
        "relation_D_single_by_q": {key: {str(q): value for q, value in qmap.items()} for key, qmap in relation_d.items()},
        "max_D_single_drift": d_drift,
        "slopes": slopes,
        "intercepts": intercepts,
        "slope_relative_disagreement": slope_disagreement,
    }


def control_nulls(rows: list[dict[str, Any]], threshold_contract: dict[str, Any]) -> dict[str, Any]:
    threshold = threshold_contract["absolute_thresholds"]["control_null_abs_max"]
    by_control: dict[str, list[float]] = defaultdict(list)
    by_stratum: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    stratum_factors = ["session", "replicate", "mapping", "delay_label", "source_order", "query_order", "q", "cyclic_origin"]
    for row in rows:
        if row.get("row_role") == "relation_control":
            value = float(row["dirty_probe_response"])
            control = row["query"]
            by_control[control].append(value)
            for factor in stratum_factors:
                by_stratum[control][f"{factor}={row[factor]}"].append(value)
    controls = {}
    for control in pub.CONTROL_ROWS:
        values = by_control.get(control, [])
        distribution = signed_distribution(values)
        per_stratum = {}
        for key, stratum_values in sorted(by_stratum.get(control, {}).items()):
            item = signed_distribution(stratum_values)
            item["passed"] = bool(stratum_values) and item["max_abs"] <= threshold and item["abs_mean"] <= threshold
            per_stratum[key] = item
        controls[control] = {
            **distribution,
            "per_stratum": per_stratum,
            "passed": bool(values)
            and distribution["abs_mean"] <= threshold
            and distribution["max_abs"] <= threshold
            and distribution["q95_abs"] <= threshold
            and all(item["passed"] for item in per_stratum.values()),
        }
    return {"passed": all(item["passed"] for item in controls.values()), "threshold": threshold, "controls": controls}


def label_scramble_test(rows: list[dict[str, Any]], threshold_contract: dict[str, Any]) -> dict[str, Any]:
    threshold = threshold_contract["resampling_thresholds"]["label_scramble_abs_max"]
    values = []
    for seed in range(64):
        rng = random.Random(0xC0DEC0DE + seed)
        scrambled = []
        block_flip: dict[str, bool] = {}
        for row in rows:
            item = dict(row)
            if item.get("row_role") == "relation_matrix":
                flip = block_flip.setdefault(item["block_id"], rng.random() < 0.5)
                if flip:
                    item["r_query"] = "relation_r1" if item["r_query"] == "relation_r0" else "relation_r0"
            scrambled.append(item)
        values.append(effect_summary(scrambled, threshold_contract)["R_match_abs_of_mean"])
    distribution = signed_distribution(values)
    return {
        "passed": distribution["abs_mean"] <= threshold and distribution["q95_abs"] <= threshold,
        "seed": "0xC0DEC0DE",
        "iterations": len(values),
        "null_distribution": distribution,
        "threshold": threshold,
    }


def heldout_transport(rows: list[dict[str, Any]], threshold_contract: dict[str, Any]) -> dict[str, Any]:
    factors = ["session", "replicate", "mapping", "delay_label", "source_order", "query_order", "q", "cyclic_origin"]
    global_summary = effect_summary(rows, threshold_contract)
    expected_sign = global_summary["sign"]
    report = {}
    for factor in factors:
        levels = {}
        for level in sorted({row[factor] for row in rows}, key=lambda value: repr(value)):
            train_rows = [row for row in rows if row[factor] != level]
            test_rows = [row for row in rows if row[factor] == level]
            test = effect_summary(test_rows, threshold_contract)
            train_absent = all(row[factor] != level for row in train_rows)
            sign_pass = test["sign"] == expected_sign and expected_sign != 0
            levels[str(level)] = {
                "train_count": len(train_rows),
                "test_count": len(test_rows),
                "held_out_level_absent_from_training": train_absent,
                "test_R_match_abs_mean": test["R_match_abs_mean"],
                "test_sign": test["sign"],
                "confidence_interval": [
                    test["R_match_mean"] - 2.0 * test["R_match_std"],
                    test["R_match_mean"] + 2.0 * test["R_match_std"],
                ],
                "null_comparison": "fixed_threshold_no_relation_lifetime_training",
                "passed": train_absent and test["passed"] and sign_pass,
            }
        report[factor] = {"levels": levels, "passed": all(item["passed"] for item in levels.values())}
    return {"passed": all(item["passed"] for item in report.values()), "factors": report}


def bootstrap_stability(rows: list[dict[str, Any]], threshold_contract: dict[str, Any]) -> dict[str, Any]:
    values, failures = r_match_values(rows)
    if failures or not values:
        return {"passed": False, "failures": failures or ["no R_match blocks"]}
    observed_sign = 1 if mean(values) > 0 else -1 if mean(values) < 0 else 0
    rng = random.Random(0xB00757A9)
    sample_means = []
    signs = []
    for _ in range(256):
        sample = [rng.choice(values) for _ in values]
        sample_mean = mean(sample)
        sample_means.append(sample_mean)
        signs.append(1 if sample_mean > 0 else -1 if sample_mean < 0 else 0)
    sign_fraction = sum(1 for sign in signs if sign == observed_sign) / len(signs)
    return {
        "passed": observed_sign != 0 and sign_fraction >= threshold_contract["resampling_thresholds"]["bootstrap_sign_fraction_min"],
        "seed": "0xB00757A9",
        "iterations": len(sample_means),
        "observed_sign": observed_sign,
        "bootstrap_sign_fraction": sign_fraction,
        "sample_mean_distribution": signed_distribution(sample_means),
        "threshold": threshold_contract["resampling_thresholds"]["bootstrap_sign_fraction_min"],
    }


def neutralized_effect(rows: list[dict[str, Any]], threshold_contract: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    """Diagnostic-only replay summary; not used as an adjudication gate."""
    grouped: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in rows:
        if row.get("row_role") == "relation_matrix":
            grouped[tuple(row[key] for key in keys)].append(float(row["dirty_probe_response"]))
    neutralized = []
    for row in rows:
        item = dict(row)
        if item.get("row_role") == "relation_matrix":
            item["dirty_probe_response"] = mean(grouped[tuple(row[key] for key in keys)])
        neutralized.append(item)
    return effect_summary(neutralized, threshold_contract)


def rows_by_block(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_block: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_block[row["block_id"]].append(row)
    return dict(by_block)


def relation_matrix_rows_by_block(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("row_role") == "relation_matrix":
            grouped[row["block_id"]].append(row)
    return dict(grouped)


def scalar_block_baselines(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    baselines: dict[str, dict[str, float]] = {}
    for block_id, block_rows in rows_by_block(rows).items():
        scalar_rows = [row for row in block_rows if row.get("row_role") == "scalar_control"]
        control_rows = [row for row in block_rows if row.get("row_role") == "relation_control"]
        a_values = [float(row["dirty_probe_response"]) for row in scalar_rows if row.get("query") == "query_A"]
        b_values = [float(row["dirty_probe_response"]) for row in scalar_rows if row.get("query") == "query_B"]
        control_values = [float(row["dirty_probe_response"]) for row in control_rows]
        scalar_center = mean([*a_values, *b_values]) if a_values or b_values else 0.0
        baselines[block_id] = {
            "scalar_center": scalar_center,
            "D_single": (mean(a_values) - mean(b_values)) if a_values and b_values else 0.0,
            "route_pressure_control": mean(control_values) if control_values else 0.0,
        }
    return baselines


def solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float]:
    size = len(vector)
    augmented = [row[:] + [vector[index]] for index, row in enumerate(matrix)]
    for pivot in range(size):
        best = max(range(pivot, size), key=lambda row: abs(augmented[row][pivot]))
        if abs(augmented[best][pivot]) < 1e-9:
            augmented[pivot][pivot] += 1e-6
            best = pivot
        if best != pivot:
            augmented[pivot], augmented[best] = augmented[best], augmented[pivot]
        scale = augmented[pivot][pivot]
        if abs(scale) < 1e-12:
            scale = 1e-12
        for column in range(pivot, size + 1):
            augmented[pivot][column] /= scale
        for row in range(size):
            if row == pivot:
                continue
            factor = augmented[row][pivot]
            if factor == 0.0:
                continue
            for column in range(pivot, size + 1):
                augmented[row][column] -= factor * augmented[pivot][column]
    return [augmented[row][size] for row in range(size)]


def fit_linear_model(feature_rows: list[list[float]], targets: list[float]) -> list[float]:
    if not feature_rows:
        return []
    width = len(feature_rows[0])
    normal = [[0.0 for _ in range(width)] for _ in range(width)]
    rhs = [0.0 for _ in range(width)]
    for features, target in zip(feature_rows, targets):
        for i, left in enumerate(features):
            rhs[i] += left * target
            for j, right in enumerate(features):
                normal[i][j] += left * right
    for index in range(width):
        normal[index][index] += 1e-6
    return solve_linear_system(normal, rhs)


def dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def category_levels(rows: list[dict[str, Any]], factors: list[str]) -> dict[str, list[Any]]:
    return {factor: sorted({row.get(factor) for row in rows}, key=lambda value: repr(value)) for factor in factors}


def scalar_feature_vector(
    row: dict[str, Any],
    baseline: dict[str, float],
    levels: dict[str, list[Any]],
    *,
    nonlinear: bool,
) -> list[float]:
    q = float(row.get("q", 0.0))
    ordinal = float(row.get("execution_ordinal", 0.0))
    position = float(row.get("block_local_position", 0.0))
    features = [
        1.0,
        q / 2048.0,
        float(row.get("bank_A_work", 0.0)) / 4096.0,
        float(row.get("bank_B_work", 0.0)) / 4096.0,
        float(row.get("total_work", 0.0)) / 4096.0,
        float(baseline.get("scalar_center", 0.0)) / 10000.0,
        float(baseline.get("D_single", 0.0)) / 1000.0,
        ordinal / 40000.0,
    ]
    if nonlinear:
        features.extend(
            [
                (q / 2048.0) ** 2,
                (position / 12.0) ** 2,
                (q / 2048.0) * (position / 12.0),
                math.sin((ordinal % 12.0) * math.pi / 6.0),
                math.cos((ordinal % 12.0) * math.pi / 6.0),
            ]
        )
    else:
        features.append(position / 12.0)
    for factor in ["session", "replicate", "mapping", "delay_label", "source_order", "query_order", "cyclic_origin"]:
        for level in levels.get(factor, []):
            features.append(1.0 if row.get(factor) == level else 0.0)
    return features


def linear_scalar_residual_rows(rows: list[dict[str, Any]], *, nonlinear: bool) -> list[dict[str, Any]]:
    matrix_rows = [row for row in rows if row.get("row_role") == "relation_matrix"]
    baselines = scalar_block_baselines(rows)
    levels = category_levels(matrix_rows, ["session", "replicate", "mapping", "delay_label", "source_order", "query_order", "cyclic_origin"])
    features = [scalar_feature_vector(row, baselines.get(row["block_id"], {}), levels, nonlinear=nonlinear) for row in matrix_rows]
    targets = [float(row["dirty_probe_response"]) for row in matrix_rows]
    coefficients = fit_linear_model(features, targets)
    residual_rows = [dict(row) for row in rows]
    by_tuple = {row["tuple_id"]: row for row in residual_rows}
    for row, feature in zip(matrix_rows, features):
        by_tuple[row["tuple_id"]]["dirty_probe_response"] = float(row["dirty_probe_response"]) - dot(coefficients, feature)
    return residual_rows


def scalar_replay_residual_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return linear_scalar_residual_rows(rows, nonlinear=False)


def nonlinear_scalar_replay_residual_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return linear_scalar_residual_rows(rows, nonlinear=True)


def additive_ab_residual_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    residual_rows = [dict(row) for row in rows]
    output_by_tuple = {row["tuple_id"]: row for row in residual_rows}
    for block_id, matrix_rows in relation_matrix_rows_by_block(rows).items():
        if len(matrix_rows) != len(pub.RELATION_CELLS):
            continue
        grand = mean([float(row["dirty_probe_response"]) for row in matrix_rows])
        prep_means = {
            relation: mean([float(row["dirty_probe_response"]) for row in matrix_rows if row["r_prepare"] == relation])
            for relation in pub.RELATIONS
        }
        query_means = {
            relation: mean([float(row["dirty_probe_response"]) for row in matrix_rows if row["r_query"] == relation])
            for relation in pub.RELATIONS
        }
        for row in matrix_rows:
            predicted = prep_means[row["r_prepare"]] + query_means[row["r_query"]] - grand
            output_by_tuple[row["tuple_id"]]["dirty_probe_response"] = float(row["dirty_probe_response"]) - predicted
    return residual_rows


def control_replay_residual_rows(rows: list[dict[str, Any]], control_name: str) -> list[dict[str, Any]]:
    residual_rows = [dict(row) for row in rows]
    output_by_tuple = {row["tuple_id"]: row for row in residual_rows}
    controls: dict[str, float] = {}
    for block_id, block_rows in rows_by_block(rows).items():
        values = [
            float(row["dirty_probe_response"])
            for row in block_rows
            if row.get("row_role") == "relation_control" and row.get("query") == control_name
        ]
        controls[block_id] = mean(values) if values else 0.0
    for row in rows:
        if row.get("row_role") != "relation_matrix":
            continue
        sign = 1.0 if row.get("relation_match") is True else -1.0
        output_by_tuple[row["tuple_id"]]["dirty_probe_response"] = float(row["dirty_probe_response"]) - sign * controls.get(row["block_id"], 0.0)
    return residual_rows


def matched_permutation_null(rows: list[dict[str, Any]], threshold_contract: dict[str, Any]) -> dict[str, Any]:
    observed = effect_summary(rows, threshold_contract)
    null_values: list[float] = []
    block_cells = []
    for matrix_rows in relation_matrix_rows_by_block(rows).values():
        if len(matrix_rows) != len(pub.RELATION_CELLS):
            continue
        block_cells.append(
            {
                (row["r_prepare"], row["r_query"]): float(row["dirty_probe_response"])
                for row in matrix_rows
            }
        )
    for seed in range(128):
        rng = random.Random(0xA77A5EED + seed)
        sample_r: list[float] = []
        for cells in block_cells:
            shuffled_values = [cells[cell] for cell in pub.RELATION_CELLS]
            rng.shuffle(shuffled_values)
            shuffled_cells = dict(zip(pub.RELATION_CELLS, shuffled_values))
            sample_r.append(
                0.5
                * (
                    (shuffled_cells[("relation_r0", "relation_r0")] + shuffled_cells[("relation_r1", "relation_r1")])
                    - (shuffled_cells[("relation_r0", "relation_r1")] + shuffled_cells[("relation_r1", "relation_r0")])
                )
            )
        null_values.append(abs(mean(sample_r)) if sample_r else 0.0)
    distribution = signed_distribution(null_values)
    threshold = threshold_contract["resampling_thresholds"]["matched_permutation_null_abs_max"]
    return {
        "passed": observed["R_match_abs_of_mean"] >= observed["threshold"]
        and observed["R_match_abs_of_mean"] > distribution["q95_abs"]
        and distribution["q95_abs"] <= threshold,
        "seed": "0xA77A5EED",
        "iterations": len(null_values),
        "observed_R_match_abs_of_mean": observed["R_match_abs_of_mean"],
        "null_distribution": distribution,
        "null_threshold": threshold,
    }


def replay_adversary_report(rows: list[dict[str, Any]], threshold_contract: dict[str, Any]) -> dict[str, Any]:
    scalar_residual = effect_summary(scalar_replay_residual_rows(rows), threshold_contract)
    nonlinear_residual = effect_summary(nonlinear_scalar_replay_residual_rows(rows), threshold_contract)
    additive_residual = effect_summary(additive_ab_residual_rows(rows), threshold_contract)
    route_residual = effect_summary(control_replay_residual_rows(rows, "route_pressure_sham"), threshold_contract)
    distance_residual = effect_summary(control_replay_residual_rows(rows, "distance_control"), threshold_contract)
    matched_null = matched_permutation_null(rows, threshold_contract)
    heldout = heldout_transport(rows, threshold_contract)
    diagnostic_neutralization = {
        "non_gating": True,
        "reason": "group-mean neutralization can algebraically erase or preserve effects and is retained only for audit visibility",
        "examples": {
            "q_only": neutralized_effect(rows, threshold_contract, ["q"]),
            "mapping_session_replicate_delay_q": neutralized_effect(
                rows,
                threshold_contract,
                ["mapping", "session", "replicate", "delay_label", "q"],
            ),
        },
    }
    adversaries = {
        "scalar_q_replay": {
            "model": "least-squares prediction from scalar-control block center, D_single, q, work totals, execution position, and frozen nuisance factors; no relation-match or relation interaction feature",
            "residual_R_match_abs_of_mean": scalar_residual["R_match_abs_of_mean"],
            "residual_R_match_abs_mean": scalar_residual["R_match_abs_mean"],
            "relation_residual_survives": scalar_residual["passed"],
            "passed": scalar_residual["passed"],
        },
        "nonlinear_scalar_q_replay": {
            "model": "distinct frozen nonlinear scalar basis using q^2, execution-position harmonics, q x position, scalar center, D_single, and nuisance factors; no relation-match or relation interaction feature",
            "residual_R_match_abs_of_mean": nonlinear_residual["R_match_abs_of_mean"],
            "residual_R_match_abs_mean": nonlinear_residual["R_match_abs_mean"],
            "relation_residual_survives": nonlinear_residual["passed"],
            "passed": nonlinear_residual["passed"],
        },
        "separable_a_b_marginal_replay": {
            "model": "per-block additive prepare/query main effects without relation interaction",
            "residual_R_match_abs_of_mean": additive_residual["R_match_abs_of_mean"],
            "residual_R_match_abs_mean": additive_residual["R_match_abs_mean"],
            "relation_interaction_residual_survives": additive_residual["passed"],
            "passed": additive_residual["passed"],
        },
        "route_pressure_replay": {
            "model": "block-level residualized contrast using physically scheduled route_pressure_sham observations",
            "residual_R_match_abs_of_mean": route_residual["R_match_abs_of_mean"],
            "residual_R_match_abs_mean": route_residual["R_match_abs_mean"],
            "matched_controls_available": any(
                row.get("row_role") == "relation_control" and row.get("query") == "route_pressure_sham" for row in rows
            ),
            "relation_residual_survives": route_residual["passed"],
            "passed": route_residual["passed"],
        },
        "distance_only_replay": {
            "model": "block-level residualized contrast using physically scheduled distance_control observations",
            "residual_R_match_abs_of_mean": distance_residual["R_match_abs_of_mean"],
            "residual_R_match_abs_mean": distance_residual["R_match_abs_mean"],
            "matched_controls_available": any(
                row.get("row_role") == "relation_control" and row.get("query") == "distance_control" for row in rows
            ),
            "relation_residual_survives": distance_residual["passed"],
            "passed": distance_residual["passed"],
        },
        "matched_permutation_null": matched_null,
        "confounding_heldout": {
            "factors": {name: item["passed"] for name, item in heldout["factors"].items()},
            "passed": heldout["passed"],
        },
    }
    return {
        "schema": "FAMILY10H_RELATION_LIFETIME_SCALAR_REPLAY_ADVERSARY_REPORT_V1",
        "null_threshold": threshold_contract["resampling_thresholds"]["matched_permutation_null_abs_max"],
        "adversaries": adversaries,
        "diagnostic_neutralized_effect": diagnostic_neutralization,
        "passed": all(item["passed"] for item in adversaries.values()),
    }


def source_off_gate(rows: list[dict[str, Any]], threshold_contract: dict[str, Any]) -> dict[str, Any]:
    values = [float(row["dirty_probe_response"]) for row in rows if row.get("row_role") == "source_off"]
    if not values:
        return {
            "status": "not_scheduled",
            "claimed_as_gate": False,
            "passed": True,
            "reason": "frozen schedule contains no legitimate source-off matrix",
        }
    threshold = threshold_contract["absolute_thresholds"]["source_off_abs_max"]
    distribution = signed_distribution(values)
    return {
        "status": "scheduled",
        "claimed_as_gate": True,
        "distribution": distribution,
        "threshold": threshold,
        "passed": distribution["max_abs"] <= threshold,
    }


def adjudicate_physical_packet(
    packet: dict[str, Any],
    schedule: dict[str, Any],
    threshold_contract: dict[str, Any] | None = None,
    *,
    require_custody: bool = True,
) -> dict[str, Any]:
    threshold_contract = threshold_contract or physical_threshold_contract()
    validation = validate_physical_packet(packet, schedule, require_custody=require_custody)
    if not validation["passed"]:
        return fail_closed_result(RESULT_INVALID, validation)
    rows = packet["raw_records"]
    global_effect = effect_summary(rows, threshold_contract)
    factors = ["session", "replicate", "mapping", "delay_label", "source_order", "query_order", "q", "cyclic_origin"]
    strata = strata_report(rows, threshold_contract, factors)
    scalar = scalar_equivalence(rows, threshold_contract)
    controls = control_nulls(rows, threshold_contract)
    scramble = label_scramble_test(rows, threshold_contract)
    heldout = heldout_transport(rows, threshold_contract)
    bootstrap = bootstrap_stability(rows, threshold_contract)
    adversary = replay_adversary_report(rows, threshold_contract)
    source_off = source_off_gate(rows, threshold_contract)
    gates = {
        "validation": validation["passed"],
        "complete_relation_matrix_global": global_effect["passed"],
        "session_strata": strata["session"]["passed"],
        "replicate_strata": strata["replicate"]["passed"],
        "mapping_strata": strata["mapping"]["passed"],
        "delay_strata": strata["delay_label"]["passed"],
        "source_order_strata": strata["source_order"]["passed"],
        "query_order_strata": strata["query_order"]["passed"],
        "q_strata": strata["q"]["passed"],
        "cyclic_origin_strata": strata["cyclic_origin"]["passed"],
        "relation_controls_null": controls["passed"],
        "scalar_q_equivalence": scalar["passed"],
        "label_scramble_collapses": scramble["passed"],
        "heldout_transport": heldout["passed"],
        "bootstrap_stability": bootstrap["passed"],
        "scalar_replay_adversary_rejected": adversary["passed"],
        "source_off_not_claimed_when_unscheduled": source_off["passed"] and (source_off.get("claimed_as_gate") is False or source_off.get("status") == "scheduled"),
        "no_aggregate_rescue": all(item["passed"] for item in strata.values()),
    }
    passed = all(gates.values())
    result_class = RESULT_CONFIRMED if passed else RESULT_NOT_CONFIRMED
    return {
        "schema": "FAMILY10H_RELATION_LIFETIME_PHYSICAL_ADJUDICATION_V1",
        "result_class": result_class,
        "passed": passed,
        "scientific_claim": POSITIVE_CLAIM if passed else NEGATIVE_CLAIM,
        "claim_boundary": claim_boundary(passed),
        "validation": validation,
        "threshold_contract": threshold_contract,
        "R_match_global": global_effect,
        "strata": strata,
        "controls": controls,
        "scalar_q_equivalence": scalar,
        "label_scramble": scramble,
        "heldout_transport": heldout,
        "bootstrap_stability": bootstrap,
        "scalar_replay_adversary": adversary,
        "source_off_gate": source_off,
        "gates": gates,
    }


def _lifetime_amplitude(expected: dict[str, Any], mode: str) -> float:
    alive = expected["source_lifetime"] == "alive_during_query"
    if mode == "positive":
        return 384.0 if alive else 0.0
    if mode in {"both_collapsed", "separable_replay"}:
        return 0.0
    if mode == "both_equally_positive":
        return 384.0
    if mode in {"scalar_replay", "route_pressure", "distance_only"}:
        return 384.0 if alive else 0.0
    if mode == "source_order_specific":
        return 768.0 if alive and expected["source_order"] == "A_then_B" else 0.0
    if mode == "query_order_specific":
        return 768.0 if alive and expected["query_order"] == "AB" else 0.0
    if mode == "mapping_specific":
        return 768.0 if alive and expected["mapping"] == "map0" else 0.0
    if mode == "session_specific":
        return 768.0 if alive and expected["session"] == "session_0" else 0.0
    if mode == "replicate_specific":
        return 768.0 if alive and expected["replicate"] == 0 else 0.0
    if mode == "origin_specific":
        return 2048.0 if alive and expected["cyclic_origin"] == 0 else 0.0
    if mode in {"q_specific", "nonlinear_q_position", "additive_imbalance"}:
        return 768.0 if alive and int(expected["q"]) > 0 else 0.0
    return 0.0


def fixture_packet(schedule: dict[str, Any], mode: str) -> dict[str, Any]:
    raw_records = []
    source_death = []
    malformed_alive_done = False
    malformed_dead_done = False
    for expected in schedule["rows"]:
        q = float(expected["q"])
        base = 10_000.0 + 0.25 * q
        value = base
        alive = expected["source_lifetime"] == "alive_during_query"
        if expected["row_role"] == "relation_matrix":
            sign = 1.0 if expected["relation_match"] else -1.0
            value += sign * _lifetime_amplitude(expected, mode)
        elif expected["row_role"] == "scalar_control":
            value += 60.0 if expected["query"] == "query_A" else -60.0
            if mode == "scalar_replay" and alive:
                value += 420.0 if expected["query"] == "query_A" else -420.0
        elif expected["row_role"] == "relation_control":
            value = 0.0
            if mode == "route_pressure" and expected["query"] == "route_pressure_sham" and alive:
                value = 800.0
            elif mode == "distance_only" and expected["query"] == "distance_control" and alive:
                value = 800.0
        ready = 1_000_000_000 + int(expected["execution_ordinal"]) * 1000
        if alive:
            query_start = ready + 10
            query_end = query_start + 100
            wait_start = query_end
            wait_end = query_end + 1
            source_exit = wait_end
            alive_start = True
            alive_end = True
            query_after_waitpid = False
            process_custody = "source_alive_during_query"
        else:
            wait_start = ready
            wait_end = ready + 1
            source_exit = wait_end
            query_start = source_exit + 1
            query_end = query_start + 100
            alive_start = False
            alive_end = False
            query_after_waitpid = True
            process_custody = "source_dead_before_query"
        if mode == "alive_dies_early" and alive and not malformed_alive_done:
            alive_end = False
            source_exit = query_start
            process_custody = "source_dead_before_query"
            malformed_alive_done = True
        if mode == "source_alive_in_dead_lane" and not alive and not malformed_dead_done:
            alive_start = True
            alive_end = True
            query_after_waitpid = False
            process_custody = "source_alive_during_query"
            malformed_dead_done = True
        record = {
            **expected,
            "dirty_probe_response": value,
            "change_to_dirty": 1.0 if value > base else 0.0,
            "cpu_cycles": 1_000_000.0 + abs(q),
            "duration_ns": 100_000.0 + int(expected["cyclic_origin"]),
            "time_enabled": 100_000.0,
            "time_running": 100_000.0,
            "pmu_event_group": pub.PMU_GROUP["name"],
            "pmu_events": pub.PMU_GROUP["events"],
            "event_ids": {name: idx + 1 for idx, name in enumerate(pub.PMU_GROUP["events"])},
            "source_cpu_before": expected["source_cpu_expected"],
            "source_cpu_after": expected["source_cpu_expected"],
            "receiver_cpu_before": expected["receiver_cpu_expected"],
            "receiver_cpu_after": expected["receiver_cpu_expected"],
            "source_pid": 10000 + expected["execution_ordinal"],
            "source_ready_monotonic_ns": ready,
            "source_exit_monotonic_ns": source_exit,
            "query_start_monotonic_ns": query_start,
            "query_end_monotonic_ns": query_end,
            "source_alive_at_query_start": alive_start,
            "source_alive_at_query_end": alive_end,
            "waitpid_start_monotonic_ns": wait_start,
            "waitpid_end_monotonic_ns": wait_end,
            "process_custody": process_custody,
            "query_hash": 424242 + expected["execution_ordinal"],
            "physical_measurement": True,
            "positive_physical_claim": False,
        }
        raw_records.append(record)
        source_death.append(
            {
                "tuple_id": expected["tuple_id"],
                "execution_ordinal": expected["execution_ordinal"],
                "source_lifetime": expected["source_lifetime"],
                "lifetime_pair_id": expected["lifetime_pair_id"],
                "lifetime_execution_order": expected["lifetime_execution_order"],
                "lifetime_hold_ns": expected["lifetime_hold_ns"],
                "source_pid": 10000 + expected["execution_ordinal"],
                "waitpid_pid": 10000 + expected["execution_ordinal"],
                "waitpid_status": "exited_0",
                "source_ready_monotonic_ns": ready,
                "source_exit_monotonic_ns": source_exit,
                "query_start_monotonic_ns": query_start,
                "query_end_monotonic_ns": query_end,
                "source_alive_at_query_start": alive_start,
                "source_alive_at_query_end": alive_end,
                "source_alive_during_query": alive_start and alive_end,
                "source_helper_survives": False,
                "open_source_ipc_after_waitpid": 0,
                "query_selected_after_waitpid": query_after_waitpid,
                "post_observation_query_or_window_selection": False,
                "process_custody": process_custody,
                "source_cpu_before": expected["source_cpu_expected"],
                "source_cpu_after": expected["source_cpu_expected"],
                "physical_measurement": True,
            }
        )
    packet = {
        "schema": "FAMILY10H_RELATION_LIFETIME_PHYSICAL_FIXTURE_PACKET_V1",
        "mode": mode,
        "raw_records": raw_records,
        "source_death_receipts": source_death,
        "feature_freeze": {
            "schema": "FAMILY10H_RELATION_LIFETIME_FEATURE_FREEZE_V1",
            "schedule_sha256": schedule["schedule_sha256"],
            "primary_endpoint": "dirty_probe_response",
            "secondary_endpoints": ["change_to_dirty", "cpu_cycles", "duration_ns"],
            "raw_record_count": len(raw_records),
            "physical_measurement": True,
            "post_observation_feature_selection": False,
        },
        "target_execution_receipt": {
            "schema": "FAMILY10H_RELATION_LIFETIME_TARGET_EXECUTION_RECEIPT_V1",
            "status": "complete",
            "returncode": 0,
            "raw_record_count": len(raw_records),
            "source_death_receipt_count": len(source_death),
            "feature_freeze_written": True,
            "physical_measurement": True,
            "pmu_opened": True,
            "live_activity": True,
            "small_wall_crossed": False,
        },
    }
    packet.update(fixture_archive_packet_material(packet, schedule, mode))
    return packet

def fail_closed_claim_state(report: dict[str, Any]) -> dict[str, Any]:
    boundary = report.get("claim_boundary", {})
    checks = {
        "result_class_not_confirmed": report.get("result_class") != RESULT_CONFIRMED,
        "positive_scientific_claim_absent": report.get("scientific_claim") != POSITIVE_CLAIM,
        "relation_coordinate_observed_false": boundary.get("reproducible_relation_match_coordinate_observed") is False,
    }
    return {
        "passed": all(checks.values()),
        "result_class": report.get("result_class"),
        "scientific_claim": report.get("scientific_claim"),
        "checks": checks,
    }


def mutated_packet_regression(schedule: dict[str, Any], label: str, mutator: Any) -> dict[str, Any]:
    packet = fixture_packet(schedule, "positive")
    mutator(packet)
    result = adjudicate_physical_packet(packet, schedule)
    claim = fail_closed_claim_state(result)
    return {
        "passed": result.get("result_class") == RESULT_INVALID and claim["passed"],
        "result_class": result.get("result_class"),
        "failures": result.get("validation", {}).get("failures", []),
        "claim_state": claim,
    }


def wrong_sha256(value: str) -> str:
    return ("0" * 64) if value != ("0" * 64) else ("1" * 64)


def wrong_sha1(value: str) -> str:
    return ("0" * 40) if value != ("0" * 40) else ("1" * 40)


def replace_archive_member(packet: dict[str, Any], label: str, replacement: bytes) -> None:
    custody = packet["custody_envelope"]
    archive_bytes, error = archive_bytes_from_custody(custody)
    if error is not None or archive_bytes is None:
        raise AssertionError(error or "archive missing")
    members = tar_member_bytes(archive_bytes)
    member = custody["evidence_inventory"][label]["member"]
    members[member] = replacement
    archive_io = io.BytesIO()
    with tarfile.open(fileobj=archive_io, mode="w") as tf:
        for name, data in sorted(members.items()):
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            info.mtime = 0
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            tf.addfile(info, io.BytesIO(data))
    packet["custody_envelope"] = custody_envelope_from_archive_bytes(archive_io.getvalue(), embed_archive_bytes=True)


def packet_mutation_regressions(schedule: dict[str, Any]) -> dict[str, Any]:
    def raw0(packet: dict[str, Any]) -> dict[str, Any]:
        return packet["raw_records"][0]

    def death0(packet: dict[str, Any]) -> dict[str, Any]:
        return packet["source_death_receipts"][0]

    def mutate_archived_raw(packet: dict[str, Any]) -> None:
        rows = json.loads(json.dumps(packet["raw_records"]))
        rows[0]["dirty_probe_response"] = float(rows[0]["dirty_probe_response"]) + 1.0
        replace_archive_member(packet, "raw_records", jsonl_bytes(rows))

    def mutate_archived_death(packet: dict[str, Any]) -> None:
        rows = json.loads(json.dumps(packet["source_death_receipts"]))
        rows[0]["waitpid_pid"] = int(rows[0]["waitpid_pid"]) + 1
        replace_archive_member(packet, "source_death_receipts", jsonl_bytes(rows))

    def mutate_archived_feature(packet: dict[str, Any]) -> None:
        feature = dict(packet["feature_freeze"])
        feature["raw_record_count"] = int(feature["raw_record_count"]) - 1
        replace_archive_member(packet, "feature_freeze", strict_json_bytes(feature))

    def mutate_archived_target(packet: dict[str, Any]) -> None:
        receipt = dict(packet["target_execution_receipt"])
        receipt["raw_record_count"] = int(receipt["raw_record_count"]) - 1
        replace_archive_member(packet, "target_execution_receipt", strict_json_bytes(receipt))

    cases = {
        "altered_block_id": lambda packet: raw0(packet).__setitem__("block_id", "mutated_block"),
        "altered_q": lambda packet: raw0(packet).__setitem__("q", int(raw0(packet)["q"]) + 1),
        "altered_session": lambda packet: raw0(packet).__setitem__("session", "mutated_session"),
        "altered_mapping": lambda packet: raw0(packet).__setitem__("mapping", "mutated_mapping"),
        "altered_delay": lambda packet: raw0(packet).__setitem__("delay_label", "mutated_delay"),
        "altered_source_order": lambda packet: raw0(packet).__setitem__("source_order", "mutated_source_order"),
        "altered_query_order": lambda packet: raw0(packet).__setitem__("query_order", "mutated_query_order"),
        "altered_cyclic_origin": lambda packet: raw0(packet).__setitem__("cyclic_origin", int(raw0(packet)["cyclic_origin"]) + 1),
        "swapped_execution_order": lambda packet: packet["raw_records"].__setitem__(
            slice(0, 2),
            [packet["raw_records"][1], packet["raw_records"][0]],
        ),
        "wrong_cpu": lambda packet: raw0(packet).__setitem__("source_cpu_before", -1),
        "wrong_event_ids": lambda packet: raw0(packet).__setitem__("event_ids", {"bad": 1}),
        "physical_measurement_false": lambda packet: raw0(packet).__setitem__("physical_measurement", False),
        "altered_feature_freeze": lambda packet: packet["feature_freeze"].__setitem__("schedule_sha256", "0" * 64),
        "source_death_pid_mismatch": lambda packet: death0(packet).__setitem__("waitpid_pid", -1),
        "wrong_64_character_target_receipt_hash": lambda packet: packet["custody_envelope"].__setitem__(
            "target_execution_receipt_sha256", wrong_sha256(packet["custody_envelope"]["target_execution_receipt_sha256"])
        ),
        "wrong_64_character_runtime_hash": lambda packet: packet["custody_envelope"].__setitem__(
            "runtime_sha256", wrong_sha256(packet["custody_envelope"]["runtime_sha256"])
        ),
        "wrong_64_character_manifest_hash": lambda packet: packet["custody_envelope"].__setitem__(
            "manifest_sha256", wrong_sha256(packet["custody_envelope"]["manifest_sha256"])
        ),
        "wrong_40_character_source_commit": lambda packet: packet["custody_envelope"].__setitem__(
            "source_authority_sha", wrong_sha1(packet["custody_envelope"]["source_authority_sha"])
        ),
        "wrong_40_character_freeze_commit": lambda packet: packet["custody_envelope"].__setitem__(
            "freeze_sha", wrong_sha1(packet["custody_envelope"]["freeze_sha"])
        ),
        "wrong_64_character_archive_hash": lambda packet: packet["custody_envelope"].__setitem__(
            "copied_back_archive_sha256", wrong_sha256(packet["custody_envelope"]["copied_back_archive_sha256"])
        ),
        "correct_archive_hash_wrong_size": lambda packet: packet["custody_envelope"].__setitem__(
            "copied_back_archive_size", int(packet["custody_envelope"]["copied_back_archive_size"]) + 1
        ),
        "altered_inventory_member_hash": lambda packet: packet["custody_envelope"]["evidence_inventory"]["raw_records"].__setitem__(
            "sha256", wrong_sha256(packet["custody_envelope"]["evidence_inventory"]["raw_records"]["sha256"])
        ),
        "altered_inventory_member_size": lambda packet: packet["custody_envelope"]["evidence_inventory"]["raw_records"].__setitem__(
            "size_bytes", int(packet["custody_envelope"]["evidence_inventory"]["raw_records"]["size_bytes"]) + 1
        ),
        "archive_containing_altered_raw_records": mutate_archived_raw,
        "archive_containing_altered_source_death_receipts": mutate_archived_death,
        "archive_containing_altered_feature_freeze": mutate_archived_feature,
        "archive_containing_altered_target_receipt": mutate_archived_target,
    }
    results = {label: mutated_packet_regression(schedule, label, mutator) for label, mutator in cases.items()}
    return {
        "schema": "FAMILY10H_RELATION_LIFETIME_PACKET_MUTATION_REGRESSIONS_V1",
        "results": results,
        "passed": all(item["passed"] for item in results.values()),
    }


def false_positive_report(report: dict[str, Any]) -> dict[str, Any]:
    adversaries = report.get("scalar_replay_adversary", {}).get("adversaries", {})
    residuals = {
        name: {
            "residual_R_match_abs_of_mean": item.get("residual_R_match_abs_of_mean"),
            "passed": item.get("passed"),
        }
        for name, item in adversaries.items()
        if isinstance(item, dict) and "residual_R_match_abs_of_mean" in item
    }
    return {
        "raw_R_match_abs_of_mean": report.get("R_match_global", {}).get("R_match_abs_of_mean"),
        "raw_R_match_abs_mean": report.get("R_match_global", {}).get("R_match_abs_mean"),
        "residual_models": residuals,
        "heldout_passed": report.get("heldout_transport", {}).get("passed"),
        "failed_gates": [key for key, value in report.get("gates", {}).items() if not value],
        "final_result_class": report.get("result_class"),
        "negative_claim_state": fail_closed_claim_state(report),
    }


def run_self_test(schedule: dict[str, Any]) -> dict[str, Any]:
    threshold = physical_threshold_contract()
    positive = adjudicate_physical_packet(fixture_packet(schedule, "positive"), schedule, threshold)
    false_positive_modes = {
        "nonlinear_q_artifact_coupled_to_execution_position": "nonlinear_q_position",
        "additive_preparation_query_effects_plus_imbalance": "additive_imbalance",
        "route_pressure_artifact": "route_pressure",
        "distance_artifact": "distance_only",
        "source_order_artifact": "source_order_specific",
        "query_order_artifact": "query_order_specific",
        "cyclic_origin_artifact": "origin_specific",
        "mapping_specific_artifact": "mapping_specific",
        "session_specific_artifact": "session_specific",
        "replicate_specific_artifact": "replicate_specific",
        "delay_specific_artifact": "delay_specific",
        "scalar_replay_artifact": "scalar_replay",
        "separable_replay_artifact": "separable_replay",
    }
    false_positive_results = {
        label: adjudicate_physical_packet(fixture_packet(schedule, mode), schedule, threshold, require_custody=False)
        for label, mode in false_positive_modes.items()
    }
    invalid_packet = fixture_packet(schedule, "positive")
    invalid_packet["source_death_receipts"] = invalid_packet["source_death_receipts"][:-1]
    invalid = adjudicate_physical_packet(invalid_packet, schedule, threshold, require_custody=False)
    packet_mutations = packet_mutation_regressions(schedule)
    false_positive_reports = {label: false_positive_report(report) for label, report in false_positive_results.items()}
    negative_claim_states = {
        **{label: fail_closed_claim_state(report) for label, report in false_positive_results.items()},
        "invalid": fail_closed_claim_state(invalid),
    }
    heldout_factor_failures = {
        "cyclic_origin_artifact": false_positive_results["cyclic_origin_artifact"]["heldout_transport"]["factors"]["cyclic_origin"]["passed"] is False,
        "mapping_specific_artifact": false_positive_results["mapping_specific_artifact"]["heldout_transport"]["factors"]["mapping"]["passed"] is False,
        "session_specific_artifact": false_positive_results["session_specific_artifact"]["heldout_transport"]["factors"]["session"]["passed"] is False,
        "replicate_specific_artifact": false_positive_results["replicate_specific_artifact"]["heldout_transport"]["factors"]["replicate"]["passed"] is False,
        "delay_specific_artifact": false_positive_results["delay_specific_artifact"]["heldout_transport"]["factors"]["delay_label"]["passed"] is False,
        "source_order_artifact": false_positive_results["source_order_artifact"]["heldout_transport"]["factors"]["source_order"]["passed"] is False,
        "query_order_artifact": false_positive_results["query_order_artifact"]["heldout_transport"]["factors"]["query_order"]["passed"] is False,
    }
    checks = {
        "positive_fixture_confirmed": positive["result_class"] == RESULT_CONFIRMED and positive["scientific_claim"] == POSITIVE_CLAIM,
        "false_positive_raw_R_match_above_floor": all(
            false_positive_reports[label]["raw_R_match_abs_of_mean"] is not None
            and false_positive_reports[label]["raw_R_match_abs_of_mean"] >= threshold["absolute_thresholds"]["r_match_abs_min"]
            for label in [
                "nonlinear_q_artifact_coupled_to_execution_position",
                "additive_preparation_query_effects_plus_imbalance",
                "route_pressure_artifact",
                "distance_artifact",
                "source_order_artifact",
                "query_order_artifact",
                "cyclic_origin_artifact",
                "mapping_specific_artifact",
                "session_specific_artifact",
                "replicate_specific_artifact",
                "scalar_replay_artifact",
            ]
        ),
        "measured_adversarial_models_reject_false_positive_interactions": all(
            report["final_result_class"] == RESULT_NOT_CONFIRMED and report["negative_claim_state"]["passed"]
            for report in false_positive_reports.values()
        ),
        "stratum_specific_artifact_fails_true_heldout": all(heldout_factor_failures.values()),
        "custody_envelope_cryptographically_bound": packet_mutations["passed"]
        and all(
            packet_mutations["results"][key]["passed"]
            for key in [
                "wrong_64_character_target_receipt_hash",
                "wrong_64_character_runtime_hash",
                "wrong_64_character_manifest_hash",
                "wrong_40_character_source_commit",
                "wrong_40_character_freeze_commit",
                "wrong_64_character_archive_hash",
                "correct_archive_hash_wrong_size",
                "altered_inventory_member_hash",
                "altered_inventory_member_size",
                "archive_containing_altered_raw_records",
                "archive_containing_altered_source_death_receipts",
                "archive_containing_altered_feature_freeze",
                "archive_containing_altered_target_receipt",
            ]
        ),
        "invalid_packet_custody_invalid": invalid["result_class"] == RESULT_INVALID,
        "packet_mutation_regressions_passed": packet_mutations["passed"],
        "negative_and_invalid_fail_closed": all(item["passed"] for item in negative_claim_states.values()),
    }
    result = {
        "schema": "FAMILY10H_RELATION_LIFETIME_PHYSICAL_ADJUDICATOR_SELF_TEST_V1",
        "threshold_contract": threshold,
        "checks": checks,
        "positive_result": {
            "result_class": positive["result_class"],
            "R_match_abs_mean": positive["R_match_global"]["R_match_abs_mean"],
            "heldout_passed": positive["heldout_transport"]["passed"],
            "adversary_passed": positive["scalar_replay_adversary"]["passed"],
            "bootstrap_sign_fraction": positive["bootstrap_stability"].get("bootstrap_sign_fraction"),
            "label_scramble_q95_abs": positive["label_scramble"]["null_distribution"]["q95_abs"],
        },
        "negative_results": {
            **{label: report["result_class"] for label, report in false_positive_results.items()},
            "invalid": invalid["result_class"],
        },
        "false_positive_fixture_results": {
            "block_id_relabeling": packet_mutations["results"]["altered_block_id"],
            "execution_order_drift": packet_mutations["results"]["swapped_execution_order"],
            **false_positive_reports,
            "genuine_relation_interaction": {
                "result_class": positive["result_class"],
                "adversary_passed": positive["scalar_replay_adversary"]["passed"],
                "raw_R_match_abs_of_mean": positive["R_match_global"]["R_match_abs_of_mean"],
            },
        },
        "heldout_factor_failure_results": heldout_factor_failures,
        "packet_mutation_regressions": packet_mutations,
        "negative_claim_states": negative_claim_states,
        "passed": all(checks.values()),
    }
    result["self_test_sha256"] = pub.digest({k: v for k, v in result.items() if k != "self_test_sha256"})
    return result
