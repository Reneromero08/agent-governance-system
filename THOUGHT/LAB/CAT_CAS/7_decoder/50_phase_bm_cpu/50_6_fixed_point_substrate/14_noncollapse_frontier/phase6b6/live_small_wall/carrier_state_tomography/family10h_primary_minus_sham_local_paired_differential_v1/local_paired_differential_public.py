from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


PACKAGE_ID = "family10h_primary_minus_sham_local_paired_differential_v1"
TRANSACTION_RUN_ID = "family10h_primary_minus_sham_local_paired_differential_v1_0"

SEGMENTED_PACKAGE_ID = "family10h_relation_spatial_pair_readout_v1_1_segmented"
SEGMENTED_SOURCE_AUTHORITY_COMMIT = "2cdec9704452669d3651a063b6fb5805913647e3"
SEGMENTED_FREEZE_COMMIT = "8720dd543947f26cbd2af18a6a3dd4870c9adf85"
LOCAL_PAIRED_AUDIT_FREEZE_COMMIT = "526a509bf6034fc7e20159541d270cb52ef04db3"
LOCAL_PAIRED_AUDIT_SHA256 = "2826e10317dd96d7f73c8471e0e6365b466a003fa59dde8c88ec200a0d1b3cf9"
PRIOR_PRIMARY_MINUS_SHAM_PACKAGE_COMMIT = "054cc36f6c4d67de69cdd1a1dc12f455edc580dd"
PRIOR_OFFICIAL_EVIDENCE_COMMIT = "dde3ac15953669f056b5f8974601b3bd1aa62f2d"

ROUND_COUNT = 2
PAIR_SAMPLE_COUNT = 256
GENERIC_CONTROL_ENVELOPE_FRACTION = 0.25

PRIMARY_QUERY = "query_relation_pair"
SHAM_QUERY = "relation_sham"
GENERIC_CONTROL_QUERIES = [
    "scrambled_pair_control",
    "distance_matched_control",
    "route_pressure_control",
]
ALL_QUERIES = [PRIMARY_QUERY, SHAM_QUERY, *GENERIC_CONTROL_QUERIES]

FACTOR_STRATA = [
    "session",
    "replicate",
    "mapping",
    "source_order",
    "query_order",
    "cyclic_origin",
]

SCHEDULE_DIRNAME = "LOCAL_PAIRED_DIFFERENTIAL_SCHEDULES"
SCHEDULE_MANIFEST_FILENAME = "LOCAL_PAIRED_DIFFERENTIAL_SCHEDULE_MANIFEST.json"
PACKAGE_MANIFEST_FILENAME = "LOCAL_PAIRED_DIFFERENTIAL_PACKAGE_MANIFEST.json"
THRESHOLD_CONTRACT_FILENAME = "LOCAL_PAIRED_DIFFERENTIAL_THRESHOLD_CONTRACT.json"
PREPARE_RESULT_FILENAME = "LOCAL_PAIRED_DIFFERENTIAL_PREPARE_RESULT.json"

RESULT_CONFIRMED = "FAMILY10H_LOCAL_PAIRED_DIFFERENTIAL_CONFIRMED_PROSPECTIVE"
RESULT_NOT_CONFIRMED = "FAMILY10H_LOCAL_PAIRED_DIFFERENTIAL_NOT_CONFIRMED_PROSPECTIVE"
RESULT_CUSTODY_INVALID = "FAMILY10H_LOCAL_PAIRED_DIFFERENTIAL_CUSTODY_INVALID"

POSITIVE_CLAIM = "PUBLIC_POST_SOURCE_LOCAL_PRIMARY_MINUS_SHAM_DIFFERENTIAL_CONFIRMED"
NEGATIVE_CLAIM = "PUBLIC_POST_SOURCE_LOCAL_PRIMARY_MINUS_SHAM_DIFFERENTIAL_NOT_ESTABLISHED"

PACKAGE_DECISION_READY = "FAMILY10H_LOCAL_PAIRED_DIFFERENTIAL_PACKAGE_READY_AWAITING_AUTHORIZATION"


def package_root() -> Path:
    return Path(__file__).resolve().parent


def carrier_state_root() -> Path:
    return package_root().parent


def segmented_package_root() -> Path:
    return carrier_state_root() / SEGMENTED_PACKAGE_ID


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return sha256_bytes(encoded)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def read_segmented_schedule_rows() -> tuple[list[str], list[dict[str, str]]]:
    schedule = segmented_package_root() / "RELATION_SPATIAL_PUBLIC_SCHEDULE.tsv"
    with schedule.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = reader.fieldnames
        if not fields:
            raise ValueError("segmented schedule has no header")
        return fields, list(reader)


def write_variant_schedule(path: Path, fields: list[str], rows: list[dict[str, str]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for index, row in enumerate(rows):
            copied = dict(row)
            copied["execution_ordinal"] = str(index)
            writer.writerow(copied)
    return {
        "path": str(path.relative_to(package_root()).as_posix()),
        "row_count": len(rows),
        "pair_observation_count": len(rows) * PAIR_SAMPLE_COUNT,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def freeze_schedules() -> dict[str, Any]:
    fields, rows = read_segmented_schedule_rows()
    by_query = {query: [row for row in rows if row.get("query") == query] for query in ALL_QUERIES}
    for query, query_rows in by_query.items():
        if len(query_rows) != 2048:
            raise ValueError(f"unexpected row count for {query}: {len(query_rows)}")

    schedule_dir = package_root() / SCHEDULE_DIRNAME
    variants: dict[str, Any] = {}
    for round_index in range(ROUND_COUNT):
        for query in ALL_QUERIES:
            name = f"round{round_index}_{query}"
            variants[name] = {
                "round": round_index,
                "query": query,
                **write_variant_schedule(schedule_dir / f"{name}.tsv", fields, by_query[query]),
            }

    manifest = {
        "schema": "FAMILY10H_LOCAL_PAIRED_DIFFERENTIAL_SCHEDULE_MANIFEST_V1",
        "package_id": PACKAGE_ID,
        "transaction_run_id": TRANSACTION_RUN_ID,
        "source_schedule_package": SEGMENTED_PACKAGE_ID,
        "source_authority_commit": SEGMENTED_SOURCE_AUTHORITY_COMMIT,
        "source_freeze_commit": SEGMENTED_FREEZE_COMMIT,
        "local_paired_audit_freeze_commit": LOCAL_PAIRED_AUDIT_FREEZE_COMMIT,
        "local_paired_audit_sha256": LOCAL_PAIRED_AUDIT_SHA256,
        "prior_primary_minus_sham_package_commit": PRIOR_PRIMARY_MINUS_SHAM_PACKAGE_COMMIT,
        "prior_official_evidence_commit": PRIOR_OFFICIAL_EVIDENCE_COMMIT,
        "round_count": ROUND_COUNT,
        "queries": ALL_QUERIES,
        "primary_query": PRIMARY_QUERY,
        "sham_query": SHAM_QUERY,
        "generic_control_queries": GENERIC_CONTROL_QUERIES,
        "factor_strata": FACTOR_STRATA,
        "fresh_process_per_variant": True,
        "interleaved_raw_control_rows": False,
        "pair_sample_count_per_row": PAIR_SAMPLE_COUNT,
        "variants": variants,
    }
    manifest["schedule_manifest_sha256"] = digest(
        {key: value for key, value in manifest.items() if key != "schedule_manifest_sha256"}
    )
    write_json(package_root() / SCHEDULE_MANIFEST_FILENAME, manifest)
    return manifest


def threshold_contract() -> dict[str, Any]:
    contract = {
        "schema": "FAMILY10H_LOCAL_PAIRED_DIFFERENTIAL_THRESHOLD_CONTRACT_V1",
        "package_id": PACKAGE_ID,
        "coordinate": {
            "R_primary": f"mean R_spatial({PRIMARY_QUERY})",
            "R_sham": f"mean R_spatial({SHAM_QUERY})",
            "D_local": "R_primary - R_sham",
        },
        "round_law": {
            "R_primary_gt_0": True,
            "R_sham_lt_0": "not_a_gate",
            "D_local_gt_0": True,
            "generic_control_envelope": "max(abs(R_scrambled), abs(R_distance), abs(R_route)) <= 0.25 * D_local",
            "generic_control_envelope_fraction": GENERIC_CONTROL_ENVELOPE_FRACTION,
        },
        "factor_stratum_law": {
            "factors": FACTOR_STRATA,
            "stratum_scope": "one_factor_matched_strata",
            "R_primary_gt_0": True,
            "R_sham_lt_0": "not_a_gate",
            "D_local_gt_0": True,
        },
        "diagnostics_not_gates": {
            "generic_matched_permutation_q99": "reported diagnostic",
            "six_factor_crossed_cells": "reported diagnostic_only_sparse_four_blocks_per_cell",
            "relation_sham_in_null_control_set": False,
            "absolute_relation_sham_sign": "reported diagnostic_not_gate",
        },
        "forbidden_post_freeze_changes": {
            "restore_sham_sign_gate": True,
            "promote_sparse_crossed_cells_to_gates": True,
            "post_observation_threshold_revision": True,
            "post_observation_feature_selection": True,
            "small_wall_promotion": True,
        },
        "claim_boundary": {
            "full_carrier_state_tomography_established": False,
            "physical_relational_memory_established": False,
            "catalytic_borrowing_established": False,
            "r2_restoration_established": False,
            "small_wall_crossed": False,
        },
    }
    contract["threshold_contract_sha256"] = digest(
        {key: value for key, value in contract.items() if key != "threshold_contract_sha256"}
    )
    return contract


def freeze_package_manifest(schedule_manifest: dict[str, Any]) -> dict[str, Any]:
    segmented = segmented_package_root()
    source_hashes = {
        "relation_spatial_runtime": sha256_file(segmented / "relation_spatial_runtime"),
        "relation_spatial_runtime.c": sha256_file(segmented / "relation_spatial_runtime.c"),
        "relation_spatial_runtime.h": sha256_file(segmented / "relation_spatial_runtime.h"),
        "relation_spatial_pmu_preflight": sha256_file(segmented / "relation_spatial_pmu_preflight"),
        "relation_spatial_pmu_preflight.c": sha256_file(segmented / "relation_spatial_pmu_preflight.c"),
        "RELATION_SPATIAL_IMPLEMENTATION_MANIFEST.json": sha256_file(
            segmented / "RELATION_SPATIAL_IMPLEMENTATION_MANIFEST.json"
        ),
        "RELATION_SPATIAL_PUBLIC_SCHEDULE.tsv": sha256_file(segmented / "RELATION_SPATIAL_PUBLIC_SCHEDULE.tsv"),
        "RELATION_GRAMMAR.json": sha256_file(segmented / "RELATION_GRAMMAR.json"),
    }
    contract = threshold_contract()
    write_json(package_root() / THRESHOLD_CONTRACT_FILENAME, contract)
    manifest = {
        "schema": "FAMILY10H_LOCAL_PAIRED_DIFFERENTIAL_PACKAGE_MANIFEST_V1",
        "package_id": PACKAGE_ID,
        "transaction_run_id": TRANSACTION_RUN_ID,
        "package_decision": PACKAGE_DECISION_READY,
        "source_schedule_package": SEGMENTED_PACKAGE_ID,
        "source_authority_commit": SEGMENTED_SOURCE_AUTHORITY_COMMIT,
        "source_freeze_commit": SEGMENTED_FREEZE_COMMIT,
        "local_paired_audit_freeze_commit": LOCAL_PAIRED_AUDIT_FREEZE_COMMIT,
        "local_paired_audit_sha256": LOCAL_PAIRED_AUDIT_SHA256,
        "prior_primary_minus_sham_package_commit": PRIOR_PRIMARY_MINUS_SHAM_PACKAGE_COMMIT,
        "prior_official_evidence_commit": PRIOR_OFFICIAL_EVIDENCE_COMMIT,
        "schedule_manifest_sha256": schedule_manifest["schedule_manifest_sha256"],
        "threshold_contract_sha256": contract["threshold_contract_sha256"],
        "runtime_source_hashes": source_hashes,
        "live_attempt_authority": False,
        "post_observation_feature_selection_allowed": False,
        "post_observation_threshold_revision_allowed": False,
        "restore_sham_sign_gate_allowed": False,
        "sparse_crossed_cells_as_gates_allowed": False,
        "small_wall_crossed": False,
    }
    manifest["package_manifest_sha256"] = digest(
        {key: value for key, value in manifest.items() if key != "package_manifest_sha256"}
    )
    write_json(package_root() / PACKAGE_MANIFEST_FILENAME, manifest)
    return manifest


def prepare_package() -> dict[str, Any]:
    schedule_manifest = freeze_schedules()
    package_manifest = freeze_package_manifest(schedule_manifest)
    result = {
        "schema": "FAMILY10H_LOCAL_PAIRED_DIFFERENTIAL_PREPARE_RESULT_V1",
        "passed": True,
        "package_decision": PACKAGE_DECISION_READY,
        "schedule_manifest_sha256": schedule_manifest["schedule_manifest_sha256"],
        "package_manifest_sha256": package_manifest["package_manifest_sha256"],
        "threshold_contract_sha256": package_manifest["threshold_contract_sha256"],
        "live_attempt_authority": False,
        "small_wall_crossed": False,
    }
    write_json(package_root() / PREPARE_RESULT_FILENAME, result)
    return result


if __name__ == "__main__":
    prepare_package()
