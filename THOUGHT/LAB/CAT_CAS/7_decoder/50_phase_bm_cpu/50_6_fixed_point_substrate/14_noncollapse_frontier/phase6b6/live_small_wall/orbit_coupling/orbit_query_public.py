#!/usr/bin/env python3
"""Public OrbitState query receiver/feature-freeze logic.

This module intentionally contains no private condition table, no member values, and
no source-work allocation law.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any


N = 256
PHASES = (0.0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0)
PHASE_NAMES = ("0", "pi/2", "pi", "3pi/2")
FORBIDDEN_RECEIVER_TERMS = (
    "condition",
    "member",
    "target",
    "plus",
    "minus",
    "lower",
    "orientation",
    "branch",
    "quantized",
    "positive_work",
    "negative_work",
    "dummy_work",
    "public_label",
    "label_swap",
    "query_phase",
    "source_phase",
    "schedule",
)


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def reconstruct_complex(responses: list[float], phases: tuple[float, ...] = PHASES) -> complex:
    if len(responses) != len(phases):
        raise ValueError("response and phase lengths differ")
    total = 0.0 + 0.0j
    for response, phase in zip(responses, phases):
        total += response * complex(math.cos(phase), math.sin(phase))
    return (2.0 / float(len(phases))) * total


def assert_receiver_manifest_blind(manifest: dict[str, Any]) -> None:
    top_allowed = {
        "schema_id",
        "modulus",
        "phase_names",
        "decoder",
        "fresh_process_replicates",
        "groups",
    }
    group_allowed = {
        "group_id",
        "replicate_index",
        "records",
        "operator",
        "base_work_units",
        "total_source_work_units",
    }
    record_allowed = {
        "opaque_run_id",
        "phase_ordinal",
        "decoder_phase_index",
        "decoder_phase_radians",
    }
    extra_top = set(manifest) - top_allowed
    if extra_top:
        raise AssertionError(f"receiver manifest has unexpected fields: {sorted(extra_top)}")
    text = json.dumps(manifest, sort_keys=True).lower()
    leaked = [term for term in FORBIDDEN_RECEIVER_TERMS if term in text]
    if leaked:
        raise AssertionError(f"receiver manifest leaks private labels: {leaked}")
    for group in manifest["groups"]:
        extra_group = set(group) - group_allowed
        if extra_group:
            raise AssertionError(f"group {group.get('group_id')} has unexpected fields: {sorted(extra_group)}")
        records = group.get("records")
        if not isinstance(records, list):
            raise AssertionError(f"group {group.get('group_id')} records are not a list")
        for record in records:
            extra_record = set(record) - record_allowed
            if extra_record:
                raise AssertionError(
                    f"record {record.get('opaque_run_id')} has unexpected fields: {sorted(extra_record)}"
                )


def extract_features(manifest: dict[str, Any], raw_records: list[dict[str, Any]]) -> dict[str, Any]:
    assert_receiver_manifest_blind(manifest)
    allowed_raw_keys = {
        "schema_id",
        "opaque_run_id",
        "group_id",
        "replicate_index",
        "phase_ordinal",
        "decoder_phase_index",
        "measurement_order",
        "positive_cycles",
        "negative_cycles",
        "positive_change_to_dirty",
        "negative_change_to_dirty",
        "positive_probe_dirty",
        "negative_probe_dirty",
        "positive_duration_ns",
        "negative_duration_ns",
        "change_to_dirty_delta",
        "probe_dirty_delta",
        "duration_delta_ns",
        "restoration_passed",
        "perf_available",
        "initial_positive_digest",
        "initial_negative_digest",
        "final_positive_digest",
        "final_negative_digest",
    }
    by_group: dict[str, list[dict[str, Any]]] = {}
    for record in raw_records:
        extra = set(record) - allowed_raw_keys
        if extra:
            raise AssertionError(f"raw record contains unexpected fields: {sorted(extra)}")
        text = json.dumps(record, sort_keys=True).lower()
        leaked = [term for term in FORBIDDEN_RECEIVER_TERMS if term in text]
        if leaked:
            raise AssertionError(f"raw record leaks private terms: {leaked}")
        by_group.setdefault(str(record["group_id"]), []).append(record)
    features: list[dict[str, Any]] = []
    for group in manifest["groups"]:
        manifest_by_id = {record["opaque_run_id"]: record for record in group["records"]}
        manifest_ids = set(manifest_by_id)
        records = sorted(by_group[group["group_id"]], key=lambda item: int(item["decoder_phase_index"]))
        if len(records) != 4:
            raise AssertionError(f"group {group['group_id']} does not have 4 records")
        if {int(record["phase_ordinal"]) for record in records} != {0, 1, 2, 3}:
            raise AssertionError(f"group {group['group_id']} has invalid phase ordinals")
        if {int(record["decoder_phase_index"]) for record in records} != {0, 1, 2, 3}:
            raise AssertionError(f"group {group['group_id']} has invalid decoder phases")
        if {record["opaque_run_id"] for record in records} != manifest_ids:
            raise AssertionError(f"group {group['group_id']} raw IDs do not match manifest")
        for record in records:
            manifest_record = manifest_by_id[record["opaque_run_id"]]
            for key in ("phase_ordinal", "decoder_phase_index"):
                if int(record[key]) != int(manifest_record[key]):
                    raise AssertionError(
                        f"group {group['group_id']} raw {key} does not match manifest"
                    )
        decoded: dict[str, Any] = {}
        for key in ("change_to_dirty_delta", "probe_dirty_delta", "duration_delta_ns"):
            values = [float(record[key]) for record in records]
            z = reconstruct_complex(values)
            decoded[key] = {"real": z.real, "imag": z.imag, "abs": abs(z), "responses": values}
        features.append(
            {
                "group_id": group["group_id"],
                "replicate_index": group["replicate_index"],
                "decoded": decoded,
                "restoration_passed": all(bool(record["restoration_passed"]) for record in records),
            }
        )
    result = {
        "schema_id": "CAT_CAS_ORBIT_QUERY_FEATURES_FROZEN_V1",
        "primary_response": "change_to_dirty_delta",
        "features": features,
    }
    result["features_sha256"] = digest({k: v for k, v in result.items() if k != "features_sha256"})
    return result
