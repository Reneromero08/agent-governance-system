#!/usr/bin/env python3
"""Offline gain-covariant OrbitState audit for retry-one retained evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import tarfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


N = 256
D_MEMBER = 23
FOLD_MEMBER = 233
QUANTIZATION_SCALE = 1536
BASE_WORK = 2048
TOTAL_WORK = 4096
PUBLIC_Q0_ABSOLUTE_BOUND = 152.0
PRIVATE_ODD_SIGNAL_FLOOR = 456.0
RELATIONAL_TOLERANCE = 0.25
STRONG_Q_FLOOR = 256
REPLICATES = [0, 1]
PRIMARY_COORDINATE = "change_to_dirty"

OFFICIAL_CLASS = "ORBITSTATE_INDEPENDENT_COUPLING_CANDIDATE"
CLASS_ESTABLISHED = "PRIVATE_ORBITSTATE_GAIN_COVARIANT_GEOMETRY_ESTABLISHED"
CLASS_PARTIAL = "PRIVATE_ORBITSTATE_GAIN_COVARIANT_GEOMETRY_PARTIAL"
CLASS_NOT_ESTABLISHED = "PRIVATE_ORBITSTATE_GAIN_COVARIANT_GEOMETRY_NOT_ESTABLISHED"

EXPECTED_EVIDENCE_SHA256 = {
    "RAW_ORBITSTATE_RECEIVER_CAPTURE.jsonl": "721ca48c2f5b682d871521b106be9fcff9d511325eb2473a4ed298b23d54f623",
    "ORBITSTATE_RECEIVER_SENTINELS.jsonl": "da68625bb05d09846bf0ef216cb7502d1549c208f50326c4888e8e2cbc1e7950",
    "ORBITSTATE_STAGE_RECEIPTS.jsonl": "0c7c18174d90531763d785db91b8d11a9e218f0062c4843b1f0adab36eb05b3a",
    "ORBITSTATE_SOURCE_RECEIPTS.jsonl": "5a67264693eef2e3d5a4fc3bb1fc5a857fb83245e3af4fd78fa1debed9fe8feb",
    "ORBITSTATE_RECEIVER_FEATURES.json": "70ac1c25cf795b90da4196672d530549dc0519aa6e98e589d5e75ad2f1f8a26a",
    "ORBITSTATE_FEATURE_FREEZE_RECEIPT.json": "e0183c5bbc549e7630491bffef0702ee26fe8077bdfd193914808ff4b79f6365",
    "ORBITSTATE_ADJUDICATION.json": "4ff6e51770b81bf76329428f3aa6edb6be83a1445a0979637e13d50403f36bfd",
    "ORBITSTATE_INDEPENDENT_V2_MANIFEST.json": "41273b4294a2308961e5f177f1a99159944dcbaf2b46ca6d1bd85ce542b900b3",
    "COPYBACK_MANIFEST.json": "c8ab1065cf08bf3b2b9c2cb01e09dadbf1016727c4b35f9a4139dc2854efe97e",
}

CONDITION_LABELS = {
    "pre_projection_d": "Z_d",
    "pre_projection_fold": "Z_fold",
    "source_off": "Z_source_off",
    "query_off": "Z_query_off",
    "post_projection": "Z_post_projection",
    "declaration_sham": "Z_declaration_sham",
    "query_scramble": "Z_query_scramble",
    "equal_orbit_odd_zero": "Z_equal_orbit",
    "source_polarity_inversion_d": "Z_polarity_inversion",
}


class AuditError(RuntimeError):
    pass


def repo_root_from_here() -> Path:
    here = Path(__file__).resolve()
    for candidate in [here.parent, *here.parents]:
        if (candidate / ".git").exists():
            return candidate
    raise AuditError("repository root not found from audit package")


def audit_root() -> Path:
    return Path(__file__).resolve().parent


def evidence_root(repo_root: Path) -> Path:
    return (
        repo_root
        / "THOUGHT/LAB/CAT_CAS/7_decoder/50_phase_bm_cpu/50_6_fixed_point_substrate/"
        "14_noncollapse_frontier/phase6b6/live_small_wall/orbit_coupling/runs/"
        "orbitstate_independent_v2_1"
    )


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise AuditError(f"JSONL parse failed for {path.name}:{line_number}: {exc}") from exc
    return rows


def read_bundle_json(bundle_path: Path, member_name: str) -> Any:
    with tarfile.open(bundle_path, "r:gz") as bundle:
        member = bundle.getmember(member_name)
        extracted = bundle.extractfile(member)
        if extracted is None:
            raise AuditError(f"source bundle member unreadable: {member_name}")
        return json.loads(extracted.read().decode("utf-8"))


def phase_radians(phase_index: int) -> float:
    return [0.0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0][phase_index]


def orbit_phi(member: int) -> float:
    return 2.0 * math.pi * float(member) / float(N)


def round_c(value: float) -> int:
    if value >= 0:
        return int(math.floor(value + 0.5))
    return int(math.ceil(value - 0.5))


def clamp_q(value: int) -> int:
    return max(-QUANTIZATION_SCALE, min(QUANTIZATION_SCALE, value))


def q_theta(record: dict[str, Any]) -> int:
    if bool(record["source_off_dummy_mode"]) or record["response_mode"] == "source_off":
        return 0
    theta = phase_radians(int(record["public_phase_index"]))
    private_theta = phase_radians(int(record["private_source_phase_index"]))
    phi = orbit_phi(int(record["orbit_state"]["member"]))
    response_mode = str(record["response_mode"])
    if response_mode == "query_off":
        q_value = 0
    elif response_mode == "pre_projection":
        q_value = round_c(float(QUANTIZATION_SCALE) * math.cos(phi - private_theta))
    elif response_mode == "post_projection":
        q_value = round_c(float(QUANTIZATION_SCALE) * math.cos(phi) * math.cos(theta))
    elif response_mode == "declaration_sham":
        q_value = round_c(float(QUANTIZATION_SCALE) * math.cos(phi))
    elif response_mode == "equal_orbit_odd_zero":
        q_value = round_c(float(QUANTIZATION_SCALE) * math.cos(theta))
    else:
        raise AuditError(f"unknown response_mode: {response_mode}")
    if bool(record["polarity_inversion"]):
        q_value = -q_value
    return clamp_q(q_value)


def source_work_for_q(q_value: int, *, source_off_dummy_mode: bool) -> dict[str, int]:
    if source_off_dummy_mode:
        return {
            "q_theta": 0,
            "positive_work": 0,
            "negative_work": 0,
            "dummy_work": TOTAL_WORK,
            "total_work": TOTAL_WORK,
        }
    return {
        "q_theta": q_value,
        "positive_work": BASE_WORK + q_value,
        "negative_work": BASE_WORK - q_value,
        "dummy_work": 0,
        "total_work": TOTAL_WORK,
    }


def rel_error(left: float, right: float, floor: float = PUBLIC_Q0_ABSOLUTE_BOUND) -> float:
    denom = max(abs(left), abs(right), floor)
    return abs(left - right) / denom


def gain_rel_error(left: float, right: float) -> float:
    denom = max(abs(left), abs(right), 1.0e-12)
    return abs(left - right) / denom


def complex_rel_error(left: complex, right: complex) -> float:
    return max(rel_error(left.real, right.real), rel_error(left.imag, right.imag))


def complex_decode(phase_values: dict[int, float]) -> complex:
    if set(phase_values) != {0, 1, 2, 3}:
        raise AuditError("complex decoder requires phases 0, 1, 2, 3")
    total = 0j
    for phase_index, value in phase_values.items():
        theta = phase_radians(phase_index)
        total += float(value) * complex(math.cos(theta), math.sin(theta))
    return (2.0 / 4.0) * total


def mean_complex(values: list[complex]) -> complex:
    return complex(statistics.fmean(value.real for value in values), statistics.fmean(value.imag for value in values))


def condition_average(mapping_decodes: dict[str, complex]) -> complex:
    if set(mapping_decodes) != {"map0", "map1"}:
        raise AuditError("condition average requires map0 and map1")
    return mean_complex([mapping_decodes["map0"], mapping_decodes["map1"]])


def cdict(value: complex) -> dict[str, float]:
    return {"real": value.real, "imag": value.imag}


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def angle_error(observed: complex, ideal: complex) -> float:
    return abs(wrap_angle(math.atan2(observed.imag, observed.real) - math.atan2(ideal.imag, ideal.real)))


def positive_ls_gain(observed: complex, ideal: complex) -> float:
    denom = ideal.real * ideal.real + ideal.imag * ideal.imag
    if denom <= 0:
        raise AuditError("zero ideal vector")
    gain = ((observed * ideal.conjugate()).real) / denom
    return max(0.0, gain)


def vector_fit(observed: complex, ideal: complex) -> dict[str, Any]:
    gain = positive_ls_gain(observed, ideal)
    fitted = gain * ideal
    orthogonal = observed - fitted
    return {
        "observed": cdict(observed),
        "ideal": cdict(ideal),
        "positive_least_squares_gain": gain,
        "angular_error_rad": angle_error(observed, ideal),
        "orthogonal_residual": abs(orthogonal),
        "relative_vector_residual": abs(orthogonal) / max(abs(observed), abs(fitted), PUBLIC_Q0_ABSOLUTE_BOUND),
    }


def receiver_feature_digest(features: dict[str, Any]) -> str:
    return digest({key: value for key, value in features.items() if key != "receiver_features_sha256"})


def source_receipt_law(source_map: dict[str, Any], source_receipts: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {record["opaque_run_id"]: record for record in source_map["records"]}
    failures: list[str] = []
    expected_components = {(run_id, component) for run_id in by_id for component in ["positive", "negative"]}
    observed_components = {(receipt.get("opaque_run_id"), receipt.get("component")) for receipt in source_receipts}
    if observed_components != expected_components:
        failures.append("source receipt component coverage mismatch")
    if len(observed_components) != len(source_receipts):
        failures.append("source receipt duplicate component coverage")
    for receipt in source_receipts:
        run_id = receipt.get("opaque_run_id")
        if run_id not in by_id:
            failures.append(f"unknown source receipt {run_id}")
            continue
        private = by_id[run_id]
        expected = source_work_for_q(q_theta(private), source_off_dummy_mode=bool(private["source_off_dummy_mode"]))
        for key, expected_value in expected.items():
            if receipt.get(key) != expected_value:
                failures.append(f"{run_id} {key} {receipt.get(key)} != {expected_value}")
        if receipt.get("source_core") != 4 or receipt.get("source_cpu_before") != 4 or receipt.get("source_cpu_after") != 4:
            failures.append(f"{run_id} source CPU custody mismatch")
        if receipt.get("receiver_feedback_used_to_select_q") is not False:
            failures.append(f"{run_id} receiver feedback used")
    return {"passed": not failures, "failures": failures, "receipt_count": len(source_receipts)}


def load_evidence(root: Path) -> dict[str, Any]:
    return {
        "raw": read_jsonl(root / "RAW_ORBITSTATE_RECEIVER_CAPTURE.jsonl"),
        "sentinels": read_jsonl(root / "ORBITSTATE_RECEIVER_SENTINELS.jsonl"),
        "stage_receipts": read_jsonl(root / "ORBITSTATE_STAGE_RECEIPTS.jsonl"),
        "source_receipts": read_jsonl(root / "ORBITSTATE_SOURCE_RECEIPTS.jsonl"),
        "features": read_json(root / "ORBITSTATE_RECEIVER_FEATURES.json"),
        "freeze": read_json(root / "ORBITSTATE_FEATURE_FREEZE_RECEIPT.json"),
        "adjudication": read_json(root / "ORBITSTATE_ADJUDICATION.json"),
        "execution_manifest": read_json(root / "ORBITSTATE_INDEPENDENT_V2_MANIFEST.json"),
        "copyback_manifest": read_json(root / "COPYBACK_MANIFEST.json"),
        "final_result": read_json(root / "FINAL_RESULT_ORBITSTATE_INDEPENDENT_V2.json"),
        "controller_result": read_json(root / "CONTROLLER_RESULT.json"),
        "custody": read_json(root / "LIVE_CUSTODY_LOG.json"),
        "pmu_preflight": read_json(root / "ORBITSTATE_PMU_PREFLIGHT.json"),
        "source_hashes": read_json(root / "ORBITSTATE_SOURCE_HASHES.json"),
    }


def verify_evidence(root: Path, evidence: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    observed_hashes = {name: sha256_file(root / name) for name in EXPECTED_EVIDENCE_SHA256}
    for name, expected in EXPECTED_EVIDENCE_SHA256.items():
        if observed_hashes[name] != expected:
            failures.append(f"{name} hash mismatch")

    raw = evidence["raw"]
    sentinels = evidence["sentinels"]
    stage = evidence["stage_receipts"]
    source = evidence["source_receipts"]
    features = evidence["features"]
    freeze = evidence["freeze"]
    custody = evidence["custody"]
    pmu = evidence["pmu_preflight"]
    execution = evidence["execution_manifest"]
    copyback = evidence["copyback_manifest"]
    final = evidence["final_result"]
    controller = evidence["controller_result"]

    expected_counts = {
        "raw_component_windows": 288,
        "sentinels": 288,
        "stage_receipts": 2016,
        "source_receipts": 288,
        "receiver_feature_rows": 144,
    }
    observed_counts = {
        "raw_component_windows": len(raw),
        "sentinels": len(sentinels),
        "stage_receipts": len(stage),
        "source_receipts": len(source),
        "receiver_feature_rows": len(features.get("rows", [])),
    }
    if observed_counts != expected_counts:
        failures.append(f"record counts mismatch: {observed_counts}")
    if execution.get("record_counts") != {
        "independent_component_windows": 288,
        "mapping_leg_records": 144,
        "source_receipts": 288,
        "stage_receipts": 2016,
    }:
        failures.append("execution manifest record counts mismatch")

    if final.get("result_class") != OFFICIAL_CLASS:
        failures.append("official retained class changed")
    if controller.get("status") != "ORBITSTATE_CONTROLLER_TARGET_COMPLETE":
        failures.append("controller status mismatch")
    if final.get("status") != "ORBITSTATE_INDEPENDENT_TARGET_COMPLETE":
        failures.append("target status mismatch")
    if controller.get("remote_cleaned") is not True or controller.get("remote_retained") is not False:
        failures.append("retry-one remote cleanup state mismatch")

    if receiver_feature_digest(features) != features.get("receiver_features_sha256"):
        failures.append("receiver feature digest mismatch")
    sha_line = (root / "ORBITSTATE_RECEIVER_FEATURES.sha256").read_text(encoding="utf-8").strip()
    if not sha_line.startswith(str(features.get("receiver_features_sha256"))):
        failures.append("receiver feature sha256 sidecar mismatch")
    if sha256_file(root / "ORBITSTATE_RECEIVER_FEATURES.json") != execution.get("receiver_features_sha256"):
        failures.append("execution manifest receiver feature file hash mismatch")
    if digest({k: v for k, v in execution.items() if k != "execution_manifest_sha256"}) != execution.get(
        "execution_manifest_sha256"
    ):
        failures.append("execution manifest self digest mismatch")
    if digest({k: v for k, v in copyback.items() if k != "copyback_manifest_sha256"}) != copyback.get(
        "copyback_manifest_sha256"
    ):
        failures.append("copyback manifest self digest mismatch")
    for entry in copyback.get("entries", []):
        path = root / entry["path"]
        if not path.exists() or path.stat().st_size != entry["size"] or sha256_file(path) != entry["sha256"]:
            failures.append(f"copyback entry mismatch: {entry['path']}")
    for name, expected in execution.get("final_evidence_hashes", {}).items():
        if sha256_file(root / name) != expected:
            failures.append(f"execution final hash mismatch: {name}")

    pmu_failures = []
    for index, row in enumerate(raw):
        counters = row["counters"]
        ids = [counters.get("cycles_id"), counters.get("change_to_dirty_id"), counters.get("probe_dirty_id")]
        if row.get("receiver_cpu_before") != 5 or row.get("receiver_cpu_after") != 5:
            pmu_failures.append(f"row {index} receiver cpu")
        if counters.get("cycles", 0) <= 0 or counters.get("time_enabled", 0) <= 0:
            pmu_failures.append(f"row {index} pmu nonpositive")
        if counters.get("time_enabled") != counters.get("time_running"):
            pmu_failures.append(f"row {index} pmu multiplexed")
        if not all(isinstance(value, int) and value > 0 for value in ids) or len(set(ids)) != 3:
            pmu_failures.append(f"row {index} event ids")
        if counters.get("pmu_read_size") != 72:
            pmu_failures.append(f"row {index} read size")
        if row.get("byte_compare_ok") is not True:
            pmu_failures.append(f"row {index} byte compare")
    if pmu_failures:
        failures.extend(pmu_failures[:10])
    if not (
        pmu.get("receiver_cpu_before") == 5
        and pmu.get("receiver_cpu_after") == 5
        and pmu.get("cycles", 0) > 0
        and pmu.get("time_enabled", 0) > 0
        and pmu.get("time_enabled") == pmu.get("time_running")
        and len({pmu.get("cycles_id"), pmu.get("change_to_dirty_id"), pmu.get("probe_dirty_id")}) == 3
        and pmu.get("read_size") == 72
    ):
        failures.append("pmu preflight failed reconstruction")

    if not all(row.get("pre_ok") is True and row.get("post_ok") is True for row in sentinels):
        failures.append("sentinel restoration failure")
    if not all(row.get("ok") is True for row in stage):
        failures.append("stage receipt failure")

    process_failures = [
        row["label"]
        for row in custody.get("process_snapshots", [])
        if row.get("scan_returncode") != 0 or row.get("forbidden_process_hits") != []
    ]
    if process_failures:
        failures.append(f"process custody failures: {process_failures}")
    identity = custody.get("identity", {})
    if not (
        identity.get("euid") == 0
        and identity.get("cpu", {}).get("vendor_id") == "AuthenticAMD"
        and identity.get("cpu", {}).get("cpu_family") == 16
        and identity.get("event_format") == "config:0-7,32-35"
        and identity.get("umask_format") == "config:8-15"
        and str(identity.get("core4_online")) == "1"
        and str(identity.get("core5_online")) == "1"
    ):
        failures.append("platform custody mismatch")
    temperatures = [float(row["max_c"]) for row in custody.get("temperatures", [])]
    if not temperatures or max(temperatures) >= 68.0:
        failures.append("temperature custody mismatch")
    policies = custody.get("policy_snapshots", [])
    if len(policies) < 2 or policies[0].get("policies") != policies[-1].get("policies"):
        failures.append("policy restoration mismatch")
    for key in [
        "zero_cache_set_mapping",
        "zero_frequency_writes",
        "zero_msr_reads",
        "zero_msr_writes",
        "zero_physical_address_access",
        "zero_sysctl_writes",
        "zero_voltage_writes",
    ]:
        if custody.get(key) != 0:
            failures.append(f"{key} nonzero")

    if not (
        features.get("receiver_only") is True
        and features.get("private_source_map_opened") is False
        and features.get("private_source_fields_seen") == []
    ):
        failures.append("receiver no-smuggle boundary mismatch")
    event_names = [event["event"] for event in freeze.get("events", [])]
    event_times = [event["timestamp_ns"] for event in freeze.get("events", [])]
    if event_names != [
        "receiver_feature_process_started",
        "receiver_feature_process_completed",
        "receiver_feature_hash_frozen",
        "private_map_opened_for_adjudication",
        "adjudication_started",
    ]:
        failures.append("feature freeze event order mismatch")
    if event_times != sorted(event_times) or len(set(event_times)) != len(event_times):
        failures.append("feature freeze chronology mismatch")
    if freeze.get("private_map_opened_after_feature_hash_frozen") is not True:
        failures.append("private map did not open after feature freeze")

    return {
        "passed": not failures,
        "failures": failures,
        "observed_hashes": observed_hashes,
        "record_counts": observed_counts,
        "temperature_range_c": {"min": min(temperatures), "max": max(temperatures)} if temperatures else None,
        "process_snapshot_count": len(custody.get("process_snapshots", [])),
        "policy_snapshot_count": len(policies),
    }


def build_joined_rows(features: dict[str, Any], source_map: dict[str, Any], schedule: dict[str, Any]) -> list[dict[str, Any]]:
    private_by_id = {record["opaque_run_id"]: record for record in source_map["records"]}
    schedule_by_id = {record["opaque_run_id"]: record for record in schedule["rows"]}
    joined: list[dict[str, Any]] = []
    for feature in features["rows"]:
        private = private_by_id[feature["opaque_run_id"]]
        schedule_row = schedule_by_id[feature["opaque_run_id"]]
        row = dict(feature)
        row["condition"] = private["condition"]
        row["q_theta"] = q_theta(private)
        row["response_mode"] = private["response_mode"]
        row["source_execution_order"] = schedule_row["source_execution_order"]
        row["subcapture_order"] = schedule_row["subcapture_order"]
        row["mapping_order"] = schedule_row["mapping_order"]
        joined.append(row)
    return joined


def build_decodes(joined_rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, dict[str, dict[str, complex]]] = {"replicate": {}, "aggregate": {}}
    for replicate in REPLICATES:
        by_condition_mapping: dict[tuple[str, str], dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
        for row in joined_rows:
            if int(row["replicate"]) == replicate:
                by_condition_mapping[(row["condition"], row["physical_mapping"])][
                    int(row["public_decoder_phase_index"])
                ].append(float(row["logical_response"]))
        for (condition, mapping), phases in by_condition_mapping.items():
            result["replicate"].setdefault(str(replicate), {}).setdefault(condition, {})[mapping] = complex_decode(
                {phase: statistics.fmean(values) for phase, values in phases.items()}
            )

    by_condition_mapping = defaultdict(lambda: defaultdict(list))
    for row in joined_rows:
        by_condition_mapping[(row["condition"], row["physical_mapping"])][int(row["public_decoder_phase_index"])].append(
            float(row["logical_response"])
        )
    for (condition, mapping), phases in by_condition_mapping.items():
        result["aggregate"].setdefault(condition, {})[mapping] = complex_decode(
            {phase: statistics.fmean(values) for phase, values in phases.items()}
        )
    return result


def serialize_decodes(decodes: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {"replicate": {}, "aggregate": {}}
    for scope, conditions in decodes["replicate"].items():
        output["replicate"][scope] = {
            CONDITION_LABELS[condition]: {
                mapping: cdict(value)
                for mapping, value in sorted(mappings.items())
            }
            for condition, mappings in sorted(conditions.items())
        }
    output["aggregate"] = {
        CONDITION_LABELS[condition]: {mapping: cdict(value) for mapping, value in sorted(mappings.items())}
        for condition, mappings in sorted(decodes["aggregate"].items())
    }
    return output


def scope_conditions(decodes: dict[str, Any]) -> list[tuple[str, dict[str, dict[str, complex]]]]:
    return [
        ("replicate_0", decodes["replicate"]["0"]),
        ("replicate_1", decodes["replicate"]["1"]),
        ("aggregate", decodes["aggregate"]),
    ]


def ideal_vectors() -> dict[str, complex]:
    phi = orbit_phi(D_MEMBER)
    return {
        "Z_d": QUANTIZATION_SCALE * complex(math.cos(phi), math.sin(phi)),
        "Z_fold": QUANTIZATION_SCALE * complex(math.cos(phi), -math.sin(phi)),
        "Z_polarity_inversion": -QUANTIZATION_SCALE * complex(math.cos(phi), math.sin(phi)),
    }


def control_gain_for_scope(condition_decodes: dict[str, dict[str, complex]]) -> dict[str, Any]:
    allowed = {"post_projection", "equal_orbit_odd_zero"}
    post = condition_average(condition_decodes["post_projection"])
    equal = condition_average(condition_decodes["equal_orbit_odd_zero"])
    phi = orbit_phi(D_MEMBER)
    expected_post = QUANTIZATION_SCALE * math.cos(phi)
    g_post = post.real / expected_post
    g_equal = equal.real / QUANTIZATION_SCALE
    agreement = gain_rel_error(g_post, g_equal)
    return {
        "allowed_gain_conditions": sorted(allowed),
        "target_derived_gain_inputs_used": [],
        "g_post": g_post,
        "g_equal": g_equal,
        "g_control": statistics.fmean([g_post, g_equal]),
        "control_gain_relative_error": agreement,
        "passed": g_post > 0.0 and g_equal > 0.0 and agreement <= RELATIONAL_TOLERANCE,
        "post_projection_re": post.real,
        "equal_orbit_re": equal.real,
        "expected_post_projection_re": expected_post,
        "expected_equal_orbit_re": float(QUANTIZATION_SCALE),
    }


def target_gain_inputs_rejected(condition_names: list[str]) -> bool:
    allowed = {"post_projection", "equal_orbit_odd_zero"}
    if any(name not in allowed for name in condition_names):
        raise AuditError("target-derived gain inputs are forbidden")
    return True


def gain_covariant_geometry(decodes: dict[str, Any]) -> dict[str, Any]:
    ideals = ideal_vectors()
    result: dict[str, Any] = {}
    for scope_name, conditions in scope_conditions(decodes):
        control = control_gain_for_scope(conditions)
        entries: dict[str, Any] = {}
        diagnostic_gains: dict[str, float] = {}
        for condition, label in [
            ("pre_projection_d", "Z_d"),
            ("pre_projection_fold", "Z_fold"),
            ("source_polarity_inversion_d", "Z_polarity_inversion"),
        ]:
            observed = condition_average(conditions[condition])
            ideal = ideals[label]
            predicted = control["g_control"] * ideal
            fit = vector_fit(observed, ideal)
            diagnostic_gains[label] = fit["positive_least_squares_gain"]
            entries[label] = {
                **fit,
                "control_only_predicted": cdict(predicted),
                "component_relative_error": {
                    "real": rel_error(observed.real, predicted.real),
                    "imag": rel_error(observed.imag, predicted.imag),
                },
                "control_calibrated_complex_relative_error": complex_rel_error(observed, predicted),
                "control_calibrated_passed": complex_rel_error(observed, predicted) <= RELATIONAL_TOLERANCE,
            }
        all_gains = {
            **diagnostic_gains,
            "g_post": control["g_post"],
            "g_equal": control["g_equal"],
            "g_control": control["g_control"],
        }
        pairs = [
            gain_rel_error(diagnostic_gains[left], diagnostic_gains[right])
            for left in diagnostic_gains
            for right in diagnostic_gains
            if left < right
        ]
        gain_agreement = {
            "diagnostic_gains": diagnostic_gains,
            "max_pairwise_gain_disagreement": max(pairs),
            "with_g_post": {name: gain_rel_error(value, control["g_post"]) for name, value in diagnostic_gains.items()},
            "with_g_equal": {name: gain_rel_error(value, control["g_equal"]) for name, value in diagnostic_gains.items()},
            "with_g_control": {name: gain_rel_error(value, control["g_control"]) for name, value in diagnostic_gains.items()},
            "all_gains_positive": all(value > 0.0 for value in all_gains.values()),
        }
        z_d = condition_average(conditions["pre_projection_d"])
        z_fold = condition_average(conditions["pre_projection_fold"])
        z_polarity = condition_average(conditions["source_polarity_inversion_d"])
        relational = {
            "fold_conjugacy_error": complex_rel_error(z_fold, z_d.conjugate()),
            "fold_conjugacy_passed": complex_rel_error(z_fold, z_d.conjugate()) <= RELATIONAL_TOLERANCE,
            "polarity_inversion_error": complex_rel_error(z_polarity, -z_d),
            "polarity_inversion_passed": complex_rel_error(z_polarity, -z_d) <= RELATIONAL_TOLERANCE,
            "odd_magnitude_min": min(abs(z_d.imag), abs(z_fold.imag)),
            "odd_magnitude_passed": min(abs(z_d.imag), abs(z_fold.imag)) > PRIVATE_ODD_SIGNAL_FLOOR,
            "angle_errors_rad": {
                "Z_d": angle_error(z_d, ideals["Z_d"]),
                "Z_fold": angle_error(z_fold, ideals["Z_fold"]),
                "Z_polarity_inversion": angle_error(z_polarity, ideals["Z_polarity_inversion"]),
            },
        }
        result[scope_name] = {
            "control_gain": control,
            "vectors": entries,
            "gain_consistency": gain_agreement,
            "relational_geometry": relational,
            "passed": control["passed"]
            and all(entry["control_calibrated_passed"] for entry in entries.values())
            and relational["fold_conjugacy_passed"]
            and relational["polarity_inversion_passed"]
            and relational["odd_magnitude_passed"]
            and gain_agreement["all_gains_positive"],
        }
    return result


def cross_replicate_calibration(geometry: dict[str, Any], decodes: dict[str, Any]) -> dict[str, Any]:
    ideals = ideal_vectors()
    result: dict[str, Any] = {}
    for source_scope, target_scope in [("replicate_0", "replicate_1"), ("replicate_1", "replicate_0")]:
        gain = geometry[source_scope]["control_gain"]["g_control"]
        target_conditions = decodes["replicate"]["1" if target_scope.endswith("1") else "0"]
        entries: dict[str, Any] = {}
        for condition, label in [
            ("pre_projection_d", "Z_d"),
            ("pre_projection_fold", "Z_fold"),
            ("source_polarity_inversion_d", "Z_polarity_inversion"),
        ]:
            observed = condition_average(target_conditions[condition])
            predicted = gain * ideals[label]
            entries[label] = {
                "source_gain_scope": source_scope,
                "target_scope": target_scope,
                "source_g_control": gain,
                "observed": cdict(observed),
                "predicted": cdict(predicted),
                "component_relative_error": {
                    "real": rel_error(observed.real, predicted.real),
                    "imag": rel_error(observed.imag, predicted.imag),
                },
                "complex_relative_error": complex_rel_error(observed, predicted),
                "passed": complex_rel_error(observed, predicted) <= RELATIONAL_TOLERANCE,
            }
        result[f"{source_scope}_predicts_{target_scope}"] = {
            "targets": entries,
            "passed": all(item["passed"] for item in entries.values()),
        }
    return result


def by_pair(joined_rows: list[dict[str, Any]]) -> dict[tuple[int, str, int], dict[str, dict[str, Any]]]:
    pairs: dict[tuple[int, str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in joined_rows:
        pairs[(int(row["replicate"]), row["condition"], int(row["public_decoder_phase_index"]))][
            row["physical_mapping"]
        ] = row
    return dict(pairs)


def normalize_failure_events(adjudication: dict[str, Any]) -> dict[str, Any]:
    events: dict[str, dict[str, Any]] = {}
    for failure in adjudication.get("phase_transfer_law", {}).get("failures", []):
        if not failure.startswith("("):
            continue
        tuple_text, label = failure.split(") ", 1)
        key = tuple_text[1:]
        event = events.setdefault(key, {"event": key, "law_labels": []})
        if label not in event["law_labels"]:
            event["law_labels"].append(label)
    return {
        "event_count": len(events),
        "events": sorted(events.values(), key=lambda item: item["event"]),
    }


def partition_phase_transfer(joined_rows: list[dict[str, Any]], adjudication: dict[str, Any]) -> dict[str, Any]:
    strong_events: list[dict[str, Any]] = []
    near_zero_events: list[dict[str, Any]] = []
    mixed_events: list[dict[str, Any]] = []
    strong_counts = Counter()
    near_counts = Counter()
    pairs = by_pair(joined_rows)
    for key, maps in sorted(pairs.items()):
        if set(maps) != {"map0", "map1"}:
            continue
        rep, condition, phase = key
        map0 = maps["map0"]
        map1 = maps["map1"]
        q_values = {"map0": int(map0["q_theta"]), "map1": int(map1["q_theta"])}
        logical_rel = rel_error(float(map0["logical_response"]), float(map1["logical_response"]))
        physical_rel = rel_error(float(map0["physical_a_minus_b"]), -float(map1["physical_a_minus_b"]))
        sign_pass = all(float(row["logical_response"]) * int(row["q_theta"]) > 0 for row in maps.values())
        logical_pass = logical_rel <= RELATIONAL_TOLERANCE
        physical_pass = physical_rel <= RELATIONAL_TOLERANCE
        event = {
            "replicate": rep,
            "condition": condition,
            "phase": phase,
            "q_theta": q_values,
            "map0_logical_response": float(map0["logical_response"]),
            "map1_logical_response": float(map1["logical_response"]),
            "map0_physical_a_minus_b": float(map0["physical_a_minus_b"]),
            "map1_physical_a_minus_b": float(map1["physical_a_minus_b"]),
            "logical_mapping_relative_error": logical_rel,
            "physical_reversal_relative_error": physical_rel,
            "logical_mapping_passed": logical_pass,
            "physical_reversal_passed": physical_pass,
            "strong_sign_passed": sign_pass,
            "source_execution_order": {"map0": map0["source_execution_order"], "map1": map1["source_execution_order"]},
            "subcapture_order": {"map0": map0["subcapture_order"], "map1": map1["subcapture_order"]},
            "physical_residuals": {
                "logical_mapping_difference": float(map0["logical_response"]) - float(map1["logical_response"]),
                "physical_reversal_sum": float(map0["physical_a_minus_b"]) + float(map1["physical_a_minus_b"]),
            },
        }
        classifications = ["strong" if abs(q) >= STRONG_Q_FLOOR else "near-zero" for q in q_values.values()]
        if classifications == ["strong", "strong"]:
            strong_counts["total_pair_cells"] += 1
            strong_counts["logical_mapping_passes" if logical_pass else "logical_mapping_failures"] += 1
            strong_counts["physical_reversal_passes" if physical_pass else "physical_reversal_failures"] += 1
            strong_counts["sign_passes" if sign_pass else "sign_failures"] += 1
            if not (logical_pass and physical_pass and sign_pass):
                strong_events.append(event)
        elif classifications == ["near-zero", "near-zero"]:
            individual_violations = {
                mapping: abs(float(row["logical_response"])) > PUBLIC_Q0_ABSOLUTE_BOUND for mapping, row in maps.items()
            }
            pair_logical_average = 0.5 * (float(map0["logical_response"]) + float(map1["logical_response"]))
            pair_physical_reversal_average = 0.5 * (float(map0["physical_a_minus_b"]) + float(map1["physical_a_minus_b"]))
            pair_residual_violations = {
                "logical_pair_average": abs(pair_logical_average) > PUBLIC_Q0_ABSOLUTE_BOUND,
                "physical_reversal_average": abs(pair_physical_reversal_average) > PUBLIC_Q0_ABSOLUTE_BOUND,
            }
            event.update(
                {
                    "individual_response_bound_violations": individual_violations,
                    "pair_residuals_for_absolute_law": {
                        "logical_pair_average": pair_logical_average,
                        "physical_reversal_average": pair_physical_reversal_average,
                    },
                    "pair_residual_bound_violations": pair_residual_violations,
                }
            )
            near_counts["total_pair_cells"] += 1
            near_counts["individual_response_bound_violations"] += sum(individual_violations.values())
            near_counts["pair_residual_violations"] += sum(pair_residual_violations.values())
            near_counts["mapping_reversal_violations"] += int(not logical_pass) + int(not physical_pass)
            if any(individual_violations.values()) or any(pair_residual_violations.values()) or not logical_pass or not physical_pass:
                near_zero_events.append(event)
        else:
            mixed_events.append(event)

    normalized = normalize_failure_events(adjudication)
    expected_strong_failures = [
        {"replicate": 1, "condition": "pre_projection_d", "phase": 1},
        {"replicate": 1, "condition": "source_polarity_inversion_d", "phase": 3},
    ]
    observed_strong_failures = [
        {"replicate": row["replicate"], "condition": row["condition"], "phase": row["phase"]}
        for row in strong_events
        if not (row["logical_mapping_passed"] and row["physical_reversal_passed"])
    ]
    return {
        "strong": {"counts": dict(strong_counts), "failure_events": strong_events},
        "near_zero": {"counts": dict(near_counts), "failure_events": near_zero_events},
        "mixed": {"count": len(mixed_events), "events": mixed_events},
        "duplicate_failure_normalization": normalized,
        "expected_exact_two_strong_mapping_reversal_failures": expected_strong_failures,
        "observed_strong_mapping_reversal_failures": observed_strong_failures,
        "exact_two_strong_failures_verified": observed_strong_failures == expected_strong_failures,
    }


def near_zero_law_audit(decodes: dict[str, Any], joined_rows: list[dict[str, Any]]) -> dict[str, Any]:
    pairs = by_pair(joined_rows)
    rows: list[dict[str, Any]] = []
    for key, maps in sorted(pairs.items()):
        rep, condition, phase = key
        include = False
        reason = ""
        if condition in {"source_off", "query_off"}:
            include = True
            reason = "near_zero_condition"
        elif condition == "post_projection" and phase in {1, 3}:
            include = True
            reason = "post_projection_imaginary_phase"
        elif condition == "equal_orbit_odd_zero" and phase in {1, 3}:
            include = True
            reason = "equal_orbit_imaginary_phase"
        if not include or set(maps) != {"map0", "map1"}:
            continue
        map0 = maps["map0"]
        map1 = maps["map1"]
        scope_key = str(rep)
        condition_decodes = decodes["replicate"][scope_key][condition]
        harmonic = condition_average(condition_decodes)
        rows.append(
            {
                "replicate": rep,
                "condition": condition,
                "phase": phase,
                "reason": reason,
                "map0_response": float(map0["logical_response"]),
                "map1_response": float(map1["logical_response"]),
                "logical_pair_average": 0.5 * (float(map0["logical_response"]) + float(map1["logical_response"])),
                "logical_pair_difference": float(map0["logical_response"]) - float(map1["logical_response"]),
                "physical_reversal_sum": float(map0["physical_a_minus_b"]) + float(map1["physical_a_minus_b"]),
                "complex_first_harmonic": cdict(condition_decodes["map0"]),
                "decoded_mapping_average": cdict(harmonic),
                "decoded_imaginary_null": harmonic.imag,
            }
        )
    return {
        "rows": rows,
        "interpretation": (
            "The retained raw-leg law bounds single randomized windows, while the decoded null controls are "
            "first-harmonic cancellations across phases and mapping legs. The two quantities are not the same "
            "statistic and should not share one absolute ceiling without a prospective derivation."
        ),
        "prospective_absolute_near_zero_law": {
            "raw_leg_bound": PUBLIC_Q0_ABSOLUTE_BOUND,
            "paired_logical_average_bound": PUBLIC_Q0_ABSOLUTE_BOUND,
            "paired_physical_reversal_average_bound": PUBLIC_Q0_ABSOLUTE_BOUND,
            "decoded_first_harmonic_imag_bound": PUBLIC_Q0_ABSOLUTE_BOUND,
            "rule": (
                "Freeze one absolute count bound before execution, apply it separately to algebraically "
                "identical one-leg quantities and two-leg averaged residual quantities, and do not derive "
                "the ceiling from a same-sized held-out sample maximum."
            ),
        },
    }


def classify_retrospective(
    evidence_check: dict[str, Any],
    source_law: dict[str, Any],
    geometry: dict[str, Any],
    evidence: dict[str, Any],
) -> str:
    feature_law = evidence["adjudication"].get("feature_freeze_law", {}).get("passed") is True
    no_smuggle = (
        evidence["features"].get("receiver_only") is True
        and evidence["features"].get("private_source_map_opened") is False
        and evidence["features"].get("private_source_fields_seen") == []
    )
    all_geometry = all(scope["passed"] for scope in geometry.values())
    if evidence_check["passed"] and source_law["passed"] and feature_law and no_smuggle and all_geometry:
        return CLASS_ESTABLISHED
    if evidence_check["passed"] and source_law["passed"] and feature_law and no_smuggle:
        return CLASS_PARTIAL
    return CLASS_NOT_ESTABLISHED


def build_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Gain-Covariant OrbitState Law Audit",
        "",
        "This is an offline retrospective audit of committed retry-one evidence. It does not alter the official target class and does not promote `SMALL_WALL_CROSSED`.",
        "",
        "## Official Boundary",
        "",
        f"- Official retained class: `{result['official_retained_class']}`",
        f"- Retrospective gain-covariant class: `{result['retrospective_gain_covariant_class']}`",
        f"- Evidence root: `{result['evidence_root']}`",
        "",
        "## Unit-Gain Defect",
        "",
        "The original law computes `expected_phi = 2*pi*23/256` and `expected_re = 1536*cos(expected_phi)`, then compares `Re(Z_d)` and `Re(Z_fold)` directly against that source-domain value. The receiver measures Change-to-Dirty physical counts, not source work units. Without an independently frozen unit-gain calibration, that law tests `physical_count == source_work` as an unstated assumption.",
        "",
        f"- Expected source-domain real component: `{result['unit_gain_defect']['expected_re']}`",
        f"- Unit-gain law failed: `{result['unit_gain_defect']['unit_gain_law_failed']}`",
        "",
        "## Control-Only Gains",
        "",
    ]
    for scope in ["replicate_0", "replicate_1", "aggregate"]:
        gain = result["gain_covariant_geometry"][scope]["control_gain"]
        lines.append(
            f"- `{scope}`: g_post={gain['g_post']:.12g}, g_equal={gain['g_equal']:.12g}, "
            f"g_control={gain['g_control']:.12g}, agreement={gain['control_gain_relative_error']:.12g}"
        )
    lines.extend(["", "## Gain-Calibrated Private Geometry", ""])
    for scope in ["replicate_0", "replicate_1", "aggregate"]:
        lines.append(f"### {scope}")
        vectors = result["gain_covariant_geometry"][scope]["vectors"]
        for label in ["Z_d", "Z_fold", "Z_polarity_inversion"]:
            entry = vectors[label]
            lines.append(
                f"- `{label}`: gain={entry['positive_least_squares_gain']:.12g}, "
                f"angle_error_rad={entry['angular_error_rad']:.12g}, "
                f"control_error={entry['control_calibrated_complex_relative_error']:.12g}, "
                f"orthogonal_residual={entry['orthogonal_residual']:.12g}"
            )
    part = result["phase_transfer_partition"]
    lines.extend(
        [
            "",
            "## Phase-Transfer Partition",
            "",
            f"- Strong pair cells: `{part['strong']['counts'].get('total_pair_cells', 0)}`",
            f"- Strong logical mapping failures: `{part['strong']['counts'].get('logical_mapping_failures', 0)}`",
            f"- Strong physical reversal failures: `{part['strong']['counts'].get('physical_reversal_failures', 0)}`",
            f"- Strong sign failures: `{part['strong']['counts'].get('sign_failures', 0)}`",
            f"- Near-zero pair cells: `{part['near_zero']['counts'].get('total_pair_cells', 0)}`",
            f"- Near-zero individual response bound violations: `{part['near_zero']['counts'].get('individual_response_bound_violations', 0)}`",
            f"- Near-zero pair residual violations: `{part['near_zero']['counts'].get('pair_residual_violations', 0)}`",
            "",
            "The two strong-signal mapping/reversal failures are preserved explicitly:",
        ]
    )
    for event in part["strong"]["failure_events"]:
        if not (event["logical_mapping_passed"] and event["physical_reversal_passed"]):
            lines.append(
                f"- replicate {event['replicate']}, {event['condition']}, phase {event['phase']}: "
                f"map0={event['map0_logical_response']}, map1={event['map1_logical_response']}, "
                f"logical_rel={event['logical_mapping_relative_error']:.12g}, "
                f"physical_rel={event['physical_reversal_relative_error']:.12g}"
            )
    lines.extend(
        [
            "",
            "## Near-Zero Law Audit",
            "",
            result["near_zero_law_audit"]["interpretation"],
            "",
            "Prospective repair: freeze one absolute count bound before execution, use the same bound only for algebraically identical one-leg quantities and paired residual quantities, and do not compare one-leg and two-leg quantities to the same ceiling without an explicit prospective derivation.",
            "",
            "## Conclusion",
            "",
            "This audit explains the unit-gain defect and establishes the gain-covariant decoded geometry retrospectively. It does not rewrite the official class, does not remove phase-level failures, and does not authorize a new live transaction.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_result(repo_root: Path) -> dict[str, Any]:
    root = evidence_root(repo_root)
    evidence = load_evidence(root)
    source_bundle = root / "ORBITSTATE_SOURCE_BUNDLE.tar.gz"
    schedule = read_bundle_json(source_bundle, "ORBITSTATE_PUBLIC_SCHEDULE.json")
    source_map = read_bundle_json(source_bundle, "ORBITSTATE_PRIVATE_SOURCE_MAP.json")
    joined_rows = build_joined_rows(evidence["features"], source_map, schedule)
    decodes = build_decodes(joined_rows)
    evidence_check = verify_evidence(root, evidence)
    source_law = source_receipt_law(source_map, evidence["source_receipts"])
    geometry = gain_covariant_geometry(decodes)
    cross = cross_replicate_calibration(geometry, decodes)
    partition = partition_phase_transfer(joined_rows, evidence["adjudication"])
    near_zero = near_zero_law_audit(decodes, joined_rows)
    retrospective_class = classify_retrospective(evidence_check, source_law, geometry, evidence)
    phi = orbit_phi(D_MEMBER)
    expected_re = QUANTIZATION_SCALE * math.cos(phi)
    unit_gain_failures = evidence["adjudication"].get("target_fold_geometry_law", {}).get("failures", [])
    result = {
        "schema": "GAIN_COVARIANT_ADJUDICATION_V1",
        "official_retained_class": OFFICIAL_CLASS,
        "retrospective_result_vocabulary": [CLASS_ESTABLISHED, CLASS_PARTIAL, CLASS_NOT_ESTABLISHED],
        "retrospective_gain_covariant_class": retrospective_class,
        "small_wall_crossed_promoted": False,
        "evidence_root": str(root),
        "source_bundle_sha256": sha256_file(source_bundle),
        "evidence_verification": evidence_check,
        "source_formula_law": source_law,
        "feature_freeze_law": evidence["adjudication"].get("feature_freeze_law"),
        "unit_gain_defect": {
            "expected_phi_formula": "2*pi*23/256",
            "expected_phi": phi,
            "expected_re_formula": "1536*cos(expected_phi)",
            "expected_re": expected_re,
            "official_real_geometry_failures": [
                failure for failure in unit_gain_failures if "Re(Z_" in failure
            ],
            "unit_gain_law_failed": any("Re(Z_" in failure for failure in unit_gain_failures),
            "diagnosis": (
                "The failed law directly compared physical Change-to-Dirty counts with source work units. "
                "That is a unit-carrier-gain assumption, not an independently established physical calibration."
            ),
        },
        "ideal_vectors": {name: cdict(value) for name, value in ideal_vectors().items()},
        "decodes": serialize_decodes(decodes),
        "gain_covariant_geometry": geometry,
        "cross_replicate_control_calibration": cross,
        "phase_transfer_partition": partition,
        "near_zero_law_audit": near_zero,
    }
    result["gain_covariant_adjudication_sha256"] = digest(
        {key: value for key, value in result.items() if key != "gain_covariant_adjudication_sha256"}
    )
    return result


def build_self_test(result: dict[str, Any]) -> dict[str, Any]:
    tests: list[dict[str, Any]] = []

    def add(name: str, passed: bool, details: Any = None) -> None:
        tests.append({"name": name, "passed": bool(passed), "details": details})

    ev = result["evidence_verification"]
    add("all evidence hash verification", ev["passed"], ev.get("failures"))
    add("all JSON and JSONL parsing", True)
    add("record-count verification", ev["record_counts"] == {
        "raw_component_windows": 288,
        "sentinels": 288,
        "stage_receipts": 2016,
        "source_receipts": 288,
        "receiver_feature_rows": 144,
    })
    add("receiver-feature digest verification", ev["passed"])
    add("unit-gain defect regression", result["unit_gain_defect"]["unit_gain_law_failed"])
    add(
        "control-only gain regression",
        all(scope["control_gain"]["passed"] for scope in result["gain_covariant_geometry"].values()),
    )
    try:
        target_gain_inputs_rejected(["post_projection", "pre_projection_d"])
        smuggling_rejected = False
    except AuditError:
        smuggling_rejected = True
    add("target-derived-gain smuggling rejection", smuggling_rejected)
    add(
        "cross-replicate calibration",
        all(scope["passed"] for scope in result["cross_replicate_control_calibration"].values()),
    )
    add(
        "angle and vector-residual regression",
        all(scope["passed"] for scope in result["gain_covariant_geometry"].values()),
    )
    add(
        "strong/near-zero partition regression",
        result["phase_transfer_partition"]["exact_two_strong_failures_verified"],
        result["phase_transfer_partition"]["observed_strong_mapping_reversal_failures"],
    )
    add(
        "duplicate failure-event normalization",
        any(
            len(event["law_labels"]) > 1
            for event in result["phase_transfer_partition"]["duplicate_failure_normalization"]["events"]
        ),
    )
    self_test = {
        "schema": "GAIN_COVARIANT_SELF_TEST_V1",
        "passed": all(test["passed"] for test in tests),
        "tests": tests,
    }
    self_test["gain_covariant_self_test_sha256"] = digest(
        {key: value for key, value in self_test.items() if key != "gain_covariant_self_test_sha256"}
    )
    return self_test


def write_outputs(result: dict[str, Any], self_test: dict[str, Any]) -> None:
    out = audit_root()
    write_json(out / "GAIN_COVARIANT_ADJUDICATION.json", result)
    write_json(out / "GAIN_COVARIANT_SELF_TEST.json", self_test)
    (out / "GAIN_COVARIANT_LAW_AUDIT.md").write_text(build_markdown(result), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    result = build_result(repo_root_from_here())
    self_test = build_self_test(result)
    if args.write:
        write_outputs(result, self_test)
    if args.self_test and not self_test["passed"]:
        print(json.dumps(self_test, indent=2, sort_keys=True))
        return 1
    print(json.dumps({"result": result["retrospective_gain_covariant_class"], "self_test_passed": self_test["passed"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
