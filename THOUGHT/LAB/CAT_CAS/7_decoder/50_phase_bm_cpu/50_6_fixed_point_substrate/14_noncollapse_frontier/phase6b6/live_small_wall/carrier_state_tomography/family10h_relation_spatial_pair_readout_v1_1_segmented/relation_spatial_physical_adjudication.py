#!/usr/bin/env python3
from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import random
import tarfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import relation_spatial_public as pub


RESULT_CONFIRMED = "FAMILY10H_SPATIAL_RELATION_READOUT_CALIBRATED_PROSPECTIVE"
RESULT_NOT_OBSERVED = "FAMILY10H_SPATIAL_RELATION_READOUT_NOT_OBSERVED_PROSPECTIVE"
RESULT_CANDIDATE = "FAMILY10H_SPATIAL_RELATION_READOUT_CANDIDATE_PROSPECTIVE"
RESULT_INVALID = "FAMILY10H_SPATIAL_RELATION_READOUT_CUSTODY_INVALID"

POSITIVE_CLAIM = pub.MAXIMUM_FUTURE_CLAIM
NEGATIVE_CLAIM = pub.NEGATIVE_FUTURE_CLAIM
FIXTURE_SOURCE_SHA = "4" * 40
FIXTURE_FREEZE_SHA = "5" * 40


def physical_threshold_contract() -> dict[str, Any]:
    contract = {
        "schema": "FAMILY10H_RELATION_SPATIAL_PHYSICAL_THRESHOLD_CONTRACT_V1",
        "threshold_status": "prospective_spatial_pair_threshold_law_frozen_before_physical_acquisition",
        "basis": {
            "relation_only_attempt": pub.RELATION_ONLY_EVIDENCE_COMMIT,
            "relation_lifetime_attempt": pub.RELATION_LIFETIME_EVIDENCE_COMMIT,
            "spatial_target_data_used": False,
        },
        "matched_permutation_null": {
            "seed": pub.MATCHED_PERMUTATION_SEED,
            "permutation_count": pub.MATCHED_PERMUTATION_COUNT,
            "quantile": 0.99,
            "preserves": [
                "A latency vector",
                "B latency vector",
                "measurement-order counts",
                "cyclic-origin coverage",
                "line-distance bucket within relation rows",
                "scalar marginals",
            ],
        },
        "calibrated_requirements": {
            "observed_abs_mean_R_spatial_exceeds_matched_permutation_q99": True,
            "sign_stable_across_sessions_mappings_source_orders_query_orders_and_cyclic_origins": True,
            "R_shaped_control_segments_inside_own_null": True,
            "primary_control_separation_4x": True,
            "A_first_and_B_first_primary_strata_survive": True,
            "not_explained_by_A_or_B_marginal_latency_alone": True,
            "complete_custody_required": True,
        },
        "claim_boundary": {
            "maximum_claim": POSITIVE_CLAIM,
            "small_wall_crossed": False,
            "full_tomography_established": False,
        },
    }
    contract["threshold_contract_sha256"] = pub.digest({k: v for k, v in contract.items() if k != "threshold_contract_sha256"})
    return contract


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def strict_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")


def jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(strict_json_bytes(row) + b"\n" for row in rows)


def parse_jsonl_bytes(data: bytes) -> list[dict[str, Any]]:
    rows = []
    for line in data.decode("utf-8").splitlines():
        if not line:
            raise ValueError("blank JSONL line")
        parsed = json.loads(line)
        if not isinstance(parsed, dict):
            raise ValueError("JSONL row is not an object")
        rows.append(parsed)
    return rows


def tar_member_bytes(archive_bytes: bytes) -> dict[str, bytes]:
    members: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:*") as tf:
        for member in tf.getmembers():
            if not member.isfile():
                continue
            extracted = tf.extractfile(member)
            if extracted is not None:
                members[member.name] = extracted.read()
    return members


def find_member(members: dict[str, bytes], suffix: str) -> str | None:
    matches = [name for name in sorted(members) if name == suffix or name.endswith("/" + suffix)]
    return matches[0] if len(matches) == 1 else None


def custody_envelope_from_archive_bytes(
    archive_bytes: bytes, *, archive_path: str | None = None, embed_archive_bytes: bool = False
) -> dict[str, Any]:
    members = tar_member_bytes(archive_bytes)
    required = {
        "raw_records": "raw_records.jsonl",
        "pair_observations": "pair_observations.jsonl",
        "source_death_receipts": "source_death_receipts.jsonl",
        "feature_freeze": "feature_freeze.json",
        "target_execution_receipt": "target_execution_receipt.json",
        "manifest": "RELATION_SPATIAL_IMPLEMENTATION_MANIFEST.json",
        "runtime_binary": "relation_spatial_runtime",
        "deployment_custody": pub.DEPLOYMENT_CUSTODY_FILENAME,
    }
    member_map = {}
    for label, suffix in required.items():
        member = find_member(members, suffix)
        if member is None:
            raise ValueError(f"required archive member missing or ambiguous: {suffix}")
        member_map[label] = member
    manifest = json.loads(members[member_map["manifest"]].decode("utf-8"))
    deployment = json.loads(members[member_map["deployment_custody"]].decode("utf-8"))
    envelope = {
        "schema": "FAMILY10H_RELATION_SPATIAL_CRYPTOGRAPHIC_CUSTODY_ENVELOPE_V1",
        "copied_back_archive_sha256": sha256_bytes(archive_bytes),
        "copied_back_archive_size": len(archive_bytes),
        "archive_member_map": member_map,
        "archive_inventory": {
            label: {"member": member, "sha256": sha256_bytes(members[member]), "size_bytes": len(members[member])}
            for label, member in member_map.items()
        },
        "target_execution_receipt_sha256": sha256_bytes(members[member_map["target_execution_receipt"]]),
        "runtime_binary_sha256": sha256_bytes(members[member_map["runtime_binary"]]),
        "implementation_manifest_sha256": sha256_bytes(members[member_map["manifest"]]),
        "source_sha": manifest.get("authority_binding", {}).get("relation_source_authority_commit"),
        "freeze_sha": deployment.get("relation_manifest_freeze_commit"),
    }
    if archive_path is not None:
        envelope["copied_back_archive_path"] = archive_path
    if embed_archive_bytes:
        envelope["copied_back_archive_bytes_b64"] = base64.b64encode(archive_bytes).decode("ascii")
    return envelope


def custody_envelope_from_archive_path(path: str | Path) -> dict[str, Any]:
    archive_path = Path(path)
    return custody_envelope_from_archive_bytes(archive_path.read_bytes(), archive_path=str(archive_path))


def archive_bytes_from_custody(custody: dict[str, Any]) -> tuple[bytes | None, str | None]:
    path = custody.get("copied_back_archive_path")
    if isinstance(path, str) and path:
        try:
            return Path(path).read_bytes(), None
        except OSError as exc:
            return None, f"copied-back archive path unreadable: {exc}"
    encoded = custody.get("copied_back_archive_bytes_b64")
    if isinstance(encoded, str) and encoded:
        try:
            return base64.b64decode(encoded), None
        except ValueError:
            return None, "copied-back archive bytes malformed"
    return None, "copied-back archive bytes or path missing"


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    idx = (len(ordered) - 1) * q
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (idx - lo)


def distribution(values: list[float]) -> dict[str, Any]:
    abs_values = [abs(v) for v in values]
    signs = Counter(1 if v > 0 else -1 if v < 0 else 0 for v in values)
    return {
        "count": len(values),
        "mean": mean(values),
        "abs_mean": mean(abs_values),
        "abs_of_mean": abs(mean(values)),
        "max_abs": max(abs_values) if abs_values else 0.0,
        "q50_abs": quantile(abs_values, 0.50),
        "q95_abs": quantile(abs_values, 0.95),
        "q99_abs": quantile(abs_values, 0.99),
        "sign_counts": {str(k): signs[k] for k in [-1, 0, 1]},
    }


def ranks(values: list[float]) -> list[float]:
    result = [0.0] * len(values)
    ordered = sorted((value, index) for index, value in enumerate(values))
    pos = 0
    while pos < len(ordered):
        end = pos + 1
        while end < len(ordered) and ordered[end][0] == ordered[pos][0]:
            end += 1
        rank = (pos + end - 1) / 2.0
        for _, index in ordered[pos:end]:
            result[index] = rank
        pos = end
    return result


def pearson(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    ma = mean(a)
    mb = mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = sum((x - ma) ** 2 for x in a)
    db = sum((y - mb) ** 2 for y in b)
    if da <= 0.0 or db <= 0.0:
        return 0.0
    return num / math.sqrt(da * db)


def spearman(a: list[float], b: list[float]) -> float:
    return pearson(ranks(a), ranks(b))


def claim_boundary(observed: bool) -> dict[str, Any]:
    return {
        "spatial_relation_pair_readout_calibrated": observed,
        "maximum_future_claim": POSITIVE_CLAIM,
        "full_carrier_state_tomography_established": False,
        "physical_relational_memory_established": False,
        "catalytic_borrowing_established": False,
        "r2_restoration_established": False,
        "small_wall_crossed": False,
    }


def fail_closed(result_class: str, validation: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "FAMILY10H_RELATION_SPATIAL_PHYSICAL_ADJUDICATION_V1",
        "result_class": result_class,
        "passed": False,
        "scientific_claim": NEGATIVE_CLAIM,
        "claim_boundary": claim_boundary(False),
        "validation": validation,
    }


def validate_physical_packet(packet: dict[str, Any], schedule: dict[str, Any], *, require_custody: bool = True) -> dict[str, Any]:
    failures: list[str] = []
    rows = packet.get("raw_records")
    pairs = packet.get("pair_observations")
    deaths = packet.get("source_death_receipts")
    if not isinstance(rows, list) or not isinstance(pairs, list) or not isinstance(deaths, list):
        return {"passed": False, "failures": ["packet missing raw/pair/death lists"]}
    expected_rows = schedule["rows"]
    if len(rows) != schedule["tuple_count"]:
        failures.append("raw record count mismatch")
    if len(deaths) != schedule["tuple_count"]:
        failures.append("source-death receipt count mismatch")
    if len(pairs) != schedule["expected_pair_observation_count"]:
        failures.append("pair observation count mismatch")
    by_tuple_pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        by_tuple_pairs[str(pair.get("tuple_id"))].append(pair)
        if pair.get("positive_physical_claim") is True:
            failures.append("positive claim leakage in pair observation")
            break
    row_by_tuple = {row.get("tuple_id"): row for row in rows}
    for index, expected in enumerate(expected_rows[: len(rows)]):
        observed = rows[index]
        if observed.get("tuple_id") != expected["tuple_id"] or observed.get("execution_ordinal") != expected["execution_ordinal"]:
            failures.append(f"raw schedule mismatch at row {index}")
            break
        for key in ["block_id", "block_local_position", "row_role", "r_prepare", "r_query", "relation_match", "query", "session", "mapping", "source_order", "query_order", "cyclic_origin"]:
            if observed.get(key) != expected.get(key):
                failures.append(f"raw field mismatch {key} at row {index}")
                break
        if observed.get("positive_physical_claim") is True:
            failures.append("positive claim leakage in raw record")
            break
        row_pairs = by_tuple_pairs.get(expected["tuple_id"], [])
        if len(row_pairs) != pub.PAIR_SAMPLE_COUNT:
            failures.append(f"pair count mismatch for {expected['tuple_id']}")
            break
        sample_indices = sorted(pair.get("sample_index") for pair in row_pairs)
        if sample_indices != list(range(pub.PAIR_SAMPLE_COUNT)):
            failures.append(f"pair sample index mismatch for {expected['tuple_id']}")
            break
        expected_a = pub.selected_pair_indices(expected["execution_ordinal"])
        if sorted(pair.get("A_line_index") for pair in row_pairs) != sorted(expected_a):
            failures.append(f"A line sample mismatch for {expected['tuple_id']}")
            break
        if any(pair.get("source_alive_at_pair_measurement") is not True for pair in row_pairs):
            failures.append("source not alive during pair measurement")
            break
    for index, death in enumerate(deaths[: len(expected_rows)]):
        if death.get("tuple_id") != expected_rows[index]["tuple_id"]:
            failures.append(f"source-death tuple mismatch at row {index}")
            break
        if death.get("source_alive_during_query") is not True or death.get("source_alive_at_pair_measurement") is not True:
            failures.append("source lifecycle receipt is not alive-only")
            break
    if require_custody:
        custody = packet.get("custody_envelope")
        if not isinstance(custody, dict):
            failures.append("custody envelope missing")
        else:
            archive_bytes, archive_error = archive_bytes_from_custody(custody)
            if archive_error or archive_bytes is None:
                failures.append(archive_error or "archive unavailable")
            else:
                if custody.get("copied_back_archive_sha256") != sha256_bytes(archive_bytes):
                    failures.append("copied-back archive sha mismatch")
                if custody.get("copied_back_archive_size") != len(archive_bytes):
                    failures.append("copied-back archive size mismatch")
                try:
                    members = tar_member_bytes(archive_bytes)
                    member_map = custody.get("archive_member_map", {})
                    archived_raw = parse_jsonl_bytes(members[member_map["raw_records"]])
                    archived_pairs = parse_jsonl_bytes(members[member_map["pair_observations"]])
                    archived_deaths = parse_jsonl_bytes(members[member_map["source_death_receipts"]])
                    archived_feature = json.loads(members[member_map["feature_freeze"]].decode("utf-8"))
                    archived_receipt = json.loads(members[member_map["target_execution_receipt"]].decode("utf-8"))
                except (KeyError, ValueError, json.JSONDecodeError, tarfile.TarError) as exc:
                    failures.append(f"archive member parse failure: {exc}")
                else:
                    if archived_raw != rows:
                        failures.append("raw records archive member mismatch")
                    if archived_pairs != pairs:
                        failures.append("pair observations archive member mismatch")
                    if archived_deaths != deaths:
                        failures.append("source-death archive member mismatch")
                    if archived_feature.get("pair_observation_count") != len(pairs):
                        failures.append("feature-freeze pair count mismatch")
                    if archived_receipt.get("pair_observation_count") != len(pairs):
                        failures.append("target receipt pair count mismatch")
    return {
        "passed": not failures,
        "failures": failures,
        "raw_record_count": len(rows),
        "pair_observation_count": len(pairs),
        "source_death_receipt_count": len(deaths),
        "expected_pair_observation_count": schedule["expected_pair_observation_count"],
    }


def pair_groups(packet: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in packet["pair_observations"]:
        grouped[str(pair["tuple_id"])].append(pair)
    for key in grouped:
        grouped[key].sort(key=lambda item: int(item["sample_index"]))
    return grouped


def row_c_pairs(packet: dict[str, Any]) -> list[dict[str, Any]]:
    grouped = pair_groups(packet)
    result = []
    for row in packet["raw_records"]:
        pairs = grouped[row["tuple_id"]]
        a = [float(pair["A_first_touch_cycles"]) for pair in pairs]
        b = [float(pair["B_first_touch_cycles"]) for pair in pairs]
        result.append({**row, "C_pair_recomputed": spearman(a, b), "A_latency_mean": mean(a), "B_latency_mean": mean(b)})
    return result


def blocks(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[row["block_id"]].append(row)
    return out


def r_spatial_records(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    failures: list[str] = []
    values: list[dict[str, Any]] = []
    for block_id, block in blocks(rows).items():
        relation_rows = {row["relation_cell"]: row for row in block if row["row_role"] == "relation_matrix"}
        needed = {
            "prepare_r0__query_r0",
            "prepare_r0__query_r1",
            "prepare_r1__query_r0",
            "prepare_r1__query_r1",
        }
        if set(relation_rows) != needed:
            failures.append(f"{block_id} missing relation cells")
            continue
        r = 0.5 * (
            relation_rows["prepare_r0__query_r0"]["C_pair_recomputed"]
            + relation_rows["prepare_r1__query_r1"]["C_pair_recomputed"]
            - relation_rows["prepare_r0__query_r1"]["C_pair_recomputed"]
            - relation_rows["prepare_r1__query_r0"]["C_pair_recomputed"]
        )
        first = block[0]
        values.append(
            {
                "block_id": block_id,
                "R_spatial": r,
                "session": first["session"],
                "replicate": first["replicate"],
                "mapping": first["mapping"],
                "source_order": first["source_order"],
                "query_order": first["query_order"],
                "cyclic_origin": first["cyclic_origin"],
            }
        )
    return values, failures


def r_summary(rows: list[dict[str, Any]], threshold_contract: dict[str, Any]) -> dict[str, Any]:
    records, failures = r_spatial_records(rows)
    values = [record["R_spatial"] for record in records]
    return {
        "R_spatial_distribution": distribution(values),
        "block_count": len(records),
        "failures": failures,
        "threshold_law": threshold_contract["matched_permutation_null"],
    }


def matched_permutation_null(packet: dict[str, Any], rows: list[dict[str, Any]], threshold_contract: dict[str, Any]) -> dict[str, Any]:
    grouped_pairs = pair_groups(packet)
    relation_blocks = blocks([row for row in rows if row["row_role"] == "relation_matrix"])
    rng = random.Random(pub.MATCHED_PERMUTATION_SEED)
    null_values: list[float] = []
    permutation_count = int(threshold_contract["matched_permutation_null"].get("permutation_count", pub.MATCHED_PERMUTATION_COUNT))
    for permutation_index in range(permutation_count):
        block_values = []
        for block in relation_blocks.values():
            c_by_cell: dict[str, float] = {}
            for row in block:
                pairs = grouped_pairs[row["tuple_id"]]
                a = [float(pair["A_first_touch_cycles"]) for pair in pairs]
                b = [float(pair["B_first_touch_cycles"]) for pair in pairs]
                order = list(range(len(b)))
                rng.shuffle(order)
                shuffled = [b[i] for i in order]
                c_by_cell[row["relation_cell"]] = spearman(a, shuffled)
            block_values.append(
                0.5
                * (
                    c_by_cell["prepare_r0__query_r0"]
                    + c_by_cell["prepare_r1__query_r1"]
                    - c_by_cell["prepare_r0__query_r1"]
                    - c_by_cell["prepare_r1__query_r0"]
                )
            )
        null_values.append(abs(mean(block_values)))
    q99 = quantile(null_values, 0.99)
    observed = r_summary(rows, threshold_contract)["R_spatial_distribution"]["abs_of_mean"]
    return {
        "passed": observed > q99,
        "observed_abs_mean_R_spatial": observed,
        "q99_abs_mean_R_spatial": q99,
        "null_distribution": distribution(null_values),
        "permutation_count": pub.MATCHED_PERMUTATION_COUNT,
        "seed": pub.MATCHED_PERMUTATION_SEED,
    }


def factor_stability(r_records: list[dict[str, Any]], factors: list[str]) -> dict[str, Any]:
    global_sign = 1 if mean([row["R_spatial"] for row in r_records]) > 0 else -1 if mean([row["R_spatial"] for row in r_records]) < 0 else 0
    reports = {}
    for factor in factors:
        levels: dict[str, list[float]] = defaultdict(list)
        for row in r_records:
            levels[str(row[factor])].append(row["R_spatial"])
        reports[factor] = {
            "passed": global_sign != 0 and all((1 if mean(vals) > 0 else -1 if mean(vals) < 0 else 0) == global_sign for vals in levels.values()),
            "levels": {level: distribution(vals) for level, vals in sorted(levels.items())},
        }
    return {"passed": all(item["passed"] for item in reports.values()), "global_sign": global_sign, "factors": reports}


def control_results(rows: list[dict[str, Any]], null_q99: float) -> dict[str, Any]:
    by_control: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row["row_role"] == "relation_control":
            by_control[row["query"]].append(row["C_pair_recomputed"])
    controls = {name: {**distribution(values), "passed": distribution(values)["abs_of_mean"] <= null_q99} for name, values in by_control.items()}
    required = ["relation_sham", "scrambled_pair_control"]
    return {
        "passed": all(controls.get(name, {}).get("passed") is True for name in required),
        "null_control_support_max_abs": null_q99,
        "controls": controls,
    }


def marginal_explanation(rows: list[dict[str, Any]], null_q99: float) -> dict[str, Any]:
    r_records, _ = r_spatial_records(rows)
    values = [record["R_spatial"] for record in r_records]
    a_means = []
    b_means = []
    for block in blocks(rows).values():
        relation_rows = [row for row in block if row["row_role"] == "relation_matrix"]
        a_means.append(mean([row["A_latency_mean"] for row in relation_rows if row["relation_match"] is True]) - mean([row["A_latency_mean"] for row in relation_rows if row["relation_match"] is False]))
        b_means.append(mean([row["B_latency_mean"] for row in relation_rows if row["relation_match"] is True]) - mean([row["B_latency_mean"] for row in relation_rows if row["relation_match"] is False]))
    a_corr = abs(pearson(values, a_means))
    b_corr = abs(pearson(values, b_means))
    return {
        "passed": a_corr < 0.5 and b_corr < 0.5,
        "A_marginal_R_correlation_abs": a_corr,
        "B_marginal_R_correlation_abs": b_corr,
        "null_q99_reference": null_q99,
    }


def measurement_order_stability(packet: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped = pair_groups(packet)
    order_reports = {}
    for measurement_order in ["A_first", "B_first"]:
        order_packet = {"raw_records": packet["raw_records"], "pair_observations": []}
        for row in packet["raw_records"]:
            selected = [pair for pair in grouped[row["tuple_id"]] if pair["measurement_order"] == measurement_order]
            if selected:
                order_packet["pair_observations"].extend(selected)
        order_rows = row_c_pairs(order_packet)
        records, failures = r_spatial_records(order_rows)
        order_reports[measurement_order] = {
            "failures": failures,
            "R_spatial_distribution": distribution([record["R_spatial"] for record in records]),
        }
    signs = [
        1 if item["R_spatial_distribution"]["mean"] > 0 else -1 if item["R_spatial_distribution"]["mean"] < 0 else 0
        for item in order_reports.values()
    ]
    return {"passed": signs[0] != 0 and signs[0] == signs[1], "orders": order_reports}



def packet_for_raw_rows(packet: dict[str, Any], raw_rows: list[dict[str, Any]]) -> dict[str, Any]:
    tuple_ids = {str(row["tuple_id"]) for row in raw_rows}
    return {
        "raw_records": raw_rows,
        "pair_observations": [pair for pair in packet["pair_observations"] if str(pair.get("tuple_id")) in tuple_ids],
        "source_death_receipts": [death for death in packet.get("source_death_receipts", []) if str(death.get("tuple_id")) in tuple_ids],
    }


def query_segment_reports(packet: dict[str, Any], threshold_contract: dict[str, Any]) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    queries = [pub.RELATION_PRIMARY_QUERY, *pub.SEGMENTED_CONTROL_QUERIES]
    for query in queries:
        raw_rows = [row for row in packet["raw_records"] if row.get("row_role") == "relation_matrix" and row.get("query") == query]
        segment_packet = packet_for_raw_rows(packet, raw_rows)
        rows = row_c_pairs(segment_packet) if raw_rows else []
        records, failures = r_spatial_records(rows) if rows else ([], [f"segment missing: {query}"])
        values = [record["R_spatial"] for record in records]
        null = matched_permutation_null(segment_packet, rows, threshold_contract) if rows else {
            "passed": False,
            "observed_abs_mean_R_spatial": 0.0,
            "q99_abs_mean_R_spatial": 0.0,
            "null_distribution": distribution([]),
            "permutation_count": 0,
            "seed": pub.MATCHED_PERMUTATION_SEED,
        }
        reports[query] = {
            "query": query,
            "raw_record_count": len(raw_rows),
            "pair_observation_count": len(segment_packet["pair_observations"]),
            "R_spatial_distribution": distribution(values),
            "block_count": len(records),
            "failures": failures,
            "matched_permutation_null": null,
            "factor_stability": factor_stability(records, ["session", "mapping", "source_order", "query_order", "cyclic_origin"]) if records else {"passed": False, "global_sign": 0, "factors": {}},
            "measurement_order_stability": measurement_order_stability(segment_packet, rows) if rows else {"passed": False, "orders": {}},
            "marginal_explanation": marginal_explanation(rows, null.get("q99_abs_mean_R_spatial", 0.0)) if rows else {"passed": False},
        }
    return reports


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
        return fail_closed(RESULT_INVALID, validation)
    rows = row_c_pairs(packet)
    r_records, relation_failures = r_spatial_records(rows)
    global_summary = r_summary(rows, threshold_contract)
    segment_reports = query_segment_reports(packet, threshold_contract)
    primary = segment_reports.get(pub.RELATION_PRIMARY_QUERY, {})
    controls = {query: segment_reports.get(query, {}) for query in pub.SEGMENTED_CONTROL_QUERIES}
    control_abs = {
        query: report.get("R_spatial_distribution", {}).get("abs_of_mean", 0.0)
        for query, report in controls.items()
    }
    primary_abs = primary.get("R_spatial_distribution", {}).get("abs_of_mean", 0.0)
    max_control_abs = max(control_abs.values()) if control_abs else 0.0
    primary_control_separation = (primary_abs / max_control_abs) if max_control_abs > 0 else None
    controls_inside_null = all(
        report.get("R_spatial_distribution", {}).get("abs_of_mean", 0.0)
        <= report.get("matched_permutation_null", {}).get("q99_abs_mean_R_spatial", 0.0)
        for report in controls.values()
    )
    primary_complete = primary.get("block_count") == schedule.get("base_condition_count", 512)
    controls_complete = all(report.get("block_count") == schedule.get("base_condition_count", 512) for report in controls.values())
    gates = {
        "validation": validation["passed"],
        "complete_relation_matrix_global": not relation_failures and len(r_records) == schedule["tuple_count"] // 4,
        "primary_segment_complete": primary_complete,
        "control_segments_complete": controls_complete,
        "primary_matched_permutation_q99_exceeded": primary.get("matched_permutation_null", {}).get("passed") is True,
        "primary_factor_sign_stability": primary.get("factor_stability", {}).get("passed") is True,
        "R_shaped_control_segments_inside_own_null": controls_inside_null,
        "primary_control_separation_4x": primary_control_separation is not None and primary_control_separation >= 4.0,
        "primary_not_explained_by_A_or_B_marginal_latency_alone": primary.get("marginal_explanation", {}).get("passed") is True,
    }
    passed = all(gates.values())
    candidate = (not passed) and gates["primary_segment_complete"] and primary_abs > 0.5 * primary.get("matched_permutation_null", {}).get("q99_abs_mean_R_spatial", 0.0)
    result_class = RESULT_CONFIRMED if passed else RESULT_CANDIDATE if candidate else RESULT_NOT_OBSERVED
    claim_positive = passed
    report = {
        "schema": "FAMILY10H_RELATION_SPATIAL_PHYSICAL_ADJUDICATION_V1",
        "result_class": result_class,
        "passed": passed,
        "candidate": candidate,
        "scientific_claim": POSITIVE_CLAIM if claim_positive else NEGATIVE_CLAIM,
        "claim_boundary": claim_boundary(claim_positive),
        "validation": validation,
        "threshold_contract": threshold_contract,
        "R_spatial_global": global_summary,
        "query_segment_reports": segment_reports,
        "primary_query": pub.RELATION_PRIMARY_QUERY,
        "control_queries": pub.SEGMENTED_CONTROL_QUERIES,
        "primary_control_separation": {
            "primary_abs_mean_R_spatial": primary_abs,
            "control_abs_mean_R_spatial": control_abs,
            "max_control_abs_mean_R_spatial": max_control_abs,
            "primary_to_max_control_ratio": primary_control_separation,
            "required_min_ratio": 4.0,
        },
        "gates": gates,
    }
    report["adjudication_sha256"] = pub.digest({k: v for k, v in report.items() if k != "adjudication_sha256"})
    return report


def fixture_packet(schedule: dict[str, Any], mode: str) -> dict[str, Any]:
    raw_rows = []
    pair_rows = []
    death_rows = []
    for row in schedule["rows"]:
        observed_row = {
            key: row[key]
            for key in [
                "tuple_id",
                "execution_ordinal",
                "block_id",
                "block_local_position",
                "row_role",
                "q",
                "r_prepare",
                "r_query",
                "relation_match",
                "query",
                "relation_cell",
                "session",
                "replicate",
                "mapping",
                "source_order",
                "query_order",
                "cyclic_origin",
                "source_cpu_expected",
                "receiver_cpu_expected",
            ]
        }
        observed_row.update(
            {
                "source_lifetime": "alive_during_query",
                "pair_measurement_count": pub.PAIR_SAMPLE_COUNT,
                "C_pair": 0.0,
                "timer_method": "rdtscp_lfence_serialized",
                "change_to_dirty": 1,
                "dirty_probe_response": 1,
                "cpu_cycles": 1000,
                "duration_ns": 1000,
                "source_alive_at_pair_measurement": mode != "source_dies",
                "physical_measurement": False,
                "positive_physical_claim": False,
            }
        )
        raw_rows.append(observed_row)
        for sample_index, a_index in enumerate(pub.selected_pair_indices(row["execution_ordinal"])):
            b_index = (a_index + (1 if row["r_query"] == "relation_r0" else -1)) % pub.LINE_COUNT
            a_cycle = 100 + ((a_index * 13 + sample_index * 7) & 31)
            if mode == "positive" and row["row_role"] == "relation_matrix" and row.get("query") == pub.RELATION_PRIMARY_QUERY:
                b_cycle = a_cycle + (sample_index & 3) if row["relation_match"] else 190 - (a_cycle & 63)
            elif mode == "A_marginal_only":
                b_cycle = 140 + ((b_index * 5 + sample_index) & 31)
                a_cycle += 40 if row["relation_match"] else 0
            elif mode == "B_marginal_only":
                b_cycle = 180 + ((b_index * 5 + sample_index) & 31) if row["relation_match"] else 140 + ((b_index * 5 + sample_index) & 31)
            elif mode == "common_mode":
                b_cycle = a_cycle + 20
            else:
                b_cycle = 140 + ((b_index * 5 + sample_index * 11 + row["execution_ordinal"]) & 31)
            pair_rows.append(
                {
                    "tuple_id": row["tuple_id"],
                    "execution_ordinal": row["execution_ordinal"],
                    "block_id": row["block_id"],
                    "block_local_position": row["block_local_position"],
                    "row_role": row["row_role"],
                    "relation_cell": row["relation_cell"],
                    "relation_match": row["relation_match"],
                    "query": row["query"],
                    "sample_index": sample_index,
                    "pair_index": a_index,
                    "A_line_index": a_index,
                    "B_line_index": b_index,
                    "measurement_order": "A_first" if sample_index % 2 == 0 else "B_first",
                    "A_first_touch_cycles": a_cycle,
                    "B_first_touch_cycles": b_cycle,
                    "source_alive_at_pair_measurement": mode != "source_dies",
                    "source_cpu_expected": 4,
                    "receiver_cpu_expected": 5,
                    "timer_method": "rdtscp_lfence_serialized",
                    "physical_measurement": False,
                    "positive_physical_claim": False,
                }
            )
        death_rows.append(
            {
                "tuple_id": row["tuple_id"],
                "execution_ordinal": row["execution_ordinal"],
                "source_lifetime": "alive_during_query",
                "source_alive_at_pair_measurement": mode != "source_dies",
                "source_alive_during_query": mode != "source_dies",
                "post_observation_query_or_window_selection": False,
                "process_custody": "source_alive_during_spatial_pair_probe",
                "physical_measurement": False,
            }
        )
    packet = {"raw_records": raw_rows, "pair_observations": pair_rows, "source_death_receipts": death_rows}
    packet.update(fixture_archive_packet_material(packet, schedule, mode))
    return packet


def fixture_archive_packet_material(packet: dict[str, Any], schedule: dict[str, Any], mode: str) -> dict[str, Any]:
    manifest = {
        "schema": "FAMILY10H_RELATION_SPATIAL_IMPLEMENTATION_MANIFEST_V1",
        "package_decision": pub.PACKAGE_DECISION_BUILD_READY,
        "schedule_sha256": schedule["schedule_sha256"],
        "authority_binding": {"relation_source_authority_commit": FIXTURE_SOURCE_SHA},
    }
    deployment = {"relation_manifest_freeze_commit": FIXTURE_FREEZE_SHA}
    feature = {
        "schema": "FAMILY10H_RELATION_SPATIAL_FEATURE_FREEZE_V1",
        "pair_observation_count": len(packet["pair_observations"]),
        "small_wall_crossed": False,
    }
    receipt = {
        "schema": "FAMILY10H_RELATION_SPATIAL_TARGET_EXECUTION_RECEIPT_V1",
        "raw_record_count": len(packet["raw_records"]),
        "pair_observation_count": len(packet["pair_observations"]),
        "source_death_receipt_count": len(packet["source_death_receipts"]),
        "small_wall_crossed": False,
    }
    archive_members = {
        "attempt/raw_records.jsonl": jsonl_bytes(packet["raw_records"]),
        "attempt/pair_observations.jsonl": jsonl_bytes(packet["pair_observations"]),
        "attempt/source_death_receipts.jsonl": jsonl_bytes(packet["source_death_receipts"]),
        "attempt/feature_freeze.json": strict_json_bytes(feature),
        "attempt/target_execution_receipt.json": strict_json_bytes(receipt),
        "attempt/RELATION_SPATIAL_IMPLEMENTATION_MANIFEST.json": strict_json_bytes(manifest),
        "attempt/relation_spatial_runtime": b"fixture-runtime",
        f"attempt/{pub.DEPLOYMENT_CUSTODY_FILENAME}": strict_json_bytes(deployment),
    }
    archive_io = io.BytesIO()
    with tarfile.open(fileobj=archive_io, mode="w") as tf:
        for name, data in sorted(archive_members.items()):
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            info.mtime = 0
            tf.addfile(info, io.BytesIO(data))
    return {"custody_envelope": custody_envelope_from_archive_bytes(archive_io.getvalue(), embed_archive_bytes=True)}


def fail_closed_claim_state(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "result_class_not_confirmed": report.get("result_class") != RESULT_CONFIRMED,
        "positive_scientific_claim_absent": report.get("scientific_claim") != POSITIVE_CLAIM,
        "positive_claim_boundary_absent": report.get("claim_boundary", {}).get("spatial_relation_pair_readout_calibrated") is not True,
        "passed": report.get("result_class") != RESULT_CONFIRMED
        and report.get("scientific_claim") != POSITIVE_CLAIM
        and report.get("claim_boundary", {}).get("spatial_relation_pair_readout_calibrated") is not True,
    }


def run_self_test(schedule: dict[str, Any]) -> dict[str, Any]:
    threshold = physical_threshold_contract()
    threshold["matched_permutation_null"] = {**threshold["matched_permutation_null"], "permutation_count": 15, "self_test_reduced_permutation_count": True}
    fixture_rows = []
    for query in [pub.RELATION_PRIMARY_QUERY, *pub.SEGMENTED_CONTROL_QUERIES]:
        fixture_rows.extend([row for row in schedule["rows"] if row.get("query") == query][:32])
    fixture_schedule = {**schedule, "rows": fixture_rows, "tuple_count": len(fixture_rows), "base_condition_count": 8, "expected_pair_observation_count": len(fixture_rows) * pub.PAIR_SAMPLE_COUNT}
    positive = adjudicate_physical_packet(fixture_packet(fixture_schedule, "positive"), fixture_schedule, threshold)
    negative_modes = {
        "equal_A_B_marginals_no_pair_coupling": "none",
        "common_mode_latency_increase_no_pair_coupling": "common_mode",
        "A_marginal_only_effect": "A_marginal_only",
        "B_marginal_only_effect": "B_marginal_only",
        "measurement_order_artifact": "none",
        "cyclic_origin_only_artifact": "none",
        "mapping_only_artifact": "none",
        "scrambled_pair_artifact": "none",
    }
    negatives = {label: adjudicate_physical_packet(fixture_packet(fixture_schedule, mode), fixture_schedule, threshold, require_custody=False) for label, mode in negative_modes.items()}
    source_dies = adjudicate_physical_packet(fixture_packet(fixture_schedule, "source_dies"), fixture_schedule, threshold, require_custody=False)
    malformed = fixture_packet(fixture_schedule, "positive")
    malformed["pair_observations"] = malformed["pair_observations"][:-1]
    malformed_result = adjudicate_physical_packet(malformed, fixture_schedule, threshold, require_custody=False)
    primary_positive = positive["query_segment_reports"][pub.RELATION_PRIMARY_QUERY]
    checks = {
        "positive_fixture_calibrated": positive["result_class"] == RESULT_CONFIRMED and positive["scientific_claim"] == POSITIVE_CLAIM,
        "negative_fixtures_fail_closed": all(fail_closed_claim_state(result)["passed"] for result in negatives.values()),
        "source_lifecycle_failure_invalid": source_dies["result_class"] == RESULT_INVALID and fail_closed_claim_state(source_dies)["passed"],
        "malformed_pair_inventory_invalid": malformed_result["result_class"] == RESULT_INVALID and fail_closed_claim_state(malformed_result)["passed"],
        "matched_permutation_threshold_applied": primary_positive["matched_permutation_null"]["q99_abs_mean_R_spatial"] > 0.0,
    }
    result = {
        "schema": "FAMILY10H_RELATION_SPATIAL_PHYSICAL_ADJUDICATOR_SELF_TEST_V1",
        "threshold_contract": threshold,
        "checks": checks,
        "positive_result": {
            "result_class": positive["result_class"],
            "scientific_claim": positive["scientific_claim"],
            "R_spatial_abs_of_mean": primary_positive["R_spatial_distribution"]["abs_of_mean"],
            "matched_permutation_q99": primary_positive["matched_permutation_null"]["q99_abs_mean_R_spatial"],
            "primary_control_separation": positive["primary_control_separation"],
        },
        "negative_results": {label: result["result_class"] for label, result in negatives.items()},
        "source_dies_result": source_dies["result_class"],
        "malformed_pair_inventory_result": malformed_result["result_class"],
        "passed": all(checks.values()),
    }
    result["self_test_sha256"] = pub.digest({k: v for k, v in result.items() if k != "self_test_sha256"})
    return result
