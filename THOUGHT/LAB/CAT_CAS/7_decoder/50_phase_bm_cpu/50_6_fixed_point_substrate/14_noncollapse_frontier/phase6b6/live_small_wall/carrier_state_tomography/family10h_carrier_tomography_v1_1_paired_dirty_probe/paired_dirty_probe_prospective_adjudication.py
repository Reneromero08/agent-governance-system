#!/usr/bin/env python3
"""Prospective paired dirty-probe adjudicator for Family 10h v1.1.

This adjudicates only the one-dimensional public scalar q-readout confirmation.
It does not promote SMALL_WALL_CROSSED, catalytic borrowing, physical relational
memory, a relational carrier, or full carrier-state tomography.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import io
import json
import math
import statistics
import sys
import tarfile
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
CARRIER_ROOT = HERE.parent
PACKAGE_ROOT = HERE
RUN_ID = "family10h_carrier_tomography_v1_1_paired_dirty_probe_0"
DEFAULT_ATTEMPT_DIR = CARRIER_ROOT / "runs" / RUN_ID / "attempt_1"

RESULT_SUPPORTED = "FAMILY10H_PAIRED_DIRTY_PROBE_Q_READOUT_CONFIRMED_PROSPECTIVE"
RESULT_NOT_SUPPORTED = "FAMILY10H_PAIRED_DIRTY_PROBE_Q_READOUT_NOT_CONFIRMED_PROSPECTIVE"
RESULT_CANDIDATE = "FAMILY10H_PAIRED_DIRTY_PROBE_Q_READOUT_CANDIDATE_PROSPECTIVE"
RESULT_INVALID = "FAMILY10H_PAIRED_DIRTY_PROBE_Q_READOUT_CUSTODY_INVALID"
POSITIVE_SCIENTIFIC_CLAIM = "PUBLIC_POST_SOURCE_SCALAR_CARRIER_Q_READOUT_CONFIRMED"
NEGATIVE_SCIENTIFIC_CLAIM = "PUBLIC_POST_SOURCE_SCALAR_CARRIER_Q_READOUT_NOT_ESTABLISHED_PROSPECTIVE"
PROSPECTIVE_CLAIM_CEILING = POSITIVE_SCIENTIFIC_CLAIM

PAIR_SOURCE_OFF_MAX_ABS = 128.0
PAIR_SOURCE_OFF_MEAN_ABS = 32.0
PAIR_Q0_MEAN_MAX_ABS = 64.0
PAIR_Q1536_MIN_ABS = 2000.0
PAIR_MODEL_MIN_R2 = 0.98
HELDOUT_MAX_REL_RMSE = 0.10
SLOPE_MAX_REL_DISAGREEMENT = 0.10
PAIR_INTERCEPT_MAX_ABS = 64.0
ODD_SYMMETRY_MAX_RELATIVE_ERROR = 0.10
SIGNAL_TO_NULL_MIN_RATIO = 20.0
CLASSIFIER_MIN_EXACT_ACCURACY = 0.95
CLASSIFIER_MIN_SIGN_ACCURACY = 1.0

ACTIVE_QS = [-1536, -1024, -512, 0, 512, 1024, 1536]
SIGN_QS = [-1536, -1024, -512, 512, 1024, 1536]
MAG_CHAIN_POS = [(512, 1024), (1024, 1536)]
MAG_CHAIN_NEG = [(-512, -1024), (-1024, -1536)]
REQUIRED_ARCHIVE_MEMBERS = {
    "raw_records.jsonl": "output/raw_records.jsonl",
    "source_death_receipts.jsonl": "output/source_death_receipts.jsonl",
    "feature_freeze.json": "output/feature_freeze.json",
    "output_target_execution_receipt.json": "output_target_execution_receipt.json",
}
THRESHOLD_PROVENANCE = {
    "attempt_3_status": "retrospective_thresholds_selected_after_attempt_3_inspection",
    "attempt_3_limitation": "attempt_3_cannot_independently_validate_thresholds_derived_after_examining_attempt_3",
    "v1_1_status": "thresholds_are_prospectively_frozen_only_for_the_proposed_v1_1_confirmation",
    "post_v1_1_revision": "no_post_v1_1_run_threshold_revision_allowed",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_public_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "family10h_carrier_tomography_public",
        PACKAGE_ROOT / "family10h_carrier_tomography_public.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen public module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def invalid_adjudication_result(validation: dict[str, Any], provenance: dict[str, Any] | None = None) -> dict[str, Any]:
    result = {
        "schema": "FAMILY10H_PAIRED_DIRTY_PROBE_PROSPECTIVE_ADJUDICATION_V1",
        "result_class": RESULT_INVALID,
        "passed": False,
        "scientific_claim": NEGATIVE_SCIENTIFIC_CLAIM,
        "prospective_claim_ceiling": PROSPECTIVE_CLAIM_CEILING,
        "claim_boundary": {
            "reproducible_public_q_dependent_response_observed": False,
            "dimension": "one_dimensional_scalar_codeword_readout",
            "full_carrier_state_tomography_established": False,
            "v1_0_retained_evidence_unchanged": True,
            "small_wall_promoted": False,
            "catalytic_borrowing_established": False,
            "physical_relational_memory_established": False,
            "relational_carrier_established": False,
        },
        "validation": validation,
        "official_result_replaced": False,
        "small_wall_promoted": False,
    }
    if provenance is not None:
        result["source_evidence"] = provenance
    return result


def parse_jsonl_bytes(data: bytes, name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, raw_line in enumerate(data.decode("utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"{name}:{line_no}: {exc}") from exc
    return rows


def load_json_path(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_archive_member_name(name: str, archive_run_id: str = RUN_ID) -> str:
    normalized = name.replace("\\", "/").lstrip("/")
    prefix = f"{archive_run_id}/"
    if normalized.startswith(prefix):
        normalized = normalized[len(prefix):]
    return normalized


def receipt_snapshot_index(copyback_receipt: dict[str, Any]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for item in copyback_receipt.get("snapshot_files", []):
        relative_path = item.get("relative_path")
        if relative_path:
            index[str(relative_path).replace("\\", "/")] = item
    return index


def verify_archive_receipts(
    attempt_dir: Path,
    tar_path: Path,
    copyback_receipt: dict[str, Any],
    inventory: dict[str, Any],
) -> dict[str, Any]:
    actual_sha = sha256_path(tar_path)
    actual_size = tar_path.stat().st_size
    expected_sha = copyback_receipt.get("local_archive_sha256")
    expected_size = copyback_receipt.get("local_archive_size")
    if expected_sha != actual_sha:
        raise RuntimeError(f"archive sha mismatch: expected {expected_sha}, observed {actual_sha}")
    if int(expected_size) != actual_size:
        raise RuntimeError(f"archive size mismatch: expected {expected_size}, observed {actual_size}")
    receipt_archive_path = Path(str(copyback_receipt.get("local_archive", ""))).resolve()
    if receipt_archive_path != tar_path.resolve():
        raise RuntimeError(f"copy-back receipt local_archive mismatch: {receipt_archive_path} != {tar_path.resolve()}")
    remote_root = inventory.get("remote_root_archive", {})
    if remote_root.get("relative_path") != tar_path.name:
        raise RuntimeError("evidence inventory remote archive path mismatch")
    if remote_root.get("sha256") != actual_sha:
        raise RuntimeError("evidence inventory remote archive sha mismatch")
    if int(remote_root.get("size", -1)) != actual_size:
        raise RuntimeError("evidence inventory remote archive size mismatch")
    files = {
        item.get("relative_path"): item
        for item in inventory.get("files", [])
    }
    inventory_archive = files.get(tar_path.name)
    if not inventory_archive:
        raise RuntimeError("evidence inventory missing archive file entry")
    if inventory_archive.get("sha256") != actual_sha:
        raise RuntimeError("evidence inventory archive file sha mismatch")
    if int(inventory_archive.get("size", -1)) != actual_size:
        raise RuntimeError("evidence inventory archive file size mismatch")
    return {
        "copyback_receipt": str(attempt_dir / f"ATTEMPT_{attempt_dir.name.split('_')[-1]}_COPYBACK_RECEIPT.json"),
        "copyback_receipt_sha256": sha256_path(attempt_dir / f"ATTEMPT_{attempt_dir.name.split('_')[-1]}_COPYBACK_RECEIPT.json"),
        "evidence_inventory": str(attempt_dir / f"ATTEMPT_{attempt_dir.name.split('_')[-1]}_EVIDENCE_INVENTORY.json"),
        "evidence_inventory_sha256": sha256_path(attempt_dir / f"ATTEMPT_{attempt_dir.name.split('_')[-1]}_EVIDENCE_INVENTORY.json"),
        "archive": str(tar_path),
        "archive_sha256": actual_sha,
        "archive_size": actual_size,
    }


def read_output_bytes_from_verified_archive(
    tar_path: Path,
    copyback_receipt: dict[str, Any],
    archive_run_id: str = RUN_ID,
) -> tuple[dict[str, bytes], dict[str, Any]]:
    matches: dict[str, list[tarfile.TarInfo]] = defaultdict(list)
    payload: dict[str, bytes] = {}
    with tarfile.open(tar_path, "r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            normalized = normalize_archive_member_name(member.name, archive_run_id=archive_run_id)
            for key, relative_path in REQUIRED_ARCHIVE_MEMBERS.items():
                if normalized == relative_path:
                    matches[key].append(member)
    missing = sorted(set(REQUIRED_ARCHIVE_MEMBERS) - set(matches))
    if missing:
        raise RuntimeError(f"remote-root archive missing output files: {missing}")
    duplicates = {
        key: [member.name for member in members]
        for key, members in matches.items()
        if len(members) != 1
    }
    if duplicates:
        raise RuntimeError(f"remote-root archive duplicate output file matches: {duplicates}")
    snapshot_index = receipt_snapshot_index(copyback_receipt)
    member_provenance: dict[str, Any] = {}
    with tarfile.open(tar_path, "r:gz") as archive:
        for key, members in matches.items():
            member = members[0]
            extracted = archive.extractfile(member)
            if extracted is None:
                raise RuntimeError(f"cannot extract {member.name}")
            data = extracted.read()
            relative_path = REQUIRED_ARCHIVE_MEMBERS[key]
            expected = snapshot_index.get(relative_path)
            if expected is None:
                raise RuntimeError(f"copy-back receipt missing snapshot hash for {relative_path}")
            observed_sha = sha256_bytes(data)
            observed_size = len(data)
            if expected.get("sha256") != observed_sha:
                raise RuntimeError(f"archive member sha mismatch for {relative_path}")
            if int(expected.get("size", -1)) != observed_size:
                raise RuntimeError(f"archive member size mismatch for {relative_path}")
            payload[key] = data
            member_provenance[key] = {
                "archive_member": member.name,
                "relative_path": relative_path,
                "sha256": observed_sha,
                "size": observed_size,
                "copyback_snapshot_bound": True,
            }
    return payload, member_provenance


def snapshot_payload_if_equal(attempt_dir: Path, archive_payload: dict[str, bytes], archive_run_id: str = RUN_ID) -> dict[str, Any]:
    snapshot_root = attempt_dir / "remote_root_snapshot" / archive_run_id
    snapshot_files = {
        "raw_records.jsonl": snapshot_root / "output" / "raw_records.jsonl",
        "source_death_receipts.jsonl": snapshot_root / "output" / "source_death_receipts.jsonl",
        "feature_freeze.json": snapshot_root / "output" / "feature_freeze.json",
        "output_target_execution_receipt.json": snapshot_root / "output_target_execution_receipt.json",
    }
    if not all(path.exists() for path in snapshot_files.values()):
        return {
            "present": False,
            "authoritative": False,
            "status": "snapshot_cache_absent_or_incomplete",
        }
    comparisons = {}
    for key, path in snapshot_files.items():
        snapshot_bytes = path.read_bytes()
        archive_bytes = archive_payload[key]
        comparisons[key] = {
            "snapshot_path": str(path),
            "snapshot_sha256": sha256_bytes(snapshot_bytes),
            "archive_sha256": sha256_bytes(archive_bytes),
            "snapshot_size": len(snapshot_bytes),
            "archive_size": len(archive_bytes),
            "matches_archive": snapshot_bytes == archive_bytes,
        }
    mismatches = [key for key, item in comparisons.items() if not item["matches_archive"]]
    if mismatches:
        raise RuntimeError(f"snapshot cache differs from verified archive members: {mismatches}")
    return {
        "present": True,
        "authoritative": False,
        "status": "snapshot_cache_verified_equal_to_archive_but_not_used_as_authority",
        "comparisons": comparisons,
    }


def read_attempt_output(attempt_dir: Path, archive_run_id: str = RUN_ID) -> tuple[dict[str, bytes], dict[str, Any]]:
    attempt_number = attempt_dir.name.split("_")[-1]
    tar_path = attempt_dir / f"ATTEMPT_{attempt_number}_REMOTE_ROOT.tar.gz"
    copyback_path = attempt_dir / f"ATTEMPT_{attempt_number}_COPYBACK_RECEIPT.json"
    inventory_path = attempt_dir / f"ATTEMPT_{attempt_number}_EVIDENCE_INVENTORY.json"
    copyback_receipt = load_json_path(copyback_path)
    inventory = load_json_path(inventory_path)
    provenance = verify_archive_receipts(attempt_dir, tar_path, copyback_receipt, inventory)
    payload, member_provenance = read_output_bytes_from_verified_archive(tar_path, copyback_receipt, archive_run_id=archive_run_id)
    provenance.update({
        "source": "verified_committed_remote_root_archive",
        "authoritative_input": tar_path.name,
        "archive_run_id": archive_run_id,
        "archive_member_hashes": member_provenance,
        "snapshot_cache": snapshot_payload_if_equal(attempt_dir, payload, archive_run_id=archive_run_id),
        "output_hashes": {
            key: {"sha256": sha256_bytes(value), "size": len(value)}
            for key, value in payload.items()
        },
    })
    return payload, provenance


def build_packet(public: Any, attempt_dir: Path, archive_run_id: str = RUN_ID) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    schedule = public.load_schedule_from_artifacts()
    payload, provenance = read_attempt_output(attempt_dir, archive_run_id=archive_run_id)
    packet = {
        "schema": "FAMILY10H_CARRIER_TOMOGRAPHY_EVIDENCE_PACKET_V1",
        "schedule_sha256": public.digest(schedule),
        "raw_records": parse_jsonl_bytes(payload["raw_records.jsonl"], "raw_records.jsonl"),
        "source_death_receipts": parse_jsonl_bytes(payload["source_death_receipts.jsonl"], "source_death_receipts.jsonl"),
        "feature_freeze": json.loads(payload["feature_freeze.json"].decode("utf-8")),
    }
    target_receipt = json.loads(payload["output_target_execution_receipt.json"].decode("utf-8"))
    provenance["target_status"] = target_receipt.get("status")
    provenance["target_returncode"] = target_receipt.get("returncode")
    provenance["target_evidence_validation"] = target_receipt.get("evidence_validation")
    return packet, schedule, provenance


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else float("nan")


def rel_disagreement(values: list[float]) -> float:
    abs_values = [abs(value) for value in values if math.isfinite(value)]
    if not abs_values:
        return float("inf")
    max_abs = max(abs_values)
    if max_abs == 0.0:
        return 0.0
    return (max_abs - min(abs_values)) / max_abs


def slope_through_origin(samples: list[dict[str, Any]]) -> float:
    denom = sum(float(item["q"]) * float(item["q"]) for item in samples if int(item["q"]) != 0)
    if denom == 0.0:
        return float("nan")
    return sum(float(item["q"]) * float(item["y"]) for item in samples if int(item["q"]) != 0) / denom


def rmse(samples: list[dict[str, Any]], slope: float) -> float:
    if not samples:
        return float("nan")
    return math.sqrt(mean([(float(item["y"]) - slope * float(item["q"])) ** 2 for item in samples]))


def r2_score(samples: list[dict[str, Any]], slope: float) -> float:
    if not samples:
        return float("nan")
    ys = [float(item["y"]) for item in samples]
    ybar = mean(ys)
    sse = sum((float(item["y"]) - slope * float(item["q"])) ** 2 for item in samples)
    sst = sum((value - ybar) ** 2 for value in ys)
    return 1.0 - sse / sst if sst else float("nan")


def ols_report(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        return {
            "count": 0,
            "slope": float("nan"),
            "intercept": float("nan"),
            "rmse": float("nan"),
            "r2": float("nan"),
            "intercept_abs_pass": False,
        }
    xs = [float(item["q"]) for item in samples]
    ys = [float(item["y"]) for item in samples]
    xbar = mean(xs)
    ybar = mean(ys)
    denom = sum((x - xbar) ** 2 for x in xs)
    slope = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys)) / denom if denom else float("nan")
    intercept = ybar - slope * xbar
    residuals = [y - (slope * x + intercept) for x, y in zip(xs, ys)]
    rmse_value = math.sqrt(mean([value * value for value in residuals]))
    sst = sum((value - ybar) ** 2 for value in ys)
    sse = sum(value * value for value in residuals)
    return {
        "count": len(samples),
        "slope": slope,
        "intercept": intercept,
        "rmse": rmse_value,
        "r2": 1.0 - sse / sst if sst else float("nan"),
        "intercept_abs_pass": abs(intercept) <= PAIR_INTERCEPT_MAX_ABS,
    }


def group_by(samples: list[dict[str, Any]], key: str) -> dict[Any, list[dict[str, Any]]]:
    groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for item in samples:
        groups[item[key]].append(item)
    return dict(groups)


def q_means(samples: list[dict[str, Any]]) -> dict[int, float]:
    by_q: dict[int, list[float]] = defaultdict(list)
    for item in samples:
        by_q[int(item["q"])].append(float(item["y"]))
    return {q: mean(values) for q, values in sorted(by_q.items())}


def pass_sign_and_magnitude(qmean: dict[int, float]) -> dict[str, Any]:
    missing = [q for q in ACTIVE_QS if q not in qmean]
    sign_failures = [q for q in SIGN_QS if q in qmean and not (qmean[q] * q > 0.0)]
    magnitude_failures: list[str] = []
    for a, b in MAG_CHAIN_POS + MAG_CHAIN_NEG:
        if a in qmean and b in qmean and not (abs(qmean[a]) < abs(qmean[b])):
            magnitude_failures.append(f"abs(q={a}) < abs(q={b})")
    q0_abs = abs(qmean.get(0, float("inf")))
    q1536_floor = all(abs(qmean.get(q, 0.0)) >= PAIR_Q1536_MIN_ABS for q in (-1536, 1536))
    return {
        "missing_q": missing,
        "sign_failures": sign_failures,
        "magnitude_failures": magnitude_failures,
        "q0_abs_mean": q0_abs,
        "q0_pass": q0_abs <= PAIR_Q0_MEAN_MAX_ABS,
        "q1536_floor_pass": q1536_floor,
        "passed": not missing and not sign_failures and not magnitude_failures and q0_abs <= PAIR_Q0_MEAN_MAX_ABS and q1536_floor,
    }


def odd_symmetry_report(qmean: dict[int, float]) -> dict[str, Any]:
    pairs: dict[str, Any] = {}
    passes: list[bool] = []
    for q in (512, 1024, 1536):
        if q not in qmean or -q not in qmean:
            pairs[str(q)] = {"passed": False, "reason": "missing q pair"}
            passes.append(False)
            continue
        residual = qmean[q] + qmean[-q]
        denominator = max(abs(qmean[q]), abs(qmean[-q]), 1.0)
        relative_error = abs(residual) / denominator
        passed = relative_error <= ODD_SYMMETRY_MAX_RELATIVE_ERROR
        pairs[str(q)] = {
            "positive_mean": qmean[q],
            "negative_mean": qmean[-q],
            "sum_residual": residual,
            "relative_error": relative_error,
            "passed": passed,
        }
        passes.append(passed)
    return {
        "threshold": ODD_SYMMETRY_MAX_RELATIVE_ERROR,
        "pairs": pairs,
        "passed": all(passes) and bool(passes),
    }


def nearest_q(y: float, slope: float) -> int:
    return min(ACTIVE_QS, key=lambda q: abs(y - slope * float(q)))


def classifier_report(samples: list[dict[str, Any]], slope: float) -> dict[str, Any]:
    labels = [str(q) for q in ACTIVE_QS]
    matrix: dict[str, dict[str, int]] = {
        label: {inner: 0 for inner in labels}
        for label in labels
    }
    correct = 0
    nonzero_sign_count = 0
    nonzero_sign_correct = 0
    for item in samples:
        actual = int(item["q"])
        predicted = nearest_q(float(item["y"]), slope)
        matrix[str(actual)][str(predicted)] += 1
        correct += int(actual == predicted)
        if actual != 0:
            nonzero_sign_count += 1
            nonzero_sign_correct += int(predicted * actual > 0)
    exact_accuracy = correct / len(samples) if samples else 0.0
    sign_accuracy = nonzero_sign_correct / nonzero_sign_count if nonzero_sign_count else 0.0
    return {
        "count": len(samples),
        "slope_used": slope,
        "exact_q_accuracy": exact_accuracy,
        "nonzero_sign_accuracy": sign_accuracy,
        "confusion_matrix": matrix,
        "passed": exact_accuracy >= CLASSIFIER_MIN_EXACT_ACCURACY and sign_accuracy >= CLASSIFIER_MIN_SIGN_ACCURACY,
    }


def heldout_classifier_report(samples: list[dict[str, Any]], factor: str) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for value in sorted({item[factor] for item in samples}, key=str):
        train = [item for item in samples if item[factor] != value]
        test = [item for item in samples if item[factor] == value]
        train_slope = slope_through_origin(train)
        report[str(value)] = classifier_report(test, train_slope)
    return report


def source_off_strata_report(source_off: list[dict[str, Any]]) -> dict[str, Any]:
    factor_reports: dict[str, Any] = {}
    for factor in ("session", "replicate", "mapping", "delay_label", "source_order"):
        levels: dict[str, Any] = {}
        for value, items in group_by(source_off, factor).items():
            abs_values = [abs(float(item["y"])) for item in items]
            levels[str(value)] = {
                "count": len(items),
                "mean_abs": mean(abs_values),
                "max_abs": max(abs_values) if abs_values else float("inf"),
                "mean_pass": mean(abs_values) <= PAIR_SOURCE_OFF_MEAN_ABS,
                "max_pass": (max(abs_values) if abs_values else float("inf")) <= PAIR_SOURCE_OFF_MAX_ABS,
            }
        factor_reports[factor] = {
            "levels": dict(sorted(levels.items())),
            "passed": all(level["mean_pass"] and level["max_pass"] for level in levels.values()) and bool(levels),
        }
    full_key_factors = ("session", "replicate", "mapping", "delay_label", "source_order")
    full_cells: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for item in source_off:
        full_cells[tuple(item[factor] for factor in full_key_factors)].append(item)
    full_cell_abs_max = max((abs(float(item["y"])) for items in full_cells.values() for item in items), default=float("inf"))
    full_cell_violations = [
        {
            "cell": "|".join(str(part) for part in cell),
            "values": [float(item["y"]) for item in items],
        }
        for cell, items in sorted(full_cells.items(), key=lambda entry: "|".join(str(part) for part in entry[0]))
        if any(abs(float(item["y"])) > PAIR_SOURCE_OFF_MAX_ABS for item in items)
    ]
    return {
        "factor_reports": factor_reports,
        "full_cell_factors": list(full_key_factors),
        "full_cell_count": len(full_cells),
        "full_cell_min_count": min((len(items) for items in full_cells.values()), default=0),
        "full_cell_max_count": max((len(items) for items in full_cells.values()), default=0),
        "full_cell_abs_max": full_cell_abs_max,
        "full_cell_violations": full_cell_violations[:16],
        "passed": (
            all(report["passed"] for report in factor_reports.values())
            and not full_cell_violations
            and len(full_cells) > 0
        ),
    }


def balance_report(active: list[dict[str, Any]], source_off: list[dict[str, Any]]) -> dict[str, Any]:
    def summarize(items: list[dict[str, Any]], factors: tuple[str, ...]) -> dict[str, Any]:
        counts = Counter(tuple(item[factor] for factor in factors) for item in items)
        return {
            "factors": list(factors),
            "total_rows": len(items),
            "unique_cells": len(counts),
            "min_cell_count": min(counts.values(), default=0),
            "max_cell_count": max(counts.values(), default=0),
            "passed": bool(counts) and min(counts.values()) == max(counts.values()) == 1,
        }

    return {
        "active_full_factor_q": summarize(active, ("session", "replicate", "mapping", "delay_label", "source_order", "q")),
        "source_off_full_factor": summarize(source_off, ("session", "replicate", "mapping", "delay_label", "source_order")),
    }


def query_observation_report(packet: dict[str, Any]) -> dict[str, Any]:
    counts = Counter((record["matrix_block"], record["query"]) for record in packet["raw_records"])
    required = [
        ("persistence_matrix", "query_A"),
        ("persistence_matrix", "query_B"),
        ("persistence_matrix", "query_A_then_B"),
        ("persistence_matrix", "query_B_then_A"),
        ("factorial_nonadditivity", "query_A_then_B"),
        ("factorial_nonadditivity", "query_B_then_A"),
    ]
    missing = [f"{matrix_block}|{query}" for matrix_block, query in required if counts[(matrix_block, query)] <= 0]
    return {
        "total_raw_records": len(packet["raw_records"]),
        "counts": {
            f"{matrix_block}|{query}": count
            for (matrix_block, query), count in sorted(counts.items())
        },
        "primary_fit_inputs": {
            "persistence_matrix|query_A": counts[("persistence_matrix", "query_A")],
            "persistence_matrix|query_B": counts[("persistence_matrix", "query_B")],
        },
        "ordered_query_observations_retained_not_fit": {
            "persistence_matrix|query_A_then_B": counts[("persistence_matrix", "query_A_then_B")],
            "persistence_matrix|query_B_then_A": counts[("persistence_matrix", "query_B_then_A")],
            "factorial_nonadditivity|query_A_then_B": counts[("factorial_nonadditivity", "query_A_then_B")],
            "factorial_nonadditivity|query_B_then_A": counts[("factorial_nonadditivity", "query_B_then_A")],
        },
        "missing_required_observations": missing,
        "passed": not missing,
    }


def build_pair_samples(packet: dict[str, Any], schedule: dict[str, Any], *, field: str, mapping_policy: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    rows_by_id = {row["tuple_id"]: row for row in schedule["rows"]}
    pair_values: dict[tuple[Any, ...], dict[str, float]] = defaultdict(dict)
    failures: list[str] = []
    for record in packet["raw_records"]:
        row = rows_by_id[record["tuple_id"]]
        if row["matrix_block"] != "persistence_matrix" or row["query"] not in {"query_A", "query_B"}:
            continue
        key = (
            row["session"],
            int(row["replicate"]),
            row["delay_label"],
            row["mapping"],
            row["source_order"],
            int(row["q"]),
            bool(row["source_off_control"]),
        )
        pair_values[key][row["query"]] = float(record[field])
    active: list[dict[str, Any]] = []
    source_off: list[dict[str, Any]] = []
    for key, values in pair_values.items():
        if set(values) != {"query_A", "query_B"}:
            failures.append(f"incomplete pair {key}: {sorted(values)}")
            continue
        session, replicate, delay_label, mapping, source_order, q, is_source_off = key
        y = values["query_A"] - values["query_B"]
        if mapping_policy == "physical_sign":
            y *= 1.0 if mapping == "map0" else -1.0
        elif mapping_policy != "logical_query_identity":
            raise ValueError(f"unknown mapping policy {mapping_policy}")
        item = {
            "session": session,
            "replicate": replicate,
            "delay_label": delay_label,
            "mapping": mapping,
            "source_order": source_order,
            "q": q,
            "source_off_control": is_source_off,
            "observable": field,
            "mapping_policy": mapping_policy,
            "y": y,
        }
        if is_source_off:
            source_off.append(item)
        else:
            active.append(item)
    return active, source_off, failures


def heldout_report(samples: list[dict[str, Any]], factor: str) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for value in sorted({item[factor] for item in samples}, key=str):
        train = [item for item in samples if item[factor] != value]
        test = [item for item in samples if item[factor] == value]
        train_slope = slope_through_origin(train)
        test_rmse = rmse(test, train_slope)
        denominator = abs(train_slope) * 1536.0
        relative = test_rmse / denominator if denominator else float("inf")
        report[str(value)] = {
            "train_count": len(train),
            "test_count": len(test),
            "train_slope": train_slope,
            "test_rmse": test_rmse,
            "relative_rmse_to_q1536": relative,
            "passed": relative <= HELDOUT_MAX_REL_RMSE,
        }
    return report


def channel_specificity_report(packet: dict[str, Any], schedule: dict[str, Any]) -> dict[str, Any]:
    fields = ["dirty_probe_response", "change_to_dirty", "cpu_cycles", "duration_ns"]
    reports: dict[str, Any] = {}
    for candidate_field in fields:
        candidate = adjudicate_packet(
            packet,
            schedule,
            field=candidate_field,
            mapping_policy="logical_query_identity",
            include_channel_specificity=False,
        )
        reports[candidate_field] = {
            "passed": candidate["passed"],
            "result_class": candidate["result_class"],
            "slope": candidate["global_fit"]["field_slope_per_q"],
            "rmse": candidate["global_fit"]["rmse"],
            "r2": candidate["global_fit"]["r2"],
            "source_off_abs_max": candidate["controls"]["source_off_abs_max"],
        }
    return {
        "diagnostic_only": True,
        "prospective_gate": False,
        "primary_endpoint": "dirty_probe_response",
        "primary_endpoint_requirement": "primary paired dirty-probe law must independently pass all primary gates",
        "secondary_channel_policy": "secondary channels may pass or fail without invalidating the primary result and may not substitute for primary endpoint failure",
        "reports": reports,
        "attempt_3_observation": "attempt_3_strongest_scalar_q_signal_is_in_dirty_probe_response_under_this_sidecar",
        "attempt_3_primary_passed": reports["dirty_probe_response"]["passed"],
        "attempt_3_secondary_channels_failed_same_law": (
            not reports["change_to_dirty"]["passed"]
            and not reports["cpu_cycles"]["passed"]
            and not reports["duration_ns"]["passed"]
        ),
    }


def adjudicate_packet(
    packet: dict[str, Any],
    schedule: dict[str, Any],
    *,
    field: str = "dirty_probe_response",
    mapping_policy: str = "logical_query_identity",
    include_channel_specificity: bool = True,
) -> dict[str, Any]:
    active, source_off, pair_failures = build_pair_samples(packet, schedule, field=field, mapping_policy=mapping_policy)
    source_abs = [abs(float(item["y"])) for item in source_off]
    global_slope = slope_through_origin(active)
    global_rmse = rmse(active, global_slope)
    global_r2 = r2_score(active, global_slope)
    strata: dict[str, Any] = {}
    stratum_passes: list[bool] = []
    by_stratum: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for item in active:
        by_stratum[(str(item["session"]), int(item["replicate"]))].append(item)
    for (session, replicate), items in sorted(by_stratum.items()):
        slope = slope_through_origin(items)
        qmean = q_means(items)
        law = pass_sign_and_magnitude(qmean)
        odd = odd_symmetry_report(qmean)
        ols = ols_report(items)
        classifier = classifier_report(items, slope)
        model_r2 = r2_score(items, slope)
        passed = (
            law["passed"]
            and odd["passed"]
            and ols["intercept_abs_pass"]
            and classifier["passed"]
            and model_r2 >= PAIR_MODEL_MIN_R2
        )
        stratum_passes.append(passed)
        strata[f"{session}:replicate_{replicate}"] = {
            "count": len(items),
            "field_slope_per_q": slope,
            "slope_dirty_probe_per_q": slope,
            "rmse": rmse(items, slope),
            "r2": model_r2,
            "q_means": {str(q): value for q, value in qmean.items()},
            "q_law": law,
            "odd_symmetry": odd,
            "ordinary_least_squares": ols,
            "nearest_q_classifier": classifier,
            "passed": passed,
        }
    global_q_means = q_means(active)
    global_odd = odd_symmetry_report(global_q_means)
    global_ols = ols_report(active)
    global_classifier = classifier_report(active, global_slope)
    mapping_slopes = {key: slope_through_origin(items) for key, items in group_by(active, "mapping").items()}
    delay_slopes = {key: slope_through_origin(items) for key, items in group_by(active, "delay_label").items()}
    session_slopes = {key: slope_through_origin(items) for key, items in group_by(active, "session").items()}
    source_order_slopes = {key: slope_through_origin(items) for key, items in group_by(active, "source_order").items()}
    mapping_disagreement = rel_disagreement(list(mapping_slopes.values()))
    delay_disagreement = rel_disagreement(list(delay_slopes.values()))
    session_disagreement = rel_disagreement(list(session_slopes.values()))
    source_order_disagreement = rel_disagreement(list(source_order_slopes.values()))
    heldout = {
        "session": heldout_report(active, "session"),
        "replicate": heldout_report(active, "replicate"),
        "mapping": heldout_report(active, "mapping"),
        "delay_label": heldout_report(active, "delay_label"),
        "source_order": heldout_report(active, "source_order"),
    }
    heldout_classifier = {
        "session": heldout_classifier_report(active, "session"),
        "replicate": heldout_classifier_report(active, "replicate"),
        "mapping": heldout_classifier_report(active, "mapping"),
        "delay_label": heldout_classifier_report(active, "delay_label"),
        "source_order": heldout_classifier_report(active, "source_order"),
    }
    control_report = {
        "source_off_pair_count": len(source_off),
        "source_off_abs_mean": mean(source_abs),
        "source_off_abs_max": max(source_abs) if source_abs else float("inf"),
        "source_off_mean_pass": mean(source_abs) <= PAIR_SOURCE_OFF_MEAN_ABS,
        "source_off_max_pass": (max(source_abs) if source_abs else float("inf")) <= PAIR_SOURCE_OFF_MAX_ABS,
    }
    query_observations = query_observation_report(packet)
    source_off_strata = source_off_strata_report(source_off)
    balance = balance_report(active, source_off)
    min_q1536_signal = min(abs(global_q_means.get(-1536, 0.0)), abs(global_q_means.get(1536, 0.0)))
    signal_to_null_ratio = min_q1536_signal / max(control_report["source_off_abs_max"], 1.0)
    channel_specificity = (
        channel_specificity_report(packet, schedule)
        if include_channel_specificity and field == "dirty_probe_response" and mapping_policy == "logical_query_identity"
        else None
    )
    gates = {
        "pair_completeness": not pair_failures,
        "query_observations_preserved": query_observations["passed"],
        "factor_balance": balance["active_full_factor_q"]["passed"] and balance["source_off_full_factor"]["passed"],
        "source_off_mean_abs": control_report["source_off_mean_pass"],
        "source_off_max_abs": control_report["source_off_max_pass"],
        "source_off_strata": source_off_strata["passed"],
        "signal_to_null_ratio": signal_to_null_ratio >= SIGNAL_TO_NULL_MIN_RATIO,
        "global_r2": global_r2 >= PAIR_MODEL_MIN_R2,
        "global_intercept": global_ols["intercept_abs_pass"],
        "global_odd_symmetry": global_odd["passed"],
        "global_classifier": global_classifier["passed"],
        "all_strata": all(stratum_passes) and bool(stratum_passes),
        "mapping_slope_agreement": mapping_disagreement <= SLOPE_MAX_REL_DISAGREEMENT,
        "delay_slope_agreement": delay_disagreement <= SLOPE_MAX_REL_DISAGREEMENT,
        "session_slope_agreement": session_disagreement <= SLOPE_MAX_REL_DISAGREEMENT,
        "source_order_slope_agreement": source_order_disagreement <= SLOPE_MAX_REL_DISAGREEMENT,
        "heldout_session": all(item["passed"] for item in heldout["session"].values()),
        "heldout_replicate": all(item["passed"] for item in heldout["replicate"].values()),
        "heldout_mapping": all(item["passed"] for item in heldout["mapping"].values()),
        "heldout_delay": all(item["passed"] for item in heldout["delay_label"].values()),
        "heldout_source_order": all(item["passed"] for item in heldout["source_order"].values()),
        "heldout_classifier_session": all(item["passed"] for item in heldout_classifier["session"].values()),
        "heldout_classifier_replicate": all(item["passed"] for item in heldout_classifier["replicate"].values()),
        "heldout_classifier_mapping": all(item["passed"] for item in heldout_classifier["mapping"].values()),
        "heldout_classifier_delay": all(item["passed"] for item in heldout_classifier["delay_label"].values()),
        "heldout_classifier_source_order": all(item["passed"] for item in heldout_classifier["source_order"].values()),
    }
    passed = all(gates.values())
    return {
        "schema": "FAMILY10H_PAIRED_DIRTY_PROBE_PROSPECTIVE_ADJUDICATION_V1",
        "result_class": RESULT_SUPPORTED if passed else RESULT_NOT_SUPPORTED,
        "passed": passed,
        "scientific_claim": POSITIVE_SCIENTIFIC_CLAIM if passed else NEGATIVE_SCIENTIFIC_CLAIM,
        "prospective_claim_ceiling": PROSPECTIVE_CLAIM_CEILING,
        "claim_boundary": {
            "reproducible_public_q_dependent_response_observed": passed,
            "dimension": "one_dimensional_scalar_codeword_readout",
            "full_carrier_state_tomography_established": False,
            "v1_0_retained_evidence_unchanged": True,
            "small_wall_promoted": False,
            "catalytic_borrowing_established": False,
            "physical_relational_memory_established": False,
            "relational_carrier_established": False,
        },
        "official_result_replaced": False,
        "small_wall_promoted": False,
        "observable_law": {
            "field": field,
            "paired_observable": "D_single = dirty_probe_response(query_A) - dirty_probe_response(query_B)",
            "mapping_policy": mapping_policy,
            "mapping_policy_reason": "Runtime maps logical query lanes through the same map variant used during preparation; map is therefore a consistency factor for logical query_A/query_B, not a sign inversion.",
            "threshold_provenance": THRESHOLD_PROVENANCE,
        },
        "thresholds": {
            "source_off_pair_abs_max": PAIR_SOURCE_OFF_MAX_ABS,
            "source_off_pair_abs_mean": PAIR_SOURCE_OFF_MEAN_ABS,
            "q0_abs_mean_max": PAIR_Q0_MEAN_MAX_ABS,
            "q1536_abs_mean_min": PAIR_Q1536_MIN_ABS,
            "model_min_r2": PAIR_MODEL_MIN_R2,
            "heldout_max_relative_rmse": HELDOUT_MAX_REL_RMSE,
            "slope_max_relative_disagreement": SLOPE_MAX_REL_DISAGREEMENT,
            "intercept_abs_max": PAIR_INTERCEPT_MAX_ABS,
            "odd_symmetry_max_relative_error": ODD_SYMMETRY_MAX_RELATIVE_ERROR,
            "signal_to_null_min_ratio": SIGNAL_TO_NULL_MIN_RATIO,
            "classifier_min_exact_accuracy": CLASSIFIER_MIN_EXACT_ACCURACY,
            "classifier_min_sign_accuracy": CLASSIFIER_MIN_SIGN_ACCURACY,
        },
        "pair_failures": pair_failures[:32],
        "pair_failure_count": len(pair_failures),
        "sample_count": len(active),
        "query_observations": query_observations,
        "factor_balance": balance,
        "global_fit": {
            "observable": field,
            "field_slope_per_q": global_slope,
            "slope_dirty_probe_per_q": global_slope,
            "rmse": global_rmse,
            "r2": global_r2,
            "q_means": {str(q): value for q, value in global_q_means.items()},
            "ordinary_least_squares": global_ols,
            "odd_symmetry": global_odd,
            "signal_to_null_ratio": signal_to_null_ratio,
            "nearest_q_classifier": global_classifier,
        },
        "controls": control_report,
        "source_off_strata": source_off_strata,
        "strata": strata,
        "mapping_slopes": mapping_slopes,
        "mapping_slope_relative_disagreement": mapping_disagreement,
        "delay_slopes": delay_slopes,
        "delay_slope_relative_disagreement": delay_disagreement,
        "session_slopes": session_slopes,
        "session_slope_relative_disagreement": session_disagreement,
        "source_order_slopes": source_order_slopes,
        "source_order_slope_relative_disagreement": source_order_disagreement,
        "heldout": heldout,
        "heldout_classifier": heldout_classifier,
        "channel_specificity": channel_specificity,
        "gates": gates,
    }


def mutate_source_off(packet: dict[str, Any], schedule: dict[str, Any]) -> dict[str, Any]:
    rows_by_id = {row["tuple_id"]: row for row in schedule["rows"]}
    mutated = json.loads(json.dumps(packet))
    for record in mutated["raw_records"]:
        row = rows_by_id[record["tuple_id"]]
        if row["matrix_block"] == "persistence_matrix" and row["source_off_control"] and row["query"] == "query_A":
            record["dirty_probe_response"] += 1000
            break
    return mutated


def mutate_flat_signal(packet: dict[str, Any], schedule: dict[str, Any]) -> dict[str, Any]:
    rows_by_id = {row["tuple_id"]: row for row in schedule["rows"]}
    mutated = json.loads(json.dumps(packet))
    for record in mutated["raw_records"]:
        row = rows_by_id[record["tuple_id"]]
        if row["matrix_block"] == "persistence_matrix" and not row["source_off_control"] and row["query"] in {"query_A", "query_B"}:
            record["dirty_probe_response"] = 2000
    return mutated


def mutate_swap_query_pair_values(packet: dict[str, Any], schedule: dict[str, Any]) -> dict[str, Any]:
    rows_by_id = {row["tuple_id"]: row for row in schedule["rows"]}
    mutated = json.loads(json.dumps(packet))
    pair_indices: dict[tuple[Any, ...], dict[str, int]] = defaultdict(dict)
    for index, record in enumerate(mutated["raw_records"]):
        row = rows_by_id[record["tuple_id"]]
        if row["matrix_block"] != "persistence_matrix" or row["query"] not in {"query_A", "query_B"}:
            continue
        key = (
            row["session"],
            int(row["replicate"]),
            row["delay_label"],
            row["mapping"],
            row["source_order"],
            int(row["q"]),
            bool(row["source_off_control"]),
        )
        pair_indices[key][row["query"]] = index
    for indices in pair_indices.values():
        if set(indices) != {"query_A", "query_B"}:
            continue
        a_index = indices["query_A"]
        b_index = indices["query_B"]
        mutated["raw_records"][a_index]["dirty_probe_response"], mutated["raw_records"][b_index]["dirty_probe_response"] = (
            mutated["raw_records"][b_index]["dirty_probe_response"],
            mutated["raw_records"][a_index]["dirty_probe_response"],
        )
    return mutated


def mutate_negate_active_q_schedule(schedule: dict[str, Any]) -> dict[str, Any]:
    mutated = json.loads(json.dumps(schedule))
    for row in mutated["rows"]:
        if row["matrix_block"] == "persistence_matrix" and row["query"] in {"query_A", "query_B"} and not row["source_off_control"]:
            row["q"] = -int(row["q"])
    return mutated


def mutate_synthetic_negative_packet(packet: dict[str, Any], schedule: dict[str, Any]) -> dict[str, Any]:
    rows_by_id = {row["tuple_id"]: row for row in schedule["rows"]}
    mutated = json.loads(json.dumps(packet))
    for record in mutated["raw_records"]:
        row = rows_by_id[record["tuple_id"]]
        if row["matrix_block"] == "persistence_matrix" and row["query"] in {"query_A", "query_B"}:
            record["dirty_probe_response"] = 0
    return mutated


def fail_closed_claim_state(report: dict[str, Any]) -> dict[str, Any]:
    boundary = report.get("claim_boundary", {})
    checks = {
        "result_class_not_supported": report.get("result_class") != RESULT_SUPPORTED,
        "positive_scientific_claim_absent": report.get("scientific_claim") != POSITIVE_SCIENTIFIC_CLAIM,
        "q_dependent_response_observed_false": boundary.get("reproducible_public_q_dependent_response_observed") is False,
    }
    return {
        "passed": all(checks.values()),
        "result_class": report.get("result_class"),
        "scientific_claim": report.get("scientific_claim"),
        "reproducible_public_q_dependent_response_observed": boundary.get("reproducible_public_q_dependent_response_observed"),
        "checks": checks,
    }


def expect_runtime_error(label: str, fn: Any) -> dict[str, Any]:
    try:
        fn()
    except RuntimeError as exc:
        return {"passed": True, "error": str(exc)}
    return {"passed": False, "error": f"{label} did not fail closed"}


def write_test_tar(
    path: Path,
    members: dict[str, bytes],
    *,
    archive_run_id: str = RUN_ID,
    duplicate: str | None = None,
    omit: str | None = None,
) -> None:
    with tarfile.open(path, "w:gz") as archive:
        for relative_path, data in members.items():
            if relative_path == omit:
                continue
            member_name = f"{archive_run_id}/{relative_path}"
            info = tarfile.TarInfo(member_name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
            if duplicate == relative_path:
                duplicate_info = tarfile.TarInfo(member_name)
                duplicate_info.size = len(data)
                archive.addfile(duplicate_info, io.BytesIO(data))


def archive_negative_self_tests(attempt_dir: Path, archive_run_id: str = RUN_ID) -> dict[str, Any]:
    attempt_number = attempt_dir.name.split("_")[-1]
    tar_path = attempt_dir / f"ATTEMPT_{attempt_number}_REMOTE_ROOT.tar.gz"
    copyback = load_json_path(attempt_dir / f"ATTEMPT_{attempt_number}_COPYBACK_RECEIPT.json")
    inventory = load_json_path(attempt_dir / f"ATTEMPT_{attempt_number}_EVIDENCE_INVENTORY.json")
    payload, _ = read_attempt_output(attempt_dir, archive_run_id=archive_run_id)
    member_payload = {
        relative_path: payload[key]
        for key, relative_path in REQUIRED_ARCHIVE_MEMBERS.items()
    }

    def with_archive_values(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
        archive_sha = sha256_path(path)
        archive_size = path.stat().st_size
        mutated_copyback = copy.deepcopy(copyback)
        mutated_inventory = copy.deepcopy(inventory)
        mutated_copyback["local_archive"] = str(path.resolve())
        mutated_copyback["local_archive_sha256"] = archive_sha
        mutated_copyback["local_archive_size"] = archive_size
        mutated_inventory["remote_root_archive"]["relative_path"] = path.name
        mutated_inventory["remote_root_archive"]["sha256"] = archive_sha
        mutated_inventory["remote_root_archive"]["size"] = archive_size
        for item in mutated_inventory["files"]:
            if item.get("relative_path") == tar_path.name:
                item["relative_path"] = path.name
                item["sha256"] = archive_sha
                item["size"] = archive_size
        return mutated_copyback, mutated_inventory

    with tempfile.TemporaryDirectory(prefix="family10h_q_readout_negative_") as tmp:
        tmpdir = Path(tmp)
        missing_tar = tmpdir / "missing_member.tar.gz"
        duplicate_tar = tmpdir / "duplicate_member.tar.gz"
        write_test_tar(missing_tar, member_payload, archive_run_id=archive_run_id, omit="output/raw_records.jsonl")
        write_test_tar(duplicate_tar, member_payload, archive_run_id=archive_run_id, duplicate="output/raw_records.jsonl")
        missing_copyback, missing_inventory = with_archive_values(missing_tar)
        duplicate_copyback, duplicate_inventory = with_archive_values(duplicate_tar)

        snapshot_attempt_dir = tmpdir / "snapshot_attempt"
        snapshot_root = snapshot_attempt_dir / "remote_root_snapshot" / archive_run_id
        (snapshot_root / "output").mkdir(parents=True)
        for key, relative_path in REQUIRED_ARCHIVE_MEMBERS.items():
            target = snapshot_root / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            data = payload[key]
            if key == "feature_freeze.json":
                data = data + b"\n"
            target.write_bytes(data)

        wrong_hash = copy.deepcopy(copyback)
        wrong_hash["local_archive_sha256"] = "0" * 64
        wrong_size = copy.deepcopy(copyback)
        wrong_size["local_archive_size"] = int(wrong_size["local_archive_size"]) + 1
        receipt_mismatch = copy.deepcopy(copyback)
        receipt_mismatch["local_archive"] = str((tmpdir / "wrong-name.tar.gz").resolve())

        tests = {
            "wrong_archive_hash": expect_runtime_error(
                "wrong_archive_hash",
                lambda: verify_archive_receipts(attempt_dir, tar_path, wrong_hash, inventory),
            ),
            "wrong_archive_size": expect_runtime_error(
                "wrong_archive_size",
                lambda: verify_archive_receipts(attempt_dir, tar_path, wrong_size, inventory),
            ),
            "copyback_receipt_mismatch": expect_runtime_error(
                "copyback_receipt_mismatch",
                lambda: verify_archive_receipts(attempt_dir, tar_path, receipt_mismatch, inventory),
            ),
            "missing_required_member": expect_runtime_error(
                "missing_required_member",
                lambda: read_output_bytes_from_verified_archive(missing_tar, missing_copyback, archive_run_id=archive_run_id),
            ),
            "duplicate_matching_archive_member": expect_runtime_error(
                "duplicate_matching_archive_member",
                lambda: read_output_bytes_from_verified_archive(duplicate_tar, duplicate_copyback, archive_run_id=archive_run_id),
            ),
            "snapshot_archive_content_mismatch": expect_runtime_error(
                "snapshot_archive_content_mismatch",
                lambda: snapshot_payload_if_equal(snapshot_attempt_dir, payload, archive_run_id=archive_run_id),
            ),
        }
    return {
        "passed": all(item["passed"] for item in tests.values()),
        "tests": tests,
    }


def run_self_tests(
    public: Any,
    packet: dict[str, Any],
    schedule: dict[str, Any],
    attempt_dir: Path,
    archive_run_id: str = RUN_ID,
) -> dict[str, Any]:
    repaired = adjudicate_packet(packet, schedule)
    change_to_dirty = adjudicate_packet(packet, schedule, field="change_to_dirty")
    cpu_cycles = adjudicate_packet(packet, schedule, field="cpu_cycles")
    duration_ns = adjudicate_packet(packet, schedule, field="duration_ns")
    legacy_mapping = adjudicate_packet(packet, schedule, mapping_policy="physical_sign")
    source_off_mutant = adjudicate_packet(mutate_source_off(packet, schedule), schedule)
    flat_mutant = adjudicate_packet(mutate_flat_signal(packet, schedule), schedule)
    swapped_query_mutant = adjudicate_packet(mutate_swap_query_pair_values(packet, schedule), schedule)
    negated_q_mutant = adjudicate_packet(packet, mutate_negate_active_q_schedule(schedule))
    synthetic_negative = adjudicate_packet(mutate_synthetic_negative_packet(packet, schedule), schedule)
    negative_reports = {
        "change_to_dirty": change_to_dirty,
        "cpu_cycles": cpu_cycles,
        "duration_ns": duration_ns,
        "legacy_mapping": legacy_mapping,
        "source_off_mutant": source_off_mutant,
        "flat_mutant": flat_mutant,
        "swapped_query_mutant": swapped_query_mutant,
        "negated_q_mutant": negated_q_mutant,
        "synthetic_negative_packet": synthetic_negative,
    }
    negative_claim_state = {
        label: fail_closed_claim_state(report) for label, report in negative_reports.items()
    }
    invalid_packet = json.loads(json.dumps(packet))
    invalid_packet["raw_records"] = []
    invalid_validation = public.validate_evidence_packet(invalid_packet, schedule)
    custody_invalid = invalid_adjudication_result(invalid_validation)
    custody_invalid_claim_state = fail_closed_claim_state(custody_invalid)
    archive_negatives = archive_negative_self_tests(attempt_dir, archive_run_id=archive_run_id)
    checks = {
        "paired_dirty_probe_v1_1_q_readout_confirmed": repaired["passed"],
        "paired_dirty_probe_v1_1_positive_claim": repaired["scientific_claim"] == POSITIVE_SCIENTIFIC_CLAIM,
        "archive_negative_self_tests": archive_negatives["passed"],
        "change_to_dirty_channel_not_sufficient": not change_to_dirty["passed"],
        "cpu_cycles_channel_not_sufficient": not cpu_cycles["passed"],
        "duration_channel_not_sufficient": not duration_ns["passed"],
        "legacy_mapping_sign_cancels_signal": not legacy_mapping["passed"],
        "source_off_pair_smuggle_rejected": not source_off_mutant["passed"],
        "flat_signal_rejected": not flat_mutant["passed"],
        "swapped_query_pair_rejected": not swapped_query_mutant["passed"],
        "negated_q_label_rejected": not negated_q_mutant["passed"],
        "synthetic_negative_packet_rejected": not synthetic_negative["passed"],
        "negative_replays_fail_closed_claim_state": all(item["passed"] for item in negative_claim_state.values()),
        "custody_invalid_packet_rejected": not invalid_validation["passed"],
        "custody_invalid_fails_closed_claim_state": custody_invalid_claim_state["passed"],
    }
    return {
        "schema": "FAMILY10H_PAIRED_DIRTY_PROBE_PROSPECTIVE_SELF_TEST_V1",
        "passed": all(checks.values()),
        "checks": checks,
        "regression_classes": {
            "repaired": repaired["result_class"],
            "change_to_dirty": change_to_dirty["result_class"],
            "cpu_cycles": cpu_cycles["result_class"],
            "duration_ns": duration_ns["result_class"],
            "legacy_mapping": legacy_mapping["result_class"],
            "source_off_mutant": source_off_mutant["result_class"],
            "flat_mutant": flat_mutant["result_class"],
            "swapped_query_mutant": swapped_query_mutant["result_class"],
            "negated_q_mutant": negated_q_mutant["result_class"],
            "synthetic_negative_packet": synthetic_negative["result_class"],
        },
        "negative_claim_state": negative_claim_state,
        "custody_invalid_claim_state": custody_invalid_claim_state,
        "custody_invalid_validation": invalid_validation,
        "legacy_mapping_gates": legacy_mapping["gates"],
        "source_off_mutant_controls": source_off_mutant["controls"],
        "swapped_query_mutant_gates": swapped_query_mutant["gates"],
        "negated_q_mutant_gates": negated_q_mutant["gates"],
        "channel_specificity": repaired["channel_specificity"],
        "archive_negative_self_tests": archive_negatives,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt-dir", type=Path, default=DEFAULT_ATTEMPT_DIR)
    parser.add_argument("--archive-run-id", default=RUN_ID)
    parser.add_argument("--out", type=Path, default=HERE / "PAIRED_DIRTY_PROBE_PROSPECTIVE_ADJUDICATION.json")
    parser.add_argument("--self-test-out", type=Path, default=HERE / "PAIRED_DIRTY_PROBE_PROSPECTIVE_SELF_TEST.json")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    public = load_public_module()
    packet, schedule, provenance = build_packet(public, args.attempt_dir, archive_run_id=args.archive_run_id)
    validation = public.validate_evidence_packet(packet, schedule)
    if not validation["passed"]:
        result = invalid_adjudication_result(validation, provenance)
    else:
        result = adjudicate_packet(packet, schedule)
        result["validation"] = validation
        result["source_evidence"] = provenance
        result["source_evidence"]["attempt_dir"] = str(args.attempt_dir)
        result["source_evidence"]["schedule_sha256"] = public.digest(schedule)
    write_json(args.out, result)
    if args.self_test:
        self_test = run_self_tests(public, packet, schedule, args.attempt_dir, archive_run_id=args.archive_run_id) if validation["passed"] else {"schema": "FAMILY10H_PAIRED_DIRTY_PROBE_PROSPECTIVE_SELF_TEST_V1", "passed": False, "reason": "base packet invalid", "validation": validation}
        write_json(args.self_test_out, self_test)
        if not self_test["passed"]:
            print(json.dumps(self_test, indent=2, sort_keys=True))
            return 1
    print(json.dumps({
        "adjudication": str(args.out),
        "result_class": result["result_class"],
        "passed": result["passed"],
        "self_test": str(args.self_test_out) if args.self_test else None,
    }, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
