#!/usr/bin/env python3
"""Paired dirty-probe repair adjudicator for Family 10h tomography.

This is an offline analysis-repair sidecar. It does not modify the frozen
family10h_carrier_tomography_v1 package, does not authorize live contact, and
does not promote SMALL_WALL_CROSSED.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import statistics
import sys
import tarfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
CARRIER_ROOT = HERE.parent
PACKAGE_ROOT = CARRIER_ROOT / "family10h_carrier_tomography_v1"
RUN_ID = "family10h_carrier_tomography_v1_0"
DEFAULT_ATTEMPT_DIR = CARRIER_ROOT / "runs" / RUN_ID / "attempt_3"

RESULT_SUPPORTED = "FAMILY10H_PAIRED_DIRTY_PROBE_TOMOGRAPHY_SUPPORTED_RETROSPECTIVE"
RESULT_NOT_SUPPORTED = "FAMILY10H_PAIRED_DIRTY_PROBE_TOMOGRAPHY_NOT_SUPPORTED_RETROSPECTIVE"
RESULT_INVALID = "FAMILY10H_PAIRED_DIRTY_PROBE_TOMOGRAPHY_CUSTODY_INVALID"

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


def read_output_bytes_from_tar(tar_path: Path) -> dict[str, bytes]:
    wanted_suffixes = {
        "raw_records.jsonl": "/output/raw_records.jsonl",
        "source_death_receipts.jsonl": "/output/source_death_receipts.jsonl",
        "feature_freeze.json": "/output/feature_freeze.json",
        "output_target_execution_receipt.json": "/output_target_execution_receipt.json",
    }
    found: dict[str, bytes] = {}
    with tarfile.open(tar_path, "r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            normalized = "/" + member.name.replace("\\", "/")
            for key, suffix in wanted_suffixes.items():
                if normalized.endswith(suffix):
                    extracted = archive.extractfile(member)
                    if extracted is None:
                        raise RuntimeError(f"cannot extract {member.name}")
                    found[key] = extracted.read()
    missing = sorted(set(wanted_suffixes) - set(found))
    if missing:
        raise RuntimeError(f"remote-root archive missing output files: {missing}")
    return found


def read_attempt_output(attempt_dir: Path) -> tuple[dict[str, bytes], dict[str, Any]]:
    snapshot_output = attempt_dir / "remote_root_snapshot" / RUN_ID / "output"
    snapshot_receipt = attempt_dir / "remote_root_snapshot" / RUN_ID / "output_target_execution_receipt.json"
    provenance: dict[str, Any]
    if snapshot_output.exists():
        payload = {
            "raw_records.jsonl": (snapshot_output / "raw_records.jsonl").read_bytes(),
            "source_death_receipts.jsonl": (snapshot_output / "source_death_receipts.jsonl").read_bytes(),
            "feature_freeze.json": (snapshot_output / "feature_freeze.json").read_bytes(),
            "output_target_execution_receipt.json": snapshot_receipt.read_bytes(),
        }
        provenance = {
            "source": "extracted_snapshot",
            "snapshot_output": str(snapshot_output),
        }
    else:
        attempt_number = attempt_dir.name.split("_")[-1]
        tar_path = attempt_dir / f"ATTEMPT_{attempt_number}_REMOTE_ROOT.tar.gz"
        payload = read_output_bytes_from_tar(tar_path)
        provenance = {
            "source": "remote_root_archive",
            "archive": str(tar_path),
            "archive_sha256": sha256_path(tar_path),
            "archive_size": tar_path.stat().st_size,
        }
    provenance["output_hashes"] = {
        key: {"sha256": sha256_bytes(value), "size": len(value)}
        for key, value in payload.items()
    }
    return payload, provenance


def build_packet(public: Any, attempt_dir: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    schedule = public.load_schedule_from_artifacts()
    payload, provenance = read_attempt_output(attempt_dir)
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
        "reports": reports,
        "passed": (
            reports["dirty_probe_response"]["passed"]
            and not reports["change_to_dirty"]["passed"]
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
    if channel_specificity is not None:
        gates["channel_specificity"] = channel_specificity["passed"]
    passed = all(gates.values())
    return {
        "schema": "FAMILY10H_PAIRED_DIRTY_PROBE_REPAIR_ADJUDICATION_V1",
        "result_class": RESULT_SUPPORTED if passed else RESULT_NOT_SUPPORTED,
        "passed": passed,
        "official_result_replaced": False,
        "small_wall_promoted": False,
        "observable_law": {
            "field": field,
            "paired_observable": "D_single = dirty_probe_response(query_A) - dirty_probe_response(query_B)",
            "mapping_policy": mapping_policy,
            "mapping_policy_reason": "Runtime maps logical query lanes through the same map variant used during preparation; map is therefore a consistency factor for logical query_A/query_B, not a sign inversion.",
            "source_of_thresholds": "prospective constants in this sidecar, not tuned by a live retry",
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


def run_self_tests(packet: dict[str, Any], schedule: dict[str, Any]) -> dict[str, Any]:
    repaired = adjudicate_packet(packet, schedule)
    change_to_dirty = adjudicate_packet(packet, schedule, field="change_to_dirty")
    cpu_cycles = adjudicate_packet(packet, schedule, field="cpu_cycles")
    duration_ns = adjudicate_packet(packet, schedule, field="duration_ns")
    legacy_mapping = adjudicate_packet(packet, schedule, mapping_policy="physical_sign")
    source_off_mutant = adjudicate_packet(mutate_source_off(packet, schedule), schedule)
    flat_mutant = adjudicate_packet(mutate_flat_signal(packet, schedule), schedule)
    swapped_query_mutant = adjudicate_packet(mutate_swap_query_pair_values(packet, schedule), schedule)
    negated_q_mutant = adjudicate_packet(packet, mutate_negate_active_q_schedule(schedule))
    checks = {
        "paired_dirty_probe_attempt3_supported": repaired["passed"],
        "change_to_dirty_channel_not_sufficient": not change_to_dirty["passed"],
        "cpu_cycles_channel_not_sufficient": not cpu_cycles["passed"],
        "duration_channel_not_sufficient": not duration_ns["passed"],
        "legacy_mapping_sign_cancels_signal": not legacy_mapping["passed"],
        "source_off_pair_smuggle_rejected": not source_off_mutant["passed"],
        "flat_signal_rejected": not flat_mutant["passed"],
        "swapped_query_pair_rejected": not swapped_query_mutant["passed"],
        "negated_q_label_rejected": not negated_q_mutant["passed"],
    }
    return {
        "schema": "FAMILY10H_PAIRED_DIRTY_PROBE_REPAIR_SELF_TEST_V1",
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
        },
        "legacy_mapping_gates": legacy_mapping["gates"],
        "source_off_mutant_controls": source_off_mutant["controls"],
        "swapped_query_mutant_gates": swapped_query_mutant["gates"],
        "negated_q_mutant_gates": negated_q_mutant["gates"],
        "channel_specificity": repaired["channel_specificity"],
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt-dir", type=Path, default=DEFAULT_ATTEMPT_DIR)
    parser.add_argument("--out", type=Path, default=HERE / "PAIRED_DIRTY_PROBE_ADJUDICATION.json")
    parser.add_argument("--self-test-out", type=Path, default=HERE / "PAIRED_DIRTY_PROBE_SELF_TEST.json")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    public = load_public_module()
    packet, schedule, provenance = build_packet(public, args.attempt_dir)
    validation = public.validate_evidence_packet(packet, schedule)
    if not validation["passed"]:
        result = {
            "schema": "FAMILY10H_PAIRED_DIRTY_PROBE_REPAIR_ADJUDICATION_V1",
            "result_class": RESULT_INVALID,
            "passed": False,
            "validation": validation,
            "source_evidence": provenance,
            "official_result_replaced": False,
            "small_wall_promoted": False,
        }
    else:
        result = adjudicate_packet(packet, schedule)
        result["validation"] = validation
        result["source_evidence"] = provenance
        result["source_evidence"]["attempt_dir"] = str(args.attempt_dir)
        result["source_evidence"]["schedule_sha256"] = public.digest(schedule)
    write_json(args.out, result)
    if args.self_test:
        self_test = run_self_tests(packet, schedule) if validation["passed"] else {"schema": "FAMILY10H_PAIRED_DIRTY_PROBE_REPAIR_SELF_TEST_V1", "passed": False, "reason": "base packet invalid", "validation": validation}
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
