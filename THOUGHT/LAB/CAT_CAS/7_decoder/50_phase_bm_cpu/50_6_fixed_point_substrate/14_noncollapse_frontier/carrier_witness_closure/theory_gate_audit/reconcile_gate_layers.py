#!/usr/bin/env python3
"""Reconcile carrier-witness gates by scientific role without changing verdicts.

This is a derived audit over committed compact evidence. It does not read or
modify raw evidence, replace the frozen seven-gate contract, or promote a new
physical claim. Its purpose is to separate protocol integrity, mode transport,
phase transport, schedule specificity, canonical-basis fidelity, and metadata
leakage checks that were previously collapsed into one pass bit.

Silent carrier-off and scramble unshared-schedule runs are the retained null
baselines. They remain controls and are never used to fit a positive model.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

SCHEMA_ID = "CAT_CAS_PHASE6B5_THEORY_GATE_AUDIT_V1"

CONTRACT_GATE_KEYS = (
    "all_rows_restore",
    "real_accuracy_ge_0_60",
    "real_vs_pseudo_floor_ge_0_95",
    "pseudo_reject_floor_ge_0_95",
    "wrong_actual_match_ge_0_60",
    "wrong_declared_match_le_0_20",
    "phase_recovered_gt_0_30",
)

ANALYZER_ADDITIONAL_KEYS = (
    "real_mode_floor_ge_0_45",
    "pseudo_declared_match_le_0_35",
)

MODE_TRANSPORT_KEYS = (
    "real_accuracy_ge_0_60",
    "real_mode_floor_ge_0_45",
    "wrong_actual_match_ge_0_60",
    "wrong_declared_match_le_0_20",
)

PHASE_TRANSPORT_KEYS = ("phase_recovered_gt_0_30",)
SHARED_SCHEDULE_SPECIFICITY_KEYS = ("pseudo_reject_floor_ge_0_95",)
CANONICAL_BASIS_FIDELITY_KEYS = (
    "real_vs_pseudo_floor_ge_0_95",
    "pseudo_reject_floor_ge_0_95",
)
METADATA_LEAKAGE_KEYS = (
    "pseudo_declared_match_le_0_35",
    "wrong_declared_match_le_0_20",
)
PROTOCOL_INTEGRITY_KEYS = ("all_rows_restore",)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def gates_pass(gates: dict[str, Any], keys: tuple[str, ...]) -> bool:
    return all(gates.get(key) is True for key in keys)


def allowed_errors(total: int, threshold: float = 0.95) -> int:
    if total < 0:
        raise ValueError("total must be non-negative")
    return total - math.ceil(threshold * total)


def analyzer_pass(gates: dict[str, Any]) -> bool:
    return bool(gates) and all(value is True for value in gates.values())


def verdict_implies_pass(verdict: Any) -> bool:
    return verdict == "PHASE4B_CROSS_CORE_PDN_LOCKIN_WITNESS"


def classify_run(run: dict[str, Any]) -> dict[str, Any]:
    gates = run.get("scientific_gates", {})
    if not isinstance(gates, dict):
        raise ValueError(f"run {run.get('run_id')} has no scientific gate map")
    contract = gates_pass(gates, CONTRACT_GATE_KEYS)
    analyzer = analyzer_pass(gates)
    return {
        "run_id": run.get("run_id"),
        "route": run.get("route"),
        "seed": run.get("seed"),
        "condition": run.get("condition"),
        "reported_scientific_pass": run.get("scientific_pass"),
        "reported_verdict": run.get("verdict"),
        "contract_seven_gate_pass": contract,
        "analyzer_nine_gate_pass": analyzer,
        "reported_pass_matches_contract": run.get("scientific_pass") is contract,
        "verdict_matches_analyzer": verdict_implies_pass(run.get("verdict")) is analyzer,
        "protocol_integrity": gates_pass(gates, PROTOCOL_INTEGRITY_KEYS),
        "mode_transport": gates_pass(gates, MODE_TRANSPORT_KEYS),
        "phase_transport": gates_pass(gates, PHASE_TRANSPORT_KEYS),
        "shared_schedule_specificity": gates_pass(
            gates, SHARED_SCHEDULE_SPECIFICITY_KEYS
        ),
        "core_carrier_transport": gates_pass(
            gates,
            MODE_TRANSPORT_KEYS
            + PHASE_TRANSPORT_KEYS
            + SHARED_SCHEDULE_SPECIFICITY_KEYS,
        ),
        "canonical_basis_fidelity": gates_pass(
            gates, CANONICAL_BASIS_FIDELITY_KEYS
        ),
        "metadata_leakage_sanity": gates_pass(gates, METADATA_LEAKAGE_KEYS),
        "gate_map": gates,
    }


def route_summary(classified: list[dict[str, Any]]) -> dict[str, Any]:
    by_route: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in classified:
        if run.get("condition") == "matrix":
            by_route[str(run.get("route"))].append(run)
    fields = (
        "contract_seven_gate_pass",
        "analyzer_nine_gate_pass",
        "protocol_integrity",
        "mode_transport",
        "phase_transport",
        "shared_schedule_specificity",
        "core_carrier_transport",
        "canonical_basis_fidelity",
        "metadata_leakage_sanity",
    )
    result: dict[str, Any] = {}
    for route, runs in sorted(by_route.items()):
        ordered = sorted(runs, key=lambda item: int(item.get("seed", -1)))
        result[route] = {
            "matrix_runs": len(ordered),
            "counts": {
                field: sum(bool(run[field]) for run in ordered) for field in fields
            },
            "seed_status": {
                str(run.get("seed")): {field: bool(run[field]) for field in fields}
                for run in ordered
            },
        }
    return result


def finite_sample_report(decomposition: dict[str, Any]) -> dict[str, Any]:
    runs = decomposition.get("runs", {})
    if not isinstance(runs, dict):
        raise ValueError("official decomposition has no runs object")
    output: dict[str, Any] = {}
    all_denominators: list[int] = []
    for run_id, run in sorted(runs.items()):
        groups = run.get("pseudo_groups", {})
        group_rows: dict[str, Any] = {}
        for mode, group in sorted(groups.items()):
            denominator = int(group.get("combined_denominator", 0))
            tolerance = allowed_errors(denominator)
            all_denominators.append(denominator)
            group_rows[mode] = {
                "combined_denominator": denominator,
                "allowed_errors_at_0_95": tolerance,
                "observed_combined_accuracy": group.get("combined_accuracy"),
                "false_accepts": group.get("false_accepts"),
                "real_false_rejects": run.get("real_modes", {})
                .get(mode, {})
                .get("false_rejects"),
                "zero_error_required": tolerance == 0,
            }
        output[run_id] = {
            "threshold": run.get("threshold"),
            "floor_mode": run.get("floor_mode"),
            "pseudo_reject_floor": run.get("metrics", {}).get(
                "pseudo_reject_floor"
            ),
            "real_accuracy": run.get("metrics", {}).get("real_accuracy"),
            "real_vs_pseudo_floor": run.get("metrics", {}).get(
                "real_vs_pseudo_floor"
            ),
            "groups": group_rows,
        }
    return {
        "threshold": 0.95,
        "minimum_combined_denominator": min(all_denominators, default=0),
        "maximum_combined_denominator": max(all_denominators, default=0),
        "groups_requiring_zero_errors": sum(
            group["zero_error_required"]
            for run in output.values()
            for group in run["groups"].values()
        ),
        "total_groups": sum(len(run["groups"]) for run in output.values()),
        "runs": output,
    }


def build_audit(
    closure_report: dict[str, Any], decomposition: dict[str, Any]
) -> dict[str, Any]:
    classified = [classify_run(run) for run in closure_report.get("runs", [])]
    analyzer_keys = sorted(
        {
            key
            for run in classified
            for key in run.get("gate_map", {}).keys()
        }
    )
    contract_keys = list(CONTRACT_GATE_KEYS)
    additional = sorted(set(analyzer_keys) - set(contract_keys))
    missing = sorted(set(contract_keys) - set(analyzer_keys))
    inconsistencies = [
        {
            "run_id": run["run_id"],
            "reported_scientific_pass": run["reported_scientific_pass"],
            "contract_seven_gate_pass": run["contract_seven_gate_pass"],
            "analyzer_nine_gate_pass": run["analyzer_nine_gate_pass"],
            "reported_verdict": run["reported_verdict"],
            "verdict_matches_analyzer": run["verdict_matches_analyzer"],
        }
        for run in classified
        if (
            not run["reported_pass_matches_contract"]
            or not run["verdict_matches_analyzer"]
            or run["contract_seven_gate_pass"] != run["analyzer_nine_gate_pass"]
        )
    ]
    return {
        "schema_id": SCHEMA_ID,
        "source_campaign_id": closure_report.get("campaign_id"),
        "source_contract_id": closure_report.get("contract_id"),
        "source_commit": closure_report.get("source_commit"),
        "claim_ceiling": "DERIVED_THEORY_TO_GATE_AUDIT_ONLY",
        "official_closure_status_unchanged": closure_report.get("status"),
        "null_baselines": ["silent_carrier_off", "scramble_unshared_schedule"],
        "gate_namespace": {
            "contract_gate_count": len(contract_keys),
            "contract_gate_keys": contract_keys,
            "analyzer_gate_count": len(analyzer_keys),
            "analyzer_gate_keys": analyzer_keys,
            "analyzer_additional_keys": additional,
            "contract_keys_missing_from_analyzer": missing,
        },
        "route_summary": route_summary(classified),
        "run_reconciliation": classified,
        "pass_namespace_inconsistencies": inconsistencies,
        "finite_sample_geometry": finite_sample_report(decomposition),
        "semantic_layers": {
            "protocol_integrity": list(PROTOCOL_INTEGRITY_KEYS),
            "mode_transport": list(MODE_TRANSPORT_KEYS),
            "phase_transport": list(PHASE_TRANSPORT_KEYS),
            "shared_schedule_specificity": list(
                SHARED_SCHEDULE_SPECIFICITY_KEYS
            ),
            "canonical_basis_fidelity": list(CANONICAL_BASIS_FIDELITY_KEYS),
            "metadata_leakage_sanity": list(METADATA_LEAKAGE_KEYS),
        },
        "interpretive_limits": [
            "Layer counts are derived diagnostics and do not replace the frozen closure verdict.",
            "The audit does not identify a physical transfer operator.",
            "The audit does not establish physical HoloGeometry or restoration.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("closure_report", type=Path)
    parser.add_argument("official_gate_decomposition", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audit = build_audit(
        load_json(args.closure_report),
        load_json(args.official_gate_decomposition),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit["route_summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
