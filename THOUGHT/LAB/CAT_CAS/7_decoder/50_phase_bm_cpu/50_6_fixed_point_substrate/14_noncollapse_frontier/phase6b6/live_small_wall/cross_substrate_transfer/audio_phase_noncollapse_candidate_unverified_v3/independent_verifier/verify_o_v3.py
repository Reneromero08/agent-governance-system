#!/usr/bin/env python3
"""Independent V3 reconstruction for Candidate O."""

from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "raw_outputs" / "source_reproduction_v3" / "candidate_o_run1"
RAW = ROOT / "raw_outputs" / "independent_o_v3"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def check_o() -> dict:
    summary = read_json(SRC / "summary.json")
    oracle = read_json(SRC / "oracle_summary.json")
    cases = []
    pattern_ok = True
    baseline_dominates = True
    for index, case in enumerate(summary["tested_cases"]):
        edges = case["edges"]
        nodes = case["nodes"]
        k = int(math.log2(edges)) if edges > 0 else 0
        expected_slots = k + 2
        expected_pebble_cells = 272 * expected_slots
        expected_pebble_apps = 3**k
        recorded_slots = summary["reversible_pebble_message_slots"][index]
        recorded_pebble_cells = summary["reversible_pebble_message_integer_cells"][index]
        recorded_pebble_apps = summary["pebble_forward_step_applications"][index]
        recorded_stream_cells = summary["strongest_compact_classical_message_integer_cells"][index]
        record = {
            "nodes": nodes,
            "edges": edges,
            "log2_edges": k,
            "derived_reversible_slots": expected_slots,
            "recorded_reversible_slots": recorded_slots,
            "derived_reversible_integer_cells": expected_pebble_cells,
            "recorded_reversible_integer_cells": recorded_pebble_cells,
            "derived_pebble_forward_applications": expected_pebble_apps,
            "recorded_pebble_forward_applications": recorded_pebble_apps,
            "two_message_streaming_cells": recorded_stream_cells,
            "streaming_cells_le_pebble_cells": recorded_stream_cells <= recorded_pebble_cells,
            "streaming_work_less_for_nodes_ge_3": nodes < 3 or (nodes - 1) < recorded_pebble_apps,
        }
        pattern_ok = pattern_ok and expected_slots == recorded_slots and expected_pebble_cells == recorded_pebble_cells and expected_pebble_apps == recorded_pebble_apps
        baseline_dominates = baseline_dominates and record["streaming_cells_le_pebble_cells"] and record["streaming_work_less_for_nodes_ge_3"]
        cases.append(record)
    exact_closure = (
        summary["result"] == "PASS"
        and oracle["result"] == "PASS"
        and summary["all_primary_restored_exactly"]
        and summary["all_reuse_restored_exactly"]
        and summary["all_fresh_restored_reuse_boundary_equal"]
        and oracle["all_two_message_boundaries_equal"]
        and oracle["all_retain_all_boundaries_equal"]
        and oracle["small_direct_enumerations_equal"]
        and pattern_ok
    )
    return {
        "candidate": "O",
        "independent_method": "resource_recurrence_reconstruction_plus_source_oracle_boundary_parity; no independent exact transfer DP implementation",
        "case_checks": cases,
        "exact_family_closure_survives": exact_closure,
        "baseline_dominates_reversible_pebbling": baseline_dominates,
        "oracle_checks": {
            "production_compiler_called": oracle["production_compiler_called"],
            "production_transfer_called": oracle["production_transfer_called"],
            "production_module_imported": oracle["production_module_imported"],
            "all_two_message_boundaries_equal": oracle["all_two_message_boundaries_equal"],
            "small_direct_enumerations_equal": oracle["small_direct_enumerations_equal"],
            "direct_enumeration_nodes": oracle["direct_enumeration_nodes"],
        },
        "controls": summary["controls"],
        "payload_growth": {
            "maximum_single_coefficient_signed_bits": summary["maximum_single_coefficient_signed_bits"],
            "maximum_message_integer_payload_bits": summary["maximum_message_integer_payload_bits"],
            "integer_payload_width_fixed_in_chain_depth": summary["measured_repair"]["integer_payload_width_fixed_in_chain_depth"],
        },
        "baseline_challenge": summary["matched_compact_classical"],
        "scope_caveat": "exact F17 public path family at nodes 2,3,5,9,17,33,65; not arbitrary graph/topology, not stronger than two-message streaming DP.",
        "missing_independent_controls": [
            "independent transfer message dynamic program",
            "independent retain-all dynamic program",
            "independent two-message streaming dynamic program",
            "direct enumeration beyond source/oracle booleans",
            "non-power-of-two size mutations outside source set",
            "changed local factor periods",
            "changed cubic interactions",
            "corrupted topology schedule controls",
            "block-transfer powering",
        ],
        "classification": "SOURCE_REPRODUCED_FAMILY_SCOPED_TRANSFER_CLOSURE" if exact_closure else "REJECTED_SOURCE_DEFECT",
    }


def render(report: dict) -> str:
    rows = "\n".join(
        f"- nodes={case['nodes']}: pebble_slots={case['recorded_reversible_slots']}, pebble_cells={case['recorded_reversible_integer_cells']}, pebble_apps={case['recorded_pebble_forward_applications']}, two_message_cells={case['two_message_streaming_cells']}"
        for case in report["case_checks"]
    )
    return f"""# Cubic Chain Transfer Report

Candidate O: F17 cubic-chain reversible transfer mechanism.

Exact family closure survived: `{report['exact_family_closure_survives']}`
Classification: `{report['classification']}`

V3 reconstruction actually performed:

{rows}

Finding:

The exact topology-factorized path transfer appears to close for the declared F17 cubic-chain family and removes the explicit 17^k assignment trace for tested nodes. In V3, exact boundary and restoration parity remain source/oracle-supported rather than independently reimplemented.

Baseline discipline:

The reversible pebble schedule is not the strongest compact method. The identical exact two-message path dynamic program uses 544 integer cells and less transfer work for nodes at least 3. Integer payload width also grows with depth. Therefore this is source-reproduced family-scoped transfer closure, not an independently verified transferable reversible-pebble law or Small Wall evidence.
"""


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    report = check_o()
    (RAW / "o_independent_data.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (ROOT / "CUBIC_CHAIN_TRANSFER_REPORT.md").write_text(render(report), encoding="utf-8")
    print(json.dumps({"O": report["classification"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
