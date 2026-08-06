#!/usr/bin/env python3
"""Summarize V3 strongest baselines and resource accounting."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw_outputs"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    ij = read_json(RAW / "independent_ij_v3" / "ij_independent_data.json")
    kl = read_json(RAW / "independent_kl_v3" / "kl_independent_data.json")
    m = read_json(RAW / "independent_m_v3" / "m_independent_data.json")
    n = read_json(ROOT / "TWO_PORT_CATVM_RUNTIME_ATTACK_REPORT.json")
    o = read_json(RAW / "independent_o_v3" / "o_independent_data.json")
    k_src = read_json(RAW / "source_reproduction_v3" / "candidate_k_run1" / "summary.json")
    l_src = read_json(RAW / "source_reproduction_v3" / "candidate_l_run1" / "result.summary.json")
    o_src = read_json(RAW / "source_reproduction_v3" / "candidate_o_run1" / "summary.json")
    resource = {
        "schema_version": "audio_noncollapse_v3_resource_accounting",
        "canonical": False,
        "small_wall_crossed": False,
        "candidates": {
            "I": {
                "fixed_object_count": "two analytic cells in declared HT theorem",
                "growing_or_unbounded_quantity": "exact projective orbit cardinality is infinite",
                "not_counted_as": ["physical_memory_lower_bound", "continuous_precision_claim"],
                "strongest_baseline": ij["candidate_i"]["strongest_baseline"],
            },
            "J": {
                "semantic_state": "Q3 symbol per register",
                "information_lower_bound_bits_per_register": ij["candidate_j"]["resource_baseline"]["information_lower_bound_bits_per_register"],
                "two_bit_packing_available": True,
                "native_complex_payload_dominated": True,
                "strongest_baseline": "packed Q3 symbolic simulator / possible program-specialized transition table",
            },
            "K": {
                "tt_cells": k_src["primary_peak_tt_complex_cells"],
                "dense_cells": k_src["matched_explicit_dense_assignment_complex_cells"],
                "dense_smaller_all_tested": k_src["matched_classical_references"]["dense_reference_smaller_in_complex_cells_for_all_cases"],
                "temporary_complex_cells": k_src["maximum_temporary_component_complex_cells"],
                "scope": "complex cell accounting, not whole-process RSS",
            },
            "L": {
                "height_formula": l_src["height_lower_bound_formula"],
                "cycle_nonzero_densities": l_src["exact_cycle_nonzero_densities"],
                "compact_exponent_ledger_upper_bound": l_src["compact_exponent_ledger_upper_bound"],
                "not_established": l_src["not_established"],
            },
            "M": {
                "resident_frame": "2x2 complex128 matrix/frame",
                "matched_compact_matrix_recurrence_identical": m["baseline_challenge"]["identical_2x2_holonomy_recurrence"],
                "runtime_advantage_claimed": m["baseline_challenge"]["runtime_advantage_claimed"],
                "closed_form_modules_available": m["baseline_challenge"]["closed_form_fixed_loop_modules_available"],
            },
            "N": {
                "classification": n["classification"],
                "packet_fail_open_cases": n["fail_open_cases"],
                "short_path_source_reproduction_passed_twice": n["short_path_source_reproduction_passed_twice"],
                "long_evidence_root_source_reproduction_failed": n["long_evidence_root_source_reproduction_failed"],
                "transfer_resource_claim_accepted": False,
            },
            "O": {
                "reversible_pebble_slots": o_src["reversible_pebble_message_slots"],
                "reversible_pebble_integer_cells": o_src["reversible_pebble_message_integer_cells"],
                "two_message_streaming_cells": o_src["strongest_compact_classical_message_integer_cells"],
                "maximum_message_integer_payload_bits": o_src["maximum_message_integer_payload_bits"],
                "maximum_single_coefficient_signed_bits": o_src["maximum_single_coefficient_signed_bits"],
                "two_message_baseline_dominates": o["baseline_dominates_reversible_pebbling"],
            },
        },
    }
    baseline_md = """# Strongest Baseline Challenge V3

Status: completed for V3 candidates I–O. No Small Wall position changed.

- I: strongest matched method is the identical exact two-cell recurrence, an indexed exact generator, or controlled approximation with an explicit error law. The surviving obstruction is only against fixed finite exact lossless quotients of the complete orbit.
- J: strongest matched method is a packed Q3 symbolic simulator, possibly with program-specialized tables. This dominates the native complex root-locked VM state in finite software scope.
- K: the direct dense assignment representation is smaller than the TT representation for every tested port count. K is retained only as source-local anti-compaction hygiene.
- L: the identical compact classical recurrence remains. The lower bound is only worst-case-through-horizon exact boundary/valuation code width without a free period index; compact exponent ledgers and indexed generators are not ruled out.
- M: the identical 2×2 matrix recurrence and closed-form fixed-loop modules remain the strongest compact baseline. M is a mechanism law, not a resource separation.
- N: no positive baseline comparison is needed because packet-layer malformed input accepted oversized/concatenated requests and the long evidence path failed before bind. N is rejected as a source defect.
- O: the identical exact two-message path dynamic program uses less storage/work than reversible pebbling for nodes at least three. O is family-scoped transfer closure, not a reversible-pebble resource law.
"""
    (ROOT / "RESOURCE_ACCOUNTING_V3.json").write_text(
        json.dumps(resource, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (ROOT / "STRONGEST_BASELINE_CHALLENGE_V3.md").write_text(
        baseline_md, encoding="utf-8"
    )
    print(json.dumps({"wrote": ["RESOURCE_ACCOUNTING_V3.json", "STRONGEST_BASELINE_CHALLENGE_V3.md"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
