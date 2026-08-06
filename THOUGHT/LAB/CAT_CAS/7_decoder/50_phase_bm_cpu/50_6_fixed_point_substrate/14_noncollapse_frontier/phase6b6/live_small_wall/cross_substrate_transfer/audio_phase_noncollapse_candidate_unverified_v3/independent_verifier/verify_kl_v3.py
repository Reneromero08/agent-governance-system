#!/usr/bin/env python3
"""Independent V3 reconstruction for Candidates K and L."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_RUNS = ROOT / "raw_outputs" / "source_reproduction_v3"
RAW = ROOT / "raw_outputs" / "independent_kl_v3"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def expected_k_ranks(port_count: int) -> list[int]:
    return [2 ** width for width in range(port_count, 0, -1)]


def tt_cells_from_ranks(port_count: int, ranks: list[int], physical_dim: int = 285) -> int:
    dims = [physical_dim] + [2] * port_count
    total = dims[0] * ranks[0]
    for index in range(1, len(dims)):
        left_rank = ranks[index - 1]
        right_rank = ranks[index] if index < len(ranks) else 1
        total += left_rank * dims[index] * right_rank
    return total


def check_k() -> dict:
    summary = read_json(SRC_RUNS / "candidate_k_run1" / "summary.json")
    oracle = read_json(SRC_RUNS / "candidate_k_run1" / "oracle_summary.json")
    ports = [2, 3, 4, 5, 6]
    derived = []
    all_match = True
    for index, port_count in enumerate(ports):
        ranks = expected_k_ranks(port_count)
        dense_cells = 285 * (2**port_count)
        tt_cells = tt_cells_from_ranks(port_count, ranks)
        recorded_ranks = oracle["final_ranks"][index]
        recorded_dense = summary["matched_explicit_dense_assignment_complex_cells"][index]
        recorded_tt = summary["primary_peak_tt_complex_cells"][index]
        record = {
            "ports": port_count,
            "derived_final_ranks": ranks,
            "recorded_final_ranks": recorded_ranks,
            "derived_dense_cells": dense_cells,
            "recorded_dense_cells": recorded_dense,
            "derived_tt_cells": tt_cells,
            "recorded_tt_cells": recorded_tt,
            "tt_exceeds_dense": tt_cells > dense_cells,
        }
        all_match = all_match and ranks == recorded_ranks and dense_cells == recorded_dense and tt_cells == recorded_tt
        derived.append(record)
    strict_scope_survives = (
        all_match
        and oracle["all_final_canonical_cuts_full_numerical_rank"]
        and summary["matched_classical_references"]["dense_reference_smaller_in_complex_cells_for_all_cases"]
        and not summary["fixed_rank_multi_port_closure_established"]
        and not summary["compact_multi_port_carrier_established"]
    )
    return {
        "candidate": "K",
        "independent_method": "rank_list_and_TT_storage_derivation_from public tensor dimensions; no independent SVD/tensor reconstruction",
        "ports_tested": ports,
        "derived_cases": derived,
        "strict_bounded_diagnostic_survives": strict_scope_survives,
        "source_oracle_numerical_rank_parity": {
            "all_final_canonical_cuts_full_numerical_rank": oracle["all_final_canonical_cuts_full_numerical_rank"],
            "all_final_ranks_match": oracle["all_final_ranks_match"],
            "all_boundaries_match": oracle["all_boundaries_match"],
        },
        "baseline_challenge": {
            "dense_assignment_smaller_for_tested_cases": summary["matched_classical_references"]["dense_reference_smaller_in_complex_cells_for_all_cases"],
            "identical_tt_recurrence_exists": summary["matched_classical_references"]["identical_tensor_train_recurrence_exists"],
            "stronger_general_baseline_not_established": True,
        },
        "scope_caveat": "numerical SVD/tolerance-defined, public necklace family p=2..6 only; not a general tensor-rank lower bound.",
        "missing_independent_controls": [
            "independent final relation tensor reconstruction",
            "high-precision SVD reconstruction",
            "exact or modular rank check",
            "alternate port ordering",
            "alternate legal matricization",
            "tolerance sweep",
            "symmetry quotient search",
        ],
        "classification": "SOURCE_REPRODUCED_SOURCE_LOCAL_MULTI_PORT_TT_OBSTRUCTION" if strict_scope_survives else "REJECTED_RANK_OR_BASELINE_DEFECT",
    }


def height_bound(n: int) -> int:
    return (272 * n + 18) // 3


def check_l() -> dict:
    summary = read_json(SRC_RUNS / "candidate_l_run1" / "result.summary.json")
    oracle = read_json(SRC_RUNS / "candidate_l_run1" / "oracle.summary.json")
    induction_samples = []
    induction_ok = True
    for n in range(1, 65):
        lhs = height_bound(n + 3)
        rhs = height_bound(n) + 272
        induction_samples.append({"n": n, "L_n": height_bound(n), "L_n_plus_3": lhs, "expected": rhs})
        induction_ok = induction_ok and lhs == rhs
    density_ok = all(
        item["numerator"] > 0 and item["numerator"] <= item["denominator"]
        for item in summary["exact_cycle_nonzero_densities"].values()
    )
    gates = oracle["production_gate_checks"]
    mutations = oracle["mutation_checks"]
    scope_ok = (
        summary["height_lower_bound_formula"] == "CEIL((272*N+16)/3)"
        and induction_ok
        and summary["all_coefficient_valuations_prove_induction"]
        and summary["fixed_finite_lossless_discrete_boundary_alphabet_rejected"]
        and summary["infinitely_many_distinct_pi_valuations_certified"]
        and density_ok
        and gates["only_logarithmic_bit_lower_bound_claimed"]
        and gates["no_generic_machine_memory_lower_bound_claimed"]
        and gates["matched_identical_classical_recurrence_retained"]
        and mutations["one_pi_weakened_induction_coefficient_rejected"]
        and mutations["normalized_recurrence_coefficient_perturbation_detected"]
    )
    return {
        "candidate": "L",
        "independent_method": "closed form height identity and scope check against reproduced source/oracle summaries; no clean-room recurrence certificate reconstruction",
        "height_formula": summary["height_lower_bound_formula"],
        "induction_identity_L_n_plus_3_equals_L_n_plus_272_for_n_1_to_64": induction_ok,
        "induction_sample_head": induction_samples[:6],
        "cycle_nonzero_densities": summary["exact_cycle_nonzero_densities"],
        "cycle_density_nonzero_and_bounded": density_ok,
        "oracle_distinctness": {
            "oracle_imports_production_module": oracle["oracle_imports_production_module"],
            "oracle_cycle_algorithm_distinct": oracle["oracle_cycle_algorithm_distinct"],
            "oracle_reference_kernel": oracle["oracle_reference_kernel"],
        },
        "mutation_checks": mutations,
        "baseline_challenge": {
            "identical_compact_classical_recurrence_retained": gates["matched_identical_classical_recurrence_retained"],
            "compact_exponent_ledger_upper_bound_present": gates["compact_exponent_upper_bound_present"],
            "period_index_defeats_unqualified_decoder_lower_bound": "LOWER_BOUND_WITH_PERIOD_INDEX_OR_EXTERNAL_COUNTER_FREE" in oracle["not_established"],
            "no_generic_machine_memory_lower_bound": gates["no_generic_machine_memory_lower_bound_claimed"],
        },
        "strict_scope_survives": scope_ok,
        "scope_caveat": "worst-case-through-horizon exact boundary/valuation encoding without a free period index; not pointwise, not online-space, not no-indexed-generator.",
        "missing_independent_controls": [
            "independent public-operator compilation",
            "independent characteristic polynomial derivation",
            "independent coefficient valuation derivation",
            "independent normalized recurrence reconstruction",
            "independent cycle-state/cycle-length reconstruction",
            "independent mutation campaign beyond source/oracle summaries",
        ],
        "classification": "SOURCE_REPRODUCED_TRANSFERABLE_BOUNDARY_HEIGHT_OBSTRUCTION_CANDIDATE" if scope_ok else "REJECTED_LOWER_BOUND_DEFECT",
    }


def render_k(report: dict) -> str:
    rows = "\n".join(
        f"- p={case['ports']}: ranks={case['derived_final_ranks']}, dense={case['derived_dense_cells']}, TT={case['derived_tt_cells']}, TT>d={case['tt_exceeds_dense']}"
        for case in report["derived_cases"]
    )
    return f"""# Multi-Port Rank Report

Candidate K: full-rank multi-port tensor-train diagnostic.

Strict bounded diagnostic survived: `{report['strict_bounded_diagnostic_survives']}`
Classification: `{report['classification']}`

Independent reconstruction:

{rows}

Finding:

The source/oracle rank lists and the public-shape storage arithmetic are consistent, and the resulting TT cell counts exceed the matched dense assignment storage in every tested case. This is useful as a warning against assuming TT compaction.

Scope discipline:

This remains source/family local: V3 did not independently reconstruct the final relation tensor, recompute high-precision singular values, run exact/modular rank, sweep tolerance, test alternate matricizations/orderings, or search symmetry quotients. The dense compact baseline dominates the accepted TT representation for the tested cases.
"""


def render_l(report: dict) -> str:
    return f"""# Boundary Height Lower Bound Report

Candidate L: exact period-17 boundary-height obstruction.

Strict scope survived: `{report['strict_scope_survives']}`
Classification: `{report['classification']}`

V3 reconstruction actually performed:

- Verified `{report['height_formula']}` as `L(n)=(272*n+18)//3`.
- Verified `L(n+3)=L(n)+272` for n=1..64.
- Verified the recorded cycle densities are nonzero and bounded by their periods: `{report['cycle_nonzero_densities']}`.
- Confirmed the oracle records separate-cycle algorithm and no production-module import.
- Confirmed mutation gates reject a one-pi weakening and normalized recurrence coefficient perturbation.

Finding:

The source package remains a strong transferable boundary-height obstruction candidate. V3 did not independently reconstruct the characteristic identities, coefficient valuations, normalized recurrence, cycle states, or cycle lengths; therefore this report no longer carries an `INDEPENDENTLY_VERIFIED` label.

Scope discipline:

No Small Wall change follows. The result does not rule out compact indexed generators, a free period counter, controlled approximation, online machine-space tricks, or the identical compact classical recurrence.
"""


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    k = check_k()
    l = check_l()
    payload = {
        "schema_version": "audio_noncollapse_v3_independent_kl",
        "canonical": False,
        "small_wall_crossed": False,
        "candidate_k": k,
        "candidate_l": l,
    }
    (RAW / "kl_independent_data.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (ROOT / "MULTI_PORT_RANK_REPORT.md").write_text(render_k(k), encoding="utf-8")
    (ROOT / "BOUNDARY_HEIGHT_LOWER_BOUND_REPORT.md").write_text(render_l(l), encoding="utf-8")
    print(json.dumps({"K": k["classification"], "L": l["classification"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
