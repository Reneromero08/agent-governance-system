#!/usr/bin/env python3
"""Fail-closed qualifier for the bounded M260 growing QND/bond diagnostic."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path


PACKAGE = Path(__file__).resolve().parents[1]
PRODUCTION = PACKAGE / "growing_even_mode_qnd_bond_diagnostic.py"
REFERENCE = PACKAGE / "tests" / "growing_even_mode_qnd_bond_separate_reference.py"
PRODUCTION_SEAL = PACKAGE / "evidence" / "PHASE_QEMU_V2_GROWING_QND_BOND_DIAGNOSTIC.json"
REFERENCE_SEAL = PACKAGE / "evidence" / "PHASE_QEMU_V2_GROWING_QND_BOND_SEPARATE_REFERENCE.json"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def regenerate(script: Path) -> bytes:
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [sys.executable, "-B", str(script)],
        cwd=PACKAGE,
        env=environment,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    require(result.stderr == b"", f"unexpected stderr from {script.name}")
    return result.stdout


def source_audit() -> None:
    production_text = PRODUCTION.read_text(encoding="utf-8")
    reference_text = REFERENCE.read_text(encoding="utf-8")
    production_tree = ast.parse(production_text)
    reference_tree = ast.parse(reference_text)

    forbidden_modules = {"numpy", "scipy", "sympy", "pickle"}
    for label, tree in (("production", production_tree), ("reference", reference_tree)):
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
            require(
                not (isinstance(node, ast.Constant) and isinstance(node.value, float)),
                f"{label} contains a floating scientific literal",
            )
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                require(node.func.id not in {"float", "eval", "exec"}, f"{label} uses {node.func.id}")
            if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
                require(len(node.elts) < 100, f"{label} contains a large hard-coded literal table")
            if isinstance(node, ast.Dict):
                require(len(node.keys) < 100, f"{label} contains a large hard-coded mapping table")
        require(not imported.intersection(forbidden_modules), f"{label} imports forbidden helpers")

    reference_imports = {
        node.module
        for node in ast.walk(reference_tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    reference_imports.update(
        alias.name
        for node in ast.walk(reference_tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    require(
        not any("growing_even_mode_qnd_bond_diagnostic" in name for name in reference_imports),
        "reference imports production",
    )
    require("DIRECT_BINOMIAL_CREATION_OPERATOR_SUBSTITUTION" in reference_text, "reference law missing")
    require("fwht(row)" in production_text, "production exact FWHT path missing")
    require("generated_fwht_table_integer_cells" in production_text, "FWHT table accounting missing")
    require("same_backing_reuse_established" in production_text, "same-backing ceiling missing")


def primary_parity(production: dict, reference: dict) -> None:
    expected = {
        4: {
            "dimension": 10,
            "supports": [4, 6, 6, 2, 1],
            "ranks": [1, 2, 2, 2, 1],
            "last": ("0", "1"),
            "central": ("0", "1"),
        },
        6: {
            "dimension": 56,
            "supports": [8, 28, 28, 50, 54],
            "ranks": [2, 8, 8, 8, 8],
            "last": ("95/128", "33/128"),
            "central": ("125/256", "131/256"),
        },
        8: {
            "dimension": 330,
            "supports": [16, 112, 112, 164, 292],
            "ranks": [1, 16, 16, 16, 18],
            "last": ("767/1024", "257/1024"),
            "central": ("273/512", "239/512"),
        },
    }
    by_n = {case["modes"]: case for case in production["primary_family"]}
    require(set(by_n) == set(expected), "production fixture set changed")

    for modes, law in expected.items():
        case = by_n[modes]
        oracle = reference["fixtures"][str(modes)]
        supports = [stage["support"] for stage in case["stages"]]
        ranks = [stage["central_schmidt_rank"] for stage in case["stages"]]
        oracle_supports = [stage["support_cells"] for stage in oracle["central_schmidt_trace"]]
        oracle_ranks = [
            stage["central_schmidt_rank"]["exact_rational_rank"]
            for stage in oracle["central_schmidt_trace"]
        ]
        require(case["sector_dimension"] == law["dimension"] == oracle["dimension"], "dimension mismatch")
        require(supports == law["supports"] == oracle_supports, f"support mismatch at n={modes}")
        require(ranks == law["ranks"] == oracle_ranks, f"rank mismatch at n={modes}")
        require(max(ranks) == oracle["peak_central_schmidt_rank"], "peak rank mismatch")
        last = case["final"]["last_mode_parity"]
        central = case["final"]["central_half_parity"]
        require((last["even_weight"], last["odd_weight"]) == law["last"], "last parity mismatch")
        require(
            law["last"]
            == (oracle["last_mode_parity"]["even_weight"], oracle["last_mode_parity"]["odd_weight"]),
            "reference last parity mismatch",
        )
        require((central["even_weight"], central["odd_weight"]) == law["central"], "central parity mismatch")
        require(
            law["central"]
            == (
                oracle["central_half_parity"]["even_weight"],
                oracle["central_half_parity"]["odd_weight"],
            ),
            "reference central parity mismatch",
        )
        require(case["controls"]["public_adjoint_exact_restore"], "production restore failed")
        require(oracle["restored_exactly"], "reference restore failed")
        require(case["controls"]["wrong_kerr_inverse_rejected"], "wrong inverse control failed")
        reuse = case["controls"]["descriptor_distinct_reuse"]
        require(reuse["consumes_returned_restored_value"], "restored value was not reused")
        require(reuse["public_adjoint_exact_restore"], "reuse did not restore")
        require(not reuse["same_backing_reuse_established"], "same-backing claim leaked")

    require(by_n[4]["final"]["last_mode_parity"]["factorized"], "n4 QND control lost")
    require(not by_n[6]["final"]["last_mode_parity"]["factorized"], "n6 unexpectedly factorized")
    require(not by_n[8]["final"]["last_mode_parity"]["factorized"], "n8 unexpectedly factorized")


def exhaustive_parity(production: dict, reference: dict) -> None:
    searches = {entry["word_form"]: entry for entry in production["n6_exhaustive_searches"]}
    require(set(searches) == {"TRANSFER_ABKAB", "ECHO_ABKBdAd"}, "search forms changed")
    for search in searches.values():
        require(search["nonempty_kerr_graphs"] == 32767, "graph count changed")
        require(search["nonempty_proper_parity_selectors"] == 62, "selector count changed")
        require(search["graph_selector_pairs_checked"] == 2031554, "pair count changed")
        sham = search["conserved_total_parity_sham"]
        require(sham["excluded_from_nontrivial_search"], "global parity sham promoted")
        require((sham["even_weight"], sham["odd_weight"]) == ("0", "1"), "global parity changed")
        algorithm = search["algorithm"]
        require(algorithm["generated_fwht_table_is_verifier_only"], "FWHT table scope widened")
        require(algorithm["generated_fwht_table_integer_cells"] == 1835008, "FWHT cells changed")
        require((algorithm["generated_fwht_table_rows"], algorithm["generated_fwht_table_columns"]) == (56, 32768), "FWHT shape changed")
        require(algorithm["fwht_integer_add_subtract_operations"] == 27525120, "FWHT work changed")
        require(algorithm["generated_table_is_not_an_accepted_carrier_or_compiler"], "verifier table promoted")
        require(algorithm["floating_point_decisions"] == 0, "float decision introduced")

    transfer = searches["TRANSFER_ABKAB"]
    echo = searches["ECHO_ABKBdAd"]
    require(transfer["deterministic_proper_selector_pairs"] == 0, "word1 gained a deterministic selector")
    require(transfer["kerr_distinguishing_deterministic_pairs"] == 0, "word1 gained a Kerr boundary")
    require(echo["deterministic_proper_selector_pairs"] == 180162, "word2 raw closure count changed")
    require(echo["graphs_with_any_deterministic_proper_selector"] == 16383, "word2 graph count changed")
    require(echo["kerr_distinguishing_deterministic_pairs"] == 0, "word2 gained a parity flip")

    oracle = reference["exhaustive_n6"]
    require(oracle["searched_selector_scope"] == "ALL_NONEMPTY_PROPER_MODE_SUBSETS", "oracle selector scope changed")
    require(oracle["word1"]["proper_nonempty_deterministic_selector_hits"] == 0, "oracle word1 changed")
    require(oracle["word2"]["proper_nonempty_deterministic_selector_closures"] == 180162, "oracle word2 closures changed")
    require(oracle["word2"]["proper_nonempty_parity_flip_closures"] == 0, "oracle word2 flip changed")
    require(oracle["word1"]["conserved_full_system_selector_hits"] == 32767, "oracle sham changed")
    require(oracle["word2"]["conserved_full_system_selector_hits"] == 32767, "oracle sham changed")


def fixed_core_parity(production: dict, reference: dict) -> None:
    expected = {
        6: {"components": [6], "support": 4, "rank": 4, "connected": True},
        8: {"components": [6, 2], "support": 4, "rank": 2, "connected": False},
    }
    controls = {entry["modes"]: entry for entry in production["fixed_core_controls"]}
    require(set(controls) == set(expected), "fixed-core cases changed")
    for modes, law in expected.items():
        case = controls[modes]
        oracle = reference["fixed_core_disconnected_controls"][str(modes)]
        require(case["geometry_component_sizes"] == law["components"] == oracle["graph_component_sizes"], "component mismatch")
        require(case["geometry_connected"] is law["connected"] is oracle["connected_public_geometry"], "connectedness mismatch")
        require(case["final"]["support"] == law["support"] == oracle["final_support_cells"], "fixed-core support mismatch")
        require(
            case["final"]["central_schmidt_rank"]
            == law["rank"]
            == oracle["final_central_schmidt_rank"]["exact_rational_rank"],
            "fixed-core rank mismatch",
        )
        require(case["all_edges_change_projective_state"], "state ablation failed")
        require(case["all_edges_change_declared_boundary"], "boundary ablation failed")
        require(oracle["both_kerr_edges_change_projective_state_and_boundary"], "oracle ablation failed")
        require(case["public_word_exact_result_free_restore"], "production fixed-core restore failed")
        require(oracle["restored_exactly"], "oracle fixed-core restore failed")
        require(case["bounded_active_core_certificate"] == (modes == 8), "bounded-core classification changed")


def ceilings(production: dict, reference: dict) -> None:
    require(production["restoration_classification"] == "EXACT_ALGEBRAIC_RESTORATION", "restoration class changed")
    require(
        production["restoration_scope"]
        == "FUNCTIONAL_NORMALIZED_EXACT_STATE_EQUALITY_AND_RETURNED_VALUE_REUSE_WITHOUT_SAME_BACKING",
        "restoration scope changed",
    )
    require(production["package_verification_level"] == "PACKAGE_SELF_REVIEW", "package level changed")
    require(
        production["claim_ceiling"]
        == "EXACT_SOFTWARE_FIXED_NUMBER_BOSONIC_ALTERNATING_MATCHING_SINGLE_PI_CROSS_KERR_QND_PARITY_DIAGNOSTIC_AT_N4_N6_N8_ONLY",
        "claim ceiling changed",
    )
    require(
        production["observed_disposition"]["classification"]
        == "DETERMINISTIC_QND_PARITY_CLOSURE_AND_CONNECTED_BOND_GROWTH_DO_NOT_COEXIST_IN_THE_TESTED_PI_CROSS_KERR_ALTERNATING_MATCHING_FAMILY",
        "disposition changed",
    )
    require(production["observed_disposition"]["exact_exhaustive_zero_kerr_distinguishing_deterministic_boundary_hit"], "zero-hit decision failed")
    comparator = production["strongest_honest_comparator"]
    require(not comparator["adaptive_tensor_network_comparator_implemented"], "unimplemented comparator promoted")
    require(not comparator["comparator_optimality_established"], "optimality overclaimed")
    require(not comparator["resource_advantage_comparison_authorized"], "resource comparison promoted")
    resources = production["resource_accounting"]
    require(resources["primary_resident_integer_coefficient_cells_n4_n6_n8"] == [10, 56, 330], "resident cells changed")
    require(resources["exhaustive_n6_generated_fwht_integer_cells_per_word"] == 1835008, "resource table changed")
    require(resources["exhaustive_n6_generated_fwht_tables_sequential_not_simultaneous"], "table lifetime changed")
    require(not resources["matching_transition_cache_and_compiler_entries_instrumented"], "compiler accounting overclaimed")
    require(not resources["whole_process_live_payload_peak_complete"], "whole-process accounting overclaimed")
    require(not resources["physical_energy_noise_precision_bandwidth_and_latency_modeled"], "physical model overclaimed")
    require(resources["resource_verification_level"] == "PACKAGE_SELF_REVIEW", "resource level changed")

    required_ceilings = {
        "software exact-arithmetic diagnostic only",
        "no physical Phase-QEMU execution",
        "no QEMU or CATVM custody enforcement",
        "no same-backing restoration or reuse claim",
        "no phase-native resource advantage",
        "no complexity lower bound",
        "no general cross-Kerr no-go",
        "no Small Wall crossing",
        "no physical-bit replacement",
        "no unbounded catalytic computation",
    }
    require(set(production["claim_ceilings"]) == required_ceilings, "production ceilings changed")
    require(all(value is False for value in reference["claim_limits"].values()), "reference claim limit promoted")
    independence = reference["implementation_independence"]
    require(not independence["imports_production"], "reference production import flag changed")
    require(not independence["imports_phase_qemu_v1"], "reference V1 import flag changed")
    require(not independence["floating_point_scientific_decisions"], "reference float flag changed")
    require(reference["verification_classification"] == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE", "reference classification changed")
    require(reference["verification_level"] == "SEPARATE_REFERENCE_PARITY", "reference level changed")


def main() -> int:
    source_audit()
    fresh_production = regenerate(PRODUCTION)
    fresh_reference = regenerate(REFERENCE)
    require(fresh_production == PRODUCTION_SEAL.read_bytes(), "production seal is stale")
    require(fresh_reference == REFERENCE_SEAL.read_bytes(), "reference seal is stale")
    production = json.loads(fresh_production)
    reference = json.loads(fresh_reference)
    primary_parity(production, reference)
    exhaustive_parity(production, reference)
    fixed_core_parity(production, reference)
    ceilings(production, reference)
    print(
        "PASS_STRICT_SCOPE "
        "M260_GROWING_QND_BOND_DIAGNOSTIC "
        "VERIFICATION=SEPARATE_REFERENCE_PARITY "
        "RESTORATION=FUNCTIONAL_EXACT_NO_SAME_BACKING"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
