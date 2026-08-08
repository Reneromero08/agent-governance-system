#!/usr/bin/env python3
"""Package the exact M217 period-10 SU(2)_8 boundary-rank diagnostic.

The public braid descriptor repeats after ten complete sweeps.  A compiled
C++ helper reconstructs the local A_9 action and obtains scalar vacuum-boundary
Berlekamp--Massey degrees at two primes that split Q(zeta_40).  A full modular
degree certifies a full exact Hankel rank because reduction cannot increase
rank and the fusion-path carrier dimension is the matching upper bound.

The exact Q(zeta_40) carrier transaction remains the M214 phase-relation
substrate.  It projects only the final vacuum boundary, reverses the actual
word on the same coefficient backing, and reuses the restored carrier for an
unrelated word.  This package is a growing-degree obstruction, not a compact
recurrence, CATVM result, phase/classical separation, or Small Wall crossing.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import su2_level8_fusion_path_braid_phase_relation as braid


PRIMARY_STRANDS = 16
PRIMARY_ROUNDS = 10
PRIMARY_FAMILY = 0
REUSE_ROUNDS = 7
REUSE_FAMILY = 1
SPLIT_PRIMES = (241, 401)
EXPECTED_DIMENSIONS = (2, 5, 14, 42, 132, 429, 1430)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def primitive_root(prime: int) -> int:
    residual = prime - 1
    factors = []
    candidate = 2
    while candidate * candidate <= residual:
        if residual % candidate == 0:
            factors.append(candidate)
            while residual % candidate == 0:
                residual //= candidate
        candidate += 1
    if residual > 1:
        factors.append(residual)
    return next(
        value
        for value in range(2, prime)
        if all(pow(value, (prime - 1) // factor, prime) != 1 for factor in factors)
    )


def root40(prime: int) -> int:
    if (prime - 1) % 40:
        raise ValueError("verification prime does not split zeta40")
    root = pow(primitive_root(prime), (prime - 1) // 40, prime)
    if (
        pow(root, 40, prime) != 1
        or pow(root, 20, prime) == 1
        or pow(root, 8, prime) == 1
    ):
        raise RuntimeError("verification root lacks exact order 40")
    return root


def evaluate_exact(value: braid.K, prime: int) -> int:
    root = root40(prime)
    total = 0
    for power_index, coefficient in enumerate(value.coefficients):
        denominator = coefficient.denominator % prime
        if denominator == 0:
            raise ZeroDivisionError("split prime divides exact denominator")
        total += (
            coefficient.numerator
            * pow(denominator, -1, prime)
            * pow(root, power_index, prime)
        )
    return total % prime


def load_core(helper: Path) -> dict[str, Any]:
    if not helper.is_file() or not os.access(helper, os.X_OK):
        raise ValueError("compiled M217 helper is missing or not executable")
    resolved = helper.resolve()
    if str(resolved).startswith(("/dev/shm/", "/run/shm/")):
        raise ValueError("RAM-backed M217 helper is forbidden")
    completed = subprocess.run(
        [str(resolved)],
        check=True,
        capture_output=True,
        text=True,
    )
    if completed.stderr:
        raise RuntimeError("M217 helper emitted unexpected stderr")
    result = json.loads(completed.stdout)
    if result.get("schema") != "cat_cas.su2_level8_period10_monodromy_krylov_core.v1":
        raise RuntimeError("M217 helper schema changed")
    return result


def exact_transactions() -> tuple[dict[str, Any], braid.K]:
    topology = braid.FusionPathTopology.compile(PRIMARY_STRANDS)
    source = braid.source_state(topology)
    carrier = braid.Carrier(braid.OpenFusionPathPort(topology, source.copy()))
    primary_program = braid.BraidProgram(
        PRIMARY_STRANDS, PRIMARY_ROUNDS, PRIMARY_FAMILY
    )
    reuse_program = braid.BraidProgram(PRIMARY_STRANDS, REUSE_ROUNDS, REUSE_FAMILY)

    primary, primary_work = braid.transaction(carrier, source, primary_program)
    reuse, reuse_work = braid.transaction(carrier, source, reuse_program)
    fresh = braid.Carrier(braid.OpenFusionPathPort(topology, source.copy()))
    fresh_reuse, fresh_reuse_work = braid.transaction(fresh, source, reuse_program)

    direct_topology, direct_state, direct_work = braid.execute_forward(primary_program)
    exact_boundary = direct_state[
        direct_topology.rank(braid.vacuum_path(PRIMARY_STRANDS))
    ]
    if braid.boundary_commitment(exact_boundary) != primary["boundary_commitment"]:
        raise RuntimeError("exact diagnostic boundary differs from restored transaction")
    if reuse["boundary_commitment"] != fresh_reuse["boundary_commitment"]:
        raise RuntimeError("restored reuse differs from fresh execution")
    if reuse["forward_state_commitment"] != fresh_reuse["forward_state_commitment"]:
        raise RuntimeError("restored reuse state differs from fresh execution")
    return (
        {
            "primary_program": {
                "strands": PRIMARY_STRANDS,
                "rounds": PRIMARY_ROUNDS,
                "family": PRIMARY_FAMILY,
                "steps": primary_program.steps,
            },
            "primary": primary,
            "primary_full_lifecycle_work": primary_work.as_dict(),
            "reuse_program": {
                "strands": PRIMARY_STRANDS,
                "rounds": REUSE_ROUNDS,
                "family": REUSE_FAMILY,
                "steps": reuse_program.steps,
            },
            "reuse": reuse,
            "reuse_full_lifecycle_work": reuse_work.as_dict(),
            "fresh_reuse": fresh_reuse,
            "fresh_reuse_full_lifecycle_work": fresh_reuse_work.as_dict(),
            "fresh_restored_reuse_boundary_agreement": True,
            "fresh_restored_reuse_state_agreement": True,
            "restoration_generation_after_reuse": carrier.restoration_generation,
            "direct_exact_boundary_verification_work": direct_work.as_dict(),
            "direct_exact_boundary_verification_field_cells": len(direct_state),
            "direct_exact_boundary_verification_payload_bits": braid.field_payload_bits(
                direct_state
            ),
            "baseline_reload_used": False,
        },
        exact_boundary,
    )


def wrong_public_program_inverse_control() -> bool:
    topology = braid.FusionPathTopology.compile(6)
    source = braid.source_state(topology)
    accepted = braid.BraidProgram(6, 2, 0)
    wrong = braid.BraidProgram(6, 2, 1)
    port = braid.OpenFusionPathPort(topology, source.copy())
    work = braid.Work()
    owner = 217001
    port.lease(owner, 1, accepted, work)
    for index in range(accepted.steps):
        port.forward(owner, accepted, index, work)
    rejected = False
    try:
        port.inverse(owner, wrong, accepted.steps - 1, work)
    except ValueError:
        rejected = True
    for index in range(accepted.steps - 1, -1, -1):
        port.inverse(owner, accepted, index, work)
    port.release(owner, accepted, work)
    if port.coefficients != source:
        raise RuntimeError("wrong-program inverse control cleanup did not restore")
    return rejected


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(
            "usage: su2_level8_period10_monodromy_krylov.py COMPILED_CORE"
        )
    here = Path(__file__).resolve().parent
    helper = Path(sys.argv[1])
    core = load_core(helper)
    transaction, exact_boundary = exact_transactions()

    cases = core["cases"]
    dimensions = [
        case["fusion_path_cells"]
        for case in cases
        if case["family"] == 0
    ]
    if tuple(dimensions) != EXPECTED_DIMENSIONS:
        raise RuntimeError("declared fusion-path dimension law changed")
    if not (
        core["period_law_verified"]
        and core["every_case_cross_prime_degree_agreement"]
        and core["every_case_full_scalar_degree"]
        and core["every_semantic_perturbation_changes_prefix"]
    ):
        raise RuntimeError("period-10 full-degree certificate failed")
    primary_core = next(
        case
        for case in cases
        if case["strands"] == PRIMARY_STRANDS
        and case["family"] == PRIMARY_FAMILY
    )
    exact_modular_boundaries = {
        str(prime): evaluate_exact(exact_boundary, prime) for prime in SPLIT_PRIMES
    }
    core_first_period_boundaries = {
        str(item["prime"]): item["first_terms"][1]
        for item in primary_core["prime_results"]
    }
    if exact_modular_boundaries != core_first_period_boundaries:
        raise RuntimeError("modular monodromy differs from exact Q(zeta40) boundary")
    if not all(
        item["undersampled_holdout_violations"] > 0
        for case in cases
        for item in case["prime_results"]
    ):
        raise RuntimeError("undersampled recurrence control did not discriminate")

    controls = braid.custody_controls()
    controls.update(
        {
            "public_round_direction_period2_and_sign_period5_combine_to_period10": True,
            "both_split_roots_have_exact_order40": all(
                pow(root40(prime), 40, prime) == 1
                and pow(root40(prime), 20, prime) != 1
                and pow(root40(prime), 8, prime) != 1
                for prime in SPLIT_PRIMES
            ),
            "all_semantic_last_gate_perturbations_change_boundary_prefix": core[
                "every_semantic_perturbation_changes_prefix"
            ],
            "every_undersampled_recurrence_fails_holdout": True,
            "exact_primary_boundary_reduces_to_both_modular_first_period_boundaries": True,
            "modular_full_hankel_rank_implies_exact_full_hankel_rank": True,
            "compiler_uses_only_public_topology_and_word": True,
            "compiler_reads_exact_final_boundary": False,
            "intermediate_fusion_path_state_projected": False,
            "wrong_public_program_inverse_rejected": (
                wrong_public_program_inverse_control()
            ),
            "snapshot_command_available": False,
        }
    )

    core_source = here / "su2_level8_period10_monodromy_krylov_core.cpp"
    substrate_source = here / "su2_level8_fusion_path_braid_phase_relation.py"
    primary_prime = primary_core["prime_results"][0]
    result = {
        "schema": "cat_cas.su2_level8_period10_monodromy_krylov.v1",
        "result": "PASS_EXACT_SU2_LEVEL8_PERIOD10_MONODROMY_FULL_BOUNDARY_KRYLOV_DEGREE_OBSTRUCTION",
        "claim": "EXACT_SPLIT_PRIME_HANKEL_CERTIFICATES_PROVE_THE_PUBLIC_PERIOD10_SU2_LEVEL8_SWEEP_MONODROMY_VACUUM_BOUNDARY_HAS_FULL_MINIMAL_SCALAR_DEGREES2_5_14_42_132_429_1430_FOR_BOTH_DECLARED_FAMILIES_ACROSS_STRANDS4TO16_SO_NO_COMPACT_FINAL_BOUNDARY_RECURRENCE_EXISTS_FOR_THIS_GROWING_FAMILY_WITH_FINAL_ONLY_BOUNDARY_EXACT_SAME_BACKING_RESTORATION_REUSE_AND_THE_IDENTICAL_CLASSICAL_TRANSFER",
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "SEPARATE_REFERENCE_PARITY",
        "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "claim_ceiling": "FORMAL_SU2_LEVEL8_TEMPERLEY_LIEB_QZETA40_PUBLIC_PERIOD10_SWEEP_MONODROMY_FAMILIES0_1_EVEN_STRANDS4_6_8_10_12_14_16_VACUUM_TO_VACUUM_SCALAR_HANKEL_RANK_AT_SPLIT_PRIMES241_401_PRIMARY16_FAMILY0_REUSE16_ROUNDS7_FAMILY1_DIRECT_PROCESS_ONLY",
        "exact_degree_law": {
            "strands": [4, 6, 8, 10, 12, 14, 16],
            "fusion_path_dimensions": list(EXPECTED_DIMENSIONS),
            "both_family_exact_scalar_minimal_degrees": list(EXPECTED_DIMENSIONS),
            "certificate": "FULL_MODULAR_SCALAR_HANKEL_RANK_GIVES_EXACT_LOWER_BOUND_AND_FUSION_PATH_DIMENSION_GIVES_MATCHING_UPPER_BOUND",
            "split_primes": list(SPLIT_PRIMES),
            "all_cases_full_at_both_primes": True,
            "uniform_fixed_degree_boundary_recurrence_rejected_for_declared_family": True,
            "exact_recurrence_coefficients_lifted": False,
            "arbitrary_braid_program_lower_bound": False,
        },
        "core_diagnostic": core,
        "exact_boundary_modular_parity": {
            "exact_boundary_commitment": transaction["primary"][
                "boundary_commitment"
            ],
            "exact_modular_boundaries": exact_modular_boundaries,
            "core_first_period_boundaries": core_first_period_boundaries,
            "agreement": True,
        },
        "transaction": transaction,
        "controls": controls,
        "resource_law": {
            "primary_exact_carrier_field_cells": transaction["primary"][
                "forward_field_cells"
            ],
            "primary_exact_carrier_payload_bits": transaction["primary"][
                "forward_payload_bits"
            ],
            "primary_modular_state_field_cells": primary_core[
                "fusion_path_cells"
            ],
            "primary_scalar_sequence_field_cells": 2
            * primary_core["fusion_path_cells"]
            + 64,
            "primary_public_recurrence_coefficient_slots": primary_prime[
                "scalar_recurrence_degree"
            ]
            + 1,
            "primary_public_recurrence_nonzero_coefficients": primary_prime[
                "nonzero_recurrence_coefficients"
            ],
            "primary_retained_topology_path_records_for_diagnostic": primary_core[
                "fusion_path_cells"
            ],
            "primary_retained_topology_path_label_cells_for_diagnostic": primary_core[
                "topology_path_label_cells"
            ],
            "primary_structural_shape_records": primary_core[
                "structural_shape_records"
            ],
            "primary_structural_shape_integer_cells": primary_core[
                "structural_shape_integer_cells"
            ],
            "primary_retained_modular_action_records_one_prime": primary_prime[
                "retained_action_records"
            ],
            "primary_retained_modular_action_integer_cells_one_prime": primary_prime[
                "retained_action_integer_cells"
            ],
            "primary_peak_bm_connection_field_cells": primary_prime[
                "peak_bm_connection_cells"
            ],
            "primary_period_applications_per_prime": 2
            * primary_core["fusion_path_cells"]
            + 64
            - 1,
            "primary_local_gate_applications_per_prime": (
                2 * primary_core["fusion_path_cells"] + 64 - 1
            )
            * 10
            * (PRIMARY_STRANDS - 1),
            "diagnostic_primes_execute_sequentially": True,
            "exact_direct_verification_carrier_field_cells": transaction[
                "direct_exact_boundary_verification_field_cells"
            ],
            "exact_direct_verification_carrier_payload_bits": transaction[
                "direct_exact_boundary_verification_payload_bits"
            ],
            "accepted_transaction_retained_inverse_history": 0,
            "controller_backend_traffic_bytes": 0,
            "snapshot_traffic_bytes": 0,
            "baseline_reload_bytes": 0,
            "compiler_and_diagnostic_costs_are_not_excluded_from_comparison": True,
            "excluded_not_zero": "TRANSIENT_CPP_VECTOR_CAPACITY_ALLOCATOR_PROCESS_IMAGE_JSON_SERIALIZATION_HASH_BYTE_TRAFFIC_AND_WHOLE_PROCESS_PEAKS",
        },
        "matched_classical_baselines": {
            "strongest_compact": "IDENTICAL_PUBLIC_PERIOD10_MONODROMY_SCALAR_KRYLOV_AND_FULL1430_CELL_FUSION_PATH_TRANSFER",
            "same_full_degree_and_recurrence_coefficient_law": True,
            "phase_specific_recurrence_reduction": False,
            "computational_advantage": False,
        },
        "claim_limits": {
            "compact_period10_boundary_recurrence_established": False,
            "uniform_fixed_degree_recurrence_for_declared_family": False,
            "arbitrary_braid_program_lower_bound": False,
            "full_state_compaction": False,
            "catvm_custody": False,
            "distinct_phase_resource_established": False,
            "computational_advantage": False,
            "small_wall_crossed": False,
            "physical_waveform_execution": False,
            "physical_bit_replacement": False,
            "catalytic_inference_established": False,
            "unbounded_computation_established": False,
        },
        "source_dependencies": {
            "m214_production_sha256": sha256_file(substrate_source),
            "m217_core_sha256": sha256_file(core_source),
        },
        "terminal": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
