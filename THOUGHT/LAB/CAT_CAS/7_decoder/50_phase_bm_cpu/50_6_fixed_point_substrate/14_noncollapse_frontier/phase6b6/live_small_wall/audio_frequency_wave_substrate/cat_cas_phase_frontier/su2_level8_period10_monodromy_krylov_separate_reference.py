#!/usr/bin/env python3
"""Package the separate M217 recurrence reconstruction.

The compiled reference imports no M217 production source and uses distinct
split primes.  This wrapper independently reduces the exact M214 period-10
vacuum boundary at those primes to bind the separately reconstructed local
operator to the exact Q(zeta_40) substrate.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import su2_level8_fusion_path_braid_phase_relation as braid


PRIMES = (641, 881)
EXPECTED_DIMENSIONS = (2, 5, 14, 42, 132, 429, 1430)


def primitive_root(prime: int) -> int:
    residual = prime - 1
    factors: list[int] = []
    divisor = 2
    while divisor * divisor <= residual:
        if residual % divisor == 0:
            factors.append(divisor)
            while residual % divisor == 0:
                residual //= divisor
        divisor += 1
    if residual > 1:
        factors.append(residual)
    for candidate in range(2, prime):
        if all(
            pow(candidate, (prime - 1) // factor, prime) != 1
            for factor in factors
        ):
            return candidate
    raise RuntimeError("separate primitive-root search failed")


def root40(prime: int) -> int:
    root = pow(primitive_root(prime), (prime - 1) // 40, prime)
    if (
        pow(root, 40, prime) != 1
        or pow(root, 20, prime) == 1
        or pow(root, 8, prime) == 1
    ):
        raise RuntimeError("separate root lacks exact order 40")
    return root


def evaluate(value: braid.K, prime: int) -> int:
    root = root40(prime)
    result = 0
    for power_index, coefficient in enumerate(value.coefficients):
        denominator = coefficient.denominator % prime
        if denominator == 0:
            raise ZeroDivisionError("separate prime divides exact denominator")
        result += (
            coefficient.numerator
            * pow(denominator, -1, prime)
            * pow(root, power_index, prime)
        )
    return result % prime


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(
            "usage: su2_level8_period10_monodromy_krylov_separate_reference.py "
            "COMPILED_REFERENCE"
        )
    helper = Path(sys.argv[1]).resolve()
    if not helper.is_file() or not os.access(helper, os.X_OK):
        raise ValueError("compiled separate M217 reference is unavailable")
    if str(helper).startswith(("/dev/shm/", "/run/shm/")):
        raise ValueError("RAM-backed M217 reference is forbidden")
    completed = subprocess.run(
        [str(helper)], check=True, capture_output=True, text=True
    )
    if completed.stderr:
        raise RuntimeError("separate M217 reference emitted stderr")
    reference = json.loads(completed.stdout)
    if reference.get("reference_imports_m217_production") is not False:
        raise RuntimeError("separate M217 import boundary changed")
    dimensions = [
        case["fusion_path_cells"]
        for case in reference["cases"]
        if case["family"] == 0
    ]
    if tuple(dimensions) != EXPECTED_DIMENSIONS:
        raise RuntimeError("separate dimension law changed")
    if not reference["all_cases_full_at_both_distinct_split_primes"]:
        raise RuntimeError("separate full-degree reconstruction failed")

    program = braid.BraidProgram(16, 10, 0)
    topology, state, work = braid.execute_forward(program)
    boundary = state[topology.rank(braid.vacuum_path(16))]
    exact = {str(prime): evaluate(boundary, prime) for prime in PRIMES}
    primary = next(
        case
        for case in reference["cases"]
        if case["strands"] == 16 and case["family"] == 0
    )
    reconstructed = {
        str(item["prime"]): item["first_terms"][1]
        for item in primary["prime_results"]
    }
    if exact != reconstructed:
        raise RuntimeError("separate modular operator differs from exact boundary")

    here = Path(__file__).resolve().parent
    core_source = here / "su2_level8_period10_monodromy_krylov_separate_reference.cpp"
    substrate_source = here / "su2_level8_fusion_path_braid_phase_relation.py"
    result = {
        "schema": "cat_cas.su2_level8_period10_monodromy_krylov_separate_reference.v1",
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "SEPARATE_REFERENCE_PARITY",
        "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "reference_imports_m217_production": False,
        "reference_algorithm": reference["reference_algorithm"],
        "distinct_split_primes": list(PRIMES),
        "all_cases_full_at_both_distinct_split_primes": True,
        "cases": reference["cases"],
        "exact_primary_boundary_parity": {
            "boundary_commitment": braid.boundary_commitment(boundary),
            "exact_modular_boundaries": exact,
            "separate_reference_first_period_boundaries": reconstructed,
            "agreement": True,
        },
        "exact_primary_boundary_verification_work": work.as_dict(),
        "source_dependencies": {
            "m214_production_sha256": hashlib.sha256(
                substrate_source.read_bytes()
            ).hexdigest(),
            "separate_reference_core_sha256": hashlib.sha256(
                core_source.read_bytes()
            ).hexdigest(),
        },
        "claim_limits": {
            "declared_period10_two_family_scope_only": True,
            "arbitrary_braid_program_lower_bound": False,
            "distinct_phase_resource_established": False,
            "computational_advantage": False,
            "small_wall_crossed": False,
            "physical_waveform_execution": False,
            "unbounded_computation_established": False,
        },
        "terminal": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
