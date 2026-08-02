from __future__ import annotations

from itertools import product
from pathlib import Path
import sys
import unittest

CAT_CAS_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_PARENT = (
    CAT_CAS_ROOT
    / "7_decoder"
    / "50_phase_bm_cpu"
    / "50_6_fixed_point_substrate"
    / "14_noncollapse_frontier"
)
sys.path.insert(0, str(PACKAGE_PARENT))

from jacobian_holonomy_small_wall_bridge_v1.bridge_model import (  # noqa: E402
    brute_force_sat_count,
    first_primes,
    reference_fiber_trace,
)
from jacobian_holonomy_small_wall_bridge_v1.proof_certificate import (  # noqa: E402
    fiber_algebra_certificate,
    formula_circuit_shape,
    logarithmic_residue_is_unit,
    prime_sieve_bound_certificate,
    target_fiber_case_split,
)


class ProofCertificateTests(unittest.TestCase):
    def test_exact_fiber_case_split_certificate_passes(self) -> None:
        certificate = fiber_algebra_certificate()
        self.assertTrue(certificate.passed)
        result = target_fiber_case_split()
        self.assertTrue(result["complete"])
        self.assertEqual(len(result["complete_fiber"]), 3)
        self.assertEqual(
            result["claim_ceiling"],
            "EXACT_FIBER_CERTIFICATE_ONLY__NATIVE_PUSHFORWARD_NOT_ESTABLISHED",
        )

    def test_each_reduced_sheet_has_unit_logarithmic_residue(self) -> None:
        for root in (-1, 0, 1):
            self.assertTrue(logarithmic_residue_is_unit(root))

    def test_formula_weight_has_linear_shared_circuit_ledger(self) -> None:
        previous = 0
        for scale in range(1, 100):
            shape = formula_circuit_shape(scale, 4 * scale)
            self.assertEqual(shape.literal_count, 12 * scale)
            self.assertLessEqual(shape.total_nodes, 40 * scale)
            self.assertGreater(shape.total_nodes, previous)
            previous = shape.total_nodes

    def test_prime_product_certificate_covers_the_full_count_range(self) -> None:
        for variable_count in range(0, 100):
            primes = first_primes(variable_count + 1)
            self.assertTrue(
                prime_sieve_bound_certificate(variable_count, primes)
            )

    def test_exhaustive_two_variable_one_and_two_clause_census(self) -> None:
        literals = (
            (0, False),
            (0, True),
            (1, False),
            (1, True),
        )
        clauses = tuple(product(literals, repeat=3))

        for clause in clauses:
            formula = (clause,)
            self.assertEqual(
                reference_fiber_trace(formula, 2),
                brute_force_sat_count(formula, 2),
            )

        for left_clause in clauses:
            for right_clause in clauses:
                formula = (left_clause, right_clause)
                self.assertEqual(
                    reference_fiber_trace(formula, 2),
                    brute_force_sat_count(formula, 2),
                )


if __name__ == "__main__":
    unittest.main()
