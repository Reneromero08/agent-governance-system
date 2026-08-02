from __future__ import annotations

from fractions import Fraction
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
    FIBER_POINTS,
    Polynomial,
    TARGET,
    brute_force_sat_count,
    first_primes,
    formula_weight,
    groebner_fiber_parameterization,
    jacobian_determinant,
    map_point,
    modular_signature,
    modular_sieve_is_nonzero,
    reference_fiber_trace,
    sheet_selectors,
    verify_exact_reference,
)


class BridgeModelTests(unittest.TestCase):
    def test_normalized_map_has_unit_jacobian(self) -> None:
        self.assertEqual(jacobian_determinant(), Polynomial.constant(1))

    def test_exact_three_point_fiber_certificate(self) -> None:
        points = tuple(
            groebner_fiber_parameterization(sheet) for sheet in (-1, 0, 1)
        )
        self.assertEqual(points, FIBER_POINTS)
        self.assertTrue(all(map_point(point) == TARGET for point in FIBER_POINTS))

    def test_sheet_selectors_are_primitive_idempotents(self) -> None:
        expected = {
            -1: (Fraction(1), Fraction(0), Fraction(0)),
            0: (Fraction(0), Fraction(1), Fraction(0)),
            1: (Fraction(0), Fraction(0), Fraction(1)),
        }
        for sheet, row in expected.items():
            selectors = sheet_selectors(sheet)
            self.assertEqual(selectors, row)
            self.assertEqual(sum(selectors), 1)
            self.assertTrue(all(value * value == value for value in selectors))

    def test_weighted_fiber_trace_matches_sat_count(self) -> None:
        examples = (
            ((((0, True), (0, True), (0, True)),), 1),
            (
                (
                    ((0, True), (1, True), (1, True)),
                    ((0, False), (1, True), (1, True)),
                ),
                2,
            ),
            (
                (
                    ((0, True), (1, True), (2, True)),
                    ((0, False), (1, False), (2, True)),
                    ((0, True), (1, False), (2, False)),
                ),
                3,
            ),
            (
                (
                    ((0, True), (0, True), (0, True)),
                    ((0, False), (0, False), (0, False)),
                ),
                1,
            ),
        )
        for formula, variable_count in examples:
            self.assertEqual(
                reference_fiber_trace(formula, variable_count),
                brute_force_sat_count(formula, variable_count),
            )

    def test_null_sheets_never_contribute(self) -> None:
        formula = (((0, True), (1, False), (1, False)),)
        for sheets in product((-1, 0, 1), repeat=2):
            if -1 in sheets:
                self.assertEqual(formula_weight(formula, sheets, 2), 0)

    def test_prime_sieve_is_total_on_reference_range(self) -> None:
        for variable_count in range(0, 10):
            self.assertEqual(len(first_primes(variable_count + 1)), variable_count + 1)
            self.assertFalse(
                modular_sieve_is_nonzero(modular_signature(0, variable_count))
            )
            for value in range(1, 2**variable_count + 1):
                self.assertTrue(
                    modular_sieve_is_nonzero(
                        modular_signature(value, variable_count)
                    )
                )

    def test_claim_ceiling_is_preserved(self) -> None:
        result = verify_exact_reference()
        for key in (
            "jacobian_unit",
            "three_fiber_points",
            "fiber_parameterization",
            "sheet_idempotents",
            "fiber_trace_matches",
            "prime_sieve_zero",
            "prime_sieve_positive",
        ):
            self.assertTrue(result[key])
        self.assertEqual(
            result["claim_ceiling"],
            "NATIVE_CATALYTIC_FIBER_PUSHFORWARD_NOT_ESTABLISHED",
        )


if __name__ == "__main__":
    unittest.main()
