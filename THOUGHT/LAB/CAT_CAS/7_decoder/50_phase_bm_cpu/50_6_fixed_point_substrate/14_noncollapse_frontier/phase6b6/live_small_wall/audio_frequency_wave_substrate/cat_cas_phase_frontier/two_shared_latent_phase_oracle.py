#!/usr/bin/env python3
"""Independent algebra oracle for the two-shared-latent-port successor."""

from __future__ import annotations

import cmath
import json
import math
import sys
from pathlib import Path


TOLERANCE = 2.0e-13


def fail(message: str) -> None:
    raise RuntimeError(message)


def determinant(state: list[complex]) -> complex:
    return state[0] * state[3] - state[1] * state[2]


def distance(left: list[complex], right: list[complex]) -> float:
    return math.sqrt(
        sum(abs(a - b) ** 2 for a, b in zip(left, right, strict=True))
    )


def controlled_phase(
    state: list[complex],
    angle: float,
) -> list[complex]:
    result = list(state)
    result[3] *= cmath.exp(1j * angle)
    return result


def rotate_a_x(
    state: list[complex],
    angle: float,
) -> list[complex]:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    result = list(state)
    for upper, lower in ((0, 2), (1, 3)):
        result[upper] = (
            cosine * state[upper] + 1j * sine * state[lower]
        )
        result[lower] = (
            1j * sine * state[upper] + cosine * state[lower]
        )
    return result


def main() -> int:
    if len(sys.argv) != 2:
        fail("usage: two_shared_latent_phase_oracle.py DIRECT_RESULT")
    production = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))

    scale = 1.0 / math.sqrt(2.0)
    port_a = [scale, cmath.rect(scale, 2.0 * math.pi / 17.0)]
    port_b = [scale, cmath.rect(scale, 4.0 * math.pi / 17.0)]
    product = [
        port_a[a] * port_b[b]
        for a in range(2)
        for b in range(2)
    ]
    theta = 2.0 * math.pi * 5.0 / 17.0
    phi = 2.0 * math.pi * 3.0 / 17.0
    entangled = controlled_phase(product, theta)
    restored = controlled_phase(entangled, -theta)

    diagonal = [1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j, cmath.exp(1j * theta)]
    factorization_residual = abs(
        diagonal[0] * diagonal[3] - diagonal[1] * diagonal[2]
    )
    product_then_joint = controlled_phase(
        rotate_a_x(product, phi), theta
    )
    joint_then_product = rotate_a_x(
        controlled_phase(product, theta), phi
    )
    noncommutation_distance = distance(
        product_then_joint, joint_then_product
    )

    product_determinant = abs(determinant(product))
    entangled_determinant = abs(determinant(entangled))
    restoration_error = distance(product, restored)
    production_probe = production.get("nonseparability_probe", {})
    checks = {
        "production_passed": production.get("result") == "PASS",
        "production_joint_cells": (
            production.get("resident_joint_complex_cells") == 1140
        ),
        "production_two_ports": (
            production.get("shared_latent_port_count") == 2
        ),
        "production_probe_nonzero": (
            float(
                production_probe.get(
                    "post_joint_maximum_fiber_determinant", 0.0
                )
            )
            > 1.0e-8
        ),
        "product_determinant_zero": product_determinant < TOLERANCE,
        "joint_determinant_nonzero": entangled_determinant > 1.0e-3,
        "controlled_phase_nonfactorable": (
            factorization_residual > 1.0e-3
        ),
        "local_joint_noncommuting": (
            noncommutation_distance > 1.0e-3
        ),
        "joint_inverse_restores": restoration_error < TOLERANCE,
    }
    if not all(checks.values()):
        fail(f"two-port independent algebra check failed: {checks}")

    result = {
        "oracle": "INDEPENDENT_PYTHON_COMPLEX_TWO_PORT_ALGEBRA",
        "result": "PASS",
        "production_backend_imported": False,
        "production_recurrence_reimplemented": False,
        "grid": 17,
        "tested_joint_residue": 5,
        "tested_local_residue": 3,
        "product_input_determinant": product_determinant,
        "post_joint_determinant": entangled_determinant,
        "controlled_phase_factorization_residual": factorization_residual,
        "local_joint_noncommutation_distance": noncommutation_distance,
        "joint_inverse_restoration_error": restoration_error,
        "checks": checks,
        "claim_ceiling": (
            "INDEPENDENT_ALGEBRA_ONLY_SINGLE_PRODUCT_FIBER_"
            "GRID17_CONTROLLED_PHASE_AND_LOCAL_X_COMPLEX128"
        ),
        "terminal": False,
    }
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
