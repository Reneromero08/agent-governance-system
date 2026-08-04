#!/usr/bin/env python3
"""Independent exact oracle for the growing-prime Weil-Gaussian package.

This file deliberately does not import the production implementation.  It
reconstructs the public recurrence with separate tuple/list code, compares the
closed and streamed cocycles along every tested trajectory, and exhaustively
checks the complete q=5 SL(2) chart against dense kernel multiplication.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ORDERS = (5, 11, 23, 29, 41, 53, 83, 89, 113)
CHECKPOINT_DEPTHS = {1, 4, 16, 64, 128, 256, 1024}
CASES = tuple((q, 256, "PRIMARY") for q in ORDERS) + (
    (113, 1024, "PRIMARY"),
    (41, 128, "ALTERNATE"),
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def prime(n: int) -> bool:
    if n < 2:
        return False
    d = 2
    while d * d <= n:
        if n % d == 0:
            return n == d
        d += 1
    return True


def public_field(q: int) -> tuple[int, int, int, int]:
    p = 2 * q + 1
    require(prime(q) and prime(p), "not a safe-prime pair")
    primitive = next(
        g for g in range(2, p) if pow(g, 2, p) != 1 and pow(g, q, p) != 1
    )
    omega = pow(primitive, 2, p)
    require(pow(omega, q, p) == 1, "bad q-th root")
    require(all(pow(omega, k, p) != 1 for k in range(1, q)), "nonprimitive q-th root")
    gauss = sum(pow(omega, x * x % q, p) for x in range(q)) % p
    require(gauss != 0, "vanishing Gauss sum")
    return p, omega, gauss, primitive


def mm(left: tuple[int, ...], right: tuple[int, ...], modulus: int) -> tuple[int, ...]:
    a, b, c, d = left
    e, f, g, h = right
    return (
        (a * e + b * g) % modulus,
        (a * f + b * h) % modulus,
        (c * e + d * g) % modulus,
        (c * f + d * h) % modulus,
    )


def mi(matrix: tuple[int, ...], modulus: int) -> tuple[int, ...]:
    a, b, c, d = matrix
    determinant = (a * d - b * c) % modulus
    require(determinant != 0, "singular matrix")
    scale = pow(determinant, -1, modulus)
    return (d * scale % modulus, -b * scale % modulus, -c * scale % modulus, a * scale % modulus)


def kernel(symplectic: tuple[int, ...], x: int, y: int, q: int, p: int, omega: int) -> int:
    a, b, c, d = symplectic
    if b % q:
        exponent = (d * x * x - 2 * x * y + a * y * y) * pow(2 * b % q, -1, q)
        return pow(omega, exponent % q, p)
    if y % q != d * x % q:
        return 0
    return pow(omega, (c * d * x * x * pow(2, -1, q)) % q, p)


def streamed_cocycle(left: tuple[int, ...], right: tuple[int, ...], field: tuple[int, ...]) -> int:
    q, p, omega, _ = field
    return sum(kernel(left, 0, y, q, p, omega) * kernel(right, y, 0, q, p, omega) for y in range(q)) % p


def closed_cocycle(left: tuple[int, ...], right: tuple[int, ...], field: tuple[int, ...]) -> int:
    q, p, _, gauss = field
    a, b, _, _ = left
    _, f, _, h = right
    if b % q == 0 or f % q == 0:
        return 1
    coefficient = (a * pow(2 * b % q, -1, q) + h * pow(2 * f % q, -1, q)) % q
    if coefficient == 0:
        return q % p
    character = pow(coefficient, (q - 1) // 2, q)
    return gauss if character == 1 else -gauss % p


# A relation is [symplectic-list, fiber-list].  Lists make the backing-identity
# check independent and explicit rather than inferred from a result flag.
def relation(symplectic: tuple[int, ...], fiber: tuple[int, ...]) -> list[list[int]]:
    return [list(symplectic), list(fiber)]


def canonical(item: list[list[int]]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    return tuple(item[0]), tuple(item[1])


def compose(left: list[list[int]], right: list[list[int]], field: tuple[int, ...], audit: dict[str, int]) -> list[list[int]]:
    q, p, _, _ = field
    left_s, left_f = tuple(left[0]), tuple(left[1])
    right_s, right_f = tuple(right[0]), tuple(right[1])
    streamed = streamed_cocycle(left_s, right_s, field)
    closed = closed_cocycle(left_s, right_s, field)
    require(streamed == closed, "streamed/closed trajectory cocycle mismatch")
    audit["trajectory_cocycle_parity_checks"] += 1
    product = mm(left_f, right_f, p)
    return relation(mm(left_s, right_s, q), tuple(closed * value % p for value in product))


def inverse(item: list[list[int]], field: tuple[int, ...], audit: dict[str, int]) -> list[list[int]]:
    q, p, _, _ = field
    inverse_s = mi(tuple(item[0]), q)
    cocycle = closed_cocycle(tuple(item[0]), inverse_s, field)
    require(cocycle == streamed_cocycle(tuple(item[0]), inverse_s, field), "inverse cocycle mismatch")
    audit["trajectory_cocycle_parity_checks"] += 1
    inverse_f = mi(tuple(item[1]), p)
    return relation(inverse_s, tuple(pow(cocycle, -1, p) * value % p for value in inverse_f))


def chirp(item: list[list[int]], row: int, column: int, field: tuple[int, ...]) -> list[list[int]]:
    q = field[0]
    left = (1, 0, 2 * row % q, 1)
    right = (1, 0, 2 * column % q, 1)
    return relation(mm(mm(left, tuple(item[0]), q), right, q), tuple(item[1]))


def stage_descriptors(index: int, family: str, q: int) -> tuple[tuple[Any, ...], ...]:
    require(family in {"PRIMARY", "ALTERNATE"}, "invalid family")
    code = 1 if family == "PRIMARY" else 2
    stages = (
        ("A", "RIGHT", 0, 0),
        ("B", "CHIRP", (3 * index + code) % q, (5 * index + 2 * code + 1) % q),
        ("B", "LEFT", 0, 0),
        ("A", "CHIRP", (7 * index + 3 * code + 1) % q, (11 * index + code + 2) % q),
    )
    return stages if family == "PRIMARY" else tuple(reversed(stages))


def seal(q: int, family: str) -> dict[str, Any]:
    require(family in {"PRIMARY", "ALTERNATE"}, "invalid seed family")
    p, omega, gauss, _ = public_field(q)
    code = 1 if family == "PRIMARY" else 2
    a = relation((1, 1, 1, 2), (1, 2 + code, 3, 7 + 3 * code))
    b = relation((2, 1, 1, 1), (2, 1 + code, 5, 3 + 3 * code))
    mi(tuple(a[1]), p)
    mi(tuple(b[1]), p)
    return {"q": q, "p": p, "omega": omega, "gauss": gauss, "family": family, "a": a, "b": b, "stage": "IDLE", "generation": 0}


def carrier_state(carrier: dict[str, Any]) -> tuple[Any, ...]:
    return (
        carrier["q"], carrier["p"], carrier["family"], canonical(carrier["a"]),
        canonical(carrier["b"]), carrier["stage"],
    )


def backing(carrier: dict[str, Any]) -> tuple[int, ...]:
    return tuple(id(value) for relation_item in (carrier["a"], carrier["b"]) for value in (relation_item, relation_item[0], relation_item[1]))


def overwrite(target: list[list[int]], source: list[list[int]]) -> None:
    target[0][:] = source[0]
    target[1][:] = source[1]


def apply(carrier: dict[str, Any], stage: tuple[Any, ...], audit: dict[str, int]) -> None:
    target_name, operation, row, column = stage
    target = carrier["a"] if target_name == "A" else carrier["b"]
    source = carrier["b"] if target_name == "A" else carrier["a"]
    if operation == "RIGHT":
        updated = compose(target, source, (carrier["q"], carrier["p"], carrier["omega"], carrier["gauss"]), audit)
    elif operation == "LEFT":
        updated = compose(source, target, (carrier["q"], carrier["p"], carrier["omega"], carrier["gauss"]), audit)
    else:
        updated = chirp(target, row, column, (carrier["q"], carrier["p"], carrier["omega"], carrier["gauss"]))
    overwrite(target, updated)


def undo(carrier: dict[str, Any], stage: tuple[Any, ...], audit: dict[str, int], wrong: bool = False) -> None:
    target_name, operation, row, column = stage
    target = carrier["a"] if target_name == "A" else carrier["b"]
    source = carrier["b"] if target_name == "A" else carrier["a"]
    field = (carrier["q"], carrier["p"], carrier["omega"], carrier["gauss"])
    if operation == "CHIRP":
        updated = chirp(target, -row + int(wrong), -column, field)
    else:
        inverse_source = inverse(source, field, audit)
        if wrong:
            inverse_source[1][0] = (inverse_source[1][0] + 1) % carrier["p"]
        updated = compose(target, inverse_source, field, audit) if operation == "RIGHT" else compose(inverse_source, target, field, audit)
    overwrite(target, updated)


def point(carrier: dict[str, Any], depth: int) -> dict[str, int]:
    q, p = carrier["q"], carrier["p"]
    return {
        "depth": depth,
        "resident_relation_components": 2,
        "resident_symplectic_cells": 8,
        "resident_fiber_cells": 8,
        "resident_two_port_cells": 16,
        "logical_resident_payload_bits": 8 * (q - 1).bit_length() + 8 * (p - 1).bit_length(),
        "ordinary_dense_two_port_relation_cells": 8 * q * q,
        "monomial_kernel_ports": sum(item[0][1] % q == 0 for item in (carrier["a"], carrier["b"])),
    }


def entry(item: list[list[int]], source: int, target: int, x: int, y: int, carrier: dict[str, Any]) -> int:
    return item[1][2 * source + target] * kernel(tuple(item[0]), x, y, carrier["q"], carrier["p"], carrier["omega"]) % carrier["p"]


def project(carrier: dict[str, Any], family: str) -> int:
    require(carrier["stage"] == "FORWARD_COMPLETE", "boundary projected outside final stage")
    q = carrier["q"]
    code = 1 if family == "PRIMARY" else 2
    probes = (
        (carrier["a"], 0, 1, (3 * code + 1) % q, (5 * code + 2) % q, 1),
        (carrier["a"], 1, 0, (7 * code + 2) % q, (11 * code + 3) % q, 2),
        (carrier["b"], 0, 0, (13 * code + 1) % q, (17 * code + 4) % q, 3),
        (carrier["b"], 1, 1, (19 * code + 2) % q, (23 * code + 5) % q, 5),
    )
    return sum(weight * entry(item, source, target, x, y, carrier) for item, source, target, x, y, weight in probes) % carrier["p"]


def digest(carrier: dict[str, Any]) -> str:
    result = hashlib.sha256()
    for item in (carrier["a"], carrier["b"]):
        for source in range(2):
            for x in range(carrier["q"]):
                for target in range(2):
                    for y in range(carrier["q"]):
                        result.update(entry(item, source, target, x, y, carrier).to_bytes(2, "big"))
    return result.hexdigest()


def execute(carrier: dict[str, Any], depth: int, family: str, audit: dict[str, int], mutation: str | None = None, reverse_modules: bool = False) -> dict[str, Any]:
    require(carrier["stage"] == "IDLE", "non-idle forward")
    before = carrier_state(carrier)
    ids = backing(carrier)
    checkpoints = []
    executed_stages = []
    for index in range(depth):
        stages = stage_descriptors(index, family, carrier["q"])
        if reverse_modules:
            stages = tuple(reversed(stages))
        for stage in stages:
            apply(carrier, stage, audit)
            executed_stages.append(stage)
        if index + 1 in CHECKPOINT_DEPTHS:
            checkpoints.append(point(carrier, index + 1))
    carrier["stage"] = "FORWARD_COMPLETE"
    boundary = project(carrier, family)
    commitment = digest(carrier)
    final_point = point(carrier, depth)
    sequence = list(reversed(executed_stages))
    if mutation == "MISSING":
        sequence = sequence[1:]
    elif mutation == "REORDER":
        sequence = list(reversed(sequence))
    for position, stage in enumerate(sequence):
        undo(carrier, stage, audit, mutation == "WRONG" and position == 0)
    carrier["stage"] = "IDLE"
    restored = carrier_state(carrier) == before
    same_backing = backing(carrier) == ids
    if mutation is None:
        require(restored and same_backing, "independent restoration failure")
        carrier["generation"] += 1
    return {
        "boundary": boundary,
        "semantic_commitment": commitment,
        "checkpoints": checkpoints,
        "final_checkpoint": final_point,
        "exact_restoration": restored,
        "same_backing": same_backing,
        "generation": carrier["generation"],
    }


def dense_product(left: tuple[int, ...], right: tuple[int, ...], field: tuple[int, ...]) -> list[list[int]]:
    q, p, omega, _ = field
    return [
        [sum(kernel(left, x, shared, q, p, omega) * kernel(right, shared, y, q, p, omega) for shared in range(q)) % p for y in range(q)]
        for x in range(q)
    ]


def exhaustive_q5() -> dict[str, Any]:
    p, omega, gauss, _ = public_field(5)
    field = (5, p, omega, gauss)
    elements = tuple(
        (a, b, c, d)
        for a in range(5) for b in range(5) for c in range(5) for d in range(5)
        if (a * d - b * c) % 5 == 1
    )
    require(len(elements) == 120, "wrong SL(2,5) cardinality")
    dense_pairs = 0
    for left in elements:
        for right in elements:
            streamed = streamed_cocycle(left, right, field)
            closed = closed_cocycle(left, right, field)
            require(streamed == closed, "q5 exhaustive cocycle mismatch")
            output = mm(left, right, 5)
            dense = dense_product(left, right, field)
            require(all(dense[x][y] == streamed * kernel(output, x, y, 5, p, omega) % p for x in range(5) for y in range(5)), "q5 exhaustive dense mismatch")
            dense_pairs += 1
    return {"sl2_elements": len(elements), "ordered_pairs": dense_pairs, "all_streamed_closed_dense_equal": True}


def compare_cases(production: dict[str, Any], audit: dict[str, int]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    indexed = {(case["q"], case["depth"], case["family"]): case for case in production["cases"]}
    comparisons = []
    for q, depth, family in CASES:
        # Both public programs borrow the same PRIMARY-sealed carrier; the
        # family selects only the public descriptor schedule.
        observed = execute(seal(q, "PRIMARY"), depth, family, audit)
        package = indexed[(q, depth, family)]
        checks = {
            "boundary": observed["boundary"] == package["boundary"],
            "semantic_commitment": observed["semantic_commitment"] == package["semantic_commitment"],
            "checkpoints": observed["checkpoints"] == package["checkpoints"],
            "final_checkpoint": observed["final_checkpoint"] == package["final_checkpoint"],
            "exact_restoration": observed["exact_restoration"] and package["exact_canonical_state_restored"],
            "same_backing": observed["same_backing"] and package["same_backing_restored"],
            "baseline_boundary": observed["boundary"] == package["matched_classical"]["boundary"],
            "baseline_commitment": observed["semantic_commitment"] == package["matched_classical"]["semantic_commitment"],
        }
        require(all(checks.values()), f"case mismatch q={q} depth={depth} family={family}: {checks}")
        comparisons.append({"q": q, "p": 2 * q + 1, "depth": depth, "family": family, "checks": checks})

    # Independent same-object reuse and fresh parity.
    reused = seal(53, "PRIMARY")
    initial = carrier_state(reused)
    ids = backing(reused)
    first = execute(reused, 4, "PRIMARY", audit)
    second = execute(reused, 64, "ALTERNATE", audit)
    fresh = execute(seal(53, "PRIMARY"), 64, "ALTERNATE", audit)
    reuse = {
        "first_generation": first["generation"] == 1,
        "second_generation": second["generation"] == 2,
        "same_backing": backing(reused) == ids,
        "exact_state_after_two_uses": carrier_state(reused) == initial,
        "second_boundary_matches_fresh": second["boundary"] == fresh["boundary"],
        "second_commitment_matches_fresh": second["semantic_commitment"] == fresh["semantic_commitment"],
        "production_reuse_tuple_matches": second["boundary"] == production["restoration_and_reuse"]["second_boundary"],
        "production_reports_no_snapshot": production["restoration_and_reuse"]["snapshot_used"] is False,
    }
    require(all(reuse.values()), "reuse mismatch")
    return comparisons, reuse


def independent_controls(audit: dict[str, int]) -> dict[str, bool]:
    failures = {}
    for mutation in ("MISSING", "WRONG", "REORDER"):
        outcome = execute(seal(23, "PRIMARY"), 4, "PRIMARY", audit, mutation=mutation)
        failures[mutation] = not outcome["exact_restoration"]
    normal = execute(seal(23, "PRIMARY"), 4, "PRIMARY", audit)
    altered = execute(seal(23, "PRIMARY"), 4, "PRIMARY", audit, reverse_modules=True)
    premature = False
    try:
        project(seal(23, "PRIMARY"), "PRIMARY")
    except AssertionError:
        premature = True
    invalid_family = False
    try:
        seal(23, "NULL")
    except AssertionError:
        invalid_family = True
    invalid_order = False
    try:
        seal(17, "PRIMARY")
    except AssertionError:
        invalid_order = True
    checks = {
        "missing_inverse_fails": failures["MISSING"],
        "wrong_inverse_fails": failures["WRONG"],
        "reordered_inverse_fails": failures["REORDER"],
        "module_reorder_changes_boundary": normal["boundary"] != altered["boundary"],
        "premature_projection_rejected": premature,
        "null_family_rejected": invalid_family,
        "non_safe_prime_rejected": invalid_order,
    }
    require(all(checks.values()), f"control failure: {checks}")
    return checks


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(production_path: Path) -> dict[str, Any]:
    production = json.loads(production_path.read_text(encoding="utf-8"))
    audit = {"trajectory_cocycle_parity_checks": 0}
    comparisons, reuse = compare_cases(production, audit)
    controls = independent_controls(audit)
    exhaustive = exhaustive_q5()
    bit_widths = sorted({case["final_checkpoint"]["logical_resident_payload_bits"] for case in production["cases"]})
    require(bit_widths == [56, 72, 88, 104, 120], "unexpected payload-width law")
    require(all(case["final_checkpoint"]["resident_two_port_cells"] == 16 for case in production["cases"]), "logical cell growth")
    return {
        "schema": "CAT_CAS_GROWING_PRIME_TWO_FIBER_WEIL_GAUSSIAN_PHASE_KERNEL_INDEPENDENT_ORACLE_V1",
        "production_result": "GROWING_PRIME_TWO_FIBER_WEIL_GAUSSIAN_PHASE_KERNEL_CLOSURE_RESULTS.json",
        "production_result_sha256": sha256(production_path),
        "independence": {
            "imports_production_module": False,
            "separate_tuple_list_recurrence": True,
            "production_projection_function_called": False,
            "production_inverse_function_called": False,
            "production_result_used_only_as_comparison_target": True,
        },
        "case_comparisons": comparisons,
        "all_11_public_cases_reconstructed": len(comparisons) == 11,
        "trajectory_cocycle_parity_checks": audit["trajectory_cocycle_parity_checks"],
        "exhaustive_q5_dense_oracle": exhaustive,
        "controls": controls,
        "restoration_and_reuse": reuse,
        "observed_resource_law": {
            "resident_two_port_field_cells": 16,
            "logical_resident_payload_bit_values": bit_widths,
            "ordinary_dense_cells": "8*Q^2",
            "accepted_streamed_cocycle_work": "Q_PHASE_TERMS_PER_COMPOSITION",
            "matched_classical_work": "CLOSED_GAUSS_COCYCLE_AFTER_ONE_PUBLIC_SCALAR_CACHE",
            "fixed_bit_width_across_unbounded_q": False,
        },
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
        "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "claim_ceiling": production["claim_ceiling"],
        "preserved_subclaims": [
            "EXACT_PROJECTIVE_WEIL_COMPOSITION",
            "SEPARABLE_PUBLIC_CHIRP_HADAMARD_INTERSECTION",
            "FIXED16_LOGICAL_FIELD_CELLS_ACROSS_TESTED_CASES",
            "EXACT_IN_PLACE_RESTORATION_AND_SAME_BACKING_REUSE",
            "IDENTICAL16_CELL_COMPACT_CLASSICAL_BASELINE",
        ],
        "rejected_interpretations": production["not_established"],
        "qualified": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--production", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = json.dumps(build(args.production), indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
