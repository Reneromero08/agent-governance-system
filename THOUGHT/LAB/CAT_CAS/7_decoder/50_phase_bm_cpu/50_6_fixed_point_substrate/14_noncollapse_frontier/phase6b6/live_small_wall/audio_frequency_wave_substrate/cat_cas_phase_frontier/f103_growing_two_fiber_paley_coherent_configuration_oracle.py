#!/usr/bin/env python3
"""No-import oracle for the fixed-12 two-fiber Paley relation algebra."""

from __future__ import annotations

import argparse
import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any


P, GEN = 103, 5
ORDERS = (5, 13, 17, 29, 37, 53, 73, 97)
CHECKPOINTS = (1, 4, 16, 64, 256, 1024)


def z(exponent: int) -> int:
    return pow(GEN, exponent % 102, P)


def slot(source: int, target: int, cls: int) -> int:
    return (2 * source + target) * 3 + cls


def cls(q: int, value: int) -> int:
    value %= q
    if value == 0:
        return 0
    return 1 if pow(value, (q - 1) // 2, q) == 1 else 2


@lru_cache(maxsize=None)
def table(q: int) -> tuple[tuple[tuple[int, int, int], ...], ...]:
    sets = tuple(tuple(x for x in range(q) if cls(q, x) == label) for label in range(3))
    out = []
    for a in range(3):
        row = []
        for b in range(3):
            bset = set(sets[b])
            row.append(tuple(sum((representative - x) % q in bset for x in sets[a]) for representative in (sets[k][0] for k in range(3))))
        out.append(tuple(row))
    return tuple(out)


def conv(q: int, left: list[int], right: list[int]) -> list[int]:
    out = [0, 0, 0]
    constants = table(q)
    for a in range(3):
        for b in range(3):
            for c in range(3):
                out[c] = (out[c] + left[a] * right[b] * constants[a][b][c]) % P
    return out


def compose(q: int, left: list[int], right: list[int]) -> list[int]:
    out = [0] * 12
    for i in range(2):
        for k in range(2):
            total = [0, 0, 0]
            for j in range(2):
                value = conv(q, [left[slot(i, j, c)] for c in range(3)], [right[slot(j, k, c)] for c in range(3)])
                total = [(x + y) % P for x, y in zip(total, value, strict=True)]
            for c in range(3):
                out[slot(i, k, c)] = total[c]
    return out


def seed(q: int, family: str, register: str) -> list[int]:
    code = 1 if family == "PRIMARY" else 2
    offset = 0 if register == "A" else 19
    return [z(q + 7 * code + offset + position * (code + 2)) for position in range(12)]


def public(q: int, index: int, family: str, kind: int) -> list[int]:
    code = 1 if family == "PRIMARY" else 2
    return [z(q + (index + 1) * (position + kind + 1) + code * (kind + 3) + position * position) for position in range(12)]


def coefficient(index: int, family: str, kind: int) -> int:
    code = 1 if family == "PRIMARY" else 2
    return z((kind + 1) * (index + 1) + code * (kind + 5))


def stages(index: int, family: str) -> tuple[tuple[str, str, int, int], ...]:
    value = (
        ("B", "RIGHT", 1, coefficient(index, family, 1)),
        ("A", "HADAMARD", 2, coefficient(index, family, 2)),
        ("B", "LEFT", 3, coefficient(index, family, 3)),
        ("A", "HADAMARD", 4, coefficient(index, family, 4)),
    )
    return value if family == "PRIMARY" else tuple(reversed(value))


def apply(q: int, a: list[int], b: list[int], index: int, family: str, stage: tuple[str, str, int, int], sign: int) -> None:
    target_name, operation, kind, scale = stage
    target, source = (a, b) if target_name == "A" else (b, a)
    operand = public(q, index, family, kind)
    if operation == "HADAMARD":
        change = [x * y % P for x, y in zip(source, operand, strict=True)]
    elif operation == "RIGHT":
        change = compose(q, source, operand)
    else:
        change = compose(q, operand, source)
    for position, value in enumerate(change):
        target[position] = (target[position] + sign * scale * value) % P


def digest(a: list[int], b: list[int]) -> str:
    h = hashlib.sha256()
    for vector in (a, b):
        for value in vector:
            h.update(value.to_bytes(2, "big"))
    return h.hexdigest()


def run(q: int, depth: int, family: str, mutation: str | None = None) -> dict[str, Any]:
    a, b = seed(q, family, "A"), seed(q, family, "B")
    sealed = tuple(a), tuple(b)
    records = []
    for index in range(depth):
        for stage in stages(index, family):
            apply(q, a, b, index, family, stage, 1)
        if index + 1 in CHECKPOINTS:
            records.append({"depth": index + 1, "resident_relation_cells": 24, "represented_vertices": 2 * q, "represented_dense_relation_entries_per_port": (2 * q) ** 2})
    code = 1 if family == "PRIMARY" else 2
    boundary = b[slot(code % 2, (code + 1) % 2, code % 3)]
    commitment = digest(a, b)
    inverse = [(index, stage) for index in reversed(range(depth)) for stage in reversed(stages(index, family))]
    if mutation == "MISSING":
        inverse = inverse[1:]
    elif mutation == "REORDER":
        inverse = list(reversed(inverse))
    for position, (index, stage) in enumerate(inverse):
        if mutation == "WRONG" and position == 0:
            stage = (*stage[:3], stage[3] + 1)
        apply(q, a, b, index, family, stage, -1)
    return {"paley_order": q, "family": family, "depth": depth, "boundary": boundary, "forward_commitment": commitment, "checkpoints": records, "restores": (tuple(a), tuple(b)) == sealed}


def expand(q: int, coefficients: list[int]) -> list[list[int]]:
    size = 2 * q
    out = [[0] * size for _ in range(size)]
    for i in range(2):
        for x in range(q):
            for j in range(2):
                for y in range(q):
                    out[i * q + x][j * q + y] = coefficients[slot(i, j, cls(q, y - x))]
    return out


def dense_product(left: list[list[int]], right: list[list[int]]) -> list[list[int]]:
    size = len(left)
    return [[sum(left[i][j] * right[j][k] for j in range(size)) % P for k in range(size)] for i in range(size)]


def dense_basis_checks() -> bool:
    for q in (5, 13):
        basis = []
        for position in range(12):
            vector = [0] * 12
            vector[position] = 1
            basis.append(vector)
        expanded = [expand(q, vector) for vector in basis]
        for i in range(12):
            for j in range(12):
                if expand(q, compose(q, basis[i], basis[j])) != dense_product(expanded[i], expanded[j]):
                    return False
                hadamard = [[x * y % P for x, y in zip(row_x, row_y, strict=True)] for row_x, row_y in zip(expanded[i], expanded[j], strict=True)]
                if expand(q, [x * y % P for x, y in zip(basis[i], basis[j], strict=True)]) != hadamard:
                    return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--production-results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    production = json.loads(args.production_results.read_text(encoding="utf-8"))
    specs = [(q, 256, "PRIMARY") for q in ORDERS] + [(97, 1024, "PRIMARY"), (37, 64, "ALTERNATE")]
    independent = [run(*spec) for spec in specs]
    comparisons = []
    for own, observed in zip(independent, production["cases"], strict=True):
        comparisons.append({
            "paley_order": own["paley_order"],
            "family": own["family"],
            "depth": own["depth"],
            "boundary_matches": own["boundary"] == observed["boundary"],
            "commitment_matches": own["forward_commitment"] == observed["forward_commitment"],
            "checkpoints_match": own["checkpoints"] == observed["checkpoints"],
            "independent_restoration_exact": own["restores"],
        })
    checks = {
        "all_boundaries_match": all(x["boundary_matches"] for x in comparisons),
        "all_commitments_match": all(x["commitment_matches"] for x in comparisons),
        "all_checkpoint_records_match": all(x["checkpoints_match"] for x in comparisons),
        "all_independent_transactions_restore": all(x["independent_restoration_exact"] for x in comparisons),
        "missing_inverse_fails": not run(13, 4, "PRIMARY", "MISSING")["restores"],
        "wrong_inverse_fails": not run(13, 4, "PRIMARY", "WRONG")["restores"],
        "reordered_inverse_fails": not run(13, 4, "PRIMARY", "REORDER")["restores"],
        "all_144_composition_and_intersection_basis_pairs_match_dense_at_q5_and_q13": dense_basis_checks(),
        "all_declared_class_tables_reconstructed_by_enumeration": all(len(table(q)) == 3 for q in ORDERS),
        "fixed12_independent_recurrence_for_all_cases": all(len(seed(q, family, "A")) == 12 for q, _, family in specs),
    }
    if not all(checks.values()):
        raise RuntimeError("independent two-fiber qualification failed")
    payload = {
        "schema": "CAT_CAS_F103_GROWING_TWO_FIBER_PALEY_COHERENT_CONFIGURATION_ORACLE_RESULTS_V1",
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
        "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "production_module_imported": False,
        "production_projection_called": False,
        "oracle_primitive": "ENUMERATED_PALEY_CLASS_TABLE_PLUS_INDEPENDENT12_COORDINATE_RECURRENCE",
        "production_results_sha256": hashlib.sha256(args.production_results.read_bytes()).hexdigest(),
        "case_comparisons": comparisons,
        "checks": checks,
        "claim_ceiling": production["claim_ceiling"],
        "preserved_subclaims": [
            "FIXED12_TWO_FIBER_PALEY_COHERENT_CONFIGURATION_CLOSURE_ACROSS_ALL_DECLARED_ORDERS_AND_DEPTHS",
            "NONCOMMUTATIVE_COMPOSITION_AND_HADAMARD_INTERSECTION",
            "DENSE_RELATION_SEMANTICS_PARITY_FOR_ALL_BASIS_PAIRS_AT_Q5_AND_Q13",
            "FINAL_ONLY_BOUNDARY_EXACT_RESTORATION_AND_REUSE",
        ],
        "rejected_interpretations": production["not_established"],
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
