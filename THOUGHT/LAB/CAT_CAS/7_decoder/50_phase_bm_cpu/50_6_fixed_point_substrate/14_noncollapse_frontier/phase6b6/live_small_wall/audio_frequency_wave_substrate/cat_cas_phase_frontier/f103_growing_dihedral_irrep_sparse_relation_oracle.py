#!/usr/bin/env python3
"""Independent group-coordinate oracle for the growing F103 dihedral package.

This file deliberately does not import the production package.  It treats full
group coordinates as primitive, independently reconstructs the public program,
and uses a separately written Fourier observer only to measure irrep support.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


P = 103
G = 5
ORDERS = (3, 6, 17, 34, 51)
CHECKPOINTS = (1, 2, 4, 8, 16)
Element = tuple[int, int]
Matrix = tuple[int, ...]


def die(message: str) -> None:
    raise RuntimeError(message)


def z(exp: int) -> int:
    return pow(G, exp % (P - 1), P)


def root(n: int) -> int:
    if (P - 1) % n:
        die("non-split order")
    return pow(G, (P - 1) // n, P)


def group(n: int) -> tuple[Element, ...]:
    return tuple((r, s) for s in (0, 1) for r in range(n))


def product(n: int, x: Element, y: Element) -> Element:
    return ((x[0] + (-y[0] if x[1] else y[0])) % n, x[1] ^ y[1])


def inv(n: int, x: Element) -> Element:
    return ((x[0] if x[1] else -x[0]) % n, x[1])


def labels(n: int) -> tuple[tuple[str, int, int, int, int], ...]:
    out: list[tuple[str, int, int, int, int]] = []
    for rs in ((1, -1) if n % 2 == 0 else (1,)):
        for ss in (1, -1):
            out.append((f"L{'P' if rs == 1 else 'M'}{'P' if ss == 1 else 'M'}", 1, rs, ss, 0))
    stop = n // 2 - 1 if n % 2 == 0 else (n - 1) // 2
    out.extend((f"K{k}", 2, 1, 1, k) for k in range(1, stop + 1))
    if sum(d * d for _, d, _, _, _ in out) != 2 * n:
        die("bad irrep dimension sum")
    return tuple(out)


def rho(n: int, descriptor: tuple[str, int, int, int, int], x: Element) -> Matrix:
    _, d, rs, ss, k = descriptor
    r, s = x
    if d == 1:
        return (pow(rs % P, r, P) * pow(ss % P, s, P) % P,)
    q = pow(root(n), k * r, P)
    qi = pow(q, -1, P)
    return (0, q, qi, 0) if s else (q, 0, 0, qi)


def mm(a: Matrix, b: Matrix, d: int) -> Matrix:
    if d == 1:
        return (a[0] * b[0] % P,)
    return (
        (a[0] * b[0] + a[1] * b[2]) % P,
        (a[0] * b[1] + a[1] * b[3]) % P,
        (a[2] * b[0] + a[3] * b[2]) % P,
        (a[2] * b[1] + a[3] * b[3]) % P,
    )


def trace_ab(a: Matrix, b: Matrix, d: int) -> int:
    if d == 1:
        return a[0] * b[0] % P
    return (a[0] * b[0] + a[1] * b[2] + a[2] * b[1] + a[3] * b[3]) % P


def fourier(n: int, vector: list[int]) -> dict[str, Matrix]:
    out: dict[str, Matrix] = {}
    for desc in labels(n):
        label, d, _, _, _ = desc
        block = [0] * (d * d)
        for value, x in zip(vector, group(n), strict=True):
            rep = rho(n, desc, x)
            for i, entry in enumerate(rep):
                block[i] = (block[i] + value * entry) % P
        if any(block):
            out[label] = tuple(block)
    return out


def inverse_fourier(n: int, blocks: dict[str, Matrix]) -> list[int]:
    descs = {desc[0]: desc for desc in labels(n)}
    scale = pow(2 * n, -1, P)
    out: list[int] = []
    for x in group(n):
        total = 0
        for label, block in blocks.items():
            desc = descs[label]
            d = desc[1]
            total += d * trace_ab(block, rho(n, desc, inv(n, x)), d)
        out.append(total * scale % P)
    return out


def seed(n: int, family: str, register: str) -> list[int]:
    code = 1 if family == "PRIMARY" else 2
    offset = 0 if register == "A" else 4
    return inverse_fourier(
        n,
        {
            "K1": (
                z(code + offset + 1),
                z(2 * code + offset + 3),
                z(3 * code + offset + 5),
                z(5 * code + offset + 7),
            )
        },
    )


def public_blocks(n: int, index: int, family: str, kind: int, operation: str) -> dict[str, Matrix]:
    code = 1 if family == "PRIMARY" else 2
    if operation == "INTERSECT":
        return {
            "LPP": (z(index + code + 3 * kind),),
            "K1": (
                z(index + code + kind + 1),
                z(2 * index + code + 2 * kind + 3),
                z(3 * index + 2 * code + kind + 5),
                z(index + 3 * code + 4 * kind + 7),
            ),
        }
    out: dict[str, Matrix] = {}
    for label, d, rs, ss, k in labels(n):
        if d == 1:
            out[label] = (z(index + code + kind + rs + 2 * ss),)
        else:
            entry = z(index + code + kind + k)
            out[label] = (1, entry, 0, 1) if kind % 2 else (1, 0, entry, 1)
    return out


def scalar(index: int, family: str, kind: int) -> int:
    code = 1 if family == "PRIMARY" else 2
    return z((kind + 1) * index + kind * code + 1)


Stage = tuple[str, str, int, str, int, int]


def program(index: int, family: str) -> tuple[Stage, ...]:
    stages = (
        ("B", "RIGHT_COMPOSE", index, family, 1, scalar(index, family, 1)),
        ("A", "INTERSECT", index, family, 2, scalar(index, family, 2)),
        ("B", "LEFT_COMPOSE", index, family, 3, scalar(index, family, 3)),
        ("A", "INTERSECT", index, family, 4, scalar(index, family, 4)),
    )
    return stages if family == "PRIMARY" else tuple(reversed(stages))


def convolution(n: int, left: list[int], right: list[int]) -> list[int]:
    points = group(n)
    at = {x: i for i, x in enumerate(points)}
    out: list[int] = []
    for target in points:
        total = 0
        for i, x in enumerate(points):
            total += left[i] * right[at[product(n, inv(n, x), target)]]
        out.append(total % P)
    return out


def apply(n: int, a: list[int], b: list[int], stage: Stage, sign: int = 1) -> tuple[list[int], list[int]]:
    target_name, operation, index, family, kind, coefficient = stage
    target, source = (a, b) if target_name == "A" else (b, a)
    public = inverse_fourier(n, public_blocks(n, index, family, kind, operation))
    if operation == "INTERSECT":
        delta = [x * y % P for x, y in zip(source, public, strict=True)]
    elif operation == "RIGHT_COMPOSE":
        delta = convolution(n, source, public)
    else:
        delta = convolution(n, public, source)
    changed = [(x + sign * coefficient * y) % P for x, y in zip(target, delta, strict=True)]
    return (changed, b) if target_name == "A" else (a, changed)


def support(n: int, a: list[int], b: list[int], depth: int) -> dict[str, int]:
    dims = {name: d for name, d, _, _, _ in labels(n)}
    ab, bb = fourier(n, a), fourier(n, b)
    ac = sum(dims[name] ** 2 for name in ab)
    bc = sum(dims[name] ** 2 for name in bb)
    return {
        "depth": depth,
        "a_active_block_capacity": ac,
        "b_active_block_capacity": bc,
        "total_active_block_capacity": ac + bc,
        "a_nonzero_coordinates": sum(sum(x != 0 for x in block) for block in ab.values()),
        "b_nonzero_coordinates": sum(sum(x != 0 for x in block) for block in bb.values()),
    }


def digest(a: list[int], b: list[int]) -> str:
    h = hashlib.sha256()
    for vector in (a, b):
        for value in vector:
            h.update(value.to_bytes(2, "big"))
    return h.hexdigest()


def run_case(n: int, depth: int, family: str, mutation: str | None = None) -> dict[str, Any]:
    a, b = seed(n, family, "A"), seed(n, family, "B")
    sealed = (tuple(a), tuple(b))
    history: list[dict[str, int]] = []
    for index in range(depth):
        for stage in program(index, family):
            a, b = apply(n, a, b, stage)
        if index + 1 in CHECKPOINTS:
            history.append(support(n, a, b, index + 1))
    code = 1 if family == "PRIMARY" else 2
    point = ((7 * code + n // 3) % n, code % 2)
    boundary = b[group(n).index(point)]
    commitment = digest(a, b)
    inverse_stages = [stage for index in reversed(range(depth)) for stage in reversed(program(index, family))]
    if mutation == "MISSING":
        inverse_stages = inverse_stages[1:]
    elif mutation == "REORDER":
        inverse_stages = list(reversed(inverse_stages))
    for position, stage in enumerate(inverse_stages):
        if mutation == "WRONG" and position == 0:
            stage = (*stage[:5], (stage[5] + 1) % P)
        a, b = apply(n, a, b, stage, -1)
    return {
        "rotation_order": n,
        "group_order": 2 * n,
        "family": family,
        "depth": depth,
        "boundary": boundary,
        "forward_commitment": commitment,
        "support_history": history,
        "final_support": history[-1],
        "exact_group_coordinate_restoration": (tuple(a), tuple(b)) == sealed,
    }


def representation_checks() -> dict[str, bool]:
    hom = True
    full_roundtrip = True
    roots = True
    for n in ORDERS:
        q = root(n)
        roots &= pow(q, n, P) == 1 and all(pow(q, k, P) != 1 for k in range(1, n))
        points = group(n)
        for desc in labels(n):
            for x in points:
                for y in points:
                    hom &= rho(n, desc, product(n, x, y)) == mm(rho(n, desc, x), rho(n, desc, y), desc[1])
        for i in range(len(points)):
            basis = [0] * len(points)
            basis[i] = 1
            full_roundtrip &= inverse_fourier(n, fourier(n, basis)) == basis
    return {
        "all_declared_roots_have_exact_order": roots,
        "all_declared_representations_are_homomorphisms": hom,
        "all_group_basis_fourier_roundtrips": full_roundtrip,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--production-results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    production = json.loads(args.production_results.read_text(encoding="utf-8"))
    specs = [(n, 16, "PRIMARY") for n in ORDERS] + [(17, 8, "ALTERNATE")]
    cases = [run_case(*spec) for spec in specs]
    comparisons = []
    for independent, observed in zip(cases, production["cases"], strict=True):
        comparisons.append(
            {
                "rotation_order": independent["rotation_order"],
                "family": independent["family"],
                "boundary_matches": independent["boundary"] == observed["boundary"],
                "commitment_matches": independent["forward_commitment"] == observed["forward_commitment"],
                "support_history_matches": independent["support_history"] == observed["support_history"],
                "final_support_matches": independent["final_support"] == observed["final_support"],
                "independent_restoration_exact": independent["exact_group_coordinate_restoration"],
            }
        )
    normal = run_case(6, 4, "PRIMARY")
    reordered_forward_a, reordered_forward_b = seed(6, "PRIMARY", "A"), seed(6, "PRIMARY", "B")
    for index in range(4):
        for stage in reversed(program(index, "PRIMARY")):
            reordered_forward_a, reordered_forward_b = apply(6, reordered_forward_a, reordered_forward_b, stage)
    point = ((7 + 6 // 3) % 6, 1)
    reordered_boundary = reordered_forward_b[group(6).index(point)]
    missing = run_case(6, 4, "PRIMARY", "MISSING")
    wrong = run_case(6, 4, "PRIMARY", "WRONG")
    reordered_inverse = run_case(6, 4, "PRIMARY", "REORDER")
    checks = representation_checks()
    checks.update(
        {
            "all_case_boundaries_match": all(item["boundary_matches"] for item in comparisons),
            "all_case_commitments_match": all(item["commitment_matches"] for item in comparisons),
            "all_support_histories_match": all(item["support_history_matches"] for item in comparisons),
            "all_final_support_records_match": all(item["final_support_matches"] for item in comparisons),
            "all_independent_forward_inverse_runs_restore": all(item["independent_restoration_exact"] for item in comparisons),
            "missing_inverse_fails_restoration": not missing["exact_group_coordinate_restoration"],
            "wrong_inverse_fails_restoration": not wrong["exact_group_coordinate_restoration"],
            "reordered_inverse_fails_restoration": not reordered_inverse["exact_group_coordinate_restoration"],
            "reordered_modules_change_boundary": normal["boundary"] != reordered_boundary,
            "all_primary_supports_reach_full_two_port_capacity": all(
                case["final_support"]["total_active_block_capacity"] == 4 * n
                for case, n in zip(cases[:5], ORDERS, strict=True)
            ),
        }
    )
    if not all(checks.values()) or not all(all(value for key, value in item.items() if key.endswith("matches") or key.startswith("independent_")) for item in comparisons):
        die("independent dihedral qualification failed")
    payload = {
        "schema": "CAT_CAS_F103_GROWING_DIHEDRAL_IRREP_SPARSE_RELATION_ORACLE_RESULTS_V1",
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
        "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "oracle_primitive_state": "FULL_DIHEDRAL_GROUP_COORDINATES",
        "production_module_imported": False,
        "production_projection_called": False,
        "production_results_sha256": hashlib.sha256(args.production_results.read_bytes()).hexdigest(),
        "case_comparisons": comparisons,
        "checks": checks,
        "observed_primary_support_histories": [case["support_history"] for case in cases[:5]],
        "claim_ceiling": production["claim_ceiling"],
        "preserved_subclaims": [
            "EXACT_DIHEDRAL_GROUP_COORDINATE_AND_IRREP_BLOCK_PARITY",
            "NONCOMMUTATIVE_COMPOSITION_AND_HADAMARD_INTERSECTION",
            "ADAPTIVE_IRREP_SUPPORT_SATURATES_FULL_TWO_PORT_CAPACITY_BY_DEPTH16_IN_ALL_FIVE_PRIMARY_CASES",
            "FINAL_ONLY_BOUNDARY_WITH_EXACT_FORWARD_INVERSE_RESTORATION",
        ],
        "rejected_interpretations": production["not_established"],
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
