#!/usr/bin/env python3
"""No-import oracle for the M181 Gauss stream/rank package."""

from __future__ import annotations

import argparse
import ast
import hashlib
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PAIRING_WEIGHTS = (1, 2, 2, 1, 2, 1, 1)
WIDTH = 10


def fail(message: str) -> None:
    raise RuntimeError(message)


def factors(number: int) -> tuple[int, ...]:
    found: list[int] = []
    candidate = 2
    while candidate * candidate <= number:
        if number % candidate == 0:
            found.append(candidate)
            while number % candidate == 0:
                number //= candidate
        candidate += 1
    if number > 1:
        found.append(number)
    return tuple(found)


def generator(prime: int) -> int:
    divisors = factors(prime - 1)
    return next(
        value
        for value in range(2, prime)
        if all(pow(value, (prime - 1) // divisor, prime) != 1 for divisor in divisors)
    )


@dataclass(frozen=True)
class Field:
    q: int
    p: int
    psi: int
    eta: int
    g: int

    @classmethod
    def make(cls, q: int, p: int) -> "Field":
        root = generator(p)
        return cls(
            q,
            p,
            pow(root, (p - 1) // q, p),
            pow(root, (p - 1) // (q - 1), p),
            generator(q),
        )


def det6(point: tuple[int, ...], q: int) -> int:
    a, b, c, d, e, f = point
    return (
        a * (d * f - e * e)
        - b * (b * f - e * c)
        + c * (b * e - d * c)
    ) % q


def minor(matrix: tuple[tuple[int, ...], ...], indexes: tuple[int, ...], q: int) -> int:
    if len(indexes) == 1:
        return matrix[indexes[0]][indexes[0]] % q
    if len(indexes) == 2:
        i, j = indexes
        return (matrix[i][i] * matrix[j][j] - matrix[i][j] ** 2) % q
    return det6(
        (
            matrix[0][0], matrix[0][1], matrix[0][2],
            matrix[1][1], matrix[1][2], matrix[2][2],
        ),
        q,
    )


def class_of(point: tuple[int, ...], q: int) -> tuple[int, int]:
    a, b, c, d, e, f = point
    matrix = ((a, b, c), (b, d, e), (c, e, f))
    for rank in (3, 2, 1):
        for indexes in itertools.combinations(range(3), rank):
            value = minor(matrix, indexes, q)
            if value:
                return rank, 1 if pow(value, (q - 1) // 2, q) == 1 else -1
    return 0, 0


def log_table(field: Field) -> dict[int, int]:
    return {pow(field.g, exponent, field.q): exponent for exponent in range(field.q - 1)}


def character(field: Field, logs: dict[int, int], value: int, exponent: int) -> int:
    if value % field.q == 0:
        fail("zero character input")
    return pow(
        field.eta,
        logs[value % field.q] * exponent % (field.q - 1),
        field.p,
    )


def direct_gauss_table(field: Field, logs: dict[int, int]) -> list[int]:
    return [
        sum(
            pow(field.psi, value, field.p) * character(field, logs, value, exponent)
            for value in range(1, field.q)
        )
        % field.p
        for exponent in range(field.q - 1)
    ]


def table_boundary(
    field: Field,
    gauss: list[int],
    logs: dict[int, int],
    a: int,
    b: int,
    point: tuple[int, ...],
    scale: int,
) -> int:
    q, p = field.q, field.p
    h = q - 1
    half = h // 2
    rank, square = class_of(point, q)
    total_character = (a + b) % h
    answer = 0
    for j in range(h):
        coefficient = pow(h, -1, p) * gauss[(a - j) % h] % p
        if rank == 3:
            gamma = gauss[j] ** 2 * gauss[(j + half) % h] * gauss[half] ** 3 % p
            determinant_factor = gamma * character(field, logs, det6(point, q), -j) % p
        elif j == 0:
            determinant_factor = (
                q**6 - q**5 - q**3 + q**2 if rank == 0 else q**2 - q**3
            ) % p
        elif j == half and rank == 1:
            determinant_factor = q**2 * h * gauss[half] ** 3 * square % p
        else:
            determinant_factor = 0
        m = (total_character - j) % h
        if scale % q:
            scale_factor = gauss[m] * character(field, logs, scale, -m) % p
        else:
            scale_factor = h % p if m == 0 else 0
        answer += coefficient * determinant_factor * scale_factor
    return answer % p


def direct_seven_dimensional_sum(
    field: Field,
    logs: dict[int, int],
    a: int,
    b: int,
    target: tuple[int, ...],
    target_scale: int,
) -> tuple[int, int]:
    q, p = field.q, field.p
    answer = 0
    nonzero_source_terms = 0
    for source in itertools.product(range(q), repeat=6):
        determinant = det6(source, q)
        if determinant == 0:
            continue
        determinant_character = character(field, logs, determinant, a)
        pairing = sum(
            weight * left * right
            for weight, left, right in zip(PAIRING_WEIGHTS[:6], target, source)
        )
        for scale in range(1, q):
            source_value = (
                pow(field.psi, determinant * pow(scale, -1, q) % q, p)
                * determinant_character
                * character(field, logs, scale, b)
            ) % p
            phase = pow(field.psi, (pairing + target_scale * scale) % q, p)
            answer += source_value * phase
            nonzero_source_terms += 1
    return answer % p, nonzero_source_terms


def rank_mod(matrix: list[list[int]], modulus: int) -> int:
    data = [row[:] for row in matrix]
    pivot_row = 0
    for column in range(len(data[0])):
        selected = next(
            (row for row in range(pivot_row, len(data)) if data[row][column] % modulus),
            None,
        )
        if selected is None:
            continue
        data[pivot_row], data[selected] = data[selected], data[pivot_row]
        scale = pow(data[pivot_row][column], -1, modulus)
        for offset in range(column, len(data[pivot_row])):
            data[pivot_row][offset] = data[pivot_row][offset] * scale % modulus
        for row in range(pivot_row + 1, len(data)):
            multiple = data[row][column] % modulus
            if multiple:
                for offset in range(column, len(data[row])):
                    data[row][offset] = (
                        data[row][offset] - multiple * data[pivot_row][offset]
                    ) % modulus
        pivot_row += 1
    return pivot_row


def recurrence_order(sequence: list[int], modulus: int) -> int:
    c = [1]
    b = [1]
    length = 0
    gap = 1
    prior_discrepancy = 1
    for n, current in enumerate(sequence):
        discrepancy = current
        for offset in range(1, length + 1):
            discrepancy += c[offset] * sequence[n - offset]
        discrepancy %= modulus
        if discrepancy == 0:
            gap += 1
            continue
        old = c[:]
        ratio = discrepancy * pow(prior_discrepancy, -1, modulus) % modulus
        if len(c) < len(b) + gap:
            c.extend([0] * (len(b) + gap - len(c)))
        for offset, value in enumerate(b):
            c[offset + gap] = (c[offset + gap] - ratio * value) % modulus
        if 2 * length <= n:
            length = n + 1 - length
            b = old
            prior_discrepancy = discrepancy
            gap = 1
        else:
            gap += 1
    return length


class TenCellMachine:
    """Independent fixed-width realization; it retains no character table."""

    def __init__(self, field: Field) -> None:
        self.field = field
        self.cells = [0] * WIDTH
        self.gauss_calls = 0
        self.orbit_terms = 0

    def gauss(self, exponent: int, destination: int) -> None:
        f = self.field
        h = f.q - 1
        self.cells[5:10] = [0, 1, 1, pow(f.eta, exponent % h, f.p), 0]
        self.gauss_calls += 1
        for _ in range(h):
            self.cells[9] = pow(f.psi, self.cells[6], f.p)
            self.cells[5] = (self.cells[5] + self.cells[9] * self.cells[7]) % f.p
            self.cells[6] = self.cells[6] * f.g % f.q
            self.cells[7] = self.cells[7] * self.cells[8] % f.p
            self.orbit_terms += 1
        if destination != 5:
            self.cells[destination] = self.cells[5]
            self.cells[5] = 0
        self.cells[6:10] = [0, 0, 0, 0]

    def chi(self, value: int, exponent: int) -> int:
        f = self.field
        self.cells[6:9] = [1, 1, pow(f.eta, exponent % (f.q - 1), f.p)]
        for _ in range(f.q - 1):
            if self.cells[6] == value % f.q:
                result = self.cells[7]
                self.cells[6:9] = [0, 0, 0]
                return result
            self.cells[6] = self.cells[6] * f.g % f.q
            self.cells[7] = self.cells[7] * self.cells[8] % f.p
        fail("machine character orbit failure")

    def term(
        self,
        a: int,
        b: int,
        point: tuple[int, ...],
        scale: int,
        j: int,
        wrong_shift: bool = False,
    ) -> int:
        f = self.field
        q, p, h = f.q, f.p, f.q - 1
        half = h // 2
        rank, square = class_of(point, q)
        m = (a + b - j) % h
        self.gauss(a - j, 2)
        if rank == 3:
            self.gauss(j, 3)
            self.gauss(j + half + (1 if wrong_shift else 0), 4)
        if scale % q:
            self.gauss(m, 5)
        if rank == 3:
            gamma = self.cells[3] ** 2 * self.cells[4] * self.cells[1] ** 3 % p
            det_factor = gamma * self.chi(det6(point, q), -j) % p
        elif j == 0:
            det_factor = (q**6 - q**5 - q**3 + q**2 if rank == 0 else q**2 - q**3) % p
        elif j == half and rank == 1:
            det_factor = q**2 * h * self.cells[1] ** 3 * square % p
        else:
            det_factor = 0
        scale_factor = (
            self.cells[5] * self.chi(scale, -m) % p
            if scale % q
            else (h % p if m == 0 else 0)
        )
        answer = pow(h, -1, p) * self.cells[2] * det_factor * scale_factor % p
        self.cells[2:6] = [0, 0, 0, 0]
        return answer

    def forward(
        self,
        a: int,
        b: int,
        point: tuple[int, ...],
        scale: int,
        omit: int | None = None,
        wrong_shift: bool = False,
    ) -> int:
        if any(self.cells):
            fail("machine entered forward dirty")
        h = self.field.q - 1
        self.gauss(h // 2, 1)
        for j in range(h):
            if omit is None or j != omit:
                self.cells[0] = (
                    self.cells[0] + self.term(a, b, point, scale, j, wrong_shift)
                ) % self.field.p
        return self.cells[0]

    def inverse(
        self,
        a: int,
        b: int,
        point: tuple[int, ...],
        scale: int,
        reverse: bool = True,
    ) -> None:
        h = self.field.q - 1
        order = range(h - 1, -1, -1) if reverse else range(h)
        for j in order:
            self.cells[0] = (
                self.cells[0] - self.term(a, b, point, scale, j)
            ) % self.field.p
        self.gauss(h // 2, 5)
        self.cells[1] = (self.cells[1] - self.cells[5]) % self.field.p
        self.cells[5] = 0


def inspect_source(path: Path) -> dict[str, Any]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    disallowed = [name for name in imports if "mellin_gauss" in name or "determinant_generating" in name]
    return {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "imports": imports,
        "imports_production_or_predecessor": bool(disallowed),
        "declares_ten_cell_workspace": "WORKSPACE_CELLS = 10" in source,
    }


def build_oracle(production_source: Path, production_result: Path) -> dict[str, Any]:
    production = json.loads(production_result.read_text(encoding="utf-8"))
    fields: dict[int, tuple[Field, dict[int, int], list[int]]] = {}
    rank_checks = []
    for item in production["rank_diagnostics"]:
        field = Field.make(item["q"], item["auxiliary_prime"])
        logs = log_table(field)
        gauss = direct_gauss_table(field, logs)
        fields[field.q] = (field, logs, gauss)
        h = field.q - 1
        a = item["determinant_character"]
        coefficients = [pow(h, -1, field.p) * gauss[(a - j) % h] % field.p for j in range(h)]
        spectrum = [
            sum(coefficients[j] * pow(field.eta, frequency * j % h, field.p) for j in range(h)) % field.p
            for frequency in range(h)
        ]
        expected = [
            pow(field.psi, pow(field.g, frequency, field.q), field.p)
            * pow(field.eta, a * frequency % h, field.p) % field.p
            for frequency in range(h)
        ]
        hankel_rank = rank_mod(
            [[coefficients[(row + column) % h] for column in range(h)] for row in range(h)],
            field.p,
        )
        order = recurrence_order(coefficients * 3, field.p)
        half = h // 2
        gamma = [gauss[j] ** 2 * gauss[(j + half) % h] * gauss[half] ** 3 % field.p for j in range(h)]
        gamma_rank = rank_mod(
            [[gamma[(row + column) % h] for column in range(h)] for row in range(h)],
            field.p,
        )
        if not (spectrum == expected and all(spectrum) and hankel_rank == order == h):
            fail("independent coefficient-rank reconstruction failed")
        if gamma_rank != item["gamma_cyclic_hankel_rank"]:
            fail("independent gamma rank differs")
        rank_checks.append({
            "q": field.q,
            "source_coefficient_fourier_support": sum(value != 0 for value in spectrum),
            "source_coefficient_cyclic_hankel_rank": hankel_rank,
            "source_coefficient_periodic_recurrence_order": order,
            "gamma_cyclic_hankel_rank": gamma_rank,
            "matches_production": True,
        })

    transactions = []
    for case in production["transaction_cases"]:
        field, logs, gauss = fields[case["q"]]
        a = case["program"]["determinant_character"]
        b = case["program"]["scale_character"]
        point = tuple(case["boundary"]["coordinates"])
        scale = case["boundary"]["scale"]
        reference = table_boundary(field, gauss, logs, a, b, point, scale)
        machine = TenCellMachine(field)
        backing = id(machine.cells)
        observed = machine.forward(a, b, point, scale)
        retained = observed
        machine.inverse(a, b, point, scale)
        primary_restored = not any(machine.cells) and id(machine.cells) == backing
        second = case["unrelated_second_program"]
        second_point = tuple(second["boundary"]["coordinates"])
        second_scale = second["boundary"]["scale"]
        second_observed = machine.forward(
            second["determinant_character"], second["scale_character"], second_point, second_scale
        )
        machine.inverse(
            second["determinant_character"], second["scale_character"], second_point, second_scale
        )
        reused_restored = not any(machine.cells) and id(machine.cells) == backing
        wrong = TenCellMachine(field)
        wrong.forward(a, b, point, scale)
        wrong.inverse((a + 1) % (field.q - 1), b, point, scale)
        omitted = TenCellMachine(field).forward(
            a, b, point, scale, omit=case["controls"]["omitted_channel"]
        )
        rank, _ = class_of(point, field.q)
        wrong_gamma_changed = None
        if rank == 3:
            wrong_gamma_changed = TenCellMachine(field).forward(
                a, b, point, scale, wrong_shift=True
            ) != reference
        if not (
            observed == retained == reference == case["materialized_reference_boundary_scalar"]
            and primary_restored
            and reused_restored
            and second_observed == second["fresh_boundary_scalar"]
            and any(wrong.cells)
            and omitted != reference
            and (wrong_gamma_changed is None or wrong_gamma_changed)
        ):
            fail("independent transaction reconstruction or attack failed")
        transactions.append({
            "q": field.q,
            "rank": rank,
            "boundary_matches_table": True,
            "ten_cell_same_backing_primary_restoration": primary_restored,
            "projected_scalar_persists_after_inverse": retained == observed,
            "actual_restored_backing_reuse_matches_fresh": second_observed == second["fresh_boundary_scalar"],
            "ten_cell_same_backing_second_restoration": reused_restored,
            "wrong_inverse_fails": any(wrong.cells),
            "omitted_channel_changes_boundary": omitted != reference,
            "wrong_gamma_shift_applicable": rank == 3,
            "wrong_gamma_shift_changes_boundary": wrong_gamma_changed,
            "snapshot_used": False,
        })

    direct_indexes = (0, 1, len(production["transaction_cases"]) - 2, len(production["transaction_cases"]) - 1)
    direct_checks = []
    for index in direct_indexes:
        case = production["transaction_cases"][index]
        field, logs, _ = fields[case["q"]]
        point = tuple(case["boundary"]["coordinates"])
        direct, terms = direct_seven_dimensional_sum(
            field,
            logs,
            case["program"]["determinant_character"],
            case["program"]["scale_character"],
            point,
            case["boundary"]["scale"],
        )
        expected = case["materialized_reference_boundary_scalar"]
        if direct != expected:
            fail("direct seven-dimensional Fourier sum differs")
        direct_checks.append({
            "transaction_index": index,
            "q": field.q,
            "boundary_rank": class_of(point, field.q)[0],
            "nonzero_source_terms_summed": terms,
            "direct_seven_dimensional_sum": direct,
            "matches_streamed_boundary": True,
        })

    source = inspect_source(production_source)
    if source["imports_production_or_predecessor"] or not source["declares_ten_cell_workspace"]:
        fail("production source isolation check failed")
    return {
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
        "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "production_source": source,
        "production_result_sha256": hashlib.sha256(production_result.read_bytes()).hexdigest(),
        "oracle_imports_production": False,
        "oracle_imports_predecessor": False,
        "independent_rank_checks": rank_checks,
        "independent_transactions": transactions,
        "direct_fourier_checks": direct_checks,
        "mutations": {
            "wrong_inverse_fails_all_transactions": all(item["wrong_inverse_fails"] for item in transactions),
            "omitted_channel_changes_all_boundaries": all(item["omitted_channel_changes_boundary"] for item in transactions),
            "wrong_gamma_shift_changes_all_applicable_boundaries": all(
                (not item["wrong_gamma_shift_applicable"])
                or item["wrong_gamma_shift_changes_boundary"]
                for item in transactions
            ),
        },
        "observed_resource_law": {
            "accepted_workspace_field_cells": WIDTH,
            "resident_tables": 0,
            "source_rank": "q-1",
            "fixed_exact_bit_width": False,
            "stream_work_per_scalar": "THETA(q^2)",
            "diagnostic_log_gauss_hankel_and_direct_sum_state_is_verification_only": True,
            "identical_classical_stream_preserved": True,
        },
        "rejected_interpretations": [
            "FIXED_LINEAR_RECURRENCE_RANK",
            "FIXED_EXACT_BIT_WIDTH",
            "NONLINEAR_PROCEDURAL_COMPRESSION_NO_GO",
            "MULTI_BOUNDARY_AMORTIZED_ADVANTAGE",
            "MACHINE_ENFORCED_HIDDEN_INTERMEDIATE_OR_CATVM_CUSTODY",
            "DISTINCT_PHASE_RESOURCE_OR_COMPUTATIONAL_ADVANTAGE",
            "SMALL_WALL_CROSSING",
            "PHYSICAL_EXECUTION_OR_REPLACEMENT_OF_PHYSICAL_BITS_WITH_PI",
            "UNBOUNDED_COMPUTATION",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--production-source", type=Path, required=True)
    parser.add_argument("--production-result", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = json.dumps(
        build_oracle(args.production_source, args.production_result),
        indent=2,
        sort_keys=True,
    ) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
