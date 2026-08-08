#!/usr/bin/env python3
"""M246 CATVM backend for exact permutation-symmetric p=5 occupation ports.

The accepted carrier stores one Q(zeta_5) amplitude for each five-symbol
occupation of n exchange-symmetric rails.  It never stores labelled 5**n
assignments or a dense occupation-transfer matrix.  The result is bounded
software evidence, not a physical waveform implementation.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import math
import socket
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable

import zeta5_normalized_cubic_fourier_coherent_port as field


P = 5
DEPTH = 3
RAILS = (2, 3, 4, 6)
PORT_TYPE = "CATVM_P5_PERMUTATION_SYMMETRIC_OCCUPATION_PORT_V1"
OUTPUT_TYPE = "QZETA5_FINAL_OCCUPATION_AMPLITUDE_V1"
CONSUMER_ID = 246001
OWNER = 246004
K = field.K
ZERO = field.ZERO
ONE = field.ONE
SQRT5 = field.SQRT5


@lru_cache(maxsize=None)
def occupations(total: int) -> tuple[tuple[int, int, int, int, int], ...]:
    values: list[tuple[int, int, int, int, int]] = []

    def visit(remaining: int, slot: int, prefix: list[int]) -> None:
        if slot == 4:
            values.append(tuple(prefix + [remaining]))  # type: ignore[arg-type]
            return
        for value in range(remaining + 1):
            visit(remaining - value, slot + 1, prefix + [value])

    visit(total, 0, [])
    return tuple(values)


@lru_cache(maxsize=None)
def occupation_index(total: int) -> dict[tuple[int, int, int, int, int], int]:
    return {value: index for index, value in enumerate(occupations(total))}


def width_for(n: int) -> int:
    return math.comb(n + 4, 4)


def descriptor_tuple(config: dict[str, Any]) -> tuple[object, ...]:
    return (
        int(config["rails"]),
        int(config["depth"]),
        tuple(int(value) % P for value in config["lambdas"]),
        tuple(int(value) % P for value in config["quadratics"]),
        tuple(int(value) % P for value in config["rungs"]),
        tuple(int(value) % P for value in config["couplings"]),
        tuple(int(value) for value in config["output_occupation"]),
    )


def descriptor_digest(descriptor: tuple[object, ...]) -> str:
    return hashlib.sha256(
        json.dumps(descriptor, separators=(",", ":")).encode()
    ).hexdigest()


def validate_descriptor(config: dict[str, Any]) -> tuple[object, ...]:
    descriptor = descriptor_tuple(config)
    n, depth, lambdas, quadratics, rungs, couplings, output = descriptor
    if n not in RAILS or depth != DEPTH:
        raise RuntimeError("M246 accepts only rails2_3_4_6 depth3")
    if not (
        len(lambdas) == depth
        and len(quadratics) == depth
        and len(rungs) == depth
        and len(couplings) == depth - 1
        and len(output) == P
        and sum(output) == n
        and all(value >= 0 for value in output)
        and all(lambdas)
        and all(rungs)
        and all(couplings)
        and tuple(output) in occupation_index(n)
    ):
        raise RuntimeError("invalid M246 symmetric occupation descriptor")
    forbidden = {
        "rail_lambdas", "left_lambdas", "right_lambdas", "labelled_output",
        "assignment_table", "transfer_matrix",
    }
    if forbidden.intersection(config):
        raise RuntimeError("labelled or exchange-breaking descriptor rejected")
    return descriptor


def phase(parameters: tuple[int, int, int], occupation: tuple[int, ...]) -> int:
    lam, quadratic, rung = parameters
    first = sum(symbol * occupation[symbol] for symbol in range(P))
    second = sum(symbol * symbol * occupation[symbol] for symbol in range(P))
    third = sum(symbol**3 * occupation[symbol] for symbol in range(P))
    return (lam * third + quadratic * second + rung * (first * first - second)) % P


def multinomial(counts: tuple[int, ...]) -> int:
    result = math.factorial(sum(counts))
    for value in counts:
        result //= math.factorial(value)
    return result


@dataclass
class Work:
    forward_modules: int = 0
    inverse_modules: int = 0
    forward_kernel_coefficient_terms: int = 0
    inverse_kernel_coefficient_terms: int = 0
    forward_orbit_dot_terms: int = 0
    inverse_orbit_dot_terms: int = 0
    normalization_field_multiplications: int = 0
    hidden_descriptor_residue_reads: int = 0
    labelled_assignment_materializations: int = 0
    transfer_matrices_materialized: int = 0
    retained_dynamic_inverse_history_entries: int = 0


def build_kernel_row(
    n: int,
    outer: tuple[int, int, int, int, int],
    beta: int,
    row: list[K],
    work: Work,
    inverse: bool,
    wrong_multiplicity: bool = False,
) -> None:
    if len(row) != width_for(n) or any(value != ZERO for value in row):
        raise RuntimeError("dirty M246 coefficient-row scratch")
    index = occupation_index(n)
    accumulated = [0] * P

    def visit(group: int, coefficient: int, exponent: int) -> None:
        if group == P:
            target = index[tuple(accumulated)]  # type: ignore[arg-type]
            multiplier = 1 if wrong_multiplicity else coefficient
            row[target] = field.k_add(
                row[target], field.k_scale(field.zeta_power(exponent), multiplier)
            )
            if inverse:
                work.inverse_kernel_coefficient_terms += 1
            else:
                work.forward_kernel_coefficient_terms += 1
            return
        count = outer[group]
        for distribution in occupations(count):
            for symbol in range(P):
                accumulated[symbol] += distribution[symbol]
            visit(
                group + 1,
                coefficient * multinomial(distribution),
                exponent + 2 * beta * group * sum(
                    symbol * distribution[symbol] for symbol in range(P)
                ),
            )
            for symbol in range(P):
                accumulated[symbol] -= distribution[symbol]

    visit(0, 1, 0)


def normalization_exponent(n: int) -> int:
    return (n + 1) // 2


def normalize_vector(values: list[K], exponent: int, n: int, work: Work) -> int:
    if n % 2:
        for index, value in enumerate(values):
            values[index] = field.k_mul(SQRT5, value)
            work.normalization_field_multiplications += 1
    exponent += normalization_exponent(n)
    return field.canonicalize_vector(values, exponent)


def public_plan_integer_cells(n: int) -> int:
    return P * sum(width_for(degree) for degree in range(n + 1)) + width_for(n)


class OccupationCarrier:
    def __init__(self, carrier_id: str, n: int) -> None:
        self.carrier_id = carrier_id
        self.n = n
        self.width = width_for(n)
        self.plan = occupations(n)
        self.index = occupation_index(n)
        self.cells: list[K] = [ZERO] * self.width
        self.scratch: list[K] = [ZERO] * self.width
        self.row: list[K] = [ZERO] * self.width
        self.lambdas = [0] * DEPTH
        self.quadratics = [0] * DEPTH
        self.rungs = [0] * DEPTH
        self.couplings = [0] * (DEPTH - 1)
        self.output_occupation = [0] * P
        self.denominator_exponent = 0
        self.cursor = 0
        self.stage = "CANONICAL"
        self.owner = 0
        self.generation = 0
        self.last_restored_generation = 0
        self.transaction_id = ""
        self.oracle_id = ""
        self.program_id = ""
        self.descriptor_digest = ""
        self.leased = False

    @property
    def delta_index(self) -> int:
        return self.index[(self.n, 0, 0, 0, 0)]

    def canonical(self) -> bool:
        return (
            self.cells == [ZERO] * self.width
            and self.scratch == [ZERO] * self.width
            and self.row == [ZERO] * self.width
            and self.lambdas == [0] * DEPTH
            and self.quadratics == [0] * DEPTH
            and self.rungs == [0] * DEPTH
            and self.couplings == [0] * (DEPTH - 1)
            and self.output_occupation == [0] * P
            and self.denominator_exponent == 0
            and self.cursor == 0
            and self.stage == "CANONICAL"
            and self.owner == 0
            and self.generation == 0
            and self.transaction_id == ""
            and self.oracle_id == ""
            and self.program_id == ""
            and self.descriptor_digest == ""
            and not self.leased
        )

    def lease(self, config: dict[str, Any], request: dict[str, Any]) -> None:
        generation = int(request["generation"])
        if (
            self.leased
            or not self.canonical()
            or int(request["owner"]) <= 0
            or not request["transaction_id"]
            or generation != self.last_restored_generation + 1
        ):
            raise RuntimeError("invalid M246 lease or generation")
        self.leased = True
        self.owner = int(request["owner"])
        self.generation = generation
        self.transaction_id = str(request["transaction_id"])
        self.oracle_id = str(request["oracle_id"])
        self.program_id = str(request["program_id"])
        self.descriptor_digest = str(config["_descriptor_digest"])
        self.stage = "LEASED"

    def load_hidden(self, config: dict[str, Any], work: Work) -> None:
        descriptor = validate_descriptor(config)
        if (
            self.stage != "LEASED"
            or descriptor[0] != self.n
            or config["_descriptor_digest"] != self.descriptor_digest
        ):
            raise RuntimeError("M246 descriptor identity mismatch")
        for target, name in (
            (self.lambdas, "lambdas"),
            (self.quadratics, "quadratics"),
            (self.rungs, "rungs"),
            (self.couplings, "couplings"),
            (self.output_occupation, "output_occupation"),
        ):
            for index, value in enumerate(config[name]):
                target[index] = int(value) % P if name != "output_occupation" else int(value)
                work.hidden_descriptor_residue_reads += 1
        self.cells[self.delta_index] = ONE
        self.stage = "FORWARD_READY"

    def require(self, config: dict[str, Any], request: dict[str, Any]) -> None:
        if (
            not self.leased
            or self.owner != int(request["owner"])
            or self.generation != int(request["generation"])
            or self.transaction_id != str(request["transaction_id"])
            or self.oracle_id != str(request["oracle_id"])
            or self.program_id != str(request["program_id"])
            or self.descriptor_digest != str(config["_descriptor_digest"])
        ):
            raise RuntimeError("M246 custody mismatch")

    def module(self, index: int) -> tuple[tuple[int, int, int], int]:
        return (
            (self.lambdas[index], self.quadratics[index], self.rungs[index]),
            1 if index == 0 else self.couplings[index - 1],
        )

    def _forward(self, module_index: int, work: Work, wrong_multiplicity: bool = False) -> None:
        if any(value != ZERO for value in self.scratch + self.row):
            raise RuntimeError("dirty M246 forward scratch")
        parameters, beta = self.module(module_index)
        work.hidden_descriptor_residue_reads += 3 + int(module_index > 0)
        if module_index == 0:
            expected = [ZERO] * self.width
            expected[self.delta_index] = ONE
            if self.cells != expected:
                raise RuntimeError("M246 first-module specialization requires delta input")
            for output_index, output in enumerate(self.plan):
                self.scratch[output_index] = field.zeta_power(phase(parameters, output))
        else:
            for output_index, output in enumerate(self.plan):
                build_kernel_row(
                    self.n, output, beta, self.row, work, False, wrong_multiplicity
                )
                accumulator = ZERO
                for input_index, coefficient in enumerate(self.row):
                    accumulator = field.k_add(
                        accumulator, field.k_mul(coefficient, self.cells[input_index])
                    )
                    work.forward_orbit_dot_terms += 1
                    self.row[input_index] = ZERO
                self.scratch[output_index] = field.k_mul(
                    field.zeta_power(phase(parameters, output)), accumulator
                )
        self.denominator_exponent = normalize_vector(
            self.scratch, self.denominator_exponent, self.n, work
        )
        for index in range(self.width):
            self.cells[index] = self.scratch[index]
            self.scratch[index] = ZERO
        work.forward_modules += 1

    def _inverse(self, module_index: int, work: Work) -> None:
        if any(value != ZERO for value in self.scratch + self.row):
            raise RuntimeError("dirty M246 inverse scratch")
        parameters, beta = self.module(module_index)
        work.hidden_descriptor_residue_reads += 3 + int(module_index > 0)
        for input_index, input_occupation in enumerate(self.plan):
            build_kernel_row(self.n, input_occupation, -beta, self.row, work, True)
            accumulator = ZERO
            for output_index, coefficient in enumerate(self.row):
                phased = field.k_mul(
                    field.zeta_power(-phase(parameters, self.plan[output_index])),
                    self.cells[output_index],
                )
                accumulator = field.k_add(
                    accumulator, field.k_mul(coefficient, phased)
                )
                work.inverse_orbit_dot_terms += 1
                self.row[output_index] = ZERO
            self.scratch[input_index] = accumulator
        self.denominator_exponent = normalize_vector(
            self.scratch, self.denominator_exponent, self.n, work
        )
        for index in range(self.width):
            self.cells[index] = self.scratch[index]
            self.scratch[index] = ZERO
        work.inverse_modules += 1

    def forward_one(self, config: dict[str, Any], request: dict[str, Any], work: Work) -> None:
        self.require(config, request)
        if self.stage not in ("FORWARD_READY", "FORWARD") or self.cursor >= DEPTH:
            raise RuntimeError("invalid M246 forward cursor")
        self._forward(self.cursor, work)
        self.cursor += 1
        self.stage = "FORWARD"

    def project(self, config: dict[str, Any], request: dict[str, Any]) -> tuple[K, int]:
        self.require(config, request)
        if (
            self.stage != "FORWARD"
            or self.cursor != DEPTH
            or any(value != ZERO for value in self.scratch + self.row)
        ):
            raise RuntimeError("premature or dirty M246 projection")
        self.stage = "PROJECTED"
        return self.cells[self.index[tuple(self.output_occupation)]], self.denominator_exponent

    def inverse_one(self, config: dict[str, Any], request: dict[str, Any], work: Work) -> None:
        self.require(config, request)
        if self.stage not in ("PROJECTED", "INVERSE") or self.cursor <= 0:
            raise RuntimeError("invalid M246 inverse cursor")
        self._inverse(self.cursor - 1, work)
        self.cursor -= 1
        self.stage = "INVERSE"

    def release(self, config: dict[str, Any], request: dict[str, Any]) -> None:
        self.require(config, request)
        expected = [ZERO] * self.width
        expected[self.delta_index] = ONE
        if (
            self.stage != "INVERSE"
            or self.cursor != 0
            or self.denominator_exponent != 0
            or self.cells != expected
            or any(value != ZERO for value in self.scratch + self.row)
        ):
            raise RuntimeError("M246 release before exact restoration")
        self.cells[self.delta_index] = ZERO
        for values in (
            self.lambdas, self.quadratics, self.rungs, self.couplings,
            self.output_occupation,
        ):
            for index in range(len(values)):
                values[index] = 0
        generation = self.generation
        self.denominator_exponent = 0
        self.cursor = 0
        self.stage = "CANONICAL"
        self.owner = 0
        self.generation = 0
        self.transaction_id = ""
        self.oracle_id = ""
        self.program_id = ""
        self.descriptor_digest = ""
        self.leased = False
        self.last_restored_generation = generation
        if not self.canonical():
            raise RuntimeError("M246 post-release canonical mismatch")


def amplitude_json(value: K, exponent: int) -> dict[str, object]:
    return {"numerator": list(value), "denominator_power5": exponent}


def run_transaction(
    carrier: OccupationCarrier, config: dict[str, Any], request: dict[str, Any]
) -> dict[str, object]:
    work = Work()
    backing_ids = (id(carrier.cells), id(carrier.scratch), id(carrier.row))
    descriptor_ids = tuple(
        id(values) for values in (
            carrier.lambdas, carrier.quadratics, carrier.rungs,
            carrier.couplings, carrier.output_occupation,
        )
    )
    retained: tuple[K, int] | None = None
    carrier.lease(config, request)
    carrier.load_hidden(config, work)
    try:
        for _ in range(DEPTH):
            carrier.forward_one(config, request, work)
            if int(config.get("inject_failure_after_modules", -1)) == carrier.cursor:
                raise RuntimeError("injected partial M246 forward failure")
        retained = carrier.project(config, request)
        if config.get("delay_before_inverse_ms"):
            time.sleep(float(config["delay_before_inverse_ms"]) / 1000.0)
        if config.get("inject_failure_after_projection"):
            raise RuntimeError("injected post-projection M246 failure")
    except Exception:
        if carrier.stage == "FORWARD":
            carrier.stage = "PROJECTED"
        while carrier.cursor:
            carrier.inverse_one(config, request, work)
        carrier.release(config, request)
        raise
    while carrier.cursor:
        carrier.inverse_one(config, request, work)
    carrier.release(config, request)
    if retained is None:
        raise RuntimeError("M246 final boundary missing")
    return {
        "rails": carrier.n,
        "occupation_dimension": carrier.width,
        "final_amplitude": amplitude_json(*retained),
        "generation": carrier.last_restored_generation,
        "same_message_backing": id(carrier.cells) == backing_ids[0],
        "same_output_scratch_backing": id(carrier.scratch) == backing_ids[1],
        "same_coefficient_row_backing": id(carrier.row) == backing_ids[2],
        "same_descriptor_backings": descriptor_ids == tuple(
            id(values) for values in (
                carrier.lambdas, carrier.quadratics, carrier.rungs,
                carrier.couplings, carrier.output_occupation,
            )
        ),
        "canonical_after_restoration": carrier.canonical(),
        "baseline_reload_used": False,
        "message_field_cells": carrier.width,
        "output_scratch_field_cells": carrier.width,
        "coefficient_row_field_cells": carrier.width,
        "total_fixed_field_backing_cells": 3 * carrier.width,
        "public_plan_integer_cells": public_plan_integer_cells(carrier.n),
        "hidden_descriptor_residue_cells": 16,
        "retained_final_amplitude_field_cells_during_inverse": 1,
        "retained_final_amplitude_denominator_exponent_scalar_cells_during_inverse": 1,
        "forward_kernel_coefficient_terms": work.forward_kernel_coefficient_terms,
        "inverse_kernel_coefficient_terms": work.inverse_kernel_coefficient_terms,
        "forward_orbit_dot_terms": work.forward_orbit_dot_terms,
        "inverse_orbit_dot_terms": work.inverse_orbit_dot_terms,
        "normalization_field_multiplications": work.normalization_field_multiplications,
        "accepted_descriptor_reads": work.hidden_descriptor_residue_reads,
        "labelled_assignment_materializations": work.labelled_assignment_materializations,
        "transfer_matrices_materialized": work.transfer_matrices_materialized,
        "retained_dynamic_inverse_history_entries": work.retained_dynamic_inverse_history_entries,
    }


def transformed(
    n: int,
    values: list[K],
    exponent: int,
    parameters: tuple[int, int, int],
    beta: int,
    inverse: bool,
    wrong_multiplicity: bool = False,
    omit_odd_sqrt: bool = False,
) -> tuple[list[K], int]:
    plan = occupations(n)
    width = len(plan)
    row = [ZERO] * width
    result = [ZERO] * width
    work = Work()
    if inverse:
        for input_index, input_occupation in enumerate(plan):
            build_kernel_row(
                n, input_occupation, -beta, row, work, True, wrong_multiplicity
            )
            accumulator = ZERO
            for output_index, coefficient in enumerate(row):
                phased = field.k_mul(
                    field.zeta_power(-phase(parameters, plan[output_index])),
                    values[output_index],
                )
                accumulator = field.k_add(accumulator, field.k_mul(coefficient, phased))
                row[output_index] = ZERO
            result[input_index] = accumulator
    else:
        for output_index, output in enumerate(plan):
            build_kernel_row(n, output, beta, row, work, False, wrong_multiplicity)
            accumulator = ZERO
            for input_index, coefficient in enumerate(row):
                accumulator = field.k_add(
                    accumulator, field.k_mul(coefficient, values[input_index])
                )
                row[input_index] = ZERO
            result[output_index] = field.k_mul(
                field.zeta_power(phase(parameters, output)), accumulator
            )
    if n % 2 and not omit_odd_sqrt:
        result = [field.k_mul(SQRT5, value) for value in result]
    exponent += normalization_exponent(n)
    return result, field.canonicalize_vector(result, exponent)


def mechanism_controls(config: dict[str, Any]) -> dict[str, bool]:
    descriptor = validate_descriptor(config)
    n = int(descriptor[0])
    modules = [
        (
            (descriptor[2][index], descriptor[3][index], descriptor[4][index]),
            1 if index == 0 else descriptor[5][index - 1],
        )
        for index in range(DEPTH)
    ]
    width = width_for(n)
    initial = [ZERO] * width
    initial[occupation_index(n)[(n, 0, 0, 0, 0)]] = ONE
    forwarded, exponent = initial, 0
    for parameters, beta in modules:
        forwarded, exponent = transformed(n, forwarded, exponent, parameters, beta, False)
    missing_inverse = forwarded != initial or exponent != 0

    wrong, wrong_exponent = forwarded, exponent
    wrong_parameters = ((modules[-1][0][0] % 4) + 1, *modules[-1][0][1:])
    wrong, wrong_exponent = transformed(
        n, wrong, wrong_exponent, wrong_parameters, modules[-1][1], True
    )
    for parameters, beta in reversed(modules[:-1]):
        wrong, wrong_exponent = transformed(n, wrong, wrong_exponent, parameters, beta, True)

    reordered, reordered_exponent = forwarded, exponent
    for parameters, beta in (modules[-2], modules[-1], modules[0]):
        reordered, reordered_exponent = transformed(
            n, reordered, reordered_exponent, parameters, beta, True
        )

    correct, correct_exponent = initial, 0
    mutated, mutated_exponent = initial, 0
    for index, (parameters, beta) in enumerate(modules):
        correct, correct_exponent = transformed(
            n, correct, correct_exponent, parameters, beta, False
        )
        mutated, mutated_exponent = transformed(
            n, mutated, mutated_exponent, parameters, beta, False,
            wrong_multiplicity=index == 1,
        )

    odd_control = True
    if n % 2:
        correct_odd, correct_odd_exponent = transformed(
            n, initial, 0, modules[0][0], modules[0][1], False
        )
        wrong_odd, wrong_odd_exponent = transformed(
            n, initial, 0, modules[0][0], modules[0][1], False,
            omit_odd_sqrt=True,
        )
        odd_control = (
            correct_odd != wrong_odd or correct_odd_exponent != wrong_odd_exponent
        )

    probe = OccupationCarrier("descriptor-control", n)
    request = {
        "oracle_id": "descriptor-control", "program_id": "descriptor-control",
        "owner": OWNER, "generation": 1, "transaction_id": "descriptor-control",
    }
    probe_config = dict(config)
    probe.lease(probe_config, request)
    mutated_config = dict(config)
    mutated_config["rungs"] = list(config["rungs"])
    mutated_config["rungs"][0] = int(mutated_config["rungs"][0]) % 4 + 1
    mutated_config["_descriptor_digest"] = descriptor_digest(
        validate_descriptor(mutated_config)
    )
    try:
        probe.require(mutated_config, request)
    except RuntimeError:
        same_id_mutation_rejected = True
    else:
        same_id_mutation_rejected = False

    dirty_row = [ZERO] * width
    dirty_row[0] = ONE
    try:
        build_kernel_row(n, occupations(n)[0], 1, dirty_row, Work(), False)
    except RuntimeError:
        dirty_row_rejected = True
    else:
        dirty_row_rejected = False

    try:
        broken = dict(config)
        broken["rail_lambdas"] = [[1] * n for _ in range(DEPTH)]
        validate_descriptor(broken)
    except RuntimeError:
        exchange_breaking_rejected = True
    else:
        exchange_breaking_rejected = False

    try:
        disconnected = dict(config)
        disconnected["rungs"] = list(config["rungs"])
        disconnected["rungs"][0] = 0
        validate_descriptor(disconnected)
    except RuntimeError:
        zero_rung_rejected = True
    else:
        zero_rung_rejected = False

    return {
        "missing_inverse_rejected": missing_inverse,
        "wrong_inverse_rejected_after_complete_inverse_word": (
            wrong != initial or wrong_exponent != 0
        ),
        "reordered_inverse_rejected": (
            reordered != initial or reordered_exponent != 0
        ),
        "wrong_multinomial_multiplicity_changes_state": (
            mutated != correct or mutated_exponent != correct_exponent
        ),
        "odd_rail_sqrt5_normalization_is_causal": odd_control,
        "same_id_changed_descriptor_rejected": same_id_mutation_rejected,
        "dirty_coefficient_row_scratch_rejected": dirty_row_rejected,
        "exchange_breaking_descriptor_rejected": exchange_breaking_rejected,
        "zero_rung_disconnected_descriptor_rejected": zero_rung_rejected,
        "retained_transfer_matrices_zero": True,
        "labelled_assignment_materializations_zero": True,
    }


class Service:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config["oracles"]
        self.carriers: dict[str, OccupationCarrier] = {}
        for item in self.config.values():
            item["_descriptor_digest"] = descriptor_digest(validate_descriptor(item))

    def carrier_for(self, oracle_id: str) -> OccupationCarrier:
        config = self.config[oracle_id]
        carrier_id = str(config["carrier_id"])
        n = int(config["rails"])
        if carrier_id not in self.carriers:
            self.carriers[carrier_id] = OccupationCarrier(carrier_id, n)
        carrier = self.carriers[carrier_id]
        if carrier.n != n:
            raise RuntimeError("M246 carrier rail type mismatch")
        return carrier

    def validate_request(self, request: dict[str, Any]) -> dict[str, Any]:
        oracle_id = str(request.get("oracle_id", ""))
        if oracle_id not in self.config:
            raise RuntimeError("unknown M246 oracle")
        config = self.config[oracle_id]
        if (
            request.get("port_type") != PORT_TYPE
            or request.get("output_type") != OUTPUT_TYPE
            or request.get("consumer_id") != CONSUMER_ID
            or request.get("program_id") != oracle_id
            or request.get("depth") != DEPTH
            or request.get("rails") != int(config["rails"])
            or request.get("owner") != OWNER
            or not isinstance(request.get("generation"), int)
            or not request.get("transaction_id")
        ):
            raise RuntimeError("invalid M246 public request")
        return config

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        command = request.get("command")
        if command == "STATUS":
            oracle_id = str(request.get("oracle_id", ""))
            if oracle_id not in self.config:
                return {"status": "REJECTED"}
            carrier = self.carrier_for(oracle_id)
            return {
                "status": "OK", "canonical": carrier.canonical(),
                "leased": carrier.leased,
                "last_restored_generation": carrier.last_restored_generation,
            }
        if command == "CONTROLS":
            oracle_id = str(request.get("oracle_id", ""))
            if oracle_id not in self.config:
                return {"status": "REJECTED"}
            return {"status": "OK", "controls": mechanism_controls(self.config[oracle_id])}
        if command == "RUN":
            try:
                config = self.validate_request(request)
                response = run_transaction(
                    self.carrier_for(str(request["oracle_id"])), config, request
                )
            except Exception as exc:
                return {"status": "REJECTED", "error_class": type(exc).__name__}
            return {"status": "OK", "response": response}
        if command in {
            "PROJECT_OCCUPATION_MESSAGE", "PROJECT_LABELLED_MESSAGE",
            "PROJECT_SCRATCH", "PROJECT_COEFFICIENT_ROW", "PROJECT_DESCRIPTOR",
            "PROJECT_INTERMEDIATE", "DENSE_ASSIGNMENTS", "DENSE_KERNEL",
            "SNAPSHOT", "RUN_SNAPSHOT", "NULL_CARRIER",
        }:
            return {"status": "REJECTED"}
        if command == "SHUTDOWN":
            return {"status": "OK", "shutdown": True}
        return {"status": "REJECTED"}


def socket_address(socket_name: str) -> str:
    if not socket_name.startswith("@catvm-m246-"):
        raise RuntimeError("M246 requires declared abstract Unix socket")
    return "\0" + socket_name[1:]


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: service.py @catvm-m246-NAME")
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(4, 0, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "M246 could not disable core dumps")
    config_line = sys.stdin.readline()
    sys.stdin.close()
    service = Service(json.loads(config_line))
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(socket_address(sys.argv[1]))
    listener.listen(8)
    running = True
    while running:
        connection, _ = listener.accept()
        try:
            payload = b""
            while not payload.endswith(b"\n"):
                chunk = connection.recv(65536)
                if not chunk:
                    break
                payload += chunk
            if not payload:
                continue
            response = service.handle(json.loads(payload))
            running = not response.get("shutdown", False)
            connection.sendall(
                json.dumps(response, sort_keys=True, separators=(",", ":")).encode()
                + b"\n"
            )
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            connection.close()
    listener.close()


if __name__ == "__main__":
    main()
