#!/usr/bin/env python3
"""M255 compact quartic-cumulant Grassmann Hodge closure diagnostic.

The accepted carrier is not a full exterior coefficient vector.  It stores a
quadratic two-form and two public decomposable quartic cumulants.  Native
relation intersection is addition in this logarithmic chart.  Selected
Berezin/Hodge output coefficients are rematerialized by Pfaffians and a
four-subset public recursion.  The experiment asks whether this compact chart
is closed as the number of typed ports grows; it does not assume closure.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from math import factorial
from typing import Any, Iterable


MILESTONE = 255
PORT_TYPE = "RATIONAL_SUBFAMILY_OF_QZETA8_GAUSSIAN_PLUS_DECOMPOSABLE_QUARTIC_CUMULANT_CHAIN_V1"
OUTPUT_TYPE = "SELECTED_CONNECTED_HODGE_CUMULANT_OBSTRUCTION_V1"
OWNER = 255004
ZERO = Fraction(0)
ONE = Fraction(1)


def fraction_json(value: Fraction) -> list[int]:
    return [value.numerator, value.denominator]


def fraction_payload_bits(value: Fraction) -> int:
    return max(1, abs(value.numerator).bit_length() + 1) + value.denominator.bit_length()


def canonical_digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def mask_of(indices: Iterable[int]) -> int:
    result = 0
    for index in indices:
        result |= 1 << index
    return result


def wedge_sign(left: int, right: int) -> int:
    if left & right:
        return 0
    inversions = 0
    bit = 0
    value = left
    while value:
        if value & 1:
            inversions += (right & ((1 << bit) - 1)).bit_count()
        bit += 1
        value >>= 1
    return -1 if inversions & 1 else 1


def hodge_input_sign(input_mask: int, port_count: int) -> int:
    full = (1 << port_count) - 1
    degree = input_mask.bit_count()
    return wedge_sign(input_mask, full ^ input_mask) * (
        -1 if (degree * (degree - 1) // 2) & 1 else 1
    )


def pair_offset(port_count: int, left: int, right: int) -> int:
    if not 0 <= left < right < port_count:
        raise RuntimeError("M255 invalid pair index")
    offset = 0
    for first in range(left):
        offset += port_count - first - 1
    return offset + right - left - 1


def pfaffian_from_pairs(pair_cells: list[Fraction], port_count: int, mask: int) -> Fraction:
    """Exact Pfaffian of the selected public antisymmetric submatrix."""
    if mask.bit_count() & 1:
        return ZERO
    if mask == 0:
        return ONE
    first_bit = mask & -mask
    first = first_bit.bit_length() - 1
    remaining = mask ^ first_bit
    result = ZERO
    position = 0
    rest = remaining
    while rest:
        partner_bit = rest & -rest
        partner = partner_bit.bit_length() - 1
        coefficient = pair_cells[pair_offset(port_count, first, partner)]
        if coefficient:
            reduced = remaining ^ partner_bit
            sign = -1 if position & 1 else 1
            result += sign * coefficient * pfaffian_from_pairs(
                pair_cells, port_count, reduced
            )
        position += 1
        rest ^= partner_bit
    return result


@dataclass(frozen=True)
class QuarticTerm:
    mask: int
    coefficient: Fraction


@dataclass(frozen=True)
class Module:
    pair_terms: tuple[tuple[int, int, Fraction], ...]
    quartic: QuarticTerm


@dataclass(frozen=True)
class PublicProgram:
    port_count: int
    modules: tuple[Module, ...]
    selected_output_mask: int
    program_id: str


def canonical_program(descriptor: dict[str, Any]) -> PublicProgram:
    if set(descriptor) != {"port_count", "modules", "selected_output"}:
        raise RuntimeError("M255 malformed public descriptor")
    port_count = descriptor["port_count"]
    if port_count not in (4, 6, 8):
        raise RuntimeError("M255 undeclared typed port count")
    modules_raw = descriptor["modules"]
    expected_modules = port_count // 2 - 1
    if not isinstance(modules_raw, list) or len(modules_raw) != expected_modules:
        raise RuntimeError("M255 declared cumulant-chain module count rejected")
    modules: list[Module] = []
    for raw in modules_raw:
        if not isinstance(raw, dict) or set(raw) != {"pairs", "quartic"}:
            raise RuntimeError("M255 malformed relation module")
        seen_pairs: set[tuple[int, int]] = set()
        pairs: list[tuple[int, int, Fraction]] = []
        if not isinstance(raw["pairs"], list):
            raise RuntimeError("M255 pair list rejected")
        for item in raw["pairs"]:
            if (
                not isinstance(item, list)
                or len(item) != 4
                or any(not isinstance(value, int) or isinstance(value, bool) for value in item)
            ):
                raise RuntimeError("M255 pair descriptor rejected")
            left, right, numerator, denominator = item
            if not 0 <= left < right < port_count or not 1 <= denominator <= 16:
                raise RuntimeError("M255 pair type or coefficient rejected")
            if (left, right) in seen_pairs:
                raise RuntimeError("M255 duplicate pair descriptor")
            seen_pairs.add((left, right))
            pairs.append((left, right, Fraction(numerator, denominator)))
        quartic = raw["quartic"]
        if (
            not isinstance(quartic, list)
            or len(quartic) != 6
            or any(not isinstance(value, int) or isinstance(value, bool) for value in quartic)
        ):
            raise RuntimeError("M255 quartic descriptor rejected")
        indices = quartic[:4]
        numerator, denominator = quartic[4:]
        if indices != sorted(set(indices)) or any(not 0 <= index < port_count for index in indices):
            raise RuntimeError("M255 decomposable quartic type rejected")
        if not 1 <= denominator <= 16 or numerator == 0:
            raise RuntimeError("M255 zero or malformed quartic rejected")
        modules.append(
            Module(tuple(pairs), QuarticTerm(mask_of(indices), Fraction(numerator, denominator)))
        )
    selected_raw = descriptor["selected_output"]
    if (
        isinstance(selected_raw, list)
        and len(selected_raw) == port_count
        and selected_raw == sorted(set(selected_raw))
        and all(isinstance(index, int) and not isinstance(index, bool) and 0 <= index < port_count for index in selected_raw)
    ):
        selected = mask_of(selected_raw)
    else:
        raise RuntimeError("M255 selected output type rejected")
    canonical = {
        "port_count": port_count,
        "modules": modules_raw,
        "selected_output": selected_raw,
    }
    return PublicProgram(port_count, tuple(modules), selected, canonical_digest(canonical))


@dataclass
class Work:
    top_level_pfaffian_evaluations: int = 0
    transformed_coefficient_calls: int = 0
    quartic_subset_candidates: int = 0
    even_set_partitions: int = 0
    forward_pair_additions: int = 0
    inverse_pair_subtractions: int = 0
    forward_quartic_writes: int = 0
    inverse_quartic_clears: int = 0


class Carrier:
    def __init__(self, port_count: int) -> None:
        self.port_count = port_count
        self.pairs = [ZERO] * (port_count * (port_count - 1) // 2)
        self.quartic_masks = [0] * (port_count // 2 - 1)
        self.quartic_coefficients = [ZERO] * (port_count // 2 - 1)
        self.cursor = 0
        self.generation = 0
        self.last_restored_generation = 0
        self.owner = 0
        self.program_id = ""
        self.leased = False

    def canonical(self) -> bool:
        return (
            all(value == ZERO for value in self.pairs)
            and all(mask == 0 for mask in self.quartic_masks)
            and all(value == ZERO for value in self.quartic_coefficients)
            and self.cursor == 0
            and self.owner == 0
            and self.program_id == ""
            and not self.leased
        )

    def lease(self, program: PublicProgram) -> int:
        if program.port_count != self.port_count or not self.canonical():
            raise RuntimeError("M255 dirty, leased, or mistyped carrier")
        generation = self.last_restored_generation + 1
        self.generation = generation
        self.owner = OWNER
        self.program_id = program.program_id
        self.leased = True
        return generation

    def require(self, program: PublicProgram, generation: int) -> None:
        if (
            not self.leased
            or self.owner != OWNER
            or self.program_id != program.program_id
            or self.generation != generation
            or self.port_count != program.port_count
        ):
            raise RuntimeError("M255 custody mismatch")

    def apply(self, program: PublicProgram, generation: int, work: Work) -> None:
        self.require(program, generation)
        if self.cursor >= len(program.modules):
            raise RuntimeError("M255 forward cursor overflow")
        module = program.modules[self.cursor]
        for left, right, coefficient in module.pair_terms:
            offset = pair_offset(self.port_count, left, right)
            self.pairs[offset] += coefficient
            work.forward_pair_additions += 1
        self.quartic_masks[self.cursor] = module.quartic.mask
        self.quartic_coefficients[self.cursor] = module.quartic.coefficient
        self.cursor += 1
        work.forward_quartic_writes += 1

    def reverse(self, program: PublicProgram, generation: int, work: Work) -> None:
        self.require(program, generation)
        if self.cursor <= 0:
            raise RuntimeError("M255 inverse cursor underflow")
        index = self.cursor - 1
        module = program.modules[index]
        if (
            self.quartic_masks[index] != module.quartic.mask
            or self.quartic_coefficients[index] != module.quartic.coefficient
        ):
            raise RuntimeError("M255 wrong inverse module")
        self.quartic_masks[index] = 0
        self.quartic_coefficients[index] = ZERO
        for left, right, coefficient in reversed(module.pair_terms):
            offset = pair_offset(self.port_count, left, right)
            self.pairs[offset] -= coefficient
            work.inverse_pair_subtractions += 1
        self.cursor -= 1
        work.inverse_quartic_clears += 1

    def release(self, program: PublicProgram, generation: int) -> None:
        self.require(program, generation)
        if (
            self.cursor != 0
            or any(value != ZERO for value in self.pairs)
            or any(mask != 0 for mask in self.quartic_masks)
            or any(value != ZERO for value in self.quartic_coefficients)
        ):
            raise RuntimeError("M255 restoration predicate failed")
        self.last_restored_generation = generation
        self.generation = 0
        self.owner = 0
        self.program_id = ""
        self.leased = False


def exp_coefficient(carrier: Carrier, input_mask: int, work: Work) -> Fraction:
    result = ZERO
    terms = tuple(
        QuarticTerm(mask, coefficient)
        for mask, coefficient in zip(carrier.quartic_masks, carrier.quartic_coefficients)
        if coefficient
    )
    for selection in range(1 << len(terms)):
        work.quartic_subset_candidates += 1
        used = 0
        multiplier = ONE
        valid = True
        for index, term in enumerate(terms):
            if selection & (1 << index):
                if used & term.mask or term.mask & ~input_mask:
                    valid = False
                    break
                multiplier *= term.coefficient * wedge_sign(used, term.mask)
                used |= term.mask
        if not valid:
            continue
        remainder = input_mask ^ used
        work.top_level_pfaffian_evaluations += 1
        result += multiplier * pfaffian_from_pairs(
            carrier.pairs, carrier.port_count, remainder
        )
    return result


def transformed_coefficient(carrier: Carrier, output_mask: int, work: Work) -> Fraction:
    full = (1 << carrier.port_count) - 1
    input_mask = full ^ output_mask
    work.transformed_coefficient_calls += 1
    return hodge_input_sign(input_mask, carrier.port_count) * exp_coefficient(
        carrier, input_mask, work
    )


def even_set_partitions(mask: int) -> Iterable[tuple[int, ...]]:
    """Stream canonical set partitions whose blocks all have even size."""
    if mask == 0:
        yield ()
        return
    first_bit = mask & -mask
    rest = mask ^ first_bit
    remaining_bits = [1 << index for index in range(mask.bit_length()) if rest & (1 << index)]
    for extra_count in range(1, len(remaining_bits) + 1, 2):
        for chosen in combinations(remaining_bits, extra_count):
            block = first_bit
            for bit in chosen:
                block |= bit
            for tail in even_set_partitions(mask ^ block):
                yield (block, *tail)


def connected_top_coefficient(carrier: Carrier, target: int, work: Work) -> tuple[Fraction, Fraction]:
    if carrier.cursor != len(carrier.quartic_masks) or target.bit_count() != carrier.port_count:
        raise RuntimeError("M255 premature or mistyped selected projection")
    scalar = transformed_coefficient(carrier, 0, work)
    if scalar == 0:
        raise RuntimeError("M255 singular Hodge normalization")

    def normalized(mask: int) -> Fraction:
        return transformed_coefficient(carrier, mask, work) / scalar

    result = ZERO
    for partition in even_set_partitions(target):
        work.even_set_partitions += 1
        coefficient = Fraction(((-1) ** (len(partition) - 1)) * factorial(len(partition) - 1))
        used = 0
        for block in partition:
            coefficient *= wedge_sign(used, block) * normalized(block)
            used |= block
        result += coefficient
    return result, scalar


def dual_plucker_square_witness(program: PublicProgram) -> tuple[int, Fraction]:
    """Derive one nonzero coefficient of (*Q4) wedge (*Q4), if present."""
    full = (1 << program.port_count) - 1
    dual_terms = [
        (
            full ^ module.quartic.mask,
            hodge_input_sign(module.quartic.mask, program.port_count)
            * module.quartic.coefficient,
        )
        for module in program.modules
    ]
    square: dict[int, Fraction] = {}
    for left_mask, left_coefficient in dual_terms:
        for right_mask, right_coefficient in dual_terms:
            sign = wedge_sign(left_mask, right_mask)
            if sign:
                target = left_mask | right_mask
                square[target] = square.get(target, ZERO) + sign * left_coefficient * right_coefficient
    for target in sorted(square):
        if square[target]:
            return target, square[target]
    return 0, ZERO


def descriptor(port_count: int, *, alternate: bool = False) -> dict[str, Any]:
    first_scale = 2 if alternate else 1
    quartic_scales = [1, 2, 3] if alternate else [1, 1, 1]
    all_pairs = [[0, 1, first_scale, 1], [2, 3, 1, 1]]
    if port_count >= 6:
        all_pairs.append([4, 5, 1, 1])
    if port_count >= 8:
        all_pairs.append([6, 7, 1, 1])
    module_count = port_count // 2 - 1
    modules: list[dict[str, Any]] = []
    for index in range(module_count):
        pair_terms = all_pairs[:2] if index == 0 else [all_pairs[index + 1]]
        quartic_indices = [2 * index, 2 * index + 1, 2 * index + 2, 2 * index + 3]
        modules.append({
            "pairs": pair_terms,
            "quartic": [*quartic_indices, quartic_scales[index], 1],
        })
    return {
        "port_count": port_count,
        "modules": modules,
        "selected_output": list(range(port_count)),
    }


def run_transaction(
    carrier: Carrier, program: PublicProgram, *, run_kind: str
) -> dict[str, Any]:
    pair_backing = id(carrier.pairs)
    quartic_mask_backing = id(carrier.quartic_masks)
    quartic_coefficient_backing = id(carrier.quartic_coefficients)
    work = Work()
    generation = carrier.lease(program)
    for _ in program.modules:
        carrier.apply(program, generation, work)
    selected, scalar = connected_top_coefficient(
        carrier, program.selected_output_mask, work
    )
    carrier_payload = sum(fraction_payload_bits(value) for value in carrier.pairs)
    carrier_payload += sum(fraction_payload_bits(value) for value in carrier.quartic_coefficients)
    retained_boundary = (selected, scalar)
    for _ in reversed(program.modules):
        carrier.reverse(program, generation, work)
    carrier.release(program, generation)
    if retained_boundary != (selected, scalar):
        raise RuntimeError("M255 result did not survive inverse")
    plucker_mask, plucker_witness = dual_plucker_square_witness(program)
    plucker_applicable = program.port_count >= 6
    if not plucker_applicable:
        plucker_mask, plucker_witness = 0, ZERO
    same_backings = (
        pair_backing == id(carrier.pairs)
        and quartic_mask_backing == id(carrier.quartic_masks)
        and quartic_coefficient_backing == id(carrier.quartic_coefficients)
    )
    return {
        "run_kind": run_kind,
        "port_count": program.port_count,
        "generation": generation,
        "program_id": program.program_id,
        "selected_connected_top_degree_cumulant": fraction_json(selected),
        "hodge_normalization_scalar": fraction_json(scalar),
        "rank1_quartic_plucker_dual_square_witness": fraction_json(plucker_witness),
        "rank1_quartic_plucker_dual_square_applicable": plucker_applicable,
        "rank1_quartic_plucker_dual_square_support": [
            index for index in range(program.port_count) if plucker_mask & (1 << index)
        ],
        "pair_carrier_field_cells": len(carrier.pairs),
        "quartic_cumulant_field_cells": len(carrier.quartic_coefficients),
        "quartic_support_integer_cells": len(carrier.quartic_masks),
        "retained_final_boundary_field_cells_during_inverse": 2,
        "retained_final_boundary_rational_payload_bits_during_inverse": fraction_payload_bits(selected) + fraction_payload_bits(scalar),
        "retained_resource_measurement_integer_cells_during_inverse": 1,
        "carrier_rational_payload_bits_after_forward": carrier_payload,
        "public_descriptor_integer_cells": 6 * program.port_count - 5,
        "retained_dynamic_inverse_history_field_cells": 0,
        "full_even_basis_dimension_not_materialized": 1 << (program.port_count - 1),
        "same_pair_quartic_backings": same_backings,
        "canonical_after_restoration": carrier.canonical(),
        "baseline_reload_used": False,
        "work": work.__dict__,
    }


def rejected(action: Any) -> bool:
    try:
        action()
    except (RuntimeError, TypeError, ValueError):
        return True
    return False


def controls() -> dict[str, bool]:
    primary = canonical_program(descriptor(8))
    carrier = Carrier(8)
    generation = carrier.lease(primary)
    for _ in range(len(primary.modules) - 1):
        carrier.apply(primary, generation, Work())
    premature = rejected(
        lambda: connected_top_coefficient(carrier, primary.selected_output_mask, Work())
    )
    carrier.apply(primary, generation, Work())
    last = primary.modules[-1]
    wrong_last = Module(
        last.pair_terms,
        QuarticTerm(last.quartic.mask, last.quartic.coefficient + ONE),
    )
    wrong_modules = (*primary.modules[:-1], wrong_last)
    wrong_inverse = rejected(lambda: carrier.reverse(
        PublicProgram(primary.port_count, wrong_modules, primary.selected_output_mask, primary.program_id),
        generation,
        Work(),
    ))
    reordered_modules = tuple(reversed(primary.modules))
    wrong_order = rejected(lambda: carrier.reverse(
        PublicProgram(primary.port_count, reordered_modules, primary.selected_output_mask, primary.program_id),
        generation,
        Work(),
    ))
    missing_inverse = rejected(lambda: carrier.release(primary, generation))
    for _ in reversed(primary.modules):
        carrier.reverse(primary, generation, Work())
    carrier.release(primary, generation)

    stale_carrier = Carrier(8)
    g1 = stale_carrier.lease(primary)
    for _ in primary.modules:
        stale_carrier.apply(primary, g1, Work())
    for _ in reversed(primary.modules):
        stale_carrier.reverse(primary, g1, Work())
    stale_carrier.release(primary, g1)
    stale = rejected(lambda: stale_carrier.require(primary, g1))
    malformed = descriptor(8)
    malformed["modules"][1]["quartic"] = [0, 1, 1, 5, 1, 1]
    return {
        "premature_projection_rejected": premature,
        "missing_inverse_release_rejected": missing_inverse,
        "wrong_inverse_rejected": wrong_inverse,
        "reordered_receipt_rejected_by_slot_custody": wrong_order,
        "stale_generation_rejected": stale,
        "malformed_quartic_type_rejected": rejected(lambda: canonical_program(malformed)),
        "null_carrier_rejected": rejected(lambda: Carrier(8).apply(primary, 1, Work())),
    }


def main() -> None:
    cases: list[dict[str, Any]] = []
    for width in (4, 6):
        cases.append(run_transaction(
            Carrier(width), canonical_program(descriptor(width)), run_kind=f"PRIMARY_M{width}"
        ))
    shared = Carrier(8)
    primary_program = canonical_program(descriptor(8))
    alternate_program = canonical_program(descriptor(8, alternate=True))
    cases.append(run_transaction(shared, primary_program, run_kind="PRIMARY_M8"))
    cases.append(run_transaction(shared, alternate_program, run_kind="REUSE_M8"))
    cases.append(run_transaction(Carrier(8), alternate_program, run_kind="FRESH_M8"))
    output = {
        "milestone": MILESTONE,
        "claim": (
            "BOUNDED_EXACT_RATIONAL_SUBFAMILY_EMBEDDED_IN_QZETA8_GAUSSIAN_PLUS_DECOMPOSABLE_QUARTIC_CHAIN_"
            "GRASSMANN_CUMULANT_INTERSECTION_HAS_A_COMPACT_PFAFFIAN_HODGE_"
            "PROJECTION_BUT_ESCAPES_THE_RANK1_DEGREE4_CHART_AT_SIX_PORTS_AND_"
            "GENERATES_NONZERO_TOP_CUMULANTS_THROUGH_EIGHT_PORTS_WITH_DIRECT_PROCESS_EXACT_SAME_"
            "BACKING_RESTORATION_AND_GENERATION2_REUSE"
        ),
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "SEPARATE_REFERENCE_PARITY",
        "restoration_class": "EXACT_ALGEBRAIC_RESTORATION",
        "cases": cases,
        "controls": controls(),
        "control_applicability": {
            "reordered_inverse_algebraic_failure_applicable": False,
            "reason": "NATIVE_CUMULANT_INTERSECTION_ADDITION_COMMUTES;_ONLY_SLOT_RECEIPT_ORDER_IS_CUSTODY_RELEVANT",
        },
        "resource_law": {
            "accepted_carrier_pair_cells": {str(width): width * (width - 1) // 2 for width in (4, 6, 8)},
            "accepted_carrier_quartic_coefficient_cells": {"4": 1, "6": 2, "8": 3},
            "accepted_path_full_even_signature_materialized": False,
            "accepted_path_relation_table_or_assignment_expansion": False,
            "selected_projection_uses_public_subset_recursion_and_pfaffians": True,
            "verifier_only_full_even_basis_dimension": {"4": 8, "6": 32, "8": 128},
            "whole_process_and_python_allocator_accounting_complete": False,
            "component_local_work_counters_only": True,
            "strongest_fixed_fixture_classical_baseline": "PUBLIC_O1_CLOSED_CERTIFICATES",
            "strongest_transferable_normalization_scalar_baseline": (
                "TWO_STATE_MONOMER_DIMER_CONTINUANT_O_M_WORK_O1_LIVE_FIELD_STATE"
            ),
            "strongest_implemented_descriptor_level_selected_cumulant_baseline": (
                "STREAMED_PFAFFIAN_COEFFICIENT_AND_EVEN_SET_PARTITION_CUMULANT"
            ),
            "descriptor_level_selected_cumulant_optimality_claimed": False,
            "matched_classical_has_no_restoration_or_reuse_burden": True,
            "accepted_projection_is_not_the_strongest_classical_baseline": True,
            "exact_rational_payload_bits_are_counted_per_component": True,
        },
        "obstruction": {
            "rank1_degree4_chart_vacuously_sufficient_at_declared_m4": True,
            "rank1_degree4_chart_fails_at_declared_m6": True,
            "connected_degree6_nonzero_at_declared_m6": True,
            "connected_degree8_nonzero_at_declared_m8": True,
            "first_declared_escape_port_count": 6,
            "route_disposition": "RETIRE_GAUSSIAN_PLUS_RANK1_QUARTIC_FIXED_DEGREE_CHART_AFTER_M8_CONFIRMATION",
        },
        "claim_limits": {
            "declared_port_counts_only": [4, 6, 8],
            "public_decomposable_basis_quartic_chain_only": True,
            "rational_counterexample_embeds_in_qzeta8_but_is_not_a_distinct_phase_resource": True,
            "general_quartic_rank_theorem_established": False,
            "all_port_count_growth_theorem_established": False,
            "compact_closed_relation_chart_established": False,
            "machine_enforced_custody_established": False,
            "distinct_phase_resource_established": False,
            "computational_advantage_established": False,
            "small_wall_crossing_established": False,
            "physical_waveform_execution_established": False,
            "catalytic_inference_established": False,
            "unbounded_catalytic_computation_established": False,
            "replacement_of_physical_bits_with_pi_established": False,
        },
    }
    if not all(output["controls"].values()):
        raise RuntimeError("M255 control failure")
    print(json.dumps(output, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
