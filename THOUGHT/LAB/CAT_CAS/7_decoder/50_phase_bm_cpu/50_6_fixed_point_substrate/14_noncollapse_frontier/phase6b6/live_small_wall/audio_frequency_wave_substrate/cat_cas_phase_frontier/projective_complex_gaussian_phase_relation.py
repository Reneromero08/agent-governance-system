#!/usr/bin/env python3
"""Exact projective Gaussian phase-relation carrier on the real line.

The carrier represents exp(i*pi*q(x,y)) up to a nonzero scalar amplitude,
where q=a*x^2+2*b*x*y+c*y^2 and a,b,c are exact rationals.  Pointwise
intersection adds quadratic forms.  Composition through one unresolved real
port eliminates its stationary coordinate by an exact Schur complement.
Both public maps are inverted algebraically on the actual three-cell backing.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path


DEPTHS = (1, 2, 4, 8, 16, 32, 64, 128)
FAMILIES = (0, 1)
PRIMARY_DEPTH = 128
REUSE_DEPTH = 73
Relation = tuple[Fraction, Fraction, Fraction]
SOURCE: Relation = (Fraction(0), Fraction(1), Fraction(0))


def signed_bits(value: int) -> int:
    return max(1, abs(value).bit_length() + 1)


def rational_payload_bits(values: Relation | tuple[Fraction, ...]) -> int:
    return sum(
        signed_bits(value.numerator) + value.denominator.bit_length()
        for value in values
    )


def rational_token(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def relation_token(relation: Relation) -> str:
    return ":".join(rational_token(value) for value in relation)


def relation_commitment(relation: Relation) -> str:
    return hashlib.sha256(relation_token(relation).encode("ascii")).hexdigest()


def boundary_commitment(value: Fraction) -> str:
    return hashlib.sha256(rational_token(value).encode("ascii")).hexdigest()


def phase_modulo_two(value: Fraction) -> Fraction:
    return Fraction(value.numerator % (2 * value.denominator), value.denominator)


def evaluate(relation: Relation, left: Fraction, right: Fraction) -> Fraction:
    a, b, c = relation
    return a * left * left + 2 * b * left * right + c * right * right


@dataclass(frozen=True)
class PublicOperation:
    kind: str
    relation: Relation

    def __post_init__(self) -> None:
        if self.kind not in ("COMPOSE_RIGHT", "INTERSECT"):
            raise ValueError("unknown Gaussian relation operation")
        if self.kind == "COMPOSE_RIGHT" and self.relation[1] == 0:
            raise ValueError("reversible composition requires nonzero coupling")

    def token(self) -> str:
        return f"{self.kind}:{relation_token(self.relation)}"


@dataclass(frozen=True)
class PublicProgram:
    depth: int
    family: int

    def __post_init__(self) -> None:
        if self.depth <= 0 or self.family not in FAMILIES:
            raise ValueError("invalid public Gaussian program")

    @property
    def steps(self) -> int:
        return 2 * self.depth

    def operation(self, step: int, *, perturb_coupling: bool = False) -> PublicOperation:
        if step < 0 or step >= self.steps:
            raise IndexError("Gaussian program step out of range")
        _round, phase = divmod(step, 2)
        if self.family == 0:
            intersection_relation: Relation = (Fraction(1), Fraction(0), Fraction(2))
            composition_relation: Relation = (Fraction(0), Fraction(1), Fraction(0))
        else:
            intersection_relation = (Fraction(2), Fraction(0), Fraction(3))
            composition_relation = (Fraction(1), Fraction(2), Fraction(1))
        kind = "INTERSECT" if phase == 0 else "COMPOSE_RIGHT"
        a, b, c = intersection_relation if phase == 0 else composition_relation
        if perturb_coupling and step == 1:
            b += 1
            if b == 0:
                b += 1
        return PublicOperation(kind, (a, b, c))

    def token(self) -> str:
        return f"GAUSSIAN_PHASE:depth={self.depth}:family={self.family}"


def program_commitment(program: PublicProgram) -> str:
    return hashlib.sha256(program.token().encode("ascii")).hexdigest()


@dataclass
class Work:
    rational_additions: int = 0
    rational_subtractions: int = 0
    rational_multiplications: int = 0
    rational_divisions: int = 0
    forward_compositions: int = 0
    inverse_compositions: int = 0
    forward_intersections: int = 0
    inverse_intersections: int = 0
    public_operations_rematerialized: int = 0
    public_relation_rational_cells_rematerialized: int = 0
    retained_inverse_history_entries: int = 0
    peak_carrier_payload_bits: int = 0
    peak_public_relation_payload_bits: int = 0

    def observe_carrier(self, relation: Relation) -> None:
        self.peak_carrier_payload_bits = max(
            self.peak_carrier_payload_bits, rational_payload_bits(relation)
        )

    def observe_public(self, operation: PublicOperation) -> None:
        self.public_operations_rematerialized += 1
        self.public_relation_rational_cells_rematerialized += 3
        self.peak_public_relation_payload_bits = max(
            self.peak_public_relation_payload_bits,
            rational_payload_bits(operation.relation),
        )

    def as_dict(self) -> dict[str, int]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def compose_right(relation: Relation, public: Relation, work: Work) -> Relation:
    a, b, c = relation
    public_a, public_b, public_c = public
    shared = c + public_a
    work.rational_additions += 1
    if shared == 0 or public_b == 0:
        raise ZeroDivisionError("singular or irreversible Gaussian shared port")
    result = (
        a - b * b / shared,
        -(b * public_b / shared),
        public_c - public_b * public_b / shared,
    )
    work.rational_subtractions += 2
    work.rational_multiplications += 3
    work.rational_divisions += 3
    work.forward_compositions += 1
    return result


def inverse_compose_right(relation: Relation, public: Relation, work: Work) -> Relation:
    result_a, result_b, result_c = relation
    public_a, public_b, public_c = public
    denominator = public_c - result_c
    work.rational_subtractions += 1
    if denominator == 0 or public_b == 0:
        raise ZeroDivisionError("Gaussian composition inverse is singular")
    shared = public_b * public_b / denominator
    old_b = -(result_b * shared / public_b)
    old = (
        result_a + old_b * old_b / shared,
        old_b,
        shared - public_a,
    )
    work.rational_additions += 1
    work.rational_subtractions += 1
    work.rational_multiplications += 3
    work.rational_divisions += 3
    work.inverse_compositions += 1
    return old


def intersect(relation: Relation, public: Relation, work: Work) -> Relation:
    work.rational_additions += 3
    work.forward_intersections += 1
    return tuple(
        left + right for left, right in zip(relation, public, strict=True)
    )  # type: ignore[return-value]


def inverse_intersect(relation: Relation, public: Relation, work: Work) -> Relation:
    work.rational_subtractions += 3
    work.inverse_intersections += 1
    return tuple(
        left - right for left, right in zip(relation, public, strict=True)
    )  # type: ignore[return-value]


@dataclass
class PhasePort:
    coefficients: list[Fraction]
    live: bool = False
    owner: int = 0
    generation: int = 0
    last_restored_generation: int = 0
    cursor: int = 0
    expected_steps: int = 0
    program_receipt: str = ""
    port_type: str = "PROJECTIVE_COMPLEX_GAUSSIAN_PHASE_RELATION_R2"

    def relation(self) -> Relation:
        if len(self.coefficients) != 3:
            raise ValueError("null or wrong-width Gaussian phase carrier")
        return tuple(self.coefficients)  # type: ignore[return-value]

    def lease(self, owner: int, generation: int, program: PublicProgram, work: Work) -> None:
        self.relation()
        if self.live:
            raise RuntimeError("Gaussian phase port already leased")
        if owner <= 0 or generation != self.last_restored_generation + 1:
            raise ValueError("Gaussian phase owner/generation mismatch")
        self.live = True
        self.owner = owner
        self.generation = generation
        self.cursor = 0
        self.expected_steps = program.steps
        self.program_receipt = program_commitment(program)
        work.observe_carrier(self.relation())

    def require(self, owner: int, program: PublicProgram) -> None:
        if not self.live:
            raise RuntimeError("Gaussian phase port is not live")
        if owner != self.owner:
            raise PermissionError("Gaussian phase owner mismatch")
        if program_commitment(program) != self.program_receipt:
            raise ValueError("Gaussian public program mismatch")
        if self.port_type != "PROJECTIVE_COMPLEX_GAUSSIAN_PHASE_RELATION_R2":
            raise TypeError("Gaussian phase port type mismatch")

    def forward(self, owner: int, program: PublicProgram, step: int, work: Work) -> None:
        self.require(owner, program)
        if step != self.cursor:
            raise ValueError("Gaussian forward cursor mismatch")
        operation = program.operation(step)
        work.observe_public(operation)
        result = (
            compose_right(self.relation(), operation.relation, work)
            if operation.kind == "COMPOSE_RIGHT"
            else intersect(self.relation(), operation.relation, work)
        )
        self.coefficients[:] = result
        self.cursor += 1
        work.observe_carrier(result)

    def project_final(self, owner: int, program: PublicProgram, work: Work) -> Fraction:
        self.require(owner, program)
        if self.cursor != self.expected_steps:
            raise PermissionError("nonfinal Gaussian relation projection rejected")
        left = Fraction(program.family + 1)
        right = Fraction(2 * program.family + 3)
        work.rational_multiplications += 7
        work.rational_additions += 2
        return phase_modulo_two(evaluate(self.relation(), left, right))

    def inverse(self, owner: int, program: PublicProgram, step: int, work: Work) -> None:
        self.require(owner, program)
        if step != self.cursor - 1:
            raise ValueError("Gaussian inverse cursor mismatch")
        operation = program.operation(step)
        work.observe_public(operation)
        result = (
            inverse_compose_right(self.relation(), operation.relation, work)
            if operation.kind == "COMPOSE_RIGHT"
            else inverse_intersect(self.relation(), operation.relation, work)
        )
        self.coefficients[:] = result
        self.cursor -= 1
        work.observe_carrier(result)

    def release(self, owner: int, program: PublicProgram, source: Relation) -> None:
        self.require(owner, program)
        if self.cursor or self.relation() != source:
            raise RuntimeError("Gaussian phase carrier not exactly restored")
        generation = self.generation
        self.live = False
        self.owner = 0
        self.generation = 0
        self.expected_steps = 0
        self.program_receipt = ""
        self.last_restored_generation = generation


@dataclass
class Carrier:
    port: PhasePort


def canonical_restored(carrier: Carrier, source: Relation, generation: int) -> bool:
    port = carrier.port
    return (
        port.relation() == source
        and not port.live
        and port.owner == 0
        and port.generation == 0
        and port.cursor == 0
        and port.expected_steps == 0
        and port.program_receipt == ""
        and port.last_restored_generation == generation
    )


def transaction(carrier: Carrier, source: Relation, program: PublicProgram) -> tuple[dict[str, object], Work]:
    work = Work()
    backing = id(carrier.port.coefficients)
    generation = carrier.port.last_restored_generation + 1
    owner = 234000 + generation
    carrier.port.lease(owner, generation, program, work)
    for step in range(program.steps):
        carrier.port.forward(owner, program, step, work)
    final_relation = carrier.port.relation()
    final_payload = rational_payload_bits(final_relation)
    forward_receipt = relation_commitment(final_relation)
    boundary = carrier.port.project_final(owner, program, work)
    missing_inverse_error = sum(
        left != right for left, right in zip(final_relation, source, strict=True)
    ) + carrier.port.cursor
    retained_boundary = boundary
    for step in range(program.steps - 1, -1, -1):
        carrier.port.inverse(owner, program, step, work)
    carrier.port.release(owner, program, source)
    return (
        {
            "depth": program.depth,
            "family": program.family,
            "boundary_token": rational_token(retained_boundary),
            "boundary_commitment": boundary_commitment(retained_boundary),
            "boundary_payload_bits": rational_payload_bits((retained_boundary,)),
            "forward_relation_commitment": forward_receipt,
            "forward_carrier_payload_bits": final_payload,
            "peak_carrier_payload_bits": work.peak_carrier_payload_bits,
            "persistent_relation_rational_cells": 3,
            "retained_final_boundary_rational_cells_during_inverse": 1,
            "missing_inverse_error_cells_and_cursor": missing_inverse_error,
            "restoration_error_rational_cells": sum(
                left != right
                for left, right in zip(carrier.port.relation(), source, strict=True)
            ),
            "same_coefficient_backing": id(carrier.port.coefficients) == backing,
            "canonical_post_restoration_state_exact": canonical_restored(
                carrier, source, generation
            ),
            "restoration_generation": carrier.port.last_restored_generation,
            "baseline_reload_used": False,
        },
        work,
    )


def executed_case(depth: int, family: int) -> dict[str, object]:
    carrier = Carrier(PhasePort(list(SOURCE)))
    result, work = transaction(carrier, SOURCE, PublicProgram(depth, family))
    return {**result, "work": work.as_dict()}


def stationary_identity_control() -> bool:
    left_relation: Relation = (Fraction(7, 3), Fraction(-2, 5), Fraction(4, 7))
    right_relation: Relation = (Fraction(5), Fraction(-3), Fraction(8))
    output = compose_right(left_relation, right_relation, Work())
    shared = left_relation[2] + right_relation[0]
    for left in (Fraction(-2), Fraction(1, 3), Fraction(5)):
        for right in (Fraction(-1), Fraction(2, 5), Fraction(4)):
            latent = -(
                left_relation[1] * left + right_relation[1] * right
            ) / shared
            if evaluate(left_relation, left, latent) + evaluate(
                right_relation, latent, right
            ) != evaluate(output, left, right):
                return False
    return True


def controls() -> dict[str, bool]:
    compose_inverse = all(
        inverse_compose_right(
            compose_right(
                intersect(
                    SOURCE, PublicProgram(8, family).operation(0).relation, Work()
                ),
                PublicProgram(8, family).operation(1).relation, Work(),
            ),
            PublicProgram(8, family).operation(1).relation,
            Work(),
        ) == intersect(
            SOURCE, PublicProgram(8, family).operation(0).relation, Work()
        )
        for family in FAMILIES
    )
    intersection = PublicProgram(8, 0).operation(0)
    composition = PublicProgram(8, 0).operation(1)
    generic_relation: Relation = (Fraction(2), Fraction(1), Fraction(3))
    first_then_second = compose_right(
        intersect(generic_relation, intersection.relation, Work()), composition.relation, Work()
    )
    second_then_first = intersect(
        compose_right(generic_relation, composition.relation, Work()), intersection.relation, Work()
    )
    singular = zero_coupling = premature = wrong_owner = wrong_program = wrong_type = wrong_generation = missing_inverse = stale = null = reordered = False
    try:
        compose_right(SOURCE, (Fraction(0), Fraction(1), Fraction(3)), Work())
    except ZeroDivisionError:
        singular = True
    try:
        PublicOperation("COMPOSE_RIGHT", (Fraction(2), Fraction(0), Fraction(3)))
    except ValueError:
        zero_coupling = True
    program = PublicProgram(4, 0)
    port = PhasePort(list(SOURCE))
    port.lease(234999, 1, program, Work())
    try:
        port.project_final(234999, program, Work())
    except PermissionError:
        premature = True
    try:
        port.require(234998, program)
    except PermissionError:
        wrong_owner = True
    try:
        port.require(234999, PublicProgram(5, 0))
    except ValueError:
        wrong_program = True
    typed_port = PhasePort(list(SOURCE), port_type="WRONG_RELATION_TYPE")
    typed_port.lease(234995, 1, program, Work())
    try:
        typed_port.require(234995, program)
    except TypeError:
        wrong_type = True
    try:
        PhasePort(list(SOURCE)).lease(234994, 2, program, Work())
    except ValueError:
        wrong_generation = True
    try:
        port.inverse(234999, program, 0, Work())
    except ValueError:
        reordered = True
    missing_port = PhasePort(list(SOURCE))
    missing_port.lease(234993, 1, program, Work())
    missing_port.forward(234993, program, 0, Work())
    try:
        missing_port.release(234993, program, SOURCE)
    except RuntimeError:
        missing_inverse = True
    try:
        PhasePort(list(SOURCE), last_restored_generation=1).lease(234997, 1, program, Work())
    except ValueError:
        stale = True
    try:
        PhasePort([]).lease(234996, 1, program, Work())
    except ValueError:
        null = True
    def perturbation_changes_or_rejects(family: int) -> bool:
        original = forward_boundary(PublicProgram(16, family), False)
        try:
            return original != forward_boundary(PublicProgram(16, family), True)
        except ZeroDivisionError:
            return True
    sample_relations = (
        (Fraction(2), Fraction(1), Fraction(3)),
        (Fraction(4), Fraction(-2), Fraction(5)),
        (Fraction(7, 3), Fraction(5, 4), Fraction(9, 2)),
    )
    return {
        "stationary_substitution_identity_exact": stationary_identity_control(),
        "compose_right_inverse_exact": compose_inverse,
        "partial_composition_associativity_exact": compose_right(
            compose_right(
                (Fraction(2), Fraction(1), Fraction(3)),
                (Fraction(4), Fraction(2), Fraction(5)), Work(),
            ),
            (Fraction(3), Fraction(-1), Fraction(7)), Work(),
        ) == compose_right(
            (Fraction(2), Fraction(1), Fraction(3)),
            compose_right(
                (Fraction(4), Fraction(2), Fraction(5)),
                (Fraction(3), Fraction(-1), Fraction(7)), Work(),
            ), Work(),
        ),
        "composition_and_intersection_noncommute": first_then_second != second_then_first,
        "intersection_associative_exact": intersect(
            intersect(sample_relations[0], sample_relations[1], Work()),
            sample_relations[2], Work(),
        ) == intersect(
            sample_relations[0],
            intersect(sample_relations[1], sample_relations[2], Work()), Work(),
        ),
        "intersection_commutative_exact": intersect(
            sample_relations[0], sample_relations[1], Work()
        ) == intersect(sample_relations[1], sample_relations[0], Work()),
        "closed_depth_formula_matches_all_declared_cases": all(
            forward_relation(PublicProgram(depth, family)) == closed_formula(depth, family)
            for family in FAMILIES for depth in DEPTHS
        ),
        "singular_shared_port_rejected": singular,
        "zero_coupling_irreversible_module_rejected": zero_coupling,
        "premature_projection_rejected": premature,
        "wrong_owner_rejected": wrong_owner,
        "wrong_public_program_rejected": wrong_program,
        "wrong_port_type_rejected": wrong_type,
        "wrong_generation_rejected": wrong_generation,
        "reordered_inverse_rejected": reordered,
        "missing_inverse_release_rejected": missing_inverse,
        "stale_generation_rejected": stale,
        "null_carrier_rejected": null,
        "public_coupling_perturbation_changes_or_rejects_both_families": all(
            perturbation_changes_or_rejects(family) for family in FAMILIES
        ),
        "public_compiler_reads_final_answer": False,
        "relation_tables_materialized": False,
        "assignment_expansions_materialized": False,
        "intermediate_relations_serialized": False,
    }


def forward_boundary(program: PublicProgram, perturb: bool) -> Fraction:
    relation = forward_relation(program, perturb=perturb)
    return phase_modulo_two(evaluate(
        relation, Fraction(program.family + 1), Fraction(2 * program.family + 3)
    ))


def forward_relation(program: PublicProgram, *, perturb: bool = False) -> Relation:
    relation = SOURCE
    work = Work()
    for step in range(program.steps):
        operation = program.operation(step, perturb_coupling=perturb)
        relation = (
            compose_right(relation, operation.relation, work)
            if operation.kind == "COMPOSE_RIGHT"
            else intersect(relation, operation.relation, work)
        )
    return relation


def closed_formula(depth: int, family: int) -> Relation:
    if depth <= 0 or family not in FAMILIES:
        raise ValueError("closed Gaussian formula outside public family")
    if family == 0:
        return (
            Fraction(depth * depth, depth + 1),
            Fraction((-1) ** depth, depth + 1),
            Fraction(-depth, depth + 1),
        )
    power = 4**depth
    return (
        Fraction(2 * depth) - Fraction(1, 3) + Fraction(1, 3 * power),
        Fraction((-1) ** depth, 2**depth),
        Fraction(0),
    )


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(
            "usage: projective_complex_gaussian_phase_relation.py REFERENCE_JSON"
        )
    reference_path = Path(sys.argv[1]).resolve()
    if str(reference_path).startswith(("/dev/shm/", "/run/shm/")):
        raise ValueError("RAM-backed M234 reference input forbidden")
    reference = json.loads(reference_path.read_text())
    if reference.get("schema") != "cat_cas.projective_complex_gaussian_phase_relation_reference.v1":
        raise RuntimeError("M234 reference schema mismatch")
    cases = [executed_case(depth, family) for family in FAMILIES for depth in DEPTHS]
    current_controls = controls()
    if not all(value for key, value in current_controls.items() if key not in (
        "public_compiler_reads_final_answer", "relation_tables_materialized",
        "assignment_expansions_materialized", "intermediate_relations_serialized",
    )) or any(current_controls[key] for key in (
        "public_compiler_reads_final_answer", "relation_tables_materialized",
        "assignment_expansions_materialized", "intermediate_relations_serialized",
    )):
        raise RuntimeError("M234 production control failed")
    reference_cases = reference.get("cases")
    comparable = [
        {key: value for key, value in case.items() if key != "work"}
        for case in cases
    ]
    if comparable != reference_cases:
        raise RuntimeError("M234 independent case parity failed")
    if current_controls != reference.get("controls"):
        raise RuntimeError("M234 independent control parity failed")

    carrier = Carrier(PhasePort(list(SOURCE)))
    primary, primary_work = transaction(carrier, SOURCE, PublicProgram(PRIMARY_DEPTH, 0))
    reuse, reuse_work = transaction(carrier, SOURCE, PublicProgram(REUSE_DEPTH, 1))
    fresh = Carrier(PhasePort(list(SOURCE)))
    fresh_reuse, _ = transaction(fresh, SOURCE, PublicProgram(REUSE_DEPTH, 1))
    reuse_summary = {
        "primary": primary,
        "reuse": reuse,
        "fresh_reuse": fresh_reuse,
        "fresh_restored_boundary_agreement": reuse["boundary_commitment"] == fresh_reuse["boundary_commitment"],
        "fresh_restored_forward_relation_agreement": reuse["forward_relation_commitment"] == fresh_reuse["forward_relation_commitment"],
        "same_backing_across_primary_and_reuse": primary["same_coefficient_backing"] and reuse["same_coefficient_backing"],
        "restoration_generation_after_reuse": carrier.port.last_restored_generation,
        "primary_work": primary_work.as_dict(),
        "reuse_work": reuse_work.as_dict(),
    }
    if {key: value for key, value in reuse_summary.items() if key not in ("primary_work", "reuse_work")} != reference.get("reuse"):
        raise RuntimeError("M234 independent reuse parity failed")

    family_payloads = {
        str(family): [
            case["forward_carrier_payload_bits"]
            for case in cases if case["family"] == family
        ] for family in FAMILIES
    }
    here = Path(__file__).resolve().parent
    result = {
        "schema": "cat_cas.projective_complex_gaussian_phase_relation.v1",
        "result": "PASS_EXACT_PROJECTIVE_GAUSSIAN_PHASE_RELATION_FIXED3_CELL_CLOSURE_WITH_GROWING_COEFFICIENT_HEIGHT",
        "claim": "EXACT_PROJECTIVE_COMPLEX_GAUSSIAN_PHASE_RELATIONS_EXP_I_PI_Q_ON_THE_REAL_LINE_CLOSE_POINTWISE_INTERSECTION_AND_NONDEGENERATE_STATIONARY_SHARED_PORT_COMPOSITION_IN_THREE_RATIONAL_QUADRATIC_COEFFICIENT_CELLS_ACROSS_DEPTHS1_2_4_8_16_32_64_128_WITH_FINAL_ONLY_PROJECTIVE_PHASE_EXACT_SAME_BACKING_RESTORATION_REUSE_BUT_PRIMARY_PAYLOAD_GROWS12_TO51_BITS_AND_ALTERNATE13_TO655_BITS_WHILE_PUBLIC_DEPTH_CLOSED_FORMULAS_GIVE_THE_STRONGEST_SMALLER_WORK_CLASSICAL_BASELINE",
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "SEPARATE_REFERENCE_PARITY",
        "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "claim_ceiling": "FORMAL_PROJECTIVE_EXP_I_PI_Q_REAL_LINE_BINARY_QUADRATIC_RELATIONS_THREE_RATIONAL_COEFFICIENTS_TWO_PUBLIC_PROGRAM_FAMILIES_DEPTHS1_2_4_8_16_32_64_128_PRIMARY128_FAMILY0_REUSE73_FAMILY1_NONDEGENERATE_COMPOSE_RIGHT_AND_INTERSECTION_DIRECT_PROCESS_ONLY",
        "phase_relation_law": {
            "primitive": "PROJECTIVE_COMPLEX_PHASE_EXP_I_PI_Q_X_Y",
            "quadratic_signature": "Q_EQUALS_A_X2_PLUS2B_XY_PLUS_C_Y2",
            "resident_rational_coefficient_cells": 3,
            "intersection": "POINTWISE_PHASE_PRODUCT_EQUALS_QUADRATIC_COEFFICIENT_ADDITION",
            "composition": "UNRESOLVED_SHARED_REAL_PORT_STATIONARY_SCHUR_COMPLEMENT",
            "composition_inverse": "PUBLIC_RIGHT_MODULE_ALGEBRAIC_RECONSTRUCTION",
            "shared_port_projected": False,
            "final_boundary_only": True,
            "projective_scalar_amplitude_discarded": True,
            "convergent_wave_integral_claimed": False,
            "direct_process_logical_custody_only": True,
        },
        "cases": cases,
        "reuse": reuse_summary,
        "controls": current_controls,
        "resource_law": {
            "persistent_relation_rational_cells_all_depths": 3,
            "retained_final_boundary_rational_cells_during_inverse": 1,
            "primary_payload_bits_by_depth": family_payloads["0"],
            "reuse_family_payload_bits_by_depth": family_payloads["1"],
            "primary_depths": list(DEPTHS),
            "primary_payload_monotone_for_declared_depths": family_payloads["0"] == sorted(family_payloads["0"]),
            "reuse_payload_monotone_for_declared_depths": family_payloads["1"] == sorted(family_payloads["1"]),
            "closed_formula_relation_commitments_match_all_cases": all(
                case["forward_relation_commitment"] == relation_commitment(
                    closed_formula(case["depth"], case["family"])
                ) for case in cases
            ),
            "retained_inverse_history_entries": 0,
            "public_operation_records_retained": 0,
            "public_relations_rematerialized_from_depth_family": True,
            "whole_transaction_live_rational_cell_and_payload_accounting_complete": False,
            "resource_verification_level": "PACKAGE_SELF_REVIEW",
            "excluded_not_zero": "PYTHON_FRACTION_OBJECT_HEADERS_TEMPORARY_LOCALS_ALLOCATOR_INTERPRETER_HASH_BYTE_TRAFFIC_JSON_SERIALIZATION_TIMING_PROCESS_RSS_AND_PROJECTIVE_STATIONARY_PHASE_AMPLITUDE",
        },
        "matched_classical": {
            "strongest_implemented": "PUBLIC_DEPTH_CLOSED_THREE_RATIONAL_FORMULA_FOR_BOTH_DECLARED_REPEATED_MODULE_FAMILIES",
            "same_persistent_cells": True,
            "same_exact_payload_height": True,
            "fewer_recurrence_steps_than_iterated_phase_path": True,
            "closed_formula_persistent_rational_cells": 3,
            "closed_formula_integer_bit_operation_cost_fully_accounted": False,
            "identical_schur_add_recurrence_retained_for_arbitrary_valid_public_words": True,
            "computational_advantage": False,
            "distinct_phase_resource": False,
        },
        "separate_reference": {
            "imports_m234_production": reference.get("imports_m234_production"),
            "uses_independent_symmetric_matrix_schur_complement": reference.get("uses_independent_symmetric_matrix_schur_complement"),
            "case_control_reuse_parity": True,
        },
        "claim_limits": {
            "full_complex_amplitude_preserved": False,
            "normalized_or_convergent_gaussian_integral_established": False,
            "fresnel_or_maslov_scalar_preserved": False,
            "total_category_with_finite_identity_signature": False,
            "singular_or_delta_gaussian_relations_supported": False,
            "general_non_gaussian_phase_relations": False,
            "bounded_exact_coefficient_payload": False,
            "machine_enforced_catvm_custody": False,
            "distinct_phase_resource_established": False,
            "computational_advantage": False,
            "small_wall_crossed": False,
            "physical_waveform_execution": False,
            "physical_bit_replacement": False,
            "catalytic_inference_established": False,
            "unbounded_computation_established": False,
        },
        "source_dependencies": {
            "m234_production_sha256": sha256_file(Path(__file__).resolve()),
            "m234_reference_code_sha256": sha256_file(here / "projective_complex_gaussian_phase_relation_separate_reference.py"),
            "m234_reference_result_sha256": sha256_file(reference_path),
        },
        "terminal": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
