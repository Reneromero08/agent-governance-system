#!/usr/bin/env python3
"""M219 exact topology-local cubic shear on the M218 skein carrier.

Each public braid first acts by the native Kauffman/Temperley--Lieb skein
operator.  A triangular cubic shear then scans the same public local
reconnection map.  Coefficients on link patterns without the local cup are
controls; the corresponding cup coefficients are targets.  Controls and
targets are disjoint, so subtraction of the same cubes is an exact inverse
without retained value history.

The experiment asks whether changing the update law, rather than merely the
linear basis, produces compact exact closure.  Logical link-pattern cell count
is fixed in each declared case, while exact coefficient height and the generic
polynomial degree are measured explicitly.  The identical classical
polynomial recurrence is retained as the strongest matched baseline.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import su2_level8_fusion_path_braid_phase_relation as braid
import su2_level8_markov_skein_krylov as skein


sys.set_int_max_str_digits(0)

EXACT_CASES = (
    *((4, rounds, 0) for rounds in range(1, 7)),
    *((6, rounds, 0) for rounds in range(1, 4)),
    *((8, rounds, 0) for rounds in range(1, 3)),
)
GENERIC_DEGREE_CASES = (
    *((4, rounds, 0) for rounds in range(1, 8)),
    *((6, rounds, 0) for rounds in range(1, 5)),
    *((8, rounds, 0) for rounds in range(1, 3)),
)
PRIMARY = (6, 3, 0)
REUSE = (6, 2, 1)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def shear_phase_power(operation: braid.BraidOperation) -> int:
    if operation.exponent == 1:
        return (7 + 2 * operation.generator) % braid.ROOT_ORDER
    if operation.exponent == -1:
        return (13 + 2 * operation.generator) % braid.ROOT_ORDER
    raise ValueError("cubic shear braid exponent must be plus or minus one")


@dataclass
class Work:
    linear: skein.Work = field(default_factory=skein.Work)
    cubic_source_scans: int = 0
    cubic_updates: int = 0
    cubic_field_multiplications: int = 0
    cubic_field_additions: int = 0
    forward_operations: int = 0
    inverse_operations: int = 0

    def observe(self, coefficients: list[braid.K], scratch: list[braid.K]) -> None:
        self.linear.observe(coefficients, scratch)

    def as_dict(self) -> dict[str, Any]:
        return {
            "linear_skein_and_boundary": self.linear.as_dict(),
            "cubic_source_scans": self.cubic_source_scans,
            "cubic_updates": self.cubic_updates,
            "cubic_field_multiplications": self.cubic_field_multiplications,
            "cubic_field_additions": self.cubic_field_additions,
            "forward_operations": self.forward_operations,
            "inverse_operations": self.inverse_operations,
        }


def apply_cubic_shear(
    coefficients: list[braid.K],
    topology: skein.DiagramTopology,
    operation: braid.BraidOperation,
    work: Work,
    *,
    inverse: bool = False,
    phase_offset: int = 0,
) -> None:
    targets = topology.e_targets[operation.generator]
    cup_flags = topology.e_delta_flags[operation.generator]
    phase = braid.K.zeta(shear_phase_power(operation) + phase_offset)
    if inverse:
        phase = braid.ZERO - phase
    sources = range(topology.dimension - 1, -1, -1) if inverse else range(topology.dimension)
    for source in sources:
        work.cubic_source_scans += 1
        if cup_flags[source]:
            continue
        target = targets[source]
        if not cup_flags[target]:
            raise RuntimeError("cubic shear target is not in the local-cup partition")
        value = coefficients[source]
        square = value * value
        cube = square * value
        injected = phase * cube
        coefficients[target] = coefficients[target] + injected
        work.cubic_field_multiplications += 3
        work.cubic_field_additions += 1
        work.cubic_updates += 1


def apply_forward_operation(
    coefficients: list[braid.K],
    scratch: list[braid.K],
    topology: skein.DiagramTopology,
    operation: braid.BraidOperation,
    work: Work,
) -> None:
    skein.apply_gate(coefficients, scratch, topology, operation, work.linear)
    apply_cubic_shear(coefficients, topology, operation, work)
    work.forward_operations += 1
    work.observe(coefficients, scratch)


def apply_inverse_operation(
    coefficients: list[braid.K],
    scratch: list[braid.K],
    topology: skein.DiagramTopology,
    operation: braid.BraidOperation,
    work: Work,
    *,
    phase_offset: int = 0,
) -> None:
    apply_cubic_shear(
        coefficients,
        topology,
        operation,
        work,
        inverse=True,
        phase_offset=phase_offset,
    )
    skein.apply_gate(
        coefficients,
        scratch,
        topology,
        braid.BraidOperation(operation.generator, -operation.exponent),
        work.linear,
    )
    work.inverse_operations += 1
    work.observe(coefficients, scratch)


Leading = tuple[tuple[int, ...], braid.K]


def add_leading(left: Leading | None, right: Leading) -> Leading:
    if left is None:
        return right
    left_key = (sum(left[0]), left[0])
    right_key = (sum(right[0]), right[0])
    if left_key > right_key:
        return left
    if right_key > left_key:
        return right
    coefficient = left[1] + right[1]
    if coefficient == braid.ZERO:
        raise RuntimeError("leading-term cancellation requires wider symbolic support")
    return left[0], coefficient


def generic_degree_case(strands: int, rounds: int, family: int) -> dict[str, Any]:
    topology = skein.DiagramTopology.compile(strands)
    dimension = topology.dimension
    leading: list[Leading] = []
    for index in range(dimension):
        exponent = [0] * dimension
        exponent[index] = 1
        leading.append((tuple(exponent), braid.ONE))
    program = braid.BraidProgram(strands, rounds, family)
    for step in range(program.steps):
        operation = program.operation(step)
        alpha, beta = braid.local_braid_scalars(operation.exponent)
        scratch: list[Leading | None] = [None] * dimension
        targets = topology.e_targets[operation.generator]
        flags = topology.e_delta_flags[operation.generator]
        for column, term in enumerate(leading):
            scratch[column] = add_leading(
                scratch[column], (term[0], alpha * term[1])
            )
            factor = skein.DELTA if flags[column] else braid.ONE
            row = targets[column]
            scratch[row] = add_leading(
                scratch[row], (term[0], beta * factor * term[1])
            )
        leading = [term for term in scratch if term is not None]
        if len(leading) != dimension:
            raise RuntimeError("linear leading-term propagation lost a coordinate")
        phase = braid.K.zeta(shear_phase_power(operation))
        for source in range(dimension):
            if flags[source]:
                continue
            term = leading[source]
            cubed = tuple(3 * value for value in term[0])
            coefficient = phase * term[1] * term[1] * term[1]
            row = targets[source]
            leading[row] = add_leading(leading[row], (cubed, coefficient))
    degrees = [sum(term[0]) for term in leading]
    digest = hashlib.sha256(
        "|".join(
            f"{','.join(map(str, term[0]))}:{term[1].token()}" for term in leading
        ).encode("ascii")
    ).hexdigest()
    return {
        "strands": strands,
        "rounds": rounds,
        "family": family,
        "link_pattern_cells": dimension,
        "coordinate_total_degrees": degrees,
        "maximum_total_degree": max(degrees),
        "expected_maximum_total_degree": 3 ** ((strands - 2) * rounds + 1),
        "leading_term_digest": digest,
        "leading_cancellation_encountered": False,
        "leading_exponent_integer_cells": dimension * dimension,
        "leading_coefficient_field_cells": dimension,
    }


def forward_state(
    program: braid.BraidProgram,
) -> tuple[skein.DiagramTopology, list[braid.K], list[braid.K], Work]:
    topology = skein.DiagramTopology.compile(program.strands)
    coefficients = skein.source_state(topology)
    scratch = [braid.ZERO] * topology.dimension
    work = Work()
    work.observe(coefficients, scratch)
    for step in range(program.steps):
        apply_forward_operation(
            coefficients, scratch, topology, program.operation(step), work
        )
    return topology, coefficients, scratch, work


def exact_case(strands: int, rounds: int, family: int) -> dict[str, Any]:
    program = braid.BraidProgram(strands, rounds, family)
    topology, coefficients, scratch, work = forward_state(program)
    boundary = skein.normalized_markov_boundary(coefficients, topology, work.linear)
    return {
        "strands": strands,
        "rounds": rounds,
        "family": family,
        "steps": program.steps,
        "link_pattern_cells": topology.dimension,
        "forward_nonzero_field_cells": sum(value != braid.ZERO for value in coefficients),
        "forward_payload_bits": braid.field_payload_bits(coefficients),
        **braid.maximum_coordinate_bits(coefficients),
        "forward_state_commitment": skein.state_commitment(coefficients),
        "boundary_commitment": braid.boundary_commitment(boundary),
    }


@dataclass
class OpenCubicSkeinPort:
    topology: skein.DiagramTopology
    coefficients: list[braid.K]
    scratch: list[braid.K]
    live: bool = False
    owner: int = 0
    lease_generation: int = 0
    cursor: int = 0
    expected_steps: int = 0
    public_program_commitment: str = ""
    topology_commitment: str = ""

    def lease(
        self, owner: int, generation: int, program: braid.BraidProgram, work: Work
    ) -> None:
        if self.live:
            raise RuntimeError("cubic skein port already live")
        if (
            len(self.coefficients) != self.topology.dimension
            or len(self.scratch) != self.topology.dimension
        ):
            raise ValueError("null or wrong-width cubic skein carrier")
        if program.strands != self.topology.strands or owner <= 0 or generation <= 0:
            raise ValueError("invalid cubic skein lease descriptor")
        self.live = True
        self.owner = owner
        self.lease_generation = generation
        self.cursor = 0
        self.expected_steps = program.steps
        self.public_program_commitment = skein.program_commitment(program)
        self.topology_commitment = self.topology.commitment
        work.linear.public_descriptor_hashes += 2
        work.linear.public_descriptor_integers_hashed += (
            3 + self.topology.retained_pairing_integer_cells
        )
        work.linear.port_leases += 1
        work.observe(self.coefficients, self.scratch)

    def require(
        self, owner: int, program: braid.BraidProgram, work: Work
    ) -> None:
        if not self.live:
            raise RuntimeError("cubic skein port is not live")
        if owner != self.owner:
            raise PermissionError("cubic skein owner mismatch")
        if skein.program_commitment(program) != self.public_program_commitment:
            raise ValueError("public cubic skein program mismatch")
        if self.topology.commitment != self.topology_commitment:
            raise ValueError("public cubic skein topology mismatch")
        work.linear.public_descriptor_hashes += 2
        work.linear.public_descriptor_integers_hashed += (
            3 + self.topology.retained_pairing_integer_cells
        )

    def forward(
        self, owner: int, program: braid.BraidProgram, index: int, work: Work
    ) -> None:
        self.require(owner, program, work)
        if index != self.cursor or index >= self.expected_steps:
            raise ValueError("forward cubic skein cursor mismatch")
        apply_forward_operation(
            self.coefficients,
            self.scratch,
            self.topology,
            program.operation(index),
            work,
        )
        self.cursor += 1

    def inverse(
        self, owner: int, program: braid.BraidProgram, index: int, work: Work
    ) -> None:
        self.require(owner, program, work)
        if index != self.cursor - 1:
            raise ValueError("inverse cubic skein cursor mismatch")
        apply_inverse_operation(
            self.coefficients,
            self.scratch,
            self.topology,
            program.operation(index),
            work,
        )
        self.cursor -= 1

    def project_final(
        self, owner: int, program: braid.BraidProgram, work: Work
    ) -> braid.K:
        self.require(owner, program, work)
        if self.cursor != self.expected_steps:
            raise PermissionError("nonfinal cubic skein projection rejected")
        return skein.normalized_markov_boundary(
            self.coefficients, self.topology, work.linear
        )

    def release(
        self, owner: int, program: braid.BraidProgram, work: Work
    ) -> int:
        self.require(owner, program, work)
        if self.cursor:
            raise RuntimeError("cubic skein port released before inverse")
        generation = self.lease_generation
        self.live = False
        self.owner = 0
        self.lease_generation = 0
        self.expected_steps = 0
        self.public_program_commitment = ""
        self.topology_commitment = ""
        self.scratch[:] = [braid.ZERO] * self.topology.dimension
        work.linear.port_releases += 1
        return generation


@dataclass
class Carrier:
    port: OpenCubicSkeinPort
    restoration_generation: int = 0


def canonical_restoration(
    carrier: Carrier, source: list[braid.K], generation: int
) -> bool:
    port = carrier.port
    return (
        port.coefficients == source
        and all(value == braid.ZERO for value in port.scratch)
        and not port.live
        and port.owner == 0
        and port.lease_generation == 0
        and port.cursor == 0
        and port.expected_steps == 0
        and port.public_program_commitment == ""
        and port.topology_commitment == ""
        and carrier.restoration_generation == generation
    )


def transaction(
    carrier: Carrier, source: list[braid.K], program: braid.BraidProgram
) -> tuple[dict[str, Any], Work]:
    coefficient_backing = id(carrier.port.coefficients)
    scratch_backing = id(carrier.port.scratch)
    generation = carrier.restoration_generation + 1
    owner = 219000 + generation
    work = Work()
    carrier.port.lease(owner, generation, program, work)
    for index in range(program.steps):
        carrier.port.forward(owner, program, index, work)
    boundary = carrier.port.project_final(owner, program, work)
    forward_state_commitment = skein.state_commitment(carrier.port.coefficients)
    work.linear.state_commitment_hashes += 1
    work.linear.state_commitment_field_cells_hashed += len(
        carrier.port.coefficients
    )
    work.linear.boundary_commitment_hashes += 1
    forward_payload = braid.field_payload_bits(carrier.port.coefficients)
    forward_nonzero = sum(value != braid.ZERO for value in carrier.port.coefficients)
    missing_inverse_error = sum(
        left != right
        for left, right in zip(carrier.port.coefficients, source, strict=True)
    ) + carrier.port.cursor
    for index in range(program.steps - 1, -1, -1):
        carrier.port.inverse(owner, program, index, work)
    carrier.restoration_generation = carrier.port.release(owner, program, work)
    return {
        "boundary_commitment": braid.boundary_commitment(boundary),
        "forward_state_commitment": forward_state_commitment,
        "forward_field_cells": len(carrier.port.coefficients),
        "forward_nonzero_field_cells": forward_nonzero,
        "forward_payload_bits": forward_payload,
        "missing_inverse_error_cells_and_cursor": missing_inverse_error,
        "restoration_error_field_cells": sum(
            left != right
            for left, right in zip(carrier.port.coefficients, source, strict=True)
        ),
        "same_coefficient_backing": id(carrier.port.coefficients) == coefficient_backing,
        "same_scratch_backing": id(carrier.port.scratch) == scratch_backing,
        "canonical_post_restoration_state_exact": canonical_restoration(
            carrier, source, generation
        ),
        "restoration_generation": carrier.restoration_generation,
        "baseline_reload_used": False,
        "work": work.as_dict(),
    }, work


def exact_transactions() -> dict[str, Any]:
    strands, rounds, family = PRIMARY
    topology = skein.DiagramTopology.compile(strands)
    source = skein.source_state(topology)
    carrier = Carrier(
        OpenCubicSkeinPort(
            topology, source.copy(), [braid.ZERO] * topology.dimension
        )
    )
    primary_program = braid.BraidProgram(strands, rounds, family)
    reuse_program = braid.BraidProgram(*REUSE)
    primary, _ = transaction(carrier, source, primary_program)
    reuse, _ = transaction(carrier, source, reuse_program)
    fresh = Carrier(
        OpenCubicSkeinPort(
            topology, source.copy(), [braid.ZERO] * topology.dimension
        )
    )
    fresh_reuse, _ = transaction(fresh, source, reuse_program)
    return {
        "primary_program": {"strands": strands, "rounds": rounds, "family": family},
        "primary": primary,
        "reuse_program": {
            "strands": REUSE[0], "rounds": REUSE[1], "family": REUSE[2]
        },
        "reuse": reuse,
        "fresh_reuse": fresh_reuse,
        "fresh_restored_reuse_boundary_agreement": (
            reuse["boundary_commitment"] == fresh_reuse["boundary_commitment"]
        ),
        "fresh_restored_reuse_state_agreement": (
            reuse["forward_state_commitment"] == fresh_reuse["forward_state_commitment"]
        ),
        "restoration_generation_after_reuse": carrier.restoration_generation,
        "baseline_reload_used": False,
    }


def controls() -> dict[str, bool]:
    topology = skein.DiagramTopology.compile(4)
    source = skein.source_state(topology)
    accepted = braid.BraidProgram(4, 2, 0)
    wrong = braid.BraidProgram(4, 2, 1)
    port = OpenCubicSkeinPort(
        topology, source.copy(), [braid.ZERO] * topology.dimension
    )
    work = Work()
    port.lease(219900, 1, accepted, work)
    wrong_owner = wrong_type = premature = wrong_program = reordered = False
    try:
        port.forward(219901, accepted, 0, work)
    except PermissionError:
        wrong_owner = True
    try:
        port.forward(219900, object(), 0, work)  # type: ignore[arg-type]
    except (AttributeError, TypeError, ValueError):
        wrong_type = True
    try:
        port.project_final(219900, accepted, work)
    except PermissionError:
        premature = True
    for index in range(accepted.steps):
        port.forward(219900, accepted, index, work)
    nonlinear_boundary = port.project_final(219900, accepted, work)
    try:
        port.inverse(219900, wrong, accepted.steps - 1, work)
    except ValueError:
        wrong_program = True
    try:
        port.inverse(219900, accepted, accepted.steps - 2, work)
    except ValueError:
        reordered = True
    missing = port.coefficients != source and port.cursor == accepted.steps
    for index in range(accepted.steps - 1, -1, -1):
        port.inverse(219900, accepted, index, work)
    port.release(219900, accepted, work)

    wrong_phase = source.copy()
    wrong_phase_scratch = [braid.ZERO] * topology.dimension
    wrong_phase_work = Work()
    for index in range(accepted.steps):
        apply_forward_operation(
            wrong_phase,
            wrong_phase_scratch,
            topology,
            accepted.operation(index),
            wrong_phase_work,
        )
    for index in range(accepted.steps - 1, -1, -1):
        apply_inverse_operation(
            wrong_phase,
            wrong_phase_scratch,
            topology,
            accepted.operation(index),
            wrong_phase_work,
            phase_offset=2 if index == accepted.steps - 1 else 0,
        )

    omitted = source.copy()
    omitted_scratch = [braid.ZERO] * topology.dimension
    omitted_work = Work()
    for index in range(accepted.steps):
        apply_forward_operation(
            omitted, omitted_scratch, topology, accepted.operation(index), omitted_work
        )
    for index in range(accepted.steps - 1, -1, -1):
        operation = accepted.operation(index)
        skein.apply_gate(
            omitted,
            omitted_scratch,
            topology,
            braid.BraidOperation(operation.generator, -operation.exponent),
            omitted_work.linear,
        )

    linear = source.copy()
    linear_scratch = [braid.ZERO] * topology.dimension
    linear_work = skein.Work()
    for index in range(accepted.steps):
        skein.apply_gate(
            linear, linear_scratch, topology, accepted.operation(index), linear_work
        )
    linear_boundary = skein.normalized_markov_boundary(linear, topology, linear_work)

    noncommuting_source = source.copy()
    noncommuting_source_scratch = [braid.ZERO] * topology.dimension
    apply_forward_operation(
        noncommuting_source,
        noncommuting_source_scratch,
        topology,
        accepted.operation(0),
        Work(),
    )
    operation = accepted.operation(1)
    gate_then_shear = noncommuting_source.copy()
    gate_then_shear_scratch = [braid.ZERO] * topology.dimension
    apply_forward_operation(
        gate_then_shear,
        gate_then_shear_scratch,
        topology,
        operation,
        Work(),
    )
    shear_then_gate = noncommuting_source.copy()
    shear_then_gate_scratch = [braid.ZERO] * topology.dimension
    shear_then_gate_work = Work()
    apply_cubic_shear(shear_then_gate, topology, operation, shear_then_gate_work)
    skein.apply_gate(
        shear_then_gate,
        shear_then_gate_scratch,
        topology,
        operation,
        shear_then_gate_work.linear,
    )

    null_rejected = False
    try:
        OpenCubicSkeinPort(topology, [], []).lease(1, 1, accepted, Work())
    except ValueError:
        null_rejected = True
    return {
        "wrong_owner_rejected": wrong_owner,
        "wrong_operation_type_rejected": wrong_type,
        "premature_projection_rejected": premature,
        "wrong_public_program_inverse_rejected": wrong_program,
        "reordered_inverse_rejected": reordered,
        "missing_inverse_detected": missing,
        "wrong_shear_phase_detected": wrong_phase != source,
        "missing_shear_inverse_detected": omitted != source,
        "null_carrier_rejected": null_rejected,
        "cubic_shear_changes_declared_boundary": nonlinear_boundary != linear_boundary,
        "skein_and_cubic_shear_noncommute": gate_then_shear != shear_then_gate,
        "intermediate_link_pattern_state_projected": False,
        "snapshot_command_available": hasattr(port, "snapshot"),
    }


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(
            "usage: su2_level8_topology_local_cubic_skein.py SEPARATE_REFERENCE_JSON"
        )
    reference_path = Path(sys.argv[1]).resolve()
    if str(reference_path).startswith(("/dev/shm/", "/run/shm/")):
        raise ValueError("RAM-backed M219 reference is forbidden")
    reference = json.loads(reference_path.read_text())
    if reference.get("schema") != "cat_cas.su2_level8_topology_local_cubic_skein_reference.v1":
        raise RuntimeError("M219 separate-reference schema changed")

    exact_cases = [exact_case(*case) for case in EXACT_CASES]
    generic_cases = [generic_degree_case(*case) for case in GENERIC_DEGREE_CASES]
    comparable_exact = [
        {key: value for key, value in case.items() if key != "work"}
        for case in exact_cases
    ]
    if comparable_exact != reference.get("exact_cases"):
        raise RuntimeError("independent exact height tuples or boundaries disagree")
    if generic_cases != reference.get("generic_degree_cases"):
        raise RuntimeError("independent generic leading-degree propagation disagrees")

    transaction_result = exact_transactions()
    reference_transaction = reference.get("transaction", {})
    for key in (
        "boundary_commitment",
        "forward_state_commitment",
        "forward_payload_bits",
        "restoration_error_field_cells",
        "canonical_post_restoration_state_exact",
    ):
        if transaction_result["primary"][key] != reference_transaction.get("primary", {}).get(key):
            raise RuntimeError(f"independent primary transaction mismatch: {key}")
    if (
        transaction_result["reuse"]["boundary_commitment"]
        != reference_transaction.get("reuse", {}).get("boundary_commitment")
        or not reference_transaction.get("primary", {}).get("same_backing_identity")
        or not reference_transaction.get("reuse", {}).get("same_backing_identity")
    ):
        raise RuntimeError("independent restored-reuse transaction mismatch")

    exact_lookup = {
        (case["strands"], case["rounds"], case["family"]): case
        for case in exact_cases
    }
    degree_lookup = {
        (case["strands"], case["rounds"], case["family"]): case
        for case in generic_cases
    }
    selected = [
        {
            "strands": strands,
            "rounds": rounds,
            "link_pattern_cells": exact_lookup[(strands, rounds, 0)]["link_pattern_cells"],
            "exact_forward_payload_bits": exact_lookup[(strands, rounds, 0)]["forward_payload_bits"],
            "maximum_signed_numerator_bits": exact_lookup[(strands, rounds, 0)]["maximum_signed_numerator_bits"],
            "generic_maximum_total_degree": degree_lookup[(strands, rounds, 0)]["maximum_total_degree"],
        }
        for strands, rounds in ((4, 6), (6, 3), (8, 2))
    ]
    all_controls = controls()
    positive_controls = {
        key: value
        for key, value in all_controls.items()
        if key not in {
            "intermediate_link_pattern_state_projected",
            "snapshot_command_available",
        }
    }
    if (
        not all(positive_controls.values())
        or all_controls["intermediate_link_pattern_state_projected"]
        or all_controls["snapshot_command_available"]
    ):
        raise RuntimeError("M219 control failed")

    here = Path(__file__).resolve().parent
    result = {
        "schema": "cat_cas.su2_level8_topology_local_cubic_skein.v1",
        "result": "PASS_EXACT_TOPOLOGY_LOCAL_CUBIC_SKEIN_HEIGHT_OBSTRUCTION",
        "claim": "EXACT_TOPOLOGY_LOCAL_REVERSIBLE_CUBIC_CYCLOTOMIC_COEFFICIENT_SHEAR_INTERLEAVED_WITH_PRETRUNCATION_TEMPERLEY_LIEB_SKEIN_RETAINS_LOGICAL_CARRIER_CELLS2_5_14_PLUS_EQUAL_SKEIN_SCRATCH_FOR_DECLARED_STRANDS4_6_8_WITH_FINAL_ONLY_BOUNDARY_EXACT_RESTORATION_REUSE_BUT_GENERIC_MAP_POLYNOMIAL_DEGREE_REACHES1594323_AT4_STRANDS_DEPTH6_1594323_AT6_STRANDS_DEPTH3_AND1594323_AT8_STRANDS_DEPTH2_WHILE_EXACT_ONE_HOT_CARRIER_PAYLOAD_REACHES4669525_5323618_6949604_BITS_AND_THE_IDENTICAL_CLASSICAL_POLYNOMIAL_RECURRENCE_REMAINS",
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "SEPARATE_REFERENCE_PARITY",
        "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "claim_ceiling": "FORMAL_PRETRUNCATION_QZETA40_NONCROSSING_LINK_PATTERN_CUBIC_SHEAR_FAMILY0_STRANDS4_DEPTH1TO6_STRANDS6_DEPTH1TO3_STRANDS8_DEPTH1TO2_GENERIC_LEADING_TERMS_THROUGH_DEPTHS7_4_2_PRIMARY6_DEPTH3_REUSE6_DEPTH2_FAMILY1_DIRECT_PROCESS_ONLY",
        "mechanism": {
            "linear_stage": "NATIVE_KAUFFMAN_TEMPERLEY_LIEB_SKEIN_A_I_PLUS_A_INVERSE_E_I",
            "nonlinear_stage": "FOR_EACH_NONCUP_LINK_PATTERN_P_ADD_ZETA40_PUBLIC_POWER_TIMES_COEFFICIENT_P_CUBED_TO_ITS_LOCAL_CUP_RECONNECTION_TARGET",
            "triangular_inverse": "SUBTRACT_IDENTICAL_CONTROL_CUBES_BEFORE_INVERSE_SKEIN_GATE",
            "topology_compiler_reads_final_answer": False,
            "retained_value_history": 0,
            "jones_wenzl_truncation_active": False,
        },
        "exact_height_cases": exact_cases,
        "generic_degree_cases": generic_cases,
        "selected_obstruction_cases": selected,
        "degree_law": {
            "declared_formula": "3^((strands-2)*rounds+1)",
            "all_declared_cases_match_formula": all(
                case["maximum_total_degree"]
                == case["expected_maximum_total_degree"]
                for case in generic_cases
            ),
            "symbolic_support_materialized": False,
            "only_exact_leading_monomial_and_coefficient_retained_per_coordinate": True,
            "arbitrary_program_or_unbounded_depth_theorem": False,
        },
        "separate_reference": {
            "imports_m219_or_m218_production": reference.get("imports_m219_or_m218_production"),
            "exact_height_tuple_parity": True,
            "exact_state_and_boundary_commitment_parity": True,
            "generic_leading_degree_parity": True,
            "exact_primary_restoration_and_reuse_parity": True,
        },
        "transaction": transaction_result,
        "controls": all_controls,
        "resource_law": {
            "logical_carrier_cells_per_declared_strand_count": {"4": 2, "6": 5, "8": 14},
            "scratch_cells_equal_carrier_cells": True,
            "retained_inverse_value_history": 0,
            "exact_payload_bits_include_every_cyclotomic_numerator_and_denominator": True,
            "generic_degree_diagnostic_cells_per_coordinate": "ONE_EXPONENT_VECTOR_AND_ONE_QZETA40_LEADING_COEFFICIENT",
            "generic_degree_diagnostic_leading_exponent_integer_cells": "LINK_PATTERN_CELLS_SQUARED",
            "primary_public_topology_pairing_integers": 30,
            "primary_public_skein_action_records": 25,
            "primary_public_skein_action_integers": 50,
            "cubic_shear_uses_existing_public_target_and_cup_flag_plan": True,
            "controller_backend_traffic_bytes": 0,
            "snapshot_traffic_bytes": 0,
            "baseline_reload_bytes": 0,
            "excluded_not_zero": "PYTHON_CONTAINER_CAPACITY_ALLOCATOR_PROCESS_IMAGE_JSON_SERIALIZATION_TIMING_AND_WHOLE_PROCESS_PEAKS",
        },
        "matched_classical_baselines": {
            "strongest_compact": "IDENTICAL_LINK_PATTERN_VECTOR_SKEIN_THEN_TRIANGULAR_CUBIC_POLYNOMIAL_RECURRENCE",
            "same_logical_cells_exact_height_and_work_law": True,
            "phase_specific_reduction": False,
            "computational_advantage": False,
        },
        "claim_limits": {
            "fixed_bounded_width_exact_state_set_established": False,
            "compact_exact_height_closure_established": False,
            "generic_symbolic_polynomial_materialized": False,
            "catvm_custody": False,
            "distinct_phase_resource_established": False,
            "computational_advantage": False,
            "small_wall_crossed": False,
            "physical_waveform_execution": False,
            "physical_bit_replacement": False,
            "catalytic_inference_established": False,
            "unbounded_computation_established": False,
        },
        "source_dependencies": {
            "m218_skein_sha256": sha256_file(here / "su2_level8_markov_skein_krylov.py"),
            "m214_field_and_word_sha256": sha256_file(here / "su2_level8_fusion_path_braid_phase_relation.py"),
            "m219_wrapper_sha256": sha256_file(Path(__file__).resolve()),
            "m219_separate_reference_code_sha256": sha256_file(
                here / "su2_level8_topology_local_cubic_skein_separate_reference.py"
            ),
            "m219_separate_reference_result_sha256": sha256_file(reference_path),
        },
        "terminal": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
