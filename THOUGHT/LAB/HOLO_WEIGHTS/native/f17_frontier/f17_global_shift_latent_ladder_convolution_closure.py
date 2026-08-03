#!/usr/bin/env python3
"""Exact global-shift quotient for a growing two-row F17 latent ladder.

Two M126 grid-boundary functions become the horizontal difference kernels of
a labelled ``F17 x F17`` latent ladder.  On the declared global-shift-invariant
sector, each 289-state labelled message is exactly determined by the relative
coordinate ``r = bottom - top``.  The quotient transfer is a 17-coordinate
cyclic convolution followed by a public rung chirp.

The accepted carrier stores the compact convolution kernel and two reversible
17-coordinate messages.  It never materializes a 289-state labelled message
or transfer table.  This is a bounded direct-process diagnostic; compact
classical group-algebra software executes the identical quotient recurrence.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

import f17_coherent_latent_basis_mismatch_grid_closure as basis
import f17_nonlinear_canonical_mps_separator_chart as backend
import f17_paired_phase_basis_holographic_matchgate_closure as matchgate


PRIME = 17
GRID_N = 4
LATENT_DIMENSION = 17
EXACT_DEPTHS = (1, 2, 4, 8, 16)
STRUCTURAL_DEPTHS = (1, 2, 4, 8, 16, 32, 64, 128)
FAMILIES = ("PRIMARY", "REUSE")
FINITE_FIELDS = ((103, 72), (137, 16))


def fail(message: str) -> None:
    raise RuntimeError(message)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def integer(alg: backend.Algebra, value: int) -> Any:
    if alg.modulus:
        return value % alg.modulus
    return alg.domain.convert(value)


def negative(alg: backend.Algebra, value: Any) -> Any:
    return alg.sub(alg.zero, value)


@dataclass(frozen=True)
class LatentLadderProgram:
    depth: int
    family: str
    basis_program: basis.CoherentBasisProgram
    rung_chirp_exponents: tuple[int, ...]

    def fingerprint(self) -> str:
        return sha256_json(
            {
                "depth": self.depth,
                "family": self.family,
                "basis_program": self.basis_program.fingerprint(),
                "rung_chirp_exponents": self.rung_chirp_exponents,
            }
        )


def compile_program(depth: int, family: str) -> LatentLadderProgram:
    if depth not in STRUCTURAL_DEPTHS:
        fail("latent ladder depth is outside the declared scope")
    if family not in FAMILIES:
        fail("unknown latent ladder family")
    variant = 0 if family == "PRIMARY" else 1
    program = LatentLadderProgram(
        depth=depth,
        family=family,
        basis_program=basis.compile_program(GRID_N, family),
        rung_chirp_exponents=tuple(
            1 + ((5 * layer + 3 * variant + 2 * depth) % 16)
            for layer in range(depth)
        ),
    )
    validate_program(program)
    return program


def validate_program(program: LatentLadderProgram) -> None:
    if program.depth not in STRUCTURAL_DEPTHS:
        fail("latent ladder depth changed")
    if program.family not in FAMILIES:
        fail("latent ladder family changed")
    if program.basis_program.n != GRID_N:
        fail("latent ladder grid topology changed")
    if program.basis_program.family != program.family:
        fail("latent ladder basis family ownership changed")
    basis.validate_program(program.basis_program)
    if len(program.rung_chirp_exponents) != program.depth:
        fail("latent ladder rung schedule changed")
    if not all(1 <= value < PRIME for value in program.rung_chirp_exponents):
        fail("latent ladder rung phase is outside F17")


@dataclass
class LatentLadderCarrier:
    n: int
    topology_fingerprint: str
    alg: backend.Algebra
    cells: list[Any]
    generation: int = 0
    lease: str | None = None
    stage: str = "RESTORED"
    grid_load_additions: int = 0
    grid_unload_additions: int = 0
    kernel_load_additions: int = 0
    kernel_unload_additions: int = 0
    seed_load_additions: int = 0
    seed_unload_additions: int = 0
    module_boundary_evaluations: int = 0
    basis_mismatch_edge_contractions: int = 0
    cyclic_convolution_field_multiplications: int = 0
    cyclic_convolution_field_additions: int = 0
    rung_chirp_field_multiplications: int = 0
    reversible_message_field_additions: int = 0
    message_coordinate_swaps: int = 0
    projection_calls: int = 0
    maximum_resident_payload_bits: int = 0
    determinant_stats: matchgate.DeterminantStats = field(
        default_factory=matchgate.DeterminantStats
    )

    @classmethod
    def create(cls, alg: backend.Algebra) -> "LatentLadderCarrier":
        edges = matchgate.grid_edges(GRID_N)
        cells = [
            alg.zero
            for _ in range(2 * len(edges) + 3 * LATENT_DIMENSION)
        ]
        carrier = cls(
            n=GRID_N,
            topology_fingerprint=sha256_json(
                {
                    "grid_n": GRID_N,
                    "grid_edges": edges,
                    "latent_geometry": "TWO_ROW_F17_GLOBAL_SHIFT_QUOTIENT",
                }
            ),
            alg=alg,
            cells=cells,
        )
        carrier.observe_resident()
        return carrier

    def backing_identity(self) -> tuple[int, int]:
        return id(self), id(self.cells)

    def exact_zero(self) -> bool:
        return (
            all(value == self.alg.zero for value in self.cells)
            and self.lease is None
            and self.stage == "RESTORED"
        )

    def observe_resident(self) -> None:
        self.maximum_resident_payload_bits = max(
            self.maximum_resident_payload_bits,
            sum(self.alg.payload_bits(value) for value in self.cells),
        )

    def digest(self) -> str:
        return sha256_json(
            {
                "n": self.n,
                "topology": self.topology_fingerprint,
                "cells": [self.alg.serialize(value) for value in self.cells],
                "generation": self.generation,
                "lease": self.lease,
                "stage": self.stage,
            }
        )


def carrier_offsets(carrier: LatentLadderCarrier) -> tuple[int, int, int, int]:
    edge_count = len(matchgate.grid_edges(carrier.n))
    kernel_start = 2 * edge_count
    u_start = kernel_start + LATENT_DIMENSION
    w_start = u_start + LATENT_DIMENSION
    return edge_count, kernel_start, u_start, w_start


def reset_transaction_observation(carrier: LatentLadderCarrier) -> None:
    carrier.grid_load_additions = 0
    carrier.grid_unload_additions = 0
    carrier.kernel_load_additions = 0
    carrier.kernel_unload_additions = 0
    carrier.seed_load_additions = 0
    carrier.seed_unload_additions = 0
    carrier.module_boundary_evaluations = 0
    carrier.basis_mismatch_edge_contractions = 0
    carrier.cyclic_convolution_field_multiplications = 0
    carrier.cyclic_convolution_field_additions = 0
    carrier.rung_chirp_field_multiplications = 0
    carrier.reversible_message_field_additions = 0
    carrier.message_coordinate_swaps = 0
    carrier.projection_calls = 0
    carrier.maximum_resident_payload_bits = 0
    carrier.determinant_stats = matchgate.DeterminantStats()


def load_grid_weights(
    carrier: LatentLadderCarrier,
    program: LatentLadderProgram,
    *,
    inverse: bool = False,
) -> None:
    index = 0
    for row in program.basis_program.module_weight_exponents:
        for exponent in row:
            value = carrier.alg.power(exponent)
            delta = negative(carrier.alg, value) if inverse else value
            carrier.cells[index] = carrier.alg.add(carrier.cells[index], delta)
            index += 1
    edge_count, _, _, _ = carrier_offsets(carrier)
    if index != 2 * edge_count:
        fail("latent ladder grid payload changed")
    if inverse:
        carrier.grid_unload_additions += index
    else:
        carrier.grid_load_additions += index
    carrier.observe_resident()


def module_signatures(
    carrier: LatentLadderCarrier,
    program: LatentLadderProgram,
) -> tuple[list[Any], list[Any]]:
    left = [
        basis.module_boundary(carrier, program.basis_program, 0, latent)
        for latent in range(LATENT_DIMENSION)
    ]
    right = [
        basis.module_boundary(carrier, program.basis_program, 1, latent)
        for latent in range(LATENT_DIMENSION)
    ]
    return left, right


def load_convolution_kernel(
    carrier: LatentLadderCarrier,
    program: LatentLadderProgram,
    *,
    inverse: bool = False,
) -> None:
    left, right = module_signatures(carrier, program)
    _, kernel_start, _, _ = carrier_offsets(carrier)
    for shift in range(LATENT_DIMENSION):
        value = carrier.alg.zero
        for latent in range(LATENT_DIMENSION):
            term = carrier.alg.mul(
                left[latent],
                right[(latent + shift) % LATENT_DIMENSION],
            )
            value = carrier.alg.add(value, term)
        delta = negative(carrier.alg, value) if inverse else value
        carrier.cells[kernel_start + shift] = carrier.alg.add(
            carrier.cells[kernel_start + shift], delta
        )
    if inverse:
        carrier.kernel_unload_additions += LATENT_DIMENSION
    else:
        carrier.kernel_load_additions += LATENT_DIMENSION
    carrier.observe_resident()


def load_seed(carrier: LatentLadderCarrier, *, inverse: bool = False) -> None:
    _, _, _, w_start = carrier_offsets(carrier)
    value = negative(carrier.alg, carrier.alg.one) if inverse else carrier.alg.one
    carrier.cells[w_start] = carrier.alg.add(carrier.cells[w_start], value)
    if inverse:
        carrier.seed_unload_additions += 1
    else:
        carrier.seed_load_additions += 1
    carrier.observe_resident()


def convolution_action(
    carrier: LatentLadderCarrier,
    source_start: int,
    rung_exponent: int,
) -> list[Any]:
    _, kernel_start, _, _ = carrier_offsets(carrier)
    output = []
    for target in range(LATENT_DIMENSION):
        value = carrier.alg.zero
        for source in range(LATENT_DIMENSION):
            term = carrier.alg.mul(
                carrier.cells[
                    kernel_start + (target - source) % LATENT_DIMENSION
                ],
                carrier.cells[source_start + source],
            )
            value = carrier.alg.add(value, term)
        value = carrier.alg.mul(
            value,
            carrier.alg.power(rung_exponent * target * target),
        )
        output.append(value)
    carrier.cyclic_convolution_field_multiplications += (
        LATENT_DIMENSION * LATENT_DIMENSION
    )
    carrier.cyclic_convolution_field_additions += (
        LATENT_DIMENSION * LATENT_DIMENSION
    )
    carrier.rung_chirp_field_multiplications += LATENT_DIMENSION
    return output


def apply_reversible_layer(
    carrier: LatentLadderCarrier,
    rung_exponent: int,
    *,
    inverse: bool = False,
) -> None:
    _, _, u_start, w_start = carrier_offsets(carrier)
    if inverse:
        for latent in range(LATENT_DIMENSION):
            carrier.cells[u_start + latent], carrier.cells[w_start + latent] = (
                carrier.cells[w_start + latent],
                carrier.cells[u_start + latent],
            )
        output = convolution_action(carrier, w_start, rung_exponent)
        for latent, value in enumerate(output):
            carrier.cells[u_start + latent] = carrier.alg.sub(
                carrier.cells[u_start + latent], value
            )
    else:
        output = convolution_action(carrier, w_start, rung_exponent)
        for latent, value in enumerate(output):
            carrier.cells[u_start + latent] = carrier.alg.add(
                carrier.cells[u_start + latent], value
            )
        for latent in range(LATENT_DIMENSION):
            carrier.cells[u_start + latent], carrier.cells[w_start + latent] = (
                carrier.cells[w_start + latent],
                carrier.cells[u_start + latent],
            )
    carrier.reversible_message_field_additions += LATENT_DIMENSION
    carrier.message_coordinate_swaps += LATENT_DIMENSION
    carrier.observe_resident()


def forward(carrier: LatentLadderCarrier, program: LatentLadderProgram) -> None:
    if not isinstance(carrier, LatentLadderCarrier) or not carrier.exact_zero():
        fail("null, leased, or unrestored latent ladder carrier")
    validate_program(program)
    carrier.lease = program.fingerprint()
    carrier.stage = "FORWARD_ACTIVE"
    load_grid_weights(carrier, program)
    load_convolution_kernel(carrier, program)
    load_seed(carrier)
    for rung_exponent in program.rung_chirp_exponents:
        apply_reversible_layer(carrier, rung_exponent)
    carrier.stage = "FORWARD_COMPLETE"
    carrier.observe_resident()


def project_boundary(
    carrier: LatentLadderCarrier,
    program: LatentLadderProgram,
) -> Any:
    if carrier.stage != "FORWARD_COMPLETE" or carrier.lease != program.fingerprint():
        fail("only the completed owned latent ladder boundary may be projected")
    carrier.projection_calls += 1
    _, _, _, w_start = carrier_offsets(carrier)
    return carrier.cells[w_start]


def inverse(carrier: LatentLadderCarrier, program: LatentLadderProgram) -> None:
    if carrier.stage != "FORWARD_COMPLETE" or carrier.lease != program.fingerprint():
        fail("inverse latent ladder program does not own the carrier lease")
    carrier.stage = "INVERSE_ACTIVE"
    for rung_exponent in reversed(program.rung_chirp_exponents):
        apply_reversible_layer(carrier, rung_exponent, inverse=True)
    load_seed(carrier, inverse=True)
    load_convolution_kernel(carrier, program, inverse=True)
    load_grid_weights(carrier, program, inverse=True)
    carrier.lease = None
    carrier.stage = "RESTORED"
    carrier.generation += 1
    carrier.observe_resident()
    if not carrier.exact_zero():
        fail("actual inverse failed exact latent ladder carrier restoration")


RESOURCE_SIGNATURE_KEYS = (
    "resident_phase_field_cells",
    "resident_grid_weight_field_cells",
    "resident_convolution_kernel_field_cells",
    "resident_quotient_message_field_cells",
    "module_boundary_evaluations",
    "basis_mismatch_edge_contractions",
    "cyclic_convolution_field_multiplications",
    "cyclic_convolution_field_additions",
    "rung_chirp_field_multiplications",
    "reversible_message_field_additions",
    "message_coordinate_swaps",
    "maximum_named_transaction_transient_field_cells",
    "resident_carrier_restoration_class",
)


def resource_signature(transaction: dict[str, Any]) -> dict[str, Any]:
    signature = {key: transaction[key] for key in RESOURCE_SIGNATURE_KEYS}
    signature["determinant_stats"] = transaction["determinant_stats"]
    return signature


def execute_transaction(
    carrier: LatentLadderCarrier,
    program: LatentLadderProgram,
) -> dict[str, Any]:
    reset_transaction_observation(carrier)
    initial = carrier.digest()
    backing = carrier.backing_identity()
    generation = carrier.generation
    forward(carrier, program)
    boundary = project_boundary(carrier, program)
    inverse(carrier, program)
    edge_count = len(matchgate.grid_edges(GRID_N))
    determinant_dimension = GRID_N * GRID_N // 2
    transient_field_cells = max(
        2 * determinant_dimension * determinant_dimension + 38,
        LATENT_DIMENSION + 2,
    )
    serialized_boundary = carrier.alg.serialize(boundary)
    return {
        "depth": program.depth,
        "family": program.family,
        "program_fingerprint": program.fingerprint(),
        "boundary": serialized_boundary,
        "grid_n": GRID_N,
        "grid_edge_count": edge_count,
        "labelled_message_field_cells_per_register": PRIME * PRIME,
        "quotient_message_field_cells_per_register": PRIME,
        "labelled_to_quotient_reduction_factor": PRIME,
        "resident_phase_field_cells": len(carrier.cells),
        "resident_grid_weight_field_cells": 2 * edge_count,
        "resident_convolution_kernel_field_cells": LATENT_DIMENSION,
        "resident_quotient_message_field_cells": 2 * LATENT_DIMENSION,
        "public_program_integer_cells": 4 * edge_count + 4 + program.depth,
        "module_boundary_evaluations": carrier.module_boundary_evaluations,
        "basis_mismatch_edge_contractions": carrier.basis_mismatch_edge_contractions,
        "cyclic_convolution_field_multiplications": carrier.cyclic_convolution_field_multiplications,
        "cyclic_convolution_field_additions": carrier.cyclic_convolution_field_additions,
        "rung_chirp_field_multiplications": carrier.rung_chirp_field_multiplications,
        "reversible_message_field_additions": carrier.reversible_message_field_additions,
        "message_coordinate_swaps": carrier.message_coordinate_swaps,
        "determinant_matrix_dimension": determinant_dimension,
        "maximum_named_transaction_transient_field_cells": transient_field_cells,
        "final_boundary_field_cells": 1,
        "final_boundary_payload_bits": carrier.alg.payload_bits(boundary),
        "final_boundary_json_bytes": len(
            json.dumps(serialized_boundary, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ),
        "intermediate_projection_calls": 0,
        "final_projection_calls": 1,
        "accepted_path_labelled_289_state_messages_materialized": False,
        "accepted_path_289_by_289_transfer_table_materialized": False,
        "accepted_path_global_assignment_enumeration": False,
        "resident_compact_17_coordinate_convolution_kernel_materialized": True,
        "maximum_resident_payload_bits": carrier.maximum_resident_payload_bits,
        "determinant_stats": carrier.determinant_stats.as_json(),
        "grid_load_additions": carrier.grid_load_additions,
        "grid_unload_additions": carrier.grid_unload_additions,
        "kernel_load_additions": carrier.kernel_load_additions,
        "kernel_unload_additions": carrier.kernel_unload_additions,
        "seed_load_additions": carrier.seed_load_additions,
        "seed_unload_additions": carrier.seed_unload_additions,
        "generation_before": generation,
        "generation_after": carrier.generation,
        "restoration_generation_increment": carrier.generation == generation + 1,
        "same_backing": carrier.backing_identity() == backing,
        "initial_digest": initial,
        "restored_digest_with_generation": carrier.digest(),
        "exact_phase_carrier_restored": carrier.exact_zero(),
        "response_released_after_restoration": True,
        "snapshot_reload_used": False,
        "inverse_history_retained": False,
        "resident_carrier_restoration_class": "EXACT_ALGEBRAIC_RESTORATION",
        "transient_signature_and_determinant_restoration_class": "NO_RESTORATION_CLAIM",
    }


def transform_vector(values: list[Any], alg: backend.Algebra, *, inverse: bool) -> list[Any]:
    scale = alg.divide(alg.one, integer(alg, PRIME)) if inverse else alg.one
    sign = -1 if inverse else 1
    result = []
    for target in range(PRIME):
        value = alg.zero
        for source, amplitude in enumerate(values):
            value = alg.add(
                value,
                alg.mul(amplitude, alg.power(sign * source * target)),
            )
        result.append(alg.mul(value, scale))
    return result


def compile_kernel_values(
    program: LatentLadderProgram,
    alg: backend.Algebra,
) -> list[Any]:
    carrier = LatentLadderCarrier.create(alg)
    load_grid_weights(carrier, program)
    load_convolution_kernel(carrier, program)
    _, kernel_start, _, _ = carrier_offsets(carrier)
    return carrier.cells[kernel_start : kernel_start + LATENT_DIMENSION]


def matched_spectral_boundary(
    program: LatentLadderProgram,
    alg: backend.Algebra,
) -> Any:
    kernel = compile_kernel_values(program, alg)
    kernel_spectrum = transform_vector(kernel, alg, inverse=False)
    u = [alg.zero for _ in range(PRIME)]
    w = [alg.one] + [alg.zero for _ in range(PRIME - 1)]
    for rung_exponent in program.rung_chirp_exponents:
        source_spectrum = transform_vector(w, alg, inverse=False)
        product = [
            alg.mul(left, right)
            for left, right in zip(kernel_spectrum, source_spectrum)
        ]
        action = transform_vector(product, alg, inverse=True)
        for latent in range(PRIME):
            action[latent] = alg.mul(
                action[latent],
                alg.power(rung_exponent * latent * latent),
            )
            u[latent] = alg.add(u[latent], action[latent])
        u, w = w, u
    return w[0]


def exact_case(depth: int, family: str) -> dict[str, Any]:
    program = compile_program(depth, family)
    alg = backend.Algebra("Q_ZETA17")
    result = execute_transaction(LatentLadderCarrier.create(alg), program)
    matched = matched_spectral_boundary(program, backend.Algebra("Q_ZETA17"))
    result["matched_spectral_group_algebra_boundary_agreement"] = (
        result["boundary"] == alg.serialize(matched)
    )
    return result


def modular_case(
    depth: int,
    family: str,
    modulus: int,
    root: int,
) -> dict[str, Any]:
    program = compile_program(depth, family)
    alg = backend.Algebra(f"F{modulus}", modulus=modulus, root=root)
    result = execute_transaction(LatentLadderCarrier.create(alg), program)
    matched = matched_spectral_boundary(
        program,
        backend.Algebra(f"F{modulus}", modulus=modulus, root=root),
    )
    result["matched_spectral_group_algebra_boundary_agreement"] = (
        result["boundary"] == alg.serialize(matched)
    )
    result["field"] = f"F{modulus}"
    return result


def quotient_action(
    values: list[Any],
    kernel: list[Any],
    rung_exponent: int,
    alg: backend.Algebra,
) -> list[Any]:
    result = []
    for target in range(PRIME):
        value = alg.zero
        for source in range(PRIME):
            value = alg.add(
                value,
                alg.mul(kernel[(target - source) % PRIME], values[source]),
            )
        result.append(
            alg.mul(value, alg.power(rung_exponent * target * target))
        )
    return result


def lift_quotient(values: list[Any]) -> list[Any]:
    return [values[(bottom - top) % PRIME] for top in range(PRIME) for bottom in range(PRIME)]


def quotient_from_labelled(values: list[Any]) -> list[Any]:
    if len(values) != PRIME * PRIME:
        fail("labelled latent message size changed")
    quotient = [values[relative] for relative in range(PRIME)]
    for top in range(PRIME):
        for bottom in range(PRIME):
            if values[top * PRIME + bottom] != quotient[(bottom - top) % PRIME]:
                fail("labelled message is outside the global-shift-invariant sector")
    return quotient


def labelled_action(
    values: list[Any],
    left: list[Any],
    right: list[Any],
    rung_exponent: int,
    alg: backend.Algebra,
) -> list[Any]:
    result = []
    for next_top in range(PRIME):
        for next_bottom in range(PRIME):
            value = alg.zero
            for top in range(PRIME):
                for bottom in range(PRIME):
                    term = alg.mul(
                        left[(next_top - top) % PRIME],
                        right[(next_bottom - bottom) % PRIME],
                    )
                    value = alg.add(
                        value,
                        alg.mul(term, values[top * PRIME + bottom]),
                    )
            relative = (next_bottom - next_top) % PRIME
            result.append(
                alg.mul(value, alg.power(rung_exponent * relative * relative))
            )
    return result


def controls() -> dict[str, Any]:
    alg = backend.Algebra("F103", modulus=103, root=72)
    program = compile_program(4, "PRIMARY")

    missing = LatentLadderCarrier.create(alg)
    forward(missing, program)
    missing_inverse_detected = not missing.exact_zero()

    wrong = LatentLadderCarrier.create(
        backend.Algebra("F103", modulus=103, root=72)
    )
    forward(wrong, program)
    wrong_inverse_ownership_detected = False
    try:
        inverse(wrong, compile_program(4, "REUSE"))
    except RuntimeError:
        wrong_inverse_ownership_detected = True

    premature = LatentLadderCarrier.create(
        backend.Algebra("F103", modulus=103, root=72)
    )
    premature_projection_rejected = False
    try:
        project_boundary(premature, program)
    except RuntimeError:
        premature_projection_rejected = True

    null_carrier_rejected = False
    try:
        forward(None, program)  # type: ignore[arg-type]
    except (RuntimeError, AttributeError):
        null_carrier_rejected = True

    reordered = LatentLadderCarrier.create(
        backend.Algebra("F103", modulus=103, root=72)
    )
    forward(reordered, program)
    for rung_exponent in program.rung_chirp_exponents:
        apply_reversible_layer(reordered, rung_exponent, inverse=True)
    reordered_inverse_detected = not reordered.exact_zero()

    base_boundary = execute_transaction(
        LatentLadderCarrier.create(
            backend.Algebra("F103", modulus=103, root=72)
        ),
        program,
    )["boundary"]
    mutated = LatentLadderProgram(
        depth=program.depth,
        family=program.family,
        basis_program=program.basis_program,
        rung_chirp_exponents=(
            1 + (program.rung_chirp_exponents[0] % 16),
            *program.rung_chirp_exponents[1:],
        ),
    )
    mutated_boundary = execute_transaction(
        LatentLadderCarrier.create(
            backend.Algebra("F103", modulus=103, root=72)
        ),
        mutated,
    )["boundary"]

    reference = LatentLadderCarrier.create(
        backend.Algebra("F103", modulus=103, root=72)
    )
    load_grid_weights(reference, program)
    left, right = module_signatures(reference, program)
    kernel = [
        sum(
            (alg.mul(left[latent], right[(latent + shift) % PRIME]) for latent in range(PRIME)),
            alg.zero,
        )
        % alg.modulus
        for shift in range(PRIME)
    ]
    quotient_seed = [alg.one] + [alg.zero for _ in range(PRIME - 1)]
    labelled_seed = lift_quotient(quotient_seed)
    labelled_output = labelled_action(
        labelled_seed,
        left,
        right,
        program.rung_chirp_exponents[0],
        alg,
    )
    quotient_output = quotient_action(
        quotient_seed,
        kernel,
        program.rung_chirp_exponents[0],
        alg,
    )
    labelled_quotient_agreement = labelled_output == lift_quotient(quotient_output)

    broken = list(labelled_output)
    for top in range(PRIME):
        for bottom in range(PRIME):
            broken[top * PRIME + bottom] = alg.mul(
                broken[top * PRIME + bottom], alg.power(top)
            )
    noninvariant_input_rejected = False
    try:
        quotient_from_labelled(broken)
    except RuntimeError:
        noninvariant_input_rejected = True

    return {
        "missing_inverse_detected": missing_inverse_detected,
        "wrong_inverse_ownership_detected": wrong_inverse_ownership_detected,
        "premature_projection_rejected": premature_projection_rejected,
        "null_carrier_rejected": null_carrier_rejected,
        "reordered_inverse_detected": reordered_inverse_detected,
        "rung_phase_mutation_changes_boundary": base_boundary != mutated_boundary,
        "labelled_289_state_oracle_matches_17_coordinate_quotient": labelled_quotient_agreement,
        "global_shift_invariant_sector_required": noninvariant_input_rejected,
        "public_topology_compilation_inspects_final_boundary": False,
        "accepted_path_labelled_289_state_messages_materialized": False,
        "accepted_path_289_by_289_transfer_table_materialized": False,
        "intermediate_quotient_messages_serialized": False,
        "snapshot_command_absent": True,
        "catvm_boundary_claimed": False,
    }


def run() -> dict[str, Any]:
    exact = [
        exact_case(depth, family)
        for family in FAMILIES
        for depth in EXACT_DEPTHS
    ]
    structural = [
        modular_case(depth, family, modulus, root)
        for modulus, root in FINITE_FIELDS
        for family in FAMILIES
        for depth in STRUCTURAL_DEPTHS
    ]

    reuse_carrier = LatentLadderCarrier.create(backend.Algebra("Q_ZETA17"))
    first = execute_transaction(reuse_carrier, compile_program(4, "PRIMARY"))
    reuse_backing = reuse_carrier.backing_identity()
    reused = execute_transaction(reuse_carrier, compile_program(8, "REUSE"))
    fresh = execute_transaction(
        LatentLadderCarrier.create(backend.Algebra("Q_ZETA17")),
        compile_program(8, "REUSE"),
    )
    reused_signature = resource_signature(reused)
    fresh_signature = resource_signature(fresh)
    if reused["boundary"] != fresh["boundary"]:
        fail("restored latent ladder carrier reuse disagrees with fresh execution")
    if reused_signature != fresh_signature:
        fail("restored latent ladder carrier reuse changed its resource signature")

    return {
        "schema": "CAT_CAS_F17_GLOBAL_SHIFT_LATENT_LADDER_CONVOLUTION_CLOSURE_V1",
        "claim": "BOUNDED_EXACT_GLOBAL_SHIFT_INVARIANT_TWO_ROW_F17_SHARED_LATENT_LADDER_REDUCES_EACH_289_STATE_LABELLED_MESSAGE_TO_17_COORDINATE_NATIVE_CYCLIC_CONVOLUTION_WITH_34_COORDINATE_REVERSIBLE_MESSAGE_CARRIER_FINAL_ONLY_PROJECTION_EXACT_RESTORATION_AND_REUSE_BUT_IDENTICAL_GROUP_ALGEBRA_CLASSICAL_RECURRENCE",
        "classification": "SOURCE_AUDITED_PACKAGE_LOCAL",
        "verification_level": "PACKAGE_SELF_REVIEW",
        "source_scope": {
            "grid_module_topology": "M126_EVEN_OPEN_SQUARE_GRID_N4_ONLY",
            "latent_topology": "TWO_ROW_OPEN_LADDER_GLOBAL_SHIFT_INVARIANT_SECTOR_ONLY",
            "exact_q_zeta17_depths": EXACT_DEPTHS,
            "dual_field_structural_depths": STRUCTURAL_DEPTHS,
            "families": FAMILIES,
            "labelled_state_coordinates_per_message": PRIME * PRIME,
            "quotient_coordinates_per_message": PRIME,
        },
        "exact_transactions": exact,
        "dual_field_structural_transactions": structural,
        "reuse": {
            "first_depth": 4,
            "reuse_depth": 8,
            "first_boundary": first["boundary"],
            "reused_boundary": reused["boundary"],
            "fresh_boundary": fresh["boundary"],
            "fresh_restored_boundary_agreement": reused["boundary"] == fresh["boundary"],
            "fresh_restored_resource_signature_agreement": reused_signature == fresh_signature,
            "same_actual_backing_across_unrelated_programs": (
                first["same_backing"]
                and reused["same_backing"]
                and reuse_carrier.backing_identity() == reuse_backing
            ),
            "generation_after_two_transactions": reuse_carrier.generation,
            "baseline_reload_used": False,
        },
        "controls": controls(),
        "resource_law": {
            "resident_phase_field_cells": "4_GRID_N_TIMES_GRID_N_MINUS_1_PLUS_51_EQUALS_99_AT_GRID_N4",
            "resident_grid_weight_field_cells": 48,
            "resident_compact_convolution_kernel_field_cells": 17,
            "resident_reversible_quotient_message_field_cells": 34,
            "labelled_reversible_message_baseline_field_cells": 578,
            "labelled_to_quotient_message_reduction_factor": 17,
            "maximum_named_transaction_transient_field_cells": 166,
            "accepted_field_operations": "O_DEPTH_TIMES_17_SQUARED_FIELD_OPERATIONS_PLUS_68_GRID_DETERMINANTS_PER_TRANSACTION; FULL_EXACT_BIT_COMPLEXITY_NOT_ESTABLISHED",
            "accepted_labelled_289_state_messages_materialized": 0,
            "accepted_289_by_289_transfer_tables_materialized": 0,
            "accepted_global_assignments_enumerated": 0,
            "compact_module_signature_field_cells_temporarily_materialized": 34,
            "inverse_history_retained": 0,
            "controller_backend_traffic_bytes": 0,
            "python_container_native_bigint_and_whole_process_memory_excluded": True,
        },
        "matched_baselines": {
            "strongest_implemented": "IDENTICAL_34_COORDINATE_REVERSIBLE_QUOTIENT_RECURRENCE_WITH_EXACT_F17_GROUP_ALGEBRA_FOURIER_DIAGONALIZATION_OF_THE_17_COORDINATE_CYCLIC_CONVOLUTION",
            "labelled_578_MESSAGE_RECURRENCE": "BOUNDED_ORACLE_ONLY_NOT_THE_MATCHED_BASELINE",
            "all_exact_and_structural_spectral_boundary_agreements": all(
                item["matched_spectral_group_algebra_boundary_agreement"]
                for item in exact + structural
            ),
            "phase_advantage_over_matched_classical": False,
        },
        "restoration": {
            "resident_grid_kernel_and_message_phase_carrier": "EXACT_ALGEBRAIC_RESTORATION",
            "transient_module_signatures_and_determinant_buffers": "NO_RESTORATION_CLAIM",
            "snapshot_reload_used": False,
            "inverse_history_retained": False,
        },
        "claim_ceiling": {
            "global_shift_quotient_applies_only_to_declared_invariant_sector": True,
            "growing_two_row_latent_ladder_depth_with_fixed_34_message_cells": True,
            "compact_17_coordinate_group_convolution_kernel": True,
            "arbitrary_latent_topology_or_noninvariant_sector": False,
            "general_planar_holant_or_relational_closure": False,
            "catvm_custody_established": False,
            "distinct_phase_resource_established": False,
            "computational_advantage_established": False,
            "small_wall_crossing_established": False,
            "physical_waveform_execution_established": False,
            "physical_bits_replaced_with_pi": False,
            "unbounded_catalytic_computation_established": False,
        },
        "next_obstruction": "THE_GLOBAL_SHIFT_QUOTIENT_CLOSES_A_GROWING_TWO_ROW_LATENT_LADDER_IN_34_REVERSIBLE_MESSAGE_CELLS_BUT_EXACT_GROUP_ALGEBRA_CLASSICAL_SOFTWARE_EXECUTES_THE_IDENTICAL_RECURRENCE_AND_THE_QUOTIENT_DOES_NOT_COVER_NONINVARIANT_OR_GROWING_TREEWIDTH_LATENT_GEOMETRY",
        "next_experiment": "NONABELIAN_OR_GROWING_TREEWIDTH_SHARED_LATENT_PHASE_GEOMETRY_WITH_EXACT_RELATION_PRESERVING_RANK_REDUCTION_OR_A_MATCHED_GROUP_ALGEBRA_NO_GO",
    }


def main() -> None:
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
