#!/usr/bin/env python3
"""Exact coherent shared-latent basis mismatch on planar F17 grids.

A 17-coordinate unresolved phase register controls every edge of two distinct
paired-basis mismatch modules.  The modules are separated by exact Fourier and
quadratic-chirp updates, so the resident operations do not commute.  Each
fixed latent coordinate still exposes a weighted planar perfect-matching
closure; the accepted path uses its Kasteleyn determinant and never expands
edge assignments or native signature tables.

This is a bounded direct-process diagnostic.  Its strongest compact classical
baseline is the identical 34-coordinate exact recurrence plus the same compact
matching closures.  No phase resource or advantage is assumed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

import f17_nonlinear_canonical_mps_separator_chart as backend
import f17_paired_phase_basis_holographic_matchgate_closure as matchgate


PRIME = 17
EXACT_SIZES = (2, 4, 6)
STRUCTURAL_SIZES = (2, 4, 6, 8, 10)
FAMILIES = ("PRIMARY", "REUSE")
FINITE_FIELDS = ((103, 72), (137, 16))
LATENT_DIMENSION = 17


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
class CoherentBasisProgram:
    n: int
    family: str
    module_weight_exponents: tuple[tuple[int, ...], tuple[int, ...]]
    module_control_exponents: tuple[tuple[int, ...], tuple[int, ...]]
    chirp_exponent: int

    def fingerprint(self) -> str:
        return sha256_json(
            {
                "n": self.n,
                "family": self.family,
                "module_weight_exponents": self.module_weight_exponents,
                "module_control_exponents": self.module_control_exponents,
                "chirp_exponent": self.chirp_exponent,
            }
        )


def compile_program(n: int, family: str) -> CoherentBasisProgram:
    if n not in STRUCTURAL_SIZES or n % 2:
        fail("coherent basis program requires a declared positive even grid")
    if family not in FAMILIES:
        fail("unknown coherent basis family")
    variant = 0 if family == "PRIMARY" else 1
    edge_count = len(matchgate.grid_edges(n))
    weight_rows = []
    control_rows = []
    for module in range(2):
        weight_rows.append(
            tuple(
                1 + ((5 * index + 7 * n + 3 * module + 4 * variant) % 16)
                for index in range(edge_count)
            )
        )
        control_rows.append(
            tuple(
                1 + ((7 * index + 5 * module + 6 * variant + n) % 16)
                for index in range(edge_count)
            )
        )
    program = CoherentBasisProgram(
        n=n,
        family=family,
        module_weight_exponents=(weight_rows[0], weight_rows[1]),
        module_control_exponents=(control_rows[0], control_rows[1]),
        chirp_exponent=3 + 2 * variant,
    )
    validate_program(program)
    return program


def validate_program(program: CoherentBasisProgram) -> None:
    if program.n not in STRUCTURAL_SIZES or program.n % 2:
        fail("coherent basis program size is outside the declared scope")
    if program.family not in FAMILIES:
        fail("coherent basis program family changed")
    edge_count = len(matchgate.grid_edges(program.n))
    for rows in (program.module_weight_exponents, program.module_control_exponents):
        if len(rows) != 2 or any(len(row) != edge_count for row in rows):
            fail("coherent basis module arity changed")
        if not all(1 <= value < PRIME for row in rows for value in row):
            fail("coherent basis phase exponent is outside F17")
    if not 1 <= program.chirp_exponent < PRIME:
        fail("chirp exponent is outside F17")


@dataclass
class CoherentBasisCarrier:
    n: int
    topology_fingerprint: str
    alg: backend.Algebra
    cells: list[Any]
    generation: int = 0
    lease: str | None = None
    stage: str = "RESTORED"
    factor_load_additions: int = 0
    factor_unload_additions: int = 0
    fourier_transforms: int = 0
    fourier_field_multiplications: int = 0
    fourier_field_additions: int = 0
    chirp_field_multiplications: int = 0
    coherent_shear_field_multiplications: int = 0
    coherent_shear_field_additions: int = 0
    basis_mismatch_edge_contractions: int = 0
    module_boundary_evaluations: int = 0
    projection_calls: int = 0
    maximum_resident_payload_bits: int = 0
    determinant_stats: matchgate.DeterminantStats = field(
        default_factory=matchgate.DeterminantStats
    )

    @classmethod
    def create(cls, n: int, alg: backend.Algebra) -> "CoherentBasisCarrier":
        edges = matchgate.grid_edges(n)
        cells = [alg.zero for _ in range(2 * len(edges) + 2 * LATENT_DIMENSION)]
        carrier = cls(
            n=n,
            topology_fingerprint=sha256_json({"n": n, "edges": edges}),
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


def load_program(
    carrier: CoherentBasisCarrier,
    program: CoherentBasisProgram,
    *,
    inverse: bool = False,
) -> None:
    """Stream public factors and the latent seed directly into resident cells."""
    index = 0
    for row in program.module_weight_exponents:
        for exponent in row:
            value = carrier.alg.power(exponent)
            delta = negative(carrier.alg, value) if inverse else value
            carrier.cells[index] = carrier.alg.add(carrier.cells[index], delta)
            index += 1
    for latent in range(2 * LATENT_DIMENSION):
        value = carrier.alg.one if latent == 0 else carrier.alg.zero
        delta = negative(carrier.alg, value) if inverse else value
        carrier.cells[index] = carrier.alg.add(carrier.cells[index], delta)
        index += 1
    if index != len(carrier.cells):
        fail("coherent basis payload does not fit the carrier")
    if inverse:
        carrier.factor_unload_additions += index
    else:
        carrier.factor_load_additions += index
    carrier.observe_resident()


def transform_vector(
    values: list[Any],
    alg: backend.Algebra,
    *,
    inverse: bool = False,
) -> list[Any]:
    scale = alg.divide(alg.one, integer(alg, LATENT_DIMENSION)) if inverse else alg.one
    sign = -1 if inverse else 1
    result = []
    for target in range(LATENT_DIMENSION):
        value = alg.zero
        for source, amplitude in enumerate(values):
            term = alg.mul(amplitude, alg.power(sign * source * target))
            value = alg.add(value, term)
        result.append(alg.mul(value, scale))
    return result


def transform_resident_segment(
    carrier: CoherentBasisCarrier,
    start: int,
    *,
    inverse: bool,
) -> None:
    values = carrier.cells[start : start + LATENT_DIMENSION]
    transformed = transform_vector(values, carrier.alg, inverse=inverse)
    carrier.cells[start : start + LATENT_DIMENSION] = transformed


def apply_fourier(carrier: CoherentBasisCarrier, *, inverse: bool = False) -> None:
    latent_start = 2 * len(matchgate.grid_edges(carrier.n))
    transform_resident_segment(carrier, latent_start, inverse=inverse)
    transform_resident_segment(
        carrier,
        latent_start + LATENT_DIMENSION,
        inverse=inverse,
    )
    carrier.fourier_transforms += 2
    carrier.fourier_field_multiplications += 2 * (
        LATENT_DIMENSION * LATENT_DIMENSION + LATENT_DIMENSION
    )
    carrier.fourier_field_additions += 2 * LATENT_DIMENSION * LATENT_DIMENSION
    carrier.observe_resident()


def apply_chirp(
    carrier: CoherentBasisCarrier,
    program: CoherentBasisProgram,
    *,
    inverse: bool = False,
) -> None:
    sign = -1 if inverse else 1
    latent_start = 2 * len(matchgate.grid_edges(carrier.n))
    w_start = latent_start + LATENT_DIMENSION
    for latent in range(LATENT_DIMENSION):
        phase = carrier.alg.power(sign * program.chirp_exponent * latent * latent)
        carrier.cells[latent_start + latent] = carrier.alg.mul(
            carrier.cells[latent_start + latent], phase
        )
        carrier.cells[w_start + latent] = carrier.alg.mul(
            carrier.cells[w_start + latent], phase
        )
    carrier.chirp_field_multiplications += 2 * LATENT_DIMENSION
    carrier.observe_resident()


def kasteleyn_matrix(
    carrier: CoherentBasisCarrier,
    program: CoherentBasisProgram,
    module: int,
    latent: int,
) -> list[list[Any]]:
    black, white = matchgate.black_white_vertices(carrier.n)
    black_index = {vertex: index for index, vertex in enumerate(black)}
    white_index = {vertex: index for index, vertex in enumerate(white)}
    matrix = [[carrier.alg.zero for _ in white] for _ in black]
    edges = matchgate.grid_edges(carrier.n)
    weight_start = module * len(edges)
    controls = program.module_control_exponents[module]
    for edge_index, edge in enumerate(edges):
        # T(zeta^(control*latent)) S(1)^T = diag(1,zeta^(control*latent)).
        weight = carrier.alg.mul(
            carrier.cells[weight_start + edge_index],
            carrier.alg.power(controls[edge_index] * latent),
        )
        carrier.basis_mismatch_edge_contractions += 1
        first, second = edge
        left, right = (first, second) if first in black_index else (second, first)
        value = (
            weight
            if matchgate.kasteleyn_edge_sign(first, second) == 1
            else negative(carrier.alg, weight)
        )
        row = black_index[left]
        column = white_index[right]
        matrix[row][column] = carrier.alg.add(matrix[row][column], value)
    return matrix


def module_boundary(
    carrier: CoherentBasisCarrier,
    program: CoherentBasisProgram,
    module: int,
    latent: int,
) -> Any:
    if module not in (0, 1):
        fail("coherent basis module index changed")
    if not 0 <= latent < LATENT_DIMENSION:
        fail("coherent basis latent coordinate changed")
    calibration = matchgate.reference_calibration_sign(carrier.n)
    value = matchgate.determinant(
        kasteleyn_matrix(carrier, program, module, latent),
        carrier.alg,
        carrier.determinant_stats,
    )
    carrier.module_boundary_evaluations += 1
    return value if calibration == 1 else negative(carrier.alg, value)


def apply_controlled_grid_shear(
    carrier: CoherentBasisCarrier,
    program: CoherentBasisProgram,
    module: int,
    *,
    inverse: bool = False,
) -> None:
    latent_start = 2 * len(matchgate.grid_edges(carrier.n))
    w_start = latent_start + LATENT_DIMENSION
    for latent in range(LATENT_DIMENSION):
        boundary = module_boundary(carrier, program, module, latent)
        term = carrier.alg.mul(boundary, carrier.cells[latent_start + latent])
        carrier.cells[w_start + latent] = (
            carrier.alg.sub(carrier.cells[w_start + latent], term)
            if inverse
            else carrier.alg.add(carrier.cells[w_start + latent], term)
        )
    carrier.coherent_shear_field_multiplications += LATENT_DIMENSION
    carrier.coherent_shear_field_additions += LATENT_DIMENSION
    carrier.observe_resident()


def forward(
    carrier: CoherentBasisCarrier,
    program: CoherentBasisProgram,
) -> None:
    if not isinstance(carrier, CoherentBasisCarrier) or not carrier.exact_zero():
        fail("null, leased, or unrestored coherent basis carrier")
    validate_program(program)
    if carrier.n != program.n:
        fail("coherent basis program does not own the carrier topology")
    carrier.lease = program.fingerprint()
    carrier.stage = "FORWARD_ACTIVE"
    load_program(carrier, program)
    apply_fourier(carrier)
    apply_controlled_grid_shear(carrier, program, 0)
    apply_chirp(carrier, program)
    apply_fourier(carrier)
    apply_controlled_grid_shear(carrier, program, 1)
    carrier.stage = "FORWARD_COMPLETE"
    carrier.observe_resident()


def project_boundary(
    carrier: CoherentBasisCarrier,
    program: CoherentBasisProgram,
) -> Any:
    if carrier.stage != "FORWARD_COMPLETE" or carrier.lease != program.fingerprint():
        fail("only the completed owned coherent boundary may be projected")
    carrier.projection_calls += 1
    edge_count = len(matchgate.grid_edges(carrier.n))
    return carrier.cells[2 * edge_count + LATENT_DIMENSION]


def inverse(
    carrier: CoherentBasisCarrier,
    program: CoherentBasisProgram,
) -> None:
    if carrier.stage != "FORWARD_COMPLETE" or carrier.lease != program.fingerprint():
        fail("inverse program does not own the coherent basis lease")
    carrier.stage = "INVERSE_ACTIVE"
    apply_controlled_grid_shear(carrier, program, 1, inverse=True)
    apply_fourier(carrier, inverse=True)
    apply_chirp(carrier, program, inverse=True)
    apply_controlled_grid_shear(carrier, program, 0, inverse=True)
    apply_fourier(carrier, inverse=True)
    load_program(carrier, program, inverse=True)
    carrier.lease = None
    carrier.stage = "RESTORED"
    carrier.generation += 1
    carrier.observe_resident()
    if not carrier.exact_zero():
        fail("actual inverse failed exact coherent basis carrier restoration")


def reset_transaction_observation(carrier: CoherentBasisCarrier) -> None:
    carrier.factor_load_additions = 0
    carrier.factor_unload_additions = 0
    carrier.fourier_transforms = 0
    carrier.fourier_field_multiplications = 0
    carrier.fourier_field_additions = 0
    carrier.chirp_field_multiplications = 0
    carrier.coherent_shear_field_multiplications = 0
    carrier.coherent_shear_field_additions = 0
    carrier.basis_mismatch_edge_contractions = 0
    carrier.module_boundary_evaluations = 0
    carrier.projection_calls = 0
    carrier.maximum_resident_payload_bits = 0
    carrier.determinant_stats = matchgate.DeterminantStats()


RESOURCE_SIGNATURE_KEYS = (
    "resident_phase_field_cells",
    "resident_grid_weight_field_cells",
    "resident_latent_field_cells",
    "module_boundary_evaluations",
    "basis_mismatch_edge_contractions",
    "fourier_transforms",
    "chirp_field_multiplications",
    "coherent_shear_field_multiplications",
    "coherent_shear_field_additions",
    "determinant_matrix_dimension",
    "maximum_named_transaction_transient_field_cells",
    "factor_load_additions",
    "factor_unload_additions",
    "resident_carrier_restoration_class",
)


def resource_signature(transaction: dict[str, Any]) -> dict[str, Any]:
    signature = {key: transaction[key] for key in RESOURCE_SIGNATURE_KEYS}
    signature["determinant_stats"] = transaction["determinant_stats"]
    return signature


def execute_transaction(
    carrier: CoherentBasisCarrier,
    program: CoherentBasisProgram,
) -> dict[str, Any]:
    reset_transaction_observation(carrier)
    initial = carrier.digest()
    backing = carrier.backing_identity()
    generation = carrier.generation
    forward(carrier, program)
    boundary = project_boundary(carrier, program)
    inverse(carrier, program)
    edge_count = len(matchgate.grid_edges(program.n))
    determinant_dimension = program.n * program.n // 2
    maximum_named_transaction_transient_field_cells = max(
        2 * determinant_dimension * determinant_dimension + 5,
        2 * LATENT_DIMENSION,
    )
    serialized_boundary = carrier.alg.serialize(boundary)
    return {
        "n": program.n,
        "family": program.family,
        "program_fingerprint": program.fingerprint(),
        "boundary": serialized_boundary,
        "edge_count": edge_count,
        "resident_phase_field_cells": len(carrier.cells),
        "resident_grid_weight_field_cells": 2 * edge_count,
        "resident_latent_field_cells": 2 * LATENT_DIMENSION,
        "unresolved_latent_dimension": LATENT_DIMENSION,
        "coherent_grid_modules": 2,
        "public_program_integer_cells": 4 * edge_count + 4,
        "compiled_topology_edge_records": edge_count,
        "module_boundary_evaluations": carrier.module_boundary_evaluations,
        "basis_mismatch_edge_contractions": carrier.basis_mismatch_edge_contractions,
        "fourier_transforms": carrier.fourier_transforms,
        "chirp_field_multiplications": carrier.chirp_field_multiplications,
        "coherent_shear_field_multiplications": carrier.coherent_shear_field_multiplications,
        "coherent_shear_field_additions": carrier.coherent_shear_field_additions,
        "determinant_matrix_dimension": determinant_dimension,
        "determinant_matrix_field_cells": determinant_dimension * determinant_dimension,
        "maximum_named_transaction_transient_field_cells": maximum_named_transaction_transient_field_cells,
        "final_boundary_field_cells": 1,
        "final_boundary_payload_bits": carrier.alg.payload_bits(boundary),
        "final_boundary_json_bytes": len(
            json.dumps(serialized_boundary, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ),
        "intermediate_projection_calls": 0,
        "final_projection_calls": 1,
        "accepted_path_edge_assignment_enumeration": False,
        "accepted_path_edge_local_native_signature_table_materialized": False,
        "accepted_path_latent_diagonal_boundary_vector_materialized": False,
        "accepted_path_latent_coordinate_count": LATENT_DIMENSION,
        "maximum_resident_payload_bits": carrier.maximum_resident_payload_bits,
        "determinant_stats": carrier.determinant_stats.as_json(),
        "factor_load_additions": carrier.factor_load_additions,
        "factor_unload_additions": carrier.factor_unload_additions,
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
        "transient_determinant_buffer_restoration_class": "NO_RESTORATION_CLAIM",
    }


def exact_case(n: int, family: str) -> dict[str, Any]:
    return execute_transaction(
        CoherentBasisCarrier.create(n, backend.Algebra("Q_ZETA17")),
        compile_program(n, family),
    )


def modular_case(
    n: int,
    family: str,
    modulus: int,
    root: int,
) -> dict[str, Any]:
    result = execute_transaction(
        CoherentBasisCarrier.create(
            n,
            backend.Algebra(f"F{modulus}", modulus=modulus, root=root),
        ),
        compile_program(n, family),
    )
    result["field"] = f"F{modulus}"
    return result


def mutated_program(program: CoherentBasisProgram) -> CoherentBasisProgram:
    rows = [list(row) for row in program.module_control_exponents]
    rows[0][0] = 1 + (rows[0][0] % 16)
    return CoherentBasisProgram(
        n=program.n,
        family=program.family,
        module_weight_exponents=program.module_weight_exponents,
        module_control_exponents=(tuple(rows[0]), tuple(rows[1])),
        chirp_exponent=program.chirp_exponent,
    )


def controls() -> dict[str, Any]:
    alg = backend.Algebra("F103", modulus=103, root=72)
    program = compile_program(4, "PRIMARY")

    seed = (alg.one, *[alg.zero for _ in range(LATENT_DIMENSION - 1)])
    fourier_then_chirp = tuple(
        alg.mul(value, alg.power(program.chirp_exponent * latent * latent))
        for latent, value in enumerate(transform_vector(seed, alg))
    )
    chirp_then_fourier = transform_vector(
        tuple(
            alg.mul(value, alg.power(program.chirp_exponent * latent * latent))
            for latent, value in enumerate(seed)
        ),
        alg,
    )

    missing = CoherentBasisCarrier.create(4, alg)
    forward(missing, program)
    missing_inverse_detected = not missing.exact_zero()

    wrong = CoherentBasisCarrier.create(
        4, backend.Algebra("F103", modulus=103, root=72)
    )
    forward(wrong, program)
    wrong_inverse_ownership_detected = False
    try:
        inverse(wrong, compile_program(4, "REUSE"))
    except RuntimeError:
        wrong_inverse_ownership_detected = True

    premature = CoherentBasisCarrier.create(
        4, backend.Algebra("F103", modulus=103, root=72)
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

    base_boundary = execute_transaction(
        CoherentBasisCarrier.create(
            4, backend.Algebra("F103", modulus=103, root=72)
        ),
        program,
    )["boundary"]
    mutated_boundary = execute_transaction(
        CoherentBasisCarrier.create(
            4, backend.Algebra("F103", modulus=103, root=72)
        ),
        mutated_program(program),
    )["boundary"]

    chirp_mutated = CoherentBasisProgram(
        n=program.n,
        family=program.family,
        module_weight_exponents=program.module_weight_exponents,
        module_control_exponents=program.module_control_exponents,
        chirp_exponent=1 + (program.chirp_exponent % 16),
    )
    chirp_boundary = execute_transaction(
        CoherentBasisCarrier.create(
            4, backend.Algebra("F103", modulus=103, root=72)
        ),
        chirp_mutated,
    )["boundary"]

    reordered = CoherentBasisCarrier.create(
        4, backend.Algebra("F103", modulus=103, root=72)
    )
    forward(reordered, program)
    apply_controlled_grid_shear(reordered, program, 0, inverse=True)
    apply_controlled_grid_shear(reordered, program, 1, inverse=True)
    apply_fourier(reordered, inverse=True)
    apply_chirp(reordered, program, inverse=True)
    apply_fourier(reordered, inverse=True)
    load_program(reordered, program, inverse=True)
    reordered.lease = None
    reordered.stage = "RESTORED"
    reordered_inverse_detected = not reordered.exact_zero()

    return {
        "missing_inverse_detected": missing_inverse_detected,
        "wrong_inverse_ownership_detected": wrong_inverse_ownership_detected,
        "premature_projection_rejected": premature_projection_rejected,
        "null_carrier_rejected": null_carrier_rejected,
        "reordered_inverse_detected": reordered_inverse_detected,
        "semantic_basis_control_mutation_changes_boundary": base_boundary != mutated_boundary,
        "coherent_chirp_mutation_changes_boundary": base_boundary != chirp_boundary,
        "fourier_chirp_order_changes_latent_state": fourier_then_chirp != chirp_then_fourier,
        "accepted_path_edge_assignment_enumeration": False,
        "accepted_path_edge_local_native_signature_table_materialized": False,
        "accepted_path_latent_diagonal_boundary_vector_materialized": False,
        "intermediate_latent_vector_serialized": False,
        "snapshot_command_absent": True,
        "catvm_boundary_claimed": False,
    }


def run() -> dict[str, Any]:
    exact = [
        exact_case(n, family)
        for family in FAMILIES
        for n in EXACT_SIZES
    ]
    structural = [
        modular_case(n, family, modulus, root)
        for modulus, root in FINITE_FIELDS
        for family in FAMILIES
        for n in STRUCTURAL_SIZES
    ]

    reuse_n = 4
    reuse_carrier = CoherentBasisCarrier.create(
        reuse_n, backend.Algebra("Q_ZETA17")
    )
    first = execute_transaction(reuse_carrier, compile_program(reuse_n, "PRIMARY"))
    reuse_backing = reuse_carrier.backing_identity()
    reused = execute_transaction(reuse_carrier, compile_program(reuse_n, "REUSE"))
    fresh = execute_transaction(
        CoherentBasisCarrier.create(reuse_n, backend.Algebra("Q_ZETA17")),
        compile_program(reuse_n, "REUSE"),
    )
    if reused["boundary"] != fresh["boundary"]:
        fail("restored coherent basis carrier reuse disagrees with fresh execution")
    reused_signature = resource_signature(reused)
    fresh_signature = resource_signature(fresh)
    if reused_signature != fresh_signature:
        fail("restored coherent basis carrier reuse changed its resource signature")

    return {
        "schema": "CAT_CAS_F17_COHERENT_LATENT_BASIS_MISMATCH_GRID_CLOSURE_V1",
        "claim": "BOUNDED_EXACT_COHERENT_SHARED_F17_LATENT_PHASE_PORT_CONTROLS_TWO_GRID_WIDE_BASIS_MISMATCH_MODULES_SEPARATED_BY_NONCOMMUTING_FOURIER_AND_CHIRP_UPDATES_WITH_FINAL_ONLY_PROJECTION_EXACT_RESTORATION_AND_REUSE_BUT_COLLAPSES_TO_IDENTICAL_34_COORDINATE_CLASSICAL_RECURRENCE",
        "classification": "SOURCE_AUDITED_PACKAGE_LOCAL",
        "verification_level": "PACKAGE_SELF_REVIEW",
        "source_scope": {
            "topology": "EVEN_OPEN_SQUARE_GRIDS_ONLY",
            "exact_q_zeta17_sizes": EXACT_SIZES,
            "dual_field_structural_sizes": STRUCTURAL_SIZES,
            "families": FAMILIES,
            "unresolved_shared_latent_dimension": LATENT_DIMENSION,
            "coherent_grid_modules": 2,
        },
        "exact_transactions": exact,
        "dual_field_structural_transactions": structural,
        "reuse": {
            "n": reuse_n,
            "first_boundary": first["boundary"],
            "reused_boundary": reused["boundary"],
            "fresh_boundary": fresh["boundary"],
            "fresh_restored_boundary_agreement": reused["boundary"] == fresh["boundary"],
            "fresh_restored_resource_signature_agreement": reused_signature == fresh_signature,
            "reused_resource_signature": reused_signature,
            "fresh_resource_signature": fresh_signature,
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
            "resident_phase_field_cells": "4N_TIMES_N_MINUS_1_PLUS_34",
            "resident_grid_weight_field_cells": "4N_TIMES_N_MINUS_1",
            "resident_latent_field_cells": 34,
            "latent_coordinate_count": 17,
            "coherent_grid_modules": 2,
            "forward_and_inverse_module_determinants": 68,
            "accepted_projection_matrix_dimension": "N_SQUARED_OVER_2",
            "accepted_named_dense_work_field_cells": "N_TO_THE_4_OVER_2_PLUS_5",
            "accepted_named_transaction_transient_field_cells": "MAX_N_TO_THE_4_OVER_2_PLUS_5_COMMA_34_AFTER_STREAMING_PUBLIC_LOADS_AND_EACH_LATENT_BOUNDARY_DIRECTLY_INTO_RESIDENT_STATE",
            "accepted_field_operation_work": "O_17_N_TO_THE_6_FIELD_OPERATIONS; PARTIAL_OBSERVED_PAYLOAD_DIAGNOSTICS_ONLY; FULL_EXACT_BIT_COMPLEXITY_NOT_ESTABLISHED",
            "native_edge_assignments_materialized": 0,
            "native_edge_local_signature_tables_materialized": 0,
            "latent_diagonal_boundary_vectors_materialized": 0,
            "compiled_load_or_unload_vectors_materialized": 0,
            "fourier_named_input_and_output_field_cells": 34,
            "final_boundary_payload_and_json_bytes_reported_per_transaction": True,
            "inverse_history_retained": 0,
            "public_program_integer_cells": "8N_TIMES_N_MINUS_1_PLUS_4",
            "controller_backend_traffic_bytes": 0,
            "control_carrier_instances": 7,
            "control_full_or_partial_forward_executions": 6,
            "control_and_verification_runs_are_sequential_not_added_to_accepted_peak": True,
            "python_container_native_bigint_and_whole_process_memory_excluded": True,
        },
        "matched_baselines": {
            "strongest_implemented": "IDENTICAL_34_COORDINATE_EXACT_LATENT_RECURRENCE_PLUS_17_SECTOR_KASTELEYN_CLOSURES_PER_MODULE",
            "phase_advantage_over_matched_classical": False,
            "edge_assignment_or_signature_expansion_is_not_the_matched_baseline": True,
        },
        "restoration": {
            "resident_weight_and_latent_phase_carrier": "EXACT_ALGEBRAIC_RESTORATION",
            "transient_kasteleyn_and_elimination_buffers": "NO_RESTORATION_CLAIM",
            "snapshot_reload_used": False,
            "inverse_history_retained": False,
        },
        "claim_ceiling": {
            "shared_latent_phase_port_is_resident_and_unprojected_until_final_boundary": True,
            "multiple_grid_wide_consumers": True,
            "noncommuting_fourier_chirp_and_controlled_shear_composition": True,
            "fixed_17_sector_classical_direct_sum_collapse": True,
            "arbitrary_latent_dimension_or_planar_holant_closure": False,
            "catvm_custody_established": False,
            "distinct_phase_resource_established": False,
            "computational_advantage_established": False,
            "small_wall_crossing_established": False,
            "physical_waveform_execution_established": False,
            "physical_bits_replaced_with_pi": False,
            "unbounded_catalytic_computation_established": False,
        },
        "next_obstruction": "COHERENCE_DEPENDENT_BASIS_CONTROL_AVOIDS_ONE_PUBLIC_SCALAR_CANCELLATION_BUT_A_FIXED_17_STATE_LATENT_PORT_IS_THE_IDENTICAL_CLASSICAL_DIRECT_SUM_SO_THE_NEXT_ROUTE_MUST_GROW_OR_CLOSE_LATENT_GEOMETRY_WITHOUT_GROWING_CLASSICAL_SECTOR_RANK",
        "next_experiment": "GROWING_SHARED_LATENT_PHASE_GEOMETRY_WITH_NATIVE_CONVOLUTION_OR_EXACT_RANK_REDUCTION_AGAINST_THE_STRONGEST_GROUP_ALGEBRA_CLASSICAL_BASELINE",
    }


def main() -> None:
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
