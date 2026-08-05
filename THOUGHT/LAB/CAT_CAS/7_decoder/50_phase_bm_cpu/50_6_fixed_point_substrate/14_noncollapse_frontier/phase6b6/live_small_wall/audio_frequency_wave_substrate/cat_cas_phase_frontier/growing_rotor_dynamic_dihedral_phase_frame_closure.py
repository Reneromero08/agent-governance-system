#!/usr/bin/env python3
"""Dynamic conjugated-symmetry frame diagnostic for exact Rotor-6.

A transported basis can follow the public Fourier network without changing
the 2,277 coefficient coordinates: the four-integer frame descriptor records
the public network endpoint and its lease.  The physical position diagonal is
not coefficientwise diagonal in that transported basis.  Its exact conjugate
is the existing M204 reflection-paired scattering operator, so the moving
frame avoids the occupation expansion and direct permanent kernel but retains
M204's search, fanout, and identical classical recurrence.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import growing_rotor_coherent_momentum_wave_streaming_closure as m204
import growing_rotor_dihedral_quotient_bosonic_fourier_diagnostic as m206
import growing_rotor_dual_position_phase_scattering as m201


GRID = m204.GRID
ROTORS = m204.ROTORS
PRIME = m204.PRIME
FORWARD_GATES = 288
INVERSE_GATES = 289
FRAME_TYPE = 20617


@dataclass
class FrameWork:
    frame_leases: int = 0
    frame_releases: int = 0
    forward_frame_cursor_steps: int = 0
    inverse_frame_cursor_steps: int = 0
    conjugated_position_diagonal_calls: int = 0

    def add(self, other: "FrameWork") -> None:
        for name in self.__dataclass_fields__:
            setattr(self, name, getattr(self, name) + getattr(other, name))

    def as_dict(self) -> dict[str, int]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass
class PhaseFrame:
    slots: list[int]

    def lease(self, owner_generation: int, work: FrameWork) -> None:
        if self.slots != [0, 0, 0, 0]:
            raise RuntimeError("phase frame was not restored before lease")
        if owner_generation <= 0:
            raise ValueError("phase frame owner generation is malformed")
        self.slots[:] = [FORWARD_GATES, FRAME_TYPE, owner_generation, 1]
        work.frame_leases += 1
        work.forward_frame_cursor_steps += FORWARD_GATES

    def require(self, owner_generation: int) -> None:
        if self.slots != [FORWARD_GATES, FRAME_TYPE, owner_generation, 1]:
            raise ValueError("phase frame type, stage, or owner mismatch")

    def release(self, owner_generation: int, work: FrameWork) -> None:
        self.require(owner_generation)
        work.inverse_frame_cursor_steps += INVERSE_GATES
        self.slots[:] = [0, 0, 0, 0]
        work.frame_releases += 1

    def project(self) -> tuple[int, ...]:
        if self.slots[3]:
            raise PermissionError("live transported phase frame projection rejected")
        return tuple(self.slots)


@dataclass
class MovingFrameCarrier:
    source: list[int]
    target: list[int]
    wave_port: m204.MomentumWavePort
    frame: PhaseFrame
    generation: int = 0


def verify_public_frame_inverse() -> dict[str, object]:
    forward, inverse = m201.compile_transforms()
    frame = m201.identity()
    for operation in forward:
        m201.apply_row_operation(frame, operation)
    forward_matrix = m201.fourier_matrix(False)
    if frame != forward_matrix:
        raise RuntimeError("public forward frame endpoint changed")
    for operation in inverse:
        m201.apply_row_operation(frame, operation)
    if frame != m201.identity():
        raise RuntimeError("public inverse frame did not restore identity")
    return {
        "forward_gates": len(forward),
        "inverse_gates": len(inverse),
        "forward_commitment": m201.network_commitment(forward),
        "inverse_commitment": m201.network_commitment(inverse),
        "verification_only_matrix_field_cells": 2 * GRID * GRID,
    }


def naive_position_diagonal(
    state: list[int], targets: tuple[tuple[int, ...], ...], step: int, tag: int
) -> list[int]:
    work = m201.Work()
    return [
        value * m201.scattering_eigenvalue(item, step, tag, work) % PRIME
        for value, item in zip(state, targets, strict=True)
    ]


def apply_conjugated_position_diagonal(
    state: list[int],
    topology: m204.CompactTopology,
    frame: PhaseFrame,
    owner_generation: int,
    step: int,
    tag: int,
    wave_port: m204.MomentumWavePort,
    frame_work: FrameWork,
) -> tuple[list[int], m204.WaveWork, dict[str, object]]:
    frame.require(owner_generation)
    frame_work.conjugated_position_diagonal_calls += 1
    return m204.apply_scattering_wave(
        state,
        topology,
        step,
        tag,
        resident_port=wave_port,
    )


def execute_word(
    source: list[int],
    topology: m204.CompactTopology,
    operations: tuple[tuple[int, int], ...],
    frame: PhaseFrame,
    wave_port: m204.MomentumWavePort,
    owner_base: int,
    *,
    targets: tuple[tuple[int, ...], ...] | None = None,
) -> tuple[list[int], m204.WaveWork, FrameWork, dict[str, object]]:
    current = source.copy()
    wave_total = m204.WaveWork()
    frame_total = FrameWork()
    diagnostics: dict[str, object] = {}
    for operation_index, (step, tag) in enumerate(operations):
        current, diagonal = m204.scalar.apply_diagonal(
            current, topology, step, tag
        )
        owner = owner_base + operation_index + 1
        frame.lease(owner, frame_total)
        transformed, scattering, wave_diagnostics = (
            apply_conjugated_position_diagonal(
                current,
                topology,
                frame,
                owner,
                step,
                tag,
                wave_port,
                frame_total,
            )
        )
        if targets is not None:
            naive = naive_position_diagonal(current, targets, step, tag)
            diagnostics["naive_position_coefficient_mismatch_cells"] = (
                m204.scalar.mismatch(transformed, naive)
            )
        frame.release(owner, frame_total)
        current = transformed
        wave_total.add(diagonal)
        wave_total.add(scattering)
        diagnostics.update(wave_diagnostics)
    if frame.slots != [0, 0, 0, 0]:
        raise RuntimeError("phase frame escaped the word")
    return current, wave_total, frame_total, diagnostics


def transaction(
    carrier: MovingFrameCarrier,
    expected_source: list[int],
    topology: m204.CompactTopology,
    operations: tuple[tuple[int, int], ...],
    targets: tuple[tuple[int, ...], ...],
    signature_order: tuple[int, ...],
) -> tuple[
    dict[str, object], str, str, m204.WaveWork, FrameWork, dict[str, object]
]:
    source_backing = id(carrier.source)
    target_backing = id(carrier.target)
    port_backing = id(carrier.wave_port.values)
    frame_backing = id(carrier.frame.slots)
    owner_base = 1000 * (carrier.generation + 1)
    forward, forward_wave, forward_frame, diagnostics = execute_word(
        carrier.source,
        topology,
        operations,
        carrier.frame,
        carrier.wave_port,
        owner_base,
        targets=targets,
    )
    carrier.target[:] = [
        (left + right) % PRIME
        for left, right in zip(carrier.target, forward, strict=True)
    ]
    projected = m204.scalar.boundary(carrier.target, topology)
    forward_commitment = m204.scalar.state_commitment(forward)
    signature_order_commitment = m204.scalar.source_commitment(
        forward, signature_order
    )
    missing_error = sum(value != 0 for value in carrier.target)
    forward.clear()
    rematerialized, reverse_wave, reverse_frame, reverse_diagnostics = execute_word(
        carrier.source,
        topology,
        operations,
        carrier.frame,
        carrier.wave_port,
        owner_base,
        targets=targets,
    )
    carrier.target[:] = [
        (left - right) % PRIME
        for left, right in zip(carrier.target, rematerialized, strict=True)
    ]
    rematerialized.clear()
    carrier.generation += 1
    wave_total = m204.WaveWork()
    wave_total.add(forward_wave)
    wave_total.add(reverse_wave)
    frame_total = FrameWork()
    frame_total.add(forward_frame)
    frame_total.add(reverse_frame)
    if diagnostics != reverse_diagnostics:
        raise RuntimeError("moving-frame rematerialization diagnostics differ")
    restoration_error = sum(
        left != right
        for left, right in zip(carrier.source, expected_source, strict=True)
    ) + sum(value != 0 for value in carrier.target)
    return (
        {
            "boundary": projected,
            "missing_inverse_error_field_cells": missing_error,
            "restoration_error_field_cells": restoration_error,
            "same_backing": (
                id(carrier.source) == source_backing
                and id(carrier.target) == target_backing
                and id(carrier.wave_port.values) == port_backing
                and id(carrier.frame.slots) == frame_backing
            ),
            "frame_restored": carrier.frame.slots == [0, 0, 0, 0],
            "wave_port_restored": (
                not carrier.wave_port.live
                and carrier.wave_port.values == [0] * len(m204.CHANNELS)
            ),
            "generation": carrier.generation,
        },
        forward_commitment,
        signature_order_commitment,
        wave_total,
        frame_total,
        diagnostics,
    )


def frame_controls() -> dict[str, bool]:
    work = FrameWork()
    frame = PhaseFrame([0, 0, 0, 0])
    frame.lease(9, work)
    wrong_owner = False
    wrong_stage = False
    premature_projection = False
    try:
        frame.require(10)
    except ValueError:
        wrong_owner = True
    frame.slots[0] -= 1
    try:
        frame.require(9)
    except ValueError:
        wrong_stage = True
    frame.slots[0] += 1
    try:
        frame.project()
    except PermissionError:
        premature_projection = True
    missing_inverse_detected = frame.slots != [0, 0, 0, 0]
    frame.release(9, work)
    return {
        "wrong_frame_owner_rejected": wrong_owner,
        "wrong_frame_stage_rejected": wrong_stage,
        "premature_frame_projection_rejected": premature_projection,
        "missing_frame_inverse_detected": missing_inverse_detected,
        "control_frame_restored": frame.slots == [0, 0, 0, 0],
    }


def main() -> None:
    public_frame = verify_public_frame_inverse()
    topology, compiler_work = m204.scalar.compile_topology()
    targets, occupation_count, zero_total_count = m206.target_topology()
    source, signature_order = m204.scalar.source_and_signature_order(topology, 0)
    primary_word = m204.scalar.predecessor.public_law.public_program(1, 0)
    reuse_word = m204.scalar.predecessor.public_law.public_program(1, 4)

    carrier = MovingFrameCarrier(
        source.copy(),
        [0] * len(source),
        m204.MomentumWavePort([0] * len(m204.CHANNELS)),
        PhaseFrame([0, 0, 0, 0]),
    )
    (
        primary,
        primary_commitment,
        primary_signature_commitment,
        primary_wave,
        primary_frame,
        primary_diagnostics,
    ) = transaction(carrier, source, topology, primary_word, targets, signature_order)
    (
        reuse,
        reuse_commitment,
        reuse_signature_commitment,
        reuse_wave,
        reuse_frame,
        reuse_diagnostics,
    ) = transaction(carrier, source, topology, reuse_word, targets, signature_order)
    fresh = MovingFrameCarrier(
        source.copy(),
        [0] * len(source),
        m204.MomentumWavePort([0] * len(m204.CHANNELS)),
        PhaseFrame([0, 0, 0, 0]),
    )
    (
        fresh_reuse,
        fresh_commitment,
        fresh_signature_commitment,
        fresh_wave,
        fresh_frame,
        fresh_diagnostics,
    ) = transaction(fresh, source, topology, reuse_word, targets, signature_order)
    controls = frame_controls()
    if (
        occupation_count != 74613
        or zero_total_count != 4389
        or len(targets) != 2277
        or primary["boundary"] != m204.scalar.predecessor.EXPECTED_PRIMARY_BOUNDARY
        or reuse["boundary"] != m204.scalar.predecessor.EXPECTED_REUSE_BOUNDARY
        or fresh_reuse["boundary"] != reuse["boundary"]
        or reuse_commitment != fresh_commitment
        or reuse_signature_commitment != fresh_signature_commitment
        or primary["restoration_error_field_cells"]
        or reuse["restoration_error_field_cells"]
        or fresh_reuse["restoration_error_field_cells"]
        or not primary["same_backing"]
        or not reuse["same_backing"]
        or not fresh_reuse["same_backing"]
        or not primary["frame_restored"]
        or not reuse["frame_restored"]
        or not fresh_reuse["frame_restored"]
        or not primary["wave_port_restored"]
        or not reuse["wave_port_restored"]
        or not fresh_reuse["wave_port_restored"]
        or carrier.generation != 2
        or primary_diagnostics["naive_position_coefficient_mismatch_cells"] == 0
        or reuse_diagnostics != fresh_diagnostics
        or not all(controls.values())
    ):
        raise RuntimeError("moving-frame transaction or controls failed")

    forward_wave = {
        name: value // 2 for name, value in primary_wave.as_dict().items()
    }
    forward_frame = {
        name: value // 2 for name, value in primary_frame.as_dict().items()
    }
    claim = (
        "EXACT_F103_ROTOR6_DYNAMIC_CONJUGATED_DIHEDRAL_FRAME_CARRIES2277_"
        "COEFFICIENTS_THROUGH577_PUBLIC_FOURIER_BASIS_STEPS_WITH_FOUR_"
        "DESCRIPTOR_INTEGERS_NO_OCCUPATION_DENSE_KERNEL_PERMANENT_OR_"
        "RETAINED_PLAN_AND_EXACT_RESTORATION_REUSE_BUT_THE_POSITION_"
        "DIAGONAL_CONJUGATE_IS_EXACTLY_M204_RETAINS5697720_SEARCHES_"
        "5534928_FANOUT_AND_AN_IDENTICAL_CLASSICAL_FRAME_RECURRENCE"
    )
    result = {
        "claim_candidate": claim,
        "claim_ceiling": (
            "GRID17_EXCHANGE_SYMMETRIC_ROTATION_REFLECTION_INVARIANT_"
            "ROTOR6_F103_ROOT72_DEPTH1_PRIMARY_REUSE_DIRECT_PROCESS_"
            "PUBLIC_ENDPOINT_FRAME_DESCRIPTOR_AND_M204_CONJUGATE_ONLY"
        ),
        "classification": "INDEPENDENTLY_VERIFIED_SOURCE_LOCAL",
        "verification_level": "SEPARATE_REFERENCE_PARITY",
        "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "result": "PASS_MOVING_FRAME_CLOSES_ONLY_BY_REUSING_M204_CONJUGATE",
        "frame_law": {
            "coefficient_cells": len(source),
            "frame_descriptor_integers": 4,
            "frame_descriptor_schema": [
                "PUBLIC_NETWORK_STAGE",
                "FRAME_TYPE",
                "OWNER_GENERATION",
                "LIVE",
            ],
            "forward_frame_steps": FORWARD_GATES,
            "inverse_frame_steps": INVERSE_GATES,
            "total_frame_steps": FORWARD_GATES + INVERSE_GATES,
            "public_frame_verification": public_frame,
            "carrier_coefficients_changed_by_frame_transport": False,
            "position_diagonal_coefficientwise_in_transported_basis": False,
            "position_diagonal_conjugate": "M204_REFLECTION_PAIRED_EIGHT_CHANNEL_SCATTERING_STREAM",
            "retained_frame_matrix_cells": 0,
            "full_occupation_scratch_cells": 0,
            "dense_2277_squared_kernel_cells": 0,
            "permanent_assignment_terms": 0,
            "retained_transition_plan_entries": 0,
        },
        "parity": {
            "primary_boundary": primary["boundary"],
            "reuse_boundary": reuse["boundary"],
            "fresh_reuse_boundary": fresh_reuse["boundary"],
            "primary_state_commitment": primary_commitment,
            "reuse_state_commitment": reuse_commitment,
            "fresh_reuse_state_commitment": fresh_commitment,
            "primary_signature_order_commitment": primary_signature_commitment,
            "reuse_signature_order_commitment": reuse_signature_commitment,
            "fresh_reuse_signature_order_commitment": fresh_signature_commitment,
            "m204_expected_primary_commitment": m204.scalar.predecessor.EXPECTED_PRIMARY_COMMITMENT,
            "fresh_restored_reuse_agreement": reuse_commitment == fresh_commitment,
            "naive_position_coefficient_mismatch_cells": primary_diagnostics[
                "naive_position_coefficient_mismatch_cells"
            ],
        },
        "transaction": {
            "primary_restoration_error_field_cells": primary[
                "restoration_error_field_cells"
            ],
            "reuse_restoration_error_field_cells": reuse[
                "restoration_error_field_cells"
            ],
            "fresh_reuse_restoration_error_field_cells": fresh_reuse[
                "restoration_error_field_cells"
            ],
            "primary_same_backing": primary["same_backing"],
            "reuse_same_backing": reuse["same_backing"],
            "fresh_reuse_same_backing": fresh_reuse["same_backing"],
            "primary_frame_restored": primary["frame_restored"],
            "reuse_frame_restored": reuse["frame_restored"],
            "fresh_reuse_frame_restored": fresh_reuse["frame_restored"],
            "restoration_generation_after_reuse": carrier.generation,
            "baseline_reload_used": False,
            "restoration_method": "EXACT_PUBLIC_FRAME_INVERSE_PLUS_TOPOLOGY_REMATERIALIZE_AND_SUBTRACT_ON_SAME_BACKINGS",
        },
        "controls": {
            **controls,
            "missing_word_inverse_error_field_cells": primary[
                "missing_inverse_error_field_cells"
            ],
            "naive_transported_coefficient_diagonal_rejected": primary_diagnostics[
                "naive_position_coefficient_mismatch_cells"
            ] > 0,
        },
        "resource_law": {
            "accepted_active_numeric_field_cells": 3 * len(source)
            + len(m204.CHANNELS)
            + 4,
            "retained_public_topology_descriptor_integers": 8943,
            "named_algorithm_field_and_descriptor_slots": 3 * len(source)
            + len(m204.CHANNELS)
            + 4
            + 8943,
            "accepted_full_lifecycle_logical_slot_peak": max(
                3 * len(source) + len(m204.CHANNELS) + 4 + 8943,
                compiler_work["compiler_logical_integer_slot_peak"],
                len(source) * (len(m204.scalar.predecessor.refined_signature(
                    m204.scalar.decode_code(topology.bracelet_codes[0])
                )) + 3),
            ),
            "primary_forward_frame_work": forward_frame,
            "primary_forward_m204_conjugate_work": forward_wave,
            "m204_sorted_searches_retained": forward_wave[
                "sorted_code_searches"
            ],
            "m204_fanout_candidates_retained": forward_wave[
                "inverse_candidate_moves"
            ],
            "verification_only_frame_matrix_field_cells": public_frame[
                "verification_only_matrix_field_cells"
            ],
            "verification_only_target_histogram_scalars": GRID * len(targets),
            "python_object_bigint_allocator_interpreter_timing_and_whole_process_peaks_excluded": True,
        },
        "matched_classical_baselines": [
            "IDENTICAL_FOUR_INTEGER_PUBLIC_FRAME_DESCRIPTOR_PLUS_M204_EIGHT_CHANNEL_VECTOR_STREAM",
            "M204_IDENTICAL_EIGHT_CHANNEL_IMPLICIT_DIHEDRAL_VECTOR_STREAM",
        ],
        "preserved_subclaims": [
            "M204_EXACT_EXECUTION_RESTORATION_AND_REUSE",
            "M206_FIXED_QUOTIENT_AND_DIRECT_KERNEL_ROUTE_DIAGNOSTIC",
            "PUBLIC_FOURIER_FRAME_ENDPOINT_AND_INVERSE_ARE_EXACT",
        ],
        "rejected_interpretations": [
            "POSITION_DIAGONAL_IS_COEFFICIENTWISE_IN_THE_TRANSPORTED_BASIS",
            "MOVING_FRAME_REMOVES_M204_SEARCH_OR_FANOUT",
            "CATVM_CUSTODY",
            "DISTINCT_PHASE_RESOURCE",
            "COMPUTATIONAL_ADVANTAGE",
            "SMALL_WALL_CROSSING",
            "PHYSICAL_WAVEFORM_EXECUTION",
            "REPLACEMENT_OF_PHYSICAL_BITS_WITH_PI",
            "UNBOUNDED_CATALYTIC_COMPUTATION",
        ],
        "catvm_custody": False,
        "distinct_phase_resource_established": False,
        "computational_advantage": False,
        "small_wall_crossed": False,
        "physical_waveform_execution": False,
        "physical_bit_replacement": False,
        "unbounded_computation_established": False,
        "terminal": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
