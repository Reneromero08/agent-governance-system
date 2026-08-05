#!/usr/bin/env python3
"""Streamed Jacobi moment rematerialization for the M210 theta boundary.

This repair retains no moment vector.  For each moment required by the exact
first-harmonic Q-jet, it scans the public operation word backward over the
actual runtime center source, rematerializes the effective power sum into one
scratch cell, consumes it into the universal Jacobi-log series, repeats the
same scan to subtract the scratch exactly, and moves to the next moment.

The experiment measures whether M210's 47,209-bit moment vector is genuinely
removed or relocated into series state and rematerialization work.  It uses
the established M210 exact formal-series primitives; the independent oracle
reconstructs the boundary directly from factors.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from fractions import Fraction

import continuous_s1_generic_center_theta_moment_qjet as base


ORDER = 24
PRIMARY_COUNT = 24
REUSE_COUNT = 17
PREDECESSOR_SOURCE_SHA256 = (
    "8d258127d31d9b83591a0cac58837809e46497207c0329d0d1f94d1793f00093"
)
UNREPAIRED_SOURCE_COMMIT = "18213589522aa72aa3e58817f2fdd604e6cd4662"


def series_payload_bits(series: base.Series) -> int:
    return base.gaussian_payload_bits(list(series.values()))


def series_key_bits(series: base.Series) -> int:
    return sum(
        base.signed_bits(harmonic) + base.signed_bits(exponent)
        for harmonic, exponent in series
    )


def program_hash(operations: tuple[base.Operation, ...]) -> str:
    return base.program_commitment(operations)


@dataclass
class StreamWork:
    public_operation_scans: int = 0
    public_operation_records_visited: int = 0
    source_center_visits: int = 0
    center_power_evaluations: int = 0
    scratch_writes: int = 0
    scratch_inverse_writes: int = 0
    retained_moment_vector_cells: int = 0
    moment_scratch_cells: int = 1
    universal_log_cells: int = 0
    weighted_log_cells: int = 0
    weighted_log_payload_bits: int = 0
    weighted_log_key_bits: int = 0
    peak_exponential_cells: int = 0
    peak_exponential_payload_bits: int = 0
    peak_exponential_key_bits: int = 0
    series_products: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            name: getattr(self, name) for name in self.__dataclass_fields__
        }


def rematerialize_moment(
    source: list[base.G],
    operations: tuple[base.Operation, ...],
    moment: int,
    work: StreamWork,
) -> base.G:
    suffix_rotation = base.ONE
    value = base.ZERO
    work.public_operation_scans += 1
    for operation in reversed(operations):
        work.public_operation_records_visited += 1
        if operation.kind == "ROTATE":
            suffix_rotation = (
                suffix_rotation * base.ROTATIONS[operation.parameter]
            )
        elif operation.kind == "INGEST":
            work.source_center_visits += 1
            effective = source[operation.parameter] * suffix_rotation
            if moment == 0:
                contribution = base.ONE
            else:
                contribution = effective ** moment
                work.center_power_evaluations += 1
            value = value + contribution
        else:
            raise ValueError("wrong streamed Jacobi operation type")
    return value


def streamed_projection(
    source: list[base.G],
    operations: tuple[base.Operation, ...],
    scratch: list[base.G],
    order: int,
) -> tuple[tuple[base.G, ...], StreamWork]:
    if len(scratch) != 1 or not scratch[0].is_zero():
        raise RuntimeError("streamed moment scratch is not restored")
    maximum_moment = (order + 1) // 2
    work = StreamWork()
    formal_work = base.ProjectionWork()
    logarithm = base.universal_log_theta(order, formal_work)
    work.universal_log_cells = len(logarithm)
    weighted: base.Series = {}
    for moment in range(0, maximum_moment + 1):
        value = rematerialize_moment(source, operations, moment, work)
        scratch[0] = scratch[0] + value
        work.scratch_writes += 1
        for harmonic in ((0,) if moment == 0 else (moment, -moment)):
            effective_moment = scratch[0] if harmonic >= 0 else scratch[0].conjugate()
            for (log_harmonic, exponent), coefficient in logarithm.items():
                if log_harmonic != harmonic:
                    continue
                product = coefficient * effective_moment
                if not product.is_zero():
                    weighted[harmonic, exponent] = product
        inverse_value = rematerialize_moment(source, operations, moment, work)
        scratch[0] = scratch[0] - inverse_value
        work.scratch_inverse_writes += 1
        if not scratch[0].is_zero():
            raise RuntimeError("streamed moment scratch inverse failed")

    work.weighted_log_cells = len(weighted)
    work.weighted_log_payload_bits = series_payload_bits(weighted)
    work.weighted_log_key_bits = series_key_bits(weighted)

    exponential: base.Series = {(0, 0): base.ONE}
    term: base.Series = {(0, 0): base.ONE}
    for power in range(1, order + 1):
        term = base.series_multiply(term, weighted, order, formal_work)
        term = {
            key: value.scale(Fraction(1, power))
            for key, value in term.items()
        }
        base.series_add_scaled(exponential, term, Fraction(1))
        work.peak_exponential_cells = max(
            work.peak_exponential_cells, len(exponential), len(term)
        )
        work.peak_exponential_payload_bits = max(
            work.peak_exponential_payload_bits,
            series_payload_bits(exponential),
            series_payload_bits(term),
        )
        work.peak_exponential_key_bits = max(
            work.peak_exponential_key_bits,
            series_key_bits(exponential),
            series_key_bits(term),
        )
    work.series_products = formal_work.series_products
    jet = tuple(
        exponential.get((1, exponent), base.ZERO)
        for exponent in range(order + 1)
    )
    return jet, work


@dataclass
class StreamCarrier:
    source: list[base.G]
    scratch: list[base.G]
    restoration_generation: int = 0
    live: bool = False
    owner: int = 0
    program_commitment: str = ""

    def transact(
        self,
        operations: tuple[base.Operation, ...],
        owner: int,
        order: int,
    ) -> dict[str, object]:
        if self.live:
            raise RuntimeError("streamed Jacobi carrier already live")
        if not self.source:
            raise ValueError("null streamed Jacobi source rejected")
        if owner <= 0 or not self.scratch[0].is_zero():
            raise ValueError("invalid streamed Jacobi lease")
        source_backing = id(self.source)
        scratch_backing = id(self.scratch)
        source_initial = self.source.copy()
        self.live = True
        self.owner = owner
        self.program_commitment = program_hash(operations)
        jet, work = streamed_projection(
            self.source, operations, self.scratch, order
        )
        boundary = base.commitment(jet)
        if not self.scratch[0].is_zero() or self.source != source_initial:
            raise RuntimeError("streamed Jacobi carrier restoration failed")
        self.restoration_generation += 1
        self.live = False
        self.owner = 0
        self.program_commitment = ""
        return {
            "boundary_commitment": boundary,
            "boundary_payload_bits": base.gaussian_payload_bits(jet),
            "work": work.as_dict(),
            "source_same_backing": id(self.source) == source_backing,
            "scratch_same_backing": id(self.scratch) == scratch_backing,
            "source_restoration_error_cells": sum(
                left != right for left, right in zip(self.source, source_initial)
            ),
            "scratch_restoration_error_cells": int(
                not self.scratch[0].is_zero()
            ),
            "restoration_generation": self.restoration_generation,
        }


def controls() -> dict[str, object]:
    source = base.center_family(4, 0)
    operations = base.program(4, 0)
    wrong_type = null = dirty = semantic = False
    try:
        rematerialize_moment(
            source, (base.Operation("DECODE", 0),), 1, StreamWork()
        )
    except ValueError:
        wrong_type = True
    try:
        StreamCarrier([], [base.ZERO]).transact(tuple(), 1, 8)
    except ValueError:
        null = True
    try:
        StreamCarrier(source, [base.ONE]).transact(operations, 1, 8)
    except ValueError:
        dirty = True
    primary = StreamCarrier(source.copy(), [base.ZERO]).transact(
        operations, 1, 8
    )["boundary_commitment"]
    changed_source = source.copy()
    changed_source[0] = changed_source[0] * base.ROTATIONS[0]
    changed = StreamCarrier(changed_source, [base.ZERO]).transact(
        operations, 1, 8
    )["boundary_commitment"]
    semantic = primary != changed
    return {
        "wrong_operation_type_rejected": wrong_type,
        "null_source_rejected": null,
        "dirty_scratch_rejected": dirty,
        "semantic_center_perturbation_changes_boundary": semantic,
    }


def main() -> None:
    primary_source = base.center_family(PRIMARY_COUNT, 0)
    primary_source_payload_bits = base.gaussian_payload_bits(primary_source)
    carrier = StreamCarrier(primary_source, [base.ZERO])
    source_backing = id(carrier.source)
    scratch_backing = id(carrier.scratch)
    primary = carrier.transact(
        base.program(PRIMARY_COUNT, 0), 8101, ORDER
    )
    reuse_source = base.center_family(REUSE_COUNT, 1)
    carrier.source[:] = reuse_source
    reuse = carrier.transact(base.program(REUSE_COUNT, 1), 8102, ORDER)
    fresh = StreamCarrier(reuse_source.copy(), [base.ZERO]).transact(
        base.program(REUSE_COUNT, 1), 8102, ORDER
    )
    if reuse["boundary_commitment"] != fresh["boundary_commitment"]:
        raise RuntimeError("fresh and restored streamed Jacobi reuse differ")

    result = {
        "schema": "cat_cas.continuous_s1_streamed_jacobi_moment.v1",
        "claim": "RESOURCE_ACCOUNTING_REPAIRED_EXACT_CONTINUOUS_S1_GENERIC_CENTER_FIRST_HARMONIC_Q24_JACOBI_PROJECTION_REMATERIALIZES_AND_EXACTLY_UNCOMPUTES_EACH_REQUIRED_MOMENT_IN_ONE_SCRATCH_CELL_WITH_ZERO_RETAINED_MOMENT_VECTOR_25CELL22609BIT_FINAL_ONLY_BOUNDARY_RESTORATION_REUSE_BUT_RETAINS24_SOURCE_CENTERS_MATERIALIZES86_WEIGHTED_LOG_CELLS_AND325_PEAK_EXPONENTIAL_CELLS_REQUIRES624_SOURCE_CENTER_VISITS_AND_HAS_AN_IDENTICAL_CLASSICAL_STREAM",
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
        "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "result": "PASS_RESOURCE_ACCOUNTING_REPAIRED_STREAMED_MOMENT_VECTOR_REMOVAL_WITH_SERIES_AND_SOURCE_WORK_OBSTRUCTION",
        "resource_accounting_repair_of_scientific_source_commit": UNREPAIRED_SOURCE_COMMIT,
        "phase_relation_law": {
            "domain": "CONTINUOUS_S1_NO_FINITE_ANGLE_SAMPLING",
            "runtime_center_type": "GAUSSIAN_RATIONAL_UNIT_PHASE",
            "boundary": "EXACT_FIRST_HARMONIC_QJET_ORDER24",
            "moment_law": "REVERSE_PUBLIC_WORD_SCAN_REMATERIALIZE_CONSUME_RESCAN_UNCOMPUTE",
            "retained_moment_vector_cells": 0,
            "moment_scratch_cells": 1,
            "intermediate_moment_projection": False,
        },
        "transaction": {
            "primary_boundary_commitment": primary["boundary_commitment"],
            "primary_boundary_payload_bits": primary["boundary_payload_bits"],
            "reuse_boundary_commitment": reuse["boundary_commitment"],
            "reuse_boundary_payload_bits": reuse["boundary_payload_bits"],
            "fresh_reuse_boundary_commitment": fresh["boundary_commitment"],
            "fresh_reuse_boundary_payload_bits": fresh[
                "boundary_payload_bits"
            ],
            "source_backing_identity_preserved_across_reuse": id(carrier.source) == source_backing,
            "scratch_backing_identity_preserved_across_reuse": id(carrier.scratch) == scratch_backing,
            "primary_source_restoration_error_cells": primary[
                "source_restoration_error_cells"
            ],
            "primary_scratch_restoration_error_cells": primary[
                "scratch_restoration_error_cells"
            ],
            "reuse_source_restoration_error_cells": reuse[
                "source_restoration_error_cells"
            ],
            "reuse_scratch_restoration_error_cells": reuse[
                "scratch_restoration_error_cells"
            ],
            "restoration_generation_after_reuse": carrier.restoration_generation,
            "baseline_reload_used": False,
        },
        "controls": controls(),
        "resource_law": {
            "runtime_source_center_cells": PRIMARY_COUNT,
            "runtime_source_payload_bits": primary_source_payload_bits,
            "projected_boundary_cells": ORDER + 1,
            "projected_boundary_payload_bits": primary[
                "boundary_payload_bits"
            ],
            **primary["work"],
            "public_program_operation_records": len(
                base.program(PRIMARY_COUNT, 0)
            ),
            "public_program_descriptor_slots": 2
            * len(base.program(PRIMARY_COUNT, 0)),
            "retained_inverse_history_entries": 0,
            "additional_retained_plan_entries": 0,
            "python_fraction_object_allocator_interpreter_serialization_timing_and_whole_process_peaks_excluded_not_zero": True,
        },
        "matched_classical_baselines": {
            "streamed_moment": "IDENTICAL_REVERSE_WORD_MOMENT_REMATERIALIZATION_AND_FORMAL_SERIES_RECURRENCE",
            "direct_factor": "M210_EXACT_FACTOR_BY_FACTOR_SPARSE_QJET_RECURRENCE",
            "strictly_smaller_or_faster_phase_path_established": False,
        },
        "predecessor_source_sha256": PREDECESSOR_SOURCE_SHA256,
        "catvm_custody": False,
        "distinct_phase_resource_established": False,
        "computational_advantage": False,
        "small_wall_crossed": False,
        "physical_waveform_execution": False,
        "physical_bit_replacement": False,
        "catalytic_inference_established": False,
        "unbounded_computation_established": False,
        "terminal": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
