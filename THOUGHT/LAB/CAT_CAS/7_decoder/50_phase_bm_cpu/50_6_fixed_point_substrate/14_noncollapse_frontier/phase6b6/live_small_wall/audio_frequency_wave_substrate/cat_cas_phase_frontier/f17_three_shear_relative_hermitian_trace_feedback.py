#!/usr/bin/env python3
"""Bounded relative-Hermitian trace feedback on the exact F17 pair carrier.

This diagnostic borrows the M116 two-by-eight carrier backing, loads a public
17-root seed vector, applies three noncommuting resident shears, and projects
one final integer trace.  The inverse shears and public seed unload restore the
actual backing exactly before an unrelated reuse transaction.

The relative trace and root injections are fixed-rank integer recurrences.
Compact classical software can execute the identical recurrence, so the
experiment tests a phase coupling law but does not establish a distinct phase
resource or a computational advantage.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from typing import Any

import f17_cubic_chain_period17_quadratic_extension_resident_carrier as pair


real = pair.real
base = pair.base
cyclo = pair.cyclo

RealElement = pair.RealElement
SplitElement = pair.SplitElement
SplitVector = pair.SplitVector

ZERO = pair.split_zero()
PUBLIC_ROOT_TABLE = (
    *pair.QUADRATIC_EXTENSION_TABLE,
    pair.full_to_split(cyclo.ring_monomial(16)),
)


def fail(message: str) -> None:
    raise RuntimeError(message)


@dataclass(frozen=True)
class Shear:
    left: int
    right: int
    target: int
    root_exponent: int

    def payload_bits(self) -> int:
        return sum(
            base.signed_bits(value)
            for value in (self.left, self.right, self.target, self.root_exponent)
        )


PUBLIC_PLAN = (
    Shear(0, 1, 2, 5),
    Shear(1, 2, 0, 11),
    Shear(2, 0, 1, 3),
)


def validate_plan(plan: tuple[Shear, ...]) -> None:
    if len(plan) != 3:
        fail("the bounded feedback diagnostic requires exactly three shears")
    for shear in plan:
        indices = (shear.left, shear.right, shear.target)
        if len(set(indices)) != 3 or any(not 0 <= index < cyclo.PRIME for index in indices):
            fail("invalid public shear topology")
        if not 0 <= shear.root_exponent < cyclo.PRIME:
            fail("invalid public root exponent")


validate_plan(PUBLIC_PLAN)


@dataclass
class TraceStats:
    real_subfield_ring_multiplications: int = 0
    real_subfield_coefficient_multiplications: int = 0
    relative_hermitian_trace_calls: int = 0
    relative_hermitian_trace_real_multiplications: int = 0
    fixed_root_injection_calls: int = 0
    fixed_root_injection_real_multiplications: int = 0
    fixed_root_action_steps: int = 0
    carrier_coordinate_additions: int = 0
    carrier_coordinate_subtractions: int = 0
    seed_loads: int = 0
    seed_unloads: int = 0
    forward_shears: int = 0
    inverse_shears: int = 0
    boundary_trace_evaluations: int = 0
    maximum_relative_trace_live_payload_bits: int = 0
    maximum_root_injection_live_payload_bits: int = 0
    maximum_shear_live_payload_bits: int = 0
    maximum_seed_vector_payload_bits: int = 0
    maximum_seed_load_unload_live_payload_bits: int = 0
    maximum_carrier_resident_payload_bits: int = 0
    maximum_accepted_resident_plus_work_payload_bits: int = 0
    maximum_boundary_payload_bits: int = 0

    def as_json(self) -> dict[str, int]:
        return {
            name: int(value)
            for name, value in vars(self).items()
        }


def real_subtract(left: RealElement, right: RealElement) -> RealElement:
    return tuple(
        left_value - right_value
        for left_value, right_value in zip(left, right, strict=True)
    )  # type: ignore[return-value]


def relative_hermitian_trace(
    left: SplitElement,
    right: SplitElement,
    stats: TraceStats,
) -> RealElement:
    """Return Tr_{Q(zeta)/Q(s1)}(left*conjugate(right)) in three products."""

    a_value, b_value = left
    c_value, d_value = right
    p_value = real.real_multiply(a_value, c_value, stats)
    q_value = real.real_multiply(b_value, d_value, stats)
    r_value = real.real_multiply(
        real.real_add(a_value, b_value),
        real.real_add(c_value, d_value),
        stats,
    )
    cross = real_subtract(real_subtract(r_value, p_value), q_value)
    result = real.real_add(
        real.real_add(pair.real_scale(p_value, 2), pair.real_scale(q_value, 2)),
        pair.real_s1_multiply(cross),
    )
    stats.relative_hermitian_trace_calls += 1
    stats.relative_hermitian_trace_real_multiplications += 3
    stats.maximum_relative_trace_live_payload_bits = max(
        stats.maximum_relative_trace_live_payload_bits,
        pair.split_payload_bits(left)
        + pair.split_payload_bits(right)
        + real.real_payload_bits(p_value)
        + real.real_payload_bits(q_value)
        + real.real_payload_bits(r_value)
        + real.real_payload_bits(cross)
        + real.real_payload_bits(result),
    )
    return result


def four_product_relative_trace(left: SplitElement, right: SplitElement) -> RealElement:
    a_value, b_value = left
    c_value, d_value = right
    ac_value = real.real_multiply(a_value, c_value)
    bd_value = real.real_multiply(b_value, d_value)
    ad_value = real.real_multiply(a_value, d_value)
    bc_value = real.real_multiply(b_value, c_value)
    return real.real_add(
        real.real_add(pair.real_scale(ac_value, 2), pair.real_scale(bd_value, 2)),
        pair.real_s1_multiply(real.real_add(ad_value, bc_value)),
    )


def fixed_root_injection(
    root_exponent: int,
    source: SplitElement,
    value: RealElement,
    stats: TraceStats,
) -> SplitElement:
    """Return zeta**k * source * value with a phase-covariant source factor."""

    scaled = (
        real.real_multiply(source[0], value, stats),
        real.real_multiply(source[1], value, stats),
    )
    output = scaled
    forward_steps = root_exponent % cyclo.PRIME
    inverse_steps = (cyclo.PRIME - forward_steps) % cyclo.PRIME
    if forward_steps <= inverse_steps:
        for _ in range(forward_steps):
            a_value, b_value = output
            next_output = (
                pair.real_negate(b_value),
                real.real_add(a_value, pair.real_s1_multiply(b_value)),
            )
            stats.maximum_root_injection_live_payload_bits = max(
                stats.maximum_root_injection_live_payload_bits,
                pair.split_payload_bits(output) + pair.split_payload_bits(next_output),
            )
            output = next_output
        steps = forward_steps
    else:
        for _ in range(inverse_steps):
            a_value, b_value = output
            next_output = (
                real.real_add(pair.real_s1_multiply(a_value), b_value),
                pair.real_negate(a_value),
            )
            stats.maximum_root_injection_live_payload_bits = max(
                stats.maximum_root_injection_live_payload_bits,
                pair.split_payload_bits(output) + pair.split_payload_bits(next_output),
            )
            output = next_output
        steps = inverse_steps
    stats.fixed_root_injection_calls += 1
    stats.fixed_root_injection_real_multiplications += 2
    stats.fixed_root_action_steps += steps
    stats.maximum_root_injection_live_payload_bits = max(
        stats.maximum_root_injection_live_payload_bits,
        pair.split_payload_bits(source)
        + real.real_payload_bits(value)
        + pair.split_payload_bits(scaled)
        + pair.split_payload_bits(output),
    )
    return output


def public_seed(
    family: str,
    *,
    global_phase_offset: int = 0,
    single_site_perturbation: tuple[int, int] | None = None,
) -> tuple[SplitElement, ...]:
    program = cyclo.adaptive.compile_program(18, family)
    coefficients = program.unary_coefficients[0]
    return tuple(
        PUBLIC_ROOT_TABLE[
            (
                cyclo.adaptive.unary_phase(coefficients, value)
                + global_phase_offset
                + (
                    single_site_perturbation[1]
                    if single_site_perturbation is not None
                    and single_site_perturbation[0] == value
                    else 0
                )
            )
            % cyclo.PRIME
        ]
        for value in range(cyclo.PRIME)
    )


def public_program_sha256(family: str) -> str:
    program = cyclo.adaptive.compile_program(18, family)
    return hashlib.sha256(cyclo.adaptive.encoded_program(program)).hexdigest()


def public_program_payload_bits(family: str) -> int:
    program = cyclo.adaptive.compile_program(18, family)
    return len(cyclo.adaptive.encoded_program(program)) * 8


@dataclass
class TraceCarrier:
    output: SplitVector
    generation: int = 0
    lease: int = 0
    active: bool = False
    pending_operations: int = 0
    phase: str = "RESTORED"
    family: str = ""

    @classmethod
    def create(cls) -> "TraceCarrier":
        return cls([ZERO for _ in range(cyclo.PRIME)])

    def backing_identity(self) -> int:
        return id(self.output)

    def all_zero(self) -> bool:
        return (
            all(value == ZERO for value in self.output)
            and not self.active
            and self.pending_operations == 0
            and self.phase == "RESTORED"
            and self.family == ""
        )

    def resident_payload_bits(self) -> int:
        state_bits = sum(
            base.signed_bits(value)
            for value in (self.generation, self.lease, self.pending_operations)
        )
        # active (1), five-valued phase (3), and empty/PRIMARY/REUSE family (2).
        return pair.split_vector_payload_bits(self.output) + state_bits + 6

    def canonical_state(self) -> dict[str, Any]:
        return {
            "all_phase_cells_zero": all(value == ZERO for value in self.output),
            "generation": self.generation,
            "lease": self.lease,
            "active": self.active,
            "pending_operations": self.pending_operations,
            "phase": self.phase,
            "family_cleared": self.family == "",
        }


def record_carrier(carrier: TraceCarrier, stats: TraceStats) -> int:
    payload = carrier.resident_payload_bits()
    stats.maximum_carrier_resident_payload_bits = max(
        stats.maximum_carrier_resident_payload_bits,
        payload,
    )
    return payload


def load_public_seed(carrier: TraceCarrier, family: str, stats: TraceStats) -> None:
    if not carrier.all_zero():
        fail("trace carrier was not restored")
    before = carrier.resident_payload_bits()
    seed = public_seed(family)
    seed_bits = pair.split_vector_payload_bits(list(seed))
    new_output = [
        pair.split_add(actual, value)
        for actual, value in zip(carrier.output, seed, strict=True)
    ]
    new_bits = pair.split_vector_payload_bits(new_output)
    stats.maximum_seed_vector_payload_bits = max(
        stats.maximum_seed_vector_payload_bits, seed_bits
    )
    stats.maximum_seed_load_unload_live_payload_bits = max(
        stats.maximum_seed_load_unload_live_payload_bits,
        before + seed_bits + new_bits,
    )
    stats.maximum_accepted_resident_plus_work_payload_bits = max(
        stats.maximum_accepted_resident_plus_work_payload_bits,
        stats.maximum_seed_load_unload_live_payload_bits,
    )
    carrier.output[:] = new_output
    carrier.lease += 1
    carrier.active = True
    carrier.pending_operations = 4
    carrier.phase = "PUBLIC_SEED_RESIDENT"
    carrier.family = family
    stats.seed_loads += 1
    stats.carrier_coordinate_additions += cyclo.PRIME * 16
    record_carrier(carrier, stats)


def apply_shear(
    carrier: TraceCarrier,
    shear: Shear,
    ordinal: int,
    stats: TraceStats,
    *,
    inverse: bool = False,
) -> None:
    if not 0 <= ordinal < len(PUBLIC_PLAN) or shear != PUBLIC_PLAN[ordinal]:
        fail("shear descriptor does not match the compiled public plan")
    if inverse:
        expected_phase = f"FORWARD_{ordinal + 1}_RESIDENT"
    else:
        expected_phase = (
            "PUBLIC_SEED_RESIDENT" if ordinal == 0 else f"FORWARD_{ordinal}_RESIDENT"
        )
    if carrier.phase != expected_phase:
        fail("trace shear order or direction changed")
    before = record_carrier(carrier, stats)
    traced = relative_hermitian_trace(
        carrier.output[shear.left], carrier.output[shear.right], stats
    )
    injected = fixed_root_injection(
        shear.root_exponent,
        carrier.output[shear.left],
        traced,
        stats,
    )
    target_before = carrier.output[shear.target]
    target_after = (
        pair.split_subtract(target_before, injected)
        if inverse
        else pair.split_add(target_before, injected)
    )
    carrier.output[shear.target] = target_after
    work = max(
        stats.maximum_relative_trace_live_payload_bits,
        stats.maximum_root_injection_live_payload_bits,
        pair.split_payload_bits(target_before)
        + pair.split_payload_bits(injected)
        + pair.split_payload_bits(target_after),
    )
    stats.maximum_shear_live_payload_bits = max(stats.maximum_shear_live_payload_bits, work)
    stats.maximum_accepted_resident_plus_work_payload_bits = max(
        stats.maximum_accepted_resident_plus_work_payload_bits,
        before + work,
    )
    if inverse:
        stats.inverse_shears += 1
        stats.carrier_coordinate_subtractions += 16
        carrier.pending_operations -= 1
        carrier.phase = "PUBLIC_SEED_RESIDENT" if ordinal == 0 else f"FORWARD_{ordinal}_RESIDENT"
    else:
        stats.forward_shears += 1
        stats.carrier_coordinate_additions += 16
        carrier.phase = f"FORWARD_{ordinal + 1}_RESIDENT"
    record_carrier(carrier, stats)


def project_final_boundary(carrier: TraceCarrier, stats: TraceStats) -> int:
    if carrier.phase != f"FORWARD_{len(PUBLIC_PLAN)}_RESIDENT":
        fail("only the final trace boundary may be projected")
    traced = relative_hermitian_trace(carrier.output[1], carrier.output[2], stats)
    boundary = real.real_trace(traced)
    stats.boundary_trace_evaluations += 1
    stats.maximum_boundary_payload_bits = max(
        stats.maximum_boundary_payload_bits,
        real.real_payload_bits(traced) + base.signed_bits(boundary),
    )
    stats.maximum_accepted_resident_plus_work_payload_bits = max(
        stats.maximum_accepted_resident_plus_work_payload_bits,
        carrier.resident_payload_bits() + stats.maximum_relative_trace_live_payload_bits,
    )
    return boundary


def unload_public_seed(carrier: TraceCarrier, family: str, stats: TraceStats) -> None:
    if carrier.phase != "PUBLIC_SEED_RESIDENT" or carrier.family != family:
        fail("public seed unload was reordered or used the wrong family")
    before = carrier.resident_payload_bits()
    seed = public_seed(family)
    seed_bits = pair.split_vector_payload_bits(list(seed))
    new_output = [
        pair.split_subtract(actual, value)
        for actual, value in zip(carrier.output, seed, strict=True)
    ]
    new_bits = pair.split_vector_payload_bits(new_output)
    stats.maximum_seed_vector_payload_bits = max(
        stats.maximum_seed_vector_payload_bits, seed_bits
    )
    stats.maximum_seed_load_unload_live_payload_bits = max(
        stats.maximum_seed_load_unload_live_payload_bits,
        before + seed_bits + new_bits,
    )
    stats.maximum_accepted_resident_plus_work_payload_bits = max(
        stats.maximum_accepted_resident_plus_work_payload_bits,
        stats.maximum_seed_load_unload_live_payload_bits,
    )
    carrier.output[:] = new_output
    stats.seed_unloads += 1
    stats.carrier_coordinate_subtractions += cyclo.PRIME * 16
    carrier.pending_operations -= 1
    carrier.active = False
    carrier.phase = "RESTORED"
    carrier.family = ""
    carrier.generation += 1
    record_carrier(carrier, stats)
    if not carrier.all_zero():
        fail("three-shear trace carrier did not restore exactly")


@dataclass
class Transaction:
    boundary: int
    stats: TraceStats
    restored_exactly: bool
    same_backing: bool


def execute_transaction(carrier: TraceCarrier, family: str) -> Transaction:
    if not isinstance(carrier, TraceCarrier):
        fail("null or invalid trace carrier")
    backing = carrier.backing_identity()
    stats = TraceStats()
    load_public_seed(carrier, family, stats)
    for ordinal, shear in enumerate(PUBLIC_PLAN):
        apply_shear(carrier, shear, ordinal, stats)
    boundary = project_final_boundary(carrier, stats)
    for ordinal in reversed(range(len(PUBLIC_PLAN))):
        apply_shear(carrier, PUBLIC_PLAN[ordinal], ordinal, stats, inverse=True)
    unload_public_seed(carrier, family, stats)
    return Transaction(
        boundary=boundary,
        stats=stats,
        restored_exactly=carrier.all_zero(),
        same_backing=carrier.backing_identity() == backing,
    )


def coupling_disabled_boundary(family: str) -> int:
    seed = public_seed(family)
    stats = TraceStats()
    return real.real_trace(relative_hermitian_trace(seed[1], seed[2], stats))


def evaluate_vector(
    seed: tuple[SplitElement, ...],
    plan: tuple[Shear, ...] = PUBLIC_PLAN,
) -> tuple[int, bool]:
    """Direct exact control evaluator with forward and reverse shears."""

    validate_plan(plan)
    vector = list(seed)
    stats = TraceStats()
    for shear in plan:
        traced = relative_hermitian_trace(
            vector[shear.left], vector[shear.right], stats
        )
        injected = fixed_root_injection(
            shear.root_exponent, vector[shear.left], traced, stats
        )
        vector[shear.target] = pair.split_add(vector[shear.target], injected)
    boundary = real.real_trace(
        relative_hermitian_trace(vector[1], vector[2], stats)
    )
    for shear in reversed(plan):
        traced = relative_hermitian_trace(
            vector[shear.left], vector[shear.right], stats
        )
        injected = fixed_root_injection(
            shear.root_exponent, vector[shear.left], traced, stats
        )
        vector[shear.target] = pair.split_subtract(vector[shear.target], injected)
    return boundary, tuple(vector) == seed


def forward_vector(
    seed: tuple[SplitElement, ...],
    plan: tuple[Shear, ...],
) -> tuple[SplitElement, ...]:
    """Return an exact control state after the declared forward plan."""

    vector = list(seed)
    stats = TraceStats()
    for shear in plan:
        traced = relative_hermitian_trace(
            vector[shear.left], vector[shear.right], stats
        )
        injected = fixed_root_injection(
            shear.root_exponent, vector[shear.left], traced, stats
        )
        vector[shear.target] = pair.split_add(vector[shear.target], injected)
    return tuple(vector)


def algebra_controls() -> dict[str, bool]:
    seeds = public_seed("PRIMARY") + public_seed("REUSE")
    trace_parity = all(
        relative_hermitian_trace(left, right, TraceStats())
        == four_product_relative_trace(left, right)
        for left in seeds[:6]
        for right in seeds[-6:]
    )
    generic_injection_parity = all(
        fixed_root_injection(
            exponent,
            pair.QUADRATIC_EXTENSION_TABLE[(exponent + 2) % 16],
            pair.REAL_ONE,
            TraceStats(),
        )
        == pair.split_multiply(
            pair.QUADRATIC_EXTENSION_TABLE[exponent],
            pair.QUADRATIC_EXTENSION_TABLE[(exponent + 2) % 16],
        )
        for exponent in (3, 5, 11)
    )
    return {
        "three_product_equals_four_product_on_36_cross_family_pairs": trace_parity,
        "fixed_root_injection_equals_generic_pair_product": generic_injection_parity,
        "carrier_integer_coordinate_count_is_272": cyclo.PRIME * 16 == 272,
        "public_plan_has_forward_feedback_dependencies": (
            PUBLIC_PLAN[1].right == PUBLIC_PLAN[0].target
            and PUBLIC_PLAN[2].right == PUBLIC_PLAN[1].target
        ),
    }


def controls() -> dict[str, bool]:
    reordered = TraceCarrier.create()
    reordered_stats = TraceStats()
    load_public_seed(reordered, "PRIMARY", reordered_stats)
    for ordinal, shear in enumerate(PUBLIC_PLAN):
        apply_shear(reordered, shear, ordinal, reordered_stats)
    reordered_before = list(reordered.output)
    reordered_rejected = False
    try:
        apply_shear(reordered, PUBLIC_PLAN[0], 0, reordered_stats, inverse=True)
    except RuntimeError:
        reordered_rejected = reordered.output == reordered_before

    premature = TraceCarrier.create()
    premature_stats = TraceStats()
    load_public_seed(premature, "PRIMARY", premature_stats)
    premature_rejected = False
    try:
        project_final_boundary(premature, premature_stats)
    except RuntimeError:
        premature_rejected = premature.phase == "PUBLIC_SEED_RESIDENT"

    missing = TraceCarrier.create()
    missing_stats = TraceStats()
    load_public_seed(missing, "PRIMARY", missing_stats)
    apply_shear(missing, PUBLIC_PLAN[0], 0, missing_stats)
    missing_detected = not missing.all_zero() and missing.pending_operations == 4

    wrong = TraceCarrier.create()
    wrong_stats = TraceStats()
    load_public_seed(wrong, "PRIMARY", wrong_stats)
    for ordinal, shear in enumerate(PUBLIC_PLAN):
        apply_shear(wrong, shear, ordinal, wrong_stats)
    wrong_before = list(wrong.output)
    wrong_shear = Shear(2, 0, 1, 4)
    wrong_descriptor_rejected = False
    try:
        apply_shear(wrong, wrong_shear, 2, wrong_stats, inverse=True)
        wrong_descriptor_rejected = False
    except RuntimeError:
        wrong_descriptor_rejected = wrong.output == wrong_before

    wrong_arithmetic_seed = public_seed("PRIMARY")
    wrong_arithmetic = list(forward_vector(wrong_arithmetic_seed, PUBLIC_PLAN))
    wrong_arithmetic_stats = TraceStats()
    for ordinal in reversed(range(len(PUBLIC_PLAN))):
        shear = PUBLIC_PLAN[ordinal]
        exponent = 4 if ordinal == 2 else shear.root_exponent
        traced = relative_hermitian_trace(
            wrong_arithmetic[shear.left],
            wrong_arithmetic[shear.right],
            wrong_arithmetic_stats,
        )
        injected = fixed_root_injection(
            exponent,
            wrong_arithmetic[shear.left],
            traced,
            wrong_arithmetic_stats,
        )
        wrong_arithmetic[shear.target] = pair.split_subtract(
            wrong_arithmetic[shear.target], injected
        )
    wrong_arithmetic_fails = tuple(wrong_arithmetic) != wrong_arithmetic_seed

    null_rejected = False
    try:
        execute_transaction(None, "PRIMARY")  # type: ignore[arg-type]
    except RuntimeError:
        null_rejected = True

    alias_rejected = False
    try:
        validate_plan((Shear(0, 1, 0, 5), *PUBLIC_PLAN[1:]))
    except RuntimeError:
        alias_rejected = True

    global_phase_invariant = all(
        evaluate_vector(public_seed(family, global_phase_offset=offset))[0]
        == evaluate_vector(public_seed(family))[0]
        for family in ("PRIMARY", "REUSE")
        for offset in (1, 8, 16)
    )
    single_site_perturbation_detected = all(
        evaluate_vector(
            public_seed(family, single_site_perturbation=(2, 1))
        )[0]
        != evaluate_vector(public_seed(family))[0]
        for family in ("PRIMARY", "REUSE")
    )
    alternate_plan = (
        Shear(0, 2, 3, 4),
        Shear(2, 3, 1, 7),
        Shear(3, 1, 0, 9),
    )
    alternate_descriptor_restores = all(
        evaluate_vector(public_seed(family), alternate_plan)[1]
        for family in ("PRIMARY", "REUSE")
    )
    pairwise_noncommuting = all(
        forward_vector(public_seed(family), (PUBLIC_PLAN[left], PUBLIC_PLAN[right]))
        != forward_vector(public_seed(family), (PUBLIC_PLAN[right], PUBLIC_PLAN[left]))
        for family in ("PRIMARY", "REUSE")
        for left, right in ((0, 1), (0, 2), (1, 2))
    )

    mutation = TraceCarrier.create()
    mutation_stats = TraceStats()
    load_public_seed(mutation, "PRIMARY", mutation_stats)
    for ordinal, shear in enumerate(PUBLIC_PLAN):
        apply_shear(mutation, shear, ordinal, mutation_stats)
    mutation.output[4] = pair.split_add(mutation.output[4], pair.split_one())
    mutation_detected = False
    try:
        for ordinal in reversed(range(len(PUBLIC_PLAN))):
            apply_shear(
                mutation,
                PUBLIC_PLAN[ordinal],
                ordinal,
                mutation_stats,
                inverse=True,
            )
        unload_public_seed(mutation, "PRIMARY", mutation_stats)
    except RuntimeError:
        mutation_detected = not mutation.all_zero()

    return {
        "reordered_inverse_rejected_before_mutation": reordered_rejected,
        "premature_projection_rejected": premature_rejected,
        "missing_inverse_leaves_detectable_resident_state": missing_detected,
        "wrong_descriptor_rejected_before_mutation": wrong_descriptor_rejected,
        "wrong_arithmetic_inverse_fails_restoration": wrong_arithmetic_fails,
        "source_target_alias_rejected": alias_rejected,
        "global_phase_rotation_invariant": global_phase_invariant,
        "single_site_phase_perturbation_changes_boundary": (
            single_site_perturbation_detected
        ),
        "alternate_valid_descriptor_restores": alternate_descriptor_restores,
        "all_three_shear_pairs_noncommute_on_both_public_families": (
            pairwise_noncommuting
        ),
        "resident_mutation_detected": mutation_detected,
        "null_carrier_rejected": null_rejected,
        "snapshot_reload_absent": True,
    }


def reuse_case() -> dict[str, Any]:
    carrier = TraceCarrier.create()
    backing = carrier.backing_identity()
    primary = execute_transaction(carrier, "PRIMARY")
    reuse = execute_transaction(carrier, "REUSE")
    fresh = execute_transaction(TraceCarrier.create(), "REUSE")
    reuse_signature = reuse.stats.as_json()
    fresh_signature = fresh.stats.as_json()
    metadata_sensitive_fields = {
        "maximum_carrier_resident_payload_bits",
        "maximum_accepted_resident_plus_work_payload_bits",
        "maximum_seed_load_unload_live_payload_bits",
    }
    reuse_structural = {
        name: value
        for name, value in reuse_signature.items()
        if name not in metadata_sensitive_fields
    }
    fresh_structural = {
        name: value
        for name, value in fresh_signature.items()
        if name not in metadata_sensitive_fields
    }
    return {
        "primary_boundary": primary.boundary,
        "reuse_boundary": reuse.boundary,
        "fresh_reuse_boundary": fresh.boundary,
        "primary_restored_exactly": primary.restored_exactly,
        "reuse_restored_exactly": reuse.restored_exactly,
        "same_original_backing": carrier.backing_identity() == backing,
        "fresh_restored_reuse_boundary_equal": reuse.boundary == fresh.boundary,
        "fresh_restored_reuse_rank_and_arithmetic_signature_equal": (
            reuse_structural == fresh_structural
        ),
        "fresh_restored_reuse_full_metadata_sensitive_signature_equal": (
            reuse_signature == fresh_signature
        ),
        "restored_minus_fresh_resident_payload_bits": (
            reuse.stats.maximum_carrier_resident_payload_bits
            - fresh.stats.maximum_carrier_resident_payload_bits
        ),
        "metadata_difference_explained_by_monotone_generation_and_lease": True,
        "generation": carrier.generation,
        "lease": carrier.lease,
        "baseline_reload": False,
        "canonical_restored_state": carrier.canonical_state(),
    }


def main() -> int:
    if len(sys.argv) != 1:
        fail("usage: f17_three_shear_relative_hermitian_trace_feedback.py")
    plan_bits = sum(shear.payload_bits() for shear in PUBLIC_PLAN)
    root_table_bits = sum(
        pair.split_payload_bits(value)
        for value in PUBLIC_ROOT_TABLE
    )
    cases: list[dict[str, Any]] = []
    for family in ("PRIMARY", "REUSE"):
        carrier = TraceCarrier.create()
        transaction = execute_transaction(carrier, family)
        program_bits = public_program_payload_bits(family)
        cases.append(
            {
                "family": family,
                "public_program_sha256": public_program_sha256(family),
                "public_program_payload_bits": program_bits,
                "boundary": transaction.boundary,
                "coupling_disabled_boundary": coupling_disabled_boundary(family),
                "declared_cross_cell_erasure_boundary": 0,
                "restored_exactly": transaction.restored_exactly,
                "same_backing": transaction.same_backing,
                "stats": transaction.stats.as_json(),
                "accepted_named_component_maxima_sum_bits": (
                    transaction.stats.maximum_accepted_resident_plus_work_payload_bits
                    + plan_bits
                    + root_table_bits
                    + program_bits
                    + transaction.stats.maximum_boundary_payload_bits
                ),
                "canonical_restored_state": carrier.canonical_state(),
            }
        )
    carrier_controls = controls()
    algebra = algebra_controls()
    reuse = reuse_case()
    result = {
        "result": "PASS",
        "experiment": "THREE_CELL_TRACE_FEEDBACK_TRIANGULAR_COUPLING_COLLAPSE_TEST",
        "claim_candidate": (
            "BOUNDED_EXACT_THREE_NONCOMMUTING_RELATIVE_HERMITIAN_"
            "TRACE_SHEARS_EXECUTE_AND_RESTORE_ON_THE_TWO_BY_EIGHT_"
            "F17_PHASE_CARRIER_BUT_COLLAPSE_TO_AN_IDENTICAL_FIXED_"
            "RANK_COMPACT_CLASSICAL_INTEGER_RECURRENCE"
        ),
        "classification_candidate": "SOURCE_AUDITED_PACKAGE_LOCAL",
        "verification_level_candidate": "PACKAGE_SELF_REVIEW",
        "restoration_class": "EXACT_ALGEBRAIC_RESTORATION",
        "execution_scope": "WARM_DIRECT_PROCESS_SOFTWARE_ONLY",
        "catvm_controller_backend_traffic_bits": 0,
        "carrier_origin": "M116_TWO_BY_EIGHT_QUADRATIC_EXTENSION_PAIR_ALGEBRA",
        "public_seed_origin": "FIRST_UNARY_PHASE_ROW_OF_PUBLIC_18_NODE_F17_PROGRAM",
        "public_plan": [vars(shear) for shear in PUBLIC_PLAN],
        "public_plan_payload_bits": plan_bits,
        "retained_public_root_table_payload_bits": root_table_bits,
        "logical_carrier_cells": cyclo.PRIME,
        "integer_coordinates_per_cell": 16,
        "logical_integer_coordinates": cyclo.PRIME * 16,
        "intermediate_phase_cells_projected": False,
        "boundary_projection": "ONE_FINAL_INTEGER_REAL_SUBFIELD_TRACE",
        "full_cyclotomic_carrier_materializations": 0,
        "split_to_full_scalar_lifts": 0,
        "retained_inverse_history_bytes": 0,
        "cases": cases,
        "algebra_controls": algebra,
        "carrier_controls": carrier_controls,
        "restoration_reuse_case": reuse,
        "relative_phase_sensitivity_diagnostic": {
            "coupling_changes_tested_boundary": all(
                case["boundary"] != case["coupling_disabled_boundary"]
                for case in cases
            ),
            "declared_cross_cell_erasure_boundary_is_zero": True,
            "executed_physical_dephasing_model": False,
            "relative_phase_sensitivity_is_distinct_resource": False,
        },
        "matched_classical": {
            "strongest_compact_recurrence": "IDENTICAL_TWO_BY_EIGHT_INTEGER_RECURRENCE",
            "relative_trace_real_multiplications": 3,
            "fixed_root_injection_real_multiplications": 2,
            "carrier_integer_coordinates": cyclo.PRIME * 16,
            "same_public_seed_plan_boundary_and_exact_arithmetic": True,
            "equal_or_lower_resource_signature_available": True,
            "comparison_establishes_advantage": False,
        },
        "resource_law": {
            "carrier_state_and_machine_metadata_counted": True,
            "relative_trace_operands_products_and_result_counted": True,
            "retained_public_root_table_and_named_injection_work_counted": True,
            "public_plan_payload_counted": True,
            "public_program_payload_counted": True,
            "seed_load_and_unload_arithmetic_counted": True,
            "projection_and_boundary_payload_counted": True,
            "restoration_verification_and_fresh_reuse_reported_separately": True,
            "root_table_compilation_occurs_outside_warm_transaction": True,
            "root_table_and_program_compilation_integer_work_bounded": False,
            "exact_real_multiply_internal_accumulator_scratch_bounded": False,
            "named_component_maxima_sum_is_complete_material_peak": False,
            "python_object_allocator_native_library_and_bigint_internal_bytes_bounded": False,
            "whole_process_peak_bounded": False,
        },
        "observation": (
            "THE_THREE_SHEAR_FEEDBACK_IS_PAIRWISE_NONCOMMUTING_ON_"
            "THE_TESTED_FAMILIES_AND_RELATIVE_PHASE_TRACE_COUPLING_"
            "CHANGES_THE_BOUNDARY_BUT_THE_COMPLETE_ACCEPTED_"
            "LAW_IS_AN_IDENTICAL_FIXED_RANK_INTEGER_RECURRENCE"
        ),
        "not_established": [
            "DISTINCT_PHASE_RESOURCE",
            "COMPUTATIONAL_ADVANTAGE",
            "SMALL_WALL_CROSSING",
            "MACHINE_ENFORCED_NO_SMUGGLE_OR_CATVM_CUSTODY",
            "CATALYTIC_INFERENCE",
            "PHYSICAL_WAVEFORM_EXECUTION",
            "REPLACEMENT_OF_PHYSICAL_BITS_WITH_PI",
            "UNBOUNDED_COMPUTATION",
        ],
        "next_experiment": "VARIABLE_RANK_PHASE_COUPLING_WITH_COMPACT_CLASSICAL_TENSOR_BASELINE",
        "next_obstruction": (
            "FIXED_RANK_RELATIVE_HERMITIAN_TRACE_FEEDBACK_REMAINS_"
            "AN_IDENTICAL_COMPACT_CLASSICAL_POLYNOMIAL_RECURRENCE"
        ),
        "terminal": False,
    }
    hard_gate = {
        "case_restoration": all(
            case["restored_exactly"]
            and case["same_backing"]
            and case["canonical_restored_state"]["all_phase_cells_zero"]
            for case in cases
        ),
        "expected_boundaries": [case["boundary"] for case in cases] == [197, 112],
        "expected_disabled_boundaries": [
            case["coupling_disabled_boundary"] for case in cases
        ] == [16, -1],
        "relative_phase_effect": result["relative_phase_sensitivity_diagnostic"][
            "coupling_changes_tested_boundary"
        ],
        "algebra": all(algebra.values()),
        "controls": all(carrier_controls.values()),
        "reuse": (
            reuse["primary_restored_exactly"]
            and reuse["reuse_restored_exactly"]
            and reuse["same_original_backing"]
            and reuse["fresh_restored_reuse_boundary_equal"]
            and reuse["fresh_restored_reuse_rank_and_arithmetic_signature_equal"]
            and not reuse["fresh_restored_reuse_full_metadata_sensitive_signature_equal"]
            and reuse["restored_minus_fresh_resident_payload_bits"] == 2
        ),
        "no_distinct_resource_claim": "DISTINCT_PHASE_RESOURCE" in result["not_established"],
    }
    if not all(hard_gate.values()):
        fail("three-shear trace diagnostic failed: " + json.dumps(hard_gate, sort_keys=True))
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
