from __future__ import annotations

from math import isclose, nan
from pathlib import Path
import sys

import pytest

PACKAGE_PARENT = (
    Path(__file__).resolve().parents[2]
    / "7_decoder"
    / "50_phase_bm_cpu"
    / "50_6_fixed_point_substrate"
    / "14_noncollapse_frontier"
)
sys.path.insert(0, str(PACKAGE_PARENT))

from constraint_relational_trace_v1.boundary_stratum_escape_audit import (  # noqa: E402
    audit_boundary_stratum_state,
    forward_compatible_memory_strata,
)
from constraint_relational_trace_v1.constraint_holo import (  # noqa: E402
    ClauseRelation,
    ConstraintHolo,
    ConstraintHoloError,
    Literal,
)
from constraint_relational_trace_v1.reduced_exact_product_phase_flow import (  # noqa: E402
    ReducedExactProductPhaseState,
)
from constraint_relational_trace_v1.self_organizing_clause_flow import (  # noqa: E402
    SelfOrganizingFlowParameters,
)


def clause(*literals: Literal) -> ClauseRelation:
    return ClauseRelation(tuple(literals))  # type: ignore[arg-type]


def symmetric_stationary_holo() -> ConstraintHolo:
    return ConstraintHolo.build(
        ("x", "y"),
        (
            clause(Literal("x"), Literal("y"), Literal("y")),
            clause(
                Literal("x", False),
                Literal("y", False),
                Literal("y", False),
            ),
        ),
    )


def memory_cap(holo: ConstraintHolo) -> float:
    parameters = SelfOrganizingFlowParameters()
    return max(
        1.0,
        parameters.long_memory_cap_factor * max(1, len(holo.clauses)),
    )


def symmetric_state(short_memory: float) -> ReducedExactProductPhaseState:
    holo = symmetric_stationary_holo()
    cap = memory_cap(holo)
    return ReducedExactProductPhaseState(
        phase_cosine=(0.0, 0.0),
        phase_sine=(1.0, 1.0),
        short_memory=(short_memory, short_memory),
        long_memory=(cap, cap),
        clause_selector=(
            (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
            (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
        ),
    )


def test_forward_compatible_inventory_contains_exactly_five_strata() -> None:
    descriptors = forward_compatible_memory_strata()
    assert len(descriptors) == 5
    assert {descriptor.label for descriptor in descriptors} == {
        "SHORT_ZERO__LONG_ZERO__C_ARBITRARY",
        "SHORT_ZERO__LONG_CAP__C_ARBITRARY",
        "SHORT_ONE__LONG_CAP__C_ARBITRARY",
        "SHORT_INTERIOR_C_EQ_GAMMA__LONG_CAP",
        "SHORT_ZERO__LONG_INTERIOR_C_EQ_DELTA",
    }
    assert sum(descriptor.has_interior_memory for descriptor in descriptors) == 2


def test_known_symmetric_non_solution_is_stationary_but_center_unresolved() -> None:
    holo = symmetric_stationary_holo()
    audit = audit_boundary_stratum_state(
        holo,
        symmetric_state(1.0),
        (
            "SHORT_ONE__LONG_CAP__C_ARBITRARY",
            "SHORT_ONE__LONG_CAP__C_ARBITRARY",
        ),
        ((True, True, True), (True, True, True)),
    )

    assert audit.pointwise_boundary_tangency_verified
    assert audit.stationary_point_verified
    assert not audit.terminal_assignment_satisfies_public_relation
    assert audit.classification == "INVARIANT_STABLE_OR_CENTER_UNRESOLVED"
    assert audit.phase_field_residual == 0.0
    assert not audit.repelling_directions
    assert all(
        isclose(clause_audit.clause_violation, 0.5)
        for clause_audit in audit.clause_audits
    )
    assert any("center directions" in item for item in audit.unresolved_conditions)


def test_same_symmetric_phase_on_short_zero_stratum_is_publicly_repelling() -> None:
    holo = symmetric_stationary_holo()
    audit = audit_boundary_stratum_state(
        holo,
        symmetric_state(0.0),
        (
            "SHORT_ZERO__LONG_CAP__C_ARBITRARY",
            "SHORT_ZERO__LONG_CAP__C_ARBITRARY",
        ),
        ((True, True, True), (True, True, True)),
    )

    assert audit.stationary_point_verified
    assert audit.classification == (
        "INVARIANT_REPELLING_IN_AT_LEAST_ONE_PUBLIC_DIRECTION"
    )
    assert audit.repelling_directions == (
        "clause[0].short_from_zero",
        "clause[1].short_from_zero",
    )


def test_satisfying_boolean_corner_is_a_satisfying_invariant_section() -> None:
    holo = ConstraintHolo.build(
        ("x", "y", "z"),
        (clause(Literal("x"), Literal("y"), Literal("z")),),
    )
    state = ReducedExactProductPhaseState(
        phase_cosine=(1.0, -1.0, -1.0),
        phase_sine=(0.0, 0.0, 0.0),
        short_memory=(0.0,),
        long_memory=(0.0,),
        clause_selector=((1.0, 0.0, 0.0),),
    )
    audit = audit_boundary_stratum_state(
        holo,
        state,
        ("SHORT_ZERO__LONG_ZERO__C_ARBITRARY",),
        ((True, False, False),),
    )

    assert audit.stationary_point_verified
    assert audit.terminal_assignment_satisfies_public_relation
    assert audit.classification == "SATISFYING_INVARIANT_SECTION"
    assert audit.clause_audits[0].clause_violation == 0.0


def test_repeated_literal_lie_derivative_sums_occurrence_contributions() -> None:
    holo = ConstraintHolo.build(
        ("x",),
        (clause(Literal("x"), Literal("x"), Literal("x")),),
    )
    cap = memory_cap(holo)
    state = ReducedExactProductPhaseState(
        phase_cosine=(0.0,),
        phase_sine=(1.0,),
        short_memory=(1.0,),
        long_memory=(cap,),
        clause_selector=((1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),),
    )

    stationary_audit = audit_boundary_stratum_state(
        holo,
        state,
        ("SHORT_ONE__LONG_CAP__C_ARBITRARY",),
        ((True, True, True),),
    )
    moving_audit = audit_boundary_stratum_state(
        holo,
        state,
        ("SHORT_ONE__LONG_CAP__C_ARBITRARY",),
        ((True, True, True),),
        proposal_kind="moving_invariant_set",
    )

    assert stationary_audit.classification == "NOT_INVARIANT"
    assert not stationary_audit.stationary_point_verified
    assert moving_audit.pointwise_boundary_tangency_verified
    assert not moving_audit.moving_invariant_set_globally_verified
    assert moving_audit.classification == "INVARIANT_STABLE_OR_CENTER_UNRESOLVED"
    assert any(
        "global moving-set closure" in item
        for item in moving_audit.unresolved_conditions
    )
    assert isclose(
        moving_audit.clause_audits[0].clause_violation_derivative,
        -2.25 * cap,
        rel_tol=1.0e-12,
        abs_tol=1.0e-9,
    )


def test_mismatched_memory_stratum_is_not_invariant() -> None:
    holo = symmetric_stationary_holo()
    audit = audit_boundary_stratum_state(
        holo,
        symmetric_state(1.0),
        (
            "SHORT_ZERO__LONG_CAP__C_ARBITRARY",
            "SHORT_ZERO__LONG_CAP__C_ARBITRARY",
        ),
        ((True, True, True), (True, True, True)),
    )
    assert audit.classification == "NOT_INVARIANT"
    assert not audit.pointwise_boundary_tangency_verified


def test_audit_rejects_incomplete_selector_support() -> None:
    holo = ConstraintHolo.build(
        ("x", "y", "z"),
        (clause(Literal("x"), Literal("y"), Literal("z")),),
    )
    state = ReducedExactProductPhaseState(
        phase_cosine=(1.0, -1.0, -1.0),
        phase_sine=(0.0, 0.0, 0.0),
        short_memory=(0.0,),
        long_memory=(0.0,),
        clause_selector=((1.0, 0.0, 0.0),),
    )
    with pytest.raises(ConstraintHoloError, match="support"):
        audit_boundary_stratum_state(
            holo,
            state,
            ("SHORT_ZERO__LONG_ZERO__C_ARBITRARY",),
            ((True, True, True),),
        )


def test_audit_rejects_nan_and_unsupported_strata() -> None:
    holo = symmetric_stationary_holo()
    bad_state = ReducedExactProductPhaseState(
        phase_cosine=(nan, 0.0),
        phase_sine=(1.0, 1.0),
        short_memory=(1.0, 1.0),
        long_memory=(memory_cap(holo), memory_cap(holo)),
        clause_selector=(
            (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
            (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
        ),
    )
    with pytest.raises(ConstraintHoloError, match="NaN or infinity"):
        audit_boundary_stratum_state(
            holo,
            bad_state,
            (
                "SHORT_ONE__LONG_CAP__C_ARBITRARY",
                "SHORT_ONE__LONG_CAP__C_ARBITRARY",
            ),
            ((True, True, True), (True, True, True)),
        )

    with pytest.raises(ConstraintHoloError, match="unsupported public-seed"):
        audit_boundary_stratum_state(
            holo,
            symmetric_state(1.0),
            ("NOT_A_STRATUM", "SHORT_ONE__LONG_CAP__C_ARBITRARY"),
            ((True, True, True), (True, True, True)),
        )
