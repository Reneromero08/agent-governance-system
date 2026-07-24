from __future__ import annotations

from pathlib import Path
import math
import sys

CAT_CAS_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_PARENT = (
    CAT_CAS_ROOT
    / "7_decoder"
    / "50_phase_bm_cpu"
    / "50_6_fixed_point_substrate"
    / "14_noncollapse_frontier"
)
sys.path.insert(0, str(PACKAGE_PARENT))

from constraint_relational_trace_v1.data_processing_amplifier_audit import (  # noqa: E402
    audit_phase_oracle_data_processing,
)


def test_unique_phase_mark_has_exponentially_small_one_shot_distance() -> None:
    audit = audit_phase_oracle_data_processing(10, 1)
    expected_overlap = 1 - 2 / 1024
    expected_distance = math.sqrt(1 - expected_overlap**2)
    assert audit.oracle_state_overlap == expected_overlap
    assert math.isclose(audit.oracle_trace_distance, expected_distance)
    assert audit.oracle_trace_distance < 0.1
    assert audit.deterministic_cptp_output_distance_upper_bound == audit.oracle_trace_distance
    assert not audit.constant_one_shot_separation_possible_under_cptp
    assert audit.postselection_or_nonstandard_dynamics_required


def test_many_marked_states_can_already_have_constant_separation() -> None:
    audit = audit_phase_oracle_data_processing(8, 64)
    assert audit.oracle_state_overlap == 0.5
    assert audit.oracle_trace_distance > 0.5
    assert audit.constant_one_shot_separation_possible_under_cptp
    assert not audit.postselection_or_nonstandard_dynamics_required


def test_unsat_and_unmarked_reference_are_identical() -> None:
    audit = audit_phase_oracle_data_processing(8, 0)
    assert audit.oracle_state_overlap == 1.0
    assert audit.oracle_trace_distance == 0.0
    assert audit.deterministic_cptp_output_distance_upper_bound == 0.0
