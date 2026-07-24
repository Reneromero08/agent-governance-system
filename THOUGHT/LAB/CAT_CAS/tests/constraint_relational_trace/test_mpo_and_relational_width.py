from __future__ import annotations

from pathlib import Path
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

from constraint_relational_trace_v1.mpo_configuration_audit import (  # noqa: E402
    audit_control_symbol_mpo_projection,
)
from constraint_relational_trace_v1.relational_width_audit import (  # noqa: E402
    audit_residual_relation_width,
    equality_relation_holo,
)


def test_historical_control_symbol_mpo_omits_machine_configuration() -> None:
    audit = audit_control_symbol_mpo_projection(
        control_states=4,
        alphabet_symbols=2,
        bounded_tape_cells=8,
    )
    assert audit.reported_bond_dimension == 8
    assert audit.full_bounded_configuration_count == 4 * 8 * 2**8
    assert audit.omitted_head_positions == 8
    assert audit.omitted_tape_configurations == 2**8
    assert not audit.exact_configuration_injective
    assert audit.invariant_scope == "finite_control_symbol_transition_graph_only"


def test_grouped_equality_order_has_exponential_midcut_width() -> None:
    pair_count = 5
    holo = equality_relation_holo(pair_count)
    grouped = tuple(f"x{index}" for index in range(1, pair_count + 1)) + tuple(
        f"y{index}" for index in range(1, pair_count + 1)
    )
    audit = audit_residual_relation_width(holo, grouped)
    assert audit.cut_widths[pair_count] == 2**pair_count
    assert audit.maximum_width == 2**pair_count
    assert audit.maximum_cut == pair_count


def test_interleaved_equality_order_keeps_width_constant() -> None:
    pair_count = 5
    holo = equality_relation_holo(pair_count)
    interleaved = tuple(
        variable
        for index in range(1, pair_count + 1)
        for variable in (f"x{index}", f"y{index}")
    )
    audit = audit_residual_relation_width(holo, interleaved)
    assert audit.maximum_width <= 3
    assert audit.maximum_width < 2**pair_count


def test_relational_width_is_a_presentation_gauge_control_not_a_global_lower_bound() -> None:
    pair_count = 4
    holo = equality_relation_holo(pair_count)
    grouped = tuple(f"x{index}" for index in range(1, pair_count + 1)) + tuple(
        f"y{index}" for index in range(1, pair_count + 1)
    )
    interleaved = tuple(
        variable
        for index in range(1, pair_count + 1)
        for variable in (f"x{index}", f"y{index}")
    )
    grouped_audit = audit_residual_relation_width(holo, grouped)
    interleaved_audit = audit_residual_relation_width(holo, interleaved)
    assert grouped_audit.maximum_width == 16
    assert interleaved_audit.maximum_width <= 3
    assert grouped_audit.maximum_width > interleaved_audit.maximum_width
