from __future__ import annotations

from pathlib import Path
import sys

import pytest

CAT_CAS_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_PARENT = (
    CAT_CAS_ROOT
    / "7_decoder"
    / "50_phase_bm_cpu"
    / "50_6_fixed_point_substrate"
    / "14_noncollapse_frontier"
)
sys.path.insert(0, str(PACKAGE_PARENT))

from constraint_relational_trace_v1.constraint_holo import (  # noqa: E402
    ConstraintHolo,
    ConstraintHoloError,
)


def test_dimacs_multi_digit_variables_are_normalized() -> None:
    holo = ConstraintHolo.from_dimacs(
        """
        p cnf 10 1
        10 -2 1 0
        """
    )
    assert len(holo.variables) == 10
    assert set(holo.variables) == {f"x{index}" for index in range(1, 11)}
    assert holo.clauses[0].variables == ("x1", "x10", "x2")


def test_dimacs_clause_count_must_match_declaration() -> None:
    with pytest.raises(ConstraintHoloError, match="clause count"):
        ConstraintHolo.from_dimacs(
            """
            p cnf 2 2
            1 1 1 0
            """
        )


def test_dimacs_rejects_clause_data_before_problem_line() -> None:
    with pytest.raises(ConstraintHoloError, match="before the problem line"):
        ConstraintHolo.from_dimacs(
            """
            1 1 1 0
            p cnf 1 1
            """
        )


def test_dimacs_rejects_second_problem_line_after_clause_data() -> None:
    with pytest.raises(ConstraintHoloError, match="after clause data"):
        ConstraintHolo.from_dimacs(
            """
            p cnf 1 1
            1 1 1 0
            p cnf 1 1
            """
        )


def test_dimacs_wraps_malformed_problem_counts() -> None:
    with pytest.raises(ConstraintHoloError, match="malformed DIMACS problem counts"):
        ConstraintHolo.from_dimacs("p cnf two 1")


def test_dimacs_wraps_malformed_literal_token() -> None:
    with pytest.raises(ConstraintHoloError, match="malformed DIMACS literal"):
        ConstraintHolo.from_dimacs(
            """
            p cnf 1 1
            1 nope 1 0
            """
        )
