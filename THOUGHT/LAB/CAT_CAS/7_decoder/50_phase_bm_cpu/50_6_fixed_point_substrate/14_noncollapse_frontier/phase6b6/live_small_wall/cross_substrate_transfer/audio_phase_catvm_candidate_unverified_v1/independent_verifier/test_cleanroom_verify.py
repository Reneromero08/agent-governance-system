from pathlib import Path

import pytest

from cleanroom_verify import (
    PublicDag,
    SeriesParallelProgram,
    dag_analysis,
    enumerate_series_parallel,
    expected_threshold_partition,
    gf2_add,
    gf2_mul,
    gf2_variable,
    parse_aspr,
    parse_public_dag,
    quotient_partition,
    replay_pebble_actions,
    restoration_controls,
    reversible_schedule_analysis,
    run_all,
    symbolic_qanf_boundary,
)


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "source_snapshot"


def test_candidate_a_primary_and_reuse_boundaries() -> None:
    primary = enumerate_series_parallel(parse_aspr(SOURCE / "catvm_primary.aspr"))
    reuse = enumerate_series_parallel(parse_aspr(SOURCE / "catvm_reuse.aspr"))
    assert primary["f_coefficients"] == [0, 1, 1, 1]
    assert primary["z_coefficients"] == [0, 2, 1, 1]
    assert reuse["z_coefficients"] == [1, 2, 0, 2]


def test_candidate_a_restoration_controls() -> None:
    controls = restoration_controls(parse_aspr(SOURCE / "catvm_primary.aspr"))
    assert controls["nominal_restored"] is True
    assert controls["wrong_g_detected"] is True
    assert controls["missing_g_detected"] is True
    assert controls["reordered_inverse_detected"] is True


def test_candidate_a_mutated_constraint_changes_boundary() -> None:
    original = parse_aspr(SOURCE / "catvm_primary.aspr")
    mutated = SeriesParallelProgram(
        original.left, original.right, (0, 0, 1, 0)
    )
    assert (
        enumerate_series_parallel(original)["z_coefficients"]
        != enumerate_series_parallel(mutated)["z_coefficients"]
    )


def test_candidate_b_graph_and_reversible_schedule() -> None:
    dag = parse_public_dag(SOURCE / "general_multi_dag_affine_topology.txt")
    graph = dag_analysis(dag)
    schedule = reversible_schedule_analysis(dag)
    assert graph["node_count"] == 15
    assert graph["root"] == 815
    assert graph["shared_fanout"] == {"805": 4, "806": 3, "807": 3, "808": 2}
    assert schedule["minimum_reversible_capacity"] == 6
    assert schedule["capacity_action_tradeoff"]["5"] is None
    assert schedule["capacity_action_tradeoff"]["6"] == 40
    assert schedule["capacity_action_tradeoff"]["7"] == 28
    assert schedule["selected_witness"]["restored_empty"] is True
    assert all(schedule["mutation_controls_detected"].values())


def test_candidate_b_rejects_cycle() -> None:
    cyclic = PublicDag(
        root=2,
        nodes=(
            # Deliberately malformed two-node cycle.
            type(parse_public_dag(SOURCE / "general_multi_dag_affine_topology.txt").nodes[0])(
                1, "compose", (2, 2)
            ),
            type(parse_public_dag(SOURCE / "general_multi_dag_affine_topology.txt").nodes[0])(
                2, "compose", (1, 1)
            ),
        ),
    )
    with pytest.raises(ValueError, match="cycle"):
        dag_analysis(cyclic)


def test_candidate_b_replay_rejects_bad_node() -> None:
    with pytest.raises(ValueError, match="outside"):
        replay_pebble_actions([0], [1], 1)


def test_gf2_polynomial_cancels_duplicate_terms() -> None:
    a = gf2_variable("a")
    b = gf2_variable("b")
    ab = gf2_mul(a, b)
    assert gf2_add(ab, ab) == frozenset()


def test_candidate_c_symbolic_formula() -> None:
    assert symbolic_qanf_boundary(0, 1, 0, 1, 0, 1) == (1, 0, 0, 0, 1)
    assert symbolic_qanf_boundary(1, 1, 1, 1, 1, 1) == (1, 1, 1, 1, 1)


@pytest.mark.parametrize("depth", range(2, 10))
@pytest.mark.parametrize("horizon", range(1, 10))
def test_candidate_d_continuation_partition(depth: int, horizon: int) -> None:
    assert quotient_partition(depth, horizon) == expected_threshold_partition(
        depth, horizon
    )


def test_full_cleanroom_result() -> None:
    result = run_all(SOURCE)
    assert result["source_results_used_as_oracle"] is False
    assert result["candidate_c"]["public_programs"] == 64
    assert result["candidate_c"]["unique_boundaries"] == 16
    assert result["candidate_d"]["all_formula_cases_match"] is True
    assert result["candidate_d"]["deliberate_overmerge_detected"] is True
    assert result["candidate_d"]["deliberate_undermerge_detected"] is True
