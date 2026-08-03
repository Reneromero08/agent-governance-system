#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 EVIDENCE_DIR" >&2
    exit 2
fi

evidence_dir=$1
mkdir -p "$evidence_dir"
frontier_dir=$(cd "$(dirname "$0")" && pwd)
repo_root=$(git -C "$frontier_dir" rev-parse --show-toplevel)
python_bin="$repo_root/.venv/bin/python"
production_seal="$frontier_dir/F17_EXCHANGE_SYMMETRIC_PHASE_MODULE_IRREDUCIBILITY_RESULTS.json"
oracle_seal="$frontier_dir/F17_EXCHANGE_SYMMETRIC_PHASE_MODULE_IRREDUCIBILITY_ORACLE_RESULTS.json"
production_replay="$evidence_dir/production.json"
oracle_replay="$evidence_dir/oracle.json"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
export LC_ALL=C
export PYTHONPATH="$frontier_dir"

nice -n 10 "$python_bin" -X dev \
    "$frontier_dir/f17_exchange_symmetric_phase_module_irreducibility.py" \
    --output "$production_replay"
cmp "$production_seal" "$production_replay"

nice -n 10 "$python_bin" -X dev \
    "$frontier_dir/f17_exchange_symmetric_phase_module_irreducibility_oracle.py" \
    --production "$production_replay" \
    --output "$oracle_replay"
cmp "$oracle_seal" "$oracle_replay"

"$python_bin" - "$production_replay" "$oracle_replay" <<'PY'
import json
import pathlib
import sys

production = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
oracle = json.loads(pathlib.Path(sys.argv[2]).read_text(encoding="utf-8"))

assert production["classification"] == "SOURCE_AUDITED_PACKAGE_LOCAL"
assert oracle["classification"] == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE"
assert oracle["verification_level"] == "INDEPENDENT_ORACLE_REEXECUTION"

expected_dimensions = {1: 17, 2: 153, 3: 969, 4: 4845}
for item in production["irreducibility_certificates"]:
    k = item["k"]
    assert item["occupation_dimension"] == expected_dimensions[k]
    assert item["p1_through_pk_signature_count"] == expected_dimensions[k]
    assert item["power_sum_signature_injective"]
    assert item["newton_denominators_invertible_mod17"]
    assert item["lifted_support_components"] == 1
    assert item["explicit_predecessor_edges_to_zero_mode"] == expected_dimensions[k] - 1
    assert item["minimum_exact_linear_quotient_dimension"] == expected_dimensions[k]
    assert item["verification_named_logical_integer_cells_conservative_sum"] > 0
    assert not item["verification_transition_edge_list_materialized"]
    assert not item["verification_predecessor_list_materialized"]
    if k >= 2:
        assert item["p1_through_p_k_minus_1_signature_count"] < expected_dimensions[k]

for item in production["exact_transactions"]:
    assert item["restored_exact_zero"]
    assert item["same_backing"]
    assert item["intermediate_projection_calls"] == 0
    assert item["final_projection_calls"] == 1
    assert item["inverse_history_cells"] == 0
    assert not item["inverse_history_retained"]
    assert not item["snapshot_reload_used"]
    assert item["public_occupation_topology_integer_cells"] == 18 * expected_dimensions[item["k"]]
    assert item["public_program_integer_cells"] > 0
    assert item["public_program_json_bytes"] > 0
    assert item["public_grid_edge_coordinate_integer_cells"] == 96
    assert item["public_grid_vertex_coordinate_integer_cells"] == 32
    assert item["determinant_matrix_dimension"] == 8
    assert item["maximum_named_transaction_transient_field_cells"] == 133
    assert item["maximum_named_transaction_transient_integer_cells"] == 48
    assert item["maximum_named_transaction_transient_payload_bits_upper_bound"] >= 0
    assert item["final_boundary_json_bytes"] > 0
    assert item["response_released_after_restoration"]
    assert item["resident_carrier_restoration_class"] == "EXACT_ALGEBRAIC_RESTORATION"
    assert item["transient_buffer_restoration_class"] == "NO_RESTORATION_CLAIM"
    assert not item["accepted_path_labelled_tensor_materialized"]
    assert not item["accepted_path_dense_occupation_operator_materialized"]
    assert not item["accepted_path_character_table_materialized"]
    assert not item["intermediate_orbit_state_serialized"]

for item in production["dual_field_structural_transactions"]:
    assert item["restored_exact_zero"]
    assert item["same_backing"]

reuse = production["reuse"]
assert reuse["fresh_restored_boundary_agreement"]
assert reuse["fresh_restored_resource_signature_agreement"]
assert reuse["same_actual_backing_across_unrelated_programs"]
assert reuse["generation_after_two_transactions"] == 2
assert not reuse["baseline_reload_used"]

controls = production["controls"]
for key in (
    "missing_inverse_detected",
    "wrong_program_ownership_rejected",
    "premature_projection_rejected",
    "reordered_inverse_detected",
    "null_carrier_rejected",
    "missing_p2_character_rejected",
    "power_sum_character_mutation_changes_boundary",
    "missing_reverse_mode_orientation_rejected",
    "mode_shear_mutation_changes_boundary",
    "p1_only_overmerges_k2",
    "disconnected_mode_graph_preserves_particle_count_components",
    "nonprimitive_root_rejected",
    "k17_newton_applicability_rejected",
):
    assert controls[key], key

assert not controls["accepted_path_character_table_materialized"]
assert not controls["accepted_path_dense_operator_materialized"]
assert not controls["accepted_path_labelled_tensor_materialized"]
assert not controls["intermediate_orbit_state_serialized"]
assert not controls["catvm_boundary_claimed"]

for item in oracle["exact_transaction_parity"]:
    assert item["boundary_agreement"]
    assert item["independent_forward_inverse_restored"]
for item in oracle["dual_field_structural_transaction_parity"]:
    assert item["boundary_agreement"]
    assert item["independent_forward_inverse_restored"]
for item in oracle["certificate_parity"]:
    assert all(
        item[key]
        for key in (
            "dimension_agreement",
            "signature_count_agreement",
            "predecessor_fingerprint_agreement",
            "full_matrix_algebra_certificate",
            "minimum_quotient_dimension_agreement",
        )
    )
for item in oracle["phase_character_orthogonality"]:
    assert item["all_17_character_sums_exact"]

ceiling = production["claim_ceiling"]
assert ceiling["uniform_linear_phase_module_quotient_only"]
assert not ceiling["nonlinear_or_program_restricted_quotient_rejected"]
assert not ceiling["arbitrary_relation_algebra_rejected"]
assert not ceiling["catvm_custody_established"]
assert not ceiling["distinct_phase_resource_established"]
assert not ceiling["computational_advantage_established"]
assert not ceiling["small_wall_crossing_established"]
assert not ceiling["physical_waveform_execution_established"]
assert not ceiling["physical_bits_replaced_with_pi"]
assert not ceiling["unbounded_catalytic_computation_established"]
PY

echo QUALIFIED_F17_EXCHANGE_SYMMETRIC_PHASE_MODULE_IRREDUCIBILITY_STRICT_SCOPE
