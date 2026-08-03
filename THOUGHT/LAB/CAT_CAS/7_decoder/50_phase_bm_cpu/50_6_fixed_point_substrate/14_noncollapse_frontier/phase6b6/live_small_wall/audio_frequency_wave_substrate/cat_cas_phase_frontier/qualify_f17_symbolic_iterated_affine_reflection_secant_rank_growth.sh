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
production_seal="$frontier_dir/F17_SYMBOLIC_ITERATED_AFFINE_REFLECTION_SECANT_RANK_GROWTH_RESULTS.json"
oracle_seal="$frontier_dir/F17_SYMBOLIC_ITERATED_AFFINE_REFLECTION_SECANT_RANK_GROWTH_ORACLE_RESULTS.json"
production_replay="$evidence_dir/production.json"
oracle_replay="$evidence_dir/oracle.json"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
export LC_ALL=C
export PYTHONPATH="$frontier_dir"

nice -n 10 "$python_bin" -X dev \
    "$frontier_dir/f17_symbolic_iterated_affine_reflection_secant_rank_growth.py" \
    --output "$production_replay"
cmp "$production_seal" "$production_replay"

nice -n 10 "$python_bin" -X dev \
    "$frontier_dir/f17_symbolic_iterated_affine_reflection_secant_rank_growth_oracle.py" \
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
assert production["verification_level"] == "PACKAGE_SELF_REVIEW"
assert oracle["classification"] == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE"
assert oracle["verification_level"] == "INDEPENDENT_ORACLE_REEXECUTION"
assert tuple(production["source_scope"]["executed_m"]) == (1, 2, 3, 4, 5, 6)

transactions = production["exact_transactions"] + production["dual_field_structural_transactions"]
assert len(transactions) == 18
for item in transactions:
    rank = 1 << item["m"]
    assert item["k"] == 2 * rank - 2
    assert item["target_rank"] == rank
    assert item["active_phase_field_cells"] == 18 * rank
    assert item["maximum_active_rank"] == rank
    assert item["maximum_coupling_transient_components"] == 2 * rank
    assert item["maximum_coupling_transient_field_cells"] == 36 * rank
    assert item["restored_exact_zero"]
    assert item["same_backing"]
    assert item["intermediate_projection_calls"] == 0
    assert item["final_projection_calls"] == 1
    assert item["inverse_history_cells"] == 0
    assert not item["inverse_history_retained"]
    assert not item["snapshot_reload_used"]
    assert item["response_released_after_restoration"]
    assert item["resident_carrier_restoration_class"] == "EXACT_ALGEBRAIC_RESTORATION"
    assert item["transient_buffer_restoration_class"] == "NO_RESTORATION_CLAIM"
    assert not item["accepted_path_catalecticant_materialized"]
    assert not item["accepted_path_occupation_vector_materialized"]
    assert item["accepted_path_explicit_coherent_component_enumeration"]
    assert item["accepted_path_resident_coherent_components"] == rank
    assert not item["accepted_path_separate_truth_table_or_assignment_buffer_materialized"]
    assert not item["accepted_path_dense_operator_materialized"]
    assert not item["intermediate_components_serialized_to_controller"]
    assert "components" not in item

certificates = production["symbolic_rank_certificates"]
assert len(certificates) == 18
q_signature = transactions[0]["algebra"]
for item in certificates:
    assert item["rank"] == 1 << item["m"]
    assert item["catalecticant_size"] == 1 << item["m"]
    assert item["declared_degree_meets_threshold"]
    assert item["public_point_set_is_integer_range_zero_to_rank_minus_one"]
    assert item["public_points_distinct_in_declared_algebra"]
    assert item["all_component_weights_nonzero"]
    assert item["factor_nonzero"]
    assert item["exact_normalized_divided_power_secant_rank"] == 1 << item["m"]
    assert item["ordinary_symmetric_waring_rank_interpretation"] == (
        item["algebra"] == q_signature
    )
    assert item["verification_carrier_restored_exact_zero"]
    assert not item["accepted_transaction_materializes_catalecticant"]
    assert not item["determinant_value_serialized"]
    assert not item["intermediate_components_serialized"]

reuse = production["reuse"]
assert reuse["fresh_restored_boundary_agreement"]
assert reuse["fresh_restored_resource_signature_agreement"]
assert reuse["same_actual_backing_across_unrelated_programs"]
assert reuse["generation_after_two_transactions"] == 2
assert not reuse["baseline_reload_used"]

for key, value in production["controls"].items():
    if key == "snapshot_command_available":
        assert not value
    else:
        assert value, key

parity_keys = (
    "program_fingerprint_agreement",
    "algebra_signature_agreement",
    "forward_commitment_agreement",
    "boundary_agreement",
    "boundary_payload_bits_agreement",
    "maximum_commitment_record_bound_honest",
    "independent_forward_inverse_restored",
    "target_rank_agreement",
    "active_field_cells_agreement",
    "inverse_transient_field_cells_agreement",
)
for item in oracle["transaction_parity"]:
    assert all(item[key] for key in parity_keys)
    certificate = item["rank_certificate"]
    assert certificate["public_point_set_agreement"]
    assert certificate["all_weights_nonzero"]
    assert certificate["independent_factorized_vandermonde_nonzero"]
    assert certificate["exact_normalized_divided_power_secant_rank"] == 1 << item["m"]
    if certificate["direct_hankel_determinant_checked"]:
        assert certificate["direct_hankel_determinant_nonzero"]

for item in oracle["direct_full_binary_moment_cases"]:
    assert item["full_binary_moment_field_cells"] == item["k"] + 1
    assert item["all_forward_component_moment_agreements"]
    assert item["final_boundary_agreement"]
    assert item["direct_moments_restored_to_seed"]
    assert item["component_chart_restored_to_seed"]

for item in oracle["compiled_two_moment_baseline_checks"]:
    assert item["boundary_agreement"]
    assert item["moment_commitment_agreement"]
    assert item["record_bytes_agreement"]
    assert item["two_dynamic_moment_cells_agreement"]
    assert item["one_sealed_boundary_cell_agreement"]
    assert item["full_state_triangular_moment_cells"] == item["k"] + 1

for item in production["compiled_atomic_weight_classical_baselines"]:
    rank = 1 << item["m"]
    assert item["resident_atomic_weight_field_cells"] == rank
    assert item["public_support_field_cells_retained"] == 0
    assert item["maximum_named_field_cells_including_update_buffer"] == 3 * rank // 2
    assert item["explicit_weight_enumeration"]
    assert item["boundary_agreement"]

for item in oracle["compiled_atomic_weight_baseline_checks"]:
    assert item["boundary_agreement"]
    assert item["atomic_weight_commitment_agreement"]
    assert item["record_bytes_agreement"]
    assert item["resident_weight_cells_agreement"]
    assert item["maximum_named_field_cells_agreement"]
    assert item["public_support_rematerialized_without_retained_cells"]
    assert item["component_weight_agreement"]

assert all(oracle["independent_mutation_checks"].values())
assert production["matched_baseline"]["strongest_descriptor_runtime"] == (
    "TWO_DYNAMIC_MOMENT_SCALARS"
)
assert production["matched_baseline"]["strongest_compact_full_state"] == (
    "TWO_TO_THE_M_ATOMIC_WEIGHTS_ON_PUBLIC_SUPPORT"
)
assert production["matched_baseline"]["independent_dense_moment_full_state"] == (
    "TWO_TIMES_TWO_TO_THE_M_MINUS_ONE_TRIANGULAR_MOMENTS"
)
assert not production["matched_baseline"]["phase_advantage_over_matched_classical"]
assert not oracle["resource_law"]["exact_payload_height_tuples_independently_reexecuted"]
assert not oracle["resource_law"]["full_exact_bit_complexity_established"]

ceiling = production["claim_ceiling"]
for key in (
    "arbitrary_interleaved_coupling_rank_law",
    "fixed_rank_closure",
    "general_gaussian_closure",
    "catvm_custody",
    "distinct_phase_resource",
    "computational_advantage",
    "small_wall_crossing",
    "physical_execution",
    "physical_bits_replaced_with_pi",
    "unbounded_catalytic_computation",
):
    assert not ceiling[key], key
PY

if rg -n 'import f17_symbolic_iterated_affine_reflection_secant_rank_growth' \
    "$frontier_dir/f17_symbolic_iterated_affine_reflection_secant_rank_growth_oracle.py"; then
    echo "oracle imports production module" >&2
    exit 1
fi

echo QUALIFIED_F17_SYMBOLIC_ITERATED_AFFINE_REFLECTION_SECANT_RANK_GROWTH_STRICT_SCOPE
