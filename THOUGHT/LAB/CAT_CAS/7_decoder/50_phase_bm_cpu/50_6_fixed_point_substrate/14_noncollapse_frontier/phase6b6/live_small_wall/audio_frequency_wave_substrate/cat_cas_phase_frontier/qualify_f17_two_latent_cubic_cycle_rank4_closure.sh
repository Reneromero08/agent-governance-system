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
production_seal="$frontier_dir/F17_TWO_LATENT_CUBIC_CYCLE_RANK4_CLOSURE_RESULTS.json"
oracle_seal="$frontier_dir/F17_TWO_LATENT_CUBIC_CYCLE_RANK4_CLOSURE_ORACLE_RESULTS.json"
production_replay="$evidence_dir/production.json"
oracle_replay="$evidence_dir/oracle.json"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
export LC_ALL=C
export PYTHONPATH="$frontier_dir"

nice -n 10 "$python_bin" -X dev \
    "$frontier_dir/f17_two_latent_cubic_cycle_rank4_closure.py" \
    --output "$production_replay"
cmp "$production_seal" "$production_replay"

nice -n 10 "$python_bin" -X dev \
    "$frontier_dir/f17_two_latent_cubic_cycle_rank4_closure_oracle.py" \
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
assert tuple(production["source_scope"]["exact_branch_counts"]) == (2, 4, 8, 16, 32, 64)
assert tuple(production["source_scope"]["dual_field_structural_branch_counts"]) == (2, 3, 4, 5, 6, 7, 8)
assert production["source_scope"]["shared_latent_ports"] == 2
assert production["source_scope"]["boolean_degree"] == 3
assert production["source_scope"]["two_branch_junction_separator_rank"] == 4

transactions = production["exact_transactions"] + production["dual_field_structural_transactions"]
transaction_index = {
    (item["branch_count"], item["family"], item["algebra"]): item
    for item in transactions
}
assert len(transactions) == 20
for item in transactions:
    branches = item["branch_count"]
    assert item["factor_count"] == 2 * branches
    assert item["local_bit_count"] == 3 * branches
    assert item["total_logical_bits"] == 2 + 3 * branches
    assert item["resident_phase_factor_field_cells"] == 4 * branches
    assert item["resident_nontrivial_theta_field_cells"] == 2 * branches
    assert item["resident_shared_latent_port_field_cells"] == 4
    assert item["exact_two_branch_junction_separator_rank"] == 4
    assert item["maximum_named_port_update_field_cells"] == 16
    assert item["intermediate_projection_calls"] == 0
    assert item["final_projection_calls"] == 1
    assert not item["accepted_path_local_assignment_enumeration"]
    assert item["accepted_path_global_assignment_or_dense_tensor_cells"] == 0
    assert item["inverse_history_cells"] == 0
    assert not item["snapshot_reload_used"]
    assert item["response_released_after_restoration"]
    assert item["restored_exact_zero"]
    assert item["same_backing"]
    assert item["resident_carrier_restoration_class"] == "EXACT_ALGEBRAIC_RESTORATION"
    assert item["compiler_commitment_and_verification_buffer_restoration_class"] == "NO_RESTORATION_CLAIM"
    assert not item["intermediate_port_or_factor_payload_exposed_in_result"]
    assert item["one_way_factor_commitment_emitted"]

for item in production["rank_certificates"]:
    assert item["all_branch_kronecker_minors_nonzero"]
    assert item["each_branch_local_to_h_k_map_rank"] == 4
    assert item["two_branch_cycle_junction_separator_rank"] == 4
    assert item["walsh_transport_rank"] == 4
    assert item["rank_three_separator_rejected"]
    assert not item["minor_values_serialized"]
    assert not item["local_or_global_assignment_tensor_materialized"]

for item in production["compiled_classical_baselines"]:
    assert item["boundary_agreement"]
    assert item["compiled_row_boundary_agreement"]
    assert item["full_signature_exact_factor_field_cells"] == 2 * item["branch_count"]
    assert item["runtime_dynamic_port_field_cells"] == 4
    transaction = transaction_index[
        (item["branch_count"], item["family"], item["algebra"])
    ]
    assert (
        item["runtime_maximum_exact_port_payload_bits"]
        == transaction["maximum_resident_port_payload_bits"]
    )
    assert item["sealed_arbitrary_port_input_final_row_field_cells"] == 4
    assert item["sealed_fixed_transaction_field_cells"] == 1
    assert not item["final_state_serialized"]
    assert not item["phase_carrier_or_snapshot_used"]

for item in oracle["transaction_parity"]:
    assert item["all_core_fields_match"], item["mismatches"]
    assert item["port_restores_exact_seed"]
    assert item["factor_pairs_restore_exact_seed"]

for item in oracle["compiled_classical_baseline_checks"]:
    assert item["row_commitment_matches"]
    assert item["row_record_bytes_match"]
    assert item["runtime_port_payload_matches_transaction"]

for item in oracle["direct_two_branch_tensor_rank_checks"]:
    assert item["left_branch_map_rank"] == 4
    assert item["right_branch_map_rank"] == 4
    assert item["full_two_branch_boundary_tensor_rank"] == 4
    assert item["selected_anchor_one_four_by_four_minor_rank"] == 4
    assert item["direct_tensor_sum_matches_recurrence"]
    assert item["full_tensor_shape"] == [8, 8]

assert all(
    value
    for key, value in production["controls"].items()
    if key != "snapshot_command_available"
)
assert not production["controls"]["snapshot_command_available"]
for mutation in oracle["independent_mutation_checks"]:
    assert all(value for key, value in mutation.items() if key != "snapshot_command_available")
    assert not mutation["snapshot_command_available"]

reuse = production["reuse"]
assert reuse["fresh_restored_boundary_agreement"]
assert reuse["fresh_restored_resource_signature_agreement"]
assert reuse["same_actual_backing_across_unrelated_programs"]
assert reuse["package_local_restoration_count_after_two_transactions"] == 2
assert not reuse["baseline_reload_used"]

assert production["matched_baseline"]["strongest_full_signature"] == "TWO_B_EXACT_PUBLIC_PHASE_FACTORS"
assert production["matched_baseline"]["strongest_final_boundary_runtime"] == "IDENTICAL_FOUR_DYNAMIC_SCALAR_RECURRENCE"
assert not production["matched_baseline"]["phase_advantage_over_matched_classical"]

ceiling = production["claim_ceiling"]
for key in (
    "arbitrary_cubic_hypergraph_closure",
    "arbitrary_port_arity",
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

if rg -n 'import f17_two_latent_cubic_cycle_rank4_closure' \
    "$frontier_dir/f17_two_latent_cubic_cycle_rank4_closure_oracle.py"; then
    echo "oracle imports production module" >&2
    exit 1
fi

echo QUALIFIED_F17_TWO_LATENT_CUBIC_CYCLE_RANK4_CLOSURE_STRICT_SCOPE
