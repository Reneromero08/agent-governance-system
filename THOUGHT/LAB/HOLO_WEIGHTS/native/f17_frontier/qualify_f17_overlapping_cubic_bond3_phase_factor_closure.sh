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
production_seal="$frontier_dir/F17_OVERLAPPING_CUBIC_BOND3_PHASE_FACTOR_CLOSURE_RESULTS.json"
oracle_seal="$frontier_dir/F17_OVERLAPPING_CUBIC_BOND3_PHASE_FACTOR_CLOSURE_ORACLE_RESULTS.json"
production_replay="$evidence_dir/production.json"
oracle_replay="$evidence_dir/oracle.json"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
export LC_ALL=C
export PYTHONPATH="$frontier_dir"

nice -n 10 "$python_bin" -X dev \
    "$frontier_dir/f17_overlapping_cubic_bond3_phase_factor_closure.py" \
    --output "$production_replay"
cmp "$production_seal" "$production_replay"

nice -n 10 "$python_bin" -X dev \
    "$frontier_dir/f17_overlapping_cubic_bond3_phase_factor_closure_oracle.py" \
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
assert tuple(production["source_scope"]["exact_depths"]) == (1, 2, 4, 8, 16, 32, 64, 128)
assert production["source_scope"]["nonaffine_boolean_degree"] == 3

transactions = production["exact_transactions"] + production["dual_field_structural_transactions"]
assert len(transactions) == 24
for item in transactions:
    depth = item["depth"]
    assert item["physical_bits"] == depth + 2
    assert item["resident_phase_factor_field_cells"] == 2 * depth
    assert item["resident_nontrivial_theta_field_cells"] == depth
    assert item["exact_maximum_mps_bond_dimension"] == (2 if depth == 1 else 3)
    assert item["maximum_local_coupling_named_field_cells"] == 4
    assert item["projection_dynamic_field_cells"] == 6
    assert item["intermediate_projection_calls"] == 0
    assert item["final_projection_calls"] == 1
    assert not item["accepted_path_assignment_enumeration"]
    assert item["accepted_path_component_weight_cells"] == 0
    assert item["accepted_path_dense_transfer_cells"] == 0
    assert item["inverse_history_cells"] == 0
    assert not item["snapshot_reload_used"]
    assert item["response_released_after_restoration"]
    assert item["restored_exact_zero"]
    assert item["same_backing"]
    assert item["resident_carrier_restoration_class"] == "EXACT_ALGEBRAIC_RESTORATION"
    assert item["projection_compiler_and_commitment_buffer_restoration_class"] == "NO_RESTORATION_CLAIM"
    assert not item["intermediate_factor_payload_exposed_in_result"]
    assert item["one_way_factor_commitment_emitted"]

for item in production["rank_certificates"]:
    assert item["all_declared_rank3_minors_nonzero"]
    assert item["exact_maximum_weight_tensor_mps_bond_dimension"] == (
        2 if item["depth"] == 1 else 3
    )
    assert item["bond_two_rejected_for_depth_at_least_two"]
    assert not item["minor_values_serialized"]
    assert not item["assignment_tensor_materialized"]

for item in production["compiled_classical_baselines"]:
    assert item["boundary_agreement"]
    assert item["compiled_boundary_agreement"]
    assert item["full_weight_signature_exact_factor_field_cells"] == item["depth"]
    assert item["final_boundary_dynamic_field_cells"] == 3
    assert item["sealed_word_three_state_chart_input_final_row_field_cells"] == 3
    assert item["sealed_fixed_initial_and_final_boundary_field_cells"] == 1

parity_keys = (
    "program_fingerprint_agreement",
    "physical_bits_agreement",
    "factor_commitment_agreement",
    "commitment_record_bytes_agreement",
    "independent_four_state_boundary_agreement",
    "boundary_payload_bits_agreement",
    "resident_payload_bits_agreement",
    "resident_factor_cells_agreement",
    "resident_nontrivial_theta_cells_agreement",
    "bond_agreement",
    "independent_local_inverse_restores_seeded_sites",
    "public_program_json_bytes_agreement",
    "restored_exact_zero_reported",
    "same_backing_reported",
    "no_assignment_or_weight_tensor_payload_in_transaction",
)
for item in oracle["transaction_parity"]:
    assert all(item[key] for key in parity_keys)

baseline_keys = (
    "boundary_agreement",
    "factor_commitment_agreement",
    "factor_record_bytes_agreement",
    "compiled_final_row_commitment_agreement",
    "compiled_final_row_record_bytes_agreement",
    "full_signature_factor_cells_agreement",
    "three_dynamic_cells_agreement",
    "three_compiled_chart_row_cells_agreement",
)
for item in oracle["compiled_classical_baseline_checks"]:
    assert all(item[key] for key in baseline_keys)

for item in oracle["direct_assignment_partition_checks"]:
    assert item["direct_partition_agrees_with_four_state_recurrence"]
    assert not item["accepted_path"]

for item in oracle["direct_local_rank_checks"]:
    assert item["exact_maximum_bond_dimension"] == (2 if item["depth"] == 1 else 3)
    assert item["all_interior_local_cross_ranks_three"]
    assert item["verification_only"]

for item in oracle["direct_full_tensor_rank_checks"]:
    assert item["exact_maximum_bond_dimension"] == (2 if item["depth"] == 1 else 3)
    assert item["verification_only_assignment_tensor_materialized"]
    assert not item["accepted_path"]

assert all(
    value
    for key, value in production["controls"].items()
    if key != "snapshot_command_available"
)
assert not production["controls"]["snapshot_command_available"]
assert all(
    value
    for key, value in oracle["independent_mutation_checks"].items()
    if key != "snapshot_command_available"
)
assert not oracle["independent_mutation_checks"]["snapshot_command_available"]

reuse = production["reuse"]
assert reuse["fresh_restored_boundary_agreement"]
assert reuse["fresh_restored_resource_signature_agreement"]
assert reuse["same_actual_backing_across_unrelated_programs"]
assert reuse["package_local_restoration_count_after_two_transactions"] == 2
assert not reuse["baseline_reload_used"]

assert production["matched_baseline"]["strongest_full_weight_signature"] == (
    "M_EXACT_PUBLIC_THETA_FACTORS"
)
assert production["matched_baseline"]["strongest_final_boundary_runtime"] == (
    "THREE_DYNAMIC_SCALARS"
)
assert not production["matched_baseline"]["phase_advantage_over_matched_classical"]

ceiling = production["claim_ceiling"]
for key in (
    "arbitrary_cubic_hypergraph_closure",
    "arbitrary_boundary_compaction",
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

if rg -n 'import f17_overlapping_cubic_bond3_phase_factor_closure' \
    "$frontier_dir/f17_overlapping_cubic_bond3_phase_factor_closure_oracle.py"; then
    echo "oracle imports production module" >&2
    exit 1
fi

echo QUALIFIED_F17_OVERLAPPING_CUBIC_BOND3_PHASE_FACTOR_CLOSURE_STRICT_SCOPE
