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
production_seal="$frontier_dir/F17_GRAY_CODE_BOND2_PHASE_FACTOR_CHART_CLOSURE_RESULTS.json"
oracle_seal="$frontier_dir/F17_GRAY_CODE_BOND2_PHASE_FACTOR_CHART_CLOSURE_ORACLE_RESULTS.json"
production_replay="$evidence_dir/production.json"
oracle_replay="$evidence_dir/oracle.json"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
export LC_ALL=C
export PYTHONPATH="$frontier_dir"

nice -n 10 "$python_bin" -X dev \
    "$frontier_dir/f17_gray_code_bond2_phase_factor_chart_closure.py" \
    --output "$production_replay"
cmp "$production_seal" "$production_replay"

nice -n 10 "$python_bin" -X dev \
    "$frontier_dir/f17_gray_code_bond2_phase_factor_chart_closure_oracle.py" \
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

transactions = production["exact_transactions"] + production["dual_field_structural_transactions"]
assert len(transactions) == 20
for item in transactions:
    depth = item["depth"]
    assert item["resident_phase_factor_field_cells"] == 2 * depth
    assert item["resident_nontrivial_eta_field_cells"] == depth
    assert item["fixed_wiring_field_cells"] == 0
    assert item["exact_maximum_mps_bond_dimension"] == (1 if depth == 1 else 2)
    assert item["maximum_local_coupling_named_field_cells"] == 4
    assert item["projection_dynamic_field_cells"] == 4
    assert item["intermediate_projection_calls"] == 0
    assert item["final_projection_calls"] == 1
    assert not item["accepted_path_explicit_component_enumeration"]
    assert item["accepted_path_component_weight_cells"] == 0
    assert item["accepted_path_catalecticant_cells"] == 0
    assert item["accepted_path_dense_operator_cells"] == 0
    assert item["inverse_history_cells"] == 0
    assert not item["snapshot_reload_used"]
    assert item["response_released_after_restoration"]
    assert item["restored_exact_zero"]
    assert item["same_backing"]
    assert item["resident_carrier_restoration_class"] == "EXACT_ALGEBRAIC_RESTORATION"
    assert item["coupling_projection_and_compiler_buffer_restoration_class"] == "NO_RESTORATION_CLAIM"
    assert not item["intermediate_factor_payload_exposed_in_result"]
    assert item["one_way_factor_commitment_emitted"]
    assert "component_weights" not in item

for item in production["fixed_bond_certificates"]:
    assert item["all_internal_edge_determinants_nonzero"]
    assert item["exact_maximum_weight_tensor_mps_bond_dimension"] == (
        1 if item["depth"] == 1 else 2
    )
    assert item["bond_one_rejected_for_depth_at_least_two"]
    assert not item["determinant_values_serialized"]
    assert not item["component_weights_materialized"]

for item in production["verification_only_expanded_weight_checks"]:
    assert item["all_gray_factor_weights_agree"]
    assert not item["accepted_path"]

for item in production["compiled_classical_baselines"]:
    assert item["boundary_agreement"]
    assert item["compiled_transfer_boundary_agreement"]
    assert item["full_weight_signature_exact_factor_field_cells"] == item["depth"]
    assert item["final_boundary_dynamic_field_cells"] == 2
    assert item["sealed_word_compiled_transfer_nonzero_field_cells"] == 3

parity_keys = (
    "program_fingerprint_agreement",
    "conceptual_component_count_agreement",
    "degree_agreement",
    "factor_commitment_agreement",
    "commitment_record_bytes_agreement",
    "closed_form_boundary_agreement",
    "boundary_payload_bits_agreement",
    "resident_payload_bits_agreement",
    "resident_phase_factor_cells_agreement",
    "resident_nontrivial_eta_cells_agreement",
    "fixed_bond_agreement",
    "independent_local_inverse_restores_seeded_sites",
    "public_program_json_bytes_agreement",
    "restored_exact_zero_reported",
    "same_backing_reported",
    "no_component_weight_payload_in_transaction",
)
for item in oracle["transaction_parity"]:
    assert all(item[key] for key in parity_keys)

baseline_keys = (
    "boundary_agreement",
    "factor_commitment_agreement",
    "factor_record_bytes_agreement",
    "compiled_transfer_commitment_agreement",
    "compiled_transfer_record_bytes_agreement",
    "full_signature_factor_cells_agreement",
    "two_dynamic_moment_cells_agreement",
    "three_compiled_transfer_cells_agreement",
)
for item in oracle["compiled_classical_baseline_checks"]:
    assert all(item[key] for key in baseline_keys)

for item in oracle["direct_expanded_mps_rank_checks"]:
    assert item["recursive_gray_agreement"]
    assert item["recursive_mps_agreement"]
    assert item["all_internal_cut_ranks_exactly_two"] == (item["depth"] >= 2)
    assert item["verification_only_component_enumeration"]

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
    "M_EXACT_PHASE_FACTORS_ON_PUBLIC_GRAY_CHAIN"
)
assert production["matched_baseline"]["strongest_final_boundary_runtime"] == (
    "TWO_DYNAMIC_MOMENT_SCALARS"
)
assert not production["matched_baseline"]["phase_advantage_over_matched_classical"]

ceiling = production["claim_ceiling"]
for key in (
    "general_coherent_polynomial_compaction",
    "arbitrary_boundary_compaction",
    "conventional_clifford_or_stabilizer_classification",
    "general_gaussian_closure_or_no_go",
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

if rg -n 'import f17_gray_code_bond2_phase_factor_chart_closure' \
    "$frontier_dir/f17_gray_code_bond2_phase_factor_chart_closure_oracle.py"; then
    echo "oracle imports production module" >&2
    exit 1
fi

echo QUALIFIED_F17_GRAY_CODE_BOND2_PHASE_FACTOR_CHART_CLOSURE_STRICT_SCOPE
