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
production_seal="$frontier_dir/F17_COHERENT_RANK2_SECANT_PHASE_COUPLING_CLOSURE_RESULTS.json"
oracle_seal="$frontier_dir/F17_COHERENT_RANK2_SECANT_PHASE_COUPLING_CLOSURE_ORACLE_RESULTS.json"
production_replay="$evidence_dir/production.json"
oracle_replay="$evidence_dir/oracle.json"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
export LC_ALL=C
export PYTHONPATH="$frontier_dir"

nice -n 10 "$python_bin" -X dev \
    "$frontier_dir/f17_coherent_rank2_secant_phase_coupling_closure.py" \
    --output "$production_replay"
cmp "$production_seal" "$production_replay"

nice -n 10 "$python_bin" -X dev \
    "$frontier_dir/f17_coherent_rank2_secant_phase_coupling_closure_oracle.py" \
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
assert tuple(production["source_scope"]["declared_degrees"]) == (4, 8, 16, 32, 64, 128)
assert tuple(item["k"] for item in production["exact_transactions"]) == (4, 8, 16, 32)

for item in (
    production["exact_transactions"]
    + production["dual_field_structural_transactions"]
):
    assert item["resident_phase_field_cells"] == 36
    assert item["resident_chart_rank"] == 2
    assert item["maximum_active_rank"] == 2
    assert item["maximum_inverse_coupling_transient_components"] == 4
    assert item["inverse_coupling_transient_field_cells"] == 72
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
    assert not item["accepted_path_occupation_vector_materialized"]
    assert not item["accepted_path_occupation_topology_materialized"]
    assert not item["accepted_path_matching_or_assignment_expansion_materialized"]
    assert not item["accepted_path_dense_operator_materialized"]
    assert not item["intermediate_secant_components_serialized_to_controller"]
    assert "components" not in item
    assert len(item["forward_secant_commitment"]) == 64

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
    "independent_rank_two_certificate_nonzero",
    "independent_forward_inverse_restored",
    "resident_field_cells_agreement",
    "inverse_transient_field_cells_agreement",
    "implicit_dimension_agreement",
)
for item in oracle["transaction_parity"]:
    assert all(item[key] for key in parity_keys)

for item in oracle["full_occupation_oracle_cases"]:
    assert item["occupation_dimension"] == 4845
    assert item["initial_coupling_full_occupation_agreement"]
    assert item["forward_full_occupation_secant_agreement"]
    assert item["forward_boundary_agreement"]
    assert item["dense_forward_inverse_restored"]
    assert item["chart_forward_inverse_restored"]
    assert item["rank_two_derivative_certificate"]["nonzero"]

for item in oracle["compiled_four_dynamic_scalar_baseline_checks"]:
    assert item["folded_endpoint_commitment_agreement"]
    assert item["maximum_record_json_bytes_agreement"]
    assert item["boundary_agreement"]
    assert item["retained_folded_endpoint_field_cells_agreement"]
    assert item["total_compiled_warm_field_cells_agreement"]
    assert item["compiler_working_field_cells_agreement"]

mutations = oracle["independent_mutation_checks"]
assert mutations["coupling_omission_changes_boundary"]
assert mutations["module_order_changes_boundary"]
assert mutations["missing_inverse_leaves_nonseed_state"]
assert mutations["wrong_coupling_inverse_rejected_by_seed_equality"]
assert mutations["eta_plus_or_minus_one_inverse_singular"]
assert mutations["rank_one_seed_to_rank_two_to_rank_one"]
assert mutations["second_generic_coupling_generates_four_distinct_terms"]
assert mutations["second_generic_coupling_observed_generated_term_count"] == 4
assert not mutations["second_generic_coupling_minimal_secant_rank_established"]
assert mutations["k_one_excluded_from_declared_program_domain"]

assert production["matched_baseline"]["strongest_declared_warm"] == (
    "COMPILED_FOUR_TOTAL_FOLDED_ENDPOINT_SCALARS"
)
assert not production["matched_baseline"]["phase_advantage_over_matched_classical"]
assert not oracle["resource_law"]["exact_payload_height_tuples_independently_reexecuted"]
assert not oracle["resource_law"]["full_exact_bit_complexity_established"]

ceiling = production["claim_ceiling"]
for key in (
    "repeated_coupling_rank_law_established",
    "m127_grid_orbit_shear_closed",
    "general_secant_rank_reduction_established",
    "gaussian_chart_closure_established",
    "catvm_custody_established",
    "distinct_phase_resource_established",
    "computational_advantage_established",
    "small_wall_crossing_established",
    "physical_waveform_execution_established",
    "physical_bits_replaced_with_pi",
    "unbounded_catalytic_computation_established",
):
    assert not ceiling[key], key
PY

if rg -n 'import f17_coherent_rank2_secant_phase_coupling_closure' \
    "$frontier_dir/f17_coherent_rank2_secant_phase_coupling_closure_oracle.py"; then
    echo "oracle imports production module" >&2
    exit 1
fi

echo QUALIFIED_F17_COHERENT_RANK2_SECANT_PHASE_COUPLING_CLOSURE_STRICT_SCOPE
