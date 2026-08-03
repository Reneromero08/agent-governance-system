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
production_seal="$frontier_dir/F17_COHERENT_VERONESE_PHASE_CHART_CLOSURE_RESULTS.json"
oracle_seal="$frontier_dir/F17_COHERENT_VERONESE_PHASE_CHART_CLOSURE_ORACLE_RESULTS.json"
production_replay="$evidence_dir/production.json"
oracle_replay="$evidence_dir/oracle.json"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
export LC_ALL=C
export PYTHONPATH="$frontier_dir"

nice -n 10 "$python_bin" -X dev \
    "$frontier_dir/f17_coherent_veronese_phase_chart_closure.py" \
    --output "$production_replay"
cmp "$production_seal" "$production_replay"

nice -n 10 "$python_bin" -X dev \
    "$frontier_dir/f17_coherent_veronese_phase_chart_closure_oracle.py" \
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

expected_k = (4, 8, 16, 32, 64, 128)
expected_exact_k = (4, 8, 16, 32)
assert tuple(item["k"] for item in production["exact_transactions"]) == expected_exact_k
assert tuple(production["source_scope"]["declared_degrees"]) == expected_k

resident_payloads = set()
resident_heights = set()
for item in production["exact_transactions"]:
    assert item["resident_phase_field_cells"] == 17
    assert item["resident_chart_rank"] == 1
    assert item["implicit_occupation_dimension_h_k"] > 17
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
    assert not item["accepted_path_character_table_materialized"]
    assert not item["accepted_path_dense_operator_materialized"]
    assert not item["accepted_path_coherent_component_expansion_materialized"]
    assert not item["intermediate_chart_serialized_to_controller"]
    assert "forward_chart" not in item
    assert len(item["forward_chart_commitment"]) == 64
    assert item["maximum_commitment_record_json_bytes"] > 0
    resident_payloads.add(item["maximum_resident_payload_bits"])
    resident_heights.add(
        (
            item["maximum_resident_numerator_signed_bits"],
            item["maximum_resident_denominator_bits"],
        )
    )
assert len(resident_payloads) == 1
assert len(resident_heights) == 1
assert production["exact_resident_payload_invariant_across_declared_exact_degrees"]

for item in production["dual_field_structural_transactions"]:
    assert item["resident_phase_field_cells"] == 17
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
    "wrong_inverse_detected",
    "reciprocal_instead_of_additive_shear_inverse_detected",
    "reordered_inverse_detected",
    "wrong_program_ownership_rejected",
    "premature_projection_rejected",
    "null_carrier_rejected",
    "invalid_degree_fixture_rejected",
    "rank_two_chart_descriptor_rejected",
    "nonintegrable_grid_orbit_shear_rejected",
    "nonadjacent_shear_rejected",
    "power_sum_character_mutation_changes_boundary",
    "mode_shear_mutation_changes_boundary",
    "rank_two_noncoherent_witness_rejected",
    "actual_m128_grid_injection_leaves_rank_one_chart",
):
    assert controls[key], key

transaction_parity = (
    oracle["exact_transaction_parity"] + oracle["dual_field_transaction_parity"]
)
for item in transaction_parity:
    assert item["program_fingerprint_agreement"]
    assert item["forward_chart_commitment_agreement"]
    assert item["boundary_agreement"]
    assert item["independent_forward_inverse_restored"]
    assert item["resident_field_cells_agreement"]
    assert item["implicit_dimension_agreement"]

for item in oracle["full_occupation_oracle_cases"]:
    assert item["occupation_dimension"] == 4845
    assert item["full_occupation_chart_agreement"]
    assert item["forward_rank_one_catalecticant_minor_zero"]
    assert item["dense_forward_inverse_restored"]
    assert item["chart_forward_inverse_restored"]

for item in oracle["individual_primitive_checks"]:
    assert item["primitive_count"] == 40
    assert item["every_individual_primitive_and_inverse_agrees"]

for item in oracle["actual_m128_grid_exit_witness_checks"]:
    assert item["entry_commitment_agreement"]
    assert item["minor_agreement"]
    assert item["rank_one_chart_rejected"]
    assert item["independent_prior_carrier_restored"]

baselines = (
    production["compiled_two_scalar_warm_classical_baselines"]["exact_q_zeta17"]
    + production["compiled_two_scalar_warm_classical_baselines"]["dual_field"]
)
for item in baselines:
    assert item["boundary_agreement"]
    assert item["compiler_working_field_cells"] == 17
    assert item["retained_warm_boundary_pair_field_cells"] == 2
    assert item["warm_named_transient_field_cells"] == 4
    assert len(item["retained_warm_boundary_pair_commitment"]) == 64
    assert item["maximum_commitment_record_json_bytes"] > 0
    assert not item["snapshot_or_phase_carrier_used"]

for item in oracle["compiled_two_scalar_warm_classical_baseline_checks"]:
    assert item["pair_commitment_agreement"]
    assert item["maximum_record_json_bytes_agreement"]
    assert item["boundary_agreement"]
    assert item["warm_retained_field_cells_agreement"]
    assert item["compiler_working_field_cells_agreement"]

assert (
    oracle["matched_baseline"]["strongest_sealed_fixture_warm"]
    == "COMPILED_TWO_SCALAR_V0_V1_RETENTION_WITH_CLOSED_FORM_OCCUPATION_PROJECTION"
)
assert not oracle["matched_baseline"]["phase_advantage_over_matched_classical"]

ceiling = production["claim_ceiling"]
assert ceiling["rank_one_coherent_chart_only"]
assert ceiling["fixed_public_primitive_schedule_only"]
assert not ceiling["m127_grid_orbit_shear_closed"]
assert not ceiling["multiple_coherent_component_closure_established"]
assert not ceiling["arbitrary_h_k_input_closed"]
assert not ceiling["general_nonlinear_quotient_established"]
assert not ceiling["catvm_custody_established"]
assert not ceiling["distinct_phase_resource_established"]
assert not ceiling["computational_advantage_established"]
assert not ceiling["small_wall_crossing_established"]
assert not ceiling["physical_waveform_execution_established"]
assert not ceiling["physical_bits_replaced_with_pi"]
assert not ceiling["unbounded_catalytic_computation_established"]
PY

echo QUALIFIED_F17_COHERENT_VERONESE_PHASE_CHART_CLOSURE_STRICT_SCOPE
