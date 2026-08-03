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
production_seal="$frontier_dir/F17_COHERENT_RANK4_DOUBLE_SECANT_PHASE_COUPLING_CLOSURE_RESULTS.json"
oracle_seal="$frontier_dir/F17_COHERENT_RANK4_DOUBLE_SECANT_PHASE_COUPLING_CLOSURE_ORACLE_RESULTS.json"
production_replay="$evidence_dir/production.json"
oracle_replay="$evidence_dir/oracle.json"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
export LC_ALL=C
export PYTHONPATH="$frontier_dir"

nice -n 10 "$python_bin" -X dev \
    "$frontier_dir/f17_coherent_rank4_double_secant_phase_coupling_closure.py" \
    --output "$production_replay"
cmp "$production_seal" "$production_replay"

nice -n 10 "$python_bin" -X dev \
    "$frontier_dir/f17_coherent_rank4_double_secant_phase_coupling_closure_oracle.py" \
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

transactions = production["exact_transactions"] + production["dual_field_structural_transactions"]
assert len(transactions) == 16
for item in transactions:
    assert item["resident_phase_field_cells"] == 72
    assert item["resident_chart_rank"] == 4
    assert item["maximum_active_rank"] == 4
    assert item["maximum_coupling_transient_components"] == 8
    assert item["maximum_coupling_transient_field_cells"] == 144
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
    assert not item["accepted_path_occupation_topology_materialized"]
    assert not item["accepted_path_matching_or_assignment_expansion_materialized"]
    assert not item["accepted_path_dense_operator_materialized"]
    assert not item["intermediate_rank4_components_serialized_to_controller"]
    assert "components" not in item
    assert len(item["forward_rank4_commitment"]) == 64

certificates = production["catalecticant_rank_certificates"]
assert len(certificates) == 16
for item in certificates:
    assert item["minor_nonzero"]
    assert item["lower_bound"] == 4
    assert item["generated_component_upper_bound"] == 4
    assert item["exact_normalized_divided_power_secant_rank"] == 4
    assert item["ordinary_symmetric_waring_rank_interpretation"] == (
        item["algebra"] == transactions[0]["algebra"]
    )
    assert item["verification_carrier_restored_to_empty"]
    assert not item["minor_value_serialized"]
    assert not item["intermediate_amplitudes_serialized"]

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
    "independent_catalecticant_rank_four",
    "resident_field_cells_agreement",
    "maximum_coupling_transient_field_cells_agreement",
    "implicit_dimension_agreement",
)
for item in oracle["transaction_parity"]:
    assert all(item[key] for key in parity_keys)

for item in oracle["catalecticant_certificate_parity"]:
    assert item["minor_nonzero_agreement"]
    assert item["exact_rank_four_agreement"]
    assert item["rank_interpretation_agreement"]

for item in oracle["full_occupation_oracle_cases"]:
    assert item["occupation_dimension"] == 4845
    assert item["first_coupling_full_occupation_agreement"]
    assert item["module_a_full_occupation_agreement"]
    assert item["second_coupling_full_occupation_agreement"]
    assert item["forward_full_occupation_rank4_agreement"]
    assert item["forward_boundary_agreement"]
    assert item["dense_forward_inverse_restored"]
    assert item["chart_forward_inverse_restored"]
    assert item["rank_four_catalecticant_certificate"][
        "exact_normalized_divided_power_secant_rank"
    ] == 4

for item in oracle["compiled_eight_total_scalar_baseline_checks"]:
    assert item["folded_endpoint_commitment_agreement"]
    assert item["maximum_record_json_bytes_agreement"]
    assert item["boundary_agreement"]
    assert item["retained_folded_endpoint_field_cells_agreement"]
    assert item["total_compiled_warm_field_cells_agreement"]
    assert item["compiler_working_field_cells_agreement"]

assert all(oracle["independent_mutation_checks"].values())
assert production["matched_baseline"]["strongest_declared_warm"] == (
    "COMPILED_EIGHT_TOTAL_FOLDED_ENDPOINT_SCALARS"
)
assert not production["matched_baseline"]["phase_advantage_over_matched_classical"]
assert not oracle["resource_law"]["exact_payload_height_tuples_independently_reexecuted"]
assert not oracle["resource_law"]["full_exact_bit_complexity_established"]

ceiling = production["claim_ceiling"]
for key in (
    "third_or_unbounded_coupling_rank_law_established",
    "fixed_rank_unbounded_depth_closure_established",
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

if rg -n 'import f17_coherent_rank4_double_secant_phase_coupling_closure' \
    "$frontier_dir/f17_coherent_rank4_double_secant_phase_coupling_closure_oracle.py"; then
    echo "oracle imports production module" >&2
    exit 1
fi

echo QUALIFIED_F17_COHERENT_RANK4_DOUBLE_SECANT_PHASE_COUPLING_CLOSURE_STRICT_SCOPE
