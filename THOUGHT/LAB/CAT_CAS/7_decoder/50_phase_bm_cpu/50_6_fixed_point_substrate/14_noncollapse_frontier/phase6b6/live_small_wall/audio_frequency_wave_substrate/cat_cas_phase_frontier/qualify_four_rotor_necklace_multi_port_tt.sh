#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
export PYTHONHASHSEED=0
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
ulimit -c 0

if [[ $# -ne 1 ]]; then
  echo \
    "usage: qualify_four_rotor_necklace_multi_port_tt.sh EVIDENCE_DIR" \
    >&2
  exit 2
fi

evidence_dir=$1
frontier_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo=$(git -C "$frontier_dir" rev-parse --show-toplevel)
python="$repo/.venv/bin/python"
source_path="$frontier_dir/four_rotor_necklace_multi_port_tt_phase.py"
oracle_path="$frontier_dir/four_rotor_necklace_multi_port_dense_oracle.py"
expected_path="$frontier_dir/MULTI_PORT_TT_RESULTS.json"
oracle_expected_path="$frontier_dir/MULTI_PORT_TT_DENSE_ORACLE_RESULTS.json"
result="$evidence_dir/result.json"
replay="$evidence_dir/replay.json"
oracle_result="$evidence_dir/oracle.json"
summary="$evidence_dir/summary.json"
oracle_summary="$evidence_dir/oracle_summary.json"

mkdir -p "$evidence_dir"
export PYTHONPYCACHEPREFIX="$evidence_dir/pycache"

for tool in jq cmp sha256sum nice; do
  command -v "$tool" >/dev/null
done
test -x "$python"
jq empty "$expected_path" "$oracle_expected_path"

"$python" -m py_compile "$source_path" "$oracle_path"
nice -n 10 "$python" -X dev "$source_path" \
  >"$result" 2>"$evidence_dir/result.stderr"
nice -n 10 "$python" -X dev "$source_path" \
  >"$replay" 2>"$evidence_dir/replay.stderr"
test ! -s "$evidence_dir/result.stderr"
test ! -s "$evidence_dir/replay.stderr"
cmp "$result" "$replay"

nice -n 10 "$python" -X dev "$oracle_path" "$result" \
  >"$oracle_result" 2>"$evidence_dir/oracle.stderr"
test ! -s "$evidence_dir/oracle.stderr"

jq -S '{
  result,
  claim_candidate,
  claim_ceiling,
  restoration_class,
  tested_port_counts:.carrier.tested_port_counts,
  primary_module_counts:[.cases[].primary_module_count],
  reuse_module_counts:[.cases[].reuse_module_count],
  primary_forward_final_ranks:[.cases[].primary_forward_final_ranks],
  primary_restored_ranks:[.cases[].primary_restored_ranks],
  primary_restoration_upper_bounds:
    [.cases[].primary_restoration_upper_bound],
  reuse_restoration_upper_bounds:
    [.cases[].reuse_restoration_upper_bound],
  fresh_restored_reuse_boundary_errors:
    [.cases[].fresh_restored_reuse_boundary_error],
  primary_peak_tt_complex_cells:
    [.cases[].primary_peak_tt_complex_cells],
  matched_explicit_dense_assignment_complex_cells:
    [.cases[].matched_explicit_dense_assignment_complex_cells],
  maximum_temporary_component_complex_cells:
    [.cases[].maximum_temporary_complex_cells],
  maximum_accounted_primary_resident_carrier_complex_cells:
    [.cases[].maximum_accounted_primary_resident_carrier_complex_cells],
  maximum_accounted_reuse_parity_resident_carrier_complex_cells:
    [.cases[].maximum_accounted_reuse_parity_resident_carrier_complex_cells],
  primary_program_sha256:[.cases[].primary_program_sha256],
  reuse_program_sha256:[.cases[].reuse_program_sha256],
  rank_observation,
  controls,
  resource_law,
  matched_classical_references,
  strongest_compact_classical_established,
  observed_obstruction,
  compact_multi_port_carrier_established,
  fixed_rank_multi_port_closure_established,
  catvm_custody_established,
  distinct_phase_resource_established,
  computational_advantage,
  small_wall_crossed,
  catalytic_inference_established,
  physical_waveform_execution,
  replacement_of_physical_bits_with_pi,
  terminal
}' "$result" >"$summary"

jq -S '{
  result,
  oracle,
  tested_port_counts,
  final_ranks:[.cases[].final_ranks],
  minimum_retained_to_cutoff_ratios:
    [.cases[].final_canonical_matricization_certificate
      .minimum_retained_to_cutoff_ratio],
  maximum_discarded_to_cutoff_ratios:
    [.cases[].final_canonical_matricization_certificate
      .maximum_discarded_to_cutoff_ratio],
  boundary_max_abs_errors:[.cases[].boundary_max_abs_error],
  restoration_l2_errors:[.cases[].restoration_l2_error],
  reuse_boundary_max_abs_errors:[.cases[].reuse_boundary_max_abs_error],
  verification_dense_complex_cells:
    [.cases[].verification_dense_complex_cells],
  all_peak_ranks_match,
  all_final_ranks_match,
  all_final_canonical_cuts_full_numerical_rank,
  all_boundaries_match,
  all_restorations_within_tolerance,
  production_backend_imported,
  production_program_compiler_called,
  production_projection_called,
  verification_dense_assignment_materialized,
  verification_dense_cells_counted,
  maximum_verification_dense_complex_cells,
  dense_operator_materialized,
  accepted_carrier_path,
  matched_identical_tt_reference_exists,
  strongest_compact_classical_established,
  fixed_rank_multi_port_closure_established,
  distinct_phase_resource_established,
  computational_advantage,
  small_wall_crossed,
  terminal
}' "$oracle_result" >"$oracle_summary"

jq -S . "$expected_path" >"$evidence_dir/expected.normalized.json"
jq -S . "$oracle_expected_path" \
  >"$evidence_dir/oracle_expected.normalized.json"
cmp "$summary" "$evidence_dir/expected.normalized.json"
cmp "$oracle_summary" "$evidence_dir/oracle_expected.normalized.json"

jq -e '
  .result == "PASS"
  and .rank_observation.first_bond_rank_equals_full_binary_width
  and (.rank_observation.fixed_rank_across_tested_port_count | not)
  and (.rank_observation.tensor_train_smaller_than_explicit_dense_assignment
    | not)
  and ([.cases[].equivalent_full_width_first_bond_materialized] | all)
  and ([.cases[].same_logical_carrier_container] | all)
  and ([.cases[].core_array_backing_preserved] | any | not)
  and ([.cases[].baseline_reload_bytes] | all(. == 0))
  and ([.cases[].retained_inverse_history_bytes] | all(. == 0))
  and ([.cases[].reported_temporary_cells_are_component_maximum_not_rss_peak]
    | all)
  and .restoration_class
    == "INVERSE_PLUS_CANONICAL_NUMERICAL_QUOTIENT"
  and .matched_classical_references.separate_reference_parity
  and (.strongest_compact_classical_established | not)
  and (.fixed_rank_multi_port_closure_established | not)
  and (.compact_multi_port_carrier_established | not)
  and (.catvm_custody_established | not)
  and (.distinct_phase_resource_established | not)
  and (.computational_advantage | not)
  and (.small_wall_crossed | not)
  and (.terminal | not)
' "$result" >/dev/null

jq -e '
  .result == "PASS"
  and .all_peak_ranks_match
  and .all_final_ranks_match
  and .all_final_canonical_cuts_full_numerical_rank
  and .all_boundaries_match
  and .all_restorations_within_tolerance
  and ([.cases[].final_canonical_matricization_certificate
    .minimum_retained_to_cutoff_ratio] | all(. > 1.0e10))
  and ([.cases[].final_canonical_matricization_certificate
    .maximum_discarded_to_cutoff_ratio] | all(. == 0.0))
  and (.production_backend_imported | not)
  and (.production_program_compiler_called | not)
  and (.production_projection_called | not)
  and .verification_dense_assignment_materialized
  and .verification_dense_cells_counted
  and (.accepted_carrier_path | not)
  and .matched_identical_tt_reference_exists
  and (.strongest_compact_classical_established | not)
  and (.distinct_phase_resource_established | not)
  and (.computational_advantage | not)
  and (.small_wall_crossed | not)
' "$oracle_result" >/dev/null

if rg -n \
  'np\\.kron|verification_dense_assignment_materialized.*true' \
  "$source_path"
then
  echo "accepted TT path contains verification-only dense construction" >&2
  exit 1
fi

jq -n \
  --arg result_sha256 "$(sha256sum "$result" | cut -d' ' -f1)" \
  --arg oracle_sha256 "$(sha256sum "$oracle_result" | cut -d' ' -f1)" \
  --arg source_sha256 "$(sha256sum "$source_path" | cut -d' ' -f1)" \
  --arg oracle_source_sha256 \
    "$(sha256sum "$oracle_path" | cut -d' ' -f1)" \
  '{
    result:"PASS",
    verification_level:"SEPARATE_REFERENCE_PARITY",
    classification:"INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
    restoration_class:"INVERSE_PLUS_CANONICAL_NUMERICAL_QUOTIENT",
    result_sha256:$result_sha256,
    oracle_sha256:$oracle_sha256,
    source_sha256:$source_sha256,
    oracle_source_sha256:$oracle_source_sha256,
    accepted_compaction_result:false,
    terminal:false
  }' >"$evidence_dir/qualification.json"

echo "multi-port TT qualification passed"
