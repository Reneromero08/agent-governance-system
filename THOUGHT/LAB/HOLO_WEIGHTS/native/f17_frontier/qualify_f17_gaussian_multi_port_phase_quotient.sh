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
    "usage: qualify_f17_gaussian_multi_port_phase_quotient.sh EVIDENCE_DIR" \
    >&2
  exit 2
fi

evidence_dir=$1
frontier_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo=$(git -C "$frontier_dir" rev-parse --show-toplevel)
python="$repo/.venv/bin/python"
source_path="$frontier_dir/f17_gaussian_multi_port_phase_quotient.py"
oracle_path="$frontier_dir/f17_gaussian_multi_port_dense_oracle.py"
expected_path="$frontier_dir/F17_GAUSSIAN_MULTI_PORT_RESULTS.json"
oracle_expected_path="$frontier_dir/F17_GAUSSIAN_MULTI_PORT_ORACLE_RESULTS.json"
result="$evidence_dir/result.json"
replay="$evidence_dir/replay.json"
oracle_result="$evidence_dir/oracle.json"
summary="$evidence_dir/summary.json"
oracle_summary="$evidence_dir/oracle_summary.json"

mkdir -p "$evidence_dir"
export PYTHONPYCACHEPREFIX="$evidence_dir/pycache"

for tool in jq cmp sha256sum nice rg; do
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
  carrier,
  port_counts:[.cases[].ports],
  primary_module_counts:[.cases[].primary_modules],
  primary_program_sha256:[.cases[].primary_program_sha256],
  reuse_program_sha256:[.cases[].reuse_program_sha256],
  primary_boundaries:[.cases[].primary_boundary],
  reuse_boundaries:[.cases[].reuse_boundary],
  resident_f17_field_cells:[.cases[].resident_f17_field_cells],
  resident_fixed_width_payload_bits:
    [.cases[].resident_fixed_width_payload_bits],
  accepted_path_baseline_plus_carrier_total_scalar_slots:
    [.cases[].accepted_path_baseline_plus_carrier_total_scalar_slots],
  verification_peak_three_carrier_total_scalar_slots:
    [.cases[].verification_peak_three_carrier_total_scalar_slots],
  maximum_execution_temporary_field_cells:
    [.cases[].maximum_execution_temporary_field_cells],
  projection_temporary_field_cells_upper_bounds:
    [.cases[].projection_temporary_field_cells_upper_bound],
  compile_determinant_search_calls:
    [.cases[].primary_compile_determinant_search_calls],
  compile_determinant_search_field_update_equivalents:
    [.cases[].primary_compile_determinant_search_field_update_equivalents],
  compile_determinant_search_field_inversions:
    [.cases[].primary_compile_determinant_search_field_inversions],
  reuse_compile_determinant_search_calls:
    [.cases[].reuse_compile_determinant_search_calls],
  reuse_compile_determinant_search_field_update_equivalents:
    [.cases[].reuse_compile_determinant_search_field_update_equivalents],
  reuse_compile_determinant_search_field_inversions:
    [.cases[].reuse_compile_determinant_search_field_inversions],
  primary_execution_field_update_equivalents:
    [.cases[].primary_execution_field_update_equivalents],
  primary_execution_field_inversions:
    [.cases[].primary_execution_field_inversions],
  dense_phase_cells_not_materialized:
    [.cases[].dense_phase_cells_not_materialized],
  all_primary_restored_exactly:
    ([.cases[].primary_restored_exactly] | all),
  all_reuse_restored_exactly:
    ([.cases[].reuse_restored_exactly] | all),
  all_fresh_restored_reuse_boundaries_equal:
    ([.cases[].fresh_restored_reuse_boundary_equal] | all),
  all_carrier_and_array_backing_preserved:
    ([.cases[] |
      .same_logical_carrier_container
      and .same_quadratic_backing
      and .same_linear_backing] | all),
  restoration_generations:[.cases[].restoration_generation],
  retained_inverse_history_bytes:
    [.cases[].retained_inverse_history_bytes],
  baseline_reload_bytes:[.cases[].baseline_reload_bytes],
  controls,
  resource_law,
  matched_compact_classical,
  phase_primitive_is_symbolic_f17_exponent_not_physical_wave,
  catvm_custody_established,
  distinct_phase_resource_established,
  computational_advantage,
  small_wall_crossed,
  catalytic_inference_established,
  physical_waveform_execution,
  replacement_of_physical_bits_with_pi,
  unbounded_computation_established,
  terminal
}' "$result" >"$summary"

jq -S '{
  result,
  oracle,
  predeclared_tolerance,
  tested_ports,
  dense_tested_ports,
  primary_program_sha256:[.cases[].primary_program_sha256],
  reuse_program_sha256:[.cases[].reuse_program_sha256],
  exact_reference_f17_field_cells:
    [.cases[].exact_reference_f17_field_cells],
  dense_phase_vector_materialized:
    [.cases[].dense_phase_vector_materialized],
  verification_dense_logical_complex_cells:
    [.cases[].verification_dense_logical_complex_cells],
  all_dense_metrics_within_tolerance:
    (.predeclared_tolerance as $tol |
      [.cases[] | select(.dense_phase_vector_materialized) |
        .forward_norm_error <= $tol
        and .primary_boundary_complex_abs_error <= $tol
        and .primary_restoration_max_abs_error <= $tol
        and .reuse_boundary_complex_abs_error <= $tol
        and .reuse_restoration_max_abs_error <= $tol] |
      all),
  all_exact_boundaries_equal,
  all_exact_restorations_equal,
  production_backend_imported,
  production_compiler_called,
  production_projection_called,
  public_descriptors_consumed,
  verification_assignment_expansion_materialized,
  verification_dense_logical_vector_cells_counted,
  dense_logical_vector_cells_are_not_process_peak,
  numpy_dense_oracle_process_peak_bounded,
  maximum_verification_dense_logical_complex_cells,
  dense_oracle_is_accepted_compact_carrier_path,
  exact_reference_uses_compact_quadratic_coefficients,
  exact_reference_reexecutes_all_tested_ports,
  exact_formula_supported_by_numerical_parity_at_dense_ports,
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
  and .carrier.tested_ports == [2,4,8,16,32]
  and (.carrier.dense_phase_vector_materialized | not)
  and (.carrier.assignment_enumeration_materialized | not)
  and (.carrier.truth_table_materialized | not)
  and .carrier.inverse_descriptors_retained == 0
  and ([.cases[] |
    .resident_f17_field_cells
      == (.ports*.ports + .ports + 1)] | all)
  and ([.cases[] |
    .resident_fixed_width_payload_bits
      == (5*(.ports*.ports + .ports + 1) + 1)] | all)
  and ([.cases[] |
    .maximum_execution_temporary_field_cells
      == (.ports*.ports + 2*.ports - 1)] | all)
  and ([.cases[] |
    .projection_temporary_field_cells_upper_bound
      == (5*.ports*.ports + .ports)] | all)
  and ([.cases[] |
    .accepted_path_baseline_plus_carrier_total_scalar_slots
      == 2*(.ports*.ports + .ports + 6)] | all)
  and ([.cases[] |
    .verification_peak_three_carrier_total_scalar_slots
      == 3*(.ports*.ports + .ports + 6)] | all)
  and ([.cases[].primary_compile_determinant_search_calls |
    . >= 1 and . <= 17] | all)
  and ([.cases[].primary_compile_determinant_search_field_inversions |
    . > 0] | all)
  and ([.cases[].primary_compile_determinant_search_field_update_equivalents |
    . > 0] | all)
  and ([.cases[].reuse_compile_determinant_search_calls |
    . >= 1 and . <= 17] | all)
  and ([.cases[].reuse_compile_determinant_search_field_inversions |
    . > 0] | all)
  and ([.cases[].reuse_compile_determinant_search_field_update_equivalents |
    . > 0] | all)
  and ([.cases[].primary_restored_exactly] | all)
  and ([.cases[].reuse_restored_exactly] | all)
  and ([.cases[].fresh_restored_reuse_boundary_equal] | all)
  and ([.cases[].same_logical_carrier_container] | all)
  and ([.cases[].same_quadratic_backing] | all)
  and ([.cases[].same_linear_backing] | all)
  and ([.cases[].restoration_generation] | all(. == 2))
  and ([.cases[].baseline_reload_bytes] | all(. == 0))
  and ([.cases[].retained_inverse_history_bytes] | all(. == 0))
  and .controls.missing_inverse_rejected
  and .controls.wrong_inverse_rejected
  and .controls.reordered_inverse_rejected
  and .resource_law.compiler_determinant_search_counted
  and .resource_law.serialized_program_descriptor_bytes_counted
  and (.resource_law.python_program_object_allocator_peak_bounded | not)
  and (.resource_law.numpy_python_allocator_os_process_peak_bounded | not)
  and .matched_compact_classical
    .identical_exact_gaussian_coefficient_recurrence_exists
  and (.matched_compact_classical.strongest_compact_classical_established
    | not)
  and .restoration_class == "EXACT_ALGEBRAIC_RESTORATION"
  and .phase_primitive_is_symbolic_f17_exponent_not_physical_wave
  and (.catvm_custody_established | not)
  and (.distinct_phase_resource_established | not)
  and (.computational_advantage | not)
  and (.small_wall_crossed | not)
  and (.catalytic_inference_established | not)
  and (.physical_waveform_execution | not)
  and (.replacement_of_physical_bits_with_pi | not)
  and (.unbounded_computation_established | not)
  and (.terminal | not)
' "$result" >/dev/null

jq -e '
  .result == "PASS"
  and .tested_ports == [2,4,8,16,32]
  and .dense_tested_ports == [2,4]
  and .all_exact_boundaries_equal
  and .all_exact_restorations_equal
  and .exact_reference_reexecutes_all_tested_ports
  and .exact_reference_uses_compact_quadratic_coefficients
  and .exact_formula_supported_by_numerical_parity_at_dense_ports
  and ([.cases[].exact_primary_boundary_equal] | all)
  and ([.cases[].exact_primary_restoration_equal] | all)
  and ([.cases[].exact_reuse_boundary_equal] | all)
  and ([.cases[].exact_reuse_restoration_equal] | all)
  and ([.cases[] |
    .exact_reference_f17_field_cells
      == (.ports*.ports + .ports + 1)] | all)
  and ([.cases[] | select(.dense_phase_vector_materialized) |
    .forward_norm_error <= 2.0e-11
    and .primary_boundary_complex_abs_error <= 2.0e-11
    and .primary_restoration_max_abs_error <= 2.0e-11
    and .reuse_boundary_complex_abs_error <= 2.0e-11
    and .reuse_restoration_max_abs_error <= 2.0e-11] | all)
  and ([.cases[] | select(.dense_phase_vector_materialized | not) |
    .verification_dense_logical_complex_cells == 0] | all)
  and (.production_backend_imported | not)
  and (.production_compiler_called | not)
  and (.production_projection_called | not)
  and .public_descriptors_consumed
  and .verification_assignment_expansion_materialized
  and .verification_dense_logical_vector_cells_counted
  and .dense_logical_vector_cells_are_not_process_peak
  and (.numpy_dense_oracle_process_peak_bounded | not)
  and (.dense_oracle_is_accepted_compact_carrier_path | not)
  and (.distinct_phase_resource_established | not)
  and (.computational_advantage | not)
  and (.small_wall_crossed | not)
  and (.terminal | not)
' "$oracle_result" >/dev/null

if rg -n \
  'complex128|np\\.exp|np\\.indices|np\\.meshgrid|np\\.kron|itertools\\.product' \
  "$source_path"
then
  echo "accepted compact source contains dense or complex construction" >&2
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
    restoration_class:"EXACT_ALGEBRAIC_RESTORATION",
    exact_reference_ports:[2,4,8,16,32],
    dense_complex_parity_ports:[2,4],
    result_sha256:$result_sha256,
    oracle_sha256:$oracle_sha256,
    source_sha256:$source_sha256,
    oracle_source_sha256:$oracle_source_sha256,
    accepted_compact_phase_quotient:true,
    distinct_phase_resource_established:false,
    computational_advantage:false,
    small_wall_crossed:false,
    terminal:false
  }' >"$evidence_dir/qualification.json"

echo "F17 Gaussian multi-port phase quotient qualification passed"
