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
    "usage: qualify_f17_cubic_latent_character_sum_quotient.sh EVIDENCE_DIR" \
    >&2
  exit 2
fi

evidence_dir=$1
frontier_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo=$(git -C "$frontier_dir" rev-parse --show-toplevel)
python="$repo/.venv/bin/python"
source_path="$frontier_dir/f17_cubic_latent_character_sum_quotient.py"
base_path="$frontier_dir/f17_gaussian_multi_port_phase_quotient.py"
oracle_path="$frontier_dir/f17_cubic_latent_character_sum_dense_oracle.py"
expected_path="$frontier_dir/F17_CUBIC_LATENT_RESULTS.json"
oracle_expected_path="$frontier_dir/F17_CUBIC_LATENT_ORACLE_RESULTS.json"
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

"$python" -m py_compile "$source_path" "$base_path" "$oracle_path"
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

jq -cS '{
  result,
  claim_candidate,
  claim_ceiling,
  restoration_class,
  carrier,
  port_counts:[.cases[].ports],
  primary_module_counts:[.cases[].primary_modules],
  primary_latent_consuming_modules:
    [.cases[].primary_latent_consuming_modules],
  primary_latent_consuming_unique_ports:
    [.cases[].primary_latent_consuming_unique_ports],
  primary_program_sha256:[.cases[].primary_program_sha256],
  reuse_program_sha256:[.cases[].reuse_program_sha256],
  primary_boundary_coefficients:
    [.cases[].primary_boundary.canonical_cyclotomic_coefficients],
  reuse_boundary_coefficients:
    [.cases[].reuse_boundary.canonical_cyclotomic_coefficients],
  primary_boundary_nonzero_coefficients:
    [.cases[].primary_boundary.canonical_nonzero_coefficients],
  reuse_boundary_nonzero_coefficients:
    [.cases[].reuse_boundary.canonical_nonzero_coefficients],
  resident_f17_field_cells:[.cases[].resident_f17_field_cells],
  resident_fixed_width_payload_bits:
    [.cases[].resident_fixed_width_payload_bits],
  resident_numpy_int64_array_payload_bytes:
    [.cases[].resident_numpy_int64_array_payload_bytes],
  accepted_path_baseline_plus_carrier_total_scalar_slots:
    [.cases[].accepted_path_baseline_plus_carrier_total_scalar_slots],
  accepted_path_baseline_plus_carrier_numpy_array_payload_bytes:
    [.cases[].accepted_path_baseline_plus_carrier_numpy_array_payload_bytes],
  verification_peak_three_carrier_total_scalar_slots:
    [.cases[].verification_peak_three_carrier_total_scalar_slots],
  maximum_execution_temporary_field_cells:
    [.cases[].maximum_execution_temporary_field_cells],
  projection_temporary_field_cells_upper_bounds:
    [.cases[].projection_temporary_field_cells_upper_bound],
  primary_boundary_descriptor_bytes:
    [.cases[].primary_boundary_descriptor_bytes],
  reuse_boundary_descriptor_bytes:
    [.cases[].reuse_boundary_descriptor_bytes],
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
      and .same_linear_backing
      and .same_latent_coupling_backing] | all),
  all_reuse_array_backing_preserved:
    ([.cases[] |
      .reuse_same_quadratic_backing
      and .reuse_same_linear_backing
      and .reuse_same_latent_coupling_backing] | all),
  restoration_generations:[.cases[].restoration_generation],
  restoration_leases:[.cases[].restoration_lease],
  primary_compiler_scratch_algebraic_scalar_slots:
    [.cases[].primary_compile.compiler_scratch_algebraic_scalar_slots],
  reuse_compiler_scratch_algebraic_scalar_slots:
    [.cases[].reuse_compile.compiler_scratch_algebraic_scalar_slots],
  primary_compiler_field_update_equivalents:
    [.cases[].primary_compile.program_simulation_field_update_equivalents],
  reuse_compiler_field_update_equivalents:
    [.cases[].reuse_compile.program_simulation_field_update_equivalents],
  primary_compiler_determinant_search_calls:
    [.cases[].primary_compile.determinant_search_calls],
  reuse_compiler_determinant_search_calls:
    [.cases[].reuse_compile.determinant_search_calls],
  retained_inverse_history_bytes:
    [.cases[].retained_inverse_history_bytes],
  baseline_reload_bytes:[.cases[].baseline_reload_bytes],
  controls,
  resource_law,
  composition_advance,
  no_smuggle_scope,
  matched_compact_classical,
  phase_primitive_is_symbolic_f17_character_sum_not_physical_wave,
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

jq -cS '{
  result,
  oracle,
  predeclared_tolerance,
  tested_ports,
  dense_tested_ports,
  delayed_latent_axis_tested_ports,
  primary_program_sha256:[.cases[].primary_program_sha256],
  reuse_program_sha256:[.cases[].reuse_program_sha256],
  exact_reference_f17_field_cells:
    [.cases[].exact_reference_f17_field_cells],
  dense_public_port_state_materialized:
    [.cases[].dense_public_port_state_materialized],
  verification_dense_public_port_logical_complex_cells:
    [.cases[].verification_dense_public_port_logical_complex_cells],
  verification_dense_phase_term_evaluations:
    [.cases[].verification_dense_phase_term_evaluations],
  verification_dense_public_fourier_transforms:
    [.cases[].verification_dense_public_fourier_transforms],
  delayed_latent_axis_materialized:
    [.cases[].delayed_latent_axis_materialized],
  verification_delayed_latent_logical_complex_cells:
    [.cases[].verification_delayed_latent_logical_complex_cells],
  verification_delayed_latent_public_fourier_transforms:
    [.cases[].verification_delayed_latent_public_fourier_transforms],
  all_dense_metrics_within_tolerance:
    (.predeclared_tolerance as $tol |
      [.cases[] | select(.dense_public_port_state_materialized) |
        .primary_boundary_complex_abs_error <= $tol
        and .primary_restoration_max_abs_error <= $tol
        and .reuse_boundary_complex_abs_error <= $tol
        and .reuse_restoration_max_abs_error <= $tol] | all),
  all_exact_boundaries_equal,
  all_exact_restorations_equal,
  production_backend_imported,
  production_compiler_called,
  production_projection_called,
  public_descriptors_consumed,
  exact_reference_reexecutes_all_tested_ports,
  dense_expansion_is_verification_only,
  independent_delayed_latent_residency_parity,
  dense_logical_cells_are_not_process_peak,
  numpy_dense_oracle_process_peak_bounded,
  exact_reference_python_allocator_peak_bounded,
  maximum_verification_dense_public_port_logical_complex_cells,
  maximum_verification_dense_phase_term_evaluations,
  exact_formula_supported_by_dense_parity_at_dense_ports,
  distinct_phase_resource_established,
  computational_advantage,
  small_wall_crossed,
  terminal
}' "$oracle_result" >"$oracle_summary"

jq -cS . "$expected_path" >"$evidence_dir/expected.normalized.json"
jq -cS . "$oracle_expected_path" \
  >"$evidence_dir/oracle_expected.normalized.json"
cmp "$summary" "$evidence_dir/expected.normalized.json"
cmp "$oracle_summary" "$evidence_dir/oracle_expected.normalized.json"

jq -e '
  .result == "PASS"
  and .carrier.tested_ports == [2,4,8,16,32]
  and .carrier.unresolved_shared_latent_coordinates == 1
  and (.carrier.latent_projected_during_forward_composition | not)
  and (.carrier.dense_public_port_phase_vector_materialized | not)
  and (.carrier.public_port_assignment_enumeration_materialized | not)
  and (.carrier.relation_table_materialized | not)
  and .carrier.inverse_descriptors_retained == 0
  and ([.cases[] |
    .resident_f17_field_cells
      == (.ports*.ports + 2*.ports + 4)] | all)
  and ([.cases[] |
    .resident_fixed_width_payload_bits
      == (5*(.ports*.ports + 2*.ports + 4) + 1)] | all)
  and ([.cases[] |
    .resident_numpy_int64_array_payload_bytes
      == 8*(.ports*.ports + 2*.ports)] | all)
  and ([.cases[] |
    .maximum_execution_temporary_field_cells
      == (.ports*.ports + 3*.ports - 1)] | all)
  and ([.cases[] |
    .projection_temporary_field_cells_upper_bound
      == (5*.ports*.ports + 3*.ports + 33)] | all)
  and ([.cases[] |
    .primary_latent_consuming_modules == .ports+2] | all)
  and ([.cases[].primary_latent_consuming_unique_ports |
    . >= 2] | all)
  and ([.cases[].primary_boundary |
    (.canonical_cyclotomic_coefficients | length) == 16
    and .latent_trace_field_evaluations == 17
    and (has("reduced_latent_cubic") | not)
    and (has("reduced_latent_quadratic") | not)
    and (has("reduced_latent_linear") | not)
    and (has("reduced_latent_constant") | not)] | all)
  and ([.cases[].reuse_boundary |
    (.canonical_cyclotomic_coefficients | length) == 16
    and .latent_trace_field_evaluations == 17] | all)
  and ([.cases[].primary_restored_exactly] | all)
  and ([.cases[].reuse_restored_exactly] | all)
  and ([.cases[].fresh_restored_reuse_boundary_equal] | all)
  and ([.cases[].same_logical_carrier_container] | all)
  and ([.cases[].same_quadratic_backing] | all)
  and ([.cases[].same_linear_backing] | all)
  and ([.cases[].same_latent_coupling_backing] | all)
  and ([.cases[].reuse_same_quadratic_backing] | all)
  and ([.cases[].reuse_same_linear_backing] | all)
  and ([.cases[].reuse_same_latent_coupling_backing] | all)
  and ([.cases[].restoration_generation] | all(. == 2))
  and ([.cases[].restoration_lease] | all(. == 3))
  and ([.cases[].baseline_reload_bytes] | all(. == 0))
  and ([.cases[].retained_inverse_history_bytes] | all(. == 0))
  and .controls.missing_inverse_rejected
  and .controls.wrong_inverse_rejected
  and .controls.reordered_inverse_rejected
  and .resource_law.latent_trace_field_evaluations == 17
  and (.resource_law.latent_trace_scales_with_ports | not)
  and .resource_law.serialized_program_descriptor_bytes_counted
  and .resource_law.serialized_boundary_descriptor_bytes_counted
  and (.resource_law
    .compiler_latent_consumer_witness_list_materialized | not)
  and (.resource_law.compiler_candidate_list_materialized | not)
  and .resource_law.semantic_fixed_width_payload_is_not_process_memory
  and (.resource_law.python_program_object_allocator_peak_bounded | not)
  and (.resource_law.numpy_python_allocator_os_process_peak_bounded | not)
  and .composition_advance
    .strictly_broader_than_quadratic_gaussian_phase_chart
  and .composition_advance
    .latent_cubic_third_finite_difference_nonzero
  and .composition_advance
    .unresolved_cubic_latent_retained_across_composition
  and .composition_advance.multiple_native_fourier_consumers
  and .composition_advance.latent_trace_only_at_final_boundary
  and (.composition_advance.multiple_cubic_latents | not)
  and (.composition_advance.arbitrary_non_gaussian_closure | not)
  and (.composition_advance.necklace_carrier_integration | not)
  and (.no_smuggle_scope.reduced_latent_polynomial_exposed | not)
  and (.no_smuggle_scope.machine_enforced_custody | not)
  and .no_smuggle_scope.scope
    == "PACKAGE_SERIALIZATION_ONLY_NOT_MACHINE_ENFORCED"
  and (.no_smuggle_scope.latent_assignments_serialized | not)
  and (.no_smuggle_scope.intermediate_coefficient_states_serialized | not)
  and .no_smuggle_scope.final_cyclotomic_boundary_only
  and .matched_compact_classical
    .identical_exact_cubic_latent_coefficient_recurrence_exists
  and (.matched_compact_classical.strongest_compact_classical_established
    | not)
  and .restoration_class == "EXACT_ALGEBRAIC_RESTORATION"
  and .phase_primitive_is_symbolic_f17_character_sum_not_physical_wave
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
  and .delayed_latent_axis_tested_ports == [2]
  and .all_exact_boundaries_equal
  and .all_exact_restorations_equal
  and .exact_reference_reexecutes_all_tested_ports
  and .exact_formula_supported_by_dense_parity_at_dense_ports
  and .independent_delayed_latent_residency_parity
  and ([.cases[].exact_primary_boundary_equal] | all)
  and ([.cases[].exact_primary_restoration_equal] | all)
  and ([.cases[].exact_reuse_boundary_equal] | all)
  and ([.cases[].exact_reuse_restoration_equal] | all)
  and ([.cases[] |
    .exact_reference_f17_field_cells
      == (.ports*.ports + 2*.ports + 4)] | all)
  and ([.cases[] | select(.dense_public_port_state_materialized) |
    .primary_boundary_complex_abs_error <= 3.0e-11
    and .primary_restoration_max_abs_error <= 3.0e-11
    and .reuse_boundary_complex_abs_error <= 3.0e-11
    and .reuse_restoration_max_abs_error <= 3.0e-11] | all)
  and ([.cases[] | select(.delayed_latent_axis_materialized) |
    .delayed_latent_axis_size_after_forward == 17
    and .delayed_latent_primary_boundary_complex_abs_error <= 3.0e-11
    and .delayed_latent_primary_restoration_max_abs_error <= 3.0e-11
    and .delayed_latent_reuse_boundary_complex_abs_error <= 3.0e-11
    and .delayed_latent_reuse_restoration_max_abs_error <= 3.0e-11] | all)
  and (.production_backend_imported | not)
  and (.production_compiler_called | not)
  and (.production_projection_called | not)
  and .public_descriptors_consumed
  and .dense_expansion_is_verification_only
  and .dense_logical_cells_are_not_process_peak
  and (.numpy_dense_oracle_process_peak_bounded | not)
  and (.exact_reference_python_allocator_peak_bounded | not)
  and (.distinct_phase_resource_established | not)
  and (.computational_advantage | not)
  and (.small_wall_crossed | not)
  and (.terminal | not)
' "$oracle_result" >/dev/null

if rg -n \
  'complex128|np\\.exp|np\\.indices|np\\.meshgrid|np\\.kron|itertools\\.product' \
  "$source_path" "$base_path"
then
  echo "accepted compact source contains dense or complex construction" >&2
  exit 1
fi

jq -n \
  --arg result_sha256 "$(sha256sum "$result" | cut -d' ' -f1)" \
  --arg oracle_sha256 "$(sha256sum "$oracle_result" | cut -d' ' -f1)" \
  --arg source_sha256 "$(sha256sum "$source_path" | cut -d' ' -f1)" \
  --arg base_source_sha256 \
    "$(sha256sum "$base_path" | cut -d' ' -f1)" \
  --arg oracle_source_sha256 \
    "$(sha256sum "$oracle_path" | cut -d' ' -f1)" \
  '{
    result:"PASS",
    verification_level:"SEPARATE_REFERENCE_PARITY",
    classification:"INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
    restoration_class:"EXACT_ALGEBRAIC_RESTORATION",
    exact_reference_ports:[2,4,8,16,32],
    dense_complex_parity_ports:[2,4],
    delayed_latent_axis_parity_ports:[2],
    result_sha256:$result_sha256,
    oracle_sha256:$oracle_sha256,
    source_sha256:$source_sha256,
    base_source_sha256:$base_source_sha256,
    oracle_source_sha256:$oracle_source_sha256,
    accepted_single_cubic_latent_phase_quotient:true,
    multiple_cubic_latents:false,
    distinct_phase_resource_established:false,
    computational_advantage:false,
    small_wall_crossed:false,
    terminal:false
  }' >"$evidence_dir/qualification.json"

echo "F17 cubic-latent character-sum qualification passed"
