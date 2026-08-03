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
    "usage: qualify_f17_interacting_cubic_latent_trace.sh EVIDENCE_DIR" \
    >&2
  exit 2
fi

evidence_dir=$1
frontier_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo=$(git -C "$frontier_dir" rev-parse --show-toplevel)
python="$repo/.venv/bin/python"
source_path="$frontier_dir/f17_interacting_cubic_latent_trace_diagnostic.py"
base_path="$frontier_dir/f17_gaussian_multi_port_phase_quotient.py"
oracle_path="$frontier_dir/f17_interacting_cubic_latent_trace_oracle.py"
expected_path="$frontier_dir/F17_INTERACTING_CUBIC_TRACE_RESULTS.json"
oracle_expected_path="$frontier_dir/F17_INTERACTING_CUBIC_TRACE_ORACLE_RESULTS.json"
provenance_path="$frontier_dir/F17_INTERACTING_CUBIC_TRACE_PROVENANCE.json"
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
jq empty "$expected_path" "$oracle_expected_path" "$provenance_path"

for sealed_path in \
  "$source_path" \
  "$base_path" \
  "$oracle_path" \
  "$expected_path" \
  "$oracle_expected_path"
do
  sealed_name=$(basename "$sealed_path")
  sealed_expected=$(jq -r --arg name "$sealed_name" \
    '.files[$name] // empty' "$provenance_path")
  test -n "$sealed_expected"
  test "$(sha256sum "$sealed_path" | cut -d' ' -f1)" = "$sealed_expected"
done

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
  classification_candidate,
  restoration_class,
  carrier,
  tested_cases:[.cases[]|{ports,latents}],
  primary_program_sha256:[.cases[].primary_program_sha256],
  reuse_program_sha256:[.cases[].reuse_program_sha256],
  primary_boundary_coefficients:
    [.cases[].primary_boundary.canonical_cyclotomic_coefficients],
  reuse_boundary_coefficients:
    [.cases[].reuse_boundary.canonical_cyclotomic_coefficients],
  resident_f17_field_cells:[.cases[].resident_f17_field_cells],
  resident_fixed_width_payload_bits:
    [.cases[].resident_fixed_width_payload_bits],
  latent_cubic_coefficient_cells:
    [.cases[].latent_cubic_coefficient_cells],
  interacting_latent_cubic_coefficient_cells:
    [.cases[].interacting_latent_cubic_coefficient_cells],
  latent_trace_assignments_streamed:
    [.cases[].latent_trace_assignments_streamed],
  latent_trace_total_polynomial_term_evaluations:
    [.cases[].latent_trace_total_polynomial_term_evaluations],
  latent_trace_constant_modulo_histogram_operations:
    [.cases[].latent_trace_constant_modulo_histogram_operations],
  projection_temporary_scalar_slots_upper_bound:
    [.cases[].projection_temporary_scalar_slots_upper_bound],
  projection_temporary_f17_field_cells_upper_bound:
    [.cases[].projection_temporary_f17_field_cells_upper_bound],
  projection_temporary_integer_count_slots:
    [.cases[].projection_temporary_integer_count_slots],
  primary_projection_histogram_integer_payload_bits:
    [.cases[].primary_projection_histogram_integer_payload_bits],
  reuse_projection_histogram_integer_payload_bits:
    [.cases[].reuse_projection_histogram_integer_payload_bits],
  primary_boundary_cyclotomic_integer_payload_bits:
    [.cases[].primary_boundary_cyclotomic_integer_payload_bits],
  reuse_boundary_cyclotomic_integer_payload_bits:
    [.cases[].reuse_boundary_cyclotomic_integer_payload_bits],
  primary_boundary_descriptor_bytes:
    [.cases[].primary_boundary_descriptor_bytes],
  reuse_boundary_descriptor_bytes:
    [.cases[].reuse_boundary_descriptor_bytes],
  primary_compile_candidate_row_masks_evaluated:
    [.cases[].primary_compile.candidate_row_masks_evaluated],
  primary_compile_selection_row_masks_evaluated:
    [.cases[].primary_compile.selection_row_masks_evaluated],
  primary_compile_latent_mask_field_tests:
    [.cases[].primary_compile.latent_mask_field_tests],
  reuse_compile_candidate_row_masks_evaluated:
    [.cases[].reuse_compile.candidate_row_masks_evaluated],
  reuse_compile_selection_row_masks_evaluated:
    [.cases[].reuse_compile.selection_row_masks_evaluated],
  reuse_compile_latent_mask_field_tests:
    [.cases[].reuse_compile.latent_mask_field_tests],
  all_primary_restored_exactly:
    ([.cases[].primary_restored_exactly]|all),
  all_reuse_restored_exactly:
    ([.cases[].reuse_restored_exactly]|all),
  all_fresh_restored_reuse_boundary_equal:
    ([.cases[].fresh_restored_reuse_boundary_equal]|all),
  all_primary_array_backing_preserved:
    ([.cases[].primary_same_array_backing[]]|all),
  all_reuse_array_backing_preserved:
    ([.cases[].reuse_same_array_backing[]]|all),
  restoration_generations:[.cases[].restoration_generation],
  restoration_leases:[.cases[].restoration_lease],
  retained_inverse_history_bytes:
    [.cases[].retained_inverse_history_bytes],
  baseline_reload_bytes:[.cases[].baseline_reload_bytes],
  controls,
  measured_obstruction,
  resource_law,
  no_smuggle_scope,
  matched_compact_classical,
  composition_scope,
  next_phase_owned_repair,
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
  tested_cases,
  dense_tested_case,
  exact_reference_f17_field_cells:
    [.cases[].exact_reference_f17_field_cells],
  exact_reference_trace_assignments:
    [.cases[].exact_reference_trace_assignments],
  dense_latent_axes_materialized:
    [.cases[].dense_latent_axes_materialized],
  dense_case:(
    .cases[]
    | select(.dense_latent_axes_materialized)
    | {
        ports,
        latents,
        dense_latent_axis_sizes_after_forward,
        dense_primary_boundary_complex_abs_error,
        dense_primary_restoration_max_abs_error,
        dense_reuse_boundary_complex_abs_error,
        dense_reuse_restoration_max_abs_error,
        verification_dense_public_fourier_transforms,
        verification_dense_logical_complex_cells
      }
  ),
  production_backend_imported,
  production_compiler_called,
  production_projection_called,
  public_descriptors_consumed,
  exact_reference_reexecutes_all_tested_cases,
  all_exact_boundaries_equal,
  all_exact_restorations_equal,
  independent_two_latent_axis_residency_parity,
  dense_expansion_is_verification_only,
  dense_logical_cells_are_not_process_peak,
  numpy_dense_oracle_process_peak_bounded,
  exact_reference_python_allocator_peak_bounded,
  maximum_verification_dense_logical_complex_cells,
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
  and .classification_candidate
    == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE"
  and .restoration_class == "EXACT_ALGEBRAIC_RESTORATION"
  and .carrier.fixed_public_ports == 4
  and .carrier.tested_latents == [1,2,3,4]
  and .carrier.dense_oracle_case == {ports:2,latents:2}
  and (.carrier.latent_projected_during_forward_composition|not)
  and (.carrier.dense_public_latent_phase_tensor_materialized|not)
  and (.carrier.relation_table_materialized|not)
  and .carrier.inverse_descriptors_retained == 0
  and ([.cases[]|select(.ports==4)|.latents] == [1,2,3,4])
  and ([.cases[]|select(.ports==4)|
    .resident_f17_field_cells] == [28,38,52,71])
  and ([.cases[]|select(.ports==4)|
    .latent_cubic_coefficient_cells] == [1,4,10,20])
  and ([.cases[]|select(.ports==4)|
    .interacting_latent_cubic_coefficient_cells] == [0,2,7,16])
  and ([.cases[]|select(.ports==4)|
    .latent_trace_assignments_streamed] == [17,289,4913,83521])
  and ([.cases[]|
    .projection_temporary_scalar_slots_upper_bound
      == (
        .projection_temporary_f17_field_cells_upper_bound
        + .projection_temporary_integer_count_slots
      )
  ]|all)
  and ([.cases[]|
    .projection_temporary_integer_count_slots == 33
  ]|all)
  and ([.cases[]|
    .primary_projection_histogram_integer_payload_bits > 0
    and .reuse_projection_histogram_integer_payload_bits > 0
    and .primary_boundary_cyclotomic_integer_payload_bits > 0
    and .reuse_boundary_cyclotomic_integer_payload_bits > 0
    and .primary_boundary_descriptor_bytes > 0
    and .reuse_boundary_descriptor_bytes > 0
  ]|all)
  and ([.cases[]|
    .primary_compile.candidate_row_masks_evaluated > 0
    and .primary_compile.selection_row_masks_evaluated > 0
    and .primary_compile.latent_mask_field_tests > 0
    and .reuse_compile.candidate_row_masks_evaluated > 0
    and .reuse_compile.selection_row_masks_evaluated > 0
    and .reuse_compile.latent_mask_field_tests > 0
  ]|all)
  and ([.cases[]|
    .resident_f17_field_cells
      == (
        .ports*.ports
        + .ports
        + .ports*.latents
        + (.latents*(.latents+1)/2)
        + .latents
        + (.latents*(.latents+1)*(.latents+2)/6)
        + 1
      )] | all)
  and ([.cases[]|
    .resident_fixed_width_payload_bits
      == 5*.resident_f17_field_cells+1] | all)
  and ([.cases[]|
    .primary_modules == .ports+3
    and .reuse_modules == .ports+3
    and .primary_latent_consuming_modules == .ports+2
    and .reuse_latent_consuming_modules == .ports+2] | all)
  and ([.cases[].primary_restored_exactly]|all)
  and ([.cases[].reuse_restored_exactly]|all)
  and ([.cases[].fresh_restored_reuse_boundary_equal]|all)
  and ([.cases[].same_logical_carrier_container]|all)
  and ([.cases[].primary_same_array_backing[]]|all)
  and ([.cases[].reuse_same_array_backing[]]|all)
  and ([.cases[].restoration_generation]|all(.==2))
  and ([.cases[].restoration_lease]|all(.==3))
  and ([.cases[].retained_inverse_history_bytes]|all(.==0))
  and ([.cases[].baseline_reload_bytes]|all(.==0))
  and ([.cases[].latent_trace_assignment_table_materialized|not]|all)
  and ([.cases[].primary_boundary|
    (.canonical_cyclotomic_coefficients|length)==16
    and (has("reduced_latent_polynomial")|not)]|all)
  and .controls.missing_inverse_rejected
  and .controls.wrong_inverse_rejected
  and .controls.reordered_inverse_rejected
  and .measured_obstruction.resident_signature_polynomial_in_latent_count
  and .measured_obstruction.generic_final_trace_exponential_in_latent_count
  and (.measured_obstruction
    .compact_multiple_latent_final_closure_established|not)
  and (.measured_obstruction
    .interaction_signature_is_current_primary_obstruction|not)
  and .measured_obstruction.final_trace_is_current_primary_obstruction
  and .resource_law.projection_streams_every_latent_assignment
  and (.resource_law.projection_assignment_table_materialized|not)
  and .resource_law.compiler_boundary_inputs == 0
  and .resource_law.compiler_answer_inputs == 0
  and .resource_law.compiler_trace_evaluations == 0
  and (.resource_law.compiler_candidate_list_materialized|not)
  and (.resource_law
    .compiler_consumer_witness_list_materialized|not)
  and (.no_smuggle_scope.machine_enforced_custody|not)
  and .no_smuggle_scope.scope
    == "PACKAGE_SERIALIZATION_ONLY_NOT_MACHINE_ENFORCED"
  and (.no_smuggle_scope.reduced_latent_polynomial_exposed|not)
  and (.no_smuggle_scope.latent_assignments_serialized|not)
  and .matched_compact_classical
    .identical_exact_multi_latent_coefficient_recurrence_exists
  and .matched_compact_classical.identical_generic_final_trace_work
  and (.matched_compact_classical
    .strongest_compact_classical_established|not)
  and .composition_scope.public_gaussian_shear_fourier_updates_native
  and .composition_scope.mixed_cubic_signature_preinitialized
  and (.composition_scope
    .mixed_cubic_coefficients_changed_by_public_modules|not)
  and (.composition_scope.native_cubic_interaction_updates_established|not)
  and (.composition_scope.arbitrary_non_gaussian_composition_established|not)
  and (.catvm_custody_established|not)
  and (.distinct_phase_resource_established|not)
  and (.computational_advantage|not)
  and (.small_wall_crossed|not)
  and (.catalytic_inference_established|not)
  and (.physical_waveform_execution|not)
  and (.replacement_of_physical_bits_with_pi|not)
  and (.unbounded_computation_established|not)
  and (.terminal|not)
' "$result" >/dev/null

jq -e '
  .result == "PASS"
  and .dense_tested_case == {ports:2,latents:2}
  and .all_exact_boundaries_equal
  and .all_exact_restorations_equal
  and .exact_reference_reexecutes_all_tested_cases
  and .independent_two_latent_axis_residency_parity
  and ([.cases[].exact_primary_boundary_equal]|all)
  and ([.cases[].exact_primary_restoration_equal]|all)
  and ([.cases[].exact_reuse_boundary_equal]|all)
  and ([.cases[].exact_reuse_restoration_equal]|all)
  and (.cases[]|select(.dense_latent_axes_materialized)|
    .dense_latent_axis_sizes_after_forward == [17,17]
    and .dense_primary_boundary_complex_abs_error <= 5.0e-11
    and .dense_primary_restoration_max_abs_error <= 5.0e-11
    and .dense_reuse_boundary_complex_abs_error <= 5.0e-11
    and .dense_reuse_restoration_max_abs_error <= 5.0e-11)
  and (.production_backend_imported|not)
  and (.production_compiler_called|not)
  and (.production_projection_called|not)
  and .public_descriptors_consumed
  and .dense_expansion_is_verification_only
  and .dense_logical_cells_are_not_process_peak
  and (.numpy_dense_oracle_process_peak_bounded|not)
  and (.exact_reference_python_allocator_peak_bounded|not)
  and (.distinct_phase_resource_established|not)
  and (.computational_advantage|not)
  and (.small_wall_crossed|not)
  and (.terminal|not)
' "$oracle_result" >/dev/null

if rg -n \
  'complex128|np\\.exp|np\\.indices|np\\.meshgrid|np\\.kron' \
  "$source_path" "$base_path"
then
  echo "accepted diagnostic source contains dense or complex construction" >&2
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
    exact_reference_cases:[
      {ports:4,latents:1},
      {ports:4,latents:2},
      {ports:4,latents:3},
      {ports:4,latents:4},
      {ports:2,latents:2}
    ],
    dense_two_latent_axis_parity_case:{ports:2,latents:2},
    result_sha256:$result_sha256,
    oracle_sha256:$oracle_sha256,
    source_sha256:$source_sha256,
    base_source_sha256:$base_source_sha256,
    oracle_source_sha256:$oracle_source_sha256,
    compact_multiple_latent_final_closure_established:false,
    distinct_phase_resource_established:false,
    computational_advantage:false,
    small_wall_crossed:false,
    terminal:false
  }' >"$evidence_dir/qualification.json"

echo "F17 interacting cubic-latent trace qualification passed"
