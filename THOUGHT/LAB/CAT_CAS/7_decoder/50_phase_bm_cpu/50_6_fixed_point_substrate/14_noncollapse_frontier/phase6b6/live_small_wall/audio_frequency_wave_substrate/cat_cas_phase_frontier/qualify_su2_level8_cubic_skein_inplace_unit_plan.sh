#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 DISK_BACKED_BUILD_DIRECTORY" >&2
    exit 2
fi
if [[ ! -d "$1" ]]; then
    echo "build directory must already exist" >&2
    exit 2
fi
build=$(realpath -e -- "$1")
case "$build" in
    /dev/shm/*|/run/shm/*)
        echo "RAM-backed build directory is forbidden" >&2
        exit 2
        ;;
esac
filesystem_type=$(findmnt -n -o FSTYPE -T "$build")
case "$filesystem_type" in
    tmpfs|ramfs)
        echo "RAM-backed build filesystem is forbidden" >&2
        exit 2
        ;;
esac

here=$(cd "$(dirname "$0")" && pwd)
production="$here/su2_level8_cubic_skein_inplace_unit_plan.py"
reference="$here/su2_level8_cubic_skein_inplace_unit_plan_separate_reference.py"
generated_reference="$build/SU2_LEVEL8_CUBIC_SKEIN_INPLACE_UNIT_PLAN_SEPARATE_REFERENCE.json"
generated_result="$build/SU2_LEVEL8_CUBIC_SKEIN_INPLACE_UNIT_PLAN_RESULTS.json"
mkdir -p "$build/tmp" "$build/xdg-cache"

TMPDIR="$build/tmp" XDG_CACHE_HOME="$build/xdg-cache" \
    PYTHONPATH="$here" PYTHONPYCACHEPREFIX="$build/pycache" \
    nice -n 10 ionice -c 2 -n 7 python3 "$reference" > "$generated_reference"
TMPDIR="$build/tmp" XDG_CACHE_HOME="$build/xdg-cache" \
    PYTHONPATH="$here" PYTHONPYCACHEPREFIX="$build/pycache" \
    nice -n 10 ionice -c 2 -n 7 \
    python3 "$production" "$generated_reference" > "$generated_result"

cmp "$generated_reference" \
    "$here/SU2_LEVEL8_CUBIC_SKEIN_INPLACE_UNIT_PLAN_SEPARATE_REFERENCE.json"
cmp "$generated_result" \
    "$here/SU2_LEVEL8_CUBIC_SKEIN_INPLACE_UNIT_PLAN_RESULTS.json"

jq -e '
    .result == "PASS_BOUNDED_EXACT_INPLACE_UNIT_PLAN_PERSISTING_RESOURCE_NO_GO"
    and .classification == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE"
    and .verification_level == "SEPARATE_REFERENCE_PARITY"
    and .restoration_classification == "EXACT_ALGEBRAIC_RESTORATION"
    and .plan_library.primitive_plan_count == 8
    and .plan_library.unit_plan_count == 7
    and .plan_library.conjugation_plan_count == 1
    and .plan_library.operation_counts.SWAP > 0
    and .plan_library.operation_counts.NEGATE > 0
    and .plan_library.operation_counts.SHEAR > 0
    and .plan_library.maximum_absolute_shear_coefficient > 0
    and .plan_library.retained_plan_integer_cells > 0
    and .plan_library.retained_plan_integer_payload_bits > 0
    and .plan_library.retained_plan_serialized_bytes > 0
    and (.plan_library.retained_inverse_plan_library | not)
    and .plan_library.compiler_matrices_built == 8
    and .plan_library.compiler_peak_dense_integer_matrix_cells == 512
    and .plan_library.compiler_peak_dense_integer_payload_bits > 0
    and .plan_library.compiler_peak_matrix_history_plan_and_public_descriptor_payload_bits
        > .plan_library.compiler_peak_dense_integer_payload_bits
    and .plan_library.compiler_public_unit_descriptor_payload_bits > 0
    and (.plan_library.compiler_scalar_arithmetic_live_payload_complete | not)
    and .plan_library.compiler_total_input_matrix_integer_payload_bits > 0
    and (.plan_library.public_compiler_input_commitment | length == 64)
    and .plan_library.compiler_input_matrices_streamed_one_at_a_time
    and .plan_library.compiler_input_matrix_field_multiplications == 112
    and .plan_library.compiler_determinant_multiply_subtracts > 0
    and .plan_library.compiler_row_integer_updates > 0
    and .plan_library.execution_dense_matrix_cells == 0
    and (.plan_library.answer_dependent_compilation | not)
    and (.plan_library.determinants | all(. == 1))
    and .compiler_controls.unit_determinant_is_one
    and .compiler_controls.basis_action_matches
    and .compiler_controls.inverse_restores_basis
    and .compiler_controls.corrupted_shear_changes_result
    and .compiler_controls.skipped_operation_changes_result
    and .compiler_controls.noncommuting_pair_witnessed
    and .compiler_controls.reordered_noncommuting_pair_changes_result
    and .compiler_controls.nonunimodular_determinant_detected
    and .compiler_controls.nonunimodular_matrix_rejected
    and .compiler_controls.nonintegral_matrix_rejected
    and .compiler_controls.public_input_mutation_changes_plan_commitment
    and .compiler_controls.norm_composite_basis_parity
    and .compiler_controls.inverse_norm_composite_basis_parity
    and .compiler_controls.unit_and_inverse_basis_parity_all
    and .compiler_controls.conjugation_basis_parity
    and .compiler_controls.nonbasis_plan_and_inverse_parity
    and .compiler_controls.wrong_inversion_flag_changes_result
    and .compiler_controls.wrong_basis_order_changes_result
    and .compiler_controls.off_by_one_repetition_changes_result
    and .compiler_controls.invalid_unit_index_rejected
    and .compiler_controls.invalid_action_kind_rejected
    and .compiler_controls.zero_exponent_rejected
    and .compiler_controls.negative_exponent_rejected
    and .compiler_controls.compilation_uses_only_public_basis_and_unit_parameters
    and .compiler_controls.public_compiler_input_commitment_reproduced
    and .compiler_controls.accepted_execution_plan_nodes_are_primitive_descriptors
    and .compiler_controls.execution_dense_matrix_cells == 0
    and .compiler_controls.deterministic_recompilation_commitment
    and .compiler_controls.verification_recompile_retained_plan_payload_bits
        == .plan_library.retained_plan_integer_payload_bits
    and .compiler_controls.verification_peak_composite_matrix_cells == 512
    and .compiler_controls.verification_peak_composite_matrix_payload_bits > 0
    and .compiler_controls.conjugation_plan_involutory
    and .static_public_resources.retained_plan_integer_payload_bits
        == .plan_library.retained_plan_integer_payload_bits
    and .static_public_resources.predecessor_descriptor_field_cells == 28
    and .static_public_resources.predecessor_descriptor_field_payload_bits > 0
    and .static_public_resources.predecessor_descriptor_parameter_integer_cells == 7
    and .static_public_resources.predecessor_descriptor_parameter_payload_bits > 0
    and .static_public_resources.total_logical_payload_bits
        == (.static_public_resources.retained_plan_integer_payload_bits
            + .static_public_resources.predecessor_descriptor_field_payload_bits
            + .static_public_resources.predecessor_descriptor_parameter_payload_bits)
    and (.cases | length == 7)
    and (.cases | all(
        .canonical_post_restoration_state_exact
        and .same_residual_backing
        and .same_unit_ledger_backing
        and .same_scratch_backing
        and (.baseline_reload_used | not)
        and .work.inplace_factor_action_calls > 0
        and .work.inplace_factor_action_repetitions > 0
        and .work.inplace_plan_reference_reads > 0
        and .work.inplace_plan_operation_steps > 0
        and .work.inplace_plan_shears > 0
        and .work.inplace_plan_swaps > 0
        and .work.inplace_plan_negations > 0
        and .work.inplace_coordinate_accumulators_materialized
            == .work.inplace_coordinate_accumulators_released
        and .work.maximum_inplace_coordinate_payload_bits > 0
        and .work.maximum_inplace_transient_rational_payload_bits > 0
        and .work.maximum_inplace_transient_rational_cells <= 2
        and .work.maximum_inplace_action_descriptor_integer_cells > 0
        and .work.maximum_inplace_plan_descriptor_refs == 4
        and .work.execution_dense_matrix_cells == 0
        and .work.unit_power_field_multiplications == 0
        and .work.direct_unit_action_calls == 0
        and .work.direct_unit_action_steps == 0
        and .work.direct_unit_action_square_multiplications == 0
        and .work.direct_unit_action_total_absolute_exponent_mass == 0
        and .work.direct_unit_action_maximum_absolute_exponent == 0
        and .work.direct_ledger_norm_action_calls == 0
        and .work.direct_ledger_scale_action_calls == 0
        and .work.direct_trace_weight_action_calls == 0
        and .work.direct_candidate_multiplier_action_calls == 0
        and .work.direct_selected_multiplier_action_calls == 0
        and .work.standalone_power_results_materialized == 0
    ))
    and .reuse.primary.restoration_generation == 1
    and .reuse.reuse.restoration_generation == 2
    and .reuse.fresh_reuse.restoration_generation == 1
    and .reuse.restoration_generation_after_reuse == 2
    and .reuse.fresh_restored_reuse_boundary_agreement
    and .reuse.fresh_restored_reuse_state_agreement
    and .predecessor_comparison.m226_semantic_parity
    and .predecessor_comparison.m226_primary_dynamic_live_payload_bits
        > .predecessor_comparison.m227_primary_dynamic_live_payload_bits
    and .predecessor_comparison.dynamic_reduction_bits
        == (.predecessor_comparison.m226_primary_dynamic_live_payload_bits
            - .predecessor_comparison.m227_primary_dynamic_live_payload_bits)
    and .predecessor_comparison.m227_retained_predecessor_descriptor_payload_bits
        == (.static_public_resources.predecessor_descriptor_field_payload_bits
            + .static_public_resources.predecessor_descriptor_parameter_payload_bits)
    and .predecessor_comparison.m227_total_static_public_resource_payload_bits
        == .static_public_resources.total_logical_payload_bits
    and .predecessor_comparison.m227_primary_dynamic_plus_static_public_payload_bits
        == (.predecessor_comparison.m227_primary_dynamic_live_payload_bits
            + .predecessor_comparison.m227_total_static_public_resource_payload_bits)
    and .predecessor_comparison.m227_plan_operation_steps
        > .predecessor_comparison.m226_direct_action_steps
    and (.lifecycle_law.every_case_dynamic_plus_static_public_below_matched_raw | not)
    and .lifecycle_law.one_owned_coordinate_accumulator
    and .lifecycle_law.execution_dense_matrix_cells_zero
    and .lifecycle_law.whole_old_and_new_field_product_coexistence_eliminated
    and .lifecycle_law.source_complete_scope
        == "INPLACE_PUBLIC_PLAN_OPERATION_INTERVALS_ONLY"
    and (.lifecycle_law.projected_boundary_retention_during_inverse_instrumented | not)
    and (.lifecycle_law.whole_transaction_live_payload_complete | not)
    and .separate_reference.imports_m227_production == false
    and .separate_reference.imports_m226_production == false
    and .separate_reference.uses_prior_standalone_m226_reference_substrate
    and .separate_reference.plan_compiler_and_resource_parity
    and .separate_reference.case_state_boundary_balance_resource_restoration_parity
    and .separate_reference.reuse_parity
    and .matched_classical_baselines.strongest_compact
        == "IDENTICAL_DETERMINISTIC_GL16Z_PLAN_COMPILER_AND_INPLACE_RATIONAL_COORDINATE_RECURRENCE"
    and (.matched_classical_baselines.phase_specific_reduction | not)
    and (.matched_classical_baselines.computational_advantage | not)
    and .resource_law.compiler_matrices_and_work_counted
    and .resource_law.retained_public_plan_library_counted
    and .resource_law.retained_predecessor_public_unit_descriptor_counted
    and .resource_law.inverse_plan_library_eliminated
    and .resource_law.compiler_input_matrices_streamed_one_at_a_time
    and .resource_law.dynamic_and_inclusive_payloads_reported_separately
    and .resource_law.every_plan_operation_and_rational_temporary_counted
    and .resource_law.verification_compiler_reexecution_excluded_from_accepted_path
    and (.resource_law.warm_runtime_measured | not)
    and (.resource_law.whole_transaction_live_payload_complete | not)
    and (.claim_limits.catvm_custody | not)
    and (.claim_limits.distinct_phase_resource_established | not)
    and (.claim_limits.computational_advantage | not)
    and (.claim_limits.small_wall_crossed | not)
    and (.claim_limits.physical_waveform_execution | not)
    and (.claim_limits.physical_bit_replacement | not)
    and (.claim_limits.catalytic_inference_established | not)
    and (.claim_limits.unbounded_computation_established | not)
    and (.terminal | not)
' "$generated_result" >/dev/null

if rg -q '^(from|import) su2_level8_cubic_skein_inplace_unit_plan([[:space:]]|$)' \
    "$reference"; then
    echo "separate reference imports M227 production" >&2
    exit 2
fi
if rg -q '^(from|import) su2_level8_cubic_skein_direct_unit_action([[:space:]]|$)' \
    "$reference"; then
    echo "separate reference imports M226 production" >&2
    exit 2
fi
rg -q 'coordinates\[operation.target\] = updated' "$production"
rg -q 'coordinates\[operation.target\] = updated' "$reference"
rg -q 'compiler_peak_dense_integer_payload_bits' "$production"
rg -q 'compiler_peak_dense_integer_payload_bits' "$reference"
rg -q 'whole_transaction_live_payload_complete": False' "$production"

echo "QUALIFIED_SU2_LEVEL8_CUBIC_SKEIN_INPLACE_UNIT_PLAN_RESOURCE_NO_GO_STRICT_SCOPE"
