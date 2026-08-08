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
production="$here/su2_level8_cubic_skein_transpose_trace_functional.py"
reference="$here/su2_level8_cubic_skein_transpose_trace_functional_separate_reference.py"
generated_reference="$build/SU2_LEVEL8_CUBIC_SKEIN_TRANSPOSE_TRACE_FUNCTIONAL_SEPARATE_REFERENCE.json"
generated_result="$build/SU2_LEVEL8_CUBIC_SKEIN_TRANSPOSE_TRACE_FUNCTIONAL_RESULTS.json"
mkdir -p "$build/tmp" "$build/xdg-cache"

TMPDIR="$build/tmp" XDG_CACHE_HOME="$build/xdg-cache" \
    PYTHONPATH="$here" PYTHONPYCACHEPREFIX="$build/pycache" \
    nice -n 10 ionice -c 2 -n 7 python3 "$reference" > "$generated_reference"
TMPDIR="$build/tmp" XDG_CACHE_HOME="$build/xdg-cache" \
    PYTHONPATH="$here" PYTHONPYCACHEPREFIX="$build/pycache" \
    nice -n 10 ionice -c 2 -n 7 \
    python3 "$production" "$generated_reference" > "$generated_result"

cmp "$generated_reference" \
    "$here/SU2_LEVEL8_CUBIC_SKEIN_TRANSPOSE_TRACE_FUNCTIONAL_SEPARATE_REFERENCE.json"
cmp "$generated_result" \
    "$here/SU2_LEVEL8_CUBIC_SKEIN_TRANSPOSE_TRACE_FUNCTIONAL_RESULTS.json"

jq -e '
    .result == "PASS_BOUNDED_EXACT_TRANSPOSE_TRACE_FUNCTIONAL_PERSISTING_RESOURCE_NO_GO"
    and .classification == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE"
    and .verification_level == "SEPARATE_REFERENCE_PARITY"
    and .restoration_classification == "EXACT_ALGEBRAIC_RESTORATION"
    and .controls.transpose_dot_identity_all_public_plans
    and .controls.inverse_transpose_restores_all_public_plans
    and .controls.composite_reverse_order_matches
    and .controls.wrong_composite_order_changes_result
    and .controls.hermitian_functional_matches_field_trace_basis
    and .controls.functional_coordinate_count_is_field_degree
    and .controls.public_plan_library_reused_without_answer_compilation
    and .plan_library.primitive_plan_count == 8
    and (.plan_library.retained_inverse_plan_library | not)
    and .plan_library.execution_dense_matrix_cells == 0
    and .static_public_resources.total_logical_payload_bits > 0
    and (.cases | length == 7)
    and (.cases | all(
        .canonical_post_restoration_state_exact
        and .same_residual_backing
        and .same_unit_ledger_backing
        and .same_scratch_backing
        and (.baseline_reload_used | not)
        and .work.trace_functional_evaluations > 0
        and .work.trace_functional_carrier_cells_scanned > 0
        and .work.trace_functional_coordinates_materialized > 0
        and (.work.trace_functional_coordinates_materialized
            == .work.trace_functional_coordinates_released)
        and .work.trace_functional_nonzero_ramanujan_terms > 0
        and .work.trace_functional_zero_ramanujan_terms_skipped > 0
        and .work.trace_functional_rational_multiplications > 0
        and .work.trace_functional_rational_additions > 0
        and .work.transpose_factor_action_calls > 0
        and .work.transpose_factor_action_repetitions > 0
        and .work.transpose_plan_reference_reads > 0
        and .work.transpose_plan_operation_steps > 0
        and .work.transpose_plan_shears > 0
        and .work.transpose_plan_swaps > 0
        and .work.transpose_plan_negations > 0
        and .work.maximum_trace_functional_payload_bits > 0
        and .work.maximum_trace_functional_transient_payload_bits > 0
        and .work.maximum_trace_functional_transient_rational_cells == 3
        and .work.maximum_transpose_plan_descriptor_refs == 4
        and .work.maximum_transpose_action_descriptor_integer_cells > 0
        and .work.maximum_transpose_absolute_shear_coefficient > 0
        and .work.materialized_trace_weight_field_cells == 0
        and .work.direct_trace_weight_rematerializations == 0
        and .work.direct_trace_energy_evaluations == 0
        and .work.direct_trace_cells_scanned == 0
        and .work.execution_dense_matrix_cells == 0
        and .work.standalone_power_results_materialized == 0
    ))
    and .reuse.primary.restoration_generation == 1
    and .reuse.reuse.restoration_generation == 2
    and .reuse.fresh_reuse.restoration_generation == 1
    and .reuse.restoration_generation_after_reuse == 2
    and .reuse.fresh_restored_reuse_boundary_agreement
    and .reuse.fresh_restored_reuse_state_agreement
    and .predecessor_comparison.m227_semantic_parity
    and .predecessor_comparison.m227_materialized_trace_weight_rematerializations > 0
    and .predecessor_comparison.m228_materialized_trace_weight_field_cells == 0
    and .predecessor_comparison.m228_trace_functional_coordinates_materialized > 0
    and .predecessor_comparison.m228_transpose_plan_operation_steps > 0
    and .predecessor_comparison.m228_primary_dynamic_live_payload_bits
        > .predecessor_comparison.matched_raw_payload_bits
    and .predecessor_comparison.m228_primary_dynamic_plus_static_public_payload_bits
        == (.predecessor_comparison.m228_primary_dynamic_live_payload_bits
            + .static_public_resources.total_logical_payload_bits)
    and (.lifecycle_law.every_case_dynamic_plus_static_public_below_matched_raw | not)
    and .lifecycle_law.materialized_trace_weight_field_cells == 0
    and .lifecycle_law.functional_coordinates_owned_and_released
    and .lifecycle_law.source_complete_scope
        == "STREAMED_FUNCTIONAL_ALL_TERM_AND_TRANSPOSE_PLAN_INTERVALS_ONLY"
    and .lifecycle_law.zero_ramanujan_skip_intervals_instrumented
    and (.lifecycle_law.projected_boundary_retention_during_inverse_instrumented | not)
    and (.lifecycle_law.whole_transaction_live_payload_complete | not)
    and (.separate_reference.imports_m228_production | not)
    and (.separate_reference.imports_m227_production | not)
    and .separate_reference.uses_prior_standalone_m227_reference_substrate
    and .separate_reference.transpose_formula_controls_parity
    and .separate_reference.case_resource_restoration_reuse_parity
    and .matched_classical_baselines.strongest_compact
        == "IDENTICAL_STREAMED16_COORDINATE_HERMITIAN_FUNCTIONAL_AND_TRANSPOSE_GL16Z_PLAN_RECURRENCE"
    and (.matched_classical_baselines.phase_specific_reduction | not)
    and (.matched_classical_baselines.computational_advantage | not)
    and .resource_law.complete_functional_vector_counted
    and .resource_law.every_nonzero_functional_update_transient_counted
    and .resource_law.every_transpose_plan_operation_and_transient_counted
    and .resource_law.retained_public_plan_and_descriptor_counted
    and .resource_law.compiler_resources_inherited_from_m227
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

if rg -q '^(from|import) su2_level8_cubic_skein_transpose_trace_functional([[:space:]]|$)' \
    "$reference"; then
    echo "separate reference imports M228 production" >&2
    exit 2
fi
if rg -q '^(from|import) su2_level8_cubic_skein_inplace_unit_plan([[:space:]]|$)' \
    "$reference"; then
    echo "separate reference imports M227 production" >&2
    exit 2
fi
rg -Fq 'for reference in reversed(refs)' "$production"
rg -Fq 'for action_index in range(len(actions) - 1, -1, -1)' "$production"
rg -Fq 'for ref_index in range(len(refs) - 1, -1, -1)' "$production"
rg -Fq 'values[operation.source] += coefficient * values[operation.target]' "$production"
rg -q 'functional\[target\] = updated' "$production"
rg -Fq 'for reference in reversed(refs)' "$reference"
rg -Fq 'for action_index in range(len(actions) - 1, -1, -1)' "$reference"
rg -Fq 'for ref_index in range(len(refs) - 1, -1, -1)' "$reference"
rg -Fq 'values[operation.source] += coefficient * values[operation.target]' "$reference"
rg -q 'functional\[target\] = updated' "$reference"
if sed -n '/^def transpose_unit_action_energy/,/^def apply_ref_plain/p' "$production" \
    | rg -q 'coordinate_accumulator|braid\.K\('; then
    echo "accepted transpose energy rematerializes a cyclotomic weight" >&2
    exit 2
fi
if sed -n '/^def transpose_unit_action_energy/,/^def apply_ref_plain/p' "$reference" \
    | rg -q 'coordinate_accumulator|base\.E\('; then
    echo "standalone transpose energy rematerializes a field weight" >&2
    exit 2
fi

echo "QUALIFIED_SU2_LEVEL8_CUBIC_SKEIN_TRANSPOSE_TRACE_FUNCTIONAL_RESOURCE_NO_GO_STRICT_SCOPE"
