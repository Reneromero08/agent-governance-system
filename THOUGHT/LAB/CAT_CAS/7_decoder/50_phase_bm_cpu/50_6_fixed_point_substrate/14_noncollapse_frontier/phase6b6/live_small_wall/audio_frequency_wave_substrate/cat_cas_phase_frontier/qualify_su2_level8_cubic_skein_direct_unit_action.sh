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
production="$here/su2_level8_cubic_skein_direct_unit_action.py"
reference="$here/su2_level8_cubic_skein_direct_unit_action_separate_reference.py"
generated_reference="$build/SU2_LEVEL8_CUBIC_SKEIN_DIRECT_UNIT_ACTION_SEPARATE_REFERENCE.json"
generated_result="$build/SU2_LEVEL8_CUBIC_SKEIN_DIRECT_UNIT_ACTION_RESULTS.json"
mkdir -p "$build/tmp" "$build/xdg-cache"

TMPDIR="$build/tmp" XDG_CACHE_HOME="$build/xdg-cache" \
    PYTHONPATH="$here" PYTHONPYCACHEPREFIX="$build/pycache" \
    nice -n 10 ionice -c 2 -n 7 python3 "$reference" > "$generated_reference"
TMPDIR="$build/tmp" XDG_CACHE_HOME="$build/xdg-cache" \
    PYTHONPATH="$here" PYTHONPYCACHEPREFIX="$build/pycache" \
    nice -n 10 ionice -c 2 -n 7 \
    python3 "$production" "$generated_reference" > "$generated_result"

cmp "$generated_reference" \
    "$here/SU2_LEVEL8_CUBIC_SKEIN_DIRECT_UNIT_ACTION_SEPARATE_REFERENCE.json"
cmp "$generated_result" \
    "$here/SU2_LEVEL8_CUBIC_SKEIN_DIRECT_UNIT_ACTION_RESULTS.json"

jq -e '
    .result == "PASS_BOUNDED_EXACT_DIRECT_UNIT_ACTION_PERSISTING_HEIGHT_NO_GO"
    and .classification == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE"
    and .verification_level == "SEPARATE_REFERENCE_PARITY"
    and .restoration_classification == "EXACT_ALGEBRAIC_RESTORATION"
    and (.mechanism.direct_trace_residual_cell_conjugates_materialized | not)
    and (.mechanism.direct_trace_norm_cell_field_products_materialized | not)
    and (.mechanism.direct_trace_weighted_cell_field_products_materialized | not)
    and (.mechanism.lifecycle_scale_conjugates_materialized | not)
    and .mechanism.lifecycle_scale_conjugates_per_balance == 0
    and .mechanism.maximum_direct_trace_caller_scalar_cells == 0
    and .mechanism.line_weights_rematerialized_from_resident_ledger
    and .mechanism.selected_multipliers_rematerialized_from_resident_ledger
    and .mechanism.dynamic_public_unit_power_realized_by_direct_fixed_descriptor_action
    and (.mechanism.standalone_dynamic_power_results_materialized | not)
    and .mechanism.terminal_outer_factor_multiplications_eliminated
    and (.mechanism.dynamic_binary_square_factors_materialized | not)
    and .mechanism.retained_power_table_cells == 0
    and .mechanism.retained_power_cache_cells == 0
    and .mechanism.maximum_fixed_public_descriptor_scalar_cells == 1
    and .mechanism.compiled_public_unit_descriptor_field_cells == 28
    and .mechanism.compiled_public_unit_descriptor_field_payload_bits > 0
    and .mechanism.compiled_public_unit_descriptor_parameter_integer_cells == 7
    and .mechanism.compiled_public_unit_descriptor_parameter_payload_bits > 0
    and .mechanism.projection_scale_built_before_final_boundary
    and (.mechanism.aggregate_norm_fields_materialized | not)
    and (.mechanism.raw_or_candidate_vectors_materialized | not)
    and .lifecycle_law.direct_trace_residual_cell_product_materialization_eliminated
    and .lifecycle_law.lifecycle_scale_conjugates_eliminated
    and .lifecycle_law.all_direct_trace_caller_field_scalars_released
    and .lifecycle_law.all_accepted_unit_powers_direct
    and .lifecycle_law.source_complete_scope == "DIRECT_REPEATED_PUBLIC_UNIT_ACTION_INTERVALS_ONLY"
    and (.lifecycle_law.projected_boundary_retention_during_inverse_instrumented | not)
    and (.lifecycle_law.whole_transaction_live_payload_complete | not)
    and .lifecycle_law.every_declared_case_above_matched_raw
    and (.lifecycle_law.all_declared_depth_above_one_smaller_than_matched_raw | not)
    and .lifecycle_law.dominant_rematerialized_contexts == ["DIRECT_PUBLIC_UNIT_TRACE_WEIGHT_STEP"]
    and (.cases | all(
        .work.weighted_cell_field_products_materialized == 0
        and .work.norm_cell_field_products_materialized == 0
        and .work.direct_trace_conjugate_field_values_materialized == 0
        and .work.lifecycle_scale_conjugate_field_values_materialized == 0
        and .work.maximum_direct_trace_caller_scalar_cells == 0
        and .work.ledger_norm_rematerializations > 0
        and .work.ledger_scale_rematerializations > 0
        and .work.direct_trace_weight_rematerializations > 0
        and .work.selected_multiplier_rematerializations > 0
        and .work.direct_unit_action_calls > 0
        and .work.direct_unit_action_calls == (
            .work.direct_ledger_norm_action_calls
            + .work.direct_ledger_scale_action_calls
            + .work.direct_trace_weight_action_calls
            + .work.direct_candidate_multiplier_action_calls
            + .work.direct_selected_multiplier_action_calls
        )
        and .work.unit_power_field_multiplications == (
            .work.direct_unit_action_steps
        )
        and .work.direct_unit_action_steps
            == .work.direct_unit_action_total_absolute_exponent_mass
        and .work.direct_unit_action_square_multiplications == 0
        and .work.direct_unit_action_maximum_absolute_exponent > 0
        and .work.maximum_fixed_public_descriptor_scalar_cells == 1
        and .work.standalone_power_results_materialized == 0
        and .work.rematerialization_field_multiplications == 0
        and .work.ledger_scale_field_multiplications == 0
        and .work.maximum_declared_live_payload_bits
            > .matched_raw_recurrence.maximum_declared_live_payload_bits
    ))
    and .separate_reference.imports_m226_production == false
    and .separate_reference.imports_m225_production == false
    and .separate_reference.imports_m224_production == false
    and .separate_reference.imports_m223_production == false
    and .separate_reference.imports_m222_production == false
    and .separate_reference.imports_m221_production == false
    and .separate_reference.uses_prior_standalone_m222_reference_substrate
    and .separate_reference.public_unit_descriptor_parity
    and .separate_reference.public_unit_descriptor.field_cells == 28
    and .separate_reference.public_unit_descriptor.field_payload_bits == .mechanism.compiled_public_unit_descriptor_field_payload_bits
    and .separate_reference.public_unit_descriptor.parameter_integer_cells == 7
    and .separate_reference.public_unit_descriptor.parameter_payload_bits == .mechanism.compiled_public_unit_descriptor_parameter_payload_bits
    and .separate_reference.reuse_parity
    and (.reuse as $reuse | (["primary", "reuse", "fresh_reuse"] | all(
        . as $section
        | $reuse[$section].canonical_post_restoration_state_exact
        and $reuse[$section].same_residual_backing
        and $reuse[$section].same_unit_ledger_backing
        and $reuse[$section].same_scratch_backing
        and ($reuse[$section].baseline_reload_used | not)
    )))
    and .reuse.primary.restoration_generation == 1
    and .reuse.reuse.restoration_generation == 2
    and .reuse.fresh_reuse.restoration_generation == 1
    and .reuse.restoration_generation_after_reuse == 2
    and .reuse.fresh_restored_reuse_boundary_agreement
    and .reuse.fresh_restored_reuse_state_agreement
    and (.predecessor_comparison as $p
        | $p.m225_semantic_parity
        and $p.m225_primary_declared_live_payload_bits
            > $p.m226_primary_declared_live_payload_bits
        and $p.reduction_bits
            == ($p.m225_primary_declared_live_payload_bits
                - $p.m226_primary_declared_live_payload_bits)
        and $p.m225_primary_binary_power_multiplications > 0
        and $p.m225_primary_binary_square_multiplications > 0
        and $p.m226_primary_terminal_factor_multiplications == 0
        and $p.m226_primary_nonzero_direct_factor_applications > 0
        and $p.m226_primary_direct_action_steps
            == $p.m226_primary_total_absolute_exponent_mass
        and $p.m226_primary_maximum_absolute_exponent > 0
        and $p.m226_primary_declared_live_payload_bits
            == ([.cases[]
                | select(
                    .strands == 4 and .rounds == 4 and .family == 0
                )
                | .work.maximum_declared_live_payload_bits] | first)
        and (.claim | contains(
            "FROM\($p.m225_primary_declared_live_payload_bits)_TO"
            + "\($p.m226_primary_declared_live_payload_bits)_BITS"
        ))
        and (.claim | contains(
            "ELIMINATES_DYNAMIC_BINARY_SQUARE_FACTORS_POWER_TABLES_AND_CACHES"
        )))
    and .controls.wrong_unit_ledger_changes_represented_state
    and .controls.reordered_inverse_rejected
    and .controls.direct_action.correct_three_step_action_matches
    and .controls.direct_action.wrong_base_changes_result
    and .controls.direct_action.wrong_sign_changes_result
    and .controls.direct_action.skipped_step_changes_result
    and .controls.direct_action.extra_step_changes_result
    and .controls.direct_action.zero_exponent_is_identity
    and .controls.direct_action.negative_exponent_rejected
    and .controls.direct_action.correct_step_count == 3
    and .controls.direct_action.correct_total_absolute_exponent_mass == 3
    and .controls.direct_action.correct_square_multiplications == 0
    and .controls.direct_action.zero_exponent_calls == 0
    and .controls.direct_action.zero_exponent_steps == 0
    and .controls.direct_action.verification_only_expected_field_multiplications == 2
    and (.controls.intermediate_actual_vector_projected | not)
    and (.controls.snapshot_command_available | not)
    and (.claim_limits.distinct_phase_resource_established | not)
    and (.claim_limits.computational_advantage | not)
    and (.claim_limits.small_wall_crossed | not)
    and (.resource_law.whole_transaction_live_payload_complete | not)
    and .resource_law.compiled_public_unit_descriptor_table_counted
    and (.terminal | not)
' "$generated_result" >/dev/null

if rg -q '^(from|import) su2_level8_cubic_skein_direct_unit_action([[:space:]]|$)' \
    "$reference"; then
    echo "separate reference imports M226 production" >&2
    exit 2
fi

if rg -q '^(from|import) su2_level8_cubic_skein_rematerialized_trace([[:space:]]|$)' \
    "$reference"; then
    echo "separate reference imports M224 production" >&2
    exit 2
fi

if rg -q '^(from|import) su2_level8_cubic_skein_fused_unit_power([[:space:]]|$)' \
    "$reference"; then
    echo "separate reference imports M225 production" >&2
    exit 2
fi

rg -q 'm221\.counted_power = reject_standalone_power' "$production"
rg -q 'prior\.counted_power = reject_standalone_power' "$reference"
rg -q 'for step in range\(exponent\):' "$production"
rg -q 'for step in range\(exponent\):' "$reference"

PYTHONPYCACHEPREFIX="$build/pycache" python3 -m py_compile \
    "$production" "$reference"

echo "QUALIFIED_SU2_LEVEL8_CUBIC_SKEIN_DIRECT_UNIT_ACTION_NO_GO_STRICT_SCOPE"
