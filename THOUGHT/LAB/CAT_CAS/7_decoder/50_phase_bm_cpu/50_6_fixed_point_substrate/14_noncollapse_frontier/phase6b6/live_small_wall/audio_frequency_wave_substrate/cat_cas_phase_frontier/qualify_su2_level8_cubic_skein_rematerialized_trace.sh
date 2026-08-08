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
production="$here/su2_level8_cubic_skein_rematerialized_trace.py"
reference="$here/su2_level8_cubic_skein_rematerialized_trace_separate_reference.py"
generated_reference="$build/SU2_LEVEL8_CUBIC_SKEIN_REMATERIALIZED_TRACE_SEPARATE_REFERENCE.json"
generated_result="$build/SU2_LEVEL8_CUBIC_SKEIN_REMATERIALIZED_TRACE_RESULTS.json"

PYTHONPATH="$here" PYTHONPYCACHEPREFIX="$build/pycache" \
    nice -n 10 ionice -c 2 -n 7 python3 "$reference" > "$generated_reference"
PYTHONPATH="$here" PYTHONPYCACHEPREFIX="$build/pycache" \
    nice -n 10 ionice -c 2 -n 7 \
    python3 "$production" "$generated_reference" > "$generated_result"

cmp "$generated_reference" \
    "$here/SU2_LEVEL8_CUBIC_SKEIN_REMATERIALIZED_TRACE_SEPARATE_REFERENCE.json"
cmp "$generated_result" \
    "$here/SU2_LEVEL8_CUBIC_SKEIN_REMATERIALIZED_TRACE_RESULTS.json"

jq -e '
    .result == "PASS_BOUNDED_EXACT_REMATERIALIZED_TRACE_PERSISTING_HEIGHT_NO_GO"
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
    and (.mechanism.aggregate_norm_fields_materialized | not)
    and (.mechanism.raw_or_candidate_vectors_materialized | not)
    and .lifecycle_law.direct_trace_residual_cell_product_materialization_eliminated
    and .lifecycle_law.lifecycle_scale_conjugates_eliminated
    and .lifecycle_law.all_direct_trace_caller_field_scalars_released
    and .lifecycle_law.every_declared_case_above_matched_raw
    and (.lifecycle_law.all_declared_depth_above_one_smaller_than_matched_raw | not)
    and .lifecycle_law.dominant_rematerialized_contexts == ["UNIT_POWER_ACCUMULATE"]
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
        and .work.maximum_declared_live_payload_bits
            > .matched_raw_recurrence.maximum_declared_live_payload_bits
    ))
    and .separate_reference.imports_m224_production == false
    and .separate_reference.imports_m223_production == false
    and .separate_reference.imports_m222_production == false
    and .separate_reference.imports_m221_production == false
    and .separate_reference.uses_prior_standalone_m222_reference_substrate
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
        | $p.m223_primary_declared_live_payload_bits
            > $p.m224_primary_declared_live_payload_bits
        and $p.reduction_bits
            == ($p.m223_primary_declared_live_payload_bits
                - $p.m224_primary_declared_live_payload_bits)
        and $p.m224_primary_declared_live_payload_bits
            == ([.cases[]
                | select(
                    .strands == 4 and .rounds == 4 and .family == 0
                )
                | .work.maximum_declared_live_payload_bits] | first)
        and (.claim | contains(
            "FROM\($p.m223_primary_declared_live_payload_bits)_TO"
            + "\($p.m224_primary_declared_live_payload_bits)_BITS"
        ))
        and (.claim | contains("ELIMINATES_ACCEPTED_PATH")))
    and .controls.wrong_unit_ledger_changes_represented_state
    and .controls.reordered_inverse_rejected
    and (.controls.intermediate_actual_vector_projected | not)
    and (.controls.snapshot_command_available | not)
    and (.claim_limits.distinct_phase_resource_established | not)
    and (.claim_limits.computational_advantage | not)
    and (.claim_limits.small_wall_crossed | not)
    and (.terminal | not)
' "$generated_result" >/dev/null

if rg -q '^(from|import) su2_level8_cubic_skein_rematerialized_trace([[:space:]]|$)' \
    "$reference"; then
    echo "separate reference imports M224 production" >&2
    exit 2
fi

PYTHONPYCACHEPREFIX="$build/pycache" python3 -m py_compile \
    "$production" "$reference"

echo "QUALIFIED_SU2_LEVEL8_CUBIC_SKEIN_REMATERIALIZED_UNIT_POWER_NO_GO_STRICT_SCOPE"
