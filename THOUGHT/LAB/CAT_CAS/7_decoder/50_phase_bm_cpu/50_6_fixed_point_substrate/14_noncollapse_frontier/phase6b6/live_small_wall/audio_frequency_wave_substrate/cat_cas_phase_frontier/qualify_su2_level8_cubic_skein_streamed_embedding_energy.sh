#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 DISK_BACKED_BUILD_DIRECTORY" >&2
    exit 2
fi

build=$1
case "$build" in
    /dev/shm/*|/run/shm/*)
        echo "RAM-backed build directory is forbidden" >&2
        exit 2
        ;;
esac

here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$build"

production="$here/su2_level8_cubic_skein_streamed_embedding_energy.py"
reference="$here/su2_level8_cubic_skein_streamed_embedding_energy_separate_reference.py"
generated_reference="$build/SU2_LEVEL8_CUBIC_SKEIN_STREAMED_EMBEDDING_ENERGY_SEPARATE_REFERENCE.json"
generated_result="$build/SU2_LEVEL8_CUBIC_SKEIN_STREAMED_EMBEDDING_ENERGY_RESULTS.json"

PYTHONPATH="$here" PYTHONPYCACHEPREFIX="$build/pycache" \
    nice -n 10 ionice -c 2 -n 7 \
    python3 "$reference" > "$generated_reference"
PYTHONPATH="$here" PYTHONPYCACHEPREFIX="$build/pycache" \
    nice -n 10 ionice -c 2 -n 7 \
    python3 "$production" "$generated_reference" > "$generated_result"

cmp "$generated_reference" \
    "$here/SU2_LEVEL8_CUBIC_SKEIN_STREAMED_EMBEDDING_ENERGY_SEPARATE_REFERENCE.json"
cmp "$generated_result" \
    "$here/SU2_LEVEL8_CUBIC_SKEIN_STREAMED_EMBEDDING_ENERGY_RESULTS.json"

jq -e '
    .result == "PASS_BOUNDED_EXACT_STREAMED_EMBEDDING_ENERGY_PERSISTING_HEIGHT_NO_GO"
    and .classification == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE"
    and .verification_level == "SEPARATE_REFERENCE_PARITY"
    and .restoration_classification == "EXACT_ALGEBRAIC_RESTORATION"
    and (.mechanism.aggregate_residual_norm_field_materialized | not)
    and (.mechanism.aggregate_actual_norm_field_materialized | not)
    and (.mechanism.aggregate_candidate_norm_field_materialized | not)
    and (.mechanism.raw_actual_vectors_materialized | not)
    and (.mechanism.candidate_residual_vectors_materialized | not)
    and .lifecycle_law.aggregate_norm_field_materialization_eliminated
    and .lifecycle_law.raw_and_candidate_vector_materialization_eliminated
    and .lifecycle_law.every_declared_case_above_matched_raw
    and (.lifecycle_law.all_declared_depth_above_one_smaller_than_matched_raw | not)
    and .lifecycle_law.dominant_streamed_energy_contexts == ["STREAMED_TRACE_LINE_WEIGHTED_CELL"]
    and (.cases | all(
        .work.aggregate_residual_norm_fields_materialized == 0
        and .work.aggregate_actual_norm_fields_materialized == 0
        and .work.aggregate_candidate_norm_fields_materialized == 0
        and .work.maximum_declared_live_payload_bits
            > .matched_raw_recurrence.maximum_declared_live_payload_bits
    ))
    and .separate_reference.imports_m222_production == false
    and .separate_reference.imports_m221_production == false
    and .separate_reference.uses_prior_standalone_m221_reference_substrate
    and .reuse.primary.canonical_post_restoration_state_exact
    and .reuse.primary.same_residual_backing
    and .reuse.primary.same_unit_ledger_backing
    and .reuse.primary.same_scratch_backing
    and .reuse.reuse.canonical_post_restoration_state_exact
    and .reuse.fresh_restored_reuse_boundary_agreement
    and .reuse.fresh_restored_reuse_state_agreement
    and .controls.wrong_unit_ledger_changes_represented_state
    and .controls.reordered_inverse_rejected
    and (.controls.raw_actual_vector_materialized | not)
    and (.controls.candidate_residual_vector_materialized | not)
    and (.controls.intermediate_actual_vector_projected | not)
    and (.controls.snapshot_command_available | not)
    and (.claim_limits.distinct_phase_resource_established | not)
    and (.claim_limits.computational_advantage | not)
    and (.claim_limits.small_wall_crossed | not)
    and (.terminal | not)
' "$generated_result" >/dev/null

if rg -q '^(from|import) su2_level8_cubic_skein_streamed_embedding_energy([[:space:]]|$)' \
    "$reference"; then
    echo "separate reference imports M222 production" >&2
    exit 2
fi

PYTHONPYCACHEPREFIX="$build/pycache" python3 -m py_compile \
    "$production" "$reference"

echo "QUALIFIED_SU2_LEVEL8_CUBIC_SKEIN_STREAMED_EMBEDDING_ENERGY_WEIGHTED_CELL_NO_GO_STRICT_SCOPE"
