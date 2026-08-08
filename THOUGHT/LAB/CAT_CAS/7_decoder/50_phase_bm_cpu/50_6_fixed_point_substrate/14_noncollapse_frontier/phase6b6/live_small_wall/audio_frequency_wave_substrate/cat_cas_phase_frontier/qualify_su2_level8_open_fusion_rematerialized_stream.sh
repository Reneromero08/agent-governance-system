#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

if [[ $# -ne 1 || ! -d "$1" ]]; then
    echo "usage: $0 DISK_BACKED_BUILD_DIRECTORY" >&2
    exit 2
fi
build=$(realpath -e -- "$1")
case "$build" in
    /dev/shm|/dev/shm/*|/run/shm|/run/shm/*)
        echo "RAM-backed M231 build directory is forbidden" >&2
        exit 2
        ;;
esac
case "$(findmnt -n -o FSTYPE -T "$build")" in
    tmpfs|ramfs)
        echo "RAM-backed M231 build filesystem is forbidden" >&2
        exit 2
        ;;
esac

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
production="$here/su2_level8_open_fusion_rematerialized_stream.py"
reference="$here/su2_level8_open_fusion_rematerialized_stream_separate_reference.py"
sealed_reference="$here/SU2_LEVEL8_OPEN_FUSION_REMATERIALIZED_STREAM_SEPARATE_REFERENCE.json"
sealed_result="$here/SU2_LEVEL8_OPEN_FUSION_REMATERIALIZED_STREAM_RESULTS.json"
generated_reference="$build/SU2_LEVEL8_OPEN_FUSION_REMATERIALIZED_STREAM_SEPARATE_REFERENCE.json"
generated_result="$build/SU2_LEVEL8_OPEN_FUSION_REMATERIALIZED_STREAM_RESULTS.json"
mkdir -p "$build/tmp" "$build/xdg-cache" "$build/pycache"

env TMPDIR="$build/tmp" TMP="$build/tmp" TEMP="$build/tmp" \
    XDG_CACHE_HOME="$build/xdg-cache" PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPYCACHEPREFIX="$build/pycache" PYTHONPATH="$here" \
    nice -n 10 ionice -c 3 python3 "$reference" > "$generated_reference"
env TMPDIR="$build/tmp" TMP="$build/tmp" TEMP="$build/tmp" \
    XDG_CACHE_HOME="$build/xdg-cache" PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPYCACHEPREFIX="$build/pycache" PYTHONPATH="$here" \
    nice -n 10 ionice -c 3 python3 "$production" "$generated_reference" \
    > "$generated_result"

cmp "$generated_reference" "$sealed_reference"
cmp "$generated_result" "$sealed_result"

jq -e '
    .result == "PASS_BOUNDED_EXACT_OPEN_FUSION_REMATERIALIZED_STREAM_REDUCES_RETAINED_RAILS_BUT_CLASSICAL_REMAINS_SMALLER"
    and .classification == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE"
    and .verification_level == "SEPARATE_REFERENCE_PARITY"
    and .restoration_classification == "EXACT_ALGEBRAIC_RESTORATION"
    and .controls.wrong_owner_rejected
    and .controls.wrong_generation_rejected
    and .controls.wrong_public_program_rejected
    and .controls.wrong_port_type_rejected
    and .controls.premature_projection_rejected
    and .controls.dirty_scalar_scratch_projection_rejected
    and .controls.internal_projection_rejected
    and .controls.missing_inverse_detected
    and .controls.release_before_inverse_rejected
    and .controls.reordered_dependent_inverse_rejected
    and .controls.dirty_scalar_scratch_rejected
    and .controls.stale_generation_rejected
    and .controls.null_carrier_rejected
    and .controls.semantic_source_perturbation_changes_boundary
    and .controls.cubic_parameter_perturbation_changes_boundary_all_families
    and .controls.output_phase_perturbation_changes_boundary_all_families
    and .controls.omitted_cubic_target_changes_boundary_all_families
    and .controls.both_public_families_restore
    and (.controls.public_topology_compilation_reads_final_answer | not)
    and (.controls.relation_tables_materialized | not)
    and (.controls.assignment_expansions_materialized | not)
    and (.cases | length == 2)
    and (.cases | all(
        .topology_descriptor_integers == 22
        and .actual_resident_internal_message_cells == 9
        and .scalar_branch_intersection_backing_cells == 3
        and .final_boundary_backing_cells == 1
        and .phase_work_backing_cells_excluding_public_input == 13
        and .phase_total_backing_cells_including_public_input == 22
        and .distinct_scalar_backings
        and .same_internal_backing_consumed_by_both_branches
        and .same_all_backings
        and .canonical_post_restoration_state_exact
        and .restoration_generation == 1
        and (.baseline_reload_used | not)
        and .work.forward_operations == 2
        and .work.inverse_operations == 2
        and .work.internal_message_productions == 1
        and .work.internal_message_clears == 1
        and .work.fusion_signature_shears == 2
        and .work.scalar_labels_forward == 9
        and .work.scalar_labels_inverse == 9
        and .work.twist_scalar_productions == 18
        and .work.twist_scalar_clears == 18
        and .work.cubic_scalar_productions == 18
        and .work.cubic_scalar_clears == 18
        and .work.intersection_scalar_productions == 18
        and .work.intersection_scalar_clears == 18
        and .work.transposed_fusion_boundary_accumulations == 9
        and .work.transposed_fusion_boundary_clears == 9
        and .work.cubic_term_rematerializations == 4
        and .work.scalar_scratch_restore_checks == 18
        and .work.internal_consumer_mask == 3
        and .work.maximum_declared_live_field_cells == 29
        and .work.maximum_declared_live_payload_bits > 0
        and .work.relation_table_cells_materialized == 0
        and .work.assignment_expansions_materialized == 0
        and .matched_compact_classical.public_input_field_cells == 9
        and .matched_compact_classical.working_backing_field_cells == 1
        and .matched_compact_classical.total_backing_field_cells_including_public_input == 10
        and .matched_compact_classical.maximum_declared_live_field_cells == 20
        and .matched_compact_classical.maximum_declared_live_payload_bits > 0
        and .matched_compact_classical.recurrence
            == "SOURCE_NEIGHBOR_REMATERIALIZED_SCALAR_BRANCH_INTERSECTION_AND_TRANSPOSED_FUSION_BOUNDARY"
        and .matched_compact_classical.boundary_commitment == .boundary_commitment
    ))
    and .reuse.primary.restoration_generation == 1
    and .reuse.reuse.restoration_generation == 2
    and .reuse.fresh_reuse.restoration_generation == 1
    and .reuse.restoration_generation_after_reuse == 2
    and .reuse.fresh_restored_reuse_boundary_agreement
    and (.reuse | [.primary, .reuse, .fresh_reuse] | all(
        .same_all_backings
        and .same_internal_backing_consumed_by_both_branches
        and .canonical_post_restoration_state_exact
        and (.baseline_reload_used | not)
    ))
    and .mechanism_law.resident_unresolved_internal_message_cells == 9
    and .mechanism_law.distinct_twist_cubic_intersection_scalar_backings == 3
    and .mechanism_law.final_boundary_backing_cells == 1
    and .mechanism_law.same_internal_message_consumed_by_both_branches
    and .mechanism_law.native_scalar_hadamard_intersection
    and .mechanism_law.topology_derived_transposed_final_fusion
    and .mechanism_law.branch_scratch_cleared_after_each_label
    and (.mechanism_law.branch_vectors_materialized | not)
    and (.mechanism_law.output_vector_materialized | not)
    and (.mechanism_law.internal_message_projected | not)
    and .mechanism_law.final_boundary_only
    and .mechanism_law.direct_process_logical_custody_only
    and .resource_law.m230_predecessor_work_backing_cells_excluding_public_input == 45
    and .resource_law.m231_work_backing_cells_excluding_public_input == 13
    and .resource_law.m231_total_backing_cells_including_public_input == 22
    and .resource_law.matched_source_rematerialized_classical_work_backing_cells_excluding_public_input == 1
    and .resource_law.matched_source_rematerialized_classical_total_backing_cells_including_public_input == 10
    and .resource_law.primary_maximum_declared_live_field_cells == 29
    and .resource_law.matched_classical_maximum_declared_live_field_cells == 20
    and .resource_law.retained_public_topology_descriptor_integers == 22
    and .resource_law.resource_measurement_verification_level == "PACKAGE_SELF_REVIEW"
    and (.resource_law.whole_transaction_live_payload_complete | not)
    and .resource_law.projected_boundary_retention_during_inverse_counted
    and .matched_compact_classical.strongest
        == "SOURCE_NEIGHBOR_REMATERIALIZED_SCALAR_BRANCH_INTERSECTION_AND_TRANSPOSED_FUSION_BOUNDARY"
    and .matched_compact_classical.boundary_agreement_all_cases
    and .matched_compact_classical.classical_work_backings_are_smaller
    and .matched_compact_classical.classical_total_backings_are_smaller
    and .matched_compact_classical.identical_algebraic_recurrence
    and (.matched_compact_classical.phase_specific_reduction | not)
    and (.matched_compact_classical.computational_advantage | not)
    and (.separate_reference.imports_m231_production | not)
    and (.separate_reference.imports_m230_production | not)
    and (.separate_reference.imports_m211_production | not)
    and .separate_reference.uses_independent_polynomial_quotient
    and .separate_reference.case_control_restoration_reuse_parity
    and (.claim_limits.arbitrary_su2_relation_signatures | not)
    and (.claim_limits.growing_interface_width | not)
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

jq -e '
    .schema == "cat_cas.su2_level8_open_fusion_rematerialized_reference.v1"
    and (.imports_m231_production | not)
    and (.imports_m230_production | not)
    and (.imports_m211_production | not)
    and .uses_independent_polynomial_quotient
    and (.cases | length == 2)
    and (.cases | all(
        .canonical_post_restoration_state_exact
        and .actual_resident_internal_message_cells == 9
        and .phase_work_backing_cells_excluding_public_input == 13
        and .phase_total_backing_cells_including_public_input == 22
        and (.baseline_reload_used | not)
    ))
    and .reuse.fresh_restored_reuse_boundary_agreement
    and .reuse.primary.restoration_generation == 1
    and .reuse.reuse.restoration_generation == 2
    and .reuse.fresh_reuse.restoration_generation == 1
    and .reuse.restoration_generation_after_reuse == 2
' "$generated_reference" >/dev/null

python3 - "$reference" <<'PY'
import ast
from pathlib import Path
import sys

tree = ast.parse(Path(sys.argv[1]).read_text(encoding="utf-8"))
imports = set()
for node in ast.walk(tree):
    if isinstance(node, ast.Import):
        imports.update(alias.name for alias in node.names)
    elif isinstance(node, ast.ImportFrom):
        imports.add(node.module or "")
for forbidden in (
    "su2_level8_open_fusion_rematerialized_stream",
    "su2_level8_open_fusion_signature_fanout_intersection",
    "root_of_unity_su2_level8_fusion_phase_relation",
):
    if forbidden in imports:
        raise SystemExit(f"M231 separate reference imports production: {forbidden}")
PY

rg -Fq 'internal_value = rematerialize_internal(label)' "$production"
rg -Fq 'source_value = rematerialize_internal(' "$production"
rg -Fq 'topology.cubic_branch.source,' "$production"
rg -Fq 'work.retained_result_values = (boundary,)' "$production"
rg -Fq 'oracle.poly_multiply(polynomial, X), oracle.E.root(parameter)' "$reference"
rg -Fq 'full_materialized_boundary(source, program)' "$reference"
if rg -q 'itertools|CartesianProduct|assignment_table|truth_table' "$production" "$reference"; then
    echo "M231 accepted path contains an expansion construct" >&2
    exit 2
fi

echo "QUALIFIED_SU2_LEVEL8_OPEN_FUSION_REMATERIALIZED_STREAM_STRICT_SCOPE"
