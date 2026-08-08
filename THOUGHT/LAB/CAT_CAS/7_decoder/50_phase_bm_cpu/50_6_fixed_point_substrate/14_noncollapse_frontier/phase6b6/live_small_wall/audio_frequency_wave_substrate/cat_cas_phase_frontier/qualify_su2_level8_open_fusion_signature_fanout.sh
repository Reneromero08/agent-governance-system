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
        echo "RAM-backed M230 build directory is forbidden" >&2
        exit 2
        ;;
esac
case "$(findmnt -n -o FSTYPE -T "$build")" in
    tmpfs|ramfs)
        echo "RAM-backed M230 build filesystem is forbidden" >&2
        exit 2
        ;;
esac

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
production="$here/su2_level8_open_fusion_signature_fanout_intersection.py"
reference="$here/su2_level8_open_fusion_signature_fanout_intersection_separate_reference.py"
sealed_reference="$here/SU2_LEVEL8_OPEN_FUSION_SIGNATURE_FANOUT_SEPARATE_REFERENCE.json"
sealed_result="$here/SU2_LEVEL8_OPEN_FUSION_SIGNATURE_FANOUT_RESULTS.json"
generated_reference="$build/SU2_LEVEL8_OPEN_FUSION_SIGNATURE_FANOUT_SEPARATE_REFERENCE.json"
generated_result="$build/SU2_LEVEL8_OPEN_FUSION_SIGNATURE_FANOUT_RESULTS.json"
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
    .result == "PASS_BOUNDED_EXACT_OPEN_FUSION_SIGNATURE_FANOUT_INTERSECTION_WITH_SMALLER_CLASSICAL_STREAM"
    and .classification == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE"
    and .verification_level == "SEPARATE_REFERENCE_PARITY"
    and .restoration_classification == "EXACT_ALGEBRAIC_RESTORATION"
    and .controls.nonfunctional_fusion_has_sixteen_directed_support_edges
    and .controls.nonfunctional_fusion_has_seven_multitarget_sources
    and .controls.wrong_owner_rejected
    and .controls.wrong_generation_rejected
    and .controls.wrong_public_program_rejected
    and .controls.wrong_port_type_rejected
    and .controls.premature_projection_rejected
    and .controls.internal_projection_rejected
    and .controls.undermerge_rejected
    and .controls.duplicate_branch_overmerge_rejected
    and .controls.missing_inverse_detected
    and .controls.reordered_dependent_inverse_rejected
    and .controls.stale_generation_rejected
    and .controls.null_carrier_rejected
    and .controls.semantic_source_perturbation_changes_boundary
    and .controls.cubic_parameter_perturbation_changes_intersection_and_boundary
    and .controls.both_public_branch_orders_restore
    and (.controls.public_topology_compilation_reads_final_answer | not)
    and (.controls.relation_tables_materialized | not)
    and (.controls.assignment_expansions_materialized | not)
    and (.cases | length == 2)
    and (.cases | all(
        .topology_descriptor_integers == 22
        and .nonfunctional_fusion_directed_support_edges == 16
        and .fusion_sources_with_two_targets == 7
        and .actual_internal_message_cells == 9
        and .distinct_branch_backings
        and .same_internal_backing_seen_by_both_branches
        and .same_rail_backings
        and .canonical_post_restoration_state_exact
        and .restoration_generation == 1
        and (.baseline_reload_used | not)
        and .forward_all_rails_payload_bits > 0
        and .work.forward_operations == 5
        and .work.inverse_operations == 5
        and .work.internal_message_productions == 1
        and .work.internal_message_clears == 1
        and .work.distinct_branch_backing_mask == 3
        and .work.fusion_signature_shears == 4
        and .work.twist_branch_shears == 2
        and .work.cubic_branch_shears == 2
        and .work.intersection_shears == 2
        and .work.maximum_declared_live_field_cells == 60
        and .work.maximum_declared_live_payload_bits > 0
        and .work.relation_table_cells_materialized == 0
        and .work.assignment_expansions_materialized == 0
        and .matched_compact_classical.public_input_field_cells == 9
        and .matched_compact_classical.working_resident_field_cells == 9
        and .matched_compact_classical.total_resident_field_cells_including_public_input == 18
        and .matched_compact_classical.materialized_twist_branch_cells == 0
        and .matched_compact_classical.materialized_cubic_branch_cells == 0
        and .matched_compact_classical.materialized_intersection_cells == 0
        and .matched_compact_classical.materialized_output_cells == 0
        and .matched_compact_classical.boundary_commitment == .boundary_commitment
    ))
    and .reuse.primary.restoration_generation == 1
    and .reuse.reuse.restoration_generation == 2
    and .reuse.fresh_reuse.restoration_generation == 1
    and .reuse.restoration_generation_after_reuse == 2
    and .reuse.fresh_restored_reuse_boundary_agreement
    and .reuse.fresh_restored_reuse_output_agreement
    and (.reuse | [.primary, .reuse, .fresh_reuse] | all(
        .same_rail_backings
        and .same_internal_backing_seen_by_both_branches
        and .distinct_branch_backings
        and .canonical_post_restoration_state_exact
        and (.baseline_reload_used | not)
    ))
    and .relation_law.nonfunctional_open_fusion_signature_constructed
    and (.relation_law.fusion_signature_support_materialized_as_table | not)
    and .relation_law.fusion_signature_analytic_neighbor_law
    and .relation_law.actual_produced_internal_relation_message_cells == 9
    and .relation_law.same_internal_message_consumed_by_two_branches
    and .relation_law.distinct_twist_and_cubic_branch_backings
    and .relation_law.native_hadamard_intersection
    and (.relation_law.internal_message_projected | not)
    and .relation_law.final_boundary_only
    and .relation_law.direct_process_logical_custody_only
    and .resource_law.resident_relation_rail_field_cells == 54
    and .resource_law.public_source_field_cells == 9
    and .resource_law.working_relation_rail_field_cells_excluding_public_source == 45
    and .resource_law.primary_maximum_declared_live_field_cells == 60
    and .resource_law.retained_public_topology_descriptor_integers == 22
    and .resource_law.kernel_transients_counted_at_observed_intervals
    and (.resource_law.projected_boundary_retention_during_inverse_counted | not)
    and (.resource_law.whole_transaction_live_payload_complete | not)
    and .matched_compact_classical.strongest
        == "NINE_CELL_INTERNAL_PLUS_STREAMED_BRANCH_INTERSECTION_AND_TRANSPOSE_FUSION_BOUNDARY"
    and .matched_compact_classical.public_input_field_cells == 9
    and .matched_compact_classical.working_resident_field_cells == 9
    and .matched_compact_classical.total_resident_field_cells_including_public_input == 18
    and .matched_compact_classical.phase_relation_working_field_cells_excluding_public_input == 45
    and .matched_compact_classical.phase_relation_total_resident_field_cells_including_public_input == 54
    and .matched_compact_classical.classical_working_is_strictly_smaller
    and .matched_compact_classical.classical_total_is_strictly_smaller
    and .matched_compact_classical.boundary_agreement_all_cases
    and .matched_compact_classical.resource_measurement_verification_level == "PACKAGE_SELF_REVIEW"
    and (.matched_compact_classical.phase_specific_reduction | not)
    and (.matched_compact_classical.computational_advantage | not)
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
    .schema == "cat_cas.su2_level8_open_fusion_signature_fanout_reference.v1"
    and (.imports_m230_production | not)
    and (.imports_m211_production | not)
    and .uses_independent_polynomial_quotient
    and (.cases | length == 2)
    and (.cases | all(.canonical_post_restoration_state_exact))
    and .reuse.fresh_restored_reuse_boundary_agreement
    and .reuse.fresh_restored_reuse_output_agreement
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
    "su2_level8_open_fusion_signature_fanout_intersection",
    "root_of_unity_su2_level8_fusion_phase_relation",
):
    if forbidden in imports:
        raise SystemExit(f"M230 separate reference imports production: {forbidden}")
PY

rg -Fq 'upper = source[label + 1]' "$production"
rg -Fq 'self.twist_branch,' "$production"
rg -Fq 'self.cubic_branch,' "$production"
rg -Fq 'product = work.multiply(left[label], right[label])' "$production"
rg -Fq 'oracle.poly_multiply(polynomial, X), oracle.E.root(signature.parameter)' "$reference"
if rg -q 'itertools|CartesianProduct|product\(' "$production" "$reference"; then
    echo "M230 accepted path contains an expansion construct" >&2
    exit 2
fi

echo "QUALIFIED_SU2_LEVEL8_OPEN_FUSION_SIGNATURE_FANOUT_STRICT_SCOPE"
