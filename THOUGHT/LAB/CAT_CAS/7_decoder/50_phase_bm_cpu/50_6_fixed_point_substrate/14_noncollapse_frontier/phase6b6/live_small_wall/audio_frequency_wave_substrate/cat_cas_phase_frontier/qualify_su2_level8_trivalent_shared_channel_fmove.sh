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
        echo "RAM-backed M232 build directory is forbidden" >&2
        exit 2
        ;;
esac
case "$(findmnt -n -o FSTYPE -T "$build")" in
    tmpfs|ramfs)
        echo "RAM-backed M232 build filesystem is forbidden" >&2
        exit 2
        ;;
esac

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
production="$here/su2_level8_trivalent_shared_channel_fmove.py"
reference="$here/su2_level8_trivalent_shared_channel_fmove_separate_reference.py"
sealed_reference="$here/SU2_LEVEL8_TRIVALENT_SHARED_CHANNEL_FMOVE_SEPARATE_REFERENCE.json"
sealed_result="$here/SU2_LEVEL8_TRIVALENT_SHARED_CHANNEL_FMOVE_RESULTS.json"
generated_reference="$build/SU2_LEVEL8_TRIVALENT_SHARED_CHANNEL_FMOVE_SEPARATE_REFERENCE.json"
generated_result="$build/SU2_LEVEL8_TRIVALENT_SHARED_CHANNEL_FMOVE_RESULTS.json"
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
    .result == "PASS_BOUNDED_EXACT_TRIVALENT_SHARED_CHANNEL_F_MOVE_BRAID_CONTRACTION_WITH_IDENTICAL_SMALLER_CLASSICAL_FACTOR_GRAPH"
    and .classification == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE"
    and .verification_level == "SEPARATE_REFERENCE_PARITY"
    and .restoration_classification == "EXACT_ALGEBRAIC_RESTORATION"
    and .controls.analytic_trivalent_admissibility_has_channels0_and2
    and .controls.invalid_trivalent_signature_rejected
    and .controls.non_diagonal_f_move_offdiagonal_nonzero
    and .controls.f_move_involution_exact
    and .controls.braid_phase_inverse_exact
    and .controls.yang_baxter_relation_exact
    and .controls.wrong_owner_rejected
    and .controls.wrong_generation_rejected
    and .controls.wrong_public_program_rejected
    and .controls.wrong_port_type_rejected
    and .controls.premature_projection_rejected
    and .controls.shared_channel_projection_rejected
    and .controls.missing_inverse_detected
    and .controls.reordered_dependent_inverse_rejected
    and .controls.wrong_inverse_braid_rejected
    and .controls.stale_generation_rejected
    and .controls.null_carrier_rejected
    and .controls.f_move_perturbation_changes_boundary_all_families
    and .controls.braid_phase_perturbation_changes_boundary_all_families
    and .controls.omitting_either_shared_channel_changes_boundary_all_families
    and .controls.public_families_have_distinct_boundaries
    and .controls.both_public_families_restore
    and (.controls.public_topology_compilation_reads_final_answer | not)
    and (.controls.fusion_relation_tables_materialized | not)
    and (.controls.assignment_expansions_materialized | not)
    and (.cases | length == 2)
    and (.cases | all(
        .topology_descriptor_integers == 11
        and .analytic_shared_channels == [0, 2]
        and .resident_unresolved_channel_cells == 2
        and .final_boundary_backing_cells == 1
        and .phase_work_backing_cells == 3
        and .same_channel_and_boundary_backings
        and .canonical_post_restoration_state_exact
        and .restoration_generation == 1
        and (.baseline_reload_used | not)
        and .work.left_vertex_message_productions == 1
        and .work.left_vertex_message_clears == 1
        and .work.associator_applications == 4
        and .work.braid_phase_applications == 2
        and .work.right_vertex_contractions == 1
        and .work.right_vertex_contraction_clears == 1
        and .work.field_additions == 17
        and .work.field_subtractions == 7
        and .work.field_multiplications == 30
        and .work.field_inversions == 2
        and .work.maximum_declared_live_field_cells == 15
        and .work.maximum_declared_live_payload_bits > 0
        and .work.relation_table_cells_materialized == 0
        and .work.assignment_expansions_materialized == 0
        and .work.intermediate_commitments_emitted == 0
        and .matched_sparse_classical.working_backing_field_cells == 1
        and .matched_sparse_classical.maximum_declared_live_field_cells == 12
        and .matched_sparse_classical.maximum_declared_live_payload_bits > 0
        and .matched_sparse_classical.field_additions == 8
        and .matched_sparse_classical.field_multiplications == 11
        and .matched_sparse_classical.field_inversions == 1
        and .matched_sparse_classical.retained_precontracted_row_field_cells == 0
        and .matched_sparse_classical.recurrence == "ANALYTIC_PRECONTRACTED_PUBLIC_B_EQUALS_F_R_F_BOUNDARY_ROW_STREAM"
        and .matched_sparse_classical.boundary_commitment == .boundary_commitment
    ))
    and .reuse.primary.restoration_generation == 1
    and .reuse.reuse.restoration_generation == 2
    and .reuse.fresh_reuse.restoration_generation == 1
    and .reuse.restoration_generation_after_reuse == 2
    and .reuse.fresh_restored_reuse_boundary_agreement
    and (.reuse | [.primary, .reuse, .fresh_reuse] | all(
        .same_channel_and_boundary_backings
        and .canonical_post_restoration_state_exact
        and (.baseline_reload_used | not)
    ))
    and .relation_law.analytic_trivalent_fusion_signatures
    and .relation_law.shared_unresolved_channel_labels == [0, 2]
    and .relation_law.shared_unresolved_channel_cells == 2
    and .relation_law.two_vertices_share_actual_channel
    and .relation_law.non_diagonal_associator
    and .relation_law.diagonal_braid_phase_in_associated_basis
    and .relation_law.native_right_vertex_contraction
    and (.relation_law.fusion_table_materialized | not)
    and (.relation_law.assignment_expansion_materialized | not)
    and (.relation_law.shared_channel_projected | not)
    and .relation_law.final_boundary_only
    and .relation_law.direct_process_logical_custody_only
    and .resource_law.phase_shared_channel_backing_cells == 2
    and .resource_law.phase_final_boundary_backing_cells == 1
    and .resource_law.phase_work_backing_cells == 3
    and .resource_law.matched_sparse_classical_work_backing_cells == 1
    and .resource_law.phase_maximum_declared_live_field_cells == 15
    and .resource_law.matched_classical_maximum_declared_live_field_cells == 12
    and .resource_law.retained_projected_boundary_during_inverse_counted
    and .resource_law.resource_measurement_verification_level == "PACKAGE_SELF_REVIEW"
    and (.resource_law.whole_transaction_live_payload_complete | not)
    and .matched_sparse_classical.strongest == "ANALYTIC_PRECONTRACTED_PUBLIC_B_EQUALS_F_R_F_BOUNDARY_ROW_STREAM"
    and .matched_sparse_classical.boundary_agreement_all_cases
    and .matched_sparse_classical.classical_work_backings_are_smaller
    and .matched_sparse_classical.identical_algebraic_contraction
    and (.matched_sparse_classical.phase_specific_reduction | not)
    and (.matched_sparse_classical.computational_advantage | not)
    and (.separate_reference.imports_m232_production | not)
    and (.separate_reference.imports_m231_production | not)
    and (.separate_reference.imports_m214_production | not)
    and .separate_reference.uses_independent_cyclotomic_polynomial_oracle
    and .separate_reference.case_control_restoration_reuse_parity
    and (.claim_limits.general_su2_fusion_category | not)
    and (.claim_limits.pentagon_or_hexagon_family_verified | not)
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
    .schema == "cat_cas.su2_level8_trivalent_shared_channel_reference.v1"
    and (.imports_m232_production | not)
    and (.imports_m231_production | not)
    and (.imports_m214_production | not)
    and .uses_independent_cyclotomic_polynomial_oracle
    and .controls.yang_baxter_relation_exact
    and (.cases | length == 2)
    and (.cases | all(
        .analytic_shared_channels == [0, 2]
        and .resident_unresolved_channel_cells == 2
        and .final_boundary_backing_cells == 1
        and .phase_work_backing_cells == 3
        and .canonical_post_restoration_state_exact
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
    "su2_level8_trivalent_shared_channel_fmove",
    "su2_level8_open_fusion_rematerialized_stream",
    "su2_level8_fusion_path_braid_phase_relation",
):
    if forbidden in imports:
        raise SystemExit(f"M232 separate reference imports production: {forbidden}")
PY

rg -Fq 'offdiagonal = work.multiply(PHI, INVERSE_DELTA)' "$production"
rg -Fq 'self.channel[index] = self.channel[index] + vertex(1, 1, label)' "$reference"
rg -Fq 'b_matrix = matrix_multiply(matrix_multiply(f_matrix, r_matrix), f_matrix)' "$production" "$reference"
rg -Fq 'ANALYTIC_PRECONTRACTED_PUBLIC_B_EQUALS_F_R_F_BOUNDARY_ROW_STREAM' "$production"
rg -Fq 'work.retained_result_values = (boundary,)' "$production"
rg -Fq 'if boundary != direct_network_boundary(program):' "$reference"
if rg -q 'itertools|CartesianProduct|assignment_table|truth_table' "$production" "$reference"; then
    echo "M232 accepted path contains an expansion/table construct" >&2
    exit 2
fi

echo "QUALIFIED_SU2_LEVEL8_TRIVALENT_SHARED_CHANNEL_FMOVE_STRICT_SCOPE"
