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
        echo "RAM-backed M234 build directory is forbidden" >&2
        exit 2
        ;;
esac
case "$(findmnt -n -o FSTYPE -T "$build")" in
    tmpfs|ramfs)
        echo "RAM-backed M234 build filesystem is forbidden" >&2
        exit 2
        ;;
esac

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
production="$here/projective_complex_gaussian_phase_relation.py"
reference="$here/projective_complex_gaussian_phase_relation_separate_reference.py"
sealed_reference="$here/PROJECTIVE_COMPLEX_GAUSSIAN_PHASE_RELATION_SEPARATE_REFERENCE.json"
sealed_result="$here/PROJECTIVE_COMPLEX_GAUSSIAN_PHASE_RELATION_RESULTS.json"
generated_reference="$build/PROJECTIVE_COMPLEX_GAUSSIAN_PHASE_RELATION_SEPARATE_REFERENCE.json"
generated_result="$build/PROJECTIVE_COMPLEX_GAUSSIAN_PHASE_RELATION_RESULTS.json"
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
    .result == "PASS_EXACT_PROJECTIVE_GAUSSIAN_PHASE_RELATION_FIXED3_CELL_CLOSURE_WITH_GROWING_COEFFICIENT_HEIGHT"
    and .classification == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE"
    and .verification_level == "SEPARATE_REFERENCE_PARITY"
    and .restoration_classification == "EXACT_ALGEBRAIC_RESTORATION"
    and (.cases | length == 16)
    and ([.cases[] | select(.family == 0) | .depth] == [1,2,4,8,16,32,64,128])
    and ([.cases[] | select(.family == 1) | .depth] == [1,2,4,8,16,32,64,128])
    and ([.cases[] | select(.family == 0) | .forward_carrier_payload_bits] == [12,15,21,27,33,39,45,51])
    and ([.cases[] | select(.family == 1) | .forward_carrier_payload_bits] == [13,19,30,51,92,173,334,655])
    and (.cases | all(
        .persistent_relation_rational_cells == 3
        and .retained_final_boundary_rational_cells_during_inverse == 1
        and .missing_inverse_error_cells_and_cursor > 0
        and .restoration_error_rational_cells == 0
        and .same_coefficient_backing
        and .canonical_post_restoration_state_exact
        and .restoration_generation == 1
        and (.baseline_reload_used | not)
        and .work.forward_compositions == .depth
        and .work.inverse_compositions == .depth
        and .work.forward_intersections == .depth
        and .work.inverse_intersections == .depth
        and .work.public_operations_rematerialized == (4 * .depth)
        and .work.public_relation_rational_cells_rematerialized == (12 * .depth)
        and .work.retained_inverse_history_entries == 0
    ))
    and (.controls | to_entries | map(select(
        .key != "public_compiler_reads_final_answer"
        and .key != "relation_tables_materialized"
        and .key != "assignment_expansions_materialized"
        and .key != "intermediate_relations_serialized"
    )) | all(.value == true))
    and (.controls.public_compiler_reads_final_answer | not)
    and (.controls.relation_tables_materialized | not)
    and (.controls.assignment_expansions_materialized | not)
    and (.controls.intermediate_relations_serialized | not)
    and .phase_relation_law.resident_rational_coefficient_cells == 3
    and (.phase_relation_law.shared_port_projected | not)
    and .phase_relation_law.final_boundary_only
    and .phase_relation_law.projective_scalar_amplitude_discarded
    and (.phase_relation_law.convergent_wave_integral_claimed | not)
    and .phase_relation_law.direct_process_logical_custody_only
    and .reuse.primary.restoration_generation == 1
    and .reuse.reuse.restoration_generation == 2
    and .reuse.fresh_reuse.restoration_generation == 1
    and .reuse.restoration_generation_after_reuse == 2
    and .reuse.fresh_restored_boundary_agreement
    and .reuse.fresh_restored_forward_relation_agreement
    and .reuse.same_backing_across_primary_and_reuse
    and (.reuse | [.primary,.reuse,.fresh_reuse] | all(
        .same_coefficient_backing
        and .canonical_post_restoration_state_exact
        and .restoration_error_rational_cells == 0
        and (.baseline_reload_used | not)
    ))
    and .resource_law.persistent_relation_rational_cells_all_depths == 3
    and .resource_law.retained_final_boundary_rational_cells_during_inverse == 1
    and .resource_law.primary_payload_bits_by_depth == [12,15,21,27,33,39,45,51]
    and .resource_law.reuse_family_payload_bits_by_depth == [13,19,30,51,92,173,334,655]
    and .resource_law.closed_formula_relation_commitments_match_all_cases
    and .resource_law.primary_payload_monotone_for_declared_depths
    and .resource_law.reuse_payload_monotone_for_declared_depths
    and .resource_law.retained_inverse_history_entries == 0
    and .resource_law.public_operation_records_retained == 0
    and .resource_law.resource_verification_level == "PACKAGE_SELF_REVIEW"
    and (.resource_law.whole_transaction_live_rational_cell_and_payload_accounting_complete | not)
    and .matched_classical.strongest_implemented == "PUBLIC_DEPTH_CLOSED_THREE_RATIONAL_FORMULA_FOR_BOTH_DECLARED_REPEATED_MODULE_FAMILIES"
    and .matched_classical.same_persistent_cells
    and .matched_classical.same_exact_payload_height
    and .matched_classical.fewer_recurrence_steps_than_iterated_phase_path
    and .matched_classical.closed_formula_persistent_rational_cells == 3
    and (.matched_classical.closed_formula_integer_bit_operation_cost_fully_accounted | not)
    and .matched_classical.identical_schur_add_recurrence_retained_for_arbitrary_valid_public_words
    and (.matched_classical.computational_advantage | not)
    and (.matched_classical.distinct_phase_resource | not)
    and (.separate_reference.imports_m234_production | not)
    and .separate_reference.uses_independent_symmetric_matrix_schur_complement
    and .separate_reference.case_control_reuse_parity
    and (.claim_limits.full_complex_amplitude_preserved | not)
    and (.claim_limits.normalized_or_convergent_gaussian_integral_established | not)
    and (.claim_limits.fresnel_or_maslov_scalar_preserved | not)
    and (.claim_limits.total_category_with_finite_identity_signature | not)
    and (.claim_limits.singular_or_delta_gaussian_relations_supported | not)
    and (.claim_limits.general_non_gaussian_phase_relations | not)
    and (.claim_limits.bounded_exact_coefficient_payload | not)
    and (.claim_limits.machine_enforced_catvm_custody | not)
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
    .schema == "cat_cas.projective_complex_gaussian_phase_relation_reference.v1"
    and (.cases | length == 16)
    and (.cases | all(
        .persistent_relation_rational_cells == 3
        and .retained_final_boundary_rational_cells_during_inverse == 1
        and .restoration_error_rational_cells == 0
        and .same_coefficient_backing
        and .canonical_post_restoration_state_exact
        and (.baseline_reload_used | not)
    ))
    and .reuse.fresh_restored_boundary_agreement
    and .reuse.fresh_restored_forward_relation_agreement
    and .reuse.same_backing_across_primary_and_reuse
    and .reuse.restoration_generation_after_reuse == 2
    and (.controls | to_entries | map(select(
        .key != "public_compiler_reads_final_answer"
        and .key != "relation_tables_materialized"
        and .key != "assignment_expansions_materialized"
        and .key != "intermediate_relations_serialized"
    )) | all(.value == true))
    and (.controls.public_compiler_reads_final_answer | not)
    and (.controls.relation_tables_materialized | not)
    and (.controls.assignment_expansions_materialized | not)
    and (.controls.intermediate_relations_serialized | not)
    and (.imports_m234_production | not)
    and .uses_independent_symmetric_matrix_schur_complement
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
if "projective_complex_gaussian_phase_relation" in imports:
    raise SystemExit("M234 separate reference imports production")
PY

rg -Fq 'shared = c + public_a' "$production"
rg -Fq 'internal_hessian = c + d' "$reference"
rg -Fq 'self.coefficients[:] = result' "$production"
rg -Fq 'carrier[:] = triple(result)' "$reference"
rg -Fq 'PUBLIC_DEPTH_CLOSED_THREE_RATIONAL_FORMULA' "$production"
if rg -q 'itertools|CartesianProduct|assignment_table|truth_table|sample_grid|quadrature' "$production" "$reference"; then
    echo "M234 source contains a forbidden expansion/sampling construct" >&2
    exit 2
fi

echo "QUALIFIED_PROJECTIVE_COMPLEX_GAUSSIAN_PHASE_RELATION_STRICT_SCOPE"
