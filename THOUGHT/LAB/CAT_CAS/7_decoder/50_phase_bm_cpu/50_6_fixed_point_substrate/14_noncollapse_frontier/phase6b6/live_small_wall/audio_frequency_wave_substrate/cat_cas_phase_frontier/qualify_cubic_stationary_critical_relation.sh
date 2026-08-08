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
        echo "RAM-backed M235 build directory is forbidden" >&2
        exit 2
        ;;
esac
case "$(findmnt -n -o FSTYPE -T "$build")" in
    tmpfs|ramfs)
        echo "RAM-backed M235 build filesystem is forbidden" >&2
        exit 2
        ;;
esac

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
production="$here/cubic_stationary_critical_relation.py"
reference="$here/cubic_stationary_critical_relation_separate_reference.py"
sealed_reference="$here/CUBIC_STATIONARY_CRITICAL_RELATION_SEPARATE_REFERENCE.json"
sealed_result="$here/CUBIC_STATIONARY_CRITICAL_RELATION_RESULTS.json"
generated_reference="$build/CUBIC_STATIONARY_CRITICAL_RELATION_SEPARATE_REFERENCE.json"
generated_result="$build/CUBIC_STATIONARY_CRITICAL_RELATION_RESULTS.json"
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
    .result == "PASS_EXACT_CUBIC_STATIONARY_CRITICAL_RELATION_BRANCH_DEGREE_GROWTH_WITH_EXACT_RESTORATION_REUSE"
    and .classification == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE"
    and .verification_level == "SEPARATE_REFERENCE_PARITY"
    and .restoration_classification == "EXACT_ALGEBRAIC_RESTORATION"
    and (.cases | length == 16)
    and ([.cases[] | select(.family == 0) | .depth] == [1,2,3,4,5,6,7,8])
    and ([.cases[] | select(.family == 0) | .final_relation_degree] == [4,10,22,46,94,190,382,766])
    and ([.cases[] | select(.family == 1) | .final_relation_degree] == [4,10,22,46,94,190,382,766])
    and ([.cases[] | select(.family == 0) | .resident_relation_coefficient_cells] == [5,11,23,47,95,191,383,767])
    and (.cases | all(
        .final_relation_degree == .distinct_complex_branch_count
        and .final_relation_degree == .expected_degree
        and .simple_branch_conditions_hold
        and .work.simple_branch_preflight_evaluations == (3 * .depth)
        and .work.public_rounds_rematerialized == (2 * .depth)
        and .work.retained_inverse_history_entries == 0
        and .retained_final_boundary_integer_cells_during_inverse == 1
        and .restoration_error_coefficient_cells == 0
        and .canonical_post_restoration_state_exact
        and .same_coefficient_backing
        and .restoration_generation == 1
        and (.baseline_reload_used | not)
        and .matched_classical.persistent_integer_cells == 2
        and .matched_classical.squarings == .depth
        and .matched_classical.subtractions == (2 * .depth)
        and .matched_classical.multiplications == (.depth + 1)
        and .matched_classical.boundary_commitment == .boundary_commitment
    ))
    and (.controls | to_entries | map(select(
        .key != "public_compiler_reads_final_answer"
        and .key != "relation_tables_materialized"
        and .key != "assignment_expansions_materialized"
        and .key != "stationary_branches_enumerated"
        and .key != "intermediate_relations_serialized"
    )) | all(.value == true))
    and (.controls.public_compiler_reads_final_answer | not)
    and (.controls.relation_tables_materialized | not)
    and (.controls.assignment_expansions_materialized | not)
    and (.controls.stationary_branches_enumerated | not)
    and (.controls.intermediate_relations_serialized | not)
    and .reuse.primary.restoration_generation == 1
    and .reuse.reuse.restoration_generation == 2
    and .reuse.fresh_reuse.restoration_generation == 1
    and .reuse.restoration_generation_after_reuse == 2
    and .reuse.fresh_restored_boundary_agreement
    and .reuse.fresh_restored_relation_agreement
    and .reuse.fresh_restored_degree_and_payload_agreement
    and .reuse.same_backing_across_primary_and_reuse
    and .phase_relation_law.native_shared_port_elimination == "R(v)->R(w^2-c)"
    and .phase_relation_law.declared_complex_branches_are_distinct
    and (.phase_relation_law.shared_stationary_port_projected | not)
    and .phase_relation_law.final_boundary_only
    and .phase_relation_law.projective_scalar_amplitude_discarded
    and (.phase_relation_law.convergent_wave_integral_claimed | not)
    and .phase_relation_law.direct_process_logical_custody_only
    and .resource_law.exact_degree_law == "3*2^depth-2"
    and .resource_law.resident_relation_coefficient_cells_by_depth == [5,11,23,47,95,191,383,767]
    and .resource_law.all_declared_relations_square_free_by_exact_inductive_preflight
    and (.resource_law.stationary_branches_enumerated | not)
    and .resource_law.retained_inverse_history_entries == 0
    and .resource_law.resource_verification_level == "PACKAGE_SELF_REVIEW"
    and (.resource_law.whole_transaction_live_integer_cell_and_payload_accounting_complete | not)
    and .matched_classical.strongest_implemented == "PUBLIC_DESCRIPTOR_BACKWARD_SCALAR_BOUNDARY_RECURRENCE"
    and .matched_classical.persistent_integer_cells == 2
    and .matched_classical.expanded_relation_cells_at_depth8 == 767
    and .matched_classical.same_exact_boundary
    and .matched_classical.retains_open_relation_signature
    and .matched_classical.integer_payload_width_grows
    and (.matched_classical.complete_integer_bit_operation_cost_fully_accounted | not)
    and (.matched_classical.computational_advantage | not)
    and (.matched_classical.distinct_phase_resource | not)
    and (.separate_reference.imports_m235_production | not)
    and .separate_reference.uses_independent_sparse_exponent_dictionary
    and .separate_reference.case_control_reuse_parity
    and (.claim_limits.normalized_or_convergent_cubic_integral_established | not)
    and (.claim_limits.fresnel_or_maslov_scalar_preserved | not)
    and (.claim_limits.general_cubic_phase_relation_algebra | not)
    and (.claim_limits.set_theoretic_relation_intersection_established | not)
    and (.claim_limits.fixed_coefficient_cell_closure | not)
    and (.claim_limits.all_non_gaussian_relations_require_exponential_state | not)
    and (.claim_limits.compact_nonlinear_or_expression_quotient_rejected | not)
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
    .schema == "cat_cas.cubic_stationary_critical_relation_reference.v1"
    and (.cases | length == 16)
    and (.cases | all(
        .final_relation_degree == .distinct_complex_branch_count
        and .simple_branch_conditions_hold
        and .restoration_error_coefficient_cells == 0
        and .canonical_post_restoration_state_exact
        and .same_coefficient_backing
        and (.baseline_reload_used | not)
    ))
    and .reuse.fresh_restored_boundary_agreement
    and .reuse.fresh_restored_relation_agreement
    and .reuse.same_backing_across_primary_and_reuse
    and .reuse.restoration_generation_after_reuse == 2
    and (.imports_m235_production | not)
    and .uses_independent_sparse_exponent_dictionary
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
if "cubic_stationary_critical_relation" in imports:
    raise SystemExit("M235 separate reference imports production")
PY

rg -Fq 'w^3/3 - (v+c)w' "$production"
rg -Fq 'R(v)->R(w^2-c)' "$production"
rg -Fq 'stationary_compose' "$reference"
if rg -q 'itertools|CartesianProduct|assignment_table|truth_table|root_list|nroots|solve\(' "$production" "$reference"; then
    echo "M235 source contains a forbidden branch enumeration or assignment construct" >&2
    exit 2
fi

echo "QUALIFIED_CUBIC_STATIONARY_CRITICAL_RELATION_STRICT_SCOPE"
