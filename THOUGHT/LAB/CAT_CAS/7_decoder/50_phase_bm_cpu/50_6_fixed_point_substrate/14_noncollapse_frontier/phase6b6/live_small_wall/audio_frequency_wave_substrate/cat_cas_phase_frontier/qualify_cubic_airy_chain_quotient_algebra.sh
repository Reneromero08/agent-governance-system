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
        echo "RAM-backed M236 build directory is forbidden" >&2
        exit 2
        ;;
esac
case "$(findmnt -n -o FSTYPE -T "$build")" in
    tmpfs|ramfs)
        echo "RAM-backed M236 build filesystem is forbidden" >&2
        exit 2
        ;;
esac

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
production="$here/cubic_airy_chain_quotient_algebra.py"
reference="$here/cubic_airy_chain_quotient_algebra_separate_reference.py"
sealed_reference="$here/CUBIC_AIRY_CHAIN_QUOTIENT_ALGEBRA_SEPARATE_REFERENCE.json"
sealed_result="$here/CUBIC_AIRY_CHAIN_QUOTIENT_ALGEBRA_RESULTS.json"
generated_reference="$build/CUBIC_AIRY_CHAIN_QUOTIENT_ALGEBRA_SEPARATE_REFERENCE.json"
generated_result="$build/CUBIC_AIRY_CHAIN_QUOTIENT_ALGEBRA_RESULTS.json"
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

production_hash=$(sha256sum "$production" | awk '{print $1}')
reference_hash=$(sha256sum "$reference" | awk '{print $1}')
[[ $(jq -r '.source_dependencies.production_sha256' "$generated_result") == "$production_hash" ]]
[[ $(jq -r '.source_dependencies.separate_reference_sha256' "$generated_result") == "$reference_hash" ]]
[[ $(jq -r '.source_sha256' "$generated_reference") == "$reference_hash" ]]

jq -e '
    .result == "PASS_EXACT_CUBIC_AIRY_CHAIN_QUOTIENT_ALGEBRA_STRICT_SCOPE"
    and .classification == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE"
    and .verification_level == "SEPARATE_REFERENCE_PARITY"
    and .restoration_classification == "EXACT_ALGEBRAIC_RESTORATION"
    and (.cases | length == 10)
    and ([.cases[] | select(.family == 0) | .length] == [2,3,4,5,6])
    and ([.cases[] | select(.family == 1) | .length] == [2,3,4,5,6])
    and ([.cases[] | select(.family == 0) | .quotient_dimension] == [2,4,8,16,32])
    and ([.cases[] | select(.family == 1) | .quotient_dimension] == [2,4,8,16,32])
    and ([.cases[] | select(.family == 0) | .critical_value_relation_payload_bits] == [11,39,144,542,2114])
    and ([.cases[] | select(.family == 1) | .critical_value_relation_payload_bits] == [9,30,121,477,1759])
    and (.cases | all(
        .endpoint_polynomial_degree == .quotient_dimension
        and .critical_value_relation_degree == .quotient_dimension
        and .endpoint_polynomial_square_free
        and .critical_value_relation_square_free
        and .critical_value_relation_annihilates_resident_action
        and .modulus_coefficient_cells == (.quotient_dimension + 1)
        and .carrier_quotient_coefficient_cells == (3 * .quotient_dimension)
        and .retained_final_relation_coefficient_cells_during_inverse == (.quotient_dimension + 1)
        and .canonical_post_inverse_state_exact
        and .same_r_s_h_modulus_backings
        and .restoration_generation == 1
        and (.baseline_reload_used | not)
        and .retained_dynamic_inverse_history_entries == 0
        and .work.forward_phase_terms == .length
        and .work.forward_maps == (.length - 1)
        and .work.inverse_phase_terms == .length
        and .work.inverse_maps == (.length - 1)
        and .work.characteristic_matrix_products == .quotient_dimension
        and .matched_classical_relation_commitment == .critical_value_relation_commitment
    ))
    and (.controls | to_entries | map(select(
        .key != "stationary_roots_or_branch_lists_materialized"
        and .key != "intermediate_quotient_elements_serialized"
        and .key != "public_compiler_reads_final_answer"
    )) | all(.value == true))
    and (.controls.stationary_roots_or_branch_lists_materialized | not)
    and (.controls.intermediate_quotient_elements_serialized | not)
    and (.controls.public_compiler_reads_final_answer | not)
    and .reuse.primary.restoration_generation == 1
    and .reuse.reuse.restoration_generation == 2
    and .reuse.fresh_reuse.restoration_generation == 1
    and .reuse.restoration_generation_after_reuse == 2
    and .reuse.fresh_restored_boundary_agreement
    and .reuse.same_backing_across_primary_and_reuse
    and .reuse.primary.canonical_post_inverse_state_exact
    and .reuse.reuse.canonical_post_inverse_state_exact
    and .reuse.fresh_reuse.canonical_post_inverse_state_exact
    and (.reuse.primary.baseline_reload_used | not)
    and (.reuse.reuse.baseline_reload_used | not)
    and (.reuse.fresh_reuse.baseline_reload_used | not)
    and .phase_relation_law.formal_cubic_signature == "S_alpha(u,v)=u*v+alpha*v^3/3"
    and .phase_relation_law.scaled_action_coordinate == "W=3*Phi"
    and .phase_relation_law.stationary_recurrence == "q_(i+1)=-q_(i-1)-alpha_i*q_i^2"
    and .phase_relation_law.endpoint_quotient == "Q[t]/(q_n(t)-z)"
    and .phase_relation_law.native_reversible_map == "T_alpha(r,s)=(s,-r-alpha*s^2)"
    and .phase_relation_law.final_relation == "Norm_(Q[t]/P_n)(W-H_n)"
    and .phase_relation_law.stationary_branches_unresolved
    and (.phase_relation_law.stationary_roots_enumerated | not)
    and (.phase_relation_law.resident_critical_value_exported | not)
    and .phase_relation_law.final_boundary_only
    and .phase_relation_law.direct_process_logical_custody_only
    and .resource_law.quotient_dimensions_by_length == [2,4,8,16,32]
    and .resource_law.carrier_coefficient_cells_by_length == [6,12,24,48,96]
    and .resource_law.immutable_modulus_coefficient_cells_by_length == [3,5,9,17,33]
    and .resource_law.final_relation_output_coefficient_cells_by_length == [3,5,9,17,33]
    and .resource_law.characteristic_matrix_cells_by_length == [4,16,64,256,1024]
    and .resource_law.characteristic_polynomial_component_peak_matrix_cells_by_length == [12,48,192,768,3072]
    and .resource_law.retained_dynamic_inverse_history_entries == 0
    and (.resource_law.carrier_initial_state_snapshot_retained | not)
    and (.resource_law.quotient_arithmetic_operation_counts_instrumented | not)
    and .resource_law.resource_verification_level == "PACKAGE_SELF_REVIEW"
    and (.resource_law.whole_transaction_live_cell_and_payload_accounting_complete | not)
    and .matched_classical.strongest_implemented == "IDENTICAL_ENDPOINT_POLYNOMIAL_QUOTIENT_RECURRENCE_AND_EXACT_CHARACTERISTIC_NORM"
    and .matched_classical.strongest_known_compact_projection == "UNIVARIATE_CHAIN_SPECIFIC_EXACT_SUBRESULTANT_PRS_RESULTANT"
    and (.matched_classical.subresultant_prs_projection_implemented | not)
    and (.matched_classical.projection_resource_comparison_authorized | not)
    and .matched_classical.same_exact_final_relation
    and .matched_classical.public_arithmetic_circuit_retained
    and .matched_classical.executed_only_after_carrier_release
    and (.matched_classical.root_enumeration_used | not)
    and (.matched_classical.computational_advantage | not)
    and (.matched_classical.distinct_phase_resource | not)
    and (.separate_reference.imports_m236_production | not)
    and .separate_reference.uses_independent_sparse_dictionary_quotient_arithmetic
    and .separate_reference.implements_independent_custody_state_machine
    and .separate_reference.case_control_reuse_parity
    and (.claim_limits | to_entries | all(.value == false))
    and (.terminal | not)
' "$generated_result" >/dev/null

jq -e '
    .schema == "cat_cas.cubic_airy_chain_quotient_algebra_reference.v1"
    and (.cases | length == 10)
    and (.cases | all(
        .endpoint_polynomial_degree == .quotient_dimension
        and .critical_value_relation_degree == .quotient_dimension
        and .endpoint_polynomial_square_free
        and .critical_value_relation_square_free
        and .critical_value_relation_annihilates_resident_action
        and .canonical_post_inverse_state_exact
        and .same_r_s_h_modulus_backings
        and (.baseline_reload_used | not)
    ))
    and .reuse.fresh_restored_boundary_agreement
    and .reuse.same_backing_across_primary_and_reuse
    and .reuse.restoration_generation_after_reuse == 2
    and (.imports_m236_production | not)
    and .uses_independent_sparse_dictionary_quotient_arithmetic
    and .implements_independent_custody_state_machine
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
if "cubic_airy_chain_quotient_algebra" in imports:
    raise SystemExit("M236 separate reference imports production")
PY

if rg -q 'itertools|CartesianProduct|assignment_table|truth_table|root_list|nroots|(^|[^[:alnum:]_])solve\(' "$production" "$reference"; then
    echo "M236 source contains a forbidden branch enumeration or assignment construct" >&2
    exit 2
fi

echo "QUALIFIED_CUBIC_AIRY_CHAIN_QUOTIENT_ALGEBRA_STRICT_SCOPE"
