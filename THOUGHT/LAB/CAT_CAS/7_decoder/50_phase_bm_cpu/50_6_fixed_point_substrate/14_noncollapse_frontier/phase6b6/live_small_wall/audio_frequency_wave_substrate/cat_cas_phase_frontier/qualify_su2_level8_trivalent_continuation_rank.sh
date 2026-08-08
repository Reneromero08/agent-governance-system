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
        echo "RAM-backed M233 build directory is forbidden" >&2
        exit 2
        ;;
esac
case "$(findmnt -n -o FSTYPE -T "$build")" in
    tmpfs|ramfs)
        echo "RAM-backed M233 build filesystem is forbidden" >&2
        exit 2
        ;;
esac

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
production="$here/su2_level8_trivalent_extension_dichotomy.py"
reference="$here/su2_level8_trivalent_extension_dichotomy_separate_reference.py"
predecessor="$here/SU2_LEVEL8_TRIVALENT_SHARED_CHANNEL_FMOVE_RESULTS.json"
sealed_reference="$here/SU2_LEVEL8_TRIVALENT_CONTINUATION_RANK_SEPARATE_REFERENCE.json"
sealed_result="$here/SU2_LEVEL8_TRIVALENT_CONTINUATION_RANK_RESULTS.json"
generated_reference="$build/SU2_LEVEL8_TRIVALENT_CONTINUATION_RANK_SEPARATE_REFERENCE.json"
generated_result="$build/SU2_LEVEL8_TRIVALENT_CONTINUATION_RANK_RESULTS.json"
mkdir -p "$build/tmp" "$build/xdg-cache" "$build/pycache"

env TMPDIR="$build/tmp" TMP="$build/tmp" TEMP="$build/tmp" \
    XDG_CACHE_HOME="$build/xdg-cache" PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPYCACHEPREFIX="$build/pycache" PYTHONPATH="$here" \
    nice -n 10 ionice -c 3 python3 "$reference" > "$generated_reference"
env TMPDIR="$build/tmp" TMP="$build/tmp" TEMP="$build/tmp" \
    XDG_CACHE_HOME="$build/xdg-cache" PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPYCACHEPREFIX="$build/pycache" PYTHONPATH="$here" \
    nice -n 10 ionice -c 3 python3 "$production" "$generated_reference" \
    "$predecessor" > "$generated_result"

cmp "$generated_reference" "$sealed_reference"
cmp "$generated_result" "$sealed_result"

jq -e '
    .result == "PASS_EXACT_GROWING_SU2_LEVEL8_TRIVALENT_CONTINUATION_RANKS_REJECT_FIXED_TWO_AND_NINE_STATE_QUOTIENTS_THROUGH_N10"
    and .classification == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE"
    and .verification_level == "SEPARATE_REFERENCE_PARITY"
    and .restoration_classification == "EXACT_ALGEBRAIC_RESTORATION"
    and ([.rank_law[].strands] == [4,6,8,10])
    and ([.rank_law[].fusion_path_dimension] == [2,5,14,42])
    and ([.rank_law[].reachable_rank_both_primes] == [[2,2],[5,5],[14,14],[42,42]])
    and ([.rank_law[].observable_rank_both_primes] == [[2,2],[5,5],[14,14],[42,42]])
    and ([.rank_law[].continuation_hankel_rank_both_primes] == [[2,2],[5,5],[14,14],[42,42]])
    and (.rank_law | all(.full_minimal_qzeta40_linear_continuation_dimension_certified))
    and (.rank_certificates | length == 8)
    and (.rank_certificates | all(
        .all_ranks_full
        and .reachable_rank == .fusion_path_dimension
        and .observable_rank == .fusion_path_dimension
        and .continuation_hankel_rank == .fusion_path_dimension
        and .verification_work.retained_path_records == .fusion_path_dimension
        and .verification_work.peak_hankel_field_cells == (.fusion_path_dimension * .fusion_path_dimension)
    ))
    and ([.rank_certificates[].prime] | unique == [241,401])
    and (.transactions | length == 4)
    and ([.transactions[].fusion_path_dimension] == [2,5,14,42])
    and (.transactions | all(
        .primary_restoration_error_field_cells == 0
        and .reuse_restoration_error_field_cells == 0
        and .primary_same_coefficient_backing
        and .reuse_same_coefficient_backing
        and .fresh_same_coefficient_backing
        and .primary_canonical_post_restoration_state_exact
        and .reuse_canonical_post_restoration_state_exact
        and .fresh_canonical_post_restoration_state_exact
        and .primary_missing_inverse_error_nonzero
        and .fresh_restored_reuse_boundary_agreement
        and .fresh_restored_reuse_state_agreement
        and .restoration_generation_after_reuse == 2
        and .fresh_restoration_generation == 1
        and (.baseline_reload_used | not)
    ))
    and (.controls | to_entries | all(.value == true))
    and .mechanism_law.multiple_overlapping_shared_channels
    and .mechanism_law.all_adjacent_noncommuting_braid_consumers
    and .mechanism_law.accepted_generator_action_is_streamed_local_one_or_two_cell_blocks
    and (.mechanism_law.accepted_dense_operator_materialized | not)
    and (.mechanism_law.accepted_fusion_relation_table_materialized | not)
    and (.mechanism_law.accepted_assignment_expansion_materialized | not)
    and .mechanism_law.global_fusion_path_coefficient_basis_materialized
    and .mechanism_law.final_vacuum_plat_scalar_only
    and .mechanism_law.forward_state_only_one_way_committed
    and .mechanism_law.direct_process_logical_custody_only
    and .mechanism_law.rank_verifier_materializes_paths_bases_and_hankel_matrix
    and ([.resource_law.accepted_transactions[].accepted_carrier_field_cells] == [2,5,14,42])
    and ([.resource_law.accepted_transactions[].retained_final_boundary_field_cells_during_inverse] == [1,1,1,1])
    and ([.resource_law.accepted_transactions[].accepted_carrier_plus_retained_boundary_field_cells] == [3,6,15,43])
    and (.resource_law.accepted_transactions | all(.retained_inverse_history_entries == 0))
    and .resource_law.rank_verifier_paths_modular_bases_and_hankel_are_verifier_only
    and .resource_law.rank_verifier_not_accepted_runtime_output
    and .resource_law.resource_verification_level == "PACKAGE_SELF_REVIEW"
    and .resource_law.verifier_peak_basis_and_hankel_cells_are_component_local_not_combined
    and (.resource_law.whole_transaction_live_field_and_payload_accounting_complete | not)
    and .obstruction.fixed_two_channel_extension_rejected_through_n10
    and .obstruction.fixed_nine_charge_label_linear_quotient_rejected_by_n8
    and .obstruction.any_fixed_qzeta40_linear_rank_at_most41_rejected_through_n10
    and (.obstruction.uniform_fixed_rank_linear_continuation_quotient_for_unbounded_family_rejected | not)
    and (.obstruction.all_n_continuation_rank_theorem_established | not)
    and (.obstruction.nonlinear_or_non_qzeta40_quotient_excluded | not)
    and .matched_classical.same_sparse_path_state_cells
    and .matched_classical.same_local_block_work
    and (.matched_classical.smaller_qzeta40_linear_continuation_quotient_exists_for_declared_language | not)
    and (.matched_classical.treewidth_optimized_scalar_word_baseline_implemented | not)
    and (.matched_classical.computational_advantage | not)
    and (.matched_classical.distinct_phase_resource | not)
    and (.separate_reference.imports_m233_production | not)
    and (.separate_reference.imports_m232_production | not)
    and (.separate_reference.imports_m214_production | not)
    and .separate_reference.uses_independent_cyclotomic_polynomial_oracle
    and .separate_reference.exact_transaction_parity
    and .separate_reference.continuation_rank_parity_at_distinct_primes
    and (.claim_limits.fixed_qzeta40_linear_quotient_rank_at_most41_through_n10 | not)
    and (.claim_limits.unbounded_continuation_rank_growth_proved | not)
    and (.claim_limits.nonlinear_quotient_excluded | not)
    and (.claim_limits.general_tensor_network_closure | not)
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
    .schema == "cat_cas.su2_level8_trivalent_continuation_rank_reference.v1"
    and .certificate_primes == [641,881]
    and ([.rank_law[].continuation_hankel_rank_both_primes] == [[2,2],[5,5],[14,14],[42,42]])
    and (.rank_cases | length == 8)
    and (.rank_cases | all(
        .reachable_rank == .fusion_path_dimension
        and .observable_rank == .fusion_path_dimension
        and .continuation_hankel_rank == .fusion_path_dimension
        and .dense_generator_matrices_are_verifier_only
    ))
    and (.transactions | length == 4)
    and (.transactions | all(
        .primary_canonical_post_restoration_state_exact
        and .reuse_canonical_post_restoration_state_exact
        and .fresh_canonical_post_restoration_state_exact
        and .primary_same_coefficient_backing
        and .reuse_same_coefficient_backing
        and .fresh_same_coefficient_backing
        and .fresh_restored_reuse_boundary_agreement
        and .fresh_restored_reuse_state_agreement
        and (.baseline_reload_used | not)
    ))
    and (.controls | to_entries | all(.value == true))
    and (.imports_m233_production | not)
    and (.imports_m232_production | not)
    and (.imports_m214_production | not)
    and .uses_independent_cyclotomic_polynomial_oracle
    and .uses_independent_dense_generator_verifier
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
    "su2_level8_trivalent_extension_dichotomy",
    "su2_level8_trivalent_shared_channel_fmove",
    "su2_level8_fusion_path_braid_phase_relation",
    "su2_level8_fusion_path_mps_rank_no_go",
):
    if forbidden in imports:
        raise SystemExit(f"M233 separate reference imports production: {forbidden}")
PY

rg -Fq 'reachable = invariant_closure(' "$production"
rg -Fq 'observable = invariant_closure(' "$production"
rg -Fq 'hankel_rank = matrix_rank(hankel, prime, work)' "$production"
rg -Fq 'carrier.port.inverse(owner, program, index, work)' "$here/su2_level8_fusion_path_braid_phase_relation.py"
rg -Fq 'matrices = [' "$reference"
rg -Fq 'dense_generator_matrices_are_verifier_only' "$reference"
if rg -q 'itertools|CartesianProduct|assignment_table|truth_table' "$production" "$reference"; then
    echo "M233 accepted source contains an assignment/table expansion construct" >&2
    exit 2
fi

echo "QUALIFIED_SU2_LEVEL8_TRIVALENT_CONTINUATION_RANK_STRICT_SCOPE"
