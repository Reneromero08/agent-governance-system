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
    echo "RAM-backed M239 build forbidden" >&2
    exit 2
    ;;
esac
case "$(findmnt -n -o FSTYPE -T "$build")" in
  tmpfs|ramfs)
    echo "RAM-backed M239 filesystem forbidden" >&2
    exit 2
    ;;
esac
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
prod="$here/zeta5_coherent_magic_injection_syndrome_port.py"
ref="$here/zeta5_coherent_magic_injection_syndrome_port_separate_reference.py"
sealed_ref="$here/ZETA5_COHERENT_MAGIC_INJECTION_SYNDROME_PORT_SEPARATE_REFERENCE.json"
sealed_result="$here/ZETA5_COHERENT_MAGIC_INJECTION_SYNDROME_PORT_RESULTS.json"
generated_ref="$build/ZETA5_COHERENT_MAGIC_INJECTION_SYNDROME_PORT_SEPARATE_REFERENCE.json"
generated_result="$build/ZETA5_COHERENT_MAGIC_INJECTION_SYNDROME_PORT_RESULTS.json"
mkdir -p "$build/tmp" "$build/xdg-cache" "$build/pycache"
run_env=(
  env
  TMPDIR="$build/tmp" TMP="$build/tmp" TEMP="$build/tmp"
  XDG_CACHE_HOME="$build/xdg-cache"
  PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX="$build/pycache"
)
"${run_env[@]}" nice -n 10 ionice -c 3 python3 "$ref" > "$generated_ref"
"${run_env[@]}" nice -n 10 ionice -c 3 python3 "$prod" "$generated_ref" > "$generated_result"
cmp "$generated_ref" "$sealed_ref"
cmp "$generated_result" "$sealed_result"
[[ $(jq -r .source_dependencies.production_sha256 "$generated_result") == "$(sha256sum "$prod" | awk '{print $1}')" ]]
[[ $(jq -r .source_dependencies.separate_reference_sha256 "$generated_result") == "$(sha256sum "$ref" | awk '{print $1}')" ]]
[[ $(jq -r .source_sha256 "$generated_ref") == "$(sha256sum "$ref" | awk '{print $1}')" ]]

jq -e '
  .result == "PASS_EXACT_ZETA5_COHERENT_MAGIC_INJECTION_SYNDROME_PORT_STRICT_SCOPE"
  and .classification == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE"
  and .verification_level == "SEPARATE_REFERENCE_PARITY"
  and .restoration_classification == "EXACT_ALGEBRAIC_RESTORATION"
  and (.cases|length)==6
  and ([.cases[]|.family]==[0,0,0,1,1,1])
  and ([.cases[]|.consumer_pair_count]==[1,2,4,1,2,4])
  and ([.cases[]|.syndrome_consumer_count]==[7,9,13,7,9,13])
  and ([.cases[]|.final_denominator_exponent]==[1,1,1,1,1,1])
  and ([.cases[]|.final_total_exact_payload_bits]==[144,120,141,144,120,122])
  and (.cases|all(
    .resident_amplitude_field_cells==25
    and .scratch_amplitude_field_cells==25
    and .selected_data_probability_nonzero
    and .normalization_exact
    and .retained_final_boundary_during_inverse
    and .canonical_post_inverse_state_exact
    and .same_amplitude_and_scratch_backings
    and .restoration_generation==1
    and (.baseline_reload_used|not)
    and .work.retained_dynamic_inverse_history_entries==0
    and .matched_compiled_injection_baseline.resident_field_cells==25
    and .matched_compiled_injection_baseline.peak_fourier_saved_input_field_cells==5
    and .matched_compiled_injection_baseline.peak_accumulator_field_cells==1
    and .matched_compiled_injection_baseline.compiled_injection_gates_removed==7
    and .matched_compiled_injection_baseline.stabilizer_component_upper_bound==5
    and .matched_compiled_injection_baseline.selected_data_probability==.selected_data_probability
    and .matched_compiled_injection_baseline.final_state_commitment==.final_state_commitment
  ))
  and (.controls|to_entries|map(select(
    .key!="syndrome_values_serialized"
    and .key!="accepted_in_place_path_retains_branch_assignment_expansion"
    and .key!="public_compiler_reads_final_answer"
  ))|all(.value==true))
  and (.controls.syndrome_values_serialized|not)
  and (.controls.accepted_in_place_path_retains_branch_assignment_expansion|not)
  and (.controls.public_compiler_reads_final_answer|not)
  and .controls.dephased_diagnostic_enumerates_five_syndrome_branches
  and .controls.dephased_measurement_is_nonrestoring_diagnostic_only
  and .controls.dephased_syndrome_changes_selected_boundary
  and .controls.remove_sum_consumer_changes_selected_boundary
  and .controls.remove_cz_consumer_changes_selected_boundary
  and .controls.adjacent_sum_cz_consumer_order_changes_full_state
  and .reuse.primary.restoration_generation==1
  and .reuse.reuse.restoration_generation==2
  and .reuse.fresh_reuse.restoration_generation==1
  and .reuse.restoration_generation_after_reuse==2
  and .reuse.fresh_restored_boundary_agreement
  and .reuse.fresh_restored_full_state_commitment_agreement
  and .reuse.fresh_restored_resource_signature_agreement
  and .reuse.same_backing_across_primary_and_reuse
  and .injection_law.field=="Q(zeta_5)"
  and .injection_law.syndrome_values==5
  and (.injection_law.same_resident_syndrome_consumers==["Q","L","G","SUM","CZ"])
  and .injection_law.fiberwise_corrections_are_stabilizer_for_fixed_syndrome
  and .injection_law.coherent_q_l_g_maps_are_not_claimed_clifford
  and (.injection_law.physical_measurement_performed|not)
  and .injection_law.direct_process_logical_custody_only
  and .matched_classical.strongest_implemented=="COMPLETE_INJECTION_IDENTITY_PLUS_EXACT25_AMPLITUDE_FIVE_INPUT_FOURIER_SCRATCH_RECURRENCE"
  and .matched_classical.stabilizer_component_upper_bound==5
  and .matched_classical.pairwise_overlap_upper_bound_for_ONE_BOUNDARY==25
  and .matched_classical.stabilizer_component_resource_implementation=="NOT_INSTRUMENTED"
  and (.matched_classical.consumer_depth_increases_component_bound|not)
  and (.matched_classical.computational_advantage|not)
  and (.matched_classical.distinct_phase_resource|not)
  and .resource_law.resident_amplitude_field_cells==25
  and .resource_law.scratch_amplitude_field_cells==25
  and .resource_law.strongest_implemented_classical_resident_field_cells==25
  and .resource_law.strongest_implemented_classical_peak_fourier_saved_inputs==5
  and .resource_law.strongest_implemented_classical_peak_accumulator_field_cells==1
  and .resource_law.accepted_phase_backings_exceed_matched_classical_field_values
  and .resource_law.retained_dynamic_inverse_history_entries==0
  and .resource_law.dephased_diagnostic_sequential_branch_count==5
  and .resource_law.dephased_diagnostic_field_cells_per_branch==25
  and .resource_law.dephased_diagnostic_excluded_from_accepted_restoring_path
  and (.resource_law.whole_transaction_live_cell_and_payload_accounting_complete|not)
  and .resource_law.python_objects_allocator_hash_serialization_rss_excluded_not_zero
  and .resource_law.resource_verification_level=="PACKAGE_SELF_REVIEW"
  and (.separate_reference.imports_m239_or_m237_production|not)
  and .separate_reference.independent_25_amplitude_recurrence
  and .separate_reference.independent_injection_identity
  and .separate_reference.independent_compiled_identity_25_amplitude_boundary
  and .separate_reference.independent_custody_state_machine
  and (.claim_limits|to_entries|all(.value==false))
  and (.terminal|not)
' "$generated_result" >/dev/null

jq -e '
  .schema=="cat_cas.zeta5_coherent_magic_injection_syndrome_port_reference.v1"
  and (.cases|length)==6
  and (.cases|all(.canonical_post_inverse_state_exact and .same_amplitude_and_scratch_backings and (.baseline_reload_used|not)))
  and .controls.dephased_syndrome_changes_selected_boundary
  and .controls.g_omission_exactly_factorizes_data_magic_and_syndrome_magic
  and (.controls.accepted_in_place_path_retains_branch_assignment_expansion|not)
  and .controls.dephased_diagnostic_enumerates_five_syndrome_branches
  and (.imports_m239_or_m237_production|not)
  and .independent_exact_25_amplitude_matrix_recurrence
  and .independent_injection_identity
  and .independent_compiled_identity_baseline
  and .independent_custody_state_machine
' "$generated_ref" >/dev/null

python3 - "$ref" <<'PY'
import ast, sys
from pathlib import Path
tree = ast.parse(Path(sys.argv[1]).read_text())
imports = {
    alias.name
    for node in ast.walk(tree) if isinstance(node, ast.Import)
    for alias in node.names
} | {
    node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
}
for forbidden in (
    "zeta5_coherent_magic_injection_syndrome_port",
    "zeta5_normalized_cubic_fourier_coherent_port",
):
    if forbidden in imports:
        raise SystemExit("standalone reference imports production")
PY
rg -q 'for gate_index in range\(last_index - 1, -1, -1\):' "$prod"
rg -q 'for i in range\(last - 1, -1, -1\):' "$ref"
if rg -q 'itertools|CartesianProduct|assignment_table|truth_table|path_list' "$prod" "$ref"; then
  echo "forbidden M239 expansion construct" >&2
  exit 2
fi
echo "QUALIFIED_ZETA5_COHERENT_MAGIC_INJECTION_SYNDROME_PORT_STRICT_SCOPE"
