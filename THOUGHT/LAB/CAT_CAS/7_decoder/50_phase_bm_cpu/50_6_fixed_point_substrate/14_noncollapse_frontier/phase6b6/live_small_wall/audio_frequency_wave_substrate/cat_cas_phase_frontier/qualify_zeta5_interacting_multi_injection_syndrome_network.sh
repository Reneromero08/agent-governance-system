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
    echo "RAM-backed M240 build forbidden" >&2
    exit 2
    ;;
esac
case "$(findmnt -n -o FSTYPE -T "$build")" in
  tmpfs|ramfs)
    echo "RAM-backed M240 filesystem forbidden" >&2
    exit 2
    ;;
esac
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
prod="$here/zeta5_interacting_multi_injection_syndrome_network.py"
ref="$here/zeta5_interacting_multi_injection_syndrome_network_separate_reference.py"
sealed_ref="$here/ZETA5_INTERACTING_MULTI_INJECTION_SYNDROME_NETWORK_SEPARATE_REFERENCE.json"
sealed_result="$here/ZETA5_INTERACTING_MULTI_INJECTION_SYNDROME_NETWORK_RESULTS.json"
generated_ref="$build/ZETA5_INTERACTING_MULTI_INJECTION_SYNDROME_NETWORK_SEPARATE_REFERENCE.json"
generated_result="$build/ZETA5_INTERACTING_MULTI_INJECTION_SYNDROME_NETWORK_RESULTS.json"
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
  .result == "PASS_EXACT_ZETA5_INTERACTING_MULTI_INJECTION_SYNDROME_NETWORK_STRICT_SCOPE"
  and .classification == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE"
  and .verification_level == "SEPARATE_REFERENCE_PARITY"
  and .restoration_classification == "EXACT_ALGEBRAIC_RESTORATION"
  and (.cases|length)==8
  and ([.cases[]|.family]==[0,1,0,1,0,1,0,1])
  and ([.cases[]|.injection_count]==[1,1,2,2,3,3,4,4])
  and ([.cases[]|.wire_count]==[2,2,3,3,4,4,5,5])
  and ([.cases[]|.syndrome_consumer_count]==[7,7,14,14,21,21,28,28])
  and ([.cases[]|.resident_amplitude_field_cells]==[25,25,125,125,625,625,3125,3125])
  and ([.cases[]|.scratch_amplitude_field_cells]==[25,25,125,125,625,625,3125,3125])
  and ([.cases[]|.restoration_generation]==[1,2,1,2,1,2,1,2])
  and (.cases|all(
    .canonical_post_inverse_state_exact
    and .same_amplitude_and_scratch_backings
    and .retained_final_boundary_during_inverse
    and .response_released_after_restoration
    and (.baseline_reload_used|not)
    and .work.retained_dynamic_inverse_history_entries==0
    and .matched_streamed_scalar_baseline.selected_data_probability==.selected_data_probability
    and .matched_streamed_scalar_baseline.selected_boundary_slice_commitment==.selected_boundary_slice_commitment
    and .matched_streamed_scalar_baseline.resident_boundary_accumulator_field_values==5
    and .matched_streamed_scalar_baseline.peak_term_field_values==1
    and .matched_streamed_scalar_baseline.materialized_assignment_table_entries==0
    and (.matched_streamed_scalar_baseline.full_amplitude_vector_retained|not)
  ))
  and (.controls|to_entries|map(select(
    .key!="accepted_path_serializes_syndrome_values"
    and .key!="accepted_path_materializes_assignment_or_history_table"
    and .key!="public_compiler_reads_final_answer"
  ))|all(.value==true))
  and (.controls.accepted_path_serializes_syndrome_values|not)
  and (.controls.accepted_path_materializes_assignment_or_history_table|not)
  and (.controls.public_compiler_reads_final_answer|not)
  and .controls.dephasing_shared_syndrome_changes_selected_boundary_counts2_3_4
  and .controls.each_declared_injection_changes_final_state_commitment
  and .controls.single_magic_wigner_l1_exact_a1_a2
  and ([.magic_laws[]|.stabilizer_component_upper_bound]==[5,25,125,625])
  and (.magic_laws|all(.product_input_law_only and (.computational_lower_bound_established|not)))
  and .reuse.primary_generations==[1,1,1,1]
  and .reuse.reuse_generations==[2,2,2,2]
  and .reuse.same_backing_reuse_at_each_declared_count
  and .reuse.fresh_restored_parity
  and (.reuse.baseline_reload_used|not)
  and .composition_law.one_actual_shared_syndrome_wire
  and .composition_law.distinct_data_wires_receive_distinct_injections
  and .composition_law.post_injection_network_uses_only_clifford_sum_cz_fourier
  and .composition_law.shared_syndrome_consumed_before_final_boundary
  and (.composition_law.syndrome_values_projected|not)
  and .composition_law.exact_l1_multiplies_across_product_magic_inputs
  and .composition_law.post_injection_clifford_network_preserves_wigner_l1
  and .composition_law.l1_growth_is_not_a_classical_runtime_lower_bound
  and .composition_law.direct_process_logical_custody_only
  and .matched_classical.strongest_implemented=="PUBLIC_INJECTION_COMPILED_STREAMED_STABILIZER_COMPONENT_AND_SYNDROME_SCALAR_BOUNDARY_RECURRENCE"
  and .matched_classical.streamed_terms_by_count==[25,125,625,3125]
  and .matched_classical.stabilizer_component_upper_bounds==[5,25,125,625]
  and .matched_classical.resident_boundary_accumulator_field_values==5
  and .matched_classical.peak_term_field_values==1
  and .matched_classical.declared_accumulator_plus_term_field_values==6
  and (.matched_classical.assignment_table_materialized|not)
  and (.matched_classical.full_amplitude_vector_retained|not)
  and (.matched_classical.optimal_stabilizer_rank_or_extent_proved|not)
  and (.matched_classical.computational_advantage|not)
  and .resource_law.phase_resident_field_cells_by_count==[25,125,625,3125]
  and .resource_law.phase_equal_scratch_field_cells_by_count==[25,125,625,3125]
  and .resource_law.phase_resident_plus_scratch_backing_cells_by_count==[50,250,1250,6250]
  and .resource_law.classical_persistent_field_values==5
  and .resource_law.classical_peak_term_field_values==1
  and .resource_law.classical_declared_accumulator_plus_term_field_values==6
  and .resource_law.accepted_phase_backings_exceed_streamed_scalar_baseline
  and .resource_law.comparison_basis=="DECLARED_QZETA5_FIELD_BACKINGS_AND_SCALAR_ACCUMULATORS_NOT_WHOLE_TRANSACTION_LIVENESS"
  and .resource_law.public_descriptors_loop_coordinates_phase_integers_and_container_state_excluded_not_zero
  and .resource_law.retained_dynamic_inverse_history_entries==0
  and .resource_law.dephased_diagnostic_excluded_from_accepted_restoring_path
  and (.resource_law.whole_transaction_live_cell_and_payload_accounting_complete|not)
  and .resource_law.python_objects_allocator_hash_serialization_rss_excluded_not_zero
  and .resource_law.resource_verification_level=="PACKAGE_SELF_REVIEW"
  and (.separate_reference.imports_m240_m239_or_m237_production|not)
  and .separate_reference.independent_power_basis_arithmetic
  and .separate_reference.independent_full_amplitude_transaction
  and .separate_reference.independent_streamed_scalar_boundary
  and .separate_reference.independent_wigner_l1_reconstruction
  and .separate_reference.independent_custody_state_machine
  and (.claim_limits|to_entries|all(.value==false))
  and (.terminal|not)
' "$generated_result" >/dev/null

jq -e '
  .schema=="cat_cas.zeta5_interacting_multi_injection_syndrome_network_reference.v1"
  and (.cases|length)==8
  and (.cases|all(.canonical_post_inverse_state_exact and .same_amplitude_and_scratch_backings and .response_released_after_restoration and (.baseline_reload_used|not)))
  and (.controls|to_entries|map(select(
    .key!="accepted_path_serializes_syndrome_values"
    and .key!="accepted_path_materializes_assignment_or_history_table"
    and .key!="public_compiler_reads_final_answer"
  ))|all(.value==true))
  and (.imports_m240_m239_or_m237_production|not)
  and .independent_power_basis_arithmetic
  and .independent_full_amplitude_transaction
  and .independent_streamed_scalar_boundary
  and .independent_wigner_l1_reconstruction
  and .independent_custody_state_machine
' "$generated_ref" >/dev/null

python3 - "$prod" "$ref" <<'PY'
import ast, inspect, sys
from pathlib import Path
for filename in sys.argv[1:]:
    ast.parse(Path(filename).read_text())
tree = ast.parse(Path(sys.argv[2]).read_text())
imports = {
    alias.name for node in ast.walk(tree) if isinstance(node, ast.Import)
    for alias in node.names
} | {node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
for forbidden in (
    "zeta5_interacting_multi_injection_syndrome_network",
    "zeta5_coherent_magic_injection_syndrome_port",
    "zeta5_normalized_cubic_fourier_coherent_port",
):
    if forbidden in imports:
        raise SystemExit("standalone M240 reference imports production")
production = ast.parse(Path(sys.argv[1]).read_text())
accepted = {"apply_gate", "run_transaction"}
for node in production.body:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in accepted:
        calls = {getattr(call.func, "id", "") for call in ast.walk(node) if isinstance(call, ast.Call)}
        if "product" in calls:
            raise SystemExit("accepted M240 path uses assignment product")
PY
echo "QUALIFIED_ZETA5_INTERACTING_MULTI_INJECTION_SYNDROME_NETWORK_STRICT_SCOPE"
