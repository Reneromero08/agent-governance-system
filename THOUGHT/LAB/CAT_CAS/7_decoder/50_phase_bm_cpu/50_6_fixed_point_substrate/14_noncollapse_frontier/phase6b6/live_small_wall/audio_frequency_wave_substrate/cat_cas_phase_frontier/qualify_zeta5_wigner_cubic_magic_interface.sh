#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
if [[ $# -ne 1 || ! -d "$1" ]]; then echo "usage: $0 DISK_BACKED_BUILD_DIRECTORY" >&2; exit 2; fi
build=$(realpath -e -- "$1")
case "$build" in /dev/shm|/dev/shm/*|/run/shm|/run/shm/*) echo "RAM-backed M238 build forbidden" >&2; exit 2;; esac
case "$(findmnt -n -o FSTYPE -T "$build")" in tmpfs|ramfs) echo "RAM-backed M238 filesystem forbidden" >&2; exit 2;; esac
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
prod="$here/zeta5_wigner_cubic_magic_interface.py"
ref="$here/zeta5_wigner_cubic_magic_interface_separate_reference.py"
m237="$here/zeta5_normalized_cubic_fourier_coherent_port.py"
sealed_ref="$here/ZETA5_WIGNER_CUBIC_MAGIC_INTERFACE_SEPARATE_REFERENCE.json"
sealed_result="$here/ZETA5_WIGNER_CUBIC_MAGIC_INTERFACE_RESULTS.json"
gen_ref="$build/ZETA5_WIGNER_CUBIC_MAGIC_INTERFACE_SEPARATE_REFERENCE.json"
gen_result="$build/ZETA5_WIGNER_CUBIC_MAGIC_INTERFACE_RESULTS.json"
mkdir -p "$build/tmp" "$build/xdg-cache" "$build/pycache"
env TMPDIR="$build/tmp" TMP="$build/tmp" TEMP="$build/tmp" XDG_CACHE_HOME="$build/xdg-cache" PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX="$build/pycache" nice -n 10 ionice -c 3 python3 "$ref" > "$gen_ref"
env TMPDIR="$build/tmp" TMP="$build/tmp" TEMP="$build/tmp" XDG_CACHE_HOME="$build/xdg-cache" PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX="$build/pycache" nice -n 10 ionice -c 3 python3 "$prod" "$gen_ref" > "$gen_result"
cmp "$gen_ref" "$sealed_ref"
cmp "$gen_result" "$sealed_result"
[[ $(jq -r .source_dependencies.production_sha256 "$gen_result") == "$(sha256sum "$prod" | awk '{print $1}')" ]]
[[ $(jq -r .source_dependencies.separate_reference_sha256 "$gen_result") == "$(sha256sum "$ref" | awk '{print $1}')" ]]
[[ $(jq -r .source_dependencies.m237_algebra_sha256 "$gen_result") == "$(sha256sum "$m237" | awk '{print $1}')" ]]
[[ $(jq -r .source_sha256 "$gen_ref") == "$(sha256sum "$ref" | awk '{print $1}')" ]]
jq -e '
 .result == "PASS_EXACT_ZETA5_WIGNER_CUBIC_MAGIC_INTERFACE_STRICT_SCOPE"
 and .classification == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE"
 and .verification_level == "SEPARATE_REFERENCE_PARITY"
 and .restoration_classification == "EXACT_ALGEBRAIC_RESTORATION"
 and (.cases|length)==6
 and ([.cases[]|select(.family==0)|.interface_qudits]==[1,2,3])
 and ([.cases[]|select(.family==1)|.interface_qudits]==[1,2,3])
 and ([.cases[]|select(.family==0)|.resident_wigner_field_cells]==[25,625,15625])
 and ([.cases[]|select(.family==1)|.resident_wigner_field_cells]==[25,625,15625])
 and ([.cases[]|select(.family==0)|.negative_cell_count]==[5,20,675])
 and ([.cases[]|select(.family==1)|.negative_cell_count]==[5,20,675])
 and ([.cases[]|select(.family==0)|.final_denominator_exponent]==[2,3,4])
 and ([.cases[]|select(.family==1)|.final_denominator_exponent]==[2,3,4])
 and ([.cases[]|select(.family==0)|.final_total_exact_payload_bits]==[196,2923,66586])
 and ([.cases[]|select(.family==1)|.final_total_exact_payload_bits]==[196,2923,66586])
 and ([.cases[]|select(.family==0)|.matched_classical.final_amplitude_payload_bits]==[42,144,1000])
 and ([.cases[]|select(.family==1)|.matched_classical.final_amplitude_payload_bits]==[42,141,980])
 and (.cases|all(
   .resident_wigner_field_cells==([25,625,15625][.interface_qudits-1])
   and .resident_wigner_integer_coordinates==4*.resident_wigner_field_cells
   and .scratch_wigner_field_cells==.resident_wigner_field_cells
   and .scratch_wigner_integer_coordinates==.resident_wigner_integer_coordinates
   and .cubic_gate_count==.interface_qudits
   and .negative_mass_positive and .normalization_exact and .purity_exact
   and .canonical_post_inverse_state_exact and .same_wigner_and_scratch_backings
   and .restoration_generation==1 and (.baseline_reload_used|not)
   and .retained_final_aggregate_during_inverse
   and .work.retained_dynamic_inverse_history_entries==0
   and .work.exact_reality_checks==.resident_wigner_field_cells
   and .matched_classical.work.resident_amplitude_field_cells==.hilbert_amplitude_cells
   and .matched_classical.work.amplitude_scratch_field_cells==.hilbert_amplitude_cells
   and .matched_classical.work.resident_wigner_field_cells==0
   and .matched_classical.negative_cell_count==.negative_cell_count
   and .matched_classical.negative_mass==.negative_mass
   and .matched_classical.selected_boundary_commitment==.selected_boundary_commitment
   and .matched_classical.final_wigner_commitment==.final_wigner_commitment))
 and (.controls|to_entries|map(select(
   .key!="resident_wigner_cells_serialized"
   and .key!="negative_cell_coordinates_serialized"
   and .key!="amplitudes_or_density_matrices_serialized"
   and .key!="path_histories_or_assignments_enumerated"
   and .key!="precomputed_cubic_kernel_tables_retained"
   and .key!="public_compiler_reads_final_answer"))|all(.value==true))
 and (.controls.resident_wigner_cells_serialized|not)
 and (.controls.negative_cell_coordinates_serialized|not)
 and (.controls.amplitudes_or_density_matrices_serialized|not)
 and (.controls.path_histories_or_assignments_enumerated|not)
 and (.controls.precomputed_cubic_kernel_tables_retained|not)
 and (.controls.public_compiler_reads_final_answer|not)
 and .reuse.primary.restoration_generation==1
 and .reuse.reuse.restoration_generation==2
 and .reuse.fresh_reuse.restoration_generation==1
 and .reuse.restoration_generation_after_reuse==2
 and .reuse.fresh_restored_boundary_agreement
 and .reuse.fresh_restored_resource_signature_agreement
 and .reuse.same_backing_across_primary_and_reuse
 and .phase_resource_diagnostic.formalism=="GROSS_ODD_PRIME_DISCRETE_WIGNER_P5"
 and .phase_resource_diagnostic.interfaces==[1,2,3]
 and .phase_resource_diagnostic.resident_wigner_cells==[25,625,15625]
 and .phase_resource_diagnostic.clifford_sham_nonnegative
 and .phase_resource_diagnostic.cubic_causes_exact_negative_mass
 and .phase_resource_diagnostic.stabilizer_relative_magic_witness
 and (.phase_resource_diagnostic.distinct_resource_unavailable_to_compact_classical_software|not)
 and .resource_law.resident_wigner_field_cells==[25,625,15625]
 and .resource_law.resident_wigner_integer_coordinates==[100,2500,62500]
 and .resource_law.same_sized_wigner_scratch_retained
 and .resource_law.stronger_amplitude_persistent_field_cells==[5,25,125]
 and .resource_law.stronger_amplitude_scratch_field_cells==[5,25,125]
 and (.resource_law.streamed_reference_retains_wigner_grid|not)
 and .resource_law.peak_formula_generated_cubic_kernel_field_cells==125
 and (.resource_law.interface_synergy_established|not)
 and (.resource_law.tensor_product_factorization_excluded|not)
 and (.resource_law.whole_transaction_live_cell_and_payload_accounting_complete|not)
 and .resource_law.python_objects_allocator_hash_serialization_rss_excluded_not_zero
 and .resource_law.resource_verification_level=="PACKAGE_SELF_REVIEW"
 and .matched_classical.clifford_sham=="AFFINE_LAGRANGIAN_STABILIZER_WIGNER_SUPPORT_O_N2_F5_STATE"
 and .matched_classical.cubic_circuit=="EXACT5_TO_THE_N_AMPLITUDE_RECURRENCE_PLUS_STREAMED_FINAL_WIGNER_AGGREGATE"
 and .matched_classical.phase_carrier_resident_cells_at_n3==15625
 and .matched_classical.amplitude_baseline_resident_cells_at_n3==125
 and .matched_classical.amplitude_baseline_is_smaller
 and (.matched_classical.explicit_gate_depth_path_sum_used|not)
 and (.matched_classical.computational_advantage|not)
 and (.matched_classical.distinct_phase_resource|not)
 and (.separate_reference.imports_m238_or_m237_production|not)
 and .separate_reference.evolves_exact_amplitudes
 and .separate_reference.streams_final_wigner_reconstruction
 and .separate_reference.implements_independent_custody_state_machine
 and (.claim_limits|to_entries|all(.value==false))
 and (.terminal|not)
' "$gen_result" >/dev/null
jq -e '
 .schema=="cat_cas.zeta5_wigner_cubic_magic_interface_reference.v1"
 and (.cases|length)==6
 and ([.cases[].interface_qudits]==[1,2,3,1,2,3])
 and (.cases|all(.negative_mass_positive and .normalization_exact and .purity_exact and .canonical_post_inverse_state_exact and .same_wigner_and_scratch_backings and .restoration_generation==1 and (.baseline_reload_used|not) and (.final_wigner_commitment|length)==64))
 and .reuse.primary.restoration_generation==1
 and .reuse.reuse.restoration_generation==2
 and .reuse.fresh_reuse.restoration_generation==1
 and .reuse.restoration_generation_after_reuse==2
 and .reuse.fresh_restored_boundary_agreement
 and .reuse.fresh_restored_resource_signature_agreement
 and .reuse.same_backing_across_primary_and_reuse
 and (.imports_m238_or_m237_production|not)
 and .evolves_exact_amplitudes
 and .streams_final_wigner_reconstruction
 and .implements_independent_custody_state_machine
' "$gen_ref" >/dev/null
python3 - "$ref" <<'PY'
import ast,sys
from pathlib import Path
tree=ast.parse(Path(sys.argv[1]).read_text())
imports={a.name for node in ast.walk(tree) if isinstance(node,ast.Import) for a in node.names}
imports|={node.module or "" for node in ast.walk(tree) if isinstance(node,ast.ImportFrom)}
for forbidden in ("zeta5_wigner_cubic_magic_interface", "zeta5_normalized_cubic_fourier_coherent_port"):
    if forbidden in imports: raise SystemExit(f"reference imports {forbidden}")
PY
rg -q 'CAT_CAS_EXACT_WIGNER_CELL_SEQUENCE_V1' "$prod" "$ref"
rg -q 'for index in range\(last_index - 1, -1, -1\):' "$prod"
rg -q 'for index in range\(len\(program.gates\) - 2, -1, -1\):' "$ref"
if rg -q 'itertools|CartesianProduct|assignment_table|truth_table|path_list|precomputed_kernel' "$prod" "$ref"; then
  echo "forbidden answer-bearing enumeration construct" >&2; exit 2
fi
echo "QUALIFIED_ZETA5_WIGNER_CUBIC_MAGIC_INTERFACE_STRICT_SCOPE"
