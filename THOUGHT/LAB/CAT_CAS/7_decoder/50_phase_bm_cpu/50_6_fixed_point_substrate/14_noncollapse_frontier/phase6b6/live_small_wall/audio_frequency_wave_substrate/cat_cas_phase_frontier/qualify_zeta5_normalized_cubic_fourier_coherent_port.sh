#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
if [[ $# -ne 1 || ! -d "$1" ]]; then echo "usage: $0 DISK_BACKED_BUILD_DIRECTORY" >&2; exit 2; fi
build=$(realpath -e -- "$1")
case "$build" in /dev/shm|/dev/shm/*|/run/shm|/run/shm/*) echo "RAM-backed M237 build forbidden" >&2; exit 2;; esac
case "$(findmnt -n -o FSTYPE -T "$build")" in tmpfs|ramfs) echo "RAM-backed M237 filesystem forbidden" >&2; exit 2;; esac
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
prod="$here/zeta5_normalized_cubic_fourier_coherent_port.py"
ref="$here/zeta5_normalized_cubic_fourier_coherent_port_separate_reference.py"
sealed_ref="$here/ZETA5_NORMALIZED_CUBIC_FOURIER_COHERENT_PORT_SEPARATE_REFERENCE.json"
sealed_result="$here/ZETA5_NORMALIZED_CUBIC_FOURIER_COHERENT_PORT_RESULTS.json"
gen_ref="$build/ZETA5_NORMALIZED_CUBIC_FOURIER_COHERENT_PORT_SEPARATE_REFERENCE.json"
gen_result="$build/ZETA5_NORMALIZED_CUBIC_FOURIER_COHERENT_PORT_RESULTS.json"
mkdir -p "$build/tmp" "$build/xdg-cache" "$build/pycache"
env TMPDIR="$build/tmp" TMP="$build/tmp" TEMP="$build/tmp" XDG_CACHE_HOME="$build/xdg-cache" PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX="$build/pycache" nice -n 10 ionice -c 3 python3 "$ref" > "$gen_ref"
env TMPDIR="$build/tmp" TMP="$build/tmp" TEMP="$build/tmp" XDG_CACHE_HOME="$build/xdg-cache" PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX="$build/pycache" nice -n 10 ionice -c 3 python3 "$prod" "$gen_ref" > "$gen_result"
cmp "$gen_ref" "$sealed_ref"
cmp "$gen_result" "$sealed_result"
[[ $(jq -r .source_dependencies.production_sha256 "$gen_result") == "$(sha256sum "$prod" | awk '{print $1}')" ]]
[[ $(jq -r .source_dependencies.separate_reference_sha256 "$gen_result") == "$(sha256sum "$ref" | awk '{print $1}')" ]]
[[ $(jq -r .source_sha256 "$gen_ref") == "$(sha256sum "$ref" | awk '{print $1}')" ]]
rg -q 'for index in range\(len\(program.gates\) - 2, -1, -1\):' "$prod"
rg -q 'for i in range\(len\(program.gates\) - 2, -1, -1\):' "$ref"
jq -e '
 .result == "PASS_EXACT_ZETA5_NORMALIZED_CUBIC_FOURIER_COHERENT_PORT_STRICT_SCOPE"
 and .classification == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE"
 and .verification_level == "SEPARATE_REFERENCE_PARITY"
 and .restoration_classification == "EXACT_ALGEBRAIC_RESTORATION"
 and (.cases|length)==12
 and ([.cases[]|select(.family==0)|.depth]==[2,4,8,16,32,64])
 and ([.cases[]|select(.family==1)|.depth]==[2,4,8,16,32,64])
 and ([.cases[]|select(.family==0)|.final_denominator_exponent]==[1,1,2,3,5,9])
 and ([.cases[]|select(.family==1)|.final_denominator_exponent]==[1,1,2,3,5,9])
 and ([.cases[]|select(.family==0)|.final_total_exact_payload_bits]==[43,43,79,115,221,416])
 and ([.cases[]|select(.family==1)|.final_total_exact_payload_bits]==[45,43,72,128,228,418])
 and (.cases|all(.logical_amplitude_cells==5 and .resident_integer_numerator_coordinates==20 and .scratch_integer_numerator_coordinates==20 and .gate_count==(1+2*.depth) and .retained_final_amplitude_integer_coordinates_during_inverse==4 and .retained_final_amplitude_denominator_exponents_during_inverse==1 and .retained_final_amplitude_denominator_material_bits_during_inverse>0 and .retained_one_way_vector_commitment_digest_bytes_during_inverse==32 and .canonical_post_inverse_state_exact and .same_psi_and_scratch_backings and .restoration_generation==1 and (.baseline_reload_used|not) and .work.retained_dynamic_inverse_history_entries==0 and .matched_classical.amplitude_commitment==.amplitude_commitment and .matched_classical.probability_commitment==.probability_commitment and .matched_classical.final_vector_commitment==.final_vector_commitment))
 and (.controls|to_entries|map(select(.key!="intermediate_or_full_vectors_serialized" and .key!="path_histories_or_assignments_enumerated" and .key!="public_compiler_reads_final_answer"))|all(.value==true))
 and (.controls.intermediate_or_full_vectors_serialized|not)
 and (.controls.path_histories_or_assignments_enumerated|not)
 and (.controls.public_compiler_reads_final_answer|not)
 and .reuse.primary.restoration_generation==1 and .reuse.reuse.restoration_generation==2 and .reuse.fresh_reuse.restoration_generation==1
 and .reuse.restoration_generation_after_reuse==2 and .reuse.fresh_restored_boundary_agreement and .reuse.fresh_restored_resource_signature_agreement and .reuse.same_backing_across_primary_and_reuse
 and .coherent_phase_law.field=="Q(zeta_5)" and .coherent_phase_law.logical_amplitude_cells==5 and .coherent_phase_law.global_phase_preserved and .coherent_phase_law.coherent_destructive_interference_exact and (.coherent_phase_law.stationary_or_path_histories_enumerated|not) and .coherent_phase_law.final_boundary_only and .coherent_phase_law.direct_process_logical_custody_only
 and .resource_law.family0_final_denominator_exponents==[1,1,2,3,5,9]
 and .resource_law.family0_final_total_exact_payload_bits==[43,43,79,115,221,416]
 and .resource_law.family1_final_total_exact_payload_bits==[45,43,72,128,228,418]
 and .resource_law.denominator_material_value_counted and .resource_law.retained_dynamic_inverse_history_entries==0 and (.resource_law.carrier_initial_state_snapshot_retained|not) and (.resource_law.whole_transaction_live_cell_and_payload_accounting_complete|not) and .resource_law.resource_verification_level=="PACKAGE_SELF_REVIEW"
 and .matched_classical.strongest_implemented=="IDENTICAL_EXACT_FIVE_AMPLITUDE_QZETA5_RECURRENCE" and .matched_classical.equivalent_single_boundary_form=="BACKWARD_FIVE_COMPONENT_ROW_RECURRENCE" and .matched_classical.executed_only_after_carrier_release and (.matched_classical.explicit_path_sum_used|not) and (.matched_classical.dephased_density_matrix_used_as_baseline|not) and (.matched_classical.computational_advantage|not) and (.matched_classical.distinct_phase_resource|not)
 and (.separate_reference.imports_m237_production|not) and .separate_reference.uses_independent_exact_matrix_recurrence and .separate_reference.implements_independent_custody_state_machine
 and (.claim_limits|to_entries|all(.value==false)) and (.terminal|not)
' "$gen_result" >/dev/null
jq -e '.schema=="cat_cas.zeta5_normalized_cubic_fourier_coherent_port_reference.v1" and (.cases|length)==12 and (.cases|all(.retained_final_amplitude_integer_coordinates_during_inverse==4 and .retained_final_amplitude_denominator_exponents_during_inverse==1 and .retained_final_amplitude_denominator_material_bits_during_inverse>0 and .retained_one_way_vector_commitment_digest_bytes_during_inverse==32 and .canonical_post_inverse_state_exact and .same_psi_and_scratch_backings and (.baseline_reload_used|not))) and .reuse.fresh_restored_boundary_agreement and .reuse.fresh_restored_resource_signature_agreement and .reuse.restoration_generation_after_reuse==2 and (.imports_m237_production|not) and .uses_independent_exact_matrix_recurrence and .implements_independent_custody_state_machine' "$gen_ref" >/dev/null
python3 - "$ref" <<'PY'
import ast,sys
from pathlib import Path
t=ast.parse(Path(sys.argv[1]).read_text())
imports={a.name for n in ast.walk(t) if isinstance(n,ast.Import) for a in n.names}|{n.module or "" for n in ast.walk(t) if isinstance(n,ast.ImportFrom)}
if "zeta5_normalized_cubic_fourier_coherent_port" in imports: raise SystemExit("reference imports production")
PY
if rg -q 'itertools|CartesianProduct|assignment_table|truth_table|path_list' "$prod" "$ref"; then echo "forbidden enumeration construct" >&2; exit 2; fi
echo "QUALIFIED_ZETA5_NORMALIZED_CUBIC_FOURIER_COHERENT_PORT_STRICT_SCOPE"
