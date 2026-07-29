#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
export PYTHONHASHSEED=0
ulimit -c 0

if [[ $# -ne 1 ]]; then
  echo "usage: qualify_phase_vm_root_bisimulation.sh EVIDENCE_DIR" >&2
  exit 2
fi

evidence_dir=$1
frontier_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo=$(git -C "$frontier_dir" rev-parse --show-toplevel)
python="$repo/.venv/bin/python"
source_path="$frontier_dir/phase_vm_root_bisimulation.c"
backend_path="$frontier_dir/streaming_phase_vm.c"
oracle_path="$frontier_dir/phase_vm_root_bisimulation_oracle.py"
qualifier_path="$frontier_dir/qualify_phase_vm_root_bisimulation.sh"
binary="$evidence_dir/phase_vm_root_bisimulation"
ubsan_binary="$evidence_dir/phase_vm_root_bisimulation_ubsan"
result="$evidence_dir/result.json"
replay="$evidence_dir/replay.json"
ubsan_result="$evidence_dir/ubsan.json"
oracle_result="$evidence_dir/oracle.json"

mkdir -p "$evidence_dir"
export PYTHONPYCACHEPREFIX="$evidence_dir/pycache"

for tool in gcc jq cmp sha256sum nice rg; do
  command -v "$tool" >/dev/null
done
test -x "$python"

gcc \
  -std=c11 \
  -O2 \
  -Wall \
  -Wextra \
  -Wpedantic \
  -Werror \
  "$source_path" \
  -lm \
  -o "$binary"

gcc \
  -std=c11 \
  -O1 \
  -g \
  -Wall \
  -Wextra \
  -Wpedantic \
  -Werror \
  -fsanitize=undefined \
  -fno-sanitize-recover=all \
  "$source_path" \
  -lm \
  -o "$ubsan_binary"

nice -n 10 "$binary" >"$result" 2>"$evidence_dir/result.stderr"
nice -n 10 "$binary" >"$replay" 2>"$evidence_dir/replay.stderr"
nice -n 10 "$ubsan_binary" \
  >"$ubsan_result" 2>"$evidence_dir/ubsan.stderr"

test ! -s "$evidence_dir/result.stderr"
test ! -s "$evidence_dir/replay.stderr"
test ! -s "$evidence_dir/ubsan.stderr"
cmp "$result" "$replay"
cmp "$result" "$ubsan_result"

"$python" -m py_compile "$oracle_path"
nice -n 10 "$python" -X dev "$oracle_path" "$result" \
  >"$oracle_result" 2>"$evidence_dir/oracle.stderr"
test ! -s "$evidence_dir/oracle.stderr"

jq -e '
  .result == "PASS"
  and .claim_candidate
    == "BOUNDED_ROOT_LOCKED_PHASE_VM_OPERATION_TRACE_CLASSICAL_BISIMULATION_WITH_NUMERICAL_RESTORATION_AND_REUSE"
  and .alphabet == "Q3_ROOTS_OF_UNITY"
  and .native_backend == "streaming_phase_vm.c"
  and .scope == "ROOT_LOCKED_PUBLIC_HOLO_OPERATION_DOMAIN"
  and .exhaustive.registers == 5
  and .exhaustive.operation_variants == 8
  and .exhaustive.input_states_per_variant == 243
  and .exhaustive.operation_cases == 1944
  and .exhaustive.forward_inverse_checkpoints == 3888
  and .exhaustive.cswap_active_cases == 81
  and .exhaustive.pcswap_active_cases == 54
  and .chained_transactions.registers == 8
  and .chained_transactions.primary_forward_steps == 12
  and .chained_transactions.reuse_forward_steps == 8
  and .chained_transactions.primary_boundary == [2,0,1,1,2,0,0,1]
  and .chained_transactions.reuse_boundary == [0,0,1,2,2,0,0,1]
  and .chained_transactions.fresh_boundary == [0,0,1,2,2,0,0,1]
  and .chained_transactions.fresh_restored_boundary_equal
  and .chained_transactions.same_carrier_backing_reused
  and .chained_transactions.restoration_generation_sequence == [1,2]
  and .chained_transactions.baseline_reload_bytes == 0
  and .trace.semantic_trace_fnv1a64 == "76c64491b21a63d2"
  and .trace.checkpoints == 3928
  and .trace.compared_relation_cells == 19760
  and .trace.intermediate_state_inspected_by_diagnostic
  and (.trace.intermediate_state_emitted | not)
  and .numerics.predeclared_tolerance == 2e-11
  and .numerics.maximum_root_distance <= 2e-11
  and .numerics.maximum_restoration_error <= 2e-11
  and .numerics.primary_restoration_error <= 2e-11
  and .numerics.reuse_restoration_error <= 2e-11
  and (.controls.missing_inverse_restored | not)
  and .controls.missing_inverse_error > 1
  and (.controls.wrong_inverse_restored | not)
  and .controls.wrong_inverse_error > 1
  and .controls.reordered_inverse_applicable
  and (.controls.reordered_inverse_restored | not)
  and .controls.reordered_inverse_error > 1
  and .matched_compact_classical.representation
    == "UINT8_Q3_SYMBOL_PER_REGISTER"
  and .matched_compact_classical.transition_parity_after_every_operation
  and .matched_compact_classical.inverse_parity_after_every_operation
  and .matched_compact_classical.implementation_payload_bytes_per_register == 1
  and .matched_compact_classical.two_bit_packing_available
  and .matched_compact_classical.program_specific_optimization_may_be_smaller
  and .resource_law.native_complex128_rails_per_register == 2
  and .resource_law.native_heap_payload_bytes_per_register == 32
  and .resource_law.classical_uint8_payload_bytes_per_register == 1
  and .resource_law.exhaustive_native_heap_payload_bytes == 160
  and .resource_law.exhaustive_classical_payload_bytes == 5
  and .resource_law.chain_native_heap_payload_bytes == 256
  and .resource_law.chain_classical_payload_bytes == 8
  and .resource_law.public_program_descriptor_shared_between_paths
  and .resource_law.trace_buffers_are_verification_only
  and (.resource_law.runtime_advantage_claimed | not)
  and (.resource_law.whole_process_peak_claimed | not)
  and .restoration_classification == "NUMERICAL_PHYSICAL_STATE_RESTORATION"
  and .lemma_scope == "FINITE_DETERMINISTIC_SOFTWARE_TRANSITION_SYSTEMS_ONLY"
  and .exceptions_not_adjudicated == [
    "PHYSICAL_ANALOG_RESOURCES",
    "EXTERNAL_ORACLES",
    "NONDETERMINISTIC_RESOURCES",
    "UNBOUNDED_PRECISION_MODELS"
  ]
  and (.catvm_custody_established | not)
  and (.distinct_phase_resource_established | not)
  and (.computational_advantage | not)
  and (.small_wall_crossed | not)
  and (.physical_waveform_execution | not)
  and (.physical_bit_replacement | not)
  and (.unbounded_computation_established | not)
  and .claim_ceiling
    == "LINUX_X86_64_DIRECT_PROCESS_COMPLEX128_Q3_ROOT_LOCKED_STREAMING_PHASE_VM_SIX_OPCODES_EXHAUSTIVE_FIVE_REGISTER_LOCAL_DOMAIN_AND_TWO_EIGHT_REGISTER_CHAINED_PROGRAMS_SOFTWARE_ONLY"
  and (.terminal | not)
' "$result" >/dev/null

jq -e '
  .result == "PASS"
  and .oracle
    == "INDEPENDENT_MPMATH80_Q3_INTERPOLATION_AND_SYMBOLIC_TRANSITION_REEXECUTION"
  and (.production_backend_imported | not)
  and .precision_decimal_digits == 80
  and .operation_cases == 1944
  and .operation_variants == 8
  and .input_states_per_variant == 243
  and .cswap_active_cases == 81
  and .pcswap_active_cases == 54
  and .complex_formula_maximum_root_error <= 1e-70
  and .semantic_trace_fnv1a64 == "76c64491b21a63d2"
  and .production_trace_hash_matches
  and .primary_boundary == [2,0,1,1,2,0,0,1]
  and .primary_boundary_matches
  and .reuse_boundary == [0,0,1,2,2,0,0,1]
  and .reuse_boundary_matches
  and .fresh_restored_reuse_boundary_equal
  and (.controls.missing_inverse_restored | not)
  and (.controls.wrong_inverse_restored | not)
  and (.controls.reordered_inverse_restored | not)
  and .all_six_native_opcode_semantics_reconstructed
  and .root_locked_symbolic_state_is_sufficient
  and .classical_uint8_payload_bytes_per_register == 1
  and .two_bit_packing_available
  and .finite_deterministic_identity_simulation_lemma_valid
  and .lemma_scope == "FINITE_DETERMINISTIC_SOFTWARE_TRANSITION_SYSTEMS_ONLY"
  and (.catvm_custody_established | not)
  and (.distinct_phase_resource_established | not)
  and (.computational_advantage | not)
  and (.small_wall_crossed | not)
  and (.terminal | not)
' "$oracle_result" >/dev/null

if rg -n \
  'witness_list|candidate_set|truth_table|assignment_expansion|dense_operator' \
  "$source_path" "$oracle_path"
then
  echo "root-bisimulation package contains forbidden extensional state" >&2
  exit 1
fi

jq -n \
  --slurpfile accepted "$result" \
  --slurpfile oracle "$oracle_result" '
  {
    result: "PASS",
    claim:
      "BOUNDED_ROOT_LOCKED_PHASE_VM_OPERATION_TRACE_CLASSICAL_BISIMULATION_WITH_NUMERICAL_RESTORATION_AND_REUSE",
    classification: "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
    claim_ceiling:
      "LINUX_X86_64_DIRECT_PROCESS_COMPLEX128_Q3_ROOT_LOCKED_STREAMING_PHASE_VM_SIX_OPCODES_EXHAUSTIVE_FIVE_REGISTER_LOCAL_DOMAIN_AND_TWO_EIGHT_REGISTER_CHAINED_PROGRAMS_SOFTWARE_ONLY",
    verification_level: "INDEPENDENT_ORACLE_REEXECUTION",
    restoration_classification: "NUMERICAL_PHYSICAL_STATE_RESTORATION",
    accepted: $accepted[0],
    independent_oracle: $oracle[0],
    finite_deterministic_software_obstruction:
      "ROOT_LOCKED_PHASE_VM_HAS_OPERATION_LEVEL_Q3_SYMBOLIC_BISIMULATION_AND_SMALLER_MATCHED_SEMANTIC_STATE",
    identity_simulation_lemma_scope:
      "FINITE_DETERMINISTIC_SOFTWARE_TRANSITION_SYSTEMS_ONLY",
    distinct_phase_resource_established: false,
    computational_advantage: false,
    small_wall_crossed: false,
    catvm_custody_established: false,
    physical_waveform_execution: false,
    physical_bit_replacement: false,
    unbounded_computation_established: false,
    terminal: false
  }
' >"$evidence_dir/qualification.json"

hash_with_label() {
  local digest
  digest=$(sha256sum "$1")
  printf '%s  %s\n' "${digest%% *}" "$2"
}

{
  hash_with_label "$source_path" "phase_vm_root_bisimulation.c"
  hash_with_label "$backend_path" "streaming_phase_vm.c"
  hash_with_label "$oracle_path" "phase_vm_root_bisimulation_oracle.py"
  hash_with_label "$qualifier_path" "qualify_phase_vm_root_bisimulation.sh"
  hash_with_label "$binary" "phase_vm_root_bisimulation"
  hash_with_label "$ubsan_binary" "phase_vm_root_bisimulation_ubsan"
  hash_with_label "$result" "result.json"
  hash_with_label "$oracle_result" "oracle.json"
  hash_with_label "$evidence_dir/qualification.json" "qualification.json"
} >"$evidence_dir/SHA256SUMS"

echo "root-locked phase-VM bisimulation qualification: PASS"
