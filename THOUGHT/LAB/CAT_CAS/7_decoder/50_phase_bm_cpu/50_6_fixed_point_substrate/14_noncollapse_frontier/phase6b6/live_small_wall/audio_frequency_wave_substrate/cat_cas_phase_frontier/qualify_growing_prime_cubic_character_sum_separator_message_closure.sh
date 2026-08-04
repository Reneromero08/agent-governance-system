#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(git -C "$HERE" rev-parse --show-toplevel)"
PYTHON="$ROOT/.venv/bin/python"
PRODUCTION="$HERE/growing_prime_cubic_character_sum_separator_message_closure.py"
GRAPH_DEPENDENCY="$HERE/growing_prime_resident_cubic_strength_character_graph_quotient.py"
RESIDENT_DEPENDENCY="$HERE/growing_prime_resident_cubic_strength_phase_port_weil_action_rank.py"
OPEN_ACTION_DEPENDENCY="$HERE/growing_prime_cubic_weil_open_interface_action_span.py"
COMPONENT_DEPENDENCY="$HERE/growing_prime_two_fiber_cubic_deformed_weil_component_rank.py"
GAUSSIAN_DEPENDENCY="$HERE/growing_prime_two_fiber_weil_gaussian_phase_kernel_closure.py"
ORACLE="$HERE/growing_prime_cubic_character_sum_separator_message_closure_independent_oracle.py"
SEALED="$HERE/GROWING_PRIME_CUBIC_CHARACTER_SUM_SEPARATOR_MESSAGE_CLOSURE_RESULTS.json"
SEALED_ORACLE="$HERE/GROWING_PRIME_CUBIC_CHARACTER_SUM_SEPARATOR_MESSAGE_CLOSURE_INDEPENDENT_ORACLE.json"
EVIDENCE="$(mktemp -d /dev/shm/ags-audio-m176-qualifier-XXXXXX)"

cp "$PRODUCTION" "$EVIDENCE/growing_prime_cubic_character_sum_separator_message_closure.py"
cp "$GRAPH_DEPENDENCY" "$EVIDENCE/growing_prime_resident_cubic_strength_character_graph_quotient.py"
cp "$RESIDENT_DEPENDENCY" "$EVIDENCE/growing_prime_resident_cubic_strength_phase_port_weil_action_rank.py"
cp "$OPEN_ACTION_DEPENDENCY" "$EVIDENCE/growing_prime_cubic_weil_open_interface_action_span.py"
cp "$COMPONENT_DEPENDENCY" "$EVIDENCE/growing_prime_two_fiber_cubic_deformed_weil_component_rank.py"
cp "$GAUSSIAN_DEPENDENCY" "$EVIDENCE/growing_prime_two_fiber_weil_gaussian_phase_kernel_closure.py"
cp "$ORACLE" "$EVIDENCE/growing_prime_cubic_character_sum_separator_message_closure_independent_oracle.py"
cp "$SEALED" "$EVIDENCE/sealed.json"
cp "$SEALED_ORACLE" "$EVIDENCE/sealed_oracle.json"

PYTHONPYCACHEPREFIX=/dev/shm "$PYTHON" -m py_compile \
  "$EVIDENCE/growing_prime_cubic_character_sum_separator_message_closure.py" \
  "$EVIDENCE/growing_prime_resident_cubic_strength_character_graph_quotient.py" \
  "$EVIDENCE/growing_prime_resident_cubic_strength_phase_port_weil_action_rank.py" \
  "$EVIDENCE/growing_prime_cubic_weil_open_interface_action_span.py" \
  "$EVIDENCE/growing_prime_two_fiber_cubic_deformed_weil_component_rank.py" \
  "$EVIDENCE/growing_prime_two_fiber_weil_gaussian_phase_kernel_closure.py" \
  "$EVIDENCE/growing_prime_cubic_character_sum_separator_message_closure_independent_oracle.py"

PYTHONPYCACHEPREFIX=/dev/shm "$PYTHON" \
  "$EVIDENCE/growing_prime_cubic_character_sum_separator_message_closure.py" \
  --output "$EVIDENCE/reexecuted.json"
cmp "$EVIDENCE/sealed.json" "$EVIDENCE/reexecuted.json"

PYTHONPYCACHEPREFIX=/dev/shm "$PYTHON" \
  "$EVIDENCE/growing_prime_cubic_character_sum_separator_message_closure_independent_oracle.py" \
  --production "$EVIDENCE/reexecuted.json" \
  --source "$EVIDENCE/growing_prime_cubic_character_sum_separator_message_closure.py" \
  --output "$EVIDENCE/reexecuted_oracle.json"
cmp "$EVIDENCE/sealed_oracle.json" "$EVIDENCE/reexecuted_oracle.json"

jq -e '
  (.cases | length) == 13 and
  ([.cases[].q] | unique) == [5,11,23,29,41,53,83,89,113] and
  ([.cases[] |
    .accepted_runtime_field_elements == (2*.q+4) and
    .accepted_runtime_bit_capacity_upper_bound == ((2*.q+1)*.data_and_message_scalar_field_cell_bit_capacity+3*.message_exponent_field_cell_bit_capacity) and
    .public_morphism_node_records == (4*.depth+1) and
    .public_morphism_payload_integer_cells == (12*.depth+4) and
    .expected_data_history_evaluations == .actual_data_history_evaluations and
    .joint_strength_data_history_evaluations == 0 and
    .matched_identical_classical_message.boundary == .boundary and
    .matched_identical_classical_message.work == .message_work and
    .matched_exact_rader_ntt_transfer.boundary == .boundary and
    .matched_exact_rader_ntt_transfer.resident_field_cells == (2*.q*.q) and
    .matched_exact_rader_ntt_transfer.retained_ntt_kernel_cache_cells == 0 and
    .matched_exact_rader_ntt_transfer.single_auxiliary_modulus_exactness_bound_checked and
    .q2_amplitude_cells_on_accepted_message_path == 0 and
    .recursive_or_dynamic_cache_entries == 0 and
    (.data_history_or_assignment_list_materialized | not) and
    (.latent_message_serialized | not) and
    (.intermediate_amplitudes_serialized | not) and
    .exact_graph_payload_restored and
    .same_backing_restored and
    (.snapshot_used | not)
  ] | all) and
  ([.controls[]] | all) and
  .restoration_and_reuse.restoration_generation == 2 and
  .restoration_and_reuse.second_matches_fresh and
  .restoration_and_reuse.second_commitment_matches_fresh and
  .restoration_and_reuse.exact_payload_restored_after_reuse and
  .restoration_and_reuse.same_backing_reused and
  (.restoration_and_reuse.snapshot_used | not) and
  .observed_resource_law.accepted_runtime_field_elements == "2*Q+4_MIXED_FQ_AND_FP_ELEMENTS" and
  .observed_resource_law.data_history_evaluations == "8*Q^DEPTH" and
  (.observed_resource_law.fixed_work_across_growing_depth_established | not) and
  .matched_baseline.identical_classical_message_all_boundaries_and_work_match and
  .matched_baseline.exact_rader_ntt_q2_transfer_all_boundaries_match and
  (.claim_boundaries[] | not)
' "$EVIDENCE/reexecuted.json" >/dev/null

jq -e '
  .qualified and
  .classification == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE" and
  .verification_level == "INDEPENDENT_ORACLE_REEXECUTION" and
  .restoration_classification == "EXACT_ALGEBRAIC_RESTORATION" and
  (.independence.imports_production_module | not) and
  (.independence.imports_predecessor_module | not) and
  .independence.production_result_used_only_as_comparison_target and
  .independence.separate_field_plan_message_dense_inverse_control_and_reuse_implementations and
  (.case_comparisons | length) == 13 and
  ([.case_comparisons[].checks[]] | all) and
  ([.controls[]] | all) and
  ([.restoration_and_reuse[]] | all) and
  ([.exhaustive_local_message_closure[]] | all) and
  ([.source_structure[]] | all) and
  .production_controls_all_pass and
  .production_reuse_passes
' "$EVIDENCE/reexecuted_oracle.json" >/dev/null

if rg -n 'import[[:space:]]+growing_prime|from[[:space:]]+growing_prime' \
  "$EVIDENCE/growing_prime_cubic_character_sum_separator_message_closure_independent_oracle.py"; then
  printf '%s\n' 'oracle dependency isolation failed' >&2
  exit 1
fi

sha256sum \
  "$EVIDENCE/growing_prime_cubic_character_sum_separator_message_closure.py" \
  "$EVIDENCE/growing_prime_resident_cubic_strength_character_graph_quotient.py" \
  "$EVIDENCE/growing_prime_resident_cubic_strength_phase_port_weil_action_rank.py" \
  "$EVIDENCE/growing_prime_cubic_character_sum_separator_message_closure_independent_oracle.py" \
  "$EVIDENCE/reexecuted.json" \
  "$EVIDENCE/reexecuted_oracle.json" > "$EVIDENCE/SHA256SUMS"

printf '%s\n' 'QUALIFIED_GROWING_PRIME_CUBIC_CHARACTER_SUM_SEPARATOR_MESSAGE_CLOSURE_STRICT_SCOPE'
printf 'evidence=%s\n' "$EVIDENCE"
