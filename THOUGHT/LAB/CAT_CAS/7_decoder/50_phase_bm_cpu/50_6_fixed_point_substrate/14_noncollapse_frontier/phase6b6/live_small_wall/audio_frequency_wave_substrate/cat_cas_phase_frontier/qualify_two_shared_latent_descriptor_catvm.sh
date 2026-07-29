#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
export PYTHONHASHSEED=0

if [[ $# -ne 1 ]]; then
  echo "usage: qualify_two_shared_latent_descriptor_catvm.sh EVIDENCE_DIR" >&2
  exit 2
fi

evidence_dir=$1
frontier_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(git -C "$frontier_dir" rev-parse --show-toplevel)
service_source="$frontier_dir/catvm_necklace_two_shared_latent_descriptor_service.cpp"
predecessor_source="$frontier_dir/catvm_necklace_two_shared_latent_service.cpp"
controller="$frontier_dir/catvm_necklace_two_shared_latent_descriptor_controller.py"
protocol="$frontier_dir/catvm_necklace_two_shared_latent_descriptor_protocol.py"
oracle="$frontier_dir/two_shared_latent_descriptor_oracle.py"
programs="$frontier_dir/TWO_SHARED_LATENT_DESCRIPTOR_PROGRAMS.json"
durable="$frontier_dir/TWO_SHARED_LATENT_DESCRIPTOR_RESULTS.json"
review="$frontier_dir/TWO_SHARED_LATENT_DESCRIPTOR_CLEANROOM_REVIEW.md"
service="$evidence_dir/descriptor_service"
sanitized_service="$evidence_dir/descriptor_service_ubsan"
predecessor="$evidence_dir/predecessor_service"
catvm_result="$evidence_dir/catvm.json"
sanitized_result="$evidence_dir/catvm_ubsan.json"
oracle_result="$evidence_dir/oracle.json"

mkdir -p "$evidence_dir"

common_flags=(
  -std=c++20
  -O2
  -Wall
  -Wextra
  -Wpedantic
  -Werror
)
g++ "${common_flags[@]}" "$service_source" -o "$service"
g++ "${common_flags[@]}" "$predecessor_source" -o "$predecessor"
g++ \
  "${common_flags[@]}" \
  -fsanitize=undefined \
  -fno-sanitize-recover=all \
  "$service_source" \
  -o "$sanitized_service"

"$repo_root/.venv/bin/python" -m py_compile \
  "$controller" "$protocol" "$oracle"

"$repo_root/.venv/bin/python" \
  "$controller" \
  "$service" \
  "$programs" \
  "$evidence_dir/regular-runs" \
  >"$catvm_result" \
  2>"$evidence_dir/controller.stderr"

UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
"$repo_root/.venv/bin/python" \
  "$controller" \
  "$sanitized_service" \
  "$programs" \
  "$evidence_dir/ubsan-runs" \
  >"$sanitized_result" \
  2>"$evidence_dir/controller-ubsan.stderr"

"$repo_root/.venv/bin/python" \
  "$oracle" \
  "$programs" \
  "$catvm_result" \
  >"$oracle_result"

test ! -s "$evidence_dir/controller.stderr"
test ! -s "$evidence_dir/controller-ubsan.stderr"
cmp "$catvm_result" "$sanitized_result"

jq -e '
  .result == "PASS"
  and .verification_classification
    == "INDEPENDENTLY_VERIFIED_SOURCE_LOCAL"
  and .restoration_class == "NUMERICAL_PHYSICAL_STATE_RESTORATION"
  and .compiler.module_counts == [6, 5, 7]
  and (.compiler.checksums | length) == 3
  and (.compiler.checksums | unique | length) == 3
  and .compiler.carrier_or_boundary_compiler_inputs == 0
  and .compiler.owners_derived_in_service
  and .compiler.stage_cut_derived_from_first_joint
  and .compiler.later_joint_required
  and .compiler.forward_topology_only
  and .compiler.retained_inverse_descriptor_count == 0
  and .compiler.slot_independent_checksum
  and .compiler.slot_independent_boundary_error < 1.2e-10
  and .compiler.valid_mutation_checksum_changed
  and .compiler.valid_mutation_boundary_effect > 1e-7
  and .custody.port_count == 2
  and .custody.resident_complex_cells == 1140
  and .custody.full_tuple_bound_per_compiled_consumer
  and .custody.rebound_at_each_generation
  and .custody.generations == [1, 2, 3]
  and .custody.same_carrier_backing
  and .custody.baseline_reload_bytes == 0
  and (
    [.families[].restoration_error]
    | all(. < 1.2e-10)
  )
  and (
    [.families[].fresh_restored_boundary_error]
    | all(. < 1.2e-10)
  )
  and (
    [.families[].fresh_restored_streamed_generator_terms_equal]
    | all(.)
  )
  and .controls.atomicity_and_inverse.missing_inverse_error > 1e-5
  and .controls.atomicity_and_inverse.reordered_inverse_error > 1e-5
  and .controls.atomicity_and_inverse.semantic_inverse_error > 1e-5
  and .controls.atomicity_and_inverse
    .authorized_stop_ack_after_rollback
  and .controls.atomicity_and_inverse.denied_stop_did_not_terminate
  and .controls.atomicity_and_inverse
    .disconnect_stage_inverse_cleanup_exit == 0
  and .controls.atomicity_and_inverse
    .disconnect_partial_load_exit == 0
  and .controls.atomicity_and_inverse.null_carrier
    == "DENIED_BEFORE_ACCESS"
  and (
    [.controls.compiler[]]
    | all(. == "DENIED" or . == "DENIED_AT_SEAL")
  )
  and .controls.wrong_outer_lease == "DENIED"
  and .controls.wrong_outer_generation == "DENIED"
  and .controls.replayed_nonce == "DENIED"
  and .controls.wrong_tuple_fields
    == "DENIED_BEFORE_CARRIER_OPERATION"
  and .controls.stale_program_identity
    == "DENIED_BEFORE_CARRIER_OPERATION"
  and .controls.premature_projection == "DENIED"
  and .controls.snapshot == "DENIED"
  and .controls.program_substitution_while_staged == "DENIED"
  and .no_smuggle.stage_boundary_values == 0
  and .no_smuggle.latent_values_in_response == 0
  and .no_smuggle.content_derived_receipts == 0
  and .no_smuggle.stdout_bytes == 0
  and .no_smuggle.stderr_bytes == 0
  and (.no_smuggle.controller_imports_backend | not)
  and (.no_smuggle.public_fixture_contains_final_answers | not)
  and .resource_law.carrier_payload_bytes == 18240
  and .resource_law.persistent_baseline_plus_carrier_payload_bytes
    == 36480
  and .resource_law.descriptor_slot_object_bytes == 128
  and .resource_law.descriptor_slot_registry_object_bytes == 384
  and .resource_law.bound_module_object_bytes == 88
  and .resource_law.accepted_peak_active_bound_program_bytes == 616
  and .resource_law.ceiling_peak_active_bound_program_bytes == 704
  and .resource_law.retained_inverse_history_bytes == 0
  and .resource_law.request_bytes == 32
  and .resource_law.response_bytes == 116
  and .resource_law.relation_table_cells == 0
  and .resource_law.assignment_cells == 0
  and .comparisons.identical_compact_1140_complex_recurrence_exists
  and .comparisons.classical_reference_verification_level
    == "PACKAGE_SELF_REVIEW"
  and (.comparisons.separate_reference_parity | not)
  and (.general_scheduler_established | not)
  and (.arbitrary_program_algebra_established | not)
  and (.distinct_phase_resource_established | not)
  and (.computational_advantage | not)
  and (.small_wall_crossed | not)
  and (.catalytic_inference_established | not)
  and (.physical_waveform_execution | not)
  and (.replacement_of_physical_bits_with_pi | not)
  and (.terminal | not)
' "$catvm_result" >/dev/null

jq -e '
  .result == "PASS"
  and (.production_backend_imported | not)
  and (.production_protocol_imported | not)
  and (.production_compiler_called | not)
  and .checks.service_checksums_match
  and .checks.three_distinct_families
  and .checks.two_joint_consumers_each
  and .checks.both_local_ports_each
  and .checks.inverse_is_reverse_topology
  and (.full_285_necklace_recurrence_reimplemented | not)
  and (.classical_reference_parity_established | not)
  and (.terminal | not)
' "$oracle_result" >/dev/null

jq -e \
  --slurpfile catvm "$catvm_result" \
  --slurpfile oracle "$oracle_result" \
  '
    .result == "PASS"
    and .classification == "INDEPENDENTLY_VERIFIED_SOURCE_LOCAL"
    and .verification_level == "CLEANROOM_ADVERSARIAL_VERIFICATION"
    and .restoration_class == "NUMERICAL_PHYSICAL_STATE_RESTORATION"
    and .catvm == $catvm[0]
    and .independent_public_topology_oracle == $oracle[0]
    and (.claim_limits.generic_scheduler | not)
    and (.claim_limits.arbitrary_program_algebra | not)
    and (.claim_limits.distinct_phase_resource | not)
    and (.claim_limits.computational_advantage | not)
    and (.claim_limits.small_wall_crossing | not)
    and (.claim_limits.catalytic_inference | not)
    and (.claim_limits.physical_waveform_execution | not)
    and (.terminal | not)
  ' "$durable" >/dev/null

if rg -n \
  'ctypes|cffi|dlopen|importlib|four_rotor_necklace|\.cpp' \
  "$controller" "$oracle"; then
  echo "controller or oracle contains a backend-loading path" >&2
  exit 1
fi

sha256sum \
  "$service_source" \
  "$predecessor_source" \
  "$controller" \
  "$protocol" \
  "$oracle" \
  "$programs" \
  "$durable" \
  "$review" \
  "$catvm_result" \
  "$sanitized_result" \
  "$oracle_result" \
  >"$evidence_dir/SHA256SUMS"

echo "two-shared-latent descriptor CATVM qualification: PASS"
