#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

if [[ $# -ne 1 ]]; then
  echo "usage: qualify_growing_rotor_implicit_dihedral_scalar_streaming.sh MANAGED_BUILD_DIR" >&2
  exit 2
fi

build_dir=$1
case "$build_dir" in
  /dev/shm|/dev/shm/*|/run/shm|/run/shm/*)
    echo "RAM-backed build directories are forbidden" >&2
    exit 2
    ;;
esac

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(git -C "$here" rev-parse --show-toplevel)
python="$repo_root/.venv/bin/python"
production="$here/growing_rotor_implicit_dihedral_scalar_streaming_closure.py"
oracle="$here/growing_rotor_implicit_dihedral_scalar_streaming_independent_oracle.py"
reference="$here/growing_rotor_open_momentum_factor_independent_oracle.py"
sealed="$here/GROWING_ROTOR_IMPLICIT_DIHEDRAL_SCALAR_STREAMING_RESULTS.json"
sealed_oracle="$here/GROWING_ROTOR_IMPLICIT_DIHEDRAL_SCALAR_STREAMING_INDEPENDENT_ORACLE.json"

mkdir -p "$build_dir/pycache"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="$build_dir/pycache"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

nice -n 10 ionice -c 3 "$python" "$production" | "$python" -c '
import json, sys
sealed = json.load(open(sys.argv[1], encoding="utf-8"))
fresh = json.load(sys.stdin)
if sealed != fresh:
    raise SystemExit("implicit-dihedral production reexecution differs from seal")
' "$sealed"

nice -n 10 ionice -c 3 "$python" "$oracle" | "$python" -c '
import json, sys
sealed = json.load(open(sys.argv[1], encoding="utf-8"))
fresh = json.load(sys.stdin)
if sealed != fresh:
    raise SystemExit("implicit-dihedral oracle reexecution differs from seal")
' "$sealed_oracle"

jq -e '
  .classification == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE" and
  .verification_level == "INDEPENDENT_ORACLE_REEXECUTION" and
  .restoration_classification == "EXACT_ALGEBRAIC_RESTORATION" and
  .result == "PASS_IMPLICIT_TOPOLOGY_STATE_REDUCTION_WITH_SEARCH_WORK_TRADEOFF" and
  .factor_law.scalar_port_field_cells == 1 and
  .factor_law.maximum_simultaneously_live_scalar_ports == 1 and
  (.factor_law.scalar_port_projected | not) and
  .factor_law.scalar_port_same_backing and
  (.factor_law.scalar_port_live_after_scattering | not) and
  .factor_law.scalar_port_value_after_scattering == 0 and
  .factor_law.open_momentum_vector_cells == 0 and
  .factor_law.full_occupation_scratch_cells == 0 and
  .factor_law.dense_2277_squared_operator_cells == 0 and
  .factor_law.permanent_assignment_terms == 0 and
  .factor_law.retained_transition_plan_entries == 0 and
  .factor_law.retained_necklace_histogram_cells == 0 and
  .factor_law.retained_hash_map_entries == 0 and
  .factor_law.retained_necklace_to_bracelet_entries == 0 and
  .factor_law.retained_reflection_map_entries == 0 and
  .parity.primary_streamed_factor_mismatch_cells == 0 and
  .parity.primary_boundary == 83 and
  .parity.reuse_boundary == 70 and
  .parity.fresh_reuse_boundary == 70 and
  .parity.fresh_restored_reuse_state_agreement and
  .transaction.primary_restoration_error_field_cells == 0 and
  .transaction.reuse_restoration_error_field_cells == 0 and
  .transaction.fresh_reuse_restoration_error_field_cells == 0 and
  .transaction.primary_same_backing and
  .transaction.reuse_same_backing and
  .transaction.fresh_reuse_same_backing and
  .transaction.primary_scalar_port_restored and
  .transaction.reuse_scalar_port_restored and
  .transaction.fresh_reuse_scalar_port_restored and
  .transaction.forward_output_released_before_inverse and
  .transaction.restoration_generation_after_reuse == 2 and
  (.transaction.baseline_reload_used | not) and
  .controls.wrong_type_rejected and
  .controls.wrong_owner_rejected and
  .controls.premature_scalar_projection_rejected and
  .controls.null_carrier_rejected and
  .controls.missing_inverse_error_field_cells == 2257 and
  .controls.wrong_inverse_error_field_cells == 2252 and
  .controls.reordered_noncommuting_error_field_cells == 2253 and
  .controls.wrong_reflection_error_field_cells == 2254 and
  .resource_law.accepted_active_numeric_field_cells == 6832 and
  .resource_law.retained_public_topology_descriptor_integers == 8943 and
  .resource_law.named_algorithm_field_and_descriptor_slots == 15775 and
  .resource_law.accepted_full_lifecycle_logical_slot_peak == 31878 and
  .resource_law.accepted_fixed_width_logical_payload_bits == 244592 and
  .resource_law.maximum_canonical_code_payload_bits == 40 and
  .resource_law.net_named_slot_saving_against_m202 == 90057 and
  .resource_law.net_named_slot_saving_against_m199 == 92168 and
  .resource_law.topology_compiler.compiler_logical_integer_slot_peak == 11072 and
  .resource_law.primary_forward_work.source_orbit_rotations == 1193808 and
  .resource_law.primary_forward_work.inverse_candidate_moves == 5534928 and
  .resource_law.primary_forward_work.histogram_digit_decodes == 3403077 and
  .resource_law.primary_forward_work.histogram_reflection_cells == 3364368 and
  .resource_law.primary_forward_work.sorted_code_searches == 5697720 and
  .resource_law.primary_forward_work.sorted_code_comparison_upper_bound == 68372640 and
  .resource_law.primary_forward_work.closure_one_body_terms == 168912 and
  .resource_law.python_tuple_bisect_bigint_allocator_interpreter_timing_and_whole_process_peaks_excluded and
  (.catvm_custody | not) and
  (.distinct_phase_resource_established | not) and
  (.computational_advantage | not) and
  (.small_wall_crossed | not) and
  (.physical_waveform_execution | not) and
  (.physical_bit_replacement | not) and
  (.unbounded_computation_established | not) and
  (.terminal | not)
' "$sealed" >/dev/null

jq -e '
  .classification == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE" and
  .verification_level == "INDEPENDENT_ORACLE_REEXECUTION" and
  .restoration_classification == "EXACT_ALGEBRAIC_RESTORATION" and
  .result == "PASS_IMPLICIT_TOPOLOGY_STATE_REDUCTION_WITH_SEARCH_WORK_TRADEOFF" and
  .independent_state.compact_codes_and_boundary_match_tuple_reference and
  .independent_state.streamed_direct_mismatch_cells == 0 and
  .independent_state.streamed_factor_mismatch_cells == 0 and
  .independent_state.primary_boundary == 83 and
  .independent_state.reuse_boundary == 70 and
  .independent_state.fresh_reuse_boundary == 70 and
  .scalar_port_verification.logical_field_cells == 1 and
  .scalar_port_verification.maximum_simultaneously_live == 1 and
  (.scalar_port_verification.projected | not) and
  .scalar_port_verification.wrong_type_rejected and
  .scalar_port_verification.wrong_owner_rejected and
  .scalar_port_verification.premature_projection_rejected and
  .scalar_port_verification.control_port_restored and
  .scalar_port_verification.accepted_path_retained_transition_plan_entries == 0 and
  .scalar_port_verification.verification_only_reference_topology_and_plans_retained and
  .scalar_port_verification.verification_only_plan_entries == 494496 and
  .transaction.primary_restoration_error_field_cells == 0 and
  .transaction.reuse_restoration_error_field_cells == 0 and
  .transaction.fresh_reuse_restoration_error_field_cells == 0 and
  .transaction.primary_same_backing and
  .transaction.reuse_same_backing and
  .transaction.fresh_reuse_same_backing and
  .transaction.fresh_restored_reuse_state_agreement and
  .transaction.forward_output_released_before_inverse and
  .transaction.restoration_generation_after_reuse == 2 and
  (.transaction.baseline_reload_used | not) and
  .resource_derivation.active_numeric_field_cells == 6832 and
  .resource_derivation.retained_public_topology_descriptor_integers == 8943 and
  .resource_derivation.named_algorithm_field_and_descriptor_slots == 15775 and
  .resource_derivation.accepted_full_lifecycle_logical_slot_peak == 31878 and
  .resource_derivation.accepted_fixed_width_logical_payload_bits == 244592 and
  .resource_derivation.maximum_canonical_code_payload_bits == 40 and
  .resource_derivation.net_named_slot_saving_against_m202 == 90057 and
  .resource_derivation.net_named_slot_saving_against_m199 == 92168 and
  .resource_derivation.topology_compiler.compiler_logical_integer_slot_peak == 11072 and
  .resource_derivation.source_orbit_rotations == 1193808 and
  .resource_derivation.inverse_candidate_moves == 5534928 and
  .resource_derivation.histogram_digit_decodes == 3403077 and
  .resource_derivation.histogram_reflection_cells == 3364368 and
  .resource_derivation.sorted_code_searches == 5697720 and
  .resource_derivation.sorted_code_comparison_upper_bound == 68372640 and
  (.production_implicit_dihedral_module_imported | not) and
  (.production_scalar_stream_module_imported | not) and
  (.production_m199_module_imported | not) and
  .prior_independent_factor_reference_reused and
  (.catvm_custody | not) and
  (.distinct_phase_resource_established | not) and
  (.computational_advantage | not) and
  (.small_wall_crossed | not) and
  (.physical_waveform_execution | not) and
  (.physical_bit_replacement | not) and
  (.unbounded_computation_established | not) and
  (.terminal | not)
' "$sealed_oracle" >/dev/null

"$python" - "$sealed" "$sealed_oracle" <<'PY'
import json
import sys

production = json.load(open(sys.argv[1], encoding="utf-8"))
oracle = json.load(open(sys.argv[2], encoding="utf-8"))
comparisons = {
    "topology": (
        production["factor_law"]["compact_topology_commitment"],
        oracle["independent_state"]["topology_commitment"],
    ),
    "primary state": (
        production["parity"]["primary_signature_order_commitment"],
        oracle["independent_state"]["primary_signature_order_commitment"],
    ),
    "primary boundary": (
        production["parity"]["primary_boundary"],
        oracle["independent_state"]["primary_boundary"],
    ),
    "reuse boundary": (
        production["parity"]["reuse_boundary"],
        oracle["independent_state"]["reuse_boundary"],
    ),
    "active numeric state": (
        production["resource_law"]["accepted_active_numeric_field_cells"],
        oracle["resource_derivation"]["active_numeric_field_cells"],
    ),
    "retained topology": (
        production["resource_law"]["retained_public_topology_descriptor_integers"],
        oracle["resource_derivation"]["retained_public_topology_descriptor_integers"],
    ),
    "named slots": (
        production["resource_law"]["named_algorithm_field_and_descriptor_slots"],
        oracle["resource_derivation"]["named_algorithm_field_and_descriptor_slots"],
    ),
    "fixed payload": (
        production["resource_law"]["accepted_fixed_width_logical_payload_bits"],
        oracle["resource_derivation"]["accepted_fixed_width_logical_payload_bits"],
    ),
    "sorted searches": (
        production["resource_law"]["primary_forward_work"]["sorted_code_searches"],
        oracle["resource_derivation"]["sorted_code_searches"],
    ),
    "comparison upper bound": (
        production["resource_law"]["primary_forward_work"]["sorted_code_comparison_upper_bound"],
        oracle["resource_derivation"]["sorted_code_comparison_upper_bound"],
    ),
}
for label, (left, right) in comparisons.items():
    if left != right:
        raise SystemExit(f"independent implicit-dihedral parity differs: {label}")
PY

"$python" - "$oracle" "$reference" <<'PY'
import ast
import pathlib
import sys

oracle = ast.parse(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
allowed = {"growing_rotor_open_momentum_factor_independent_oracle"}
for node in ast.walk(oracle):
    if isinstance(node, ast.Import):
        names = {alias.name for alias in node.names}
    elif isinstance(node, ast.ImportFrom):
        names = {node.module or ""}
    else:
        continue
    forbidden = {
        name for name in names
        if ("growing_rotor" in name or "cat_cas" in name) and name not in allowed
    }
    if forbidden:
        raise SystemExit(f"implicit-dihedral oracle production dependency: {sorted(forbidden)}")

reference = ast.parse(pathlib.Path(sys.argv[2]).read_text(encoding="utf-8"))
for node in ast.walk(reference):
    if isinstance(node, ast.Import):
        names = [alias.name for alias in node.names]
    elif isinstance(node, ast.ImportFrom):
        names = [node.module or ""]
    else:
        continue
    if any("growing_rotor" in name or "cat_cas" in name for name in names):
        raise SystemExit("prior independent factor reference imports production")
PY

sha256sum "$production" "$oracle" "$sealed" "$sealed_oracle"
printf '%s\n' 'QUALIFIED_GROWING_ROTOR_IMPLICIT_DIHEDRAL_SCALAR_STREAMING_STRICT_SCOPE'
printf 'evidence=tracked-in-place:%s\n' "$here"
