#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

if [[ $# -ne 1 ]]; then
  echo "usage: qualify_growing_rotor_coherent_momentum_wave_streaming.sh MANAGED_BUILD_DIR" >&2
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
production="$here/growing_rotor_coherent_momentum_wave_streaming_closure.py"
oracle="$here/growing_rotor_coherent_momentum_wave_streaming_independent_oracle.py"
reference="$here/growing_rotor_implicit_dihedral_scalar_streaming_independent_oracle.py"
prior_reference="$here/growing_rotor_open_momentum_factor_independent_oracle.py"
sealed="$here/GROWING_ROTOR_COHERENT_MOMENTUM_WAVE_STREAMING_RESULTS.json"
sealed_oracle="$here/GROWING_ROTOR_COHERENT_MOMENTUM_WAVE_STREAMING_INDEPENDENT_ORACLE.json"

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
    raise SystemExit("momentum-wave production reexecution differs from seal")
' "$sealed"

nice -n 10 ionice -c 3 "$python" "$oracle" | "$python" -c '
import json, sys
sealed = json.load(open(sys.argv[1], encoding="utf-8"))
fresh = json.load(sys.stdin)
if sealed != fresh:
    raise SystemExit("momentum-wave oracle reexecution differs from seal")
' "$sealed_oracle"

jq -e '
  .classification == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE" and
  .verification_level == "INDEPENDENT_ORACLE_REEXECUTION" and
  .restoration_classification == "EXACT_ALGEBRAIC_RESTORATION" and
  .result == "PASS_CHANNEL_FUSION_REDUCES_ORBIT_AND_DECODE_WORK_BUT_NOT_SEARCH_OR_CLASSICAL_EQUIVALENCE" and
  .wave_law.channel_type == "REFLECTION_PAIRED_F103_MOMENTUM_WAVE8" and
  .wave_law.channels == [1,2,3,4,5,6,7,8] and
  .wave_law.wave_port_field_cells == 8 and
  .wave_law.maximum_simultaneously_live_wave_ports == 1 and
  (.wave_law.individual_channels_projected | not) and
  .wave_law.wave_port_same_backing and
  (.wave_law.wave_port_live_after_scattering | not) and
  .wave_law.wave_port_values_after_scattering == [0,0,0,0,0,0,0,0] and
  (.wave_law.physical_simultaneous_superposition_established | not) and
  .wave_law.open_momentum_vector_cells == 0 and
  .wave_law.full_occupation_scratch_cells == 0 and
  .wave_law.dense_2277_squared_operator_cells == 0 and
  .wave_law.permanent_assignment_terms == 0 and
  .wave_law.retained_transition_plan_entries == 0 and
  .parity.primary_wave_scalar_stream_mismatch_cells == 0 and
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
  .transaction.primary_wave_port_restored and
  .transaction.reuse_wave_port_restored and
  .transaction.fresh_reuse_wave_port_restored and
  .transaction.forward_output_released_before_inverse and
  .transaction.restoration_generation_after_reuse == 2 and
  (.transaction.baseline_reload_used | not) and
  .controls.wrong_type_rejected and
  .controls.wrong_owner_rejected and
  .controls.premature_wave_projection_rejected and
  .controls.null_carrier_rejected and
  .controls.missing_inverse_error_field_cells == 2257 and
  .controls.wrong_inverse_error_field_cells == 2252 and
  .controls.reordered_noncommuting_error_field_cells == 2253 and
  .controls.wrong_reflection_error_field_cells == 2254 and
  .resource_law.accepted_active_numeric_field_cells == 6839 and
  .resource_law.retained_public_topology_descriptor_integers == 8943 and
  .resource_law.named_algorithm_field_and_descriptor_slots == 15782 and
  .resource_law.accepted_fixed_width_logical_payload_bits == 244641 and
  .resource_law.accepted_full_lifecycle_logical_slot_peak == 31878 and
  .resource_law.primary_forward_work.wave_port_leases == 4389 and
  .resource_law.primary_forward_work.wave_port_clear_field_cells == 35112 and
  .resource_law.primary_forward_work.source_orbit_rotations == 149226 and
  .resource_law.primary_forward_work.histogram_digit_decodes == 2880786 and
  .resource_law.primary_forward_work.histogram_reflection_cells == 2842077 and
  .resource_law.primary_forward_work.inverse_candidate_moves == 5534928 and
  .resource_law.primary_forward_work.sorted_code_searches == 5697720 and
  .resource_law.primary_forward_work.sorted_code_comparison_upper_bound == 68372640 and
  .resource_law.source_orbit_rotation_reduction_against_m203 == 1044582 and
  .resource_law.histogram_digit_decode_reduction_against_m203 == 522291 and
  .resource_law.histogram_reflection_cell_reduction_against_m203 == 522291 and
  .resource_law.sorted_search_reduction_against_m203 == 0 and
  .resource_law.fanout_candidate_reduction_against_m203 == 0 and
  (.catvm_custody | not) and
  (.distinct_phase_resource_established | not) and
  (.computational_advantage | not) and
  (.small_wall_crossed | not) and
  (.physical_waveform_execution | not) and
  (.physical_simultaneous_superposition | not) and
  (.physical_bit_replacement | not) and
  (.unbounded_computation_established | not) and
  (.terminal | not)
' "$sealed" >/dev/null

jq -e '
  .classification == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE" and
  .verification_level == "INDEPENDENT_ORACLE_REEXECUTION" and
  .restoration_classification == "EXACT_ALGEBRAIC_RESTORATION" and
  .result == "PASS_CHANNEL_FUSION_REDUCES_ORBIT_AND_DECODE_WORK_BUT_NOT_SEARCH_OR_CLASSICAL_EQUIVALENCE" and
  .independent_state.compact_codes_and_boundary_match_tuple_reference and
  .independent_state.wave_scalar_mismatch_cells == 0 and
  .independent_state.wave_direct_mismatch_cells == 0 and
  .independent_state.wave_factor_mismatch_cells == 0 and
  .independent_state.primary_boundary == 83 and
  .independent_state.reuse_boundary == 70 and
  .independent_state.fresh_reuse_boundary == 70 and
  .wave_port_verification.channel_type == "REFLECTION_PAIRED_F103_MOMENTUM_WAVE8" and
  .wave_port_verification.channels == [1,2,3,4,5,6,7,8] and
  .wave_port_verification.logical_field_cells == 8 and
  .wave_port_verification.maximum_simultaneously_live == 1 and
  (.wave_port_verification.individual_channels_projected | not) and
  .wave_port_verification.wrong_type_rejected and
  .wave_port_verification.wrong_owner_rejected and
  .wave_port_verification.premature_projection_rejected and
  .wave_port_verification.null_carrier_rejected and
  .wave_port_verification.control_port_restored and
  .wave_port_verification.accepted_path_retained_transition_plan_entries == 0 and
  .wave_port_verification.verification_only_reference_topology_and_plans_retained and
  .wave_port_verification.verification_only_plan_entries == 494496 and
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
  .resource_derivation.active_numeric_field_cells == 6839 and
  .resource_derivation.retained_public_topology_descriptor_integers == 8943 and
  .resource_derivation.named_algorithm_field_and_descriptor_slots == 15782 and
  .resource_derivation.accepted_fixed_width_logical_payload_bits == 244641 and
  .resource_derivation.accepted_full_lifecycle_logical_slot_peak == 31878 and
  .resource_derivation.source_orbit_rotations == 149226 and
  .resource_derivation.histogram_digit_decodes == 2880786 and
  .resource_derivation.histogram_reflection_cells == 2842077 and
  .resource_derivation.inverse_candidate_moves == 5534928 and
  .resource_derivation.sorted_code_searches == 5697720 and
  .resource_derivation.sorted_code_comparison_upper_bound == 68372640 and
  (.production_wave_module_imported | not) and
  (.production_implicit_dihedral_module_imported | not) and
  (.production_scalar_stream_module_imported | not) and
  (.production_m199_module_imported | not) and
  .prior_independent_implicit_dihedral_reference_reused and
  (.catvm_custody | not) and
  (.distinct_phase_resource_established | not) and
  (.computational_advantage | not) and
  (.small_wall_crossed | not) and
  (.physical_waveform_execution | not) and
  (.physical_simultaneous_superposition | not) and
  (.physical_bit_replacement | not) and
  (.unbounded_computation_established | not) and
  (.terminal | not)
' "$sealed_oracle" >/dev/null

"$python" - "$sealed" "$sealed_oracle" <<'PY'
import json, sys
production = json.load(open(sys.argv[1], encoding="utf-8"))
oracle = json.load(open(sys.argv[2], encoding="utf-8"))
comparisons = {
    "topology": (production["wave_law"]["compact_topology_commitment"], oracle["independent_state"]["topology_commitment"]),
    "primary state": (production["parity"]["primary_signature_order_commitment"], oracle["independent_state"]["primary_signature_order_commitment"]),
    "primary boundary": (production["parity"]["primary_boundary"], oracle["independent_state"]["primary_boundary"]),
    "reuse boundary": (production["parity"]["reuse_boundary"], oracle["independent_state"]["reuse_boundary"]),
    "active state": (production["resource_law"]["accepted_active_numeric_field_cells"], oracle["resource_derivation"]["active_numeric_field_cells"]),
    "named slots": (production["resource_law"]["named_algorithm_field_and_descriptor_slots"], oracle["resource_derivation"]["named_algorithm_field_and_descriptor_slots"]),
    "orbit rotations": (production["resource_law"]["primary_forward_work"]["source_orbit_rotations"], oracle["resource_derivation"]["source_orbit_rotations"]),
    "sorted searches": (production["resource_law"]["primary_forward_work"]["sorted_code_searches"], oracle["resource_derivation"]["sorted_code_searches"]),
}
for label, (left, right) in comparisons.items():
    if left != right:
        raise SystemExit(f"independent momentum-wave parity differs: {label}")
PY

"$python" - "$oracle" "$reference" "$prior_reference" <<'PY'
import ast, pathlib, sys

def imported(path):
    tree = ast.parse(pathlib.Path(path).read_text(encoding="utf-8"))
    result = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            result.add(node.module or "")
    return result

oracle_imports = imported(sys.argv[1])
allowed = {"growing_rotor_implicit_dihedral_scalar_streaming_independent_oracle"}
forbidden = {name for name in oracle_imports if ("growing_rotor" in name or "cat_cas" in name) and name not in allowed}
if forbidden:
    raise SystemExit(f"momentum-wave oracle production dependency: {sorted(forbidden)}")

reference_imports = imported(sys.argv[2])
allowed_reference = {"growing_rotor_open_momentum_factor_independent_oracle"}
forbidden = {name for name in reference_imports if ("growing_rotor" in name or "cat_cas" in name) and name not in allowed_reference}
if forbidden:
    raise SystemExit(f"implicit-dihedral reference production dependency: {sorted(forbidden)}")

if any("growing_rotor" in name or "cat_cas" in name for name in imported(sys.argv[3])):
    raise SystemExit("prior independent factor reference imports production")
PY

sha256sum "$production" "$oracle" "$sealed" "$sealed_oracle"
printf '%s\n' 'QUALIFIED_GROWING_ROTOR_COHERENT_MOMENTUM_WAVE_STREAMING_STRICT_SCOPE'
printf 'evidence=tracked-in-place:%s\n' "$here"
