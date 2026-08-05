#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

if [[ $# -ne 1 ]]; then
  echo "usage: qualify_growing_rotor_dual_position_phase_scattering.sh MANAGED_BUILD_DIR" >&2
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
production="$here/growing_rotor_dual_position_phase_scattering.py"
oracle="$here/growing_rotor_dual_position_phase_scattering_independent_oracle.py"
reference="$here/growing_rotor_open_momentum_factor_independent_oracle.py"
sealed="$here/GROWING_ROTOR_DUAL_POSITION_PHASE_SCATTERING_RESULTS.json"
sealed_oracle="$here/GROWING_ROTOR_DUAL_POSITION_PHASE_SCATTERING_INDEPENDENT_ORACLE.json"

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
    raise SystemExit("position-dual production reexecution differs from seal")
' "$sealed"

nice -n 10 ionice -c 3 "$python" "$oracle" | "$python" -c '
import json, sys
sealed = json.load(open(sys.argv[1], encoding="utf-8"))
fresh = json.load(sys.stdin)
if sealed != fresh:
    raise SystemExit("position-dual oracle reexecution differs from seal")
' "$sealed_oracle"

jq -e '
  .classification == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE" and
  .verification_level == "INDEPENDENT_ORACLE_REEXECUTION" and
  .restoration_classification == "EXACT_ALGEBRAIC_RESTORATION" and
  .result == "PASS_NEGATIVE_COMPACTION_RESULT" and
  .phase_law.single_particle_transform == "EXACT_F103_ORDER17_PHASE_FOURIER" and
  .phase_law.forward_elementary_operations == 288 and
  .phase_law.inverse_elementary_operations == 289 and
  .phase_law.full_occupation_scratch_cells == 74613 and
  .phase_law.position_zero_total_sector_cells == 4389 and
  .phase_law.position_reflection_closed_cells == 2277 and
  .phase_law.open_momentum_port_cells == 0 and
  .phase_law.dense_2277_squared_operator_cells == 0 and
  .phase_law.permanent_assignment_terms == 0 and
  .parity.primary_dual_factor_mismatch_cells == 0 and
  .parity.primary_boundary == 83 and
  .parity.reuse_boundary == 70 and
  .parity.fresh_reuse_boundary == 70 and
  .parity.fresh_restored_reuse_boundary_parity and
  .parity.primary_position_diagnostics.position_off_sector_nonzero_cells == 0 and
  .parity.primary_position_diagnostics.bracelet_closure_error_cells == 0 and
  .transaction.primary_restoration_error_field_cells == 0 and
  .transaction.reuse_restoration_error_field_cells == 0 and
  .transaction.fresh_reuse_restoration_error_field_cells == 0 and
  .transaction.primary_same_backing and
  .transaction.reuse_same_backing and
  .transaction.fresh_reuse_same_backing and
  .transaction.restoration_generation_after_reuse == 2 and
  .transaction.restoration_method == "EXACT_TOPOLOGY_REMATERIALIZE_AND_SUBTRACT_ON_SAME_TARGET_BACKING" and
  (.transaction.baseline_reload_used | not) and
  .controls.missing_inverse_error_field_cells > 0 and
  .controls.reordered_noncommuting_error_field_cells > 0 and
  .resource_law.accepted_active_numeric_field_cells == 81444 and
  .resource_law.public_occupation_topology_descriptor_integers == 1343034 and
  .resource_law.predecessor_public_topology_descriptor_integers == 96723 and
  .resource_law.retained_transform_network_descriptor_integers == 2308 and
  .resource_law.named_algorithm_field_and_descriptor_slots == 1523509 and
  .resource_law.primary_forward_work.shear_blocks == 8434176 and
  .resource_law.primary_forward_work.shear_polynomial_terms == 33829728 and
  .resource_law.primary_forward_work.diagonal_character_terms == 40589472 and
  .resource_law.m199_factor_conservative_named_field_cells == 11220 and
  .resource_law.full_occupation_basis_is_explicitly_enumerated and
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
  .result == "PASS_NEGATIVE_COMPACTION_RESULT" and
  .fourier_verification.forward_inverse_matrix_mismatch_cells == 0 and
  .fourier_verification.forward_elementary_operations == 288 and
  .fourier_verification.inverse_elementary_operations == 289 and
  .fourier_verification.forward_inverse_shears == 544 and
  .fourier_verification.forward_inverse_scales == 33 and
  .fourier_verification.forward_inverse_swaps == 0 and
  .spectral_verification.occupation_histograms == 74613 and
  .spectral_verification.zero_total_coordinate_cells == 4389 and
  .spectral_verification.pair_identity_mismatches_across_all_nonzero_momenta == 0 and
  .spectral_verification.primary_weighted_identity_mismatch_cells == 0 and
  .spectral_verification.reuse_weighted_identity_mismatch_cells == 0 and
  .spectral_verification.primary_omit_particle_correction_mutated_occupations == 74613 and
  .spectral_verification.reuse_omit_particle_correction_mutated_occupations == 74613 and
  .independent_final_state.direct_two_body_raw_terms == 684624 and
  .independent_final_state.direct_two_body_csr_nonzeros == 172838 and
  .independent_final_state.direct_factor_mismatch_cells == 0 and
  .independent_final_state.primary_boundary == 83 and
  .independent_final_state.reuse_boundary == 70 and
  .independent_final_state.fresh_reuse_boundary == 70 and
  .transaction.primary_restoration_error_field_cells == 0 and
  .transaction.reuse_restoration_error_field_cells == 0 and
  .transaction.fresh_reuse_restoration_error_field_cells == 0 and
  .transaction.primary_same_backing and
  .transaction.reuse_same_backing and
  .transaction.fresh_reuse_same_backing and
  .transaction.fresh_restored_reuse_state_agreement and
  .controls.missing_fourier_inverse_normalization_matrix_mismatch_cells == 17 and
  .controls.nonprimitive_phase_root_matrix_mismatch_cells == 272 and
  .controls.underreported_4389_cell_position_basis_shortfall_cells == 70224 and
  .resource_derivation.active_numeric_field_cells == 81444 and
  .resource_derivation.named_algorithm_field_and_descriptor_slots == 1523509 and
  .resource_derivation.forward_inverse_shear_blocks == 8434176 and
  .resource_derivation.forward_inverse_shear_polynomial_terms == 33829728 and
  .resource_derivation.diagonal_character_terms_per_scattering == 40589472 and
  (.production_position_dual_module_imported | not) and
  (.production_position_dual_transform_called | not) and
  .prior_independent_factor_reference_reused and
  (.network_boundary_nonzero_counts_independently_reexecuted | not) and
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
    "forward network": (
        production["phase_law"]["forward_network_commitment"],
        oracle["fourier_verification"]["forward_network_commitment"],
    ),
    "inverse network": (
        production["phase_law"]["inverse_network_commitment"],
        oracle["fourier_verification"]["inverse_network_commitment"],
    ),
    "topology": (
        production["phase_law"]["predecessor_topology_commitment"],
        oracle["independent_final_state"]["topology_commitment"],
    ),
    "primary state": (
        production["parity"]["primary_signature_order_commitment"],
        oracle["independent_final_state"]["primary_signature_order_commitment"],
    ),
    "primary boundary": (
        production["parity"]["primary_boundary"],
        oracle["independent_final_state"]["primary_boundary"],
    ),
    "reuse boundary": (
        production["parity"]["reuse_boundary"],
        oracle["independent_final_state"]["reuse_boundary"],
    ),
    "named slots": (
        production["resource_law"]["named_algorithm_field_and_descriptor_slots"],
        oracle["resource_derivation"]["named_algorithm_field_and_descriptor_slots"],
    ),
    "shear blocks": (
        production["resource_law"]["primary_forward_work"]["shear_blocks"],
        oracle["resource_derivation"]["forward_inverse_shear_blocks"],
    ),
    "shear terms": (
        production["resource_law"]["primary_forward_work"]["shear_polynomial_terms"],
        oracle["resource_derivation"]["forward_inverse_shear_polynomial_terms"],
    ),
}
for label, (left, right) in comparisons.items():
    if left != right:
        raise SystemExit(f"independent position-dual parity differs: {label}")
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
        raise SystemExit(f"position-dual oracle production dependency: {sorted(forbidden)}")

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
printf '%s\n' 'QUALIFIED_GROWING_ROTOR_DUAL_POSITION_PHASE_SCATTERING_STRICT_SCOPE'
printf 'evidence=tracked-in-place:%s\n' "$here"
