#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

if [[ $# -ne 1 ]]; then
  echo "usage: qualify_growing_rotor_dynamic_dihedral_phase_frame.sh MANAGED_BUILD_DIR" >&2
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
production="$here/growing_rotor_dynamic_dihedral_phase_frame_closure.py"
oracle="$here/growing_rotor_dynamic_dihedral_phase_frame_independent_oracle.py"
sealed="$here/GROWING_ROTOR_DYNAMIC_DIHEDRAL_PHASE_FRAME_RESULTS.json"
sealed_oracle="$here/GROWING_ROTOR_DYNAMIC_DIHEDRAL_PHASE_FRAME_INDEPENDENT_ORACLE.json"
m204="$here/GROWING_ROTOR_COHERENT_MOMENTUM_WAVE_STREAMING_RESULTS.json"

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
    raise SystemExit("dynamic phase-frame production reexecution differs from seal")
' "$sealed"

nice -n 10 ionice -c 3 "$python" "$oracle" | "$python" -c '
import json, sys
sealed = json.load(open(sys.argv[1], encoding="utf-8"))
fresh = json.load(sys.stdin)
if sealed != fresh:
    raise SystemExit("dynamic phase-frame oracle reexecution differs from seal")
' "$sealed_oracle"

jq -e '
  .classification == "INDEPENDENTLY_VERIFIED_SOURCE_LOCAL" and
  .verification_level == "SEPARATE_REFERENCE_PARITY" and
  .restoration_classification == "EXACT_ALGEBRAIC_RESTORATION" and
  .result == "PASS_MOVING_FRAME_CLOSES_ONLY_BY_REUSING_M204_CONJUGATE" and
  .frame_law.coefficient_cells == 2277 and
  .frame_law.frame_descriptor_integers == 4 and
  .frame_law.forward_frame_steps == 288 and
  .frame_law.inverse_frame_steps == 289 and
  .frame_law.total_frame_steps == 577 and
  (.frame_law.carrier_coefficients_changed_by_frame_transport | not) and
  (.frame_law.position_diagonal_coefficientwise_in_transported_basis | not) and
  .frame_law.position_diagonal_conjugate == "M204_REFLECTION_PAIRED_EIGHT_CHANNEL_SCATTERING_STREAM" and
  .frame_law.retained_frame_matrix_cells == 0 and
  .frame_law.full_occupation_scratch_cells == 0 and
  .frame_law.dense_2277_squared_kernel_cells == 0 and
  .frame_law.permanent_assignment_terms == 0 and
  .frame_law.retained_transition_plan_entries == 0 and
  .parity.primary_boundary == 83 and
  .parity.reuse_boundary == 70 and
  .parity.fresh_reuse_boundary == 70 and
  .parity.fresh_restored_reuse_agreement and
  .parity.naive_position_coefficient_mismatch_cells == 2255 and
  .transaction.primary_restoration_error_field_cells == 0 and
  .transaction.reuse_restoration_error_field_cells == 0 and
  .transaction.fresh_reuse_restoration_error_field_cells == 0 and
  .transaction.primary_same_backing and
  .transaction.reuse_same_backing and
  .transaction.fresh_reuse_same_backing and
  .transaction.primary_frame_restored and
  .transaction.reuse_frame_restored and
  .transaction.fresh_reuse_frame_restored and
  .transaction.restoration_generation_after_reuse == 2 and
  (.transaction.baseline_reload_used | not) and
  .controls.wrong_frame_owner_rejected and
  .controls.wrong_frame_stage_rejected and
  .controls.premature_frame_projection_rejected and
  .controls.missing_frame_inverse_detected and
  .controls.missing_word_inverse_error_field_cells == 2257 and
  .controls.naive_transported_coefficient_diagonal_rejected and
  .resource_law.accepted_active_numeric_field_cells == 6843 and
  .resource_law.retained_public_topology_descriptor_integers == 8943 and
  .resource_law.accepted_full_lifecycle_logical_slot_peak == 31878 and
  .resource_law.m204_sorted_searches_retained == 5697720 and
  .resource_law.m204_fanout_candidates_retained == 5534928 and
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
  .classification == "INDEPENDENTLY_VERIFIED_SOURCE_LOCAL" and
  .verification_level == "SEPARATE_REFERENCE_PARITY" and
  .restoration_classification == "NO_RESTORATION_CLAIM" and
  (.oracle_imports_cat_cas_modules | not) and
  .transferable_rotor2_certificate.occupation_cells == 153 and
  .transferable_rotor2_certificate.momentum_bracelet_cells == 9 and
  .transferable_rotor2_certificate.zero_total_position_cells == 9 and
  .transferable_rotor2_certificate.reflection_closed_position_cells == 9 and
  .transferable_rotor2_certificate.forward_inverse_identity and
  .transferable_rotor2_certificate.conjugated_direct_scattering_mismatch_cells == 0 and
  .transferable_rotor2_certificate.naive_coefficientwise_diagonal_mismatch_cells == 79 and
  .transferable_rotor2_certificate.forward_nonzero_cells == 81 and
  .transferable_rotor2_certificate.conjugated_nonzero_cells == 78 and
  (.rotor6_execution_independently_reexecuted | not) and
  .rotor6_parity_uses_prior_independently_verified_m204_boundary_and_commitment and
  (.distinct_phase_resource_established | not) and
  (.terminal | not)
' "$sealed_oracle" >/dev/null

"$python" - "$sealed" "$m204" <<'PY'
import json, sys
moving = json.load(open(sys.argv[1], encoding="utf-8"))
baseline = json.load(open(sys.argv[2], encoding="utf-8"))
comparisons = {
    "primary boundary": (
        moving["parity"]["primary_boundary"],
        baseline["parity"]["primary_boundary"],
    ),
    "reuse boundary": (
        moving["parity"]["reuse_boundary"],
        baseline["parity"]["reuse_boundary"],
    ),
    "fresh reuse boundary": (
        moving["parity"]["fresh_reuse_boundary"],
        baseline["parity"]["fresh_reuse_boundary"],
    ),
    "primary signature-order commitment": (
        moving["parity"]["primary_signature_order_commitment"],
        baseline["parity"]["primary_signature_order_commitment"],
    ),
    "sorted searches": (
        moving["resource_law"]["m204_sorted_searches_retained"],
        baseline["resource_law"]["primary_forward_work"]["sorted_code_searches"],
    ),
    "fanout candidates": (
        moving["resource_law"]["m204_fanout_candidates_retained"],
        baseline["resource_law"]["primary_forward_work"]["inverse_candidate_moves"],
    ),
}
for label, (left, right) in comparisons.items():
    if left != right:
        raise SystemExit(f"dynamic phase-frame M204 parity differs: {label}")
PY

"$python" - "$oracle" <<'PY'
import ast, pathlib, sys
tree = ast.parse(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
imports = set()
for node in ast.walk(tree):
    if isinstance(node, ast.Import):
        imports.update(alias.name for alias in node.names)
    elif isinstance(node, ast.ImportFrom):
        imports.add(node.module or "")
if any("growing_rotor" in name or "cat_cas" in name for name in imports):
    raise SystemExit("dynamic phase-frame oracle imports CAT_CAS production")
PY

sha256sum "$production" "$oracle" "$sealed" "$sealed_oracle" "$m204"
printf '%s\n' 'QUALIFIED_GROWING_ROTOR_DYNAMIC_DIHEDRAL_PHASE_FRAME_SOURCE_LOCAL'
printf 'evidence=tracked-in-place:%s\n' "$here"
