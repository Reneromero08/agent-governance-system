#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

if [[ $# -ne 1 ]]; then
  echo "usage: qualify_necklace_offdiagonal_pair_scattering_closure.sh MANAGED_BUILD_DIR" >&2
  exit 2
fi

build_dir=$1
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(git -C "$here" rev-parse --show-toplevel)
python="$repo_root/.venv/bin/python"
production="$here/necklace_offdiagonal_pair_scattering_closure.cpp"
oracle="$here/necklace_offdiagonal_pair_scattering_independent_oracle.py"
sealed="$here/NECKLACE_OFFDIAGONAL_PAIR_SCATTERING_CLOSURE_RESULTS.json"
sealed_oracle="$here/NECKLACE_OFFDIAGONAL_PAIR_SCATTERING_INDEPENDENT_ORACLE.json"
binary="$build_dir/necklace_offdiagonal_pair_scattering_closure"

mkdir -p "$build_dir"
g++ -std=c++20 -O2 -Wall -Wextra -Wpedantic -Werror \
  "$production" -o "$binary"

"$python" -c '
import json, sys
sealed = json.load(open(sys.argv[1], encoding="utf-8"))
fresh = json.load(open(sys.argv[2], encoding="utf-8"))
if sealed != fresh:
    raise SystemExit("off-diagonal production reexecution differs from seal")
' "$sealed" <(nice -n 10 taskset -c 0-3 "$binary")

export PYTHONDONTWRITEBYTECODE=1
unset PYTHONPYCACHEPREFIX
"$python" -c '
import json, sys
sealed = json.load(open(sys.argv[1], encoding="utf-8"))
fresh = json.load(open(sys.argv[2], encoding="utf-8"))
if sealed != fresh:
    raise SystemExit("off-diagonal oracle reexecution differs from seal")
' "$sealed_oracle" <(nice -n 10 taskset -c 0-3 "$python" "$oracle")

jq -e '
  .classification == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE" and
  .verification_level == "INDEPENDENT_ORACLE_REEXECUTION" and
  .restoration_classification == "NUMERICAL_PHYSICAL_STATE_RESTORATION" and
  .result == "PASS" and
  .mechanism.necklace_cells == 285 and
  .mechanism.signed_scattering_shifts == 16 and
  .mechanism.momentum_conserving and
  .mechanism.rotation_invariant and
  .mechanism.exchange_symmetric and
  .mechanism.off_diagonal_terms_present and
  .mechanism.genuine_double_move_terms_present and
  .mechanism.matrix_free_hermitian_probe_error < 2e-13 and
  .mechanism.radius_bound < 6 and
  .mechanism.chebyshev_tail_bound < 1e-13 and
  .restoration.primary_error < 1.2e-10 and
  .restoration.reuse_error < 1.2e-10 and
  .restoration.fresh_restored_reuse_boundary_error < 1.2e-10 and
  .restoration.repeated_reuse_cycles == 32 and
  .restoration.repeated_reuse_max_error < 4e-10 and
  .restoration.same_backing_primary and
  .restoration.same_backing_reuse and
  (.restoration.baseline_reload_used | not) and
  .controls.missing_inverse_error > 1e-6 and
  .controls.wrong_inverse_error > 1e-6 and
  .controls.reordered_inverse_error > 1e-6 and
  .controls.zero_scattering_boundary_difference > 1e-8 and
  .controls.swapped_phase_scattering_boundary_difference > 1e-8 and
  .controls.null_carrier_rejected and
  .resources.accepted_resident_complex_cells == 285 and
  .resources.accepted_temporary_necklace_complex_cells == 855 and
  .resources.accepted_temporary_occupation_complex_cells == 0 and
  .resources.accepted_retained_dense_operator_complex_cells == 0 and
  .resources.accepted_retained_pair_transition_table_cells == 0 and
  .resources.accepted_retained_inverse_history_bytes == 0 and
  (.accepted_path_occupation_vector_materialized | not) and
  (.accepted_path_dense_285_operator_materialized | not) and
  (.accepted_path_pair_transition_table_materialized | not) and
  (.public_topology_compilation_inspects_final_answer | not) and
  (.response_order_machine_enforced | not) and
  .matched_classical_recurrence == "IDENTICAL_285_COMPLEX_NECKLACE_PAIR_SCATTERING_AND_DIAGONAL_PHASE_RECURRENCE" and
  (.distinct_phase_resource_established | not) and
  (.computational_advantage | not) and
  (.small_wall_crossed | not) and
  (.catvm_custody | not) and
  (.physical_waveform_execution | not) and
  (.physical_bit_replacement | not) and
  (.unbounded_computation_established | not) and
  (.terminal | not)
' "$sealed" >/dev/null

jq -e '
  .classification == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE" and
  .verification_level == "INDEPENDENT_ORACLE_REEXECUTION" and
  .restoration_classification == "NUMERICAL_PHYSICAL_STATE_RESTORATION" and
  .oracle_representation == "FULL_4845_CELL_EXCHANGE_SYMMETRIC_OCCUPATION_SPACE_VERIFICATION_ONLY" and
  .occupation_cells == 4845 and
  .necklace_cells_reconstructed == 285 and
  .scattering_basis_count == 8 and
  .scattering_basis_streamed_terms == 707472 and
  .scattering_basis_total_nnz == 350608 and
  .weighted_hermitian_max_error < 2e-13 and
  .rotation_invariance_error < 2e-10 and
  .primary.restoration_error < 2e-10 and
  .reuse.restoration_error < 2e-10 and
  .controls.missing_inverse_error > 1e-6 and
  .controls.wrong_inverse_error > 1e-6 and
  .controls.reordered_inverse_error > 1e-6 and
  .controls.zero_scattering_boundary_difference > 1e-8 and
  .controls.swapped_phase_scattering_boundary_difference > 1e-8 and
  .accepted_path_occupation_cells == 0 and
  .accepted_path_sparse_generator_cells == 0 and
  .accepted_path_dense_necklace_operator_cells == 0 and
  (.distinct_phase_resource_established | not) and
  (.computational_advantage | not) and
  (.small_wall_crossed | not)
' "$sealed_oracle" >/dev/null

"$python" -c '
import json, sys
production = json.load(open(sys.argv[1], encoding="utf-8"))
oracle = json.load(open(sys.argv[2], encoding="utf-8"))
for production_key, oracle_key in (("primary_boundary", "primary"), ("reuse_boundary", "reuse")):
    error = max(abs(a-b) for a,b in zip(production[production_key], oracle[oracle_key]["boundary"], strict=True))
    if error > 1e-10:
        raise SystemExit(f"independent occupation boundary differs: {production_key}")
for key in ("missing_inverse_error", "wrong_inverse_error", "reordered_inverse_error", "zero_scattering_boundary_difference", "swapped_phase_scattering_boundary_difference"):
    if abs(production["controls"][key] - oracle["controls"][key]) > 1e-10:
        raise SystemExit(f"independent control differs: {key}")
' "$sealed" "$sealed_oracle"

"$python" -c '
import ast, pathlib, sys
tree = ast.parse(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
for node in ast.walk(tree):
    if isinstance(node, ast.Import):
        names = [alias.name for alias in node.names]
    elif isinstance(node, ast.ImportFrom):
        names = [node.module or ""]
    else:
        continue
    if any("necklace" in name or "four_rotor" in name for name in names):
        raise SystemExit("off-diagonal oracle production dependency isolation failed")
' "$oracle"

sha256sum "$production" "$oracle" \
  "$here/necklace_two_body_phase_interaction_closure.cpp" \
  "$here/four_rotor_necklace_generator_phase.cpp" \
  "$sealed" "$sealed_oracle"
printf '%s\n' 'QUALIFIED_NECKLACE_OFFDIAGONAL_PAIR_SCATTERING_CLOSURE_STRICT_SCOPE'
printf 'evidence=tracked-in-place:%s\n' "$here"
