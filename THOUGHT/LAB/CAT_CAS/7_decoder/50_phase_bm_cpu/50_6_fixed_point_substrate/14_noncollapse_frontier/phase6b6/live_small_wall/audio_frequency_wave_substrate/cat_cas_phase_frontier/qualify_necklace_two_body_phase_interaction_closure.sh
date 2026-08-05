#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

if [[ $# -ne 1 ]]; then
  echo "usage: qualify_necklace_two_body_phase_interaction_closure.sh MANAGED_BUILD_DIR" >&2
  exit 2
fi

build_dir=$1
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(git -C "$here" rev-parse --show-toplevel)
python="$repo_root/.venv/bin/python"
production="$here/necklace_two_body_phase_interaction_closure.cpp"
oracle="$here/necklace_two_body_phase_interaction_independent_oracle.py"
sealed="$here/NECKLACE_TWO_BODY_PHASE_INTERACTION_CLOSURE_RESULTS.json"
sealed_oracle="$here/NECKLACE_TWO_BODY_PHASE_INTERACTION_INDEPENDENT_ORACLE.json"
binary="$build_dir/necklace_two_body_phase_interaction_closure"

mkdir -p "$build_dir"
g++ \
  -std=c++20 \
  -O2 \
  -Wall \
  -Wextra \
  -Wpedantic \
  -Werror \
  "$production" \
  -o "$binary"

"$python" -c '
import json
import sys
with open(sys.argv[1], encoding="utf-8") as handle:
    sealed = json.load(handle)
with open(sys.argv[2], encoding="utf-8") as handle:
    fresh = json.load(handle)
if sealed != fresh:
    raise SystemExit("two-body production reexecution differs from seal")
' "$sealed" <(nice -n 10 taskset -c 0-3 "$binary")

export PYTHONDONTWRITEBYTECODE=1
unset PYTHONPYCACHEPREFIX
"$python" -c '
import json
import sys
with open(sys.argv[1], encoding="utf-8") as handle:
    sealed = json.load(handle)
with open(sys.argv[2], encoding="utf-8") as handle:
    fresh = json.load(handle)
if sealed != fresh:
    raise SystemExit("two-body oracle reexecution differs from seal")
' "$sealed_oracle" <(nice -n 10 taskset -c 0-3 "$python" "$oracle")

jq -e '
  .classification == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE" and
  .verification_level == "INDEPENDENT_ORACLE_REEXECUTION" and
  .restoration_classification == "NUMERICAL_PHYSICAL_STATE_RESTORATION" and
  .result == "PASS" and
  .topology.necklace_cells == 285 and
  .topology.pair_distance_channels == 9 and
  .topology.pairs_per_histogram == 6 and
  .topology.distinct_pair_signatures == 165 and
  .topology.rotation_invariant and
  .topology.exchange_symmetric and
  .topology.collision_degenerate_states_split and
  .parity.one_step_givens_l2_error < 6e-11 and
  .parity.depth4_givens_l2_error < 6e-11 and
  .parity.one_step_streamed_permanent_l2_error < 6e-11 and
  .parity.depth4_streamed_permanent_l2_error < 6e-11 and
  .restoration.primary_error < 6e-11 and
  .restoration.reuse_error < 6e-11 and
  .restoration.fresh_restored_reuse_boundary_error < 6e-11 and
  .restoration.repeated_reuse_cycles == 32 and
  .restoration.repeated_reuse_max_error < 2e-10 and
  .restoration.same_backing_primary and
  .restoration.same_backing_reuse and
  (.restoration.baseline_reload_used | not) and
  .restoration.restoration_generation_after_reuse == 2 and
  .controls.missing_inverse_error > 1e-6 and
  .controls.wrong_inverse_error > 1e-6 and
  .controls.reordered_inverse_error > 1e-6 and
  .controls.collision_only_boundary_difference > 1e-8 and
  .controls.swapped_module_boundary_difference > 1e-8 and
  .controls.null_carrier_rejected and
  .resources.accepted_resident_complex_cells == 285 and
  .resources.accepted_temporary_necklace_complex_cells == 855 and
  .resources.accepted_temporary_occupation_complex_cells == 0 and
  .resources.accepted_retained_dense_operator_complex_cells == 0 and
  .resources.accepted_retained_inverse_history_bytes == 0 and
  .resources.accepted_pair_plan_integer_cells == 9 and
  .resources.maximum_named_engine_bytes == 33542 and
  (.accepted_path_labelled_wave_materialized | not) and
  (.accepted_path_occupation_vector_materialized | not) and
  (.accepted_path_dense_285_operator_materialized | not) and
  (.accepted_path_pair_energy_table_materialized | not) and
  (.public_topology_compilation_inspects_final_answer | not) and
  (.response_order_machine_enforced | not) and
  .matched_bosonic_classical_recurrence == "IDENTICAL_285_COMPLEX_NECKLACE_GENERATOR_AND_DIAGONAL_PAIR_PHASE_RECURRENCE" and
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
  .oracle_representation == "DENSE_LABELLED_17_TO_THE_4_COMPLEX_WAVE_VERIFICATION_ONLY" and
  .labelled_complex_cells == 83521 and
  .dense_work_complex_cells == 83521 and
  .dense_pair_exponent_integer_cells == 83521 and
  .dense_pair_signature_integer_cells == 751689 and
  .necklace_cells_reconstructed == 285 and
  .distinct_pair_signatures == 165 and
  .rotation_invariant and
  .exchange_symmetric and
  .collision_degenerate_states_split and
  .primary.norm_error < 2e-10 and
  .primary.restoration_error < 2e-10 and
  .reuse.norm_error < 2e-10 and
  .reuse.restoration_error < 2e-10 and
  .controls.missing_inverse_error > 1e-6 and
  .controls.wrong_inverse_error > 1e-6 and
  .controls.reordered_inverse_error > 1e-6 and
  .accepted_path_dense_labelled_wave_cells == 0 and
  .accepted_path_dense_pair_signature_cells == 0 and
  (.distinct_phase_resource_established | not) and
  (.small_wall_crossed | not)
' "$sealed_oracle" >/dev/null

"$python" -c '
import json
import sys
production = json.load(open(sys.argv[1], encoding="utf-8"))
oracle = json.load(open(sys.argv[2], encoding="utf-8"))
for key, oracle_key in (("primary_boundary", "primary"), ("reuse_boundary", "reuse")):
    difference = max(
        abs(left - right)
        for left, right in zip(
            production[key], oracle[oracle_key]["boundary"], strict=True
        )
    )
    if difference > 1e-10:
        raise SystemExit(f"independent labelled boundary differs: {key}")
for key in ("missing_inverse_error", "wrong_inverse_error", "reordered_inverse_error"):
    if abs(production["controls"][key] - oracle["controls"][key]) > 1e-10:
        raise SystemExit(f"independent inverse control differs: {key}")
' "$sealed" "$sealed_oracle"

"$python" -c '
import ast
import pathlib
import sys
tree = ast.parse(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
for node in ast.walk(tree):
    if isinstance(node, ast.Import):
        names = [alias.name for alias in node.names]
    elif isinstance(node, ast.ImportFrom):
        names = [node.module or ""]
    else:
        continue
    if any("necklace_two_body" in name or "four_rotor" in name for name in names):
        raise SystemExit("two-body oracle production dependency isolation failed")
' "$oracle"

sha256sum \
  "$production" \
  "$oracle" \
  "$here/four_rotor_necklace_generator_phase.cpp" \
  "$here/four_rotor_bosonic_givens_phase.cpp" \
  "$here/four_rotor_necklace_orbit_phase.cpp" \
  "$sealed" \
  "$sealed_oracle"
printf '%s\n' 'QUALIFIED_NECKLACE_TWO_BODY_PHASE_INTERACTION_CLOSURE_STRICT_SCOPE'
printf 'evidence=tracked-in-place:%s\n' "$here"
