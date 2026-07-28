#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
export PYTHONHASHSEED=0
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
ulimit -c 0

if [[ $# -ne 1 ]]; then
    echo "usage: qualify_four_rotor_rotation_quotient_phase.sh EVIDENCE_DIR" >&2
    exit 2
fi

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)
python="$repo/.venv/bin/python"
out=$1
mkdir -p "$out"
program="$here/four_rotor_rotation_quotient_phase.py"
dense="$here/four_rotor_cyclic_phase_law.py"
qualifier="$here/qualify_four_rotor_rotation_quotient_phase.sh"

for tool in jq sha256sum nice taskset; do
    command -v "$tool" >/dev/null
done
[[ -x "$python" ]]
"$python" -m py_compile "$program" "$dense"

env PYTHONPATH="$here" nice -n 10 taskset -c 0-3 \
    "$python" -X dev "$program" \
    >"$out/result.json" 2>"$out/stderr"
[[ ! -s "$out/stderr" ]]

jq -e '
  .result == "PASS"
  and .quotient_geometry
    == "THREE_RELATIVE_ANGLES_WITH_TOTAL_MOMENTUM_ZERO_MOD17"
  and .original_angle_coordinates == 4
  and .quotient_angle_coordinates == 3
  and .global_rotation_coordinates_removed == 1
  and .onsite_phase_updates == 0
  and .state_reduction_factor == 17
  and (.depth_growth | map(.depth)) == [1,2,4,8,16,32,64]
  and (.depth_growth
    | all(.full_state_error_vs_dense_lift < 2e-11))
  and (.depth_growth | all(.boundary_error_vs_dense < 2e-11))
  and (.depth_growth | all(.restoration_error < 2e-11))
  and .depth_independent_memory_signature
  and .primary.full_state_error_vs_dense_lift < 2e-11
  and .primary.boundary_error_vs_dense < 2e-11
  and .primary.restoration_error < 2e-11
  and .primary.restoration_generation == 1
  and .primary.actual_inverse_restoration
  and .primary.actual_restored_carrier_reuse_ready
  and .primary.resources.quotient_carrier_complex_cells == 4913
  and .primary.resources.quotient_carrier_payload_bytes == 78608
  and .primary.resources.dense_four_rotor_complex_cells == 83521
  and .primary.resources.dense_to_quotient_state_ratio == 17
  and .primary.resources.retained_public_plan_bytes == 78880
  and .primary.resources.plan_compilation_peak_payload_bytes == 157760
  and .primary.resources.maximum_pair_factor_cells == 289
  and .primary.resources.maximum_explicit_engine_array_bytes == 236368
  and .primary.resources.maximum_explicit_wrapper_array_bytes == 314976
  and .primary.resources.verification_dense_lift_bytes == 2672672
  and .primary.resources.verification_lift_index_bytes == 7072
  and .primary.resources.verification_peak_explicit_array_bytes == 2915840
  and .primary.resources.verification_baseline_reload_count == 0
  and (.primary.resources.verification_baseline_used_for_restoration | not)
  and .primary.resources.retained_inverse_history_bytes == 0
  and .primary.resources.fft_backing_buffer_preserved
  and .reuse.restoration_generation == 2
  and .reuse.restoration_error < 2e-11
  and .fresh_restored_boundary_error < 2e-11
  and .controls.missing_inverse_error > 1e-4
  and .controls.wrong_inverse_error > 1e-4
  and .controls.reordered_inverse_error > 1e-4
  and .matched_classical_rotation_quotient_identical
  and (.distinct_phase_resource_established | not)
  and (.computational_advantage | not)
  and (.small_wall_crossed | not)
  and (.unbounded_computation_established | not)
  and (.physical_waveform_execution | not)
  and (.terminal | not)
' "$out/result.json" >/dev/null

sha256sum "$program" "$dense" "$qualifier" "$out/result.json" \
    >"$out/SHA256SUMS"
echo "four-rotor global-rotation quotient qualification: PASS"
