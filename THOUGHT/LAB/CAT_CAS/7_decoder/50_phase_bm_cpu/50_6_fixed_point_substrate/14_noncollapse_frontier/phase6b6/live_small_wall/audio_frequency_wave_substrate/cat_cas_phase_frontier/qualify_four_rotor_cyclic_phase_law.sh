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
    echo "usage: qualify_four_rotor_cyclic_phase_law.sh EVIDENCE_DIR" >&2
    exit 2
fi

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)
python="$repo/.venv/bin/python"
out=$1
mkdir -p "$out"
program="$here/four_rotor_cyclic_phase_law.py"
qualifier="$here/qualify_four_rotor_cyclic_phase_law.sh"

for tool in jq sha256sum nice taskset; do
    command -v "$tool" >/dev/null
done
[[ -x "$python" ]]
"$python" -m py_compile "$program"

env PYTHONPATH="$here" nice -n 10 taskset -c 0-3 \
    "$python" -X dev "$program" \
    >"$out/result.json" 2>"$out/stderr"
[[ ! -s "$out/stderr" ]]

jq -e '
  .result == "PASS"
  and .primary.restoration_error < 2e-11
  and .primary.restoration_generation == 1
  and .primary.actual_inverse_restoration
  and .primary.actual_restored_carrier_reuse_ready
  and .primary.verification_baseline_reload_count == 0
  and (.primary.verification_baseline_used_for_restoration | not)
  and .primary.resources.retained_inverse_history_bytes == 0
  and .primary.resources.dense_operator_materialization_bytes == 0
  and .primary.resources.decoded_intermediate_bytes == 0
  and .primary.resources.fft_backing_buffer_preserved
  and .reuse.restoration_generation == 2
  and .fresh_restored_boundary_error < 2e-11
  and .matched_direct_boundary_error < 2e-11
  and .depth_independent_memory_signature
  and (.depth_growth | map(.depth)) == [1,2,4,8,16,32,64]
  and (.depth_growth | map(.carrier_payload_bytes) | unique | length) == 1
  and (.depth_growth
    | map(.maximum_accounted_engine_array_bytes)
    | unique
    | length) == 1
  and (.depth_growth
    | map(.wrapper_accounted_peak_array_bytes)
    | unique
    | length) == 1
  and (.depth_growth
    | map(.retained_inverse_history_bytes)
    | unique) == [0]
  and (.depth_growth | all(.fft_backing_buffer_preserved))
  and .dense_cyclic_path_avoids_sector_inverse_work
    .sector_rhs_rematerializations == 0
  and .dense_cyclic_path_avoids_sector_inverse_work
    .gram_rematerializations == 0
  and .controls.missing_inverse_error > 1e-4
  and .controls.wrong_inverse_error > 1e-4
  and .controls.reordered_inverse_error > 1e-4
  and .matched_classical_cyclic_fft_identical
  and (.compact_tt_advantage_established | not)
  and (.distinct_phase_resource_established | not)
  and (.computational_advantage | not)
  and (.small_wall_crossed | not)
  and (.unbounded_computation_established | not)
  and (.physical_waveform_execution | not)
  and (.terminal | not)
' "$out/result.json" >/dev/null

sha256sum "$program" "$qualifier" "$out/result.json" \
    >"$out/SHA256SUMS"
echo "four-rotor cyclic phase-law qualification: PASS"
