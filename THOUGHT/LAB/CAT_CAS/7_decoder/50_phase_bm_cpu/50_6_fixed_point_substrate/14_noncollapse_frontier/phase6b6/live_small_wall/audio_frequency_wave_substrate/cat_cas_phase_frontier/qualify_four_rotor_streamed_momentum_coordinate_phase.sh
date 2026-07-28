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
    echo "usage: qualify_four_rotor_streamed_momentum_coordinate_phase.sh EVIDENCE_DIR" >&2
    exit 2
fi

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)
python="$repo/.venv/bin/python"
out=$1
mkdir -p "$out"
program="$here/four_rotor_streamed_momentum_coordinate_phase.py"
quotient="$here/four_rotor_rotation_quotient_phase.py"
dense="$here/four_rotor_cyclic_phase_law.py"
qualifier="$here/qualify_four_rotor_streamed_momentum_coordinate_phase.sh"

for tool in jq sha256sum nice taskset; do
    command -v "$tool" >/dev/null
done
[[ -x "$python" ]]
"$python" -m py_compile "$program" "$quotient" "$dense"

env PYTHONPATH="$here" nice -n 10 taskset -c 0-3 \
    "$python" -X dev "$program" \
    >"$out/result.json" 2>"$out/stderr"
[[ ! -s "$out/stderr" ]]

jq -e '
  .result == "PASS"
  and (.depth_growth | map(.depth)) == [1,2,4,8,16,32,64]
  and (.depth_growth
    | all(.full_state_error_vs_dense_lift < 2e-11))
  and (.depth_growth | all(.boundary_error_vs_dense < 2e-11))
  and (.depth_growth | all(.restoration_error < 2e-11))
  and .depth_independent_resource_signature
  and .primary.actual_inverse_restoration
  and .primary.actual_restored_carrier_reuse_ready
  and .primary.full_state_error_vs_dense_lift < 2e-11
  and .primary.boundary_error_vs_dense < 2e-11
  and .primary.restoration_error < 2e-11
  and .primary.resources.quotient_carrier_complex_cells == 4913
  and .primary.resources.quotient_carrier_payload_bytes == 78608
  and .primary.resources.retained_public_plan_bytes == 408
  and .primary.resources.plan_compilation_peak_payload_bytes == 408
  and .primary.resources.maximum_explicit_engine_array_bytes == 118592
  and .primary.resources.maximum_explicit_wrapper_array_bytes == 197200
  and .primary.resources.maximum_pair_factor_cells == 289
  and .primary.resources.maximum_free_slice_cells == 17
  and .primary.resources.total_momentum_coordinate_closures == 18496
  and (.primary.resources.dense_free_phase_plan_materialized | not)
  and .primary.resources.retained_inverse_history_bytes == 0
  and .primary.resources.fft_backing_buffer_preserved
  and .primary.resources.verification_baseline_reload_count == 0
  and (.primary.resources.verification_baseline_used_for_restoration | not)
  and .primary.resources.verification_peak_explicit_array_bytes == 2837368
  and .reuse.restoration_generation == 2
  and .reuse.restoration_error < 2e-11
  and .fresh_restored_boundary_error < 2e-11
  and .controls.missing_inverse_error > 1e-4
  and .controls.wrong_inverse_error > 1e-4
  and .controls.reordered_inverse_error > 1e-4
  and .previous_dense_quotient_retained_plan_bytes == 78880
  and .streamed_retained_plan_bytes == 408
  and .retained_plan_reduction_factor > 193
  and .unprojected_topology_derived_internal_coordinate
    == "TOTAL_MOMENTUM_N0_MOD17"
  and (.internal_coordinate_projected | not)
  and (.phase_resident_unresolved_internal_port_established | not)
  and (.warm_time_reduction_established | not)
  and .matched_classical_streamed_quotient_identical
  and (.distinct_phase_resource_established | not)
  and (.computational_advantage | not)
  and (.small_wall_crossed | not)
  and (.unbounded_computation_established | not)
  and (.terminal | not)
' "$out/result.json" >/dev/null

sha256sum "$program" "$quotient" "$dense" "$qualifier" \
    "$out/result.json" >"$out/SHA256SUMS"
echo "four-rotor streamed total-momentum coordinate qualification: PASS"
