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
    echo "usage: qualify_four_rotor_compact_cyclic_phase_tt.sh EVIDENCE_DIR" >&2
    exit 2
fi

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)
python="$repo/.venv/bin/python"
out=$1
mkdir -p "$out"
program="$here/four_rotor_compact_cyclic_phase_tt.py"
qualifier="$here/qualify_four_rotor_compact_cyclic_phase_tt.sh"

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
  and (.depth_growth | map(.depth)) == [1,2,3,4,5]
  and (.depth_growth | map(.central_boundary_rank))
    == [17,92,201,280,289]
  and (.depth_growth | map(.rank_history[-1][1]))
    == [17,95,207,281,289]
  and (.depth_growth | all(.boundary_error_vs_dense < 2e-9))
  and (.depth_growth
    | all(.postclosure_restoration_error < 2e-9))
  and .primary.actual_inverse_restoration
  and .primary.actual_restored_carrier_reuse_ready
  and .primary.inverse_restoration_error < 2e-9
  and .primary.postclosure_restoration_error < 2e-9
  and .primary.closure.bond_ranks_after == [1,1,1]
  and .primary.resources.maximum_bond_rank == 289
  and .primary.resources.maximum_logical_tt_cells == 167620
  and .primary.resources.maximum_two_site_core_cells == 83521
  and .primary.resources.retained_inverse_history_bytes == 0
  and (.primary.resources.physical_coordinate_dense_wave_materialized | not)
  and (.primary.resources.simultaneous_peak_payload_established | not)
  and .reuse.restoration_generation == 2
  and .fresh_restored_boundary_error < 2e-9
  and .controls.missing_inverse_error > 1e-4
  and .controls.wrong_inverse_error > 1e-4
  and .controls.reordered_inverse_error > 1e-4
  and .exact_central_rank_ceiling == 289
  and .central_rank_fraction_of_exact_width == 1
  and .central_rank_near_saturated_exact_width
  and .dense_cyclic_carrier_cells == 83521
  and .logical_tt_over_dense_cell_ratio > 2
  and .dense_equivalent_interface_core_materialized
  and .identified_obstruction
    == "COMPACT_CYCLIC_PHASE_TT_CENTRAL_RANK_SATURATES_DENSE_INTERFACE"
  and .matched_classical_cyclic_tt_identical
  and (.compact_fixed_rank_across_depth_established | not)
  and (.distinct_phase_resource_established | not)
  and (.computational_advantage | not)
  and (.small_wall_crossed | not)
  and (.unbounded_computation_established | not)
  and (.terminal | not)
' "$out/result.json" >/dev/null

sha256sum "$program" "$qualifier" "$out/result.json" \
    >"$out/SHA256SUMS"
echo "four-rotor compact cyclic phase TT qualification: PASS"
