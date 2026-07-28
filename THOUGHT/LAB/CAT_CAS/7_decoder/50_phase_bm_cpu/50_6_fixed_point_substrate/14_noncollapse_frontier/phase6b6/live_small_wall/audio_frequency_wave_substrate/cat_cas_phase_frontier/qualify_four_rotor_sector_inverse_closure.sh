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
    echo "usage: qualify_four_rotor_sector_inverse_closure.sh EVIDENCE_DIR" >&2
    exit 2
fi

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)
python="$repo/.venv/bin/python"
out=$1
mkdir -p "$out"

for tool in jq sha256sum nice taskset; do
    command -v "$tool" >/dev/null
done
[[ -x "$python" ]]

program="$here/four_rotor_sector_inverse_closure.py"
incremental="$here/four_rotor_incremental_schmidt_closure.py"
reference="$here/four_rotor_kicked_phase_tt.py"
matrix_free="$here/four_rotor_kicked_phase_tt_matrix_free.py"
canonical="$here/four_rotor_post_inverse_canonical_closure.py"
qualifier="$here/qualify_four_rotor_sector_inverse_closure.sh"

"$python" -m py_compile \
    "$program" "$incremental" "$reference" "$matrix_free" "$canonical"

env PYTHONPATH="$here" nice -n 10 taskset -c 0-3 \
    "$python" -X dev "$program" \
    >"$out/result.json" 2>"$out/stderr"
[[ ! -s "$out/stderr" ]]

jq -e '
  .result == "PASS"
  and .claim_candidate
    == "BOUNDED_TOPOLOGY_DERIVED_TOTAL_MOMENTUM_SECTOR_INVERSE_PHASE_CLOSURE_UPDATE_REDUCTION_WITH_GRAM_REMATERIALIZATION_OBSTRUCTION_ACTUAL_RESTORATION_AND_REUSE"
  and .primary.central_rank_history == [11,47,117]
  and .primary.resources.forward_incremental_updates == 117
  and .primary.resources.inverse_sector_closures == 9
  and .primary.resources.total_closure_updates == 126
  and .primary.resources.gram_rematerializations == 18
  and .primary.resources.sector_rhs_rematerializations == 73167
  and .primary.resources.public_plan_complex_cells == 16269
  and .primary.resources.public_plan_pivot_cells == 841
  and .primary.resources.maximum_sector_condition < 2
  and .primary.resources.maximum_sector_inverse_residual < 1e-12
  and .primary.resources.retained_inverse_history_bytes == 0
  and .primary.resources.probe_columns == 0
  and .primary.inverse_restoration_error < 5e-5
  and .primary.postclosure_restoration_error < 5e-5
  and .primary.closure.bond_ranks_after == [1,1,1]
  and .primary.actual_inverse_restoration
  and .primary.actual_restored_carrier_reuse_ready
  and .reuse.restoration_generation == 2
  and .fresh_restored_boundary_error < 1e-5
  and .fresh_restored_resource_signature_exact
  and .controls.missing_inverse_error > 1e-4
  and .controls.wrong_inverse_error > 1e-4
  and .controls.reordered_inverse_error > 1e-4
  and .tradeoff.closure_update_reduction_established
  and (.tradeoff.peak_memory_reduction_established | not)
  and (.tradeoff.warm_time_reduction_established | not)
  and .tradeoff.sector_over_incremental_peak_ratio > 2
  and .tradeoff.identified_obstruction
    == "EXACT_GRAM_AND_SECTOR_RHS_REMATERIALIZATION_COST"
  and .matched_classical_sector_algorithm_identical
  and (.distinct_phase_resource_established | not)
  and (.computational_advantage | not)
  and (.small_wall_crossed | not)
  and (.unbounded_computation_established | not)
  and (.terminal | not)
' "$out/result.json" >/dev/null

sha256sum \
    "$program" "$incremental" "$reference" "$matrix_free" \
    "$canonical" "$qualifier" "$out/result.json" \
    >"$out/SHA256SUMS"

echo "four-rotor sector inverse closure qualification: PASS"
