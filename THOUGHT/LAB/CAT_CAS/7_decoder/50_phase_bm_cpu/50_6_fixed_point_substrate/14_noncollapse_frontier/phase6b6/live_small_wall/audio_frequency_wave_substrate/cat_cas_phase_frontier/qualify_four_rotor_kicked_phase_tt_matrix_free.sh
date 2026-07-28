#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
export PYTHONHASHSEED=0
ulimit -c 0

if [[ $# -ne 1 ]]; then
    echo "usage: qualify_four_rotor_kicked_phase_tt_matrix_free.sh EVIDENCE_DIR" >&2
    exit 2
fi

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)
python="$repo/.venv/bin/python"
program="$here/four_rotor_kicked_phase_tt_matrix_free.py"
reference="$here/four_rotor_kicked_phase_tt.py"
out=$1
mkdir -p "$out"

for tool in jq rg sha256sum nice; do
    command -v "$tool" >/dev/null
done
"$python" -m py_compile "$program" "$reference"
env PYTHONPATH="$here" nice -n 10 "$python" -X dev "$program" \
    >"$out/result.json" 2>"$out/stderr"
[[ ! -s "$out/stderr" ]]

jq -e '
  .result == "PASS"
  and .claim_candidate
    == "BOUNDED_MATRIX_FREE_STREAMED_BESSEL_SCHMIDT_CLOSURE_WITHOUT_EXPANDED_MPO_OR_DENSE_INTERFACE_CORE_WITH_ACTUAL_RESTORATION_AND_REUSE"
  and .primary.actual_inverse_restoration
  and .primary.restoration_generation == 1
  and .primary.restoration_error
    <= .predeclared_tolerances.restoration_l2
  and .reuse.actual_inverse_restoration
  and .reuse.restoration_generation == 2
  and .reuse.restoration_error
    <= .predeclared_tolerances.restoration_l2
  and .primary.retained_inverse_history_bytes == 0
  and (.primary.dense_interface_core_materialized | not)
  and (.primary.expanded_mpo_bond_materialized | not)
  and .boundary_error_against_dense_reference
    <= .predeclared_tolerances.boundary_parity
  and ((.primary.central_rank_history[0]
    - .dense_reference_same_tolerance.central_rank_history[0]) | fabs) <= 1
  and ((.primary.central_rank_history[1]
    - .dense_reference_same_tolerance.central_rank_history[1]) | fabs) <= 1
  and ((.primary.central_rank_history[2]
    - .dense_reference_same_tolerance.central_rank_history[2]) | fabs) <= 1
  and .resource_repair.expanded_mpo_bond_eliminated
  and .resource_repair.dense_interface_core_eliminated
  and .resource_repair.public_bessel_terms_rematerialized
  and .resource_repair.frobenius_tail_certified
  and .resource_repair.matrix_free_peak_payload_bytes
    < .resource_repair.prior_strict_path_peak_payload_bytes
  and (.resource_repair.matrix_free_peak_below_dense_equivalent | not)
  and .resource_repair.largest_workspace_array_cells
    < .resource_repair.dense_interface_core_cells
  and .controls.missing_inverse_error > 1e-4
  and .controls.wrong_inverse_error > 1e-4
  and .controls.reordered_inverse_error > 1e-4
  and .snapshot_sham.snapshot_loaded
  and (.snapshot_sham.actual_inverse_restoration | not)
  and .snapshot_sham.restoration_generation == 0
  and .snapshot_sham.reload_error == 0
  and .matched_classical_tt_is_identical_representation
  and (.fixed_central_rank_closure | not)
  and (.distinct_phase_resource_established | not)
  and (.computational_advantage | not)
  and (.small_wall_crossed | not)
  and (.unbounded_computation_established | not)
  and (.physical_waveform_execution | not)
  and (.terminal | not)
' "$out/result.json" >/dev/null

sha256sum "$program" "$reference" "$out/result.json" \
    >"$out/SHA256SUMS"
echo "matrix-free four-rotor kicked-phase TT qualification: PASS"
