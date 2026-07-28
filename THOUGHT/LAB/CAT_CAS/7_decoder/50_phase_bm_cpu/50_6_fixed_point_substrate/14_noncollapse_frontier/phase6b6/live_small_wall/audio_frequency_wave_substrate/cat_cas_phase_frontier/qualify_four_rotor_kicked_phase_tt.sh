#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
export PYTHONHASHSEED=0
ulimit -c 0

if [[ $# -ne 1 ]]; then
    echo "usage: qualify_four_rotor_kicked_phase_tt.sh EVIDENCE_DIR" >&2
    exit 2
fi

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)
python="$repo/.venv/bin/python"
program="$here/four_rotor_kicked_phase_tt.py"
out=$1
mkdir -p "$out"

for tool in jq rg sha256sum nice; do
    command -v "$tool" >/dev/null
done
"$python" -m py_compile "$program"
env PYTHONPATH="$here" nice -n 10 "$python" -X dev "$program" \
    >"$out/result.json" 2>"$out/stderr"
[[ ! -s "$out/stderr" ]]

jq -e '
  .result == "PASS"
  and .claim_candidate
    == "BOUNDED_FOUR_ROTOR_NONSEPARABLE_CONTINUOUS_KICKED_PHASE_FOURIER_TT_CENTRAL_INTERFACE_RANK_GROWTH_WITH_ACTUAL_RESTORATION_AND_REUSE"
  and (.primary.central_rank_history | length == 3)
  and .primary.central_rank_history[0]
    < .primary.central_rank_history[1]
  and .primary.central_rank_history[1]
    < .primary.central_rank_history[2]
  and .primary.actual_inverse_restoration
  and .primary.restoration_generation == 1
  and (.primary.snapshot_loaded | not)
  and .primary.restoration_error
    <= .predeclared_tolerances.restoration_l2
  and .reuse.actual_inverse_restoration
  and .reuse.restoration_generation == 2
  and (.reuse.snapshot_loaded | not)
  and .reuse.restoration_error
    <= .predeclared_tolerances.restoration_l2
  and .primary.retained_inverse_history_bytes == 0
  and .mode_guard_boundary_errors_against_radius14[1]
    <= .predeclared_tolerances.boundary_guard
  and .controls.coupling_material
  and .controls.separable_central_rank == 1
  and .controls.missing_inverse_error > 1e-4
  and .controls.wrong_inverse_error > 1e-4
  and .controls.reordered_inverse_error > 1e-4
  and .snapshot_sham.snapshot_loaded
  and (.snapshot_sham.actual_inverse_restoration | not)
  and .snapshot_sham.restoration_generation == 0
  and .snapshot_sham.reload_error == 0
  and .snapshot_sham.reuse_consumed_reloaded_carrier
  and .snapshot_sham.reuse_reload_error == 0
  and .snapshot_sham.total_creation_and_copy_payload_bytes == 7424
  and (.global_dense_wave_materialized | not)
  and (.two_site_dense_wave_materialized | not)
  and .factorized_coupling_mpo
  and .unresolved_nonseparable_phase_relation
  and .resource_comparison.matched_classical_tt_is_identical_representation
  and (.fixed_central_rank_closure | not)
  and (.distinct_phase_resource_established | not)
  and (.computational_advantage | not)
  and (.small_wall_crossed | not)
  and (.unbounded_computation_established | not)
  and (.physical_waveform_execution | not)
  and (.terminal | not)
' "$out/result.json" >/dev/null

if jq -c '{
  primary:.primary.boundary,
  reuse:.reuse.boundary,
  primary_generation:.primary.restoration_generation,
  reuse_generation:.reuse.restoration_generation
}' "$out/result.json" | rg -n \
    'tensor|singular|intermediate|candidate|assignment|wavefunction'
then
    echo "four-rotor phase TT result smuggled resident state" >&2
    exit 1
fi

sha256sum "$program" "$out/result.json" >"$out/SHA256SUMS"
echo "four-rotor kicked-phase TT qualification: PASS"
