#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
export PYTHONHASHSEED=0
ulimit -c 0

if [[ $# -ne 1 ]]; then
    echo "usage: qualify_continuous_kicked_phase_fourier.sh EVIDENCE_DIR" >&2
    exit 2
fi

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)
python="$repo/.venv/bin/python"
program="$here/continuous_kicked_phase_fourier.py"
out=$1
mkdir -p "$out"

for tool in jq rg sha256sum nice; do
    command -v "$tool" >/dev/null
done
[[ -x "$python" ]]
"$python" -m py_compile "$program"
env PYTHONPATH="$here" nice -n 10 "$python" -X dev "$program" \
    >"$out/result.json" 2>"$out/stderr"
[[ ! -s "$out/stderr" ]]

jq -e '
  .result == "PASS"
  and .claim_candidate
    == "BOUNDED_NUMERICAL_CONTINUOUS_IRRATIONAL_KICKED_PHASE_COHERENT_FOURIER_LOCALIZATION_CONTRAST_WITH_EFFECTIVE_BANDWIDTH_PLATEAU_ACTUAL_RESTORATION_AND_REUSE"
  and .periodic_effective_fixed_bandwidth_observed
  and (.periodic_localization_window_radii | max)
    - (.periodic_localization_window_radii | min) <= 2
  and (.scrambled_control_radii[-1]
    > 20 * (.periodic_localization_window_radii | max))
  and .periodic_cross_grid_boundary_max_error
    <= .predeclared_tolerances.cross_grid_boundary
  and .scrambled_cross_grid_boundary_max_error
    <= .predeclared_tolerances.cross_grid_boundary
  and .extended_precision_boundary_error
    <= .predeclared_tolerances.cross_grid_boundary
  and .primary.actual_inverse_restoration
  and .primary.restoration_generation == 1
  and (.primary.snapshot_loaded | not)
  and .reuse.actual_inverse_restoration
  and .reuse.restoration_generation == 2
  and (.reuse.snapshot_loaded | not)
  and .maximum_repeated_reuse_restoration_error
    <= .predeclared_tolerances.restoration_linf
  and .primary.retained_inverse_history_bytes == 0
  and .reuse.retained_inverse_history_bytes == 0
  and .matched_sparse_bessel_baseline.total_l2_error_against_fft
    <= .predeclared_tolerances.sparse_bessel_l2
  and .matched_sparse_bessel_baseline.classical_equivalent_of_phase_harmonic_recurrence
  and .snapshot_sham.snapshot_loaded
  and (.snapshot_sham.actual_inverse_restoration | not)
  and .snapshot_sham.restoration_generation == 0
  and .snapshot_sham.total_payload_copy_bytes == 131072
  and .snapshot_sham.primary_reload_error == 0
  and .snapshot_sham.reuse_consumed_reloaded_execution_carrier
  and .snapshot_sham.reuse_reload_error == 0
  and .controls.missing_inverse_error > 1e-5
  and .controls.wrong_inverse_error > 1e-5
  and .controls.reordered_inverse_error > 1e-5
  and .controls.fourier_disabled_difference > 1e-5
  and (.exact_finite_fourier_support | not)
  and .infinite_bessel_support_of_generic_kick
  and .phase_is_primitive_wave_amplitude
  and .unresolved_phase_preserved_across_composition
  and (.machine_enforced_hidden_intermediate | not)
  and (.distinct_phase_resource_established | not)
  and (.asymptotic_dynamical_localization_established | not)
  and (.scrambled_control_asymptotic_delocalization_established | not)
  and (.computational_advantage | not)
  and (.small_wall_crossed | not)
  and (.unbounded_computation_established | not)
  and (.physical_waveform_execution | not)
  and (.terminal | not)
' "$out/result.json" >/dev/null

if jq -c '{
  primary:.primary.boundary,
  reuse:.reuse.boundary,
  periodic_localization_window_radii,
  scrambled_control_radii
}' "$out/result.json" | rg -n \
    'samples|momentum|wavefunction|intermediate|candidate|tensor|assignment'
then
    echo "continuous kicked-phase result smuggled resident state" >&2
    exit 1
fi

sha256sum "$program" "$out/result.json" >"$out/SHA256SUMS"
echo "continuous kicked-phase Fourier qualification: PASS"
