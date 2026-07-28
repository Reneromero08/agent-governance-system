#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
export PYTHONHASHSEED=0
ulimit -c 0

if [[ $# -ne 1 ]]; then
    echo "usage: qualify_cyclotomic_f5_cubic_tt.sh EVIDENCE_DIR" >&2
    exit 2
fi

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)
python="$repo/.venv/bin/python"
out=$1
mkdir -p "$out"
phase="$here/cyclotomic_f5_cubic_tt_phase.py"
reference="$here/cyclotomic_f5_cubic_tt_reference.py"

for tool in jq cmp rg sha256sum nice; do
    command -v "$tool" >/dev/null
done
[[ -x "$python" ]]
"$python" -m py_compile "$phase" "$reference"

nice -n 10 env PYTHONPATH="$here" "$python" -X dev "$phase" \
    >"$out/phase.json" 2>"$out/phase.stderr"
nice -n 10 env PYTHONPATH="$here" "$python" -X dev "$phase" \
    >"$out/phase_replay.json" 2>"$out/replay.stderr"
cmp "$out/phase.json" "$out/phase_replay.json"
[[ ! -s "$out/phase.stderr" ]]
[[ ! -s "$out/replay.stderr" ]]

nice -n 10 env PYTHONPATH="$here" "$python" -X dev "$reference" \
    >"$out/reference.json" 2>"$out/reference.stderr"
[[ ! -s "$out/reference.stderr" ]]

jq -e '
  .result == "PASS"
  and .claim_candidate
    == "BOUNDED_EXACT_CYCLOTOMIC_CUBIC_PHASE_FOURIER_TENSOR_TRAIN_RANK_GROWTH_WITH_RESTORATION"
  and .field == "Q(ZETA5)"
  and .central_bond_ranks == [4,14,64]
  and [.fixtures[].width] == [2,4,6]
  and [.fixtures[].rounds] == [1,4,5]
  and [.fixtures[].central_bond_rank] == [4,14,64]
  and [.fixtures[].maximum_live_tensor_cells] == [40,750,9710]
  and [.fixtures[].maximum_merged_cells] == [25,625,4900]
  and ([.fixtures[].maximum_logical_coefficient_bytes] | all(. > 0))
  and ([.fixtures[].compiled_topology_logical_bytes] | all(. > 0))
  and ([.fixtures[].carrier_creation_tensor_cells] | all(. > 0))
  and ([.fixtures[].boundary_contraction_terms] | all(. > 0))
  and ([.fixtures[].maximum_factorization_scratch_logical_bytes] | all(. > 0))
  and ([.fixtures[].factorization_field_additions] | all(. > 0))
  and ([.fixtures[].factorization_field_multiplications] | all(. > 0))
  and ([.fixtures[].factorization_field_divisions] | all(. > 0))
  and ([.fixtures[].restoration_field_inversions] | all(. > 0))
  and ([.fixtures[].restoration_field_multiplications] | all(. > 0))
  and ([.fixtures[].restoration_verification_cells] | all(. > 0))
  and ([.fixtures[].actual_inverse_restoration] | all)
  and ([.fixtures[].retained_inverse_matrices] | all(. == 0))
  and ([.fixtures[].dense_assignment_cells] | all(. == 0))
  and ([.fixtures[].matched_exact_tt_is_same_as_accepted_representation] | all)
  and ([.fixtures[].actual_python_allocation_measured] | all(. | not))
  and .controls.identity_gate_operator_schmidt_rank == 1
  and .controls.separable_gate_operator_schmidt_rank == 1
  and .controls.bilinear_clifford_gate_operator_schmidt_rank == 5
  and .controls.cubic_gate_operator_schmidt_rank == 4
  and (.controls.missing_inverse_restored | not)
  and (.controls.wrong_inverse_restored | not)
  and (.controls.reordered_inverse_restored | not)
  and .controls.forced_rank_cap_rejected
  and .controls.snapshot_baseline_tensor_cells == 20
  and .controls.snapshot_forward_tensor_cells == 200
  and .controls.snapshot_creation_traffic_bytes == 160
  and .controls.snapshot_reload_traffic_bytes == 160
  and .controls.snapshot_forward_boundary_matches
  and (.controls.snapshot_actual_inverse | not)
  and .controls.snapshot_restoration_generation == 0
  and .controls.snapshot_reuse_restored
  and .same_carrier_transactions == 2
  and .actual_restored_carrier_reuse
  and .fourier_disabled_boundary_differs
  and .phase_is_primitive_wave_coupling
  and .roots_alone_not_claimed_as_resource
  and (.fixed_rank_closure_established | not)
  and (.distinct_phase_resource_established | not)
  and (.computational_advantage | not)
  and (.small_wall_crossed | not)
  and (.physical_waveform_execution | not)
  and (.terminal | not)
' "$out/phase.json" >/dev/null

jq -e '
  .result == "PASS"
  and .oracle
    == "INDEPENDENT_DUAL_FINITE_FIELD_TENSOR_TRAIN_RANK_AND_BOUNDARY_CERTIFICATE"
  and .central_bond_ranks == [4,14,64]
  and [.fields[].prime] == [11,31]
  and ([.fields[].fixtures[].bond_ranks] | length == 6)
  and (.dense_assignment_expansion | not)
  and .modular_rank_lower_bounds_exact_rank
  and (.terminal | not)
' "$out/reference.json" >/dev/null

for denied in project-intermediate null-carrier; do
    set +e
    env PYTHONPATH="$here" "$python" "$phase" "--${denied}" \
        >"$out/${denied}.stdout" \
        2>"$out/${denied}.stderr"
    rc=$?
    set -e
    [[ "$rc" -eq 2 ]]
    [[ ! -s "$out/${denied}.stdout" ]]
done
rg -q '^cyclotomic TT intermediate projection denied$' \
    "$out/project-intermediate.stderr"
rg -q '^invalid cyclotomic TT carrier$' \
    "$out/null-carrier.stderr"

if rg -n \
    'witness_list|candidate_set|truth_table|assignment_expansion|dense_state|statevector' \
    "$phase"
then
    echo "cyclotomic TT phase path contains forbidden extensional state" >&2
    exit 1
fi
rg -q 'dense_assignment_expansion.*False' "$reference"
rg -q 'exact skeleton factorization failed' "$phase"
rg -q 'modular skeleton reconstruction failed' "$reference"

jq -n \
  --slurpfile phase "$out/phase.json" \
  --slurpfile reference "$out/reference.json" '
  {
    result: "PASS",
    claim:
      "BOUNDED_EXACT_CYCLOTOMIC_CUBIC_PHASE_FOURIER_TENSOR_TRAIN_SEQUENTIAL_RANK_GROWTH_WITH_ACTUAL_RESTORATION_AND_REUSE",
    claim_ceiling:
      "BOUNDED_F5_WIDTHS2_4_6_CENTRAL_CROSSINGS1_2_3_EXACT_Q_ZETA5_SOFTWARE_TENSOR_TRAIN_ONLY",
    accepted: $phase[0],
    independent_dual_field_certificate: $reference[0],
    matched_exact_tt_has_identical_rank_and_state_growth: true,
    distinct_phase_resource_established: false,
    computational_advantage: false,
    small_wall_crossed: false,
    terminal: false
  }
' >"$out/qualification.json"

sha256sum \
    "$phase" "$reference" \
    "$out/phase.json" "$out/reference.json" \
    "$out/qualification.json" \
    >"$out/SHA256SUMS"

echo "cyclotomic F5 cubic TT qualification: PASS"
