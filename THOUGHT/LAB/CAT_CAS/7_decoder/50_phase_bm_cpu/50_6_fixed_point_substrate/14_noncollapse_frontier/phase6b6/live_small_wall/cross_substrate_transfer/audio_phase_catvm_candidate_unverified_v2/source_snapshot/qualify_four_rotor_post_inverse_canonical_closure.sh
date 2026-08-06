#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: qualify_four_rotor_post_inverse_canonical_closure.sh EVIDENCE_DIR" >&2
  exit 2
fi

evidence_dir=$1
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(git -C "$script_dir" rev-parse --show-toplevel)
python_bin="$repo_root/.venv/bin/python"

mkdir -p "$evidence_dir"
"$python_bin" "$script_dir/four_rotor_post_inverse_canonical_closure.py" \
  >"$evidence_dir/result.json" 2>"$evidence_dir/stderr"
jq -e '
  .result == "PASS"
  and .primary_actual_inverse.missing_closure_bond_ranks == [29,166,29]
  and .primary_actual_inverse.closure.bond_ranks_after == [1,1,1]
  and .primary_actual_inverse.closure.baseline_state_consulted == false
  and .primary_actual_inverse.closure.snapshot_loaded == false
  and .primary_actual_inverse.closure.retained_backing_matches_logical_cells
      == true
  and .actual_restored_reuse.central_rank_history
      == .fresh_reuse_baseline.central_rank_history
  and .fresh_restored_resource_signature_exact == true
  and .retained_numpy_backing_allocations_counted == true
  and .actual_restored_reuse.ending_bond_ranks == [1,1,1]
  and .actual_carrier_generation == 2
  and .snapshot_loaded_on_accepted_path == false
  and .retained_inverse_history_bytes == 0
  and .computational_advantage == false
  and .small_wall_crossed == false
  and .terminal == false
' "$evidence_dir/result.json" >/dev/null

sha256sum \
  "$script_dir/four_rotor_post_inverse_canonical_closure.py" \
  "$script_dir/four_rotor_kicked_phase_tt.py" \
  "$script_dir/four_rotor_kicked_phase_tt_matrix_free.py" \
  "$evidence_dir/result.json" \
  >"$evidence_dir/SHA256SUMS"

echo "four-rotor post-inverse canonical closure qualification: PASS"
