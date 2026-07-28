#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: qualify_four_rotor_incremental_schmidt_closure.sh EVIDENCE_DIR" >&2
  exit 2
fi

evidence_dir=$1
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(git -C "$script_dir" rev-parse --show-toplevel)
python_bin="$repo_root/.venv/bin/python"

mkdir -p "$evidence_dir"
"$python_bin" "$script_dir/four_rotor_incremental_schmidt_closure.py" \
  >"$evidence_dir/result.json" 2>"$evidence_dir/stderr"
jq -e '
  .result == "PASS"
  and .primary.stats.probe_columns == 0
  and .primary.stats.maximum_total_live_cells < 707281
  and .primary.stats.maximum_coupling_declared_l2_bound <= 1e-6
  and .primary.stats.combined_declared_l2_bound
      <= (.primary.stats.coupling_applications * 1e-6)
  and .primary.stats.maximum_basis_orthogonality_error <= 1e-10
  and .primary.closure.bond_ranks_after == [1,1,1]
  and .reuse.restoration_generation == 2
  and .fresh_restored_resource_signature_exact == true
  and .controls.missing_inverse_error > 1e-4
  and .controls.wrong_inverse_error > 1e-4
  and .controls.reordered_inverse_error > 1e-4
  and .resource_comparison.below_dense_equivalent_memory == true
  and .matched_classical_incremental_tt_is_identical == true
  and .computational_advantage == false
  and .small_wall_crossed == false
  and .terminal == false
' "$evidence_dir/result.json" >/dev/null

sha256sum \
  "$script_dir/four_rotor_incremental_schmidt_closure.py" \
  "$script_dir/four_rotor_post_inverse_canonical_closure.py" \
  "$script_dir/four_rotor_kicked_phase_tt.py" \
  "$script_dir/four_rotor_kicked_phase_tt_matrix_free.py" \
  "$evidence_dir/result.json" \
  >"$evidence_dir/SHA256SUMS"

echo "four-rotor incremental Schmidt closure qualification: PASS"
