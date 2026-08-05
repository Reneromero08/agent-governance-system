#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

if [[ $# -ne 1 ]]; then
  echo "usage: qualify_growing_rotor_open_momentum_factor_closure.sh MANAGED_BUILD_DIR" >&2
  exit 2
fi

build_dir=$1
case "$build_dir" in
  /dev/shm|/dev/shm/*|/run/shm|/run/shm/*)
    echo "RAM-backed build directories are forbidden" >&2
    exit 2
    ;;
esac

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(git -C "$here" rev-parse --show-toplevel)
python="$repo_root/.venv/bin/python"
production="$here/growing_rotor_open_momentum_factor_closure.py"
oracle="$here/growing_rotor_open_momentum_factor_independent_oracle.py"
sealed="$here/GROWING_ROTOR_OPEN_MOMENTUM_FACTOR_RESULTS.json"
sealed_oracle="$here/GROWING_ROTOR_OPEN_MOMENTUM_FACTOR_INDEPENDENT_ORACLE.json"

mkdir -p "$build_dir/pycache"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="$build_dir/pycache"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

nice -n 10 ionice -c 3 "$python" "$production" | "$python" -c '
import json, sys
sealed = json.load(open(sys.argv[1], encoding="utf-8"))
fresh = json.load(sys.stdin)
if sealed != fresh:
    raise SystemExit("open-momentum production reexecution differs from seal")
' "$sealed"

nice -n 10 ionice -c 3 "$python" "$oracle" | "$python" -c '
import json, sys
sealed = json.load(open(sys.argv[1], encoding="utf-8"))
fresh = json.load(sys.stdin)
if sealed != fresh:
    raise SystemExit("open-momentum oracle reexecution differs from seal")
' "$sealed_oracle"

jq -e '
  .classification == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE" and
  .verification_level == "INDEPENDENT_ORACLE_REEXECUTION" and
  .restoration_classification == "EXACT_ALGEBRAIC_RESTORATION" and
  .result == "PASS" and
  .factor_law.identity == "K_Q_EQUALS_A_MINUS_Q_COMPOSE_A_Q_MINUS_ROTOR_COUNT_IDENTITY" and
  .factor_law.reflection_pairs == [[1,16],[2,15],[3,14],[4,13],[5,12],[6,11],[7,10],[8,9]] and
  .factor_law.public_scattering_weights_are_equal_within_each_pair and
  .factor_law.open_port_logical_cells == 4389 and
  .factor_law.final_closed_bracelet_cells == 2277 and
  .factor_law.maximum_simultaneously_live_open_ports == 1 and
  (.factor_law.intermediate_projected | not) and
  .factor_law.intermediate_released_to_zero_after_each_pair and
  .factor_law.retained_transition_plan_entries == 0 and
  .factor_law.retained_inverse_history_bytes == 0 and
  .topology.occupation_histograms == 74613 and
  .topology.necklace_cells == 4389 and
  .topology.bracelet_cells == 2277 and
  (.topology.public_topology_compile_reads_final_answers | not) and
  .transaction.primary_boundary == 83 and
  .transaction.primary_signature_order_commitment == "834956d4d03066d651390a4e2d4b8c0b0940e8169f0b1fb7dfb62d201679c05e" and
  .transaction.reuse_boundary == 70 and
  .transaction.fresh_reuse_boundary == 70 and
  .transaction.fresh_restored_reuse_state_agreement and
  .transaction.primary_restoration_error_field_cells == 0 and
  .transaction.reuse_restoration_error_field_cells == 0 and
  .transaction.same_backing_primary and
  .transaction.same_backing_reuse and
  .transaction.restoration_generation_after_reuse == 2 and
  (.transaction.baseline_reload_used | not) and
  .controls.missing_inverse_error_field_cells > 0 and
  .controls.wrong_inverse_error_field_cells > 0 and
  .controls.reordered_inverse_error_field_cells > 0 and
  .controls.wrong_reflection_pair_error_field_cells > 0 and
  .controls.null_carrier_rejected and
  .controls.wrong_owner_rejected and
  .controls.wrong_momentum_type_rejected and
  .controls.premature_intermediate_projection_rejected and
  .controls.control_port_released_to_zero and
  .resource_law.accepted_one_body_terms_per_scattering == 331704 and
  .resource_law.accepted_first_pass_terms_per_scattering == 162792 and
  .resource_law.accepted_closure_terms_per_scattering == 168912 and
  .resource_law.accepted_cyclic_code_candidates_per_scattering == 5638968 and
  .resource_law.accepted_cyclic_code_rolling_updates_per_scattering == 5307264 and
  .resource_law.accepted_port_clear_field_cells_per_scattering == 35112 and
  .resource_law.prior_m197_two_body_moves_per_scattering == 684624 and
  .resource_law.prior_m197_triangle_monomial_evaluations_per_scattering == 24767280 and
  .resource_law.accepted_retained_transition_plan_entries == 0 and
  .resource_law.accepted_retained_inverse_history_bytes == 0 and
  .resource_law.relation_table_or_assignment_expansion_cells == 0 and
  (.catvm_custody | not) and
  (.distinct_phase_resource_established | not) and
  (.computational_advantage | not) and
  (.small_wall_crossed | not) and
  (.physical_waveform_execution | not) and
  (.physical_bit_replacement | not) and
  (.unbounded_computation_established | not) and
  (.terminal | not)
' "$sealed" >/dev/null

jq -e '
  .classification == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE" and
  .verification_level == "INDEPENDENT_ORACLE_REEXECUTION" and
  .restoration_classification == "EXACT_ALGEBRAIC_RESTORATION" and
  .result == "PASS" and
  .topology.occupation_histograms == 74613 and
  .topology.necklace_cells == 4389 and
  .topology.bracelet_cells == 2277 and
  .factor_verification.oracle_factor_channels == 16 and
  .factor_verification.oracle_first_pass_plan_entries == 325584 and
  .factor_verification.oracle_closure_plan_entries == 168912 and
  .factor_verification.oracle_full_unpaired_factor_terms_per_scattering == 494496 and
  .factor_verification.production_reflection_paired_term_law == 331704 and
  .factor_verification.direct_two_body_raw_terms == 684624 and
  .factor_verification.direct_primary_csr_nonzeros == 172838 and
  .factor_verification.primary_factor_direct_mismatch_cells == 0 and
  .factor_verification.reuse_factor_fresh_transaction_mismatch_cells == 0 and
  .transaction.primary_boundary == 83 and
  .transaction.reuse_boundary == 70 and
  .transaction.fresh_reuse_boundary == 70 and
  .transaction.primary_restoration_error_field_cells == 0 and
  .transaction.reuse_restoration_error_field_cells == 0 and
  .transaction.fresh_restoration_error_field_cells == 0 and
  .transaction.same_backing_primary and
  .transaction.same_backing_reuse and
  .transaction.same_backing_fresh and
  .transaction.fresh_restored_reuse_state_agreement and
  (.transaction.baseline_reload_used | not) and
  .controls.missing_inverse_error_field_cells > 0 and
  .controls.wrong_inverse_error_field_cells > 0 and
  .controls.reordered_inverse_error_field_cells > 0 and
  .controls.omitted_identity_correction_error_field_cells > 0 and
  (.production_source_imported | not) and
  (.production_projection_called | not) and
  (.production_inverse_called | not) and
  .oracle_direct_csr_and_one_body_plans_are_verification_only and
  (.catvm_custody | not) and
  (.distinct_phase_resource_established | not) and
  (.computational_advantage | not) and
  (.small_wall_crossed | not) and
  (.physical_waveform_execution | not) and
  (.physical_bit_replacement | not) and
  (.unbounded_computation_established | not) and
  (.terminal | not)
' "$sealed_oracle" >/dev/null

"$python" - "$sealed" "$sealed_oracle" <<'PY'
import json
import sys

production = json.load(open(sys.argv[1], encoding="utf-8"))
oracle = json.load(open(sys.argv[2], encoding="utf-8"))
for key in ("occupation_histograms", "necklace_cells", "bracelet_cells", "topology_commitment"):
    if production["topology"][key] != oracle["topology"][key]:
        raise SystemExit(f"independent open-momentum topology differs: {key}")
if production["transaction"]["primary_boundary"] != oracle["transaction"]["primary_boundary"]:
    raise SystemExit("independent primary boundary differs")
if production["transaction"]["reuse_boundary"] != oracle["transaction"]["reuse_boundary"]:
    raise SystemExit("independent reuse boundary differs")
if production["transaction"]["primary_signature_order_commitment"] != oracle["factor_verification"]["primary_signature_order_commitment"]:
    raise SystemExit("independent primary state commitment differs")
if production["resource_law"]["accepted_one_body_terms_per_scattering"] != oracle["factor_verification"]["production_reflection_paired_term_law"]:
    raise SystemExit("independent reflection-paired term law differs")
for key in ("missing_inverse_error_field_cells", "wrong_inverse_error_field_cells", "reordered_inverse_error_field_cells"):
    if production["controls"][key] != oracle["controls"][key]:
        raise SystemExit(f"independent control differs: {key}")
PY

"$python" - "$oracle" <<'PY'
import ast
import pathlib
import sys

tree = ast.parse(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
for node in ast.walk(tree):
    if isinstance(node, ast.Import):
        names = [alias.name for alias in node.names]
    elif isinstance(node, ast.ImportFrom):
        names = [node.module or ""]
    else:
        continue
    if any("growing_rotor" in name or "cat_cas" in name for name in names):
        raise SystemExit("open-momentum oracle production dependency isolation failed")
PY

sha256sum "$production" "$oracle" "$sealed" "$sealed_oracle"
printf '%s\n' 'QUALIFIED_GROWING_ROTOR_OPEN_MOMENTUM_FACTOR_CLOSURE_STRICT_SCOPE'
printf 'evidence=tracked-in-place:%s\n' "$here"
