#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

if [[ $# -ne 1 ]]; then
  echo "usage: qualify_su2_level8_fusion_path_mps_rank.sh MANAGED_BUILD_DIR" >&2
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
production="$here/su2_level8_fusion_path_mps_rank_no_go.py"
reference="$here/su2_level8_fusion_path_mps_rank_separate_reference.py"
substrate="$here/su2_level8_fusion_path_braid_phase_relation.py"
sealed="$here/SU2_LEVEL8_FUSION_PATH_MPS_RANK_NO_GO_RESULTS.json"
sealed_reference="$here/SU2_LEVEL8_FUSION_PATH_MPS_RANK_SEPARATE_REFERENCE.json"

mkdir -p "$build_dir/pycache"
filesystem_type=$(findmnt -n -o FSTYPE -T "$build_dir")
case "$filesystem_type" in
  tmpfs|ramfs)
    echo "managed build directory resolves to a RAM-backed filesystem" >&2
    exit 2
    ;;
esac
export TMPDIR="$build_dir"
export TMP="$build_dir"
export TEMP="$build_dir"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="$build_dir/pycache"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

nice -n 10 ionice -c 3 "$python" "$production" | "$python" -c '
import json, sys
sealed = json.load(open(sys.argv[1], encoding="utf-8"))
fresh = json.load(sys.stdin)
if sealed != fresh:
    raise SystemExit("SU2 level-8 MPS rank production differs from seal")
' "$sealed"

nice -n 10 ionice -c 3 "$python" "$reference" | "$python" -c '
import json, sys
sealed = json.load(open(sys.argv[1], encoding="utf-8"))
fresh = json.load(sys.stdin)
if sealed != fresh:
    raise SystemExit("SU2 level-8 MPS rank reference differs from seal")
' "$sealed_reference"

jq -e '
  .classification == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE" and
  .verification_level == "SEPARATE_REFERENCE_PARITY" and
  .restoration_classification == "EXACT_ALGEBRAIC_RESTORATION" and
  .result == "PASS_EXACT_SU2_LEVEL8_FUSION_PATH_MAXIMAL_SECTOR_SCHMIDT_RANK_NO_FIXED_BOND_MPS" and
  .exact_rank_law.strands == [4,6,8,10,12,14,16] and
  .exact_rank_law.both_family_maximum_bond_ranks == [2,3,6,10,20,35,70] and
  .exact_rank_law.every_declared_sector_reaches_upper_bound_at_both_split_primes and
  .exact_rank_law.therefore_exact_qzeta40_rank_is_maximal and
  .exact_rank_law.uniform_fixed_bond_exact_mps_rejected_for_declared_growing_family and
  ([.cases[].cross_prime_rank_agreement] | all) and
  ([.cases[].all_sector_ranks_maximal] | all) and
  .transaction.primary_restoration_error_field_cells == 0 and
  .transaction.primary_same_coefficient_backing and
  .transaction.primary_canonical_post_restoration_state_exact and
  .transaction.reuse_restoration_error_field_cells == 0 and
  .transaction.reuse_same_coefficient_backing and
  .transaction.reuse_canonical_post_restoration_state_exact and
  .transaction.fresh_restored_reuse_boundary_agreement and
  .transaction.fresh_restored_reuse_state_agreement and
  .transaction.restoration_generation_after_reuse == 2 and
  (.transaction.baseline_reload_used | not) and
  .controls.invalid_composite_modulus_rejected and
  .controls.invalid_nonsplit_prime_rejected and
  .controls.both_split_roots_have_exact_order40 and
  .resource_law.primary_direct_fusion_path_field_cells == 1430 and
  .resource_law.primary_direct_fusion_path_payload_bits == 256269 and
  .resource_law.primary_maximum_exact_bond_rank == 70 and
  .resource_law.primary_canonical_dense_sector_mps_field_cells == 4110 and
  .resource_law.primary_peak_modular_matrix_cells == 1430 and
  .resource_law.primary_verification_path_records == 1430 and
  .resource_law.primary_verification_path_label_cells == 24310 and
  .resource_law.rank_certification_is_verification_instrument_not_accepted_runtime_output and
  .resource_law.accepted_transaction_retained_inverse_history == 0 and
  .matched_classical_baselines.strongest_compact == "IDENTICAL_EXACT_SECTOR_RANK_ANYON_MPS_AND_SMALLER_DIRECT_FUSION_PATH_RECURRENCE" and
  (.matched_classical_baselines.phase_specific_rank_reduction | not) and
  (.matched_classical_baselines.computational_advantage | not) and
  (.claim_limits.native_compact_mps_transaction_established | not) and
  (.claim_limits.uniform_fixed_bond_exact_mps | not) and
  (.claim_limits.rank_certificates_are_runtime_outputs | not) and
  (.claim_limits.catvm_custody | not) and
  (.claim_limits.distinct_phase_resource_established | not) and
  (.claim_limits.small_wall_crossed | not) and
  (.claim_limits.physical_waveform_execution | not) and
  (.claim_limits.physical_bit_replacement | not) and
  (.claim_limits.catalytic_inference_established | not) and
  (.claim_limits.unbounded_computation_established | not) and
  (.terminal | not)
' "$sealed" >/dev/null

jq -e '
  .classification == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE" and
  .verification_level == "SEPARATE_REFERENCE_PARITY" and
  .restoration_classification == "EXACT_ALGEBRAIC_RESTORATION" and
  (.reference_imports_m215_production | not) and
  .reference_algorithm == "INCREMENTAL_MODULAR_COLUMN_BASIS_AT_DISTINCT_SPLIT_PRIMES641_881" and
  .both_reference_primes_exact_order40 and
  .all_case_maximum_bond_ranks == [2,3,6,10,20,35,70] and
  .primary_canonical_dense_sector_mps_field_cells == 4110 and
  .primary_direct_fusion_path_field_cells == 1430 and
  .uniform_fixed_bond_exact_mps_rejected and
  (.distinct_phase_resource_established | not) and
  (.terminal | not)
' "$sealed_reference" >/dev/null

"$python" - "$sealed" "$sealed_reference" "$substrate" <<'PY'
import hashlib, json, sys
production = json.load(open(sys.argv[1], encoding="utf-8"))
reference = json.load(open(sys.argv[2], encoding="utf-8"))
substrate_hash = hashlib.sha256(open(sys.argv[3], "rb").read()).hexdigest()
if production["source_dependencies"]["m214_production_sha256"] != substrate_hash:
    raise SystemExit("M214 substrate dependency hash mismatch")
lookup = {(case["strands"], case["family"]): case for case in production["cases"]}
for case in reference["cases"]:
    candidate = lookup[(case["strands"], case["family"])]
    if candidate["prime_certificates"]["241"]["cut_total_ranks"] != case["cut_ranks"]:
        raise SystemExit("separate-reference cut-rank parity failed")
    if candidate["maximum_exact_sector_schmidt_bond_rank"] != case["maximum_bond_rank"]:
        raise SystemExit("separate-reference maximum-rank parity failed")
    if candidate["canonical_dense_sector_mps_field_cells"] != case["canonical_dense_sector_mps_field_cells"]:
        raise SystemExit("separate-reference MPS-allocation parity failed")
PY

sha256sum "$production" "$reference" "$substrate" "$sealed" "$sealed_reference"
echo "QUALIFIED_SU2_LEVEL8_FUSION_PATH_MPS_RANK_STRICT_SCOPE"
echo "evidence=tracked-in-place:$here"
