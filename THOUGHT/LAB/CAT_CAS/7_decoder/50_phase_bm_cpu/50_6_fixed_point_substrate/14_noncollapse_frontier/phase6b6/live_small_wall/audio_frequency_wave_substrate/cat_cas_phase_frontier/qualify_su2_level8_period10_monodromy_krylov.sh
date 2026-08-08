#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

if [[ $# -ne 1 ]]; then
  echo "usage: qualify_su2_level8_period10_monodromy_krylov.sh MANAGED_BUILD_DIR" >&2
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
production="$here/su2_level8_period10_monodromy_krylov.py"
core="$here/su2_level8_period10_monodromy_krylov_core.cpp"
reference="$here/su2_level8_period10_monodromy_krylov_separate_reference.py"
reference_core="$here/su2_level8_period10_monodromy_krylov_separate_reference.cpp"
substrate="$here/su2_level8_fusion_path_braid_phase_relation.py"
sealed="$here/SU2_LEVEL8_PERIOD10_MONODROMY_KRYLOV_RESULTS.json"
sealed_reference="$here/SU2_LEVEL8_PERIOD10_MONODROMY_KRYLOV_SEPARATE_REFERENCE.json"

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

g++ -std=c++20 -O3 -Wall -Wextra -Werror -pedantic \
  "$core" -o "$build_dir/m217-core"
g++ -std=c++20 -O3 -Wall -Wextra -Werror -pedantic \
  "$reference_core" -o "$build_dir/m217-reference"

nice -n 10 ionice -c 3 "$python" "$production" "$build_dir/m217-core" \
  | "$python" -c '
import json, sys
sealed = json.load(open(sys.argv[1], encoding="utf-8"))
fresh = json.load(sys.stdin)
if sealed != fresh:
    raise SystemExit("M217 production differs from seal")
' "$sealed"

nice -n 10 ionice -c 3 "$python" "$reference" "$build_dir/m217-reference" \
  | "$python" -c '
import json, sys
sealed = json.load(open(sys.argv[1], encoding="utf-8"))
fresh = json.load(sys.stdin)
if sealed != fresh:
    raise SystemExit("M217 separate reference differs from seal")
' "$sealed_reference"

jq -e '
  .classification == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE" and
  .verification_level == "SEPARATE_REFERENCE_PARITY" and
  .restoration_classification == "EXACT_ALGEBRAIC_RESTORATION" and
  .result == "PASS_EXACT_SU2_LEVEL8_PERIOD10_MONODROMY_FULL_BOUNDARY_KRYLOV_DEGREE_OBSTRUCTION" and
  .exact_degree_law.strands == [4,6,8,10,12,14,16] and
  .exact_degree_law.fusion_path_dimensions == [2,5,14,42,132,429,1430] and
  .exact_degree_law.both_family_exact_scalar_minimal_degrees == [2,5,14,42,132,429,1430] and
  .exact_degree_law.all_cases_full_at_both_primes and
  .exact_degree_law.uniform_fixed_degree_boundary_recurrence_rejected_for_declared_family and
  (.exact_degree_law.exact_recurrence_coefficients_lifted | not) and
  (.exact_degree_law.arbitrary_braid_program_lower_bound | not) and
  .core_diagnostic.period_law_verified and
  .core_diagnostic.every_case_cross_prime_degree_agreement and
  .core_diagnostic.every_case_full_scalar_degree and
  .core_diagnostic.every_semantic_perturbation_changes_prefix and
  ([.core_diagnostic.cases[].prime_results[].training_violations] | all(. == 0)) and
  ([.core_diagnostic.cases[].prime_results[].holdout_violations] | all(. == 0)) and
  ([.core_diagnostic.cases[].prime_results[].undersampled_holdout_violations] | all(. > 0)) and
  ([.core_diagnostic.cases[].prime_results[].semantic_perturbation_changes_prefix] | all) and
  .exact_boundary_modular_parity.agreement and
  .transaction.primary.restoration_error_field_cells == 0 and
  .transaction.primary.same_coefficient_backing and
  .transaction.primary.canonical_post_restoration_state_exact and
  .transaction.reuse.restoration_error_field_cells == 0 and
  .transaction.reuse.same_coefficient_backing and
  .transaction.reuse.canonical_post_restoration_state_exact and
  .transaction.fresh_restored_reuse_boundary_agreement and
  .transaction.fresh_restored_reuse_state_agreement and
  .transaction.restoration_generation_after_reuse == 2 and
  (.transaction.baseline_reload_used | not) and
  .controls.wrong_owner_rejected and
  .controls.wrong_operation_type_rejected and
  .controls.wrong_public_program_inverse_rejected and
  .controls.premature_projection_rejected and
  .controls.missing_inverse_detected and
  .controls.reordered_inverse_rejected and
  .controls.null_carrier_rejected and
  (.controls.snapshot_command_available | not) and
  .controls.compiler_uses_only_public_topology_and_word and
  (.controls.compiler_reads_exact_final_boundary | not) and
  (.controls.intermediate_fusion_path_state_projected | not) and
  .resource_law.primary_exact_carrier_field_cells == 1430 and
  .resource_law.primary_modular_state_field_cells == 1430 and
  .resource_law.primary_scalar_sequence_field_cells == 2924 and
  .resource_law.primary_public_recurrence_coefficient_slots == 1431 and
  .resource_law.accepted_transaction_retained_inverse_history == 0 and
  .resource_law.controller_backend_traffic_bytes == 0 and
  .resource_law.snapshot_traffic_bytes == 0 and
  .resource_law.baseline_reload_bytes == 0 and
  .matched_classical_baselines.strongest_compact == "IDENTICAL_PUBLIC_PERIOD10_MONODROMY_SCALAR_KRYLOV_AND_FULL1430_CELL_FUSION_PATH_TRANSFER" and
  .matched_classical_baselines.same_full_degree_and_recurrence_coefficient_law and
  (.matched_classical_baselines.phase_specific_recurrence_reduction | not) and
  (.matched_classical_baselines.computational_advantage | not) and
  (.claim_limits.compact_period10_boundary_recurrence_established | not) and
  (.claim_limits.catvm_custody | not) and
  (.claim_limits.distinct_phase_resource_established | not) and
  (.claim_limits.computational_advantage | not) and
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
  (.reference_imports_m217_production | not) and
  .distinct_split_primes == [641,881] and
  .all_cases_full_at_both_distinct_split_primes and
  ([.cases[].fusion_path_cells] == [2,5,14,42,132,429,1430,2,5,14,42,132,429,1430]) and
  ([.cases[].prime_results[].scalar_recurrence_degree] == [2,2,5,5,14,14,42,42,132,132,429,429,1430,1430,2,2,5,5,14,14,42,42,132,132,429,429,1430,1430]) and
  ([.cases[].prime_results[].holdout_violations] | all(. == 0)) and
  .exact_primary_boundary_parity.agreement and
  (.claim_limits.arbitrary_braid_program_lower_bound | not) and
  (.claim_limits.distinct_phase_resource_established | not) and
  (.claim_limits.computational_advantage | not) and
  (.claim_limits.small_wall_crossed | not) and
  (.claim_limits.physical_waveform_execution | not) and
  (.claim_limits.unbounded_computation_established | not) and
  (.terminal | not)
' "$sealed_reference" >/dev/null

if grep -Eq '^[[:space:]]*#include[[:space:]]*[<"]su2_level8_period10_monodromy_krylov(_core)?' "$reference_core"; then
  echo "separate reference core includes M217 production" >&2
  exit 1
fi
if grep -Eq '(^|[[:space:]])(from|import)[[:space:]]+su2_level8_period10_monodromy_krylov([[:space:].]|$)' "$reference"; then
  echo "separate reference wrapper imports M217 production" >&2
  exit 1
fi

"$python" - "$sealed" "$sealed_reference" "$core" "$reference_core" "$substrate" <<'PY'
import hashlib, json, sys
production = json.load(open(sys.argv[1], encoding="utf-8"))
reference = json.load(open(sys.argv[2], encoding="utf-8"))
core_hash = hashlib.sha256(open(sys.argv[3], "rb").read()).hexdigest()
reference_hash = hashlib.sha256(open(sys.argv[4], "rb").read()).hexdigest()
substrate_hash = hashlib.sha256(open(sys.argv[5], "rb").read()).hexdigest()
if production["source_dependencies"] != {
    "m214_production_sha256": substrate_hash,
    "m217_core_sha256": core_hash,
}:
    raise SystemExit("M217 production dependency hashes changed")
if reference["source_dependencies"] != {
    "m214_production_sha256": substrate_hash,
    "separate_reference_core_sha256": reference_hash,
}:
    raise SystemExit("M217 reference dependency hashes changed")
expected = [2, 5, 14, 42, 132, 429, 1430]
for family in (0, 1):
    prod = [
        case for case in production["core_diagnostic"]["cases"]
        if case["family"] == family
    ]
    ref = [case for case in reference["cases"] if case["family"] == family]
    if [case["fusion_path_cells"] for case in prod] != expected:
        raise SystemExit("M217 production family dimensions changed")
    if [case["fusion_path_cells"] for case in ref] != expected:
        raise SystemExit("M217 reference family dimensions changed")
    for cases in (prod, ref):
        for case in cases:
            if any(
                item["scalar_recurrence_degree"] != case["fusion_path_cells"]
                for item in case["prime_results"]
            ):
                raise SystemExit("M217 production/reference full-degree parity failed")
if (
    production["exact_boundary_modular_parity"]["exact_boundary_commitment"]
    != reference["exact_primary_boundary_parity"]["boundary_commitment"]
):
    raise SystemExit("M217 exact primary boundary commitments differ")
PY

sha256sum \
  "$production" "$core" "$reference" "$reference_core" "$substrate" \
  "$sealed" "$sealed_reference"
echo "QUALIFIED_SU2_LEVEL8_PERIOD10_MONODROMY_FULL_BOUNDARY_DEGREE_OBSTRUCTION_STRICT_SCOPE"
echo "evidence=tracked-in-place:$here"
