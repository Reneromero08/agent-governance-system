#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

if [[ $# -ne 1 ]]; then
  echo "usage: qualify_continuous_s1_generic_center_theta_moment.sh MANAGED_BUILD_DIR" >&2
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
production="$here/continuous_s1_generic_center_theta_moment_qjet.py"
oracle="$here/continuous_s1_generic_center_theta_moment_independent_oracle.py"
sealed="$here/CONTINUOUS_S1_GENERIC_CENTER_THETA_MOMENT_RESULTS.json"
sealed_oracle="$here/CONTINUOUS_S1_GENERIC_CENTER_THETA_MOMENT_INDEPENDENT_ORACLE.json"

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
    raise SystemExit("generic-center theta production differs from seal")
' "$sealed"

nice -n 10 ionice -c 3 "$python" "$oracle" | "$python" -c '
import json, sys
sealed = json.load(open(sys.argv[1], encoding="utf-8"))
fresh = json.load(sys.stdin)
if sealed != fresh:
    raise SystemExit("generic-center theta oracle differs from seal")
' "$sealed_oracle"

jq -e '
  .classification == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE" and
  .verification_level == "INDEPENDENT_ORACLE_REEXECUTION" and
  .restoration_classification == "EXACT_ALGEBRAIC_RESTORATION" and
  .result == "PASS_EXACT_GENERIC_CENTER_MOMENT_QJET_WITH_PRECISION_HIERARCHY_AND_SOURCE_RETENTION_OBSTRUCTION" and
  .phase_relation_law.domain == "CONTINUOUS_S1_NO_FINITE_ANGLE_SAMPLING" and
  .phase_relation_law.runtime_center_type == "GAUSSIAN_RATIONAL_UNIT_PHASE" and
  (.phase_relation_law.finite_phase_center_alphabet | not) and
  .phase_relation_law.native_moment_law == "LOG_THETA_WEIGHTED_BY_RUNTIME_POWER_SUMS_THEN_FORMAL_EXP" and
  .phase_relation_law.multiple_noncommuting_consumers and
  (.phase_relation_law.intermediate_moment_projection | not) and
  (.phase_relation_law.truth_table_or_assignment_expansion | not) and
  .moment_hierarchy.first_harmonic_qjet_order_j_requires_moments_through == "FLOOR_JPLUS1_OVER_2" and
  (.moment_hierarchy.fixed_moment_count_for_unbounded_precision | not) and
  (.moment_hierarchy.full_infinite_theta_scalar_evaluated | not) and
  (.precision_cases | length) == 14 and
  .precision_cases[0].q_jet_order == 2 and
  .precision_cases[0].resident_moment_cells == 2 and
  .precision_cases[-2].q_jet_order == 24 and
  .precision_cases[-2].family == 0 and
  .precision_cases[-2].resident_moment_cells == 13 and
  .precision_cases[-2].resident_moment_payload_bits == 47209 and
  .precision_cases[-2].runtime_source_center_cells == 24 and
  .precision_cases[-2].runtime_source_payload_bits == 763 and
  .precision_cases[-2].projection_peak_series_cells == 319 and
  .precision_cases[-2].projection_series_products == 106586 and
  .transaction.source_backing_identity_preserved_across_program_reuse and
  .transaction.moment_backing_identity_preserved_across_program_reuse and
  .transaction.primary_restoration_error_moment_cells == 0 and
  .transaction.reuse_restoration_error_moment_cells == 0 and
  .transaction.restoration_generation_after_reuse == 2 and
  .transaction.reuse_boundary_commitment == .transaction.fresh_reuse_boundary_commitment and
  (.transaction.baseline_reload_used | not) and
  .controls.wrong_owner_rejected and
  .controls.wrong_operation_type_rejected and
  .controls.premature_projection_rejected and
  .controls.missing_inverse_detected and
  .controls.reordered_inverse_rejected and
  .controls.null_carrier_rejected and
  .controls.module_order_moments_differ and
  .controls.module_order_boundary_changes and
  .controls.control_port_restored and
  .resource_law.primary_runtime_source_center_cells == 24 and
  .resource_law.primary_resident_moment_cells == 13 and
  .resource_law.primary_resident_moment_payload_bits == 47209 and
  .resource_law.primary_compiled_public_program_operation_records == 34 and
  .resource_law.primary_compiled_public_program_descriptor_slots == 68 and
  .resource_law.retained_inverse_history_entries == 0 and
  .resource_law.additional_retained_plan_entries_beyond_public_program == 0 and
  (.catvm_custody | not) and
  (.distinct_phase_resource_established | not) and
  (.computational_advantage | not) and
  (.small_wall_crossed | not) and
  (.physical_waveform_execution | not) and
  (.physical_bit_replacement | not) and
  (.catalytic_inference_established | not) and
  (.unbounded_computation_established | not) and
  (.terminal | not)
' "$sealed" >/dev/null

jq -e '
  .classification == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE" and
  .verification_level == "INDEPENDENT_ORACLE_REEXECUTION" and
  .restoration_classification == "EXACT_ALGEBRAIC_RESTORATION" and
  (.oracle_imports_cat_cas_modules | not) and
  (.precision_cases | length) == 14 and
  .precision_cases[-2].q_jet_order == 24 and
  .precision_cases[-2].moment_cells == 13 and
  .precision_cases[-2].source_unchanged and
  .module_order_center_lists_differ and
  .module_order_boundary_changes and
  (.finite_angle_sampling_used | not) and
  (.full_infinite_theta_scalar_evaluated | not) and
  (.distinct_phase_resource_established | not) and
  (.terminal | not)
' "$sealed_oracle" >/dev/null

"$python" - "$sealed" "$sealed_oracle" <<'PY'
import json, sys
production = json.load(open(sys.argv[1], encoding="utf-8"))
oracle = json.load(open(sys.argv[2], encoding="utf-8"))
for left, right in zip(
    production["precision_cases"], oracle["precision_cases"], strict=True
):
    for left_field, right_field in (
        ("q_jet_order", "q_jet_order"),
        ("family", "family"),
        ("resident_moment_cells", "moment_cells"),
        ("resident_moment_commitment", "moment_commitment"),
        ("boundary_commitment", "boundary_commitment"),
    ):
        if left[left_field] != right[right_field]:
            raise SystemExit(
                f"generic-center parity differs: order={left['q_jet_order']} "
                f"family={left['family']} field={left_field}"
            )
if (
    production["transaction"]["primary_boundary_commitment"]
    != oracle["primary_boundary_commitment"]
):
    raise SystemExit("generic-center primary boundary parity differs")
if (
    production["transaction"]["reuse_boundary_commitment"]
    != oracle["reuse_boundary_commitment"]
):
    raise SystemExit("generic-center reuse boundary parity differs")
PY

"$python" - "$oracle" <<'PY'
import ast, pathlib, sys
tree = ast.parse(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
imports = set()
for node in ast.walk(tree):
    if isinstance(node, ast.Import):
        imports.update(alias.name for alias in node.names)
    elif isinstance(node, ast.ImportFrom):
        imports.add(node.module or "")
if any("continuous_s1_generic" in name or "cat_cas" in name for name in imports):
    raise SystemExit("generic-center oracle imports CAT_CAS production")
PY

sha256sum "$production" "$oracle" "$sealed" "$sealed_oracle"
printf '%s\n' 'QUALIFIED_CONTINUOUS_S1_GENERIC_CENTER_THETA_MOMENT_STRICT_SCOPE'
printf 'evidence=tracked-in-place:%s\n' "$here"
