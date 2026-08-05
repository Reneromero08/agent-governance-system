#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

if [[ $# -ne 1 ]]; then
  echo "usage: qualify_continuous_s1_streamed_jacobi_moment.sh MANAGED_BUILD_DIR" >&2
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
production="$here/continuous_s1_streamed_jacobi_moment_rematerialization.py"
oracle="$here/continuous_s1_streamed_jacobi_moment_independent_oracle.py"
sealed="$here/CONTINUOUS_S1_STREAMED_JACOBI_MOMENT_RESULTS.json"
sealed_oracle="$here/CONTINUOUS_S1_STREAMED_JACOBI_MOMENT_INDEPENDENT_ORACLE.json"

mkdir -p "$build_dir/pycache"
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
    raise SystemExit("streamed Jacobi production differs from seal")
' "$sealed"

nice -n 10 ionice -c 3 "$python" "$oracle" | "$python" -c '
import json, sys
sealed = json.load(open(sys.argv[1], encoding="utf-8"))
fresh = json.load(sys.stdin)
if sealed != fresh:
    raise SystemExit("streamed Jacobi oracle differs from seal")
' "$sealed_oracle"

jq -e '
  .classification == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE" and
  .verification_level == "INDEPENDENT_ORACLE_REEXECUTION" and
  .restoration_classification == "EXACT_ALGEBRAIC_RESTORATION" and
  .result == "PASS_STREAMED_MOMENT_VECTOR_REMOVAL_WITH_SERIES_AND_SOURCE_WORK_OBSTRUCTION" and
  .phase_relation_law.domain == "CONTINUOUS_S1_NO_FINITE_ANGLE_SAMPLING" and
  .phase_relation_law.runtime_center_type == "GAUSSIAN_RATIONAL_UNIT_PHASE" and
  .phase_relation_law.boundary == "EXACT_FIRST_HARMONIC_QJET_ORDER24" and
  .phase_relation_law.moment_law == "REVERSE_PUBLIC_WORD_SCAN_REMATERIALIZE_CONSUME_RESCAN_UNCOMPUTE" and
  .phase_relation_law.retained_moment_vector_cells == 0 and
  .phase_relation_law.moment_scratch_cells == 1 and
  (.phase_relation_law.intermediate_moment_projection | not) and
  .transaction.primary_boundary_commitment == "9f89424531b038f72b07b278e2f61e8f2716d49080d8e768e89e4928f243c5ca" and
  .transaction.reuse_boundary_commitment == "d66f614f650a9390b9ec2a6bf9dd503043d3f0f4b99a7a353a28890bd9004de7" and
  .transaction.reuse_boundary_commitment == .transaction.fresh_reuse_boundary_commitment and
  .transaction.source_backing_identity_preserved_across_reuse and
  .transaction.scratch_backing_identity_preserved_across_reuse and
  .transaction.primary_source_restoration_error_cells == 0 and
  .transaction.primary_scratch_restoration_error_cells == 0 and
  .transaction.reuse_source_restoration_error_cells == 0 and
  .transaction.reuse_scratch_restoration_error_cells == 0 and
  .transaction.restoration_generation_after_reuse == 2 and
  (.transaction.baseline_reload_used | not) and
  .controls.wrong_operation_type_rejected and
  .controls.null_source_rejected and
  .controls.dirty_scratch_rejected and
  .controls.semantic_center_perturbation_changes_boundary and
  .resource_law.runtime_source_center_cells == 24 and
  .resource_law.runtime_source_payload_bits == 763 and
  .resource_law.public_operation_scans == 26 and
  .resource_law.public_operation_records_visited == 884 and
  .resource_law.source_center_visits == 624 and
  .resource_law.center_power_evaluations == 576 and
  .resource_law.retained_moment_vector_cells == 0 and
  .resource_law.moment_scratch_cells == 1 and
  .resource_law.universal_log_cells == 110 and
  .resource_law.weighted_log_cells == 86 and
  .resource_law.weighted_log_payload_bits == 172243 and
  .resource_law.peak_exponential_cells == 325 and
  .resource_law.peak_exponential_payload_bits == 2480635 and
  .resource_law.series_products == 106586 and
  .resource_law.public_program_operation_records == 34 and
  .resource_law.public_program_descriptor_slots == 68 and
  .resource_law.retained_inverse_history_entries == 0 and
  .resource_law.additional_retained_plan_entries == 0 and
  .matched_classical_baselines.streamed_moment == "IDENTICAL_REVERSE_WORD_MOMENT_REMATERIALIZATION_AND_FORMAL_SERIES_RECURRENCE" and
  .matched_classical_baselines.direct_factor == "M210_EXACT_FACTOR_BY_FACTOR_SPARSE_QJET_RECURRENCE" and
  (.matched_classical_baselines.strictly_smaller_or_faster_phase_path_established | not) and
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
  .primary.boundary_commitment == "9f89424531b038f72b07b278e2f61e8f2716d49080d8e768e89e4928f243c5ca" and
  .primary.boundary_commitment == .primary.direct_factor_boundary_commitment and
  .reuse.boundary_commitment == "d66f614f650a9390b9ec2a6bf9dd503043d3f0f4b99a7a353a28890bd9004de7" and
  .reuse.boundary_commitment == .reuse.direct_factor_boundary_commitment and
  .primary.source_unchanged and
  .reuse.source_unchanged and
  .primary.source_payload_bits == 763 and
  .primary.metrics.scratch_restored and
  .primary.metrics.retained_moment_vector_cells == 0 and
  .primary.metrics.source_center_visits == 624 and
  .primary.metrics.weighted_log_payload_bits == 172243 and
  .primary.metrics.peak_exponential_payload_bits == 2480635 and
  .primary_expected_resource_tuple_reproduced and
  .streamed_and_direct_boundaries_match and
  (.finite_angle_sampling_used | not) and
  (.full_infinite_theta_scalar_evaluated | not) and
  (.distinct_phase_resource_established | not) and
  (.terminal | not)
' "$sealed_oracle" >/dev/null

"$python" - "$sealed" "$sealed_oracle" <<'PY'
import json, sys
production = json.load(open(sys.argv[1], encoding="utf-8"))
oracle = json.load(open(sys.argv[2], encoding="utf-8"))
for production_field, oracle_field in (
    ("primary_boundary_commitment", "primary"),
    ("reuse_boundary_commitment", "reuse"),
):
    if (
        production["transaction"][production_field]
        != oracle[oracle_field]["boundary_commitment"]
    ):
        raise SystemExit(f"streamed Jacobi {oracle_field} boundary parity differs")
for field in (
    "public_operation_scans",
    "public_operation_records_visited",
    "source_center_visits",
    "center_power_evaluations",
    "scratch_writes",
    "scratch_inverse_writes",
    "retained_moment_vector_cells",
    "moment_scratch_cells",
    "universal_log_cells",
    "weighted_log_cells",
    "weighted_log_payload_bits",
    "weighted_log_key_bits",
    "peak_exponential_cells",
    "peak_exponential_payload_bits",
    "peak_exponential_key_bits",
    "series_products",
):
    if production["resource_law"][field] != oracle["primary"]["metrics"][field]:
        raise SystemExit(f"streamed Jacobi primary resource parity differs: {field}")
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
if any("continuous_s1" in name or "cat_cas" in name for name in imports):
    raise SystemExit("streamed Jacobi oracle imports CAT_CAS production")
PY

sha256sum "$production" "$oracle" "$sealed" "$sealed_oracle"
printf '%s\n' 'QUALIFIED_CONTINUOUS_S1_STREAMED_JACOBI_MOMENT_STRICT_SCOPE'
printf 'evidence=tracked-in-place:%s\n' "$here"
