#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

if [[ $# -ne 1 ]]; then
  echo "usage: qualify_continuous_s1_open_phase_relation_laurent.sh MANAGED_BUILD_DIR" >&2
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
production="$here/continuous_s1_open_phase_relation_laurent_closure.py"
oracle="$here/continuous_s1_open_phase_relation_laurent_independent_oracle.py"
sealed="$here/CONTINUOUS_S1_OPEN_PHASE_RELATION_LAURENT_RESULTS.json"
sealed_oracle="$here/CONTINUOUS_S1_OPEN_PHASE_RELATION_LAURENT_INDEPENDENT_ORACLE.json"

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
    raise SystemExit("continuous-S1 production reexecution differs from seal")
' "$sealed"

nice -n 10 ionice -c 3 "$python" "$oracle" | "$python" -c '
import json, sys
sealed = json.load(open(sys.argv[1], encoding="utf-8"))
fresh = json.load(sys.stdin)
if sealed != fresh:
    raise SystemExit("continuous-S1 oracle reexecution differs from seal")
' "$sealed_oracle"

jq -e '
  .classification == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE" and
  .verification_level == "INDEPENDENT_ORACLE_REEXECUTION" and
  .restoration_classification == "EXACT_ALGEBRAIC_RESTORATION" and
  .result == "PASS_EXACT_S1_RELATION_CLOSURE_WITH_GROWING_HARMONIC_RANK" and
  .phase_relation_law.domain == "CONTINUOUS_S1_NO_FINITE_ANGLE_SAMPLING" and
  .phase_relation_law.relation_composition == "HARMONIC_HADAMARD_PUBLIC_ROTATION_MULTIPLIER" and
  .phase_relation_law.relation_intersection == "LAURENT_COEFFICIENT_CONVOLUTION" and
  .phase_relation_law.shared_unresolved_angle_ports == 1 and
  .phase_relation_law.multiple_noncommuting_consumers and
  (.phase_relation_law.intermediate_coefficient_projection | not) and
  (.phase_relation_law.truth_table_or_assignment_expansion | not) and
  (.phase_relation_law.finite_cyclic_group_reduction | not) and
  .analytic_rank_law.degree_at_depth_d == "D_PLUS_1" and
  .analytic_rank_law.coefficient_cells_at_depth_d == "D_PLUS_2" and
  .analytic_rank_law.finite_support_hankel_rank_at_depth_d == "D_PLUS_2" and
  (.analytic_rank_law.fixed_bounded_degree_reduced_rational_chart_for_declared_family | not) and
  .analytic_rank_law.procedural_public_word_rematerialization_remains and
  (.depth_cases | length) == 14 and
  .depth_cases[-2].depth == 64 and
  .depth_cases[-2].family == 0 and
  .depth_cases[-2].reduced_rational_numerator_degree == 65 and
  .depth_cases[-2].finite_support_hankel_rank == 66 and
  .depth_cases[-2].relation_coefficient_cells == 66 and
  .depth_cases[-2].relation_payload_bits == 1121831 and
  .depth_cases[-2].maximum_signed_numerator_bits == 6334 and
  .depth_cases[-2].maximum_denominator_bits == 6332 and
  .depth_cases[-2].scalar_boundary_stream.resident_gaussian_rational_cells == 2 and
  .depth_cases[-2].scalar_boundary_stream.warm_named_gaussian_rational_cell_peak == 4 and
  .transaction.primary_restoration_error_relation_cells == 0 and
  .transaction.reuse_restoration_error_relation_cells == 0 and
  .transaction.fresh_reuse_restoration_error_relation_cells == 0 and
  .transaction.primary_same_backing and
  .transaction.reuse_same_backing and
  .transaction.fresh_reuse_same_backing and
  .transaction.restoration_generation_after_reuse == 2 and
  (.transaction.baseline_reload_used | not) and
  .controls.wrong_owner_rejected and
  .controls.wrong_operation_type_rejected and
  .controls.premature_projection_rejected and
  .controls.missing_inverse_detected and
  .controls.reordered_inverse_rejected and
  .controls.null_carrier_rejected and
  .controls.module_order_noncommuting_mismatch_cells == 2 and
  .controls.semantic_perturbation_changes_boundary and
  .controls.control_port_restored and
  .resource_law.primary_compiled_public_program_operation_records == 128 and
  .resource_law.primary_compiled_public_program_descriptor_slots == 256 and
  .resource_law.retained_inverse_history_entries == 0 and
  .resource_law.additional_retained_factor_plan_entries_beyond_public_program == 0 and
  .resource_law.temporary_full_coefficient_vector_cells == 0 and
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
  (.depth_cases | length) == 14 and
  .depth_cases[-2].depth == 64 and
  .depth_cases[-2].finite_support_hankel_rank == 66 and
  .depth_cases[-2].relation_coefficient_cells == 66 and
  .depth_cases[-2].relation_payload_bits == 1121831 and
  .depth_cases[-2].exact_reverse_restored and
  .module_order_noncommuting_mismatch_cells == 2 and
  (.finite_angle_sampling_used | not) and
  (.distinct_phase_resource_established | not) and
  (.terminal | not)
' "$sealed_oracle" >/dev/null

"$python" - "$sealed" "$sealed_oracle" <<'PY'
import json, sys
production = json.load(open(sys.argv[1], encoding="utf-8"))
oracle = json.load(open(sys.argv[2], encoding="utf-8"))
fields = [
    "depth",
    "family",
    "reduced_rational_numerator_degree",
    "reduced_rational_denominator_degree",
    "finite_support_hankel_rank",
    "relation_coefficient_cells",
    "nonzero_harmonic_coefficients",
    "relation_payload_bits",
    "state_commitment",
    "boundary_commitment",
]
for left, right in zip(
    production["depth_cases"], oracle["depth_cases"], strict=True
):
    for field in fields:
        if left[field] != right[field]:
            raise SystemExit(
                f"continuous-S1 independent parity differs: "
                f"depth={left['depth']} family={left['family']} field={field}"
            )
if (
    production["transaction"]["primary_boundary_commitment"]
    != oracle["primary_depth64_boundary_commitment"]
):
    raise SystemExit("continuous-S1 primary boundary parity differs")
if (
    production["transaction"]["reuse_boundary_commitment"]
    != oracle["reuse_depth37_boundary_commitment"]
):
    raise SystemExit("continuous-S1 reuse boundary parity differs")
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
    raise SystemExit("continuous-S1 oracle imports CAT_CAS production")
PY

sha256sum "$production" "$oracle" "$sealed" "$sealed_oracle"
printf '%s\n' 'QUALIFIED_CONTINUOUS_S1_OPEN_PHASE_RELATION_LAURENT_STRICT_SCOPE'
printf 'evidence=tracked-in-place:%s\n' "$here"
