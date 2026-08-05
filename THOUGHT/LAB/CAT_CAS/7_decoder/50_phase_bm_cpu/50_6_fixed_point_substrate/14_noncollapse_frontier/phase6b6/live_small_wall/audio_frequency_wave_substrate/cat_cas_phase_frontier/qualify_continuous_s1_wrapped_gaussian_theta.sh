#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

if [[ $# -ne 1 ]]; then
  echo "usage: qualify_continuous_s1_wrapped_gaussian_theta.sh MANAGED_BUILD_DIR" >&2
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
production="$here/continuous_s1_wrapped_gaussian_theta_phase_chart.py"
oracle="$here/continuous_s1_wrapped_gaussian_theta_independent_oracle.py"
sealed="$here/CONTINUOUS_S1_WRAPPED_GAUSSIAN_THETA_RESULTS.json"
sealed_oracle="$here/CONTINUOUS_S1_WRAPPED_GAUSSIAN_THETA_INDEPENDENT_ORACLE.json"

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
    raise SystemExit("continuous-S1 theta production differs from seal")
' "$sealed"

nice -n 10 ionice -c 3 "$python" "$oracle" | "$python" -c '
import json, sys
sealed = json.load(open(sys.argv[1], encoding="utf-8"))
fresh = json.load(sys.stdin)
if sealed != fresh:
    raise SystemExit("continuous-S1 theta oracle differs from seal")
' "$sealed_oracle"

jq -e '
  .classification == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE" and
  .verification_level == "INDEPENDENT_ORACLE_REEXECUTION" and
  .restoration_classification == "EXACT_ALGEBRAIC_RESTORATION" and
  .result == "PASS_EXACT_THETA_DESCRIPTOR_WITH_GROWING_DISCRIMINANT_FIBER_OBSTRUCTION" and
  .phase_relation_law.domain == "CONTINUOUS_S1_NO_FINITE_ANGLE_SAMPLING" and
  .phase_relation_law.relation_composition == "HARMONIC_HADAMARD_PARAMETER_ADDITION" and
  .phase_relation_law.relation_intersection == "HARMONIC_CONVOLUTION_A_LATTICE_THETA_CHART" and
  .phase_relation_law.shared_unresolved_angle_ports == 1 and
  .phase_relation_law.multiple_noncommuting_consumers and
  (.phase_relation_law.intermediate_harmonic_projection | not) and
  (.phase_relation_law.truth_table_or_assignment_expansion | not) and
  (.phase_relation_law.finite_cyclic_group_reduction | not) and
  .composition_certificate.all_exact and
  .composition_certificate.diffusion_parameter_law == "A_PLUS_B" and
  .composition_certificate.phase_parameter_law == "P_PLUS_R_MOD4" and
  .theta_fiber_law.lattice_rank == "D_MINUS_1" and
  .theta_fiber_law.discriminant_fibers == "D" and
  (.theta_fiber_law.fixed_parity_closure_across_depth | not) and
  .theta_fiber_law.four_count_descriptor_names_full_declared_analytic_relation and
  (.theta_fiber_law.full_infinite_theta_scalar_evaluated | not) and
  (.factor_cases | length) == 14 and
  .factor_cases[-2].total_factors == 64 and
  .factor_cases[-2].lattice_rank == 63 and
  .factor_cases[-2].reduced_gram_determinant == 64 and
  .factor_cases[-2].discriminant_fibers == 64 and
  .factor_cases[-2].resident_factor_count_cells == 4 and
  .factor_cases[-2].projection_peak_sparse_cells == 325 and
  .factor_cases[-2].projection_sparse_transitions == 100809 and
  .transaction.primary_same_backing and
  .transaction.reuse_same_backing and
  .transaction.carrier_backing_identity_preserved_across_both_programs and
  .transaction.primary_restoration_error_count_cells == 0 and
  .transaction.reuse_restoration_error_count_cells == 0 and
  .transaction.restoration_generation_after_reuse == 2 and
  .transaction.reuse_boundary_commitment == .transaction.fresh_reuse_boundary_commitment and
  (.transaction.baseline_reload_used | not) and
  .controls.wrong_owner_rejected and
  .controls.wrong_operation_type_rejected and
  .controls.premature_projection_rejected and
  .controls.missing_inverse_detected and
  .controls.reordered_inverse_rejected and
  .controls.null_carrier_rejected and
  .controls.module_order_counts_differ and
  .controls.module_order_boundary_changes and
  .controls.fixed_parity_overmerge_collisions_at_factor64 == 62 and
  .controls.control_port_restored and
  .resource_law.primary_resident_relation_descriptor_integer_cells == 4 and
  .resource_law.primary_compiled_public_program_operation_records == 126 and
  .resource_law.primary_compiled_public_program_descriptor_slots == 252 and
  .resource_law.primary_projection_q_jet_order == 24 and
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
  .composition_parameter_addition_exact and
  (.factor_cases | length) == 14 and
  .factor_cases[-2].total_factors == 64 and
  .factor_cases[-2].lattice_rank == 63 and
  .factor_cases[-2].reduced_gram_determinant == 64 and
  .factor_cases[-2].discriminant_fibers == 64 and
  .factor_cases[-2].exact_reverse_restored and
  .module_order_counts_differ and
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
fields = [
    "total_factors",
    "family",
    "boundary_commitment",
    "lattice_rank",
    "reduced_gram_determinant",
    "discriminant_fibers",
]
for left, right in zip(
    production["factor_cases"], oracle["factor_cases"], strict=True
):
    for field in fields:
        if left[field] != right[field]:
            raise SystemExit(
                f"theta independent parity differs: "
                f"factor={left['total_factors']} family={left['family']} "
                f"field={field}"
            )
if (
    production["transaction"]["primary_boundary_commitment"]
    != oracle["primary_boundary_commitment"]
):
    raise SystemExit("theta primary boundary parity differs")
if (
    production["transaction"]["reuse_boundary_commitment"]
    != oracle["reuse_boundary_commitment"]
):
    raise SystemExit("theta reuse boundary parity differs")
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
if any("continuous_s1_wrapped" in name or "cat_cas" in name for name in imports):
    raise SystemExit("theta oracle imports CAT_CAS production")
PY

sha256sum "$production" "$oracle" "$sealed" "$sealed_oracle"
printf '%s\n' 'QUALIFIED_CONTINUOUS_S1_WRAPPED_GAUSSIAN_THETA_STRICT_SCOPE'
printf 'evidence=tracked-in-place:%s\n' "$here"
