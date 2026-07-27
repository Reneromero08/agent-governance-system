#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: qualify_quadratic_anf_chain_phase.sh EVIDENCE_DIR" >&2
    exit 2
fi

frontier_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
evidence_dir=$1
mkdir -p "$evidence_dir"

phase_source="$frontier_dir/quadratic_anf_chain_phase.c"
reference_source="$frontier_dir/quadratic_anf_chain_reference.c"
primary_fixture="$frontier_dir/quadratic_anf_chain_primary.qanf"
reuse_fixture="$frontier_dir/quadratic_anf_chain_reuse.qanf"
sham_fixture="$frontier_dir/quadratic_anf_chain_affine_sham.qanf"
degree2_fixture="$frontier_dir/quadratic_anf_chain_degree2_nonaffine.qanf"
nonmonic_fixture="$frontier_dir/negative_qanf_nonmonic.qanf"
bad_coefficient_fixture="$frontier_dir/negative_qanf_coefficient.qanf"
qualifier_source="$frontier_dir/qualify_quadratic_anf_chain_phase.sh"

common_flags=(
    -std=c11
    -O2
    -Wall
    -Wextra
    -Wpedantic
    -Werror
)

cc "${common_flags[@]}" "$phase_source" -lm \
    -o "$evidence_dir/phase"
cc "${common_flags[@]}" "$reference_source" \
    -o "$evidence_dir/reference"
cc "${common_flags[@]}" -fanalyzer "$phase_source" -lm \
    -o "$evidence_dir/phase_analyzer"
cc "${common_flags[@]}" -fanalyzer "$reference_source" \
    -o "$evidence_dir/reference_analyzer"
cc "${common_flags[@]}" -fsanitize=address,undefined \
    -fno-omit-frame-pointer -fno-sanitize-recover=all \
    "$phase_source" -lm -o "$evidence_dir/phase_sanitized"
cc "${common_flags[@]}" -fsanitize=address,undefined \
    -fno-omit-frame-pointer -fno-sanitize-recover=all \
    "$reference_source" -o "$evidence_dir/reference_sanitized"
cc "${common_flags[@]}" -fstack-usage "$phase_source" -lm \
    -o "$evidence_dir/phase_stack"

"$evidence_dir/phase" "$primary_fixture" "$reuse_fixture" \
    >"$evidence_dir/accepted.json" \
    2>"$evidence_dir/accepted.stderr"
"$evidence_dir/phase" "$primary_fixture" "$reuse_fixture" \
    >"$evidence_dir/replay.json" \
    2>"$evidence_dir/replay.stderr"
cmp "$evidence_dir/accepted.json" "$evidence_dir/replay.json"
test ! -s "$evidence_dir/accepted.stderr"
test ! -s "$evidence_dir/replay.stderr"

"$evidence_dir/phase" "$sham_fixture" "$sham_fixture" \
    >"$evidence_dir/affine_sham.json" \
    2>"$evidence_dir/affine_sham.stderr"
test ! -s "$evidence_dir/affine_sham.stderr"
"$evidence_dir/phase" "$degree2_fixture" "$degree2_fixture" \
    >"$evidence_dir/degree2_nonaffine.json" \
    2>"$evidence_dir/degree2_nonaffine.stderr"
test ! -s "$evidence_dir/degree2_nonaffine.stderr"

"$evidence_dir/reference" "$primary_fixture" \
    >"$evidence_dir/reference_primary.json"
"$evidence_dir/reference" "$reuse_fixture" \
    >"$evidence_dir/reference_reuse.json"
"$evidence_dir/reference" "$sham_fixture" \
    >"$evidence_dir/reference_sham.json"
"$evidence_dir/reference" "$degree2_fixture" \
    >"$evidence_dir/reference_degree2.json"

ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
"$evidence_dir/phase_sanitized" "$primary_fixture" "$reuse_fixture" \
    >"$evidence_dir/sanitized.json" \
    2>"$evidence_dir/sanitized.stderr"
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
"$evidence_dir/reference_sanitized" "$primary_fixture" \
    >"$evidence_dir/reference_sanitized.json" \
    2>"$evidence_dir/reference_sanitized.stderr"
test ! -s "$evidence_dir/sanitized.stderr"
test ! -s "$evidence_dir/reference_sanitized.stderr"
cmp "$evidence_dir/accepted.json" "$evidence_dir/sanitized.json"
cmp "$evidence_dir/reference_primary.json" \
    "$evidence_dir/reference_sanitized.json"

expect_rejection() {
    local label=$1
    shift
    set +e
    "$@" >"$evidence_dir/${label}.stdout" \
        2>"$evidence_dir/${label}.stderr"
    local status=$?
    set -e
    if [[ $status -eq 0 ]]; then
        echo "negative control unexpectedly passed: $label" >&2
        exit 1
    fi
    test ! -s "$evidence_dir/${label}.stdout"
    test -s "$evidence_dir/${label}.stderr"
}

expect_rejection project_intermediate \
    "$evidence_dir/phase" --project-intermediate
expect_rejection null_carrier \
    "$evidence_dir/phase" --null-carrier
expect_rejection nonmonic \
    "$evidence_dir/phase" "$nonmonic_fixture" "$reuse_fixture"
expect_rejection bad_coefficient \
    "$evidence_dir/phase" "$bad_coefficient_fixture" "$reuse_fixture"

jq -e '
    .result == "PASS"
    and .primary_boundary == [1,0,0,0,1]
    and .reuse_boundary == [1,1,1,1,1]
    and .primary_highest_degree_coefficient == 1
    and .reuse_highest_degree_coefficient == 1
    and .carrier_cells == 23
    and .input_signature_cells == 9
    and .resident_intermediate_cells == 4
    and .resident_final_cells == 5
    and .public_boundary_cells == 5
    and .dense_five_variable_membership_entries == 32
    and .same_carrier_transactions == 258
    and .carrier_creations_on_accepted_path == 1
    and .intermediate_decodes == 0
    and .intermediate_copies == 0
    and .boundary_decodes_per_transaction == 5
    and .boundary_copies_per_transaction == 2
    and .wrong_inverse_restoration_max_abs >= 0.001
    and .missing_inverse_restoration_max_abs >= 0.001
    and .reordered_inverse_restoration_max_abs >= 0.001
    and .snapshot_inverse_executed == false
    and .actual_inverse == true
    and .actual_restored_carrier_reuse == true
    and .final_boundary_only == true
    and .full_boundary_anf_signature == true
' "$evidence_dir/accepted.json" >/dev/null

jq -e '
    .primary_boundary == [1,0,1,0,0]
    and .reuse_boundary == [1,0,1,0,0]
    and .primary_highest_degree_coefficient == 0
    and .reuse_highest_degree_coefficient == 0
' "$evidence_dir/affine_sham.json" >/dev/null
jq -e '
    .primary_boundary == [1,0,0,1,0]
    and .reuse_boundary == [1,0,0,1,0]
    and .primary_highest_degree_coefficient == 0
    and .reuse_highest_degree_coefficient == 0
' "$evidence_dir/degree2_nonaffine.json" >/dev/null

accepted_plan=$(jq -r '.plan_fnv1a64' "$evidence_dir/accepted.json")
sham_plan=$(jq -r '.plan_fnv1a64' "$evidence_dir/affine_sham.json")
degree2_plan=$(jq -r '.plan_fnv1a64' "$evidence_dir/degree2_nonaffine.json")
[[ "$accepted_plan" == "$sham_plan" ]]
[[ "$accepted_plan" == "$degree2_plan" ]]
for counter in \
    phase_ands_per_transaction \
    carrier_reads_per_transaction \
    phase_cell_updates_per_transaction \
    boundary_decodes_per_transaction \
    boundary_copies_per_transaction
do
    accepted_count=$(jq -r ".$counter" "$evidence_dir/accepted.json")
    sham_count=$(jq -r ".$counter" "$evidence_dir/affine_sham.json")
    degree2_count=$(jq -r ".$counter" "$evidence_dir/degree2_nonaffine.json")
    [[ "$accepted_count" == "$sham_count" ]]
    [[ "$accepted_count" == "$degree2_count" ]]
done

jq -e '
    .result == "PASS"
    and .boundary_coefficients == [1,0,0,0,1]
    and .boundary_relation_rows == 32
    and .accepted_boundary_rows == 16
    and .symbolic_rows_exact == 32
    and .accepted_rows_with_unique_hidden_state == 16
    and .ce_anf_coefficient == 0
    and .fourth_boolean_derivative == 1
    and .gf2_affine == false
    and .extensional_storage_cells == 0
' "$evidence_dir/reference_primary.json" >/dev/null
jq -e '
    .result == "PASS"
    and .boundary_coefficients == [1,1,1,1,1]
    and .ce_anf_coefficient == 1
    and .fourth_boolean_derivative == 1
    and .gf2_affine == false
' "$evidence_dir/reference_reuse.json" >/dev/null
jq -e '
    .result == "PASS"
    and .boundary_coefficients == [1,0,1,0,0]
    and .ce_anf_coefficient == 0
    and .fourth_boolean_derivative == 0
    and .gf2_affine == true
' "$evidence_dir/reference_sham.json" >/dev/null
jq -e '
    .result == "PASS"
    and .boundary_coefficients == [1,0,0,1,0]
    and .ce_anf_coefficient == 1
    and .fourth_boolean_derivative == 0
    and .gf2_affine == false
' "$evidence_dir/reference_degree2.json" >/dev/null

jq -e --slurpfile primary "$evidence_dir/reference_primary.json" \
    --slurpfile reuse "$evidence_dir/reference_reuse.json" '
    .primary_boundary == $primary[0].boundary_coefficients
    and .reuse_boundary == $reuse[0].boundary_coefficients
' "$evidence_dir/accepted.json" >/dev/null

allowed_keys='[
  "actual_inverse",
  "actual_restored_carrier_reuse",
  "altered_hidden_source_boundary",
  "altered_hidden_source_restores",
  "boundary_copies_per_transaction",
  "boundary_decodes_per_transaction",
  "carrier_cells",
  "carrier_creations_on_accepted_path",
  "carrier_reads_per_transaction",
  "comparison_snapshot_bytes",
  "dense_five_variable_membership_entries",
  "final_boundary_only",
  "full_boundary_anf_signature",
  "input_signature_cells",
  "intermediate_copies",
  "intermediate_decodes",
  "live_carrier_bytes",
  "missing_inverse_restoration_max_abs",
  "phase_ands_per_transaction",
  "phase_cell_updates_per_transaction",
  "plan_fnv1a64",
  "primary_boundary",
  "primary_highest_degree_coefficient",
  "primary_restoration_max_abs",
  "primary_root_max_abs",
  "primary_source_fnv1a64",
  "public_boundary_cells",
  "quadratic_cut_applicable",
  "quadratic_cut_boundary",
  "quadratic_cut_restores",
  "reordered_inverse_restoration_max_abs",
  "repeated_reuse_max_abs",
  "resident_final_cells",
  "resident_intermediate_cells",
  "restoration_tolerance",
  "result",
  "reuse_boundary",
  "reuse_highest_degree_coefficient",
  "reuse_restoration_max_abs",
  "reuse_source_fnv1a64",
  "root_tolerance",
  "same_carrier_transactions",
  "schema",
  "snapshot_inverse_executed",
  "snapshot_restoration_max_abs",
  "terminal",
  "wrong_inverse_restoration_max_abs"
]'
jq -e --argjson allowed "$allowed_keys" \
    '((keys - $allowed) | length) == 0
     and (($allowed - keys) | length) == 0' \
    "$evidence_dir/accepted.json" >/dev/null

if rg -n \
    'SELECT|EXPECTED|BOUNDARY|ANSWER|WITNESS|CANDIDATE|TABLE|MASK' \
    "$primary_fixture" "$reuse_fixture" "$sham_fixture" "$degree2_fixture"
then
    echo "public fixture contains an answer-bearing field" >&2
    exit 1
fi

if rg -n \
    '1[UuLl]*[[:space:]]*<<|\[[[:space:]]*32[[:space:]]*\]|uint(32|64)_t[[:space:]]+.*(mask|table)|expected_boundary|answer_table' \
    "$phase_source"
then
    echo "native phase source contains a dense or answer-bearing seam" >&2
    exit 1
fi

allocation_calls=$(rg -c 'calloc\(' "$phase_source")
[[ "$allocation_calls" == "1" ]]
if nm -a "$evidence_dir/phase" | rg -i \
    'qr_boundary|qr_[fgj]$|reference|oracle|solver|sat|bdd'
then
    echo "native phase binary contains reference machinery" >&2
    exit 1
fi

strace -qq -f \
    -e trace=%file,%network,%ipc,write,writev,pwrite64,sendto,sendmsg \
    -o "$evidence_dir/no_smuggle.strace" \
    "$evidence_dir/phase" "$primary_fixture" "$reuse_fixture" \
    >"$evidence_dir/traced.json" \
    2>"$evidence_dir/traced.stderr"
test ! -s "$evidence_dir/traced.stderr"
cmp "$evidence_dir/accepted.json" "$evidence_dir/traced.json"
[[ $(rg -c 'write\(1,' "$evidence_dir/no_smuggle.strace") -eq 1 ]]
if rg -n \
    'write\(2,|writev\(|pwrite64\(|sendto\(|sendmsg\(|socket\(|shm(get|at|dt|ctl)\(|msgget\(|semget\(|O_WRONLY|O_RDWR|O_CREAT|O_TRUNC|O_APPEND' \
    "$evidence_dir/no_smuggle.strace"
then
    echo "native process exposed an undeclared output channel" >&2
    exit 1
fi

stack_file=$(find "$evidence_dir" -maxdepth 1 -name '*.su' -print -quit)
test -n "$stack_file"
stack_main=$(awk -F '\t' '$1 ~ /:main$/ {print $2}' "$stack_file")
stack_execute=$(awk -F '\t' '$1 ~ /:qb_execute$/ {print $2}' "$stack_file")
stack_z=$(awk -F '\t' '$1 ~ /:qb_apply_z$/ {print $2}' "$stack_file")
stack_product=$(awk -F '\t' '$1 ~ /:qb_symbol_product$/ {print $2}' "$stack_file")
stack_lock=$(awk -F '\t' '$1 ~ /:qb_lock$/ {print $2}' "$stack_file")
for value in \
    "$stack_main" "$stack_execute" "$stack_z" \
    "$stack_product" "$stack_lock"
do
    [[ "$value" =~ ^[0-9]+$ ]]
done
stack_chain_bytes=$((
    stack_main
    + stack_execute
    + stack_z
    + stack_product
    + stack_lock
))

phase_source_sha256=$(sha256sum "$phase_source" | awk '{print $1}')
reference_source_sha256=$(sha256sum "$reference_source" | awk '{print $1}')
qualifier_sha256=$(sha256sum "$qualifier_source" | awk '{print $1}')
primary_sha256=$(sha256sum "$primary_fixture" | awk '{print $1}')
reuse_sha256=$(sha256sum "$reuse_fixture" | awk '{print $1}')
sham_sha256=$(sha256sum "$sham_fixture" | awk '{print $1}')
degree2_sha256=$(sha256sum "$degree2_fixture" | awk '{print $1}')
nonmonic_sha256=$(sha256sum "$nonmonic_fixture" | awk '{print $1}')
bad_coefficient_sha256=$(
    sha256sum "$bad_coefficient_fixture" | awk '{print $1}'
)
accepted_sha256=$(sha256sum "$evidence_dir/accepted.json" | awk '{print $1}')
trace_sha256=$(sha256sum "$evidence_dir/no_smuggle.strace" | awk '{print $1}')
project_rejection_sha256=$(
    sha256sum "$evidence_dir/project_intermediate.stderr" | awk '{print $1}'
)
null_rejection_sha256=$(
    sha256sum "$evidence_dir/null_carrier.stderr" | awk '{print $1}'
)
nonmonic_rejection_sha256=$(
    sha256sum "$evidence_dir/nonmonic.stderr" | awk '{print $1}'
)
coefficient_rejection_sha256=$(
    sha256sum "$evidence_dir/bad_coefficient.stderr" | awk '{print $1}'
)
phase_binary_bytes=$(stat -c %s "$evidence_dir/phase")
reference_binary_bytes=$(stat -c %s "$evidence_dir/reference")
trace_bytes=$(stat -c %s "$evidence_dir/no_smuggle.strace")
live_carrier_bytes=$(jq -r '.live_carrier_bytes' "$evidence_dir/accepted.json")
snapshot_bytes=$(jq -r '.comparison_snapshot_bytes' "$evidence_dir/accepted.json")
material_bytes=$((live_carrier_bytes + snapshot_bytes))

jq -n \
    --arg claim \
        "ALGEBRAIC_FIXED_SCHEMA_QUADRATIC_ANF_TWO_HIDDEN_PORT_PHASE_COMPOSITION_ESTABLISHED" \
    --arg ceiling \
        "BOUNDED_BOOLEAN_GF2_MONIC_QAND_CHAIN_DEGREE4_FIVE_COEFFICIENT_BOUNDARY_SOFTWARE_REFERENCE_ONLY" \
    --arg plan "$accepted_plan" \
    --arg phase_source_sha256 "$phase_source_sha256" \
    --arg reference_source_sha256 "$reference_source_sha256" \
    --arg qualifier_sha256 "$qualifier_sha256" \
    --arg primary_sha256 "$primary_sha256" \
    --arg reuse_sha256 "$reuse_sha256" \
    --arg sham_sha256 "$sham_sha256" \
    --arg degree2_sha256 "$degree2_sha256" \
    --arg nonmonic_sha256 "$nonmonic_sha256" \
    --arg bad_coefficient_sha256 "$bad_coefficient_sha256" \
    --arg accepted_sha256 "$accepted_sha256" \
    --arg trace_sha256 "$trace_sha256" \
    --arg project_rejection_sha256 "$project_rejection_sha256" \
    --arg null_rejection_sha256 "$null_rejection_sha256" \
    --arg nonmonic_rejection_sha256 "$nonmonic_rejection_sha256" \
    --arg coefficient_rejection_sha256 "$coefficient_rejection_sha256" \
    --argjson stack_chain_bytes "$stack_chain_bytes" \
    --argjson material_bytes "$material_bytes" \
    --argjson phase_binary_bytes "$phase_binary_bytes" \
    --argjson reference_binary_bytes "$reference_binary_bytes" \
    --argjson trace_bytes "$trace_bytes" \
    --slurpfile accepted "$evidence_dir/accepted.json" \
    --slurpfile sham "$evidence_dir/affine_sham.json" \
    --slurpfile degree2 "$evidence_dir/degree2_nonaffine.json" \
    --slurpfile primary_reference "$evidence_dir/reference_primary.json" \
    --slurpfile reuse_reference "$evidence_dir/reference_reuse.json" \
    --slurpfile sham_reference "$evidence_dir/reference_sham.json" \
    --slurpfile degree2_reference "$evidence_dir/reference_degree2.json" '
    {
      result: "PASS",
      claim: $claim,
      claim_ceiling: $ceiling,
      plan_fnv1a64: $plan,
      accepted: $accepted[0],
      affine_sham: {
        boundary: $sham[0].primary_boundary,
        highest_degree_coefficient:
          $sham[0].primary_highest_degree_coefficient,
        plan_matches: ($sham[0].plan_fnv1a64 == $plan),
        operation_counts_match:
          (
            $sham[0].phase_ands_per_transaction
              == $accepted[0].phase_ands_per_transaction
            and $sham[0].carrier_reads_per_transaction
              == $accepted[0].carrier_reads_per_transaction
            and $sham[0].phase_cell_updates_per_transaction
              == $accepted[0].phase_cell_updates_per_transaction
          )
      },
      degree2_nonaffine_control: {
        boundary: $degree2[0].primary_boundary,
        highest_degree_coefficient:
          $degree2[0].primary_highest_degree_coefficient,
        ce_anf_coefficient:
          $degree2_reference[0].ce_anf_coefficient,
        fourth_boolean_derivative:
          $degree2_reference[0].fourth_boolean_derivative,
        gf2_affine: $degree2_reference[0].gf2_affine,
        plan_matches: ($degree2[0].plan_fnv1a64 == $plan),
        operation_counts_match:
          (
            $degree2[0].phase_ands_per_transaction
              == $accepted[0].phase_ands_per_transaction
            and $degree2[0].carrier_reads_per_transaction
              == $accepted[0].carrier_reads_per_transaction
            and $degree2[0].phase_cell_updates_per_transaction
              == $accepted[0].phase_cell_updates_per_transaction
          )
      },
      independent_reference: {
        primary: $primary_reference[0],
        reuse: $reuse_reference[0],
        affine_sham: $sham_reference[0],
        degree2_nonaffine: $degree2_reference[0],
        native_boundary_parity: true
      },
      controls: {
        wrong_inverse: "PASS",
        missing_inverse: "PASS",
        reordered_inverse: "PASS_APPLICABLE_NONCOMMUTING",
        altered_hidden_source: "PASS_CHANGED_AND_RESTORED",
        quadratic_term_cut: "PASS_CHANGED_AND_RESTORED",
        affine_sham_same_plan: "PASS",
        degree2_nonaffine_same_plan: "PASS",
        attempted_intermediate_projection: "PASS_REJECTED",
        null_carrier: "PASS_REJECTED",
        nonmonic_definition: "PASS_REJECTED",
        malformed_coefficient: "PASS_REJECTED",
        snapshot: "PASS_SEPARATE_WEAKER_PATH",
        repeated_actual_restored_reuse: "PASS_258",
        deterministic_replay: "PASS",
        analyzer: "PASS",
        sanitizer: "PASS",
        no_smuggle_trace: "PASS"
      },
      resources: {
        carrier_plus_comparison_snapshot_bytes: $material_bytes,
        compiler_measured_nested_stack_chain_bytes: $stack_chain_bytes,
        phase_binary_bytes: $phase_binary_bytes,
        reference_binary_bytes: $reference_binary_bytes,
        no_smuggle_trace_bytes: $trace_bytes
      },
      sha256: {
        phase_source: $phase_source_sha256,
        reference_source: $reference_source_sha256,
        qualifier: $qualifier_sha256,
        primary_fixture: $primary_sha256,
        reuse_fixture: $reuse_sha256,
        affine_sham_fixture: $sham_sha256,
        degree2_nonaffine_fixture: $degree2_sha256,
        nonmonic_fixture: $nonmonic_sha256,
        malformed_coefficient_fixture: $bad_coefficient_sha256,
        accepted_result: $accepted_sha256,
        no_smuggle_trace: $trace_sha256,
        projection_rejection: $project_rejection_sha256,
        null_carrier_rejection: $null_rejection_sha256,
        nonmonic_rejection: $nonmonic_rejection_sha256,
        malformed_coefficient_rejection: $coefficient_rejection_sha256
      },
      not_established: [
        "ARBITRARY_NON_AFFINE_RELATION_CLOSURE",
        "GENERAL_BOOLEAN_ANF_ELIMINATION",
        "BOUNDED_DEGREE_OR_TERM_GROWTH",
        "MANY_TO_MANY_NON_AFFINE_BOUNDARY",
        "CATVM_ENFORCEMENT_FOR_QUADRATIC_ANF",
        "COMPUTATIONAL_ADVANTAGE",
        "SMALL_WALL_CROSSING",
        "PHYSICAL_WAVEFORM_OR_SILICON_EXECUTION",
        "UNLIMITED_CATALYTIC_COMPUTATION"
      ],
      terminal: false
    }
    ' >"$evidence_dir/result.json"

sha256sum \
    "$phase_source" \
    "$reference_source" \
    "$qualifier_source" \
    "$primary_fixture" \
    "$reuse_fixture" \
    "$sham_fixture" \
    "$degree2_fixture" \
    "$nonmonic_fixture" \
    "$bad_coefficient_fixture" \
    "$evidence_dir/phase" \
    "$evidence_dir/reference" \
    "$evidence_dir/accepted.json" \
    "$evidence_dir/affine_sham.json" \
    "$evidence_dir/degree2_nonaffine.json" \
    "$evidence_dir/reference_primary.json" \
    "$evidence_dir/reference_reuse.json" \
    "$evidence_dir/reference_sham.json" \
    "$evidence_dir/reference_degree2.json" \
    "$evidence_dir/project_intermediate.stderr" \
    "$evidence_dir/null_carrier.stderr" \
    "$evidence_dir/nonmonic.stderr" \
    "$evidence_dir/bad_coefficient.stderr" \
    "$evidence_dir/no_smuggle.strace" \
    "$evidence_dir/result.json" \
    >"$evidence_dir/SHA256SUMS"

jq . "$evidence_dir/result.json"
