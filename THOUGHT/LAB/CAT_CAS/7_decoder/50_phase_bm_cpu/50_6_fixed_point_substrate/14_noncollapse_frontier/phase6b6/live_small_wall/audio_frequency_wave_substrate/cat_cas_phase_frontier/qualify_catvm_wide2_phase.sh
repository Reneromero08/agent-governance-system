#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "usage: $0 SOURCE_DIR WORK_DIR" >&2
    exit 2
fi

source_dir=$(cd "$1" && pwd)
work_dir=$2
mkdir -p "$work_dir"
work_dir=$(cd "$work_dir" && pwd)

core_source="$source_dir/catvm_wide2_core.c"
core_header="$source_dir/catvm_wide2_core.h"
service_source="$source_dir/catvm_wide2_service.c"
shared_service_source="$source_dir/catvm_phase_service.c"
controller_source="$source_dir/catvm_wide2_controller.c"
shared_controller_source="$source_dir/catvm_phase_controller.c"
direct_source="$source_dir/catvm_wide2_direct.c"
wide_source="$source_dir/algebraic_wide_interface_phase.c"
series_source="$source_dir/algebraic_series_parallel_phase.c"
primary_fixture="$source_dir/wide_interface_primary.wr2"
reuse_fixture="$source_dir/wide_interface_reuse.wr2"

service="$work_dir/catvm_wide2_service"
controller="$work_dir/catvm_wide2_controller"
direct="$work_dir/catvm_wide2_direct"
wide="$work_dir/algebraic_wide_interface_phase"
service_sanitized="$work_dir/catvm_wide2_service_sanitized"
controller_sanitized="$work_dir/catvm_wide2_controller_sanitized"

gcc -std=c11 -O2 -Wall -Wextra -Werror -pedantic \
    "$core_source" "$service_source" \
    -o "$service" -lm -lseccomp
gcc -std=c11 -O2 -Wall -Wextra -Werror -pedantic \
    "$controller_source" -o "$controller"
gcc -std=c11 -O2 -Wall -Wextra -Werror -pedantic \
    "$core_source" "$direct_source" -o "$direct" -lm
gcc -std=c11 -O2 -Wall -Wextra -Werror -pedantic \
    "$wide_source" -o "$wide" -lm
gcc -std=c11 -O1 -g -Wall -Wextra -Werror -pedantic \
    -DCATVM_SANITIZER_BUILD=1 \
    -fsanitize=address,undefined \
    -fno-sanitize-recover=undefined \
    -fno-omit-frame-pointer \
    "$core_source" "$service_source" \
    -o "$service_sanitized" -lm -lseccomp
gcc -std=c11 -O1 -g -Wall -Wextra -Werror -pedantic \
    -DCATVM_SANITIZER_BUILD=1 \
    -fsanitize=address,undefined \
    -fno-sanitize-recover=undefined \
    -fno-omit-frame-pointer \
    "$controller_source" -o "$controller_sanitized"
gcc -std=c11 -O2 -Wall -Wextra -Werror -pedantic -fanalyzer \
    "$core_source" "$service_source" \
    -o "$work_dir/service_analyzer_build" -lm -lseccomp
gcc -std=c11 -O2 -Wall -Wextra -Werror -pedantic -fanalyzer \
    "$controller_source" -o "$work_dir/controller_analyzer_build"

compiler_builds=8
if command -v clang >/dev/null 2>&1; then
    clang -std=c11 -O2 -Wall -Wextra -Werror -pedantic \
        "$core_source" "$service_source" \
        -o "$work_dir/service_clang" -lm -lseccomp
    clang -std=c11 -O2 -Wall -Wextra -Werror -pedantic \
        "$controller_source" -o "$work_dir/controller_clang"
    compiler_builds=$((compiler_builds + 2))
fi

run_case() {
    local case_name=$1
    local service_binary=$2
    local controller_binary=$3
    local backend=$4
    local restore_mode=$5
    local scenario=$6
    local cycles=$7
    local case_dir="$work_dir/$case_name"
    mkdir "$case_dir"
    local socket_path="$case_dir/service.sock"

    ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
    UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
        "$service_binary" \
        "$socket_path" "$backend" "$restore_mode" \
        >"$case_dir/service.stdout" \
        2>"$case_dir/service.stderr" &
    local service_pid=$!
    local ready=0
    for _ in $(seq 1 400); do
        if [ -S "$socket_path" ]; then
            ready=1
            break
        fi
        if ! kill -0 "$service_pid" 2>/dev/null; then
            break
        fi
        sleep 0.01
    done
    if [ "$ready" -ne 1 ]; then
        set +e
        wait "$service_pid"
        set -e
        echo "service did not become ready: $case_name" >&2
        return 1
    fi

    set +e
    ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
    UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
        "$controller_binary" \
        "$socket_path" "$scenario" "$cycles" \
        >"$case_dir/result.json" \
        2>"$case_dir/controller.stderr"
    local controller_result=$?
    wait "$service_pid"
    local service_result=$?
    set -e

    test "$controller_result" -eq 0
    test "$service_result" -eq 0
    test ! -e "$socket_path"
    test ! -s "$case_dir/service.stdout"
    test ! -s "$case_dir/service.stderr"
    test ! -s "$case_dir/controller.stderr"
    jq -e '.result == "PASS"' "$case_dir/result.json" >/dev/null
}

run_case accepted \
    "$service" "$controller" \
    in-place correct accepted 256
run_case accepted_replay \
    "$service" "$controller" \
    in-place correct accepted 256
run_case wrong_g \
    "$service" "$controller" \
    in-place wrong-g wrong-g 0
run_case missing_g \
    "$service" "$controller" \
    in-place missing-g missing-g 0
run_case reordered \
    "$service" "$controller" \
    in-place reordered reordered 0
run_case null_carrier \
    "$service" "$controller" \
    null correct null 0
run_case snapshot \
    "$service" "$controller" \
    snapshot snapshot snapshot 32
run_case sanitized_accepted \
    "$service_sanitized" "$controller_sanitized" \
    in-place correct accepted 32
run_case sanitized_snapshot \
    "$service_sanitized" "$controller_sanitized" \
    snapshot snapshot snapshot 8

"$direct" 1000 >"$work_dir/warm_direct.json"
run_case warm_inert \
    "$service" "$controller" \
    null correct transport 1000
run_case warm_snapshot \
    "$service" "$controller" \
    snapshot snapshot snapshot 998
run_case warm_in_place \
    "$service" "$controller" \
    in-place correct accepted 998

(
    cd "$source_dir"
    "$wide" \
        "$(basename "$primary_fixture")" \
        "$(basename "$reuse_fixture")"
) >"$work_dir/branch_native_reference.jsonl"

jq 'del(.transaction_wall_ns, .average_transaction_wall_ns)' \
    "$work_dir/accepted/result.json" \
    >"$work_dir/accepted/normalized.json"
jq 'del(.transaction_wall_ns, .average_transaction_wall_ns)' \
    "$work_dir/accepted_replay/result.json" \
    >"$work_dir/accepted_replay/normalized.json"
cmp \
    "$work_dir/accepted/normalized.json" \
    "$work_dir/accepted_replay/normalized.json"

accepted_keys='[
  "alternating_repeat_cycles",
  "average_transaction_wall_ns",
  "carrier_cells",
  "claim",
  "compiled_morphisms",
  "compiled_morphism_descriptor_bytes",
  "compiled_program_bytes",
  "final_restoration",
  "intermediate_projection_denied",
  "logical_carrier_bytes",
  "mapped_locked_bytes",
  "maximum_restoration_error",
  "maximum_temporary_complex_values",
  "physical_complex_values",
  "primary_boundary",
  "proc_fd_denied",
  "proc_maps_denied",
  "proc_mem_denied",
  "process_vm_readv_denied",
  "ptrace_denied",
  "request_bytes",
  "request_packets",
  "response_bytes",
  "response_packets",
  "restoration_generations",
  "restoration_tolerance",
  "result",
  "reuse_boundary",
  "same_carrier_creation_count",
  "scenario",
  "seccomp_enforced",
  "transaction_wall_ns",
  "pidfd_getfd_denied"
]'
boundary_keys='[
  "coefficients",
  "decoded_intermediate_coefficients",
  "event",
  "fnv1a64",
  "maximum_root_error",
  "ok",
  "port"
]'
restoration_keys='[
  "actual_inverse",
  "backend",
  "backend_queue_empty",
  "boundary_decodes",
  "carrier_creations",
  "carrier_within_tolerance",
  "coefficient_accumulation_additions",
  "contract2_workspace_cleared",
  "control_discriminated",
  "event",
  "generation",
  "generation_transition_exact",
  "invariant_state_exact",
  "inverse_factor_recomputations",
  "maximum_abs_error",
  "native_compose_calls",
  "native_contract2_calls",
  "native_intersection_calls",
  "native_symbol_products",
  "ok",
  "phase_cell_updates",
  "protocol_rx_cleared",
  "restoration_cell_checks",
  "restriction_and_intersection_additions",
  "snapshot_bytes_reloaded",
  "snapshot_bytes_written",
  "snapshot_reload",
  "tolerance",
  "transient_state_exact"
]'
for case_name in accepted accepted_replay sanitized_accepted; do
    jq -e \
        --argjson outer "$accepted_keys" \
        --argjson boundary "$boundary_keys" \
        --argjson restoration "$restoration_keys" '
        (keys | sort) == ($outer | sort)
        and (.primary_boundary | keys | sort) == ($boundary | sort)
        and (.reuse_boundary | keys | sort) == ($boundary | sort)
        and (.final_restoration | keys | sort)
            == ($restoration | sort)
    ' "$work_dir/$case_name/result.json" >/dev/null
done

jq -e '
    .claim
        == "CATVM_WIDTH2_OPEN_INTERMEDIATE_CONTRACT2_COMPOSITION_"
            + "ESTABLISHED_ON_PHASE_BACKEND"
    and .scenario == "accepted-in-place"
    and .seccomp_enforced
    and .intermediate_projection_denied
    and .proc_mem_denied
    and .proc_maps_denied
    and .proc_fd_denied
    and .process_vm_readv_denied
    and .ptrace_denied
    and .pidfd_getfd_denied
    and .same_carrier_creation_count == 1
    and .carrier_cells == 96
    and .physical_complex_values == 192
    and .logical_carrier_bytes == 3072
    and .mapped_locked_bytes == 8192
    and .compiled_program_bytes == 192
    and .compiled_morphisms == 2
    and .compiled_morphism_descriptor_bytes == 16
    and .maximum_temporary_complex_values == 240
    and .restoration_generations == 258
    and .alternating_repeat_cycles == 256
    and .maximum_restoration_error <= .restoration_tolerance
    and .primary_boundary.coefficients
        == [1,0,2,1,2,1,2,1,0,0,1,1,1,1,1,1]
    and .reuse_boundary.coefficients
        == [1,0,2,1,0,0,1,1,2,1,2,1,1,1,1,1]
    and .primary_boundary.decoded_intermediate_coefficients == 0
    and .reuse_boundary.decoded_intermediate_coefficients == 0
    and .primary_boundary.fnv1a64 != .reuse_boundary.fnv1a64
    and .final_restoration.actual_inverse
    and (.final_restoration.snapshot_reload | not)
    and .final_restoration.contract2_workspace_cleared
    and .final_restoration.snapshot_bytes_written == 0
    and .final_restoration.snapshot_bytes_reloaded == 0
    and .final_restoration.native_compose_calls == 0
    and .final_restoration.native_intersection_calls == 0
    and .final_restoration.native_contract2_calls == 1032
    and .final_restoration.native_symbol_products == 1849344
    and .final_restoration.coefficient_accumulation_additions
        == 1849344
    and .final_restoration.restriction_and_intersection_additions
        == 115584
    and .final_restoration.phase_cell_updates == 49536
    and .final_restoration.inverse_factor_recomputations == 516
    and .final_restoration.boundary_decodes == 4128
    and .final_restoration.restoration_cell_checks == 24768
    and .final_restoration.protocol_rx_cleared
    and .final_restoration.backend_queue_empty
    and .final_restoration.transient_state_exact
' "$work_dir/accepted/result.json" >/dev/null

jq -e '
    .claim
        == "CATVM_WIDTH2_SNAPSHOT_BACKED_TRANSACTIONAL_REUSE_"
            + "ESTABLISHED"
    and .scenario == "snapshot-baseline"
    and .same_carrier_creation_count == 1
    and .mapped_locked_bytes == 12288
    and (.final_restoration.actual_inverse | not)
    and .final_restoration.snapshot_reload
    and .final_restoration.contract2_workspace_cleared
    and .final_restoration.snapshot_bytes_written == 52224
    and .final_restoration.snapshot_bytes_reloaded == 52224
    and .final_restoration.inverse_factor_recomputations == 0
    and .maximum_restoration_error == 0
' "$work_dir/snapshot/result.json" >/dev/null

for control in wrong_g missing_g reordered; do
    jq -e '
        .prospectively_applicable
        and .restoration_failure_detected
        and .restoration.control_discriminated
        and (.restoration.carrier_within_tolerance | not)
        and .restoration.contract2_workspace_cleared
        and .restoration.maximum_abs_error >= 0.001
    ' "$work_dir/$control/result.json" >/dev/null
done

jq -e '
    .carrier_required
    and (.final_boundary_emitted | not)
' "$work_dir/null_carrier/result.json" >/dev/null

jq -s -e '
    .[0].boundary_coefficients
        == .[9].primary_boundary.coefficients
    and .[1].boundary_coefficients
        == .[9].reuse_boundary.coefficients
' \
    "$work_dir/branch_native_reference.jsonl" \
    "$work_dir/accepted/result.json" >/dev/null

jq -e '
    .enforced_machine_boundary == false
    and .cycles == 1000
    and .carrier_creations == 1
    and .carrier_cells == 96
    and .workspace_complex_values == 240
    and .native_contract2_calls == 4000
    and .native_symbol_products == 7168000
    and .coefficient_accumulation_additions == 7168000
    and .restriction_and_intersection_additions == 448000
    and .phase_cell_updates == 192000
    and .boundary_decodes == 16000
    and .inverse_factor_recomputations == 2000
    and .snapshot_bytes_written == 0
    and .snapshot_bytes_reloaded == 0
    and .restoration_cell_checks == 96000
    and .maximum_restoration_error <= 2e-12
' "$work_dir/warm_direct.json" >/dev/null

test "$(rg -n 'decode_final_root' "$core_source" | wc -l)" -eq 2
awk '
    /int catvm_project_final/ {inside=1}
    /int catvm_restore/ {inside=0}
    inside {print}
' "$core_source" >"$work_dir/final_projection_slice.c"
if rg -q 'INTERMEDIATE_Y_START' "$work_dir/final_projection_slice.c"; then
    echo "width-two final projection can address the intermediate" >&2
    exit 1
fi
if nm "$controller" | rg -q \
    'catvm_|decode_final_root|symbol_product|wide_poly_multiply|boolean_norm'
then
    echo "width-two controller links phase or carrier symbols" >&2
    exit 1
fi
if rg -q \
    'shm_open|shmget|shmat|memfd_create|MAP_SHARED|SCM_RIGHTS|fork\(|exec[lvpe]*\(' \
    "$core_source" "$shared_service_source"
then
    echo "width-two backend contains forbidden shared or child mechanism" >&2
    exit 1
fi
test "$(rg -n '\bsend\(' "$shared_service_source" | wc -l)" -eq 1
test "$(rg -n '\brecv\(' "$shared_service_source" | wc -l)" -eq 2
test "$(rg -n 'MSG_PEEK \| MSG_DONTWAIT' "$shared_service_source" | wc -l)" -eq 1
if rg -q '\b(printf|fprintf|puts|fputs|fwrite|write)\(' \
    "$shared_service_source" "$service_source"
then
    echo "width-two service contains a stdout/stderr/file write path" >&2
    exit 1
fi
if rg -q \
    '"(intermediate_coefficients|intermediate_hash|phase_cells|witnesses|candidate_sets|truth_tables|assignments)"' \
    "$work_dir"
then
    echo "width-two qualification artifact exposes forbidden state" >&2
    exit 1
fi
test "$(rg -n 'contract2_workspace' "$core_header" | wc -l)" -eq 1
test "$(rg -n 'sizeof\(machine->contract2_workspace\)' "$core_source" | wc -l)" -eq 2
test "$(rg -n 'machine->contract2_workspace' "$core_source" | wc -l)" -eq 7

sha256sum \
    "$core_source" \
    "$core_header" \
    "$service_source" \
    "$shared_service_source" \
    "$controller_source" \
    "$shared_controller_source" \
    "$direct_source" \
    "$wide_source" \
    "$primary_fixture" \
    "$reuse_fixture" \
    >"$work_dir/source_sha256.txt"
sha256sum \
    "$work_dir/accepted/result.json" \
    "$work_dir/snapshot/result.json" \
    "$work_dir/wrong_g/result.json" \
    "$work_dir/missing_g/result.json" \
    "$work_dir/reordered/result.json" \
    "$work_dir/null_carrier/result.json" \
    "$work_dir/warm_direct.json" \
    "$work_dir/warm_inert/result.json" \
    "$work_dir/warm_snapshot/result.json" \
    "$work_dir/warm_in_place/result.json" \
    >"$work_dir/result_sha256.txt"

jq -n \
    --argjson accepted "$(cat "$work_dir/accepted/result.json")" \
    --argjson direct "$(cat "$work_dir/warm_direct.json")" \
    --argjson inert "$(cat "$work_dir/warm_inert/result.json")" \
    --argjson snapshot "$(cat "$work_dir/warm_snapshot/result.json")" \
    --argjson inplace "$(cat "$work_dir/warm_in_place/result.json")" \
    --argjson compiler_builds "$compiler_builds" \
    '{
        result: "PASS",
        claim:
            "CATVM_WIDTH2_OPEN_INTERMEDIATE_CONTRACT2_COMPOSITION_"
            + "ESTABLISHED_ON_PHASE_BACKEND",
        compiler_builds: $compiler_builds,
        sanitizers: "PASS_IN_PLACE_AND_SNAPSHOT",
        carrier_cells: $accepted.carrier_cells,
        carrier_creations: $accepted.same_carrier_creation_count,
        locked_mapping_bytes: $accepted.mapped_locked_bytes,
        contract2_workspace_complex_values:
            $accepted.maximum_temporary_complex_values,
        restoration_generations: $accepted.restoration_generations,
        maximum_restoration_error: $accepted.maximum_restoration_error,
        restoration_tolerance: $accepted.restoration_tolerance,
        native_contract2_calls:
            $accepted.final_restoration.native_contract2_calls,
        inverse_controls: {
            wrong: "PASS",
            missing: "PASS",
            reordered_noncommuting: "PASS"
        },
        no_smuggle: {
            projection_denied: true,
            process_access_denied: true,
            strict_protocol: true,
            service_output_bytes: 0,
            intermediate_decodes: 0,
            workspace_cleared: true
        },
        snapshot_claim:
            "CATVM_WIDTH2_SNAPSHOT_BACKED_TRANSACTIONAL_REUSE_ESTABLISHED",
        warm_average_transaction_ns: {
            direct_process_phase: $direct.average_transaction_wall_ns,
            isolated_inert_boundary:
                $inert.average_transaction_wall_ns,
            snapshot_catvm: $snapshot.average_transaction_wall_ns,
            in_place_catvm: $inplace.average_transaction_wall_ns
        },
        terminal: false
    }'
