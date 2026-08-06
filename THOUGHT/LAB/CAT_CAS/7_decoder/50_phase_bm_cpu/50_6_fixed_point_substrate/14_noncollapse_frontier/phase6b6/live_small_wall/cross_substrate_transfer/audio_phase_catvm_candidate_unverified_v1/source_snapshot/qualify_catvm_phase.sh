#!/bin/sh
set -eu

source_dir=${1:-.}
work_dir=${2:-/tmp/catvm_phase_qualification}

if [ -e "$work_dir" ]; then
    test -d "$work_dir"
    test -z "$(find "$work_dir" -mindepth 1 -maxdepth 1 -print -quit)"
else
    mkdir -p "$work_dir"
fi

core_source="$source_dir/catvm_phase_core.c"
core_header="$source_dir/catvm_phase_core.h"
service_source="$source_dir/catvm_phase_service.c"
controller_source="$source_dir/catvm_phase_controller.c"
direct_source="$source_dir/catvm_phase_direct.c"
series_source="$source_dir/algebraic_series_parallel_phase.c"
primary_fixture="$source_dir/catvm_primary.aspr"
reuse_fixture="$source_dir/catvm_reuse.aspr"

service="$work_dir/catvm_phase_service"
controller="$work_dir/catvm_phase_controller"
direct="$work_dir/catvm_phase_direct"
series="$work_dir/algebraic_series_parallel_phase"
service_sanitized="$work_dir/catvm_phase_service_sanitized"
controller_sanitized="$work_dir/catvm_phase_controller_sanitized"

gcc -std=c11 -O2 -Wall -Wextra -Werror -pedantic \
    "$core_source" "$service_source" \
    -o "$service" -lm -lseccomp
gcc -std=c11 -O2 -Wall -Wextra -Werror -pedantic \
    "$controller_source" -o "$controller"
gcc -std=c11 -O2 -Wall -Wextra -Werror -pedantic \
    "$core_source" "$direct_source" -o "$direct" -lm
gcc -std=c11 -O2 -Wall -Wextra -Werror -pedantic \
    "$series_source" -o "$series" -lm
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
    case_name=$1
    service_binary=$2
    controller_binary=$3
    backend=$4
    restore_mode=$5
    scenario=$6
    cycles=$7
    case_dir="$work_dir/$case_name"
    mkdir "$case_dir"
    socket_path="$case_dir/service.sock"

    ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
    UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
        "$service_binary" \
        "$socket_path" "$backend" "$restore_mode" \
        >"$case_dir/service.stdout" \
        2>"$case_dir/service.stderr" &
    service_pid=$!
    ready=0
    attempt=0
    while [ "$attempt" -lt 400 ]; do
        if [ -S "$socket_path" ]; then
            ready=1
            break
        fi
        if ! kill -0 "$service_pid" 2>/dev/null; then
            break
        fi
        attempt=$((attempt + 1))
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
    controller_result=$?
    wait "$service_pid"
    service_result=$?
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
    in-place correct accepted 1000
run_case accepted_replay \
    "$service" "$controller" \
    in-place correct accepted 1000
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
    snapshot snapshot snapshot 64
run_case sanitized_accepted \
    "$service_sanitized" "$controller_sanitized" \
    in-place correct accepted 64
run_case sanitized_snapshot \
    "$service_sanitized" "$controller_sanitized" \
    snapshot snapshot snapshot 16

"$direct" 10000 >"$work_dir/warm_direct.json"
run_case warm_inert \
    "$service" "$controller" \
    null correct transport 10000
run_case warm_snapshot \
    "$service" "$controller" \
    snapshot snapshot snapshot 9998
run_case warm_in_place \
    "$service" "$controller" \
    in-place correct accepted 9998

"$series" "$primary_fixture" "$reuse_fixture" \
    >"$work_dir/branch_native_reference.jsonl"

jq 'del(.transaction_wall_ns, .average_transaction_wall_ns)' \
    "$work_dir/accepted/result.json" \
    >"$work_dir/accepted/normalized.json"
jq 'del(.transaction_wall_ns, .average_transaction_wall_ns)' \
    "$work_dir/accepted_replay/result.json" \
    >"$work_dir/accepted_replay/normalized.json"
cmp \
    "$work_dir/accepted/normalized.json" \
    "$work_dir/accepted_replay/normalized.json"

jq -e '
    .claim
        == "CATVM_OPEN_INTERMEDIATE_COMPOSITION_ESTABLISHED_ON_PHASE_BACKEND"
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
    and .carrier_cells == 24
    and .physical_complex_values == 48
    and .logical_carrier_bytes == 768
    and .mapped_locked_bytes == 4096
    and .compiled_program_bytes == 48
    and .compiled_morphisms == 2
    and .maximum_temporary_complex_values == 52
    and .restoration_generations == 1002
    and .alternating_repeat_cycles == 1000
    and .maximum_restoration_error <= .restoration_tolerance
    and .primary_boundary.decoded_intermediate_coefficients == 0
    and .reuse_boundary.decoded_intermediate_coefficients == 0
    and .primary_boundary.fnv1a64 != .reuse_boundary.fnv1a64
    and .final_restoration.actual_inverse
    and (.final_restoration.snapshot_reload | not)
    and .final_restoration.snapshot_bytes_written == 0
    and .final_restoration.snapshot_bytes_reloaded == 0
    and .final_restoration.native_compose_calls == 2004
    and .final_restoration.native_intersection_calls == 2004
    and .final_restoration.native_symbol_products == 224448
    and .final_restoration.phase_cell_updates == 48096
    and .final_restoration.inverse_factor_recomputations == 2004
    and .final_restoration.boundary_decodes == 4008
    and .final_restoration.restoration_cell_checks == 24048
    and .final_restoration.protocol_rx_cleared
    and .final_restoration.backend_queue_empty
    and .final_restoration.transient_state_exact
' "$work_dir/accepted/result.json" >/dev/null

jq -e '
    .claim == "CATVM_SNAPSHOT_BACKED_TRANSACTIONAL_REUSE_ESTABLISHED"
    and .scenario == "snapshot-baseline"
    and .same_carrier_creation_count == 1
    and .mapped_locked_bytes == 8192
    and (.final_restoration.actual_inverse | not)
    and .final_restoration.snapshot_reload
    and .final_restoration.snapshot_bytes_written == 25344
    and .final_restoration.snapshot_bytes_reloaded == 25344
    and .final_restoration.inverse_factor_recomputations == 0
    and .maximum_restoration_error == 0
' "$work_dir/snapshot/result.json" >/dev/null

for control in wrong_g missing_g reordered
do
    jq -e '
        .prospectively_applicable
        and .restoration_failure_detected
        and .restoration.control_discriminated
        and (.restoration.carrier_within_tolerance | not)
        and .restoration.maximum_abs_error >= 0.001
    ' "$work_dir/$control/result.json" >/dev/null
done

jq -e '
    .carrier_required
    and (.final_boundary_emitted | not)
' "$work_dir/null_carrier/result.json" >/dev/null

jq -s -e '
    .[0].boundary_coefficients
        == .[7].primary_boundary.coefficients
    and .[1].boundary_coefficients
        == .[7].reuse_boundary.coefficients
' \
    "$work_dir/branch_native_reference.jsonl" \
    "$work_dir/accepted/result.json" >/dev/null

jq -e '
    .enforced_machine_boundary == false
    and .cycles == 10000
    and .carrier_creations == 1
    and .native_compose_calls == 20000
    and .native_intersection_calls == 20000
    and .native_symbol_products == 2240000
    and .phase_cell_updates == 480000
    and .boundary_decodes == 40000
    and .inverse_factor_recomputations == 20000
    and .snapshot_bytes_written == 0
    and .snapshot_bytes_reloaded == 0
    and .restoration_cell_checks == 240000
' "$work_dir/warm_direct.json" >/dev/null

test "$(rg -n 'decode_final_root' "$core_source" | wc -l)" -eq 2
awk '
    /int catvm_project_final/ {inside=1}
    /int catvm_restore/ {inside=0}
    inside {print}
' "$core_source" >"$work_dir/final_projection_slice.c"
if rg -q 'INTERMEDIATE_Y_START' "$work_dir/final_projection_slice.c"; then
    echo "final projection slice can address the intermediate" >&2
    exit 1
fi
if rg -q \
    'catvm_phase_core|complex\.h|decode_final_root|symbol_product|poly_multiply|struct catvm_machine' \
    "$controller_source"
then
    echo "controller contains phase or carrier implementation" >&2
    exit 1
fi
if nm "$controller" | rg -q \
    'catvm_|decode_final_root|symbol_product|poly_multiply'
then
    echo "controller binary links phase or carrier symbols" >&2
    exit 1
fi
if rg -q \
    'shm_open|shmget|shmat|memfd_create|MAP_SHARED|SCM_RIGHTS|fork\(|exec[lvpe]*\(' \
    "$core_source" "$service_source"
then
    echo "backend contains forbidden shared or child process mechanism" >&2
    exit 1
fi
test "$(rg -n '\bsend\(' "$service_source" | wc -l)" -eq 1
test "$(rg -n '\brecv\(' "$service_source" | wc -l)" -eq 2
test "$(rg -n 'MSG_PEEK \| MSG_DONTWAIT' "$service_source" | wc -l)" -eq 1
if rg -q '\b(printf|fprintf|puts|fputs|fwrite|write)\(' \
    "$service_source"
then
    echo "service contains a stdout/stderr/file write path" >&2
    exit 1
fi
if rg -q \
    '"(intermediate_coefficients|intermediate_hash|phase_cells|witnesses|candidate_sets|truth_tables|assignments)"' \
    "$work_dir"
then
    echo "qualification artifact contains forbidden intermediate material" >&2
    exit 1
fi

sha256sum \
    "$core_source" \
    "$core_header" \
    "$service_source" \
    "$controller_source" \
    "$direct_source" \
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
            "CATVM_OPEN_INTERMEDIATE_COMPOSITION_ESTABLISHED_ON_PHASE_BACKEND",
        compiler_builds: $compiler_builds,
        sanitizers: "PASS_IN_PLACE_AND_SNAPSHOT",
        carrier_cells: $accepted.carrier_cells,
        carrier_creations: $accepted.same_carrier_creation_count,
        restoration_generations: $accepted.restoration_generations,
        maximum_restoration_error: $accepted.maximum_restoration_error,
        restoration_tolerance: $accepted.restoration_tolerance,
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
            intermediate_decodes: 0
        },
        snapshot_claim:
            "CATVM_SNAPSHOT_BACKED_TRANSACTIONAL_REUSE_ESTABLISHED",
        warm_average_transaction_ns: {
            direct_process_phase: $direct.average_transaction_wall_ns,
            isolated_inert_boundary:
                $inert.average_transaction_wall_ns,
            snapshot_catvm: $snapshot.average_transaction_wall_ns,
            in_place_catvm: $inplace.average_transaction_wall_ns
        },
        terminal: false
    }'
