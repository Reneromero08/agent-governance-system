#define _POSIX_C_SOURCE 200809L

/*
 * Warm direct-process baseline for the same native phase transaction.
 * This is explicitly not an enforced CATVM boundary.
 */

#include "catvm_phase_core.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static const struct catvm_program PRIMARY = {
    .left = {0, 1, 2, 0},
    .right = {0, 1, 2, 0},
    .constraint = {0, 1, 0, 0}
};

static const struct catvm_program REUSE = {
    .left = {2, 1, 1, 0},
    .right = {0, 1, 2, 0},
    .constraint = {0, 0, 1, 0}
};

static int run_transaction(
    struct catvm_machine *machine,
    const struct catvm_program *program
) {
    struct catvm_projection projection;
    struct catvm_restoration restoration;
    const int ok = (
        catvm_seal(machine, program)
        && catvm_apply_f(machine)
        && catvm_apply_g(machine)
        && catvm_project_final(machine, &projection)
        && catvm_restore(
            machine,
            CATVM_RESTORE_CORRECT,
            &restoration
        )
        && restoration.used_actual_inverse
        && !restoration.used_snapshot_reload
        && restoration.carrier_within_tolerance
        && restoration.transient_state_exact
    );
    memset(&projection, 0, sizeof(projection));
    memset(&restoration, 0, sizeof(restoration));
    return ok;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        return 2;
    }
    char *end = NULL;
    errno = 0;
    const unsigned long cycles_value = strtoul(argv[1], &end, 10);
    if (
        errno != 0
        || end == argv[1]
        || *end != '\0'
        || cycles_value == 0UL
        || cycles_value > 10000000UL
    ) {
        return 2;
    }
    struct catvm_machine machine;
    if (!catvm_machine_init(&machine, CATVM_BACKEND_IN_PLACE)) {
        return 1;
    }
    for (size_t warmup = 0U; warmup < 1000U; ++warmup) {
        if (!run_transaction(
            &machine,
            (warmup & 1U) == 0U ? &PRIMARY : &REUSE
        )) {
            catvm_machine_destroy(&machine);
            return 1;
        }
    }
    memset(&machine.resources, 0, sizeof(machine.resources));
    machine.restoration_generation = 0U;
    struct timespec start;
    struct timespec finish;
    if (clock_gettime(CLOCK_MONOTONIC, &start) != 0) {
        catvm_machine_destroy(&machine);
        return 1;
    }
    for (size_t cycle = 0U; cycle < (size_t)cycles_value; ++cycle) {
        if (!run_transaction(
            &machine,
            (cycle & 1U) == 0U ? &PRIMARY : &REUSE
        )) {
            catvm_machine_destroy(&machine);
            return 1;
        }
    }
    if (clock_gettime(CLOCK_MONOTONIC, &finish) != 0) {
        catvm_machine_destroy(&machine);
        return 1;
    }
    const uint64_t elapsed =
        (uint64_t)(finish.tv_sec - start.tv_sec) * UINT64_C(1000000000)
        + (uint64_t)(finish.tv_nsec - start.tv_nsec);
    printf(
        "{\"result\":\"PASS\","
        "\"scenario\":\"warm-direct-process-phase\","
        "\"enforced_machine_boundary\":false,"
        "\"cycles\":%lu,\"warmup_cycles\":1000,"
        "\"transaction_wall_ns\":%llu,"
        "\"average_transaction_wall_ns\":%.3f,"
        "\"carrier_cells\":%u,\"carrier_creations\":1,"
        "\"native_compose_calls\":%llu,"
        "\"native_intersection_calls\":%llu,"
        "\"native_symbol_products\":%llu,"
        "\"phase_cell_updates\":%llu,"
        "\"boundary_decodes\":%llu,"
        "\"inverse_factor_recomputations\":%llu,"
        "\"snapshot_bytes_written\":%llu,"
        "\"snapshot_bytes_reloaded\":%llu,"
        "\"restoration_cell_checks\":%llu,"
        "\"maximum_restoration_error\":%.12g}\n",
        cycles_value,
        (unsigned long long)elapsed,
        (double)elapsed / (double)cycles_value,
        CATVM_CARRIER_CELLS,
        (unsigned long long)machine.resources.native_compose_calls,
        (unsigned long long)machine.resources.native_intersection_calls,
        (unsigned long long)machine.resources.native_symbol_products,
        (unsigned long long)machine.resources.phase_cell_updates,
        (unsigned long long)machine.resources.boundary_decodes,
        (unsigned long long)
            machine.resources.inverse_factor_recomputations,
        (unsigned long long)machine.resources.snapshot_bytes_written,
        (unsigned long long)machine.resources.snapshot_bytes_reloaded,
        (unsigned long long)machine.resources.restoration_cell_checks,
        catvm_carrier_maximum_error(&machine)
    );
    catvm_machine_destroy(&machine);
    return 0;
}
