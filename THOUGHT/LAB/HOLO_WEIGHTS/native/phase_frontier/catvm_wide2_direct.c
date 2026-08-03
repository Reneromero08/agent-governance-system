#define _POSIX_C_SOURCE 200809L

/*
 * Warm direct-process comparison for the CATVM width-two backend.
 * This has no enforced machine boundary and cannot support the CATVM claim.
 */

#include "catvm_wide2_core.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static const struct catvm_program primary = {
    .left = {0,1,1,0,1,1,0,0,1,0,1,0,0,0,0,0},
    .right = {1,1,2,0,1,1,0,0,2,0,2,0,0,0,0,0},
    .constraint = {0,1,1,0,1,0,1,0,1,1,0,0,0,0,0,0}
};

static const struct catvm_program reuse = {
    .left = {0,1,1,0,1,1,0,0,1,0,1,0,0,0,0,0},
    .right = {1,1,2,0,1,1,0,0,2,0,2,0,0,0,0,0},
    .constraint = {0,1,1,0,1,1,0,0,1,0,1,0,0,0,0,0}
};

int main(int argc, char **argv) {
    if (argc != 2) {
        return 2;
    }
    errno = 0;
    char *end = NULL;
    const unsigned long cycles = strtoul(argv[1], &end, 10);
    if (
        errno != 0
        || end == argv[1]
        || *end != '\0'
        || cycles == 0UL
        || cycles > 100000UL
    ) {
        return 2;
    }
    struct catvm_machine machine;
    if (!catvm_machine_init(&machine, CATVM_BACKEND_IN_PLACE)) {
        return 1;
    }
    struct timespec start;
    struct timespec finish;
    struct catvm_projection projection;
    struct catvm_restoration restoration;
    double maximum_restoration_error = 0.0;
    if (clock_gettime(CLOCK_MONOTONIC, &start) != 0) {
        catvm_machine_destroy(&machine);
        return 1;
    }
    for (unsigned long cycle = 0UL; cycle < cycles; ++cycle) {
        const struct catvm_program *program =
            (cycle & 1UL) == 0UL ? &primary : &reuse;
        if (
            !catvm_seal(&machine, program)
            || !catvm_apply_f(&machine)
            || !catvm_apply_g(&machine)
            || !catvm_project_final(&machine, &projection)
            || !catvm_restore(
                &machine,
                CATVM_RESTORE_CORRECT,
                &restoration
            )
            || !restoration.transient_state_exact
            || !restoration.carrier_within_tolerance
        ) {
            catvm_machine_destroy(&machine);
            return 1;
        }
        if (
            restoration.maximum_abs_error
                > maximum_restoration_error
        ) {
            maximum_restoration_error =
                restoration.maximum_abs_error;
        }
    }
    if (clock_gettime(CLOCK_MONOTONIC, &finish) != 0) {
        catvm_machine_destroy(&machine);
        return 1;
    }
    const uint64_t elapsed =
        (uint64_t)(finish.tv_sec - start.tv_sec)
            * UINT64_C(1000000000)
        + (uint64_t)(finish.tv_nsec - start.tv_nsec);
    printf(
        "{\"result\":\"PASS\","
        "\"scenario\":\"warm-direct-width2-phase\","
        "\"enforced_machine_boundary\":false,"
        "\"cycles\":%lu,\"carrier_creations\":%llu,"
        "\"carrier_cells\":96,\"workspace_complex_values\":240,"
        "\"native_contract2_calls\":%llu,"
        "\"native_symbol_products\":%llu,"
        "\"coefficient_accumulation_additions\":%llu,"
        "\"restriction_and_intersection_additions\":%llu,"
        "\"phase_cell_updates\":%llu,"
        "\"boundary_decodes\":%llu,"
        "\"inverse_factor_recomputations\":%llu,"
        "\"snapshot_bytes_written\":%llu,"
        "\"snapshot_bytes_reloaded\":%llu,"
        "\"restoration_cell_checks\":%llu,"
        "\"maximum_restoration_error\":%.12g,"
        "\"transaction_wall_ns\":%llu,"
        "\"average_transaction_wall_ns\":%.3f}\n",
        cycles,
        (unsigned long long)machine.carrier_creation_count,
        (unsigned long long)machine.resources.native_contract2_calls,
        (unsigned long long)machine.resources.native_symbol_products,
        (unsigned long long)
            machine.resources.coefficient_accumulation_additions,
        (unsigned long long)
            machine.resources.restriction_and_intersection_additions,
        (unsigned long long)machine.resources.phase_cell_updates,
        (unsigned long long)machine.resources.boundary_decodes,
        (unsigned long long)
            machine.resources.inverse_factor_recomputations,
        (unsigned long long)machine.resources.snapshot_bytes_written,
        (unsigned long long)machine.resources.snapshot_bytes_reloaded,
        (unsigned long long)machine.resources.restoration_cell_checks,
        maximum_restoration_error,
        (unsigned long long)elapsed,
        (double)elapsed / (double)cycles
    );
    catvm_machine_destroy(&machine);
    return maximum_restoration_error <= CATVM_RESTORATION_TOLERANCE
        ? 0
        : 1;
}
