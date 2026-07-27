#define _GNU_SOURCE

/*
 * Bounded CATVM phase backend.
 *
 * The native polynomial kernels are the branch-native four-phase
 * composition and intersection laws from algebraic_series_parallel_phase.c.
 * F composes two public relations into the resident Y cells.  G intersects
 * those actual cells with a third public relation and writes resident Z.
 * Only the final boundary cells are decoded.
 */

#include "catvm_phase_core.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#define INPUT_LEFT_START 0U
#define INPUT_RIGHT_START 4U
#define INPUT_CONSTRAINT_START 8U
#define INTERMEDIATE_Y_START 12U
#define FINAL_Z_START 16U
#define BOUNDARY_START 20U
#define PHASE_LOCK_PASSES 3U

static const uint64_t CATVM_LEASE_ID = UINT64_C(0x434154564d000001);

static void secure_zero(void *memory, size_t bytes) {
    volatile unsigned char *cursor = memory;
    while (bytes > 0U) {
        *cursor = 0U;
        ++cursor;
        --bytes;
    }
}

static int f3(int value) {
    int result = value % 3;
    if (result < 0) {
        result += 3;
    }
    return result;
}

static double complex unit(double complex value) {
    const double magnitude = cabs(value);
    if (!(magnitude > 0.0) || !isfinite(magnitude)) {
        return CMPLX(NAN, NAN);
    }
    return value / magnitude;
}

/*
 * Continuous three-well phase lock.  This is phase-label blind:
 * each legal cube root is a fixed point of unit(2z + conj(z)^2).
 */
static double complex lock_f3_phase(double complex value) {
    double complex locked = unit(value);
    for (size_t pass = 0U; pass < PHASE_LOCK_PASSES; ++pass) {
        locked = unit(2.0 * locked + conj(locked) * conj(locked));
    }
    return locked;
}

static double complex root3(int amount) {
    const int value = f3(amount);
    if (value == 0) {
        return 1.0 + 0.0 * I;
    }
    if (value == 1) {
        return -0.5 + 0.86602540378443864676 * I;
    }
    return -0.5 - 0.86602540378443864676 * I;
}

static double complex relative(
    const struct catvm_machine *machine,
    size_t cell
) {
    return machine->working[cell] * conj(machine->baseline[cell]);
}

static int multiply_cell(
    struct catvm_machine *machine,
    size_t cell,
    double complex factor
) {
    const double complex next = unit(relative(machine, cell) * factor);
    if (!isfinite(creal(next)) || !isfinite(cimag(next))) {
        return 0;
    }
    machine->working[cell] = next * machine->baseline[cell];
    ++machine->resources.phase_cell_updates;
    return 1;
}

static void read_poly(
    const struct catvm_machine *machine,
    size_t start,
    double complex output[CATVM_RELATION_CELLS]
) {
    for (size_t index = 0U; index < CATVM_RELATION_CELLS; ++index) {
        output[index] = relative(machine, start + index);
    }
}

static double complex symbol_product(
    struct catvm_machine *machine,
    double complex left,
    double complex right
) {
    ++machine->resources.native_symbol_products;
    const double complex left_squared = conj(left);
    const double complex right_squared = conj(right);
    const double complex product = left * right;
    return lock_f3_phase(
        (
            1.0
            + left
            + left_squared
            + right
            + right_squared
            + root3(2) * (product + conj(product))
            + root3(1) * (
                left * right_squared
                + left_squared * right
            )
        ) / 3.0
    );
}

static void poly_multiply(
    struct catvm_machine *machine,
    const double complex left[CATVM_RELATION_CELLS],
    const double complex right[CATVM_RELATION_CELLS],
    double complex output[CATVM_RELATION_CELLS]
) {
    for (size_t out = 0U; out < CATVM_RELATION_CELLS; ++out) {
        output[out] = 1.0 + 0.0 * I;
        for (size_t l = 0U; l < CATVM_RELATION_CELLS; ++l) {
            for (size_t r = 0U; r < CATVM_RELATION_CELLS; ++r) {
                if ((l | r) == out) {
                    output[out] = unit(
                        output[out]
                        * symbol_product(machine, left[l], right[r])
                    );
                }
            }
        }
    }
}

static void poly_add(
    const double complex left[CATVM_RELATION_CELLS],
    const double complex right[CATVM_RELATION_CELLS],
    double complex output[CATVM_RELATION_CELLS]
) {
    for (size_t index = 0U; index < CATVM_RELATION_CELLS; ++index) {
        output[index] = lock_f3_phase(left[index] * right[index]);
    }
}

static void exact_compose_factors(
    struct catvm_machine *machine,
    size_t left_start,
    size_t right_start,
    double complex output[CATVM_RELATION_CELLS]
) {
    ++machine->resources.native_compose_calls;
    double complex left[CATVM_RELATION_CELLS];
    double complex right[CATVM_RELATION_CELLS];
    read_poly(machine, left_start, left);
    read_poly(machine, right_start, right);
    const double complex zero = root3(0);
    double complex f0[CATVM_RELATION_CELLS] = {
        left[0], left[1], zero, zero
    };
    double complex f1[CATVM_RELATION_CELLS] = {
        unit(left[0] * left[2]),
        unit(left[1] * left[3]),
        zero,
        zero
    };
    double complex g0[CATVM_RELATION_CELLS] = {
        right[0], zero, right[2], zero
    };
    double complex g1[CATVM_RELATION_CELLS] = {
        unit(right[0] * right[1]),
        zero,
        unit(right[2] * right[3]),
        zero
    };
    double complex f0s[CATVM_RELATION_CELLS];
    double complex f1s[CATVM_RELATION_CELLS];
    double complex g0s[CATVM_RELATION_CELLS];
    double complex g1s[CATVM_RELATION_CELLS];
    double complex k0[CATVM_RELATION_CELLS];
    double complex k1[CATVM_RELATION_CELLS];
    poly_multiply(machine, f0, f0, f0s);
    poly_multiply(machine, f1, f1, f1s);
    poly_multiply(machine, g0, g0, g0s);
    poly_multiply(machine, g1, g1, g1s);
    poly_add(f0s, g0s, k0);
    poly_add(f1s, g1s, k1);
    poly_multiply(machine, k0, k1, output);
    secure_zero(left, sizeof(left));
    secure_zero(right, sizeof(right));
    secure_zero(f0, sizeof(f0));
    secure_zero(f1, sizeof(f1));
    secure_zero(g0, sizeof(g0));
    secure_zero(g1, sizeof(g1));
    secure_zero(f0s, sizeof(f0s));
    secure_zero(f1s, sizeof(f1s));
    secure_zero(g0s, sizeof(g0s));
    secure_zero(g1s, sizeof(g1s));
    secure_zero(k0, sizeof(k0));
    secure_zero(k1, sizeof(k1));
}

static void exact_intersection_factors(
    struct catvm_machine *machine,
    size_t left_start,
    size_t right_start,
    double complex output[CATVM_RELATION_CELLS]
) {
    ++machine->resources.native_intersection_calls;
    double complex left[CATVM_RELATION_CELLS];
    double complex right[CATVM_RELATION_CELLS];
    double complex left_squared[CATVM_RELATION_CELLS];
    double complex right_squared[CATVM_RELATION_CELLS];
    read_poly(machine, left_start, left);
    read_poly(machine, right_start, right);
    poly_multiply(machine, left, left, left_squared);
    poly_multiply(machine, right, right, right_squared);
    poly_add(left_squared, right_squared, output);
    secure_zero(left, sizeof(left));
    secure_zero(right, sizeof(right));
    secure_zero(left_squared, sizeof(left_squared));
    secure_zero(right_squared, sizeof(right_squared));
}

static int apply_factor(
    struct catvm_machine *machine,
    size_t output_start,
    const double complex factor[CATVM_RELATION_CELLS],
    int inverse
) {
    for (size_t index = 0U; index < CATVM_RELATION_CELLS; ++index) {
        if (!multiply_cell(
            machine,
            output_start + index,
            inverse ? conj(factor[index]) : factor[index]
        )) {
            return 0;
        }
    }
    return 1;
}

static int apply_encoding(
    struct catvm_machine *machine,
    size_t start,
    const int coefficient[CATVM_RELATION_CELLS],
    int inverse
) {
    for (size_t index = 0U; index < CATVM_RELATION_CELLS; ++index) {
        const double complex factor = root3(coefficient[index]);
        if (!multiply_cell(
            machine,
            start + index,
            inverse ? conj(factor) : factor
        )) {
            return 0;
        }
    }
    return 1;
}

static uint64_t hash_bytes(
    uint64_t hash,
    const unsigned char *bytes,
    size_t count
) {
    for (size_t index = 0U; index < count; ++index) {
        hash ^= (uint64_t)bytes[index];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static uint64_t digest_bytes(const void *memory, size_t bytes) {
    return hash_bytes(
        UINT64_C(14695981039346656037),
        memory,
        bytes
    );
}

static int bytes_are_zero(const void *memory, size_t bytes) {
    const unsigned char *cursor = memory;
    for (size_t index = 0U; index < bytes; ++index) {
        if (cursor[index] != 0U) {
            return 0;
        }
    }
    return 1;
}

static uint64_t expected_topology_digest(void) {
    static const char topology[] =
        "BOOLEAN_F3_COMPOSE_Y_THEN_INTERSECT_Z_24_CELLS";
    return digest_bytes(topology, sizeof(topology) - 1U);
}

/*
 * This is the sole phase-label decode in the core.  Its only caller is
 * catvm_project_final(), and that caller passes BOUNDARY_START.
 */
static int decode_final_root(
    double complex value,
    double *distance_out
) {
    int best = 0;
    double best_distance = INFINITY;
    for (int symbol = 0; symbol < 3; ++symbol) {
        const double distance = cabs(value - root3(symbol));
        if (distance < best_distance) {
            best = symbol;
            best_distance = distance;
        }
    }
    *distance_out = best_distance;
    return best;
}

static int remove_boundary_copy(struct catvm_machine *machine) {
    double complex factor[CATVM_RELATION_CELLS];
    read_poly(machine, FINAL_Z_START, factor);
    const int ok = apply_factor(machine, BOUNDARY_START, factor, 1);
    secure_zero(factor, sizeof(factor));
    machine->open_boundary = 0;
    return ok;
}

static int inverse_g(
    struct catvm_machine *machine,
    int wrong_factor
) {
    double complex factor[CATVM_RELATION_CELLS];
    double complex applied[CATVM_RELATION_CELLS];
    exact_intersection_factors(
        machine,
        INTERMEDIATE_Y_START,
        INPUT_CONSTRAINT_START,
        factor
    );
    ++machine->resources.inverse_factor_recomputations;
    if (wrong_factor) {
        for (size_t index = 0U; index < CATVM_RELATION_CELLS; ++index) {
            applied[index] =
                factor[(index + 1U) % CATVM_RELATION_CELLS];
        }
    } else {
        memcpy(applied, factor, sizeof(applied));
    }
    const int factor_ok = apply_factor(
        machine,
        FINAL_Z_START,
        applied,
        1
    );
    const int encoding_ok = apply_encoding(
        machine,
        INPUT_CONSTRAINT_START,
        machine->program.constraint,
        1
    );
    secure_zero(factor, sizeof(factor));
    secure_zero(applied, sizeof(applied));
    return factor_ok && encoding_ok;
}

static int inverse_f(struct catvm_machine *machine) {
    double complex factor[CATVM_RELATION_CELLS];
    exact_compose_factors(
        machine,
        INPUT_LEFT_START,
        INPUT_RIGHT_START,
        factor
    );
    ++machine->resources.inverse_factor_recomputations;
    const int factor_ok = apply_factor(
        machine,
        INTERMEDIATE_Y_START,
        factor,
        1
    );
    const int right_ok = apply_encoding(
        machine,
        INPUT_RIGHT_START,
        machine->program.right,
        1
    );
    const int left_ok = apply_encoding(
        machine,
        INPUT_LEFT_START,
        machine->program.left,
        1
    );
    secure_zero(factor, sizeof(factor));
    return factor_ok && right_ok && left_ok;
}

static void reset_discrete_transaction_state(
    struct catvm_machine *machine
) {
    secure_zero(&machine->program, sizeof(machine->program));
    machine->morphism_depth = 0U;
    machine->open_boundary = 0;
    machine->program_loaded = 0;
    machine->pending_operations = 0;
}

const char *catvm_state_name(enum catvm_machine_state state) {
    switch (state) {
        case CATVM_STATE_IDLE:
            return "IDLE";
        case CATVM_STATE_SEALED:
            return "SEALED";
        case CATVM_STATE_AFTER_F:
            return "Y_RESIDENT";
        case CATVM_STATE_AFTER_G:
            return "Z_RESIDENT";
        case CATVM_STATE_PROJECTED:
            return "FINAL_PROJECTED";
        case CATVM_STATE_FAILED:
            return "FAILED";
    }
    return "INVALID";
}

const char *catvm_backend_name(enum catvm_backend_kind backend) {
    switch (backend) {
        case CATVM_BACKEND_IN_PLACE:
            return "IN_PLACE_PHASE";
        case CATVM_BACKEND_SNAPSHOT:
            return "SNAPSHOT_BASELINE";
        case CATVM_BACKEND_NULL:
            return "NULL_CARRIER";
    }
    return "INVALID";
}

int catvm_machine_init(
    struct catvm_machine *machine,
    enum catvm_backend_kind backend
) {
    if (machine == NULL) {
        return 0;
    }
    memset(machine, 0, sizeof(*machine));
    machine->backend = backend;
    machine->state = CATVM_STATE_IDLE;
    machine->lease_id = CATVM_LEASE_ID;
    machine->topology_digest = expected_topology_digest();
    machine->carrier_enabled = backend != CATVM_BACKEND_NULL;
    if (!machine->carrier_enabled) {
        return 1;
    }
    if (backend == CATVM_BACKEND_SNAPSHOT) {
        const long page_size = sysconf(_SC_PAGESIZE);
        if (page_size <= 0) {
            return 0;
        }
        machine->snapshot_mapped_bytes = (size_t)page_size;
        machine->snapshot = mmap(
            NULL,
            machine->snapshot_mapped_bytes,
            PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS,
            -1,
            0
        );
        if (machine->snapshot == MAP_FAILED) {
            machine->snapshot = NULL;
            return 0;
        }
        if (
            mlock(machine->snapshot, machine->snapshot_mapped_bytes) != 0
            || madvise(
                machine->snapshot,
                machine->snapshot_mapped_bytes,
                MADV_DONTDUMP
            ) != 0
            || madvise(
                machine->snapshot,
                machine->snapshot_mapped_bytes,
                MADV_DONTFORK
            ) != 0
        ) {
            (void)munlock(
                machine->snapshot,
                machine->snapshot_mapped_bytes
            );
            (void)munmap(
                machine->snapshot,
                machine->snapshot_mapped_bytes
            );
            machine->snapshot = NULL;
            machine->snapshot_mapped_bytes = 0U;
            return 0;
        }
    }
    for (size_t index = 0U; index < CATVM_CARRIER_CELLS; ++index) {
        const double angle =
            0.229
            + 0.061 * (double)index
            + 0.014 * sin(0.27 * (double)index + 0.023 * 5303.0);
        machine->baseline[index] = cexp(I * angle);
        machine->working[index] = machine->baseline[index];
    }
    machine->baseline_digest = digest_bytes(
        machine->baseline,
        sizeof(machine->baseline)
    );
    machine->carrier_creation_count = 1U;
    return 1;
}

void catvm_machine_destroy(struct catvm_machine *machine) {
    if (machine == NULL) {
        return;
    }
    if (machine->snapshot != NULL) {
        secure_zero(
            machine->snapshot,
            CATVM_CARRIER_CELLS * sizeof(*machine->snapshot)
        );
        (void)munlock(
            machine->snapshot,
            machine->snapshot_mapped_bytes
        );
        (void)munmap(
            machine->snapshot,
            machine->snapshot_mapped_bytes
        );
        machine->snapshot = NULL;
    }
    secure_zero(machine, sizeof(*machine));
}

int catvm_program_valid(const struct catvm_program *program) {
    if (program == NULL) {
        return 0;
    }
    for (size_t index = 0U; index < CATVM_RELATION_CELLS; ++index) {
        if (
            program->left[index] < 0
            || program->left[index] > 2
            || program->right[index] < 0
            || program->right[index] > 2
            || program->constraint[index] < 0
            || program->constraint[index] > 2
        ) {
            return 0;
        }
    }
    return 1;
}

int catvm_seal(
    struct catvm_machine *machine,
    const struct catvm_program *program
) {
    if (
        machine == NULL
        || !machine->carrier_enabled
        || machine->state != CATVM_STATE_IDLE
        || !catvm_program_valid(program)
        || !catvm_machine_idle_exact(machine)
    ) {
        return 0;
    }
    if (machine->backend == CATVM_BACKEND_SNAPSHOT) {
        if (machine->snapshot == NULL || machine->snapshot_valid) {
            return 0;
        }
        memcpy(
            machine->snapshot,
            machine->working,
            CATVM_CARRIER_CELLS * sizeof(*machine->snapshot)
        );
        machine->resources.snapshot_bytes_written +=
            CATVM_CARRIER_CELLS * sizeof(*machine->snapshot);
        machine->snapshot_valid = 1;
    } else if (machine->snapshot != NULL) {
        return 0;
    }
    memcpy(&machine->program, program, sizeof(machine->program));
    machine->program_loaded = 1;
    machine->state = CATVM_STATE_SEALED;
    ++machine->resources.transactions_sealed;
    return 1;
}

int catvm_apply_f(struct catvm_machine *machine) {
    if (
        machine == NULL
        || !machine->carrier_enabled
        || machine->state != CATVM_STATE_SEALED
        || !machine->program_loaded
    ) {
        return 0;
    }
    machine->pending_operations = 1;
    double complex factor[CATVM_RELATION_CELLS];
    const int left_ok = apply_encoding(
        machine,
        INPUT_LEFT_START,
        machine->program.left,
        0
    );
    const int right_ok = apply_encoding(
        machine,
        INPUT_RIGHT_START,
        machine->program.right,
        0
    );
    exact_compose_factors(
        machine,
        INPUT_LEFT_START,
        INPUT_RIGHT_START,
        factor
    );
    const int output_ok = apply_factor(
        machine,
        INTERMEDIATE_Y_START,
        factor,
        0
    );
    secure_zero(factor, sizeof(factor));
    machine->pending_operations = 0;
    if (!(left_ok && right_ok && output_ok)) {
        machine->state = CATVM_STATE_FAILED;
        return 0;
    }
    machine->morphism_depth = 1U;
    machine->state = CATVM_STATE_AFTER_F;
    return 1;
}

int catvm_apply_g(struct catvm_machine *machine) {
    if (
        machine == NULL
        || !machine->carrier_enabled
        || machine->state != CATVM_STATE_AFTER_F
        || machine->morphism_depth != 1U
    ) {
        return 0;
    }
    machine->pending_operations = 1;
    double complex factor[CATVM_RELATION_CELLS];
    const int encoding_ok = apply_encoding(
        machine,
        INPUT_CONSTRAINT_START,
        machine->program.constraint,
        0
    );
    exact_intersection_factors(
        machine,
        INTERMEDIATE_Y_START,
        INPUT_CONSTRAINT_START,
        factor
    );
    const int output_ok = apply_factor(
        machine,
        FINAL_Z_START,
        factor,
        0
    );
    secure_zero(factor, sizeof(factor));
    machine->pending_operations = 0;
    if (!(encoding_ok && output_ok)) {
        machine->state = CATVM_STATE_FAILED;
        return 0;
    }
    machine->morphism_depth = 2U;
    machine->state = CATVM_STATE_AFTER_G;
    return 1;
}

int catvm_project_final(
    struct catvm_machine *machine,
    struct catvm_projection *projection
) {
    if (
        machine == NULL
        || projection == NULL
        || !machine->carrier_enabled
        || machine->state != CATVM_STATE_AFTER_G
        || machine->morphism_depth != 2U
    ) {
        return 0;
    }
    memset(projection, 0, sizeof(*projection));
    double complex factor[CATVM_RELATION_CELLS];
    read_poly(machine, FINAL_Z_START, factor);
    if (!apply_factor(machine, BOUNDARY_START, factor, 0)) {
        secure_zero(factor, sizeof(factor));
        machine->state = CATVM_STATE_FAILED;
        return 0;
    }
    secure_zero(factor, sizeof(factor));
    projection->hash = UINT64_C(14695981039346656037);
    static const unsigned char port_name[] = "FINAL_Z";
    projection->hash = hash_bytes(
        projection->hash,
        port_name,
        sizeof(port_name) - 1U
    );
    for (size_t index = 0U; index < CATVM_RELATION_CELLS; ++index) {
        double distance = 0.0;
        projection->coefficient[index] = decode_final_root(
            relative(machine, BOUNDARY_START + index),
            &distance
        );
        const unsigned char byte =
            (unsigned char)projection->coefficient[index];
        projection->hash = hash_bytes(projection->hash, &byte, 1U);
        if (distance > projection->maximum_root_error) {
            projection->maximum_root_error = distance;
        }
        ++machine->resources.boundary_decodes;
    }
    machine->open_boundary = 1;
    machine->state = CATVM_STATE_PROJECTED;
    return projection->maximum_root_error <= CATVM_ROOT_TOLERANCE;
}

int catvm_restore(
    struct catvm_machine *machine,
    enum catvm_restore_mode mode,
    struct catvm_restoration *restoration
) {
    if (
        machine == NULL
        || restoration == NULL
        || !machine->carrier_enabled
        || machine->state != CATVM_STATE_PROJECTED
        || !machine->open_boundary
    ) {
        return 0;
    }
    memset(restoration, 0, sizeof(*restoration));
    restoration->mode = mode;
    restoration->generation_before = machine->restoration_generation;
    restoration->lease_id_before = machine->lease_id;
    restoration->reordered_pair_applicable = 1;

    int operation_ok = 1;
    if (mode == CATVM_RESTORE_SNAPSHOT) {
        if (
            machine->backend != CATVM_BACKEND_SNAPSHOT
            || machine->snapshot == NULL
            || !machine->snapshot_valid
        ) {
            return 0;
        }
        memcpy(
            machine->working,
            machine->snapshot,
            CATVM_CARRIER_CELLS * sizeof(*machine->working)
        );
        machine->resources.snapshot_bytes_reloaded +=
            CATVM_CARRIER_CELLS * sizeof(*machine->working);
        secure_zero(
            machine->snapshot,
            CATVM_CARRIER_CELLS * sizeof(*machine->snapshot)
        );
        machine->snapshot_valid = 0;
        restoration->used_snapshot_reload = 1;
    } else {
        restoration->used_actual_inverse = 1;
        operation_ok = remove_boundary_copy(machine);
        if (mode == CATVM_RESTORE_REORDERED) {
            operation_ok = inverse_f(machine) && operation_ok;
            operation_ok = inverse_g(machine, 0) && operation_ok;
        } else {
            if (mode == CATVM_RESTORE_MISSING_G) {
                operation_ok = apply_encoding(
                    machine,
                    INPUT_CONSTRAINT_START,
                    machine->program.constraint,
                    1
                ) && operation_ok;
            } else {
                operation_ok = inverse_g(
                    machine,
                    mode == CATVM_RESTORE_WRONG_G
                ) && operation_ok;
            }
            operation_ok = inverse_f(machine) && operation_ok;
        }
    }

    reset_discrete_transaction_state(machine);
    ++machine->restoration_generation;
    restoration->maximum_abs_error = catvm_carrier_maximum_error(machine);
    restoration->carrier_integrity_max_abs =
        catvm_carrier_integrity(machine);
    restoration->maximum_transient_root_error =
        catvm_transient_maximum_root_error(machine);
    machine->resources.restoration_cell_checks += CATVM_CARRIER_CELLS;
    restoration->generation_after = machine->restoration_generation;
    restoration->lease_id_after = machine->lease_id;
    restoration->morphism_depth_after = machine->morphism_depth;
    restoration->open_boundary_after = machine->open_boundary;
    restoration->program_loaded_after = machine->program_loaded;
    restoration->pending_operations_after = machine->pending_operations;
    restoration->invariant_state_exact =
        restoration->lease_id_after == restoration->lease_id_before
        && machine->topology_digest == expected_topology_digest()
        && machine->baseline_digest
            == digest_bytes(
                machine->baseline,
                sizeof(machine->baseline)
            )
        && machine->carrier_creation_count == 1U
        && machine->backend != CATVM_BACKEND_NULL
        && machine->carrier_enabled;
    restoration->generation_transition_exact =
        restoration->generation_after
            == restoration->generation_before + 1U;
    restoration->carrier_within_tolerance =
        restoration->maximum_abs_error
            <= CATVM_RESTORATION_TOLERANCE
        && restoration->carrier_integrity_max_abs
            <= CATVM_RESTORATION_TOLERANCE;

    const int expected_success =
        mode == CATVM_RESTORE_CORRECT
        || mode == CATVM_RESTORE_SNAPSHOT;
    if (
        operation_ok
        && expected_success
        && restoration->carrier_within_tolerance
        && restoration->invariant_state_exact
        && restoration->generation_transition_exact
    ) {
        machine->state = CATVM_STATE_IDLE;
    } else {
        machine->state = CATVM_STATE_FAILED;
    }
    restoration->transient_state_exact =
        machine->state == CATVM_STATE_IDLE
        && machine->morphism_depth == 0U
        && !machine->open_boundary
        && !machine->program_loaded
        && bytes_are_zero(
            &machine->program,
            sizeof(machine->program)
        )
        && machine->pending_operations == 0
        && (
            (
                machine->backend == CATVM_BACKEND_IN_PLACE
                && machine->snapshot == NULL
            )
            || (
                machine->backend == CATVM_BACKEND_SNAPSHOT
                && machine->snapshot != NULL
                && !machine->snapshot_valid
            )
        );
    return operation_ok;
}

double catvm_carrier_maximum_error(const struct catvm_machine *machine) {
    if (machine == NULL || !machine->carrier_enabled) {
        return INFINITY;
    }
    double maximum = 0.0;
    for (size_t index = 0U; index < CATVM_CARRIER_CELLS; ++index) {
        const double error = cabs(
            machine->working[index] - machine->baseline[index]
        );
        if (!isfinite(error)) {
            return INFINITY;
        }
        if (error > maximum) {
            maximum = error;
        }
    }
    return maximum;
}

double catvm_carrier_integrity(const struct catvm_machine *machine) {
    if (machine == NULL || !machine->carrier_enabled) {
        return INFINITY;
    }
    double maximum = 0.0;
    for (size_t index = 0U; index < CATVM_CARRIER_CELLS; ++index) {
        const double error = fabs(cabs(machine->working[index]) - 1.0);
        if (!isfinite(error)) {
            return INFINITY;
        }
        if (error > maximum) {
            maximum = error;
        }
    }
    return maximum;
}

double catvm_transient_maximum_root_error(
    const struct catvm_machine *machine
) {
    if (machine == NULL || !machine->carrier_enabled) {
        return INFINITY;
    }
    double maximum = 0.0;
    for (size_t index = 0U; index < CATVM_CARRIER_CELLS; ++index) {
        const double error = cabs(relative(machine, index) - root3(0));
        if (!isfinite(error)) {
            return INFINITY;
        }
        if (error > maximum) {
            maximum = error;
        }
    }
    return maximum;
}

int catvm_machine_idle_exact(const struct catvm_machine *machine) {
    return (
        machine != NULL
        && machine->carrier_enabled
        && machine->state == CATVM_STATE_IDLE
        && machine->morphism_depth == 0U
        && !machine->open_boundary
        && !machine->program_loaded
        && bytes_are_zero(
            &machine->program,
            sizeof(machine->program)
        )
        && machine->pending_operations == 0
        && (
            (
                machine->backend == CATVM_BACKEND_IN_PLACE
                && machine->snapshot == NULL
            )
            || (
                machine->backend == CATVM_BACKEND_SNAPSHOT
                && machine->snapshot != NULL
                && !machine->snapshot_valid
            )
        )
        && catvm_carrier_maximum_error(machine)
            <= CATVM_RESTORATION_TOLERANCE
    );
}
