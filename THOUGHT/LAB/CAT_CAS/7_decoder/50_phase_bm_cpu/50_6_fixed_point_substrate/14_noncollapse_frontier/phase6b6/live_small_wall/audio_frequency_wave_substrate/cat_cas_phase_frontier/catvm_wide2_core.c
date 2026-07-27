#define _GNU_SOURCE

/*
 * Fixed width-two CATVM backend.
 *
 * F and G are the reviewed CONTRACT2 law.  All 240 complex workspace values
 * live inside the locked machine mapping and are wiped after every call.
 * Only the final 16-cell boundary is decoded.
 */

#include "catvm_wide2_core.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#define INPUT_LEFT_START 0U
#define INPUT_RIGHT_START 16U
#define INPUT_CONSTRAINT_START 32U
#define INTERMEDIATE_Y_START 48U
#define FINAL_Z_START 64U
#define BOUNDARY_START 80U
#define PHASE_LOCK_PASSES 3U

#define WS_LEFT 0U
#define WS_RIGHT 16U
#define WS_LEFT_SQUARED 32U
#define WS_RIGHT_SQUARED 48U
#define WS_INTERSECTION 64U
#define WS_FIRST_NORM 128U
#define WS_OUTPUT 160U
#define WS_RESTRICT_ZERO 176U
#define WS_RESTRICT_ONE 208U

static const uint64_t CATVM_LEASE_ID = UINT64_C(0x434154564d020001);
static const uint64_t MORPHISM_F = UINT64_C(0x434f4e5452414331);
static const uint64_t MORPHISM_G = UINT64_C(0x434f4e5452414332);

static void secure_zero(void *memory, size_t bytes) {
    volatile unsigned char *cursor = memory;
    while (bytes > 0U) {
        *cursor = 0U;
        ++cursor;
        --bytes;
    }
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

static int f3(int value) {
    int result = value % 3;
    return result < 0 ? result + 3 : result;
}

static double complex unit(double complex value) {
    const double magnitude = cabs(value);
    if (!(magnitude > 0.0) || !isfinite(magnitude)) {
        return CMPLX(NAN, NAN);
    }
    return value / magnitude;
}

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

static uint64_t expected_topology_digest(void) {
    static const char topology[] =
        "BOOLEAN_F3_WIDTH2_CONTRACT2_CHAIN_96_CELLS";
    return digest_bytes(topology, sizeof(topology) - 1U);
}

static uint64_t expected_morphism_digest(void) {
    static const uint64_t morphism[2] = {MORPHISM_F, MORPHISM_G};
    return digest_bytes(morphism, sizeof(morphism));
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

static void read_relation(
    const struct catvm_machine *machine,
    size_t start,
    double complex *output
) {
    for (size_t index = 0U; index < CATVM_RELATION_CELLS; ++index) {
        output[index] = relative(machine, start + index);
    }
}

static void wide_poly_multiply(
    struct catvm_machine *machine,
    const double complex *left,
    const double complex *right,
    double complex *output
) {
    for (size_t out = 0U; out < CATVM_RELATION_CELLS; ++out) {
        output[out] = root3(0);
        for (size_t l = 0U; l < CATVM_RELATION_CELLS; ++l) {
            for (size_t r = 0U; r < CATVM_RELATION_CELLS; ++r) {
                if ((l | r) == out) {
                    output[out] = lock_f3_phase(
                        output[out]
                        * symbol_product(machine, left[l], right[r])
                    );
                    ++machine->resources
                        .coefficient_accumulation_additions;
                }
            }
        }
    }
}

static size_t insert_zero_bit(size_t mask, size_t bit) {
    const size_t lower = ((size_t)1U << bit) - 1U;
    return (mask & lower) | ((mask & ~lower) << 1U);
}

static void boolean_norm(
    struct catvm_machine *machine,
    const double complex *input,
    size_t input_variables,
    size_t eliminated_bit,
    double complex *output
) {
    const size_t output_count =
        (size_t)1U << (input_variables - 1U);
    double complex *restrict_zero =
        &machine->contract2_workspace[WS_RESTRICT_ZERO];
    double complex *restrict_one =
        &machine->contract2_workspace[WS_RESTRICT_ONE];
    for (size_t index = 0U; index < output_count; ++index) {
        const size_t without = insert_zero_bit(index, eliminated_bit);
        restrict_zero[index] = input[without];
        restrict_one[index] = lock_f3_phase(
            input[without]
            * input[without | ((size_t)1U << eliminated_bit)]
        );
        ++machine->resources.restriction_and_intersection_additions;
    }
    for (size_t out = 0U; out < output_count; ++out) {
        output[out] = root3(0);
        for (size_t left = 0U; left < output_count; ++left) {
            for (size_t right = 0U; right < output_count; ++right) {
                if ((left | right) == out) {
                    output[out] = lock_f3_phase(
                        output[out]
                        * symbol_product(
                            machine,
                            restrict_zero[left],
                            restrict_one[right]
                        )
                    );
                    ++machine->resources
                        .coefficient_accumulation_additions;
                }
            }
        }
    }
}

static int workspace_is_zero(const struct catvm_machine *machine) {
    return bytes_are_zero(
        machine->contract2_workspace,
        sizeof(machine->contract2_workspace)
    );
}

static int contract2_apply(
    struct catvm_machine *machine,
    size_t left_start,
    size_t right_start,
    size_t output_start,
    int inverse,
    int wrong_inverse
) {
    if (!workspace_is_zero(machine)) {
        return 0;
    }
    double complex *workspace = machine->contract2_workspace;
    double complex *left = &workspace[WS_LEFT];
    double complex *right = &workspace[WS_RIGHT];
    double complex *left_squared = &workspace[WS_LEFT_SQUARED];
    double complex *right_squared = &workspace[WS_RIGHT_SQUARED];
    double complex *intersection = &workspace[WS_INTERSECTION];
    double complex *first_norm = &workspace[WS_FIRST_NORM];
    double complex *output = &workspace[WS_OUTPUT];

    ++machine->resources.native_contract2_calls;
    read_relation(machine, left_start, left);
    read_relation(machine, right_start, right);
    wide_poly_multiply(machine, left, left, left_squared);
    wide_poly_multiply(machine, right, right, right_squared);

    for (size_t mask = 0U; mask < 64U; ++mask) {
        double complex f = root3(0);
        double complex g = root3(0);
        if ((mask & (size_t)0x30U) == 0U) {
            f = left_squared[mask & 15U];
        }
        if ((mask & (size_t)0x03U) == 0U) {
            const size_t local_g = (
                ((mask >> 2U) & 3U)
                | (((mask >> 4U) & 3U) << 2U)
            );
            g = right_squared[local_g];
        }
        intersection[mask] = lock_f3_phase(f * g);
        ++machine->resources.restriction_and_intersection_additions;
    }
    boolean_norm(machine, intersection, 6U, 2U, first_norm);
    boolean_norm(machine, first_norm, 5U, 2U, output);

    int ok = 1;
    for (size_t index = 0U; index < CATVM_RELATION_CELLS; ++index) {
        const size_t source = wrong_inverse
            ? (index + 1U) % CATVM_RELATION_CELLS
            : index;
        ok = multiply_cell(
            machine,
            output_start + index,
            inverse ? conj(output[source]) : output[source]
        ) && ok;
    }
    secure_zero(
        machine->contract2_workspace,
        sizeof(machine->contract2_workspace)
    );
    return ok && workspace_is_zero(machine);
}

static int apply_encoding(
    struct catvm_machine *machine,
    size_t start,
    const int coefficient[CATVM_RELATION_CELLS],
    int inverse
) {
    int ok = 1;
    for (size_t index = 0U; index < CATVM_RELATION_CELLS; ++index) {
        const double complex factor = root3(coefficient[index]);
        ok = multiply_cell(
            machine,
            start + index,
            inverse ? conj(factor) : factor
        ) && ok;
    }
    return ok;
}

/*
 * This is the sole phase-label decode.  Its only caller projects cells
 * BOUNDARY_START..BOUNDARY_START+15.
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
    int ok = 1;
    for (size_t index = 0U; index < CATVM_RELATION_CELLS; ++index) {
        ok = multiply_cell(
            machine,
            BOUNDARY_START + index,
            conj(relative(machine, FINAL_Z_START + index))
        ) && ok;
    }
    machine->open_boundary = 0;
    return ok;
}

static int inverse_g(
    struct catvm_machine *machine,
    int wrong_factor
) {
    ++machine->resources.inverse_factor_recomputations;
    const int factor_ok = contract2_apply(
        machine,
        INTERMEDIATE_Y_START,
        INPUT_CONSTRAINT_START,
        FINAL_Z_START,
        1,
        wrong_factor
    );
    const int encoding_ok = apply_encoding(
        machine,
        INPUT_CONSTRAINT_START,
        machine->program.constraint,
        1
    );
    return factor_ok && encoding_ok;
}

static int inverse_f(struct catvm_machine *machine) {
    ++machine->resources.inverse_factor_recomputations;
    const int factor_ok = contract2_apply(
        machine,
        INPUT_LEFT_START,
        INPUT_RIGHT_START,
        INTERMEDIATE_Y_START,
        1,
        0
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
    return factor_ok && right_ok && left_ok;
}

static void reset_discrete_transaction_state(
    struct catvm_machine *machine
) {
    secure_zero(&machine->program, sizeof(machine->program));
    secure_zero(machine->morphism_stack, sizeof(machine->morphism_stack));
    machine->morphism_depth = 0U;
    machine->resident_internal_messages = 0U;
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
    machine->compiled_morphism_digest = expected_morphism_digest();
    machine->carrier_enabled = backend != CATVM_BACKEND_NULL;
    if (!machine->carrier_enabled) {
        return 1;
    }
    if (backend == CATVM_BACKEND_SNAPSHOT) {
        const long page_size_long = sysconf(_SC_PAGESIZE);
        if (page_size_long <= 0) {
            return 0;
        }
        const size_t page_size = (size_t)page_size_long;
        const size_t snapshot_bytes =
            CATVM_CARRIER_CELLS * sizeof(*machine->snapshot);
        machine->snapshot_mapped_bytes = (
            (snapshot_bytes + page_size - 1U) / page_size
        ) * page_size;
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
            0.193
            + 0.047 * (double)index
            + 0.012 * sin(0.21 * (double)index + 0.019 * 6203.0);
        machine->baseline[index] = cexp(I * angle);
        machine->working[index] = machine->baseline[index];
    }
    machine->baseline_digest = digest_bytes(
        machine->baseline,
        sizeof(machine->baseline)
    );
    machine->carrier_creation_count = 1U;
    return workspace_is_zero(machine);
}

void catvm_machine_destroy(struct catvm_machine *machine) {
    if (machine == NULL) {
        return;
    }
    if (machine->snapshot != NULL) {
        secure_zero(
            machine->snapshot,
            machine->snapshot_mapped_bytes
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
        || !workspace_is_zero(machine)
    ) {
        return 0;
    }
    machine->pending_operations = 1;
    const int left_ok = apply_encoding(
        machine, INPUT_LEFT_START, machine->program.left, 0
    );
    const int right_ok = apply_encoding(
        machine, INPUT_RIGHT_START, machine->program.right, 0
    );
    const int output_ok = contract2_apply(
        machine,
        INPUT_LEFT_START,
        INPUT_RIGHT_START,
        INTERMEDIATE_Y_START,
        0,
        0
    );
    machine->pending_operations = 0;
    if (!(left_ok && right_ok && output_ok)) {
        machine->state = CATVM_STATE_FAILED;
        return 0;
    }
    machine->morphism_stack[0] = MORPHISM_F;
    machine->morphism_depth = 1U;
    machine->resident_internal_messages = 1U;
    machine->state = CATVM_STATE_AFTER_F;
    return 1;
}

int catvm_apply_g(struct catvm_machine *machine) {
    if (
        machine == NULL
        || !machine->carrier_enabled
        || machine->state != CATVM_STATE_AFTER_F
        || machine->morphism_depth != 1U
        || machine->resident_internal_messages != 1U
        || !workspace_is_zero(machine)
    ) {
        return 0;
    }
    machine->pending_operations = 1;
    const int encoding_ok = apply_encoding(
        machine,
        INPUT_CONSTRAINT_START,
        machine->program.constraint,
        0
    );
    const int output_ok = contract2_apply(
        machine,
        INTERMEDIATE_Y_START,
        INPUT_CONSTRAINT_START,
        FINAL_Z_START,
        0,
        0
    );
    machine->pending_operations = 0;
    if (!(encoding_ok && output_ok)) {
        machine->state = CATVM_STATE_FAILED;
        return 0;
    }
    machine->morphism_stack[1] = MORPHISM_G;
    machine->morphism_depth = 2U;
    machine->resident_internal_messages = 2U;
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
        || machine->resident_internal_messages != 2U
        || !workspace_is_zero(machine)
    ) {
        return 0;
    }
    memset(projection, 0, sizeof(*projection));
    for (size_t index = 0U; index < CATVM_RELATION_CELLS; ++index) {
        if (!multiply_cell(
            machine,
            BOUNDARY_START + index,
            relative(machine, FINAL_Z_START + index)
        )) {
            machine->state = CATVM_STATE_FAILED;
            return 0;
        }
    }
    projection->hash = UINT64_C(14695981039346656037);
    static const unsigned char port_name[] = "FINAL_WIDTH2_Z";
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
        || !workspace_is_zero(machine)
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
            machine->snapshot_mapped_bytes
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
    restoration->maximum_abs_error =
        catvm_carrier_maximum_error(machine);
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
        && machine->compiled_morphism_digest
            == expected_morphism_digest()
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
        restoration->maximum_abs_error <= CATVM_RESTORATION_TOLERANCE
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
        && machine->resident_internal_messages == 0U
        && bytes_are_zero(
            machine->morphism_stack,
            sizeof(machine->morphism_stack)
        )
        && !machine->open_boundary
        && !machine->program_loaded
        && bytes_are_zero(&machine->program, sizeof(machine->program))
        && !machine->pending_operations
        && workspace_is_zero(machine)
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
    restoration->workspace_cleared = workspace_is_zero(machine);
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
        && machine->resident_internal_messages == 0U
        && bytes_are_zero(
            machine->morphism_stack,
            sizeof(machine->morphism_stack)
        )
        && !machine->open_boundary
        && !machine->program_loaded
        && bytes_are_zero(&machine->program, sizeof(machine->program))
        && !machine->pending_operations
        && workspace_is_zero(machine)
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
