/*
 * Mutable CAT_CAS frontier: spatially layered phase Fredkin fabric.
 *
 * Each layer partitions 3*width data relations into disjoint triples. Public
 * phase-resident program enables and data controls drive PCSWAP interactions
 * without decoded feedback. A persistent pthread pool exposes the independent
 * gates in one layer to the available CPU cores; logical depth and total
 * conventional work are reported separately.
 */

#define _GNU_SOURCE
#include <complex.h>
#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define Q 3
#define RESTORATION_MAX 2.0e-11
#define LOCK_ERROR_TRIGGER 1.0e-12

struct parameters {
    size_t width;
    size_t depth;
    uint64_t cycles;
    int threads;
    uint64_t program_seed;
    size_t program_registers;
    size_t data_registers;
    size_t registers;
    uint64_t logical_depth;
    uint64_t total_gates;
};

struct carrier {
    size_t registers;
    double complex *baseline;
    double complex *working;
};

struct execution {
    uint64_t full_boundary_fnv1a64;
    uint64_t data_boundary_fnv1a64;
    size_t boundary_nonzero;
    double maximum_root_distance;
    double displacement_l2;
    double restoration_max_abs;
    uint64_t forward_ns;
    uint64_t inverse_ns;
};

static uint64_t monotonic_ns(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC_RAW, &value) != 0) {
        perror("clock_gettime");
        exit(2);
    }
    return (uint64_t)value.tv_sec * UINT64_C(1000000000)
        + (uint64_t)value.tv_nsec;
}

static uint64_t parse_u64(const char *text, const char *name) {
    char *end = NULL;
    if (text[0] < '0' || text[0] > '9') {
        fprintf(stderr, "%s must be a positive decimal integer\n", name);
        exit(2);
    }
    errno = 0;
    const unsigned long long value = strtoull(text, &end, 10);
    if (
        end == text
        || *end != '\0'
        || errno == ERANGE
        || value == 0
    ) {
        fprintf(stderr, "%s must be a positive decimal integer\n", name);
        exit(2);
    }
    return (uint64_t)value;
}

static uint64_t checked_product_u64(
    uint64_t left, uint64_t right, const char *name
) {
    if (left != 0 && right > UINT64_MAX / left) {
        fprintf(stderr, "%s overflows uint64\n", name);
        exit(2);
    }
    return left * right;
}

static size_t checked_product_size(
    size_t left, size_t right, const char *name
) {
    if (left != 0 && right > SIZE_MAX / left) {
        fprintf(stderr, "%s overflows size_t\n", name);
        exit(2);
    }
    return left * right;
}

static struct parameters parse_parameters(int argc, char **argv) {
    if (argc != 6) {
        fprintf(
            stderr,
            "usage: %s WIDTH DEPTH CYCLES THREADS PROGRAM_SEED\n",
            argv[0]
        );
        exit(2);
    }
    const uint64_t width_u64 = parse_u64(argv[1], "width");
    const uint64_t depth_u64 = parse_u64(argv[2], "depth");
    const uint64_t cycles = parse_u64(argv[3], "cycles");
    const uint64_t threads_u64 = parse_u64(argv[4], "threads");
    const uint64_t program_seed = parse_u64(argv[5], "program seed");
    if (
        width_u64 > SIZE_MAX
        || depth_u64 > SIZE_MAX
        || threads_u64 > INT32_MAX
    ) {
        fprintf(stderr, "parameter exceeds host representation\n");
        exit(2);
    }
    const size_t width = (size_t)width_u64;
    const size_t depth = (size_t)depth_u64;
    const size_t program_registers =
        checked_product_size(width, depth, "program registers");
    const size_t data_registers =
        checked_product_size(width, 3U, "data registers");
    if (program_registers > SIZE_MAX - data_registers) {
        fprintf(stderr, "total registers overflow size_t\n");
        exit(2);
    }
    const uint64_t logical_depth =
        checked_product_u64((uint64_t)depth, cycles, "logical depth");
    const uint64_t total_gates =
        checked_product_u64((uint64_t)width, logical_depth, "total gates");
    const size_t registers = program_registers + data_registers;
    if (registers > SIZE_MAX / sizeof(double complex)) {
        fprintf(stderr, "carrier allocation size overflows\n");
        exit(2);
    }
    return (struct parameters){
        .width = width,
        .depth = depth,
        .cycles = cycles,
        .threads = (int)threads_u64,
        .program_seed = program_seed,
        .program_registers = program_registers,
        .data_registers = data_registers,
        .registers = registers,
        .logical_depth = logical_depth,
        .total_gates = total_gates
    };
}

static double complex unit(double complex value) {
    const double magnitude = cabs(value);
    if (!(magnitude > 0.0) || !isfinite(magnitude)) {
        fprintf(stderr, "collapsed or nonfinite phase relation\n");
        exit(2);
    }
    return value / magnitude;
}

static double complex root_of_unity(int value) {
    int reduced = value % Q;
    if (reduced < 0) {
        reduced += Q;
    }
    if (reduced == 0) {
        return 1.0 + 0.0 * I;
    }
    if (reduced == 1) {
        return -0.5 + 0.86602540378443864676 * I;
    }
    return -0.5 - 0.86602540378443864676 * I;
}

static double complex phase_lock(double complex value) {
    double complex locked = value;
    const double complex cubed = locked * locked * locked;
    const double error_force = cimag(cubed);
    const double real_part = creal(locked);
    const double imag_part = cimag(locked);
    const double magnitude_error =
        real_part * real_part + imag_part * imag_part - 1.0;
    if (
        fabs(error_force) <= LOCK_ERROR_TRIGGER
        && fabs(magnitude_error) <= LOCK_ERROR_TRIGGER
    ) {
        return locked;
    }
    locked *= 1.0 - I * (error_force / 3.0);
    return unit(locked);
}

static double complex product_factor(
    double complex left, double complex right
) {
    /*
     * Every native relation is locked to the unit-circle F3 carrier, where
     * z^2 == conj(z). Preserve the phase polynomial while avoiding redundant
     * complex multiplications.
     */
    const double complex left_squared = conj(left);
    const double complex right_squared = conj(right);
    const double complex product = left * right;
    const double complex both_squared = conj(product);
    const double complex left_right_squared = left * right_squared;
    const double complex left_squared_right = left_squared * right;
    return (
        1.0
        + left
        + left_squared
        + right
        + right_squared
        + root_of_unity(2) * (product + both_squared)
        + root_of_unity(1)
            * (left_right_squared + left_squared_right)
    ) / 3.0;
}

static double complex boolean_one_indicator(double complex control) {
    const double complex squared_value = conj(control);
    const double complex squared_symbol =
        product_factor(control, control);
    return phase_lock(
        squared_value * squared_symbol * squared_symbol
    );
}

static double complex relation(
    const struct carrier *carrier, size_t index
) {
    return carrier->working[index] * conj(carrier->baseline[index]);
}

static void write_relation(
    struct carrier *carrier, size_t index, double complex value
) {
    carrier->working[index] =
        phase_lock(value) * carrier->baseline[index];
}

static void write_relation_unlocked(
    struct carrier *carrier, size_t index, double complex value
) {
    carrier->working[index] = value * carrier->baseline[index];
}

static int program_symbol(
    const struct parameters *parameters, size_t layer, size_t gate
) {
    const uint64_t mixed =
        (uint64_t)gate * UINT64_C(17)
        + (uint64_t)layer * UINT64_C(13)
        + parameters->program_seed;
    return (mixed & UINT64_C(1)) == 0 ? 0 : 1;
}

static int data_symbol(size_t index) {
    return (int)(
        (
            (uint64_t)index * (uint64_t)index
            + UINT64_C(2) * (uint64_t)index
            + UINT64_C(1)
        ) % Q
    );
}

static int input_symbol(
    const struct parameters *parameters, size_t index
) {
    if (index < parameters->program_registers) {
        const size_t layer = index / parameters->width;
        const size_t gate = index % parameters->width;
        return program_symbol(parameters, layer, gate);
    }
    return data_symbol(index - parameters->program_registers);
}

static struct carrier make_carrier(size_t registers, int identity) {
    struct carrier result = {
        .registers = registers,
        .baseline = calloc(registers, sizeof(double complex)),
        .working = calloc(registers, sizeof(double complex))
    };
    if (result.baseline == NULL || result.working == NULL) {
        fprintf(stderr, "carrier allocation failed\n");
        exit(2);
    }
    for (size_t index = 0; index < registers; ++index) {
        const double angle =
            0.149
            + 0.031 * (double)index
            + 0.017 * sin(0.11 * (double)index + identity * 0.09);
        const double complex rail = cexp(I * angle);
        result.baseline[index] = rail;
        result.working[index] = rail;
    }
    return result;
}

static struct carrier clone_carrier(const struct carrier *source) {
    struct carrier result = make_carrier(source->registers, 0);
    memcpy(
        result.baseline,
        source->baseline,
        source->registers * sizeof(double complex)
    );
    memcpy(
        result.working,
        source->working,
        source->registers * sizeof(double complex)
    );
    return result;
}

static void free_carrier(struct carrier *carrier) {
    free(carrier->baseline);
    free(carrier->working);
    carrier->baseline = NULL;
    carrier->working = NULL;
    carrier->registers = 0;
}

static void load_input(
    struct carrier *carrier, const struct parameters *parameters
) {
    for (size_t index = 0; index < parameters->registers; ++index) {
        carrier->working[index] *=
            root_of_unity(input_symbol(parameters, index));
    }
}

static void unload_input(
    struct carrier *carrier, const struct parameters *parameters
) {
    for (size_t index = 0; index < parameters->registers; ++index) {
        carrier->working[index] *=
            conj(root_of_unity(input_symbol(parameters, index)));
    }
}

static void gate_indices(
    const struct parameters *parameters,
    size_t layer,
    size_t gate,
    size_t *program,
    size_t *control,
    size_t *left,
    size_t *right
) {
    const size_t shift = layer % 3U;
    const size_t start =
        (3U * gate + shift) % parameters->data_registers;
    const size_t data_base = parameters->program_registers;
    *program = layer * parameters->width + gate;
    *control = data_base + start;
    *left = data_base + (start + 1U) % parameters->data_registers;
    *right = data_base + (start + 2U) % parameters->data_registers;
}

static void apply_pcswap(
    struct carrier *carrier,
    size_t program,
    size_t control_index,
    size_t left_index,
    size_t right_index
) {
    const double complex routed_control = product_factor(
        relation(carrier, program),
        relation(carrier, control_index)
    );
    const double complex control =
        boolean_one_indicator(routed_control);
    const double complex left = relation(carrier, left_index);
    const double complex right = relation(carrier, right_index);
    const double complex control_left =
        product_factor(control, left);
    const double complex control_right =
        product_factor(control, right);
    write_relation(
        carrier,
        left_index,
        left * control_right * conj(control_left)
    );
    write_relation(
        carrier,
        right_index,
        right * control_left * conj(control_right)
    );
}

static void apply_gate(
    struct carrier *carrier,
    const struct parameters *parameters,
    size_t layer,
    size_t gate
) {
    size_t program = 0;
    size_t control = 0;
    size_t left = 0;
    size_t right = 0;
    gate_indices(
        parameters,
        layer,
        gate,
        &program,
        &control,
        &left,
        &right
    );
    apply_pcswap(
        carrier,
        program,
        control,
        left,
        right
    );
}

struct worker_pool;

struct worker_context {
    struct worker_pool *pool;
    size_t worker_id;
};

struct worker_pool {
    size_t thread_count;
    pthread_t *threads;
    struct worker_context *contexts;
    pthread_mutex_t mutex;
    pthread_cond_t start_condition;
    pthread_cond_t done_condition;
    uint64_t generation;
    size_t completed;
    int stop;
    struct carrier *job_carrier;
    const struct parameters *job_parameters;
    size_t job_layer;
};

static void pthread_fatal(int error, const char *operation) {
    if (error != 0) {
        fprintf(stderr, "%s failed: %s\n", operation, strerror(error));
        exit(2);
    }
}

static void *worker_main(void *opaque) {
    struct worker_context *context = opaque;
    struct worker_pool *pool = context->pool;
    uint64_t observed_generation = 0;
    pthread_fatal(
        pthread_mutex_lock(&pool->mutex),
        "worker mutex lock"
    );
    for (;;) {
        while (
            !pool->stop
            && pool->generation == observed_generation
        ) {
            pthread_fatal(
                pthread_cond_wait(
                    &pool->start_condition,
                    &pool->mutex
                ),
                "worker start wait"
            );
        }
        if (pool->stop) {
            pthread_fatal(
                pthread_mutex_unlock(&pool->mutex),
                "worker stop unlock"
            );
            return NULL;
        }

        struct carrier *carrier = pool->job_carrier;
        const struct parameters *parameters =
            pool->job_parameters;
        const size_t layer = pool->job_layer;
        const uint64_t generation = pool->generation;
        const size_t quotient =
            parameters->width / pool->thread_count;
        const size_t remainder =
            parameters->width % pool->thread_count;
        const size_t extra_before =
            context->worker_id < remainder
                ? context->worker_id
                : remainder;
        const size_t begin =
            context->worker_id * quotient + extra_before;
        const size_t end =
            begin
            + quotient
            + (context->worker_id < remainder ? 1U : 0U);
        pthread_fatal(
            pthread_mutex_unlock(&pool->mutex),
            "worker job unlock"
        );

        for (size_t gate = begin; gate < end; ++gate) {
            apply_gate(carrier, parameters, layer, gate);
        }

        pthread_fatal(
            pthread_mutex_lock(&pool->mutex),
            "worker completion lock"
        );
        ++pool->completed;
        if (pool->completed == pool->thread_count) {
            pthread_fatal(
                pthread_cond_signal(&pool->done_condition),
                "worker completion signal"
            );
        }
        observed_generation = generation;
    }
}

static void initialize_worker_pool(
    struct worker_pool *pool, size_t thread_count
) {
    memset(pool, 0, sizeof(*pool));
    pool->thread_count = thread_count;
    pool->threads = calloc(thread_count, sizeof(pthread_t));
    pool->contexts = calloc(
        thread_count,
        sizeof(struct worker_context)
    );
    if (pool->threads == NULL || pool->contexts == NULL) {
        fprintf(stderr, "worker-pool allocation failed\n");
        exit(2);
    }
    pthread_fatal(
        pthread_mutex_init(&pool->mutex, NULL),
        "worker-pool mutex initialization"
    );
    pthread_fatal(
        pthread_cond_init(&pool->start_condition, NULL),
        "worker-pool start-condition initialization"
    );
    pthread_fatal(
        pthread_cond_init(&pool->done_condition, NULL),
        "worker-pool done-condition initialization"
    );
    for (size_t index = 0; index < thread_count; ++index) {
        pool->contexts[index] = (struct worker_context){
            .pool = pool,
            .worker_id = index
        };
        pthread_fatal(
            pthread_create(
                &pool->threads[index],
                NULL,
                worker_main,
                &pool->contexts[index]
            ),
            "worker creation"
        );
    }
}

static void dispatch_layer(
    struct worker_pool *pool,
    struct carrier *carrier,
    const struct parameters *parameters,
    size_t layer
) {
    pthread_fatal(
        pthread_mutex_lock(&pool->mutex),
        "dispatcher mutex lock"
    );
    pool->job_carrier = carrier;
    pool->job_parameters = parameters;
    pool->job_layer = layer;
    pool->completed = 0;
    ++pool->generation;
    pthread_fatal(
        pthread_cond_broadcast(&pool->start_condition),
        "dispatcher start broadcast"
    );
    while (pool->completed != pool->thread_count) {
        pthread_fatal(
            pthread_cond_wait(
                &pool->done_condition,
                &pool->mutex
            ),
            "dispatcher completion wait"
        );
    }
    pthread_fatal(
        pthread_mutex_unlock(&pool->mutex),
        "dispatcher mutex unlock"
    );
}

static void free_worker_pool(struct worker_pool *pool) {
    pthread_fatal(
        pthread_mutex_lock(&pool->mutex),
        "worker-pool stop lock"
    );
    pool->stop = 1;
    pthread_fatal(
        pthread_cond_broadcast(&pool->start_condition),
        "worker-pool stop broadcast"
    );
    pthread_fatal(
        pthread_mutex_unlock(&pool->mutex),
        "worker-pool stop unlock"
    );
    for (size_t index = 0; index < pool->thread_count; ++index) {
        pthread_fatal(
            pthread_join(pool->threads[index], NULL),
            "worker join"
        );
    }
    pthread_fatal(
        pthread_cond_destroy(&pool->done_condition),
        "worker-pool done-condition destruction"
    );
    pthread_fatal(
        pthread_cond_destroy(&pool->start_condition),
        "worker-pool start-condition destruction"
    );
    pthread_fatal(
        pthread_mutex_destroy(&pool->mutex),
        "worker-pool mutex destruction"
    );
    free(pool->contexts);
    free(pool->threads);
    memset(pool, 0, sizeof(*pool));
}

static void apply_forward_fabric(
    struct worker_pool *pool,
    struct carrier *carrier,
    const struct parameters *parameters
) {
    for (uint64_t cycle = 0; cycle < parameters->cycles; ++cycle) {
        for (size_t layer = 0; layer < parameters->depth; ++layer) {
            dispatch_layer(
                pool,
                carrier,
                parameters,
                layer
            );
        }
    }
}

static void apply_inverse_fabric(
    struct worker_pool *pool,
    struct carrier *carrier,
    const struct parameters *parameters
) {
    for (uint64_t cycle = parameters->cycles; cycle-- > 0;) {
        for (size_t layer = parameters->depth; layer-- > 0;) {
            dispatch_layer(
                pool,
                carrier,
                parameters,
                layer
            );
        }
    }
}

static double carrier_error(
    const struct carrier *left, const struct carrier *right
) {
    double maximum = 0.0;
    for (size_t index = 0; index < left->registers; ++index) {
        const double baseline_error =
            cabs(left->baseline[index] - right->baseline[index]);
        const double working_error =
            cabs(left->working[index] - right->working[index]);
        if (baseline_error > maximum) {
            maximum = baseline_error;
        }
        if (working_error > maximum) {
            maximum = working_error;
        }
    }
    return maximum;
}

static double carrier_displacement(
    const struct carrier *left, const struct carrier *right
) {
    double sum = 0.0;
    for (size_t index = 0; index < left->registers; ++index) {
        const double baseline_error =
            cabs(left->baseline[index] - right->baseline[index]);
        const double working_error =
            cabs(left->working[index] - right->working[index]);
        sum += baseline_error * baseline_error;
        sum += working_error * working_error;
    }
    return sqrt(sum);
}

static int decode_relation(double complex value, double *distance) {
    double best = INFINITY;
    int best_symbol = -1;
    for (int symbol = 0; symbol < Q; ++symbol) {
        const double candidate =
            cabs(unit(value) - root_of_unity(symbol));
        if (candidate < best) {
            best = candidate;
            best_symbol = symbol;
        }
    }
    *distance = best;
    return best_symbol;
}

static uint64_t fnv1a_update(uint64_t hash, unsigned char value) {
    hash ^= (uint64_t)value;
    hash *= UINT64_C(1099511628211);
    return hash;
}

static struct execution execute(
    struct worker_pool *pool,
    struct carrier *borrowed,
    const struct parameters *parameters,
    int inverse_mode
) {
    struct carrier working = clone_carrier(borrowed);
    double complex *latch = calloc(
        parameters->registers, sizeof(double complex)
    );
    if (latch == NULL) {
        fprintf(stderr, "boundary latch allocation failed\n");
        exit(2);
    }

    load_input(&working, parameters);
    const uint64_t forward_start = monotonic_ns();
    apply_forward_fabric(pool, &working, parameters);
    const uint64_t forward_ns = monotonic_ns() - forward_start;
    const double displacement =
        carrier_displacement(&working, borrowed);
    for (size_t index = 0; index < parameters->registers; ++index) {
        latch[index] = relation(&working, index);
    }

    const uint64_t inverse_start = monotonic_ns();
    if (inverse_mode != 2) {
        apply_inverse_fabric(pool, &working, parameters);
        if (inverse_mode == 1) {
            const size_t target = parameters->program_registers;
            write_relation_unlocked(
                &working,
                target,
                relation(&working, target) * root_of_unity(1)
            );
        }
        unload_input(&working, parameters);
    }
    const uint64_t inverse_ns = monotonic_ns() - inverse_start;

    uint64_t full_hash = UINT64_C(14695981039346656037);
    uint64_t data_hash = UINT64_C(14695981039346656037);
    size_t nonzero = 0;
    double maximum_root_distance = 0.0;
    for (size_t index = 0; index < parameters->registers; ++index) {
        double distance = 0.0;
        const int symbol = decode_relation(latch[index], &distance);
        const unsigned char byte = (unsigned char)symbol;
        full_hash = fnv1a_update(full_hash, byte);
        if (index >= parameters->program_registers) {
            data_hash = fnv1a_update(data_hash, byte);
        }
        if (symbol != 0) {
            ++nonzero;
        }
        if (distance > maximum_root_distance) {
            maximum_root_distance = distance;
        }
    }

    const double restoration = carrier_error(&working, borrowed);
    if (inverse_mode == 0) {
        memcpy(
            borrowed->baseline,
            working.baseline,
            parameters->registers * sizeof(double complex)
        );
        memcpy(
            borrowed->working,
            working.working,
            parameters->registers * sizeof(double complex)
        );
    }
    free(latch);
    free_carrier(&working);
    return (struct execution){
        .full_boundary_fnv1a64 = full_hash,
        .data_boundary_fnv1a64 = data_hash,
        .boundary_nonzero = nonzero,
        .maximum_root_distance = maximum_root_distance,
        .displacement_l2 = displacement,
        .restoration_max_abs = restoration,
        .forward_ns = forward_ns,
        .inverse_ns = inverse_ns
    };
}

static void print_execution(
    const char *mode,
    const struct parameters *parameters,
    const struct execution *execution
) {
    printf(
        "{\"mode\":\"%s\",\"width\":%zu,\"depth\":%zu,"
        "\"cycles\":%llu,\"threads\":%d,"
        "\"logical_depth\":%llu,\"total_gates\":%llu,"
        "\"program_phase_registers\":%zu,"
        "\"data_phase_registers\":%zu,\"registers\":%zu,"
        "\"resident_complex_cells\":%zu,"
        "\"full_boundary_fnv1a64\":\"%016llx\","
        "\"data_boundary_fnv1a64\":\"%016llx\","
        "\"boundary_nonzero\":%zu,"
        "\"root_distance_max\":%.12g,"
        "\"displacement_l2\":%.12g,"
        "\"restoration_max_abs\":%.12g,"
        "\"restoration_pass\":%s,"
        "\"forward_ns\":%llu,\"inverse_ns\":%llu}\n",
        mode,
        parameters->width,
        parameters->depth,
        (unsigned long long)parameters->cycles,
        parameters->threads,
        (unsigned long long)parameters->logical_depth,
        (unsigned long long)parameters->total_gates,
        parameters->program_registers,
        parameters->data_registers,
        parameters->registers,
        2U * parameters->registers,
        (unsigned long long)execution->full_boundary_fnv1a64,
        (unsigned long long)execution->data_boundary_fnv1a64,
        execution->boundary_nonzero,
        execution->maximum_root_distance,
        execution->displacement_l2,
        execution->restoration_max_abs,
        execution->restoration_max_abs <= RESTORATION_MAX
            ? "true"
            : "false",
        (unsigned long long)execution->forward_ns,
        (unsigned long long)execution->inverse_ns
    );
}

int main(int argc, char **argv) {
    struct parameters parameters = parse_parameters(argc, argv);
    struct worker_pool pool;
    initialize_worker_pool(
        &pool,
        (size_t)parameters.threads
    );

    struct carrier borrowed = make_carrier(parameters.registers, 193);
    const struct execution first =
        execute(&pool, &borrowed, &parameters, 0);
    print_execution("spatial-phase", &parameters, &first);
    const struct execution reuse =
        execute(&pool, &borrowed, &parameters, 0);
    print_execution("actual-restored-reuse", &parameters, &reuse);

    struct parameters variant = parameters;
    variant.program_seed += UINT64_C(1);
    const struct execution cross_program =
        execute(&pool, &borrowed, &variant, 0);
    print_execution("restored-cross-program-reuse", &variant, &cross_program);

    const struct execution wrong =
        execute(&pool, &borrowed, &parameters, 1);
    print_execution("wrong-inverse", &parameters, &wrong);
    const struct execution omitted =
        execute(&pool, &borrowed, &parameters, 2);
    print_execution("omitted-inverse", &parameters, &omitted);

    const int program_variant_changes_data =
        first.data_boundary_fnv1a64
        != cross_program.data_boundary_fnv1a64;
    printf(
        "{\"mode\":\"controls\","
        "\"program_variant_changes_data\":%s,"
        "\"process_exit_requires_sensitivity\":false}\n",
        program_variant_changes_data ? "true" : "false"
    );

    const int result = (
        first.restoration_max_abs <= RESTORATION_MAX
        && reuse.restoration_max_abs <= RESTORATION_MAX
        && cross_program.restoration_max_abs <= RESTORATION_MAX
        && wrong.restoration_max_abs > RESTORATION_MAX
        && omitted.restoration_max_abs > RESTORATION_MAX
        && first.full_boundary_fnv1a64 == reuse.full_boundary_fnv1a64
        && first.data_boundary_fnv1a64 == reuse.data_boundary_fnv1a64
    ) ? 0 : 1;
    free_carrier(&borrowed);
    free_worker_pool(&pool);
    return result;
}
