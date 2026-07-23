/*
 * Independent compact reference for spatial_phase_fredkin.c.
 *
 * This executable contains no complex carrier code and is never linked into
 * the native engine. It evaluates the same public geometry afterward.
 */

#define _GNU_SOURCE
#include <errno.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define Q 3

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
    return (struct parameters){
        .width = width,
        .depth = depth,
        .cycles = cycles,
        .threads = (int)threads_u64,
        .program_seed = program_seed,
        .program_registers = program_registers,
        .data_registers = data_registers,
        .registers = program_registers + data_registers,
        .logical_depth = logical_depth,
        .total_gates = total_gates
    };
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

static void load_input(
    uint8_t *state, const struct parameters *parameters
) {
    for (size_t index = 0; index < parameters->program_registers; ++index) {
        const size_t layer = index / parameters->width;
        const size_t gate = index % parameters->width;
        state[index] = (uint8_t)program_symbol(parameters, layer, gate);
    }
    for (size_t index = 0; index < parameters->data_registers; ++index) {
        state[parameters->program_registers + index] =
            (uint8_t)data_symbol(index);
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

static void apply_gate(
    uint8_t *state,
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
    if (
        (
            state[program] * state[control]
        ) % Q == 1U
    ) {
        const uint8_t temporary = state[left];
        state[left] = state[right];
        state[right] = temporary;
    }
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
    uint8_t *job_state;
    const struct parameters *job_parameters;
    size_t job_layer;
};

static void pthread_fatal(int error, const char *operation) {
    if (error != 0) {
        fprintf(stderr, "%s failed\n", operation);
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

        uint8_t *state = pool->job_state;
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
            apply_gate(state, parameters, layer, gate);
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
    *pool = (struct worker_pool){0};
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
    uint8_t *state,
    const struct parameters *parameters,
    size_t layer
) {
    pthread_fatal(
        pthread_mutex_lock(&pool->mutex),
        "dispatcher mutex lock"
    );
    pool->job_state = state;
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
}

static void apply_forward_fabric(
    struct worker_pool *pool,
    uint8_t *state,
    const struct parameters *parameters
) {
    for (uint64_t cycle = 0; cycle < parameters->cycles; ++cycle) {
        for (size_t layer = 0; layer < parameters->depth; ++layer) {
            dispatch_layer(pool, state, parameters, layer);
        }
    }
}

static uint64_t fnv1a_update(uint64_t hash, unsigned char value) {
    hash ^= (uint64_t)value;
    hash *= UINT64_C(1099511628211);
    return hash;
}

int main(int argc, char **argv) {
    const struct parameters parameters = parse_parameters(argc, argv);
    struct worker_pool pool;
    initialize_worker_pool(
        &pool,
        (size_t)parameters.threads
    );
    uint8_t *state = calloc(parameters.registers, sizeof(uint8_t));
    if (state == NULL) {
        fprintf(stderr, "reference allocation failed\n");
        return 2;
    }
    load_input(state, &parameters);

    const uint64_t start = monotonic_ns();
    apply_forward_fabric(&pool, state, &parameters);
    const uint64_t forward_ns = monotonic_ns() - start;

    uint64_t full_hash = UINT64_C(14695981039346656037);
    uint64_t data_hash = UINT64_C(14695981039346656037);
    size_t nonzero = 0;
    for (size_t index = 0; index < parameters.registers; ++index) {
        full_hash = fnv1a_update(full_hash, state[index]);
        if (index >= parameters.program_registers) {
            data_hash = fnv1a_update(data_hash, state[index]);
        }
        if (state[index] != 0U) {
            ++nonzero;
        }
    }

    printf(
        "{\"width\":%zu,\"depth\":%zu,\"cycles\":%llu,"
        "\"threads\":%d,\"logical_depth\":%llu,"
        "\"total_gates\":%llu,\"program_phase_registers\":%zu,"
        "\"data_phase_registers\":%zu,\"registers\":%zu,"
        "\"full_boundary_fnv1a64\":\"%016llx\","
        "\"data_boundary_fnv1a64\":\"%016llx\","
        "\"boundary_nonzero\":%zu,\"forward_ns\":%llu}\n",
        parameters.width,
        parameters.depth,
        (unsigned long long)parameters.cycles,
        parameters.threads,
        (unsigned long long)parameters.logical_depth,
        (unsigned long long)parameters.total_gates,
        parameters.program_registers,
        parameters.data_registers,
        parameters.registers,
        (unsigned long long)full_hash,
        (unsigned long long)data_hash,
        nonzero,
        (unsigned long long)forward_ns
    );
    free(state);
    free_worker_pool(&pool);
    return 0;
}
