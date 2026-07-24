/*
 * Mutable CAT_CAS frontier: dependency-layered pthread phase VM.
 *
 * This executable reuses the reviewed complex phase algebra and strict public
 * .holo parser from streaming_phase_vm.c. It adds only a dependency scheduler
 * and a persistent pthread runtime. Instructions in one layer access disjoint
 * phase registers, so they commute and may evolve concurrently. The scheduler
 * inspects register topology only; it never evaluates phase data or results.
 */

#define main streaming_phase_vm_original_main
#include "streaming_phase_vm.c"
#undef main

#include <pthread.h>

#define PARALLEL_MIN_INSTRUCTIONS 256U

struct parallel_schedule {
    struct instruction *instructions;
    size_t *layer_offsets;
    size_t layers;
    size_t maximum_layer_width;
    size_t parallel_eligible_layers;
    uint64_t passes;
    uint64_t steps;
};

struct parallel_pool;

struct parallel_worker {
    struct parallel_pool *pool;
    size_t worker_id;
};

struct parallel_pool {
    size_t thread_count;
    pthread_t *threads;
    struct parallel_worker *workers;
    pthread_mutex_t mutex;
    pthread_cond_t start_condition;
    pthread_cond_t done_condition;
    uint64_t generation;
    size_t completed;
    int stop;
    int inverse;
    struct carrier *carrier;
    const struct parallel_schedule *schedule;
    size_t layer;
};

static void thread_fatal(int error, const char *operation) {
    if (error != 0) {
        fprintf(stderr, "%s failed: %s\n", operation, strerror(error));
        exit(2);
    }
}

static size_t raw_instruction_registers(
    const struct instruction *instruction,
    size_t output[4]
) {
    if (instruction->op == OP_ROT) {
        output[0] = instruction->target;
        return 1;
    }
    if (instruction->op == OP_ADD) {
        output[0] = instruction->a;
        output[1] = instruction->target;
        return 2;
    }
    if (instruction->op == OP_MULADD) {
        output[0] = instruction->a;
        output[1] = instruction->b;
        output[2] = instruction->target;
        return 3;
    }
    if (instruction->op == OP_SWAP) {
        output[0] = instruction->a;
        output[1] = instruction->b;
        return 2;
    }
    if (instruction->op == OP_CSWAP) {
        output[0] = instruction->target;
        output[1] = instruction->a;
        output[2] = instruction->b;
        return 3;
    }
    if (instruction->op == OP_PCSWAP) {
        output[0] = instruction->target;
        output[1] = instruction->a;
        output[2] = instruction->b;
        output[3] = instruction->c;
        return 4;
    }
    fprintf(stderr, "unsupported instruction in dependency scheduler\n");
    exit(2);
}

static size_t instruction_registers(
    const struct instruction *instruction,
    size_t output[4]
) {
    size_t raw[4] = {0, 0, 0, 0};
    const size_t raw_count =
        raw_instruction_registers(instruction, raw);
    size_t unique_count = 0;
    for (size_t item = 0; item < raw_count; ++item) {
        int present = 0;
        for (size_t prior = 0; prior < unique_count; ++prior) {
            if (output[prior] == raw[item]) {
                present = 1;
                break;
            }
        }
        if (!present) {
            output[unique_count++] = raw[item];
        }
    }
    return unique_count;
}

static void verify_schedule(
    const struct instruction *instructions,
    const size_t *offsets,
    size_t layers,
    size_t registers,
    size_t expected_instructions
) {
    if (offsets[layers] != expected_instructions) {
        fprintf(stderr, "dependency schedule lost an instruction\n");
        exit(2);
    }
    size_t *seen_layer = calloc(registers, sizeof(size_t));
    if (seen_layer == NULL) {
        fprintf(stderr, "schedule-verification allocation failed\n");
        exit(2);
    }
    for (size_t layer = 0; layer < layers; ++layer) {
        const size_t marker = layer + 1U;
        for (
            size_t index = offsets[layer];
            index < offsets[layer + 1U];
            ++index
        ) {
            size_t touched[4] = {0, 0, 0, 0};
            const size_t count = instruction_registers(
                &instructions[index],
                touched
            );
            for (size_t item = 0; item < count; ++item) {
                if (seen_layer[touched[item]] == marker) {
                    fprintf(
                        stderr,
                        "dependency layer contains a register collision\n"
                    );
                    exit(2);
                }
                seen_layer[touched[item]] = marker;
            }
        }
    }
    free(seen_layer);
}

static struct parallel_schedule make_schedule(
    const struct public_program *program
) {
    if (
        program->instruction_count == 0
        || program->instruction_count > SIZE_MAX
    ) {
        fprintf(stderr, "parallel program instruction count is invalid\n");
        exit(2);
    }
    const size_t instruction_count =
        (size_t)program->instruction_count;
    size_t *ready = calloc(program->registers, sizeof(size_t));
    size_t *assigned = calloc(instruction_count, sizeof(size_t));
    if (ready == NULL || assigned == NULL) {
        fprintf(stderr, "dependency-scheduler allocation failed\n");
        exit(2);
    }

    size_t layers = 0;
    for (size_t index = 0; index < instruction_count; ++index) {
        size_t registers[4] = {0, 0, 0, 0};
        const size_t count = instruction_registers(
            &program->instructions[index],
            registers
        );
        size_t layer = 0;
        for (size_t item = 0; item < count; ++item) {
            if (ready[registers[item]] > layer) {
                layer = ready[registers[item]];
            }
        }
        if (layer == SIZE_MAX) {
            fprintf(stderr, "dependency depth overflows size_t\n");
            exit(2);
        }
        assigned[index] = layer;
        for (size_t item = 0; item < count; ++item) {
            ready[registers[item]] = layer + 1U;
        }
        if (layer + 1U > layers) {
            layers = layer + 1U;
        }
    }
    free(ready);

    if (layers == SIZE_MAX) {
        fprintf(stderr, "layer-offset allocation overflows\n");
        exit(2);
    }
    size_t *offsets = calloc(layers + 1U, sizeof(size_t));
    size_t *cursor = calloc(layers, sizeof(size_t));
    struct instruction *ordered = calloc(
        instruction_count,
        sizeof(struct instruction)
    );
    if (offsets == NULL || cursor == NULL || ordered == NULL) {
        fprintf(stderr, "layer-order allocation failed\n");
        exit(2);
    }
    for (size_t index = 0; index < instruction_count; ++index) {
        ++offsets[assigned[index] + 1U];
    }
    for (size_t layer = 0; layer < layers; ++layer) {
        offsets[layer + 1U] += offsets[layer];
        cursor[layer] = offsets[layer];
    }
    for (size_t index = 0; index < instruction_count; ++index) {
        const size_t layer = assigned[index];
        ordered[cursor[layer]++] = program->instructions[index];
    }
    free(cursor);
    free(assigned);
    verify_schedule(
        ordered,
        offsets,
        layers,
        program->registers,
        instruction_count
    );

    if (
        program->passes > UINT64_MAX / program->instruction_count
    ) {
        fprintf(stderr, "parallel program step count overflows uint64\n");
        exit(2);
    }
    size_t maximum_layer_width = 0;
    size_t parallel_eligible_layers = 0;
    for (size_t layer = 0; layer < layers; ++layer) {
        const size_t width =
            offsets[layer + 1U] - offsets[layer];
        if (width > maximum_layer_width) {
            maximum_layer_width = width;
        }
        if (width >= PARALLEL_MIN_INSTRUCTIONS) {
            ++parallel_eligible_layers;
        }
    }
    return (struct parallel_schedule){
        .instructions = ordered,
        .layer_offsets = offsets,
        .layers = layers,
        .maximum_layer_width = maximum_layer_width,
        .parallel_eligible_layers = parallel_eligible_layers,
        .passes = program->passes,
        .steps = program->passes * program->instruction_count
    };
}

static void free_schedule(struct parallel_schedule *schedule) {
    free(schedule->instructions);
    free(schedule->layer_offsets);
    *schedule = (struct parallel_schedule){0};
}

static void *parallel_worker_main(void *opaque) {
    struct parallel_worker *worker = opaque;
    struct parallel_pool *pool = worker->pool;
    uint64_t observed_generation = 0;
    thread_fatal(
        pthread_mutex_lock(&pool->mutex),
        "parallel worker lock"
    );
    for (;;) {
        while (
            !pool->stop
            && pool->generation == observed_generation
        ) {
            thread_fatal(
                pthread_cond_wait(
                    &pool->start_condition,
                    &pool->mutex
                ),
                "parallel worker start wait"
            );
        }
        if (pool->stop) {
            thread_fatal(
                pthread_mutex_unlock(&pool->mutex),
                "parallel worker stop unlock"
            );
            return NULL;
        }

        struct carrier *carrier = pool->carrier;
        const struct parallel_schedule *schedule = pool->schedule;
        const size_t layer = pool->layer;
        const int inverse = pool->inverse;
        const uint64_t generation = pool->generation;
        const size_t layer_begin = schedule->layer_offsets[layer];
        const size_t layer_length =
            schedule->layer_offsets[layer + 1U] - layer_begin;
        const size_t quotient = layer_length / pool->thread_count;
        const size_t remainder = layer_length % pool->thread_count;
        const size_t extra_before =
            worker->worker_id < remainder
                ? worker->worker_id
                : remainder;
        const size_t begin =
            layer_begin
            + worker->worker_id * quotient
            + extra_before;
        const size_t end =
            begin
            + quotient
            + (worker->worker_id < remainder ? 1U : 0U);
        thread_fatal(
            pthread_mutex_unlock(&pool->mutex),
            "parallel worker job unlock"
        );

        for (size_t index = begin; index < end; ++index) {
            if (inverse) {
                apply_inverse(
                    carrier,
                    &schedule->instructions[index],
                    0
                );
            } else {
                apply_forward(
                    carrier,
                    &schedule->instructions[index]
                );
            }
        }

        thread_fatal(
            pthread_mutex_lock(&pool->mutex),
            "parallel worker completion lock"
        );
        ++pool->completed;
        if (pool->completed == pool->thread_count) {
            thread_fatal(
                pthread_cond_signal(&pool->done_condition),
                "parallel worker completion signal"
            );
        }
        observed_generation = generation;
    }
}

static void initialize_parallel_pool(
    struct parallel_pool *pool, size_t thread_count
) {
    *pool = (struct parallel_pool){0};
    pool->thread_count = thread_count;
    pool->threads = calloc(thread_count, sizeof(pthread_t));
    pool->workers = calloc(
        thread_count,
        sizeof(struct parallel_worker)
    );
    if (pool->threads == NULL || pool->workers == NULL) {
        fprintf(stderr, "parallel worker-pool allocation failed\n");
        exit(2);
    }
    thread_fatal(
        pthread_mutex_init(&pool->mutex, NULL),
        "parallel mutex initialization"
    );
    thread_fatal(
        pthread_cond_init(&pool->start_condition, NULL),
        "parallel start-condition initialization"
    );
    thread_fatal(
        pthread_cond_init(&pool->done_condition, NULL),
        "parallel done-condition initialization"
    );
    for (size_t index = 0; index < thread_count; ++index) {
        pool->workers[index] = (struct parallel_worker){
            .pool = pool,
            .worker_id = index
        };
        thread_fatal(
            pthread_create(
                &pool->threads[index],
                NULL,
                parallel_worker_main,
                &pool->workers[index]
            ),
            "parallel worker creation"
        );
    }
}

static void dispatch_parallel_layer(
    struct parallel_pool *pool,
    struct carrier *carrier,
    const struct parallel_schedule *schedule,
    size_t layer,
    int inverse
) {
    const size_t begin = schedule->layer_offsets[layer];
    const size_t end = schedule->layer_offsets[layer + 1U];
    if (
        pool->thread_count == 1U
        || end - begin < PARALLEL_MIN_INSTRUCTIONS
    ) {
        for (size_t index = begin; index < end; ++index) {
            if (inverse) {
                apply_inverse(
                    carrier,
                    &schedule->instructions[index],
                    0
                );
            } else {
                apply_forward(
                    carrier,
                    &schedule->instructions[index]
                );
            }
        }
        return;
    }
    thread_fatal(
        pthread_mutex_lock(&pool->mutex),
        "parallel dispatch lock"
    );
    pool->carrier = carrier;
    pool->schedule = schedule;
    pool->layer = layer;
    pool->inverse = inverse;
    pool->completed = 0;
    ++pool->generation;
    thread_fatal(
        pthread_cond_broadcast(&pool->start_condition),
        "parallel dispatch broadcast"
    );
    while (pool->completed != pool->thread_count) {
        thread_fatal(
            pthread_cond_wait(
                &pool->done_condition,
                &pool->mutex
            ),
            "parallel dispatch completion wait"
        );
    }
    thread_fatal(
        pthread_mutex_unlock(&pool->mutex),
        "parallel dispatch unlock"
    );
}

static void free_parallel_pool(struct parallel_pool *pool) {
    thread_fatal(
        pthread_mutex_lock(&pool->mutex),
        "parallel pool stop lock"
    );
    pool->stop = 1;
    thread_fatal(
        pthread_cond_broadcast(&pool->start_condition),
        "parallel pool stop broadcast"
    );
    thread_fatal(
        pthread_mutex_unlock(&pool->mutex),
        "parallel pool stop unlock"
    );
    for (size_t index = 0; index < pool->thread_count; ++index) {
        thread_fatal(
            pthread_join(pool->threads[index], NULL),
            "parallel worker join"
        );
    }
    thread_fatal(
        pthread_cond_destroy(&pool->done_condition),
        "parallel done-condition destruction"
    );
    thread_fatal(
        pthread_cond_destroy(&pool->start_condition),
        "parallel start-condition destruction"
    );
    thread_fatal(
        pthread_mutex_destroy(&pool->mutex),
        "parallel mutex destruction"
    );
    free(pool->workers);
    free(pool->threads);
    *pool = (struct parallel_pool){0};
}

static void run_parallel_forward(
    struct parallel_pool *pool,
    struct carrier *carrier,
    const struct parallel_schedule *schedule
) {
    for (uint64_t pass = 0; pass < schedule->passes; ++pass) {
        for (size_t layer = 0; layer < schedule->layers; ++layer) {
            dispatch_parallel_layer(
                pool,
                carrier,
                schedule,
                layer,
                0
            );
        }
    }
}

static void run_parallel_inverse(
    struct parallel_pool *pool,
    struct carrier *carrier,
    const struct parallel_schedule *schedule
) {
    for (uint64_t pass = schedule->passes; pass-- > 0;) {
        for (size_t layer = schedule->layers; layer-- > 0;) {
            dispatch_parallel_layer(
                pool,
                carrier,
                schedule,
                layer,
                1
            );
        }
    }
}

static struct execution execute_parallel(
    struct parallel_pool *pool,
    struct carrier *borrowed,
    const struct instruction_source *source,
    const struct parallel_schedule *schedule,
    int inverse_mode
) {
    struct carrier working = clone_carrier(borrowed);
    double complex *latch = calloc(
        working.registers,
        sizeof(double complex)
    );
    int *symbols = calloc(working.registers, sizeof(int));
    if (latch == NULL || symbols == NULL) {
        fprintf(stderr, "parallel latch allocation failed\n");
        exit(2);
    }

    load_input(&working, source);
    const uint64_t forward_start = monotonic_ns();
    run_parallel_forward(pool, &working, schedule);
    const uint64_t forward_ns = monotonic_ns() - forward_start;
    const double displacement =
        carrier_displacement(&working, borrowed);
    for (size_t index = 0; index < working.registers; ++index) {
        latch[index] = relation(&working, index);
    }

    const uint64_t inverse_start = monotonic_ns();
    if (inverse_mode != 2) {
        run_parallel_inverse(pool, &working, schedule);
        if (inverse_mode == 1) {
            write_relation_unlocked(
                &working,
                0,
                relation(&working, 0) * root_of_unity(1)
            );
        }
        unload_input(&working, source);
    }
    const uint64_t inverse_ns = monotonic_ns() - inverse_start;

    double maximum_root_distance = 0.0;
    for (size_t index = 0; index < working.registers; ++index) {
        double distance = 0.0;
        symbols[index] = decode_relation(latch[index], &distance);
        if (distance > maximum_root_distance) {
            maximum_root_distance = distance;
        }
    }
    const double restoration = carrier_error(&working, borrowed);
    if (inverse_mode == 0) {
        memcpy(
            borrowed->baseline,
            working.baseline,
            working.registers * sizeof(double complex)
        );
        memcpy(
            borrowed->working,
            working.working,
            working.registers * sizeof(double complex)
        );
    }
    free(latch);
    free_carrier(&working);
    return (struct execution){
        .steps = schedule->steps,
        .family = -2,
        .registers = borrowed->registers,
        .boundary_symbols = symbols,
        .maximum_root_distance = maximum_root_distance,
        .displacement_l2 = displacement,
        .restoration_max_abs = restoration,
        .forward_ns = forward_ns,
        .inverse_ns = inverse_ns,
        .public_program_fnv1a64 = source->public_program_fnv1a64
    };
}

static int same_boundary(
    const struct execution *left,
    const struct execution *right
) {
    return (
        left->registers == right->registers
        && memcmp(
            left->boundary_symbols,
            right->boundary_symbols,
            left->registers * sizeof(int)
        ) == 0
    );
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(
            stderr,
            "usage: %s THREADS PROGRAM.holo\n",
            argv[0]
        );
        return 2;
    }
    const uint64_t thread_count_u64 =
        parse_positive_decimal(argv[1], "thread count");
    if (thread_count_u64 > SIZE_MAX) {
        fprintf(stderr, "thread count exceeds host representation\n");
        return 2;
    }

    struct public_program program = read_public_program(argv[2]);
    struct parallel_schedule schedule = make_schedule(&program);
    if (
        schedule.layers > 0
        && program.passes > UINT64_MAX / (uint64_t)schedule.layers
    ) {
        fprintf(stderr, "logical depth overflows uint64\n");
        return 2;
    }
    const struct instruction_source source = {
        .steps = schedule.steps,
        .family = -2,
        .registers = program.registers,
        .input_symbols = program.input_symbols,
        .instructions = program.instructions,
        .instruction_period = program.instruction_count,
        .public_program_fnv1a64 = program.fnv1a64
    };
    struct parallel_pool pool;
    initialize_parallel_pool(
        &pool,
        (size_t)thread_count_u64
    );
    struct carrier carrier = make_carrier(program.registers, 131);

    struct execution first = execute_parallel(
        &pool,
        &carrier,
        &source,
        &schedule,
        0
    );
    print_execution(&first, "parallel-public-program");
    struct execution reuse = execute_parallel(
        &pool,
        &carrier,
        &source,
        &schedule,
        0
    );
    print_execution(&reuse, "parallel-actual-restored-reuse");
    struct execution wrong = execute_parallel(
        &pool,
        &carrier,
        &source,
        &schedule,
        1
    );
    print_execution(&wrong, "parallel-wrong-inverse");
    struct execution omitted = execute_parallel(
        &pool,
        &carrier,
        &source,
        &schedule,
        2
    );
    print_execution(&omitted, "parallel-omitted-inverse");
    printf(
        "{\"mode\":\"parallel-schedule\","
        "\"threads\":%zu,\"stored_instructions\":%llu,"
        "\"parallel_min_instructions\":%u,"
        "\"layers_per_pass\":%zu,"
        "\"maximum_layer_width\":%zu,"
        "\"parallel_eligible_layers_per_pass\":%zu,"
        "\"passes\":%llu,"
        "\"logical_depth\":%llu,\"total_steps\":%llu}\n",
        (size_t)thread_count_u64,
        (unsigned long long)program.instruction_count,
        PARALLEL_MIN_INSTRUCTIONS,
        schedule.layers,
        schedule.maximum_layer_width,
        schedule.parallel_eligible_layers,
        (unsigned long long)program.passes,
        (unsigned long long)(
            (uint64_t)schedule.layers * program.passes
        ),
        (unsigned long long)schedule.steps
    );

    const int inverse_controls_applicable =
        first.displacement_l2 > RESTORATION_MAX;
    printf(
        "{\"mode\":\"parallel-inverse-controls\","
        "\"applicable\":%s,"
        "\"forward_displacement_l2\":%.12g,"
        "\"reason\":\"%s\"}\n",
        inverse_controls_applicable ? "true" : "false",
        first.displacement_l2,
        inverse_controls_applicable
            ? "FORWARD_CHANGED_CARRIER"
            : "FORWARD_WAS_IDENTITY"
    );
    const int result = (
        first.restoration_max_abs <= RESTORATION_MAX
        && reuse.restoration_max_abs <= RESTORATION_MAX
        && (
            !inverse_controls_applicable
            || (
                wrong.restoration_max_abs > RESTORATION_MAX
                && omitted.restoration_max_abs > RESTORATION_MAX
            )
        )
        && same_boundary(&first, &reuse)
    ) ? 0 : 1;
    free_execution(&first);
    free_execution(&reuse);
    free_execution(&wrong);
    free_execution(&omitted);
    free_carrier(&carrier);
    free_parallel_pool(&pool);
    free_schedule(&schedule);
    free_public_program(&program);
    return result;
}
